import argparse
import torch
import torch.nn as nn
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gcn_wrapper import GCNWrapper
from models.graph_adapter import SceneGraphEmbedder
from models.dual_lora import DualInputLoRALinear
from processors.masactrl_processor import MasaCtrlSelfAttnProcessor
from stage2_utils.inversion import ddim_inversion, null_text_inversion, AttentionStore
from stage2_utils.warping import warp_tensor
from stage2_utils.data_loader import VGDataset
from stage2_utils.daam import GLOBAL_ATTENTION_STORE, DAAMCrossAttnProcessor, aggregate_attention_maps, visualize_and_save_maps

def inject_dual_lora(unet):
    lora_params = []
    for name, module in unet.named_modules():
        if name.endswith('attn2'): # Cross Attention
            # Replace to_k
            if isinstance(module.to_k, nn.Linear):
                module.to_k = DualInputLoRALinear(module.to_k)
                lora_params.extend([p for p in module.to_k.parameters() if p.requires_grad])
            
            # Replace to_v
            if isinstance(module.to_v, nn.Linear):
                module.to_v = DualInputLoRALinear(module.to_v)
                lora_params.extend([p for p in module.to_v.parameters() if p.requires_grad])
    return lora_params

def register_attention_control(pipe, attn_store, is_inversion=False):
    # Helper to register MasaCtrl processors AND DAAM processors
    
    attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        if name.endswith("attn1.processor"): # Self Attention -> MasaCtrl
            # attn_procs[name] = MasaCtrlSelfAttnProcessor(attn_store, is_inversion)
            attn_procs[name] = pipe.unet.attn_processors[name]
        elif name.endswith("attn2.processor"): # Cross Attention -> DAAM
            # We only use DAAM during generation (not inversion usually, but can be useful)
            # If is_inversion is True, we might skip DAAM or use it too.
            # Let's use it always to capture maps.
            attn_procs[name] = DAAMCrossAttnProcessor(layer_name=name)
        else:
            attn_procs[name] = pipe.unet.attn_processors[name]
            
    pipe.unet.set_attn_processor(attn_procs)

def prepare_batch_for_embedder(obj_vecs, pred_vecs, triples, obj_to_img, triple_to_img, device):
    """
    Converts raw GCN outputs into padded batches for SceneGraphEmbedder.
    """
    batch_size = obj_to_img.max().item() + 1
    
    batch_gcn_vectors = []
    batch_token_types = []
    batch_obj_idx = []
    batch_sub_ptr = []
    batch_obj_ptr = []
    
    max_len = 0
    
    for i in range(batch_size):
        # 1. Extract features for this image
        curr_obj_mask = (obj_to_img == i)
        curr_pred_mask = (triple_to_img == i)
        
        curr_obj_vecs = obj_vecs[curr_obj_mask]
        curr_pred_vecs = pred_vecs[curr_pred_mask]
        
        num_objs = curr_obj_vecs.shape[0]
        num_rels = curr_pred_vecs.shape[0]
        
        # 2. Concatenate vectors
        if num_objs == 0:
             curr_seq_vecs = torch.zeros((0, curr_obj_vecs.shape[-1]), device=device)
        else:
             curr_seq_vecs = torch.cat([curr_obj_vecs, curr_pred_vecs], dim=0)
        
        # 3. Build Indices
        curr_token_types = [0] * num_objs + [1] * num_rels
        curr_obj_idx = list(range(num_objs)) + [0] * num_rels
        
        global_obj_indices = torch.where(curr_obj_mask)[0]
        global_to_local = {g_idx.item(): l_idx for l_idx, g_idx in enumerate(global_obj_indices)}
        
        curr_triples = triples[curr_pred_mask]
        
        curr_sub_ptr = [0] * num_objs
        curr_obj_ptr = [0] * num_objs
        
        for t in curr_triples:
            s_global, _, o_global = t.tolist()
            s_local = global_to_local.get(s_global, 0)
            o_local = global_to_local.get(o_global, 0)
            curr_sub_ptr.append(s_local)
            curr_obj_ptr.append(o_local)
            
        batch_gcn_vectors.append(curr_seq_vecs)
        batch_token_types.append(torch.tensor(curr_token_types, device=device))
        batch_obj_idx.append(torch.tensor(curr_obj_idx, device=device))
        batch_sub_ptr.append(torch.tensor(curr_sub_ptr, device=device))
        batch_obj_ptr.append(torch.tensor(curr_obj_ptr, device=device))
        
        max_len = max(max_len, len(curr_token_types))
        
    def pad_tensor(t_list, pad_val=0):
        padded = []
        for t in t_list:
            pad_len = max_len - t.shape[0]
            if pad_len > 0:
                if t.dim() == 1:
                    p = torch.full((pad_len,), pad_val, device=device, dtype=t.dtype)
                    padded.append(torch.cat([t, p]))
                else:
                    p = torch.full((pad_len, t.shape[1]), pad_val, device=device, dtype=t.dtype)
                    padded.append(torch.cat([t, p], dim=0))
            else:
                padded.append(t)
        return torch.stack(padded)

    padded_gcn_vectors = pad_tensor(batch_gcn_vectors, 0.0)
    padded_token_types = pad_tensor(batch_token_types, 0) 
    padded_obj_idx = pad_tensor(batch_obj_idx, 0)
    padded_sub_ptr = pad_tensor(batch_sub_ptr, 0)
    padded_obj_ptr = pad_tensor(batch_obj_ptr, 0)
    
    return padded_gcn_vectors, padded_token_types, padded_obj_idx, padded_sub_ptr, padded_obj_ptr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default="checkpoints/adapter.pth")
    parser.add_argument("--image_index", type=int, default=2524, help="Index of the image in VG dataset to edit")
    parser.add_argument("--output_dir", type=str, default="outputs_daam")
    parser.add_argument("--edit_type", type=str, default="reconstruction", choices=["reconstruction", "move_object", "replace_object"])
    parser.add_argument("--inversion_type", type=str, default="ddim", choices=["ddim", "null_text"])
    parser.add_argument("--num_inference_steps", type=int, default=20, help="Number of denoising steps")
    parser.add_argument("--source_object", type=str, default="horse", help="Object to replace (for replace_object edit)")
    parser.add_argument("--target_object", type=str, default="zebra", help="Target object (for replace_object edit)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.cuda.empty_cache()
    
    # 1. Load Models
    print("Loading models...")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    pipe.enable_attention_slicing()
    
    # Inject Dual LoRA
    inject_dual_lora(pipe.unet)
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gcn = GCNWrapper(
        vocab_file=os.path.join(project_root, "datasets/vg/vocab.json"),
        checkpoint_path=os.path.join(project_root, "pretrained/sip_vg.pt")
    ).to(device, dtype=torch.float16)
    gcn.eval()
    
    adapter = SceneGraphEmbedder().to(device)
    if os.path.exists(args.adapter_path):
        checkpoint = torch.load(args.adapter_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            adapter.load_state_dict(checkpoint["model_state_dict"])
        else:
            adapter.load_state_dict(checkpoint)
        print(f"Loaded adapter from {args.adapter_path}")
    else:
        print(f"Warning: Adapter checkpoint {args.adapter_path} not found. Using random weights.")
    adapter.to(dtype=torch.float16)
    adapter.eval()
    
    # 2. Prepare Data
    print(f"Loading data (Index {args.image_index})...")
    dataset = VGDataset(
        vocab_path=os.path.join(project_root, "datasets/vg/vocab.json"),
        h5_path=os.path.join(project_root, "datasets/vg/val.h5"), # Use val set
        image_dir=os.path.join(project_root, "datasets/vg/images"),
        image_size=(512, 512), # SD uses 512
        max_objects=10
    )
    
    # Get item
    img_tensor, objs, boxes, triples = dataset[args.image_index]
    
    # Prepare batch
    img_tensor = img_tensor.unsqueeze(0).to(device) # (1, 3, H, W)
    objs = objs.unsqueeze(0).to(device)
    boxes = boxes.unsqueeze(0).to(device)
    triples = triples.unsqueeze(0).to(device)
    
    # Get Graph Embedding
    with torch.no_grad():
        obj_to_img = torch.zeros(objs.size(1), dtype=torch.long).to(device)
        triple_to_img = torch.zeros(triples.size(1), dtype=torch.long).to(device)
        objs_flat = objs.view(-1)
        boxes_flat = boxes.view(-1, 4)
        triples_flat = triples.view(-1, 3)
        
        graphs = [objs_flat, boxes_flat, triples_flat, obj_to_img, triple_to_img]
        
        # Use get_raw_features instead of forward
        obj_vecs, pred_vecs = gcn.get_raw_features(graphs)
        obj_vecs = obj_vecs.to(dtype=torch.float16)
        pred_vecs = pred_vecs.to(dtype=torch.float16)
            
        gcn_vectors, token_types, obj_idx, sub_ptr, obj_ptr = prepare_batch_for_embedder(
            obj_vecs, pred_vecs, triples_flat, obj_to_img, triple_to_img, device
        )
        
        # Adapter Forward
        x_clean, x_mixed = adapter(gcn_vectors, token_types, obj_idx, sub_ptr, obj_ptr)
        
        # Combine for Dual Input
        cond_embeddings = torch.cat([x_clean, x_mixed], dim=-1).to(dtype=torch.float16)
        
        # Free GCN and Adapter
        gcn.to("cpu")
        adapter.to("cpu")
        del gcn, adapter, obj_vecs, pred_vecs, gcn_vectors, x_clean, x_mixed
        torch.cuda.empty_cache()
        
        # Pad to 77 for SD compatibility (required for concatenation with uncond_embeddings)
        B, N, C = cond_embeddings.shape
        if N < 77:
            padding = torch.zeros(B, 77 - N, C, device=device, dtype=cond_embeddings.dtype)
            cond_embeddings = torch.cat([cond_embeddings, padding], dim=1)
        elif N > 77:
            cond_embeddings = cond_embeddings[:, :77, :]
            
    # 3. Inversion
    print(f"Running Inversion ({args.inversion_type})...")
    
    # Inverse ImageNet Norm
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    img_unnorm = img_tensor * std + mean
    img_sd = (img_unnorm * 2 - 1).to(dtype=torch.float16) # [0, 1] -> [-1, 1]
    
    if args.inversion_type == "null_text":
        z_T, uncond_embeddings_list = null_text_inversion(pipe, img_sd, cond_embeddings, num_inference_steps=args.num_inference_steps)
    else:
        z_T, _ = ddim_inversion(pipe, img_sd, cond_embeddings, num_inference_steps=args.num_inference_steps)
        uncond_embeddings_list = None
    
    # 4. Generation with Editing
    print(f"Generating ({args.edit_type})...")
    
    # Setup Processors
    attn_store = AttentionStore()
    register_attention_control(pipe, attn_store, is_inversion=False) # We are generating now
    
    # Reset DAAM Store
    GLOBAL_ATTENTION_STORE.reset()
    
    if args.edit_type == "move_object":
        # Example: Move object 0
        old_box = boxes_flat[0].tolist() # [x1, y1, x2, y2]
        new_box = [old_box[0]+0.1, old_box[1], old_box[2]+0.1, old_box[3]]
        print(f"Moving object from {old_box} to {new_box}")
        
        # Update warping fn in all processors
        for proc in pipe.unet.attn_processors.values():
            if isinstance(proc, MasaCtrlSelfAttnProcessor):
                proc.old_box = old_box
                proc.new_box = new_box
                proc.warping_fn = warp_tensor
    
    # Generation Loop
    latents = z_T
    pipe.scheduler.set_timesteps(args.num_inference_steps)
    timesteps = pipe.scheduler.timesteps
    
    # Manual Loop to support DAAM callback
    
    # Prepare uncond embeddings
    if uncond_embeddings_list is None:
        uncond_embeddings = torch.zeros_like(cond_embeddings)
    
    for i, t in enumerate(tqdm(timesteps, desc="Generating")):
        # DAAM Step
        GLOBAL_ATTENTION_STORE.next_step()
        
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        if args.inversion_type == "null_text":
            uncond_embedding = uncond_embeddings_list[i]
        else:
            uncond_embedding = uncond_embeddings
            
        concat_embeds = torch.cat([uncond_embedding, cond_embeddings])
        
        noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=concat_embeds).sample
        
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        
        guidance_scale = 7.5
        if args.edit_type == "reconstruction" and args.inversion_type == "ddim":
             guidance_scale = 1.0
             
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
        
    # Decode
    with torch.no_grad():
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        image = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True]*image.shape[0])[0]
            
    image.save(os.path.join(args.output_dir, f"result_{args.image_index}_{args.edit_type}_{args.inversion_type}.png"))
    
    # Save original
    img_pil = Image.fromarray((img_unnorm[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    img_pil.save(os.path.join(args.output_dir, f"original_{args.image_index}.png"))
    
    # Visualize DAAM Maps
    print("Processing Attention Maps...")
    vocab = dataset.vocab
    obj_names = [vocab['object_idx_to_name'][i.item()] for i in objs[0]]
    
    labels = []
    for i, name in enumerate(obj_names):
        labels.append(f"Obj{i}: {name}")
        
    # Relations
    for i in range(triples.size(1)):
        s, p, o = triples[0, i].tolist()
        p_name = vocab['pred_idx_to_name'][p]
        s_name = obj_names[s] if s < len(obj_names) else "?"
        o_name = obj_names[o] if o < len(obj_names) else "?"
        labels.append(f"Rel: {s_name}-{p_name}-{o_name}")
        
    final_maps = aggregate_attention_maps()
    if final_maps is not None:
        visualize_and_save_maps(image, final_maps, labels, os.path.join(args.output_dir, f"attention_{args.image_index}.png"))

    print("Sampling done.")

if __name__ == "__main__":
    main()
