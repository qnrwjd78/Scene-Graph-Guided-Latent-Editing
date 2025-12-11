import argparse
import torch
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gcn_wrapper import GCNWrapper
from models.graph_adapter import SceneGraphEmbedder
from models.dual_lora import DualInputLoRALinear
from processors.masactrl_processor import MasaCtrlSelfAttnProcessor
from stage2_utils.inversion import ddim_inversion, null_text_inversion, AttentionStore
from stage2_utils.warping import warp_tensor
from stage2_utils.data_loader import VGDataset
from stage2_utils.attention_map import AttentionStore as MapStore, register_attention_map_saver

# --- Helpers ---

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

def prepare_batch_for_embedder(obj_vecs, pred_vecs, triples, obj_to_img, triple_to_img, device):
    batch_size = obj_to_img.max().item() + 1
    
    batch_gcn_vectors = []
    batch_token_types = []
    batch_obj_idx = []
    batch_sub_ptr = []
    batch_obj_ptr = []
    
    max_len = 0
    
    for i in range(batch_size):
        curr_obj_mask = (obj_to_img == i)
        curr_pred_mask = (triple_to_img == i)
        
        curr_obj_vecs = obj_vecs[curr_obj_mask]
        curr_pred_vecs = pred_vecs[curr_pred_mask]
        curr_triples = triples[curr_pred_mask]
        
        # Remove SEP token (__image__ node) and associated relationships
        if curr_obj_vecs.shape[0] > 0:
             # The last object is always __image__
             curr_obj_vecs = curr_obj_vecs[:-1]
             
             # Filter predicates connected to the last object
             global_obj_indices = torch.where(curr_obj_mask)[0]
             sep_node_global_idx = global_obj_indices[-1]
             
             valid_pred_mask = (curr_triples[:, 2] != sep_node_global_idx)
             
             curr_pred_vecs = curr_pred_vecs[valid_pred_mask]
             curr_triples = curr_triples[valid_pred_mask]
             
             # Update global_obj_indices to exclude SEP node
             global_obj_indices = global_obj_indices[:-1]
        else:
             global_obj_indices = torch.tensor([], device=device)
        
        num_objs = curr_obj_vecs.shape[0]
        num_rels = curr_pred_vecs.shape[0]
        
        if num_objs == 0:
             curr_seq_vecs = torch.zeros((0, curr_obj_vecs.shape[-1]), device=device)
        else:
             curr_seq_vecs = torch.cat([curr_obj_vecs, curr_pred_vecs], dim=0)
        
        curr_token_types = [0] * num_objs + [1] * num_rels
        curr_obj_idx = list(range(num_objs)) + [0] * num_rels
        
        global_to_local = {g_idx.item(): l_idx for l_idx, g_idx in enumerate(global_obj_indices)}
        
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

def register_attention_control(pipe, attn_store, is_inversion=False):
    attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        if name.endswith("attn1.processor"): # Self Attention
            attn_procs[name] = MasaCtrlSelfAttnProcessor(attn_store, is_inversion)
        else: # Cross Attention (attn2)
            attn_procs[name] = pipe.unet.attn_processors[name]
            
    pipe.unet.set_attn_processor(attn_procs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument("--image_index", type=int, default=2524, help="Index of the image in VG dataset to edit")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--edit_type", type=str, default="reconstruction", choices=["reconstruction", "move_object", "replace_object"])
    parser.add_argument("--inversion_type", type=str, default="ddim", choices=["ddim", "null_text"])
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--save_attn_map", action="store_true", help="Save cross-attention maps")
    parser.add_argument("--dataset", type=str, default="vg", help="Dataset to use: vg or vg_clevr")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Models
    print("Loading models...")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    
    # Inject Dual LoRA
    inject_dual_lora(pipe.unet)
    
    # GCN
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gcn = GCNWrapper(
        vocab_file=os.path.join(project_root, f"datasets/{args.dataset}/vocab.json"),
        checkpoint_path=os.path.join(project_root, "pretrained/sip_vg.pt")
    ).to(device)
    gcn.eval()
    
    # Adapter
    adapter = SceneGraphEmbedder().to(device)
    
    # Load Checkpoint
    if os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                adapter.load_state_dict(checkpoint['model_state_dict'])
            if 'lora_state_dict' in checkpoint:
                pipe.unet.load_state_dict(checkpoint['lora_state_dict'], strict=False)
                print("Loaded LoRA weights.")
        else:
            adapter.load_state_dict(checkpoint)
            print("Loaded adapter weights (old format).")
    else:
        print(f"Checkpoint {args.checkpoint_path} not found!")
        return

    adapter.eval()
    
    # 2. Prepare Data
    print(f"Loading data (Index {args.image_index})...")
    dataset = VGDataset(
        vocab_path=os.path.join(project_root, f"datasets/{args.dataset}/vocab.json"),
        h5_path=os.path.join(project_root, f"datasets/{args.dataset}/val.h5"),
        image_dir=os.path.join(project_root, f"datasets/{args.dataset}/images"),
        image_size=(512, 512),
        max_objects=10
    )
    
    # Get item
    img_tensor, objs, boxes, triples = dataset[args.image_index]
    
    # Prepare batch
    img_tensor = img_tensor.unsqueeze(0).to(device) # (1, 3, H, W)
    objs = objs.to(device)
    boxes = boxes.to(device)
    triples = triples.to(device)
    
    # Fake batch indices
    obj_to_img = torch.zeros(objs.shape[0], dtype=torch.long, device=device)
    triple_to_img = torch.zeros(triples.shape[0], dtype=torch.long, device=device)
    
    graphs = [objs, boxes, triples, obj_to_img, triple_to_img]
    
    # Get Graph Embedding
    with torch.no_grad():
        obj_vecs, pred_vecs = gcn.get_raw_features(graphs)
        
        gcn_vectors, token_types, obj_idx, sub_ptr, obj_ptr = prepare_batch_for_embedder(
            obj_vecs, pred_vecs, triples, obj_to_img, triple_to_img, device
        )
        
        x_mixed = adapter(gcn_vectors, token_types, obj_idx, sub_ptr, obj_ptr)
        cond_embeddings = x_mixed
        
        # Pad to 77 if needed
        B, N, C = cond_embeddings.shape
        if N < 77:
            padding = torch.zeros(B, 77 - N, C, device=device)
            cond_embeddings = torch.cat([cond_embeddings, padding], dim=1)
        elif N > 77:
            cond_embeddings = cond_embeddings[:, :77, :]

    # 3. Inversion
    print(f"Running Inversion ({args.inversion_type})...")
    
    # Inverse ImageNet Norm
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    img_unnorm = img_tensor * std + mean
    img_sd = img_unnorm * 2 - 1 # [0, 1] -> [-1, 1]
    
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
    
    # Register Attention Map Saver if needed
    map_store = None
    if args.save_attn_map:
        # Note: register_attention_control replaces processors.
        # register_attention_map_saver ALSO replaces processors.
        # They conflict if we want BOTH MasaCtrl AND Map Saving.
        # MasaCtrl is for Self-Attention. Map Saver is for Cross-Attention (usually).
        # Our Map Saver replaces ALL processors.
        # We need to be careful.
        
        # If we want both, we need to merge them.
        # MasaCtrl only touches "attn1" (Self).
        # Map Saver touches everything but we only care about Cross ("attn2") for visualization.
        
        # Let's manually inject Map Saver into Cross Attn processors ONLY.
        map_store = MapStore(save_dir=args.output_dir, res=32)
        
        # Iterate existing processors (which might be MasaCtrl or Default)
        new_procs = {}
        for name, proc in pipe.unet.attn_processors.items():
            if name.endswith("attn2.processor"): # Cross Attention
                # Replace with Map Saver Processor
                # We need to import the class
                from stage2_utils.attention_map import AttentionMapSaverProcessor
                
                # Determine place
                if "down_blocks" in name: place = "down"
                elif "mid_block" in name: place = "mid"
                elif "up_blocks" in name: place = "up"
                else: place = "unknown"
                
                new_procs[name] = AttentionMapSaverProcessor(map_store, place)
            else:
                # Keep existing (MasaCtrl or Default)
                new_procs[name] = proc
        
        pipe.unet.set_attn_processor(new_procs)
    
    if args.edit_type == "move_object":
        # Example: Move object 0
        # We need to know which object is which.
        # boxes is (N, 4).
        # Let's just move the first object slightly for demo.
        old_box = boxes[0].tolist() # [x1, y1, x2, y2]
        new_box = [old_box[0]+0.1, old_box[1], old_box[2]+0.1, old_box[3]]
        print(f"Moving object 0 from {old_box} to {new_box}")
        
        # Update warping fn in all processors
        for proc in pipe.unet.attn_processors.values():
            if isinstance(proc, MasaCtrlSelfAttnProcessor):
                proc.old_box = old_box
                proc.new_box = new_box
                proc.warping_fn = warp_tensor
    
    # Generation Loop
    pipe.scheduler.set_timesteps(args.num_inference_steps)
    timesteps = pipe.scheduler.timesteps
    
    # --- 1. Generate with CFG 7.5 ---
    latents = z_T.clone()
    image_cfg75 = None
    
    if args.inversion_type == "null_text":
        print("Generating with CFG 7.5 (Null-text)...")
        for i, t in enumerate(tqdm(timesteps, desc="Generating CFG 7.5")):
            uncond_embedding = uncond_embeddings_list[i]
            
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            
            concat_embeds = torch.cat([uncond_embedding, cond_embeddings])
            
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=concat_embeds).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + 7.5 * (noise_pred_text - noise_pred_uncond)
            
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
            
        # Decode
        with torch.no_grad():
            image_cfg75 = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            image_cfg75 = pipe.image_processor.postprocess(image_cfg75, output_type="pil", do_denormalize=[True]*image_cfg75.shape[0])[0]
            
    else:
        # Standard DDIM Generation
        guidance_scale = 7.5
        if args.edit_type == "reconstruction" and args.inversion_type == "ddim":
            print("For DDIM reconstruction, setting guidance_scale=1.0 to match inversion.")
            guidance_scale = 1.0
            
        # Create negative_prompt_embeds (Unconditional)
        B = cond_embeddings.shape[0]
        uncond_input = pipe.tokenizer([""] * B, padding="max_length", max_length=77, return_tensors="pt")
        with torch.no_grad():
            uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
        negative_prompt_embeds = torch.cat([uncond_embeddings, uncond_embeddings], dim=-1)
            
        image_cfg75 = pipe(
            prompt_embeds=cond_embeddings,
            negative_prompt_embeds=negative_prompt_embeds,
            latents=z_T,
            guidance_scale=guidance_scale,
            num_inference_steps=args.num_inference_steps
        ).images[0]
    
    image_cfg75.save(os.path.join(args.output_dir, f"result_{args.image_index}_{args.edit_type}_{args.inversion_type}_cfg7.5.png"))

    # --- 2. Generate with CFG 1.0 (No CFG) ---
    latents = z_T.clone()
    image_cfg10 = None
    
    if args.inversion_type == "null_text":
        print("Generating with CFG 1.0 (Null-text)...")
        for i, t in enumerate(tqdm(timesteps, desc="Generating CFG 1.0")):
            uncond_embedding = uncond_embeddings_list[i]
            
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
            
            concat_embeds = torch.cat([uncond_embedding, cond_embeddings])
            
            noise_pred = pipe.unet(latent_model_input, t, encoder_hidden_states=concat_embeds).sample
            
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            # CFG 1.0: noise_pred = noise_pred_text
            noise_pred = noise_pred_text
            
            latents = pipe.scheduler.step(noise_pred, t, latents).prev_sample
            
        # Decode
        with torch.no_grad():
            image_cfg10 = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            image_cfg10 = pipe.image_processor.postprocess(image_cfg10, output_type="pil", do_denormalize=[True]*image_cfg10.shape[0])[0]
            
    else:
        # Standard DDIM Generation
        # Create negative_prompt_embeds (Unconditional)
        B = cond_embeddings.shape[0]
        uncond_input = pipe.tokenizer([""] * B, padding="max_length", max_length=77, return_tensors="pt")
        with torch.no_grad():
            uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
        negative_prompt_embeds = torch.cat([uncond_embeddings, uncond_embeddings], dim=-1)
            
        image_cfg10 = pipe(
            prompt_embeds=cond_embeddings,
            negative_prompt_embeds=negative_prompt_embeds,
            latents=z_T,
            guidance_scale=1.0,
            num_inference_steps=args.num_inference_steps
        ).images[0]
    
    image_cfg10.save(os.path.join(args.output_dir, f"result_{args.image_index}_{args.edit_type}_{args.inversion_type}_cfg1.0.png"))
    
    # Save Attention Map
    if args.save_attn_map and map_store is not None:
        # Get labels
        obj_names = []
        for idx in objs:
            name = dataset.vocab['object_idx_to_name'][idx.item()]
            obj_names.append(name)
        
        all_labels = obj_names[:]
        for t in triples:
            p_idx = t[1].item()
            p_name = dataset.vocab['pred_idx_to_name'][p_idx]
            all_labels.append(p_name)
            
        map_store.save_attention_maps(all_labels, save_name=f"attn_map_{args.image_index}_{args.edit_type}")
    
    # Save original
    img_pil = Image.fromarray((img_unnorm[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    img_pil.save(os.path.join(args.output_dir, f"original_{args.image_index}.png"))
    
    print("Sampling done.")

if __name__ == "__main__":
    main()
