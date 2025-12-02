import argparse
import torch
import os
import sys
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn as nn

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gcn_wrapper import GCNWrapper
from models.graph_adapter import SceneGraphEmbedder
from models.dual_lora import DualInputLoRALinear
from stage2_utils.data_loader import VGDataset
from stage2_utils.attention_map import AttentionStore, register_attention_map_saver

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
        
        # 2. Concatenate vectors
        if num_objs == 0:
             curr_seq_vecs = torch.zeros((0, curr_obj_vecs.shape[-1]), device=device)
        else:
             curr_seq_vecs = torch.cat([curr_obj_vecs, curr_pred_vecs], dim=0)
        
        # 3. Build Indices
        curr_token_types = [0] * num_objs + [1] * num_rels
        curr_obj_idx = list(range(num_objs)) + [0] * num_rels
        
        # Pointers
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
        
    # 4. Pad
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
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the checkpoint file (e.g., checkpoints/adapter_epoch_10.pth)")
    parser.add_argument("--image_index", type=int, default=0, help="Index of the image in VG dataset to generate")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--save_attn_map", action="store_true", help="Save cross-attention maps")
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
        vocab_file=os.path.join(project_root, "datasets/vg/vocab.json"),
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
        vocab_path=os.path.join(project_root, "datasets/vg/vocab.json"),
        h5_path=os.path.join(project_root, "datasets/vg/val.h5"),
        image_dir=os.path.join(project_root, "datasets/vg/images"),
        image_size=(512, 512),
        max_objects=10
    )
    
    # Get item
    img_tensor, objs, boxes, triples = dataset[args.image_index]
    
    # Prepare batch (Batch size 1)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    objs = objs.to(device)
    boxes = boxes.to(device)
    triples = triples.to(device)
    
    # Fake batch indices
    obj_to_img = torch.zeros(objs.shape[0], dtype=torch.long, device=device)
    triple_to_img = torch.zeros(triples.shape[0], dtype=torch.long, device=device)
    
    graphs = [objs, boxes, triples, obj_to_img, triple_to_img]
    
    # 3. Inference
    with torch.no_grad():
        # Encode Graph
        obj_vecs, pred_vecs = gcn.get_raw_features(graphs)
        
        # Prepare Batch
        gcn_vectors, token_types, obj_idx, sub_ptr, obj_ptr = prepare_batch_for_embedder(
            obj_vecs, pred_vecs, triples, obj_to_img, triple_to_img, device
        )
        
        # Adapter Forward
        x_clean, x_mixed = adapter(gcn_vectors, token_types, obj_idx, sub_ptr, obj_ptr)
        context_embedding = torch.cat([x_clean, x_mixed], dim=-1)
        
        # Pad to 77 for SD compatibility
        B, N, C = context_embedding.shape
        if N < 77:
            padding = torch.zeros(B, 77 - N, C, device=device)
            context_embedding = torch.cat([context_embedding, padding], dim=1)
        elif N > 77:
            context_embedding = context_embedding[:, :77, :]
            
        # Create negative_prompt_embeds (Unconditional)
        # We need 1536 dim (768*2)
        uncond_input = pipe.tokenizer([""] * B, padding="max_length", max_length=77, return_tensors="pt")
        with torch.no_grad():
            uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
        negative_prompt_embeds = torch.cat([uncond_embeddings, uncond_embeddings], dim=-1)
        
        # Register Attention Saver if needed
        attn_store = None
        if args.save_attn_map:
            attn_store = AttentionStore(save_dir=args.output_dir, res=32)
            register_attention_map_saver(pipe, attn_store)
        
        # Generate
        print("Generating image...")
        image = pipe(
            prompt_embeds=context_embedding,
            negative_prompt_embeds=negative_prompt_embeds,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale
        ).images[0]
        
        save_path = os.path.join(args.output_dir, f"sample_{args.image_index}.png")
        image.save(save_path)
        print(f"Saved generated image to {save_path}")
        
        # Construct Scene Graph Text
        sg_text = f"Image Index: {args.image_index}\n\nObjects:\n"
        obj_names = []
        for idx, obj_idx in enumerate(objs):
            name = dataset.vocab['object_idx_to_name'][obj_idx.item()]
            obj_names.append(name)
            sg_text += f"[{idx}] {name}\n"
        
        sg_text += "\nRelationships:\n"
        for t in triples:
            s_idx = t[0].item()
            p_idx = t[1].item()
            o_idx = t[2].item()
            
            if s_idx < len(obj_names) and o_idx < len(obj_names):
                s_name = obj_names[s_idx]
                p_name = dataset.vocab['pred_idx_to_name'][p_idx]
                o_name = obj_names[o_idx]
                sg_text += f"[{s_idx}]{s_name} --[{p_name}]--> [{o_idx}]{o_name}\n"
            
        # Save Scene Graph Text
        sg_path = os.path.join(args.output_dir, f"scene_graph_{args.image_index}.txt")
        with open(sg_path, "w") as f:
            f.write(sg_text)
        print(f"Saved scene graph text to {sg_path}")

        # Save Attention Map
        if args.save_attn_map and attn_store is not None:
            # Use the names we just extracted
            all_labels = obj_names[:]
            for t in triples:
                p_idx = t[1].item()
                p_name = dataset.vocab['pred_idx_to_name'][p_idx]
                all_labels.append(p_name)
                
            attn_store.save_attention_maps(all_labels, save_name=f"attn_map_{args.image_index}")
        
        # Save Original for comparison
        # Inverse Normalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        img_unnorm = img_tensor * std + mean
        img_pil = Image.fromarray((img_unnorm[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
        img_pil.save(os.path.join(args.output_dir, f"original_{args.image_index}.png"))

if __name__ == "__main__":
    main()
