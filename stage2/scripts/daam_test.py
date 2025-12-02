import argparse
import torch
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import torch.nn as nn
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDIMScheduler
from einops import rearrange

# Add stage2 to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gcn_wrapper import GCNWrapper
from models.graph_adapter import SceneGraphEmbedder
from models.dual_lora import DualInputLoRALinear
from stage2_utils.data_loader import VGDataset, vg_collate_fn
from stage2_utils.daam import GLOBAL_ATTENTION_STORE, register_daam_processors, aggregate_attention_maps, visualize_and_save_maps

# ============================================================================
# Utils
# ============================================================================

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

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default="checkpoints/adapter.pth")
    parser.add_argument("--output_dir", type=str, default="daam_results")
    parser.add_argument("--start_idx", type=int, default=2500)
    parser.add_argument("--end_idx", type=int, default=2505)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Models
    print("Loading models...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", local_files_only=True)
    except Exception:
        print("Model not found in cache, downloading...")
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    
    # Inject Dual LoRA
    inject_dual_lora(pipe.unet)
    
    # Load GCN
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gcn = GCNWrapper(
        vocab_file=os.path.join(project_root, "datasets/vg/vocab.json"),
        checkpoint_path=os.path.join(project_root, "pretrained/sip_vg.pt") 
    ).to(device)
    gcn.eval()
    
    # Load Adapter
    adapter = SceneGraphEmbedder().to(device)
    
    if os.path.exists(args.adapter_path):
        print(f"Loading adapter from {args.adapter_path}")
        checkpoint = torch.load(args.adapter_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            adapter.load_state_dict(checkpoint['model_state_dict'])
            if 'lora_state_dict' in checkpoint:
                pipe.unet.load_state_dict(checkpoint['lora_state_dict'], strict=False)
                print("Loaded LoRA weights")
        else:
            adapter.load_state_dict(checkpoint)
    else:
        print(f"Warning: Adapter checkpoint {args.adapter_path} not found. Using random weights.")
    
    adapter.eval()
    
    # Register DAAM Processors
    register_daam_processors(pipe)
    
    # 2. Data
    print("Loading data...")
    dataset = VGDataset(
        vocab_path=os.path.join(project_root, "datasets/vg/vocab.json"),
        h5_path=os.path.join(project_root, "datasets/vg/test.h5"),
        image_dir=os.path.join(project_root, "datasets/vg/images"),
        image_size=(512, 512),
        normalize_images=False, # We normalize manually to keep original clean
        max_samples=None
    )
    
    # 3. Inference Loop
    vocab = dataset.vocab
    
    # We iterate manually
    for idx in range(args.start_idx, args.end_idx + 1):
        if idx >= len(dataset): break
        print(f"Processing image {idx}...")
        
        # Get item
        img_tensor, objs, boxes, triples = dataset[idx]
        
        # Prepare batch (size 1)
        images = img_tensor.unsqueeze(0).to(device)
        objects = objs.unsqueeze(0).to(device)
        boxes = boxes.unsqueeze(0).to(device)
        triples_batch = triples.unsqueeze(0).to(device)
        
        # Create dummy mappings for batch size 1
        obj_to_img = torch.zeros(objects.size(1), dtype=torch.long).to(device)
        triple_to_img = torch.zeros(triples_batch.size(1), dtype=torch.long).to(device)
        
        graphs = [objects.view(-1), boxes.view(-1, 4), triples_batch.view(-1, 3), obj_to_img, triple_to_img]
        
        # Encode Graph
        with torch.no_grad():
            obj_vecs, pred_vecs = gcn.get_raw_features(graphs)
            
            gcn_vectors, token_types, obj_idx, sub_ptr, obj_ptr = prepare_batch_for_embedder(
                obj_vecs, pred_vecs, triples_batch.view(-1, 3), obj_to_img, triple_to_img, device
            )
            
            # Adapter Forward
            x_clean, x_mixed = adapter(gcn_vectors, token_types, obj_idx, sub_ptr, obj_ptr)
            
            # Combine for Dual Input
            context_embedding = torch.cat([x_clean, x_mixed], dim=-1)
        
        # Generate
        GLOBAL_ATTENTION_STORE.reset()
        
        # Callback to advance step
        def callback_fn(step, timestep, latents):
            GLOBAL_ATTENTION_STORE.next_step()
            
        negative_prompt_embeds = torch.zeros_like(context_embedding)
            
        with torch.no_grad():
            image = pipe(
                prompt_embeds=context_embedding,
                negative_prompt_embeds=negative_prompt_embeds,
                num_inference_steps=50,
                guidance_scale=7.5,
                callback=callback_fn,
                callback_steps=1
            ).images[0]
            
        # Save Image
        image.save(os.path.join(args.output_dir, f"{idx}_generated.png"))
        
        # Process Attention Maps
        print("Processing Attention Maps...")
        
        obj_names = [vocab['object_idx_to_name'][i.item()] for i in objects[0]]
        
        labels = []
        # Objects
        for i, name in enumerate(obj_names):
            labels.append(f"Obj{i}: {name}")
            
        # Relations
        for i in range(triples_batch.size(1)):
            s, p, o = triples_batch[0, i].tolist()
            p_name = vocab['pred_idx_to_name'][p]
            s_name = obj_names[s] if s < len(obj_names) else "?"
            o_name = obj_names[o] if o < len(obj_names) else "?"
            labels.append(f"Rel: {s_name}-{p_name}-{o_name}")
            
        final_maps = aggregate_attention_maps()
        
        if final_maps is not None:
            visualize_and_save_maps(image, final_maps, labels, os.path.join(args.output_dir, f"{idx}_attention.png"))

if __name__ == "__main__":
    main()
