import torch
import sys

ckpt_path = "/root/Scene-Graph-Guided-Latent-Editing/logs/sgclip_new_embedding_v1/checkpoints/epoch_6.pt"
try:
    sd = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    
    keys = list(sd.keys())
    print(f"Total keys: {len(keys)}")
    
    diffusion_keys = [k for k in keys if k.startswith("model.diffusion_model")]
    print(f"Diffusion keys: {len(diffusion_keys)}")
    if len(diffusion_keys) > 0:
        print("Diffusion keys found (Example):", diffusion_keys[:5])
    else:
        print("No diffusion keys found.")
        
except Exception as e:
    print(f"Error loading checkpoint: {e}")
