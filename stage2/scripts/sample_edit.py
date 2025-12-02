import argparse
import torch
import os
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
from diffusers import StableDiffusionPipeline, DDIMScheduler

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gcn_wrapper import GCNWrapper
from models.graph_adapter import GraphAdapter
from processors.masactrl_processor import MasaCtrlSelfAttnProcessor
from processors.box_attn_processor import BoxGuidedCrossAttnProcessor
from stage2_utils.inversion import ddim_inversion, null_text_inversion, AttentionStore
from stage2_utils.warping import warp_tensor
from stage2_utils.data_loader import VGDataset, imagenet_preprocess

def register_attention_control(pipe, attn_store, is_inversion=False):
    # Helper to register MasaCtrl processors
    # We want to replace SelfAttention in UNet
    # And optionally CrossAttention
    
    # Traverse UNet
    # For simplicity, we replace all SelfAttn with MasaCtrl
    # And all CrossAttn with BoxGuided (if needed, but maybe not for now)
    
    attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        if name.endswith("attn1.processor"): # Self Attention
            attn_procs[name] = MasaCtrlSelfAttnProcessor(attn_store, is_inversion)
        else: # Cross Attention (attn2)
            # Keep default or use BoxGuided
            # For now, keep default unless we want box guidance
            attn_procs[name] = pipe.unet.attn_processors[name]
            
    pipe.unet.set_attn_processor(attn_procs)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default="checkpoints/adapter.pth")
    parser.add_argument("--image_index", type=int, default=2524, help="Index of the image in VG dataset to edit")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--edit_type", type=str, default="reconstruction", choices=["reconstruction", "move_object", "replace_object"])
    parser.add_argument("--inversion_type", type=str, default="ddim", choices=["ddim", "null_text"])
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of denoising steps")
    parser.add_argument("--source_object", type=str, default="horse", help="Object to replace (for replace_object edit)")
    parser.add_argument("--target_object", type=str, default="zebra", help="Target object (for replace_object edit)")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Models
    print("Loading models...")
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    
    gcn = GCNWrapper(
        vocab_file="./datasets/vg/vocab.json",
        checkpoint_path="./pretrained/sip_vg.pt"
    ).to(device)
    gcn.eval()
    
    adapter = GraphAdapter().to(device)
    if os.path.exists(args.adapter_path):
        checkpoint = torch.load(args.adapter_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            adapter.load_state_dict(checkpoint["model_state_dict"])
        else:
            adapter.load_state_dict(checkpoint)
        print(f"Loaded adapter from {args.adapter_path}")
    else:
        print(f"Warning: Adapter checkpoint {args.adapter_path} not found. Using random weights.")
    adapter.eval()
    
    # 2. Prepare Data
    print(f"Loading data (Index {args.image_index})...")
    dataset = VGDataset(
        vocab_path="./datasets/vg/vocab.json",
        h5_path="./datasets/vg/val.h5", # Use val set
        image_dir="./datasets/vg/images",
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
        local_feats, global_feats = gcn(graphs)

        cond_embeddings = adapter(local_feats)
        
        # Pad to 77 for SD compatibility (required for concatenation with uncond_embeddings)
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
    
    # If edit_type is move_object, we modify the boxes in the graph and re-compute embeddings?
    # Or we just use the warping in MasaCtrl.
    
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
    
    # If using standard DDIM, we can use pipe() but we need to ensure processors are used.
    # Since we registered processors, pipe() will use them.
    # But pipe() doesn't support per-step uncond embeddings.
    # So we use manual loop if null_text is used.
    
    if args.inversion_type == "null_text":
        print("Using Null-text generation loop...")
        for i, t in enumerate(tqdm(timesteps, desc="Generating")):
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
            image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            image = pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True]*image.shape[0])[0]
            
    else:
        # Standard DDIM Generation
        # We can use pipe directly
        # For reconstruction, we should use the same guidance scale as inversion (usually 1.0)
        # unless we use Null-text inversion to close the gap.
        # If inversion was done with scale 1.0 (default in ddim_inversion code), 
        # generation with 7.5 will fail if the condition is not perfect.
        
        guidance_scale = 7.5
        if args.edit_type == "reconstruction" and args.inversion_type == "ddim":
            print("For DDIM reconstruction, setting guidance_scale=1.0 to match inversion.")
            guidance_scale = 1.0
            
        image = pipe(
            prompt_embeds=cond_embeddings,
            latents=z_T,
            guidance_scale=guidance_scale,
            num_inference_steps=args.num_inference_steps
        ).images[0]
    
    image.save(os.path.join(args.output_dir, f"result_{args.image_index}_{args.edit_type}_{args.inversion_type}.png"))
    
    # Save original
    img_pil = Image.fromarray((img_unnorm[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    img_pil.save(os.path.join(args.output_dir, f"original_{args.image_index}.png"))
    
    print("Sampling done.")

if __name__ == "__main__":
    main()
