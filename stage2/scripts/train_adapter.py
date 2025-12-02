import argparse
import torch
import os
import sys
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gcn_wrapper import GCNWrapper
from models.graph_adapter import SceneGraphEmbedder
from models.dual_lora import DualInputLoRALinear
from stage2_utils.data_loader import VGDataset, vg_collate_fn

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
        # If no objects, handle gracefully (though VG usually has objects)
        if num_objs == 0:
             # Create dummy
             curr_seq_vecs = torch.zeros((0, curr_obj_vecs.shape[-1]), device=device)
        else:
             curr_seq_vecs = torch.cat([curr_obj_vecs, curr_pred_vecs], dim=0)
        
        # 3. Build Indices
        # Token Types: 0 for Obj, 1 for Rel
        curr_token_types = [0] * num_objs + [1] * num_rels
        
        # Object Index: 0..N-1 for Objs, 0 for Rels
        curr_obj_idx = list(range(num_objs)) + [0] * num_rels
        
        # Pointers
        # Need to map global object indices to local indices
        global_obj_indices = torch.where(curr_obj_mask)[0]
        global_to_local = {g_idx.item(): l_idx for l_idx, g_idx in enumerate(global_obj_indices)}
        
        curr_triples = triples[curr_pred_mask] # (T, 3) -> s, p, o (global indices)
        
        curr_sub_ptr = [0] * num_objs
        curr_obj_ptr = [0] * num_objs
        
        for t in curr_triples:
            s_global, _, o_global = t.tolist()
            s_local = global_to_local.get(s_global, 0) # Default to 0 if not found (shouldn't happen)
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
    # Helper to pad
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
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit the number of training samples for debugging")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.")
    args = parser.parse_args()
    
    # Setup logging
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.save_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tensorboard'))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Models
    # SD (Frozen)
    try:
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", local_files_only=True)
    except Exception:
        print("Model not found in cache, downloading...")
        pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    
    # Inject Dual LoRA
    lora_params = inject_dual_lora(pipe.unet)
    print(f"Injected Dual LoRA. Trainable LoRA params: {len(lora_params)}")
    
    # GCN (Frozen)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gcn = GCNWrapper(
        vocab_file=os.path.join(project_root, "datasets/vg/vocab.json"),
        checkpoint_path=os.path.join(project_root, "pretrained/sip_vg.pt") 
    ).to(device)
    
    # Adapter (Trainable)
    adapter = SceneGraphEmbedder().to(device)
    
    # Optimizer: Adapter + LoRA
    optimizer = torch.optim.Adam([
        {'params': adapter.parameters()},
        {'params': lora_params}
    ], lr=args.lr)
    
    start_epoch = 0
    global_step = 0
    
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                adapter.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'epoch' in checkpoint:
                    start_epoch = checkpoint['epoch']
                if 'global_step' in checkpoint:
                    global_step = checkpoint['global_step']
                
                # Load LoRA weights if available
                if 'lora_state_dict' in checkpoint:
                    pipe.unet.load_state_dict(checkpoint['lora_state_dict'], strict=False)
                    
                print(f"Resumed from epoch {start_epoch}, global step {global_step}")
            else:
                # Old format (just adapter)
                adapter.load_state_dict(checkpoint)
                print("Loaded adapter weights from old format checkpoint")
        else:
            print(f"No checkpoint found at {args.resume}")

    # 2. Data
    dataset = VGDataset(
        vocab_path=os.path.join(project_root, "datasets/vg/vocab.json"),
        h5_path=os.path.join(project_root, "datasets/vg/train.h5"),
        image_dir=os.path.join(project_root, "datasets/vg/images"),
        image_size=(512, 512),
        normalize_images=False,
        max_samples=args.max_samples
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=vg_collate_fn)
    
    # 3. Training Loop
    print("Starting training...")
    logging.info("Starting training...")
    
    for epoch in range(start_epoch, args.epochs):
        adapter.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            # Unpack batch
            images, objects, boxes, triples, obj_to_img, triple_to_img = batch
            
            images = images.to(device)
            objects = objects.to(device)
            boxes = boxes.to(device)
            triples = triples.to(device)
            obj_to_img = obj_to_img.to(device)
            triple_to_img = triple_to_img.to(device)
            
            graphs = [objects, boxes, triples, obj_to_img, triple_to_img]
            
            # Normalize images to [-1, 1]
            images = images * 2.0 - 1.0
            
            # 1. Encode Image -> Latents
            with torch.no_grad():
                latents = pipe.vae.encode(images).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
            
            # 2. Encode Graph
            with torch.no_grad():
                obj_vecs, pred_vecs = gcn.get_raw_features(graphs)
            
            # 3. Prepare Batch for Embedder
            gcn_vectors, token_types, obj_idx, sub_ptr, obj_ptr = prepare_batch_for_embedder(
                obj_vecs, pred_vecs, triples, obj_to_img, triple_to_img, device
            )
            
            # 4. Adapter Forward
            x_clean, x_mixed = adapter(gcn_vectors, token_types, obj_idx, sub_ptr, obj_ptr)
            
            # 5. Combine for Dual Input
            # Concatenate along feature dimension: (B, Seq, 768*2)
            context_embedding = torch.cat([x_clean, x_mixed], dim=-1)
            
            # 6. Diffusion Loss
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise
            # Pass context_embedding to encoder_hidden_states
            noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=context_embedding).sample
            
            loss = F.mse_loss(noise_pred, noise)
            
            # Gradient Accumulation
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            global_step += 1
            pbar.set_postfix(loss=loss.item() * args.gradient_accumulation_steps)
            writer.add_scalar("Loss/train", loss.item() * args.gradient_accumulation_steps, global_step)
            
        # Save Checkpoint
        save_path = os.path.join(args.save_dir, f"adapter_epoch_{epoch+1}.pth")
        
        # Save LoRA weights
        lora_state_dict = {k: v for k, v in pipe.unet.state_dict().items() if "lora" in k}
        
        torch.save({
            'epoch': epoch + 1,
            'global_step': global_step,
            'model_state_dict': adapter.state_dict(),
            'lora_state_dict': lora_state_dict,
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    main()
