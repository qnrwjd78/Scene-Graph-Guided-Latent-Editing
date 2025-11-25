import argparse
import torch
import os
import sys
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gcn_wrapper import GCNWrapper
from models.graph_adapter import GraphAdapter
from utils.data_loader import VGDataset, vg_collate_fn
import torch.nn.functional as F

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Models
    # SD (Frozen)
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
    pipe.to(device)
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(False)
    
    # GCN (Frozen)
    gcn = GCNWrapper(
        vocab_file="../Data/datasets/vg/vocab.json",
        checkpoint_path="../pretrained/sip_vg.pt" # Adjust path
    ).to(device)
    
    # Adapter (Trainable)
    adapter = GraphAdapter().to(device)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=args.lr)
    
    # 2. Data
    dataset = VGDataset(
        vocab_path="../Data/datasets/vg/vocab.json",
        h5_path="../Data/datasets/vg/train.h5",
        image_dir="../Data/datasets/vg/images"
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=vg_collate_fn)
    
    # 3. Training Loop
    print("Starting training...")
    for epoch in range(args.epochs):
        for batch in dataloader:
            # Unpack batch
            # vg_collate_fn returns: (imgs, objs, boxes, triples, obj_to_img, triple_to_img)
            images, objects, boxes, triples, obj_to_img, triple_to_img = batch
            
            images = images.to(device)
            objects = objects.to(device)
            boxes = boxes.to(device)
            triples = triples.to(device)
            obj_to_img = obj_to_img.to(device)
            triple_to_img = triple_to_img.to(device)
            
            graphs = [objects, boxes, triples, obj_to_img, triple_to_img]
            
            # 1. Encode Image -> Latents
            with torch.no_grad():
                latents = pipe.vae.encode(images).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
            
            # 2. Encode Graph
            # GCNWrapper needs to return the graph embedding.
            # We assume GCNWrapper returns (local, global).
            local_graph_feats, global_graph_feats = gcn(graphs)
            
            # Pad local features to (Batch, Max_Objs, Dim)
            max_objs = 0
            for i in range(args.batch_size):
                n_objs = (obj_to_img == i).sum().item()
                if n_objs > max_objs:
                    max_objs = n_objs
            
            batch_local_feats = torch.zeros(args.batch_size, max_objs, local_graph_feats.shape[1], device=device)
            
            for i in range(args.batch_size):
                mask = (obj_to_img == i)
                objs_feats = local_graph_feats[mask]
                n = objs_feats.shape[0]
                batch_local_feats[i, :n] = objs_feats
                
            # Adapt features
            cond_embeds = adapter(batch_local_feats) # (B, N, 768)
            
            # 3. Forward SD
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, pipe.scheduler.num_train_timesteps, (latents.shape[0],), device=device)
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
            
            pred_noise = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=cond_embeds).sample
            
            # 4. Loss
            loss = F.mse_loss(pred_noise, noise)
            
            # 5. Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch} Loss: {loss.item()}")
            
        print(f"Epoch {epoch} completed.")
        
    # Save
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(adapter.state_dict(), os.path.join(args.save_dir, "adapter.pth"))

if __name__ == "__main__":
    main()
