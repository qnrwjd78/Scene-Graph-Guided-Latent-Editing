import torch
import os
import sys
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gcn_wrapper import GCNWrapper
from models.graph_adapter import SceneGraphEmbedder
from stage2_utils.data_loader import VGDataset, vg_collate_fn
from scripts.train_adapter import prepare_batch_for_embedder

def check_gcn_health():
    # Force CPU for debugging to avoid OOM if training is running
    device = torch.device("cpu")
    print(f"Using device: {device}")

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 1. Load GCN
    print("Loading GCN...")
    gcn = GCNWrapper(
        vocab_file=os.path.join(project_root, "datasets/vg/vocab.json"),
        checkpoint_path=os.path.join(project_root, "pretrained/sip_vg.pt") 
    ).to(device)
    gcn.eval()
    
    # 2. Load Adapter
    print("Loading Adapter...")
    adapter = SceneGraphEmbedder().to(device)
    adapter.eval()

    # 3. Load Data (Batch size 4 to compare different items)
    print("Loading Dataset...")
    dataset = VGDataset(
        vocab_path=os.path.join(project_root, "datasets/vg/vocab.json"),
        h5_path=os.path.join(project_root, "datasets/vg/train.h5"),
        image_dir=os.path.join(project_root, "datasets/vg/images"),
        image_size=(512, 512),
        normalize_images=False,
        max_samples=100
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=vg_collate_fn)
    
    # 4. Get a batch
    batch = next(iter(dataloader))
    images, objects, boxes, triples, obj_to_img, triple_to_img = batch
    
    objects = objects.to(device)
    boxes = boxes.to(device)
    triples = triples.to(device)
    obj_to_img = obj_to_img.to(device)
    triple_to_img = triple_to_img.to(device)
    
    graphs = [objects, boxes, triples, obj_to_img, triple_to_img]
    
    print("\n--- Input Data Stats ---")
    print(f"Batch Size: {images.shape[0]}")
    print(f"Num Objects: {objects.shape[0]}")
    print(f"Num Triples: {triples.shape[0]}")
    
    # 5. Run GCN
    print("\n--- Running GCN ---")
    with torch.no_grad():
        obj_vecs, pred_vecs = gcn.get_raw_features(graphs)
        
    print(f"Object Vectors Shape: {obj_vecs.shape}")
    print(f"Predicate Vectors Shape: {pred_vecs.shape}")
    
    # 6. Prepare Batch for Adapter
    print("\n--- Preparing Batch for Adapter ---")
    gcn_vectors, token_types, obj_idx, sub_ptr, obj_ptr = prepare_batch_for_embedder(
        obj_vecs, pred_vecs, triples, obj_to_img, triple_to_img, device
    )
    
    print(f"GCN Vectors (Padded): {gcn_vectors.shape}")
    print(f"Token Types: {token_types.shape}")
    
    # 7. Run Adapter
    print("\n--- Running Adapter ---")
    with torch.no_grad():
        x_mixed = adapter(gcn_vectors, token_types, obj_idx, sub_ptr, obj_ptr)
        
    print(f"Adapter Output Shape: {x_mixed.shape}")
    print(f"Adapter Output Mean: {x_mixed.mean().item():.4f}, Std: {x_mixed.std().item():.4f}")
    
    # 8. Check Diversity of Adapter Output
    # Compare the first token (usually an object) of each batch item
    first_tokens = x_mixed[:, 0, :] # (Batch, Dim)
    
    # Normalize
    first_tokens_norm = F.normalize(first_tokens, p=2, dim=1)
    sim_matrix = torch.mm(first_tokens_norm, first_tokens_norm.t())
    
    print("\n--- Cosine Similarity between Adapter outputs (first token) ---")
    print(sim_matrix)
    
    mask = ~torch.eye(images.shape[0], dtype=bool, device=device)
    avg_sim = sim_matrix[mask].mean().item()
    print(f"\nAverage Similarity between Adapter outputs: {avg_sim:.4f}")
    
    if avg_sim > 0.99:
        print("!!! CRITICAL: Adapter outputs are identical across batch. Something is wrong.")
    else:
        print("OK: Adapter outputs are distinct.")

if __name__ == "__main__":
    check_gcn_health()
