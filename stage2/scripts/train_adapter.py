import argparse
import torch
import os
import sys
import logging
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gcn_wrapper import GCNWrapper
from models.graph_adapter import SceneGraphEmbedder
from models.dual_lora import DualInputLoRALinear
from stage2_utils.data_loader import VGDataset, vg_collate_fn
from stage2_utils.attention_map import AttentionStore, register_attention_map_saver
from stage2_utils.visualization import draw_scene_graph_matplotlib

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
        global_to_local = {g_idx.item(): l_idx for l_idx, g_idx in enumerate(global_obj_indices)}
        
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
    parser.add_argument("--lora_lr", type=float, default=1e-4, help="Learning rate for LoRA")
    parser.add_argument("--adapter_lr", type=float, default=1e-3, help="Learning rate for Adapter")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--tensorboard_dir", type=str, default=None, help="Custom path for tensorboard logs")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit the number of training samples")
    parser.add_argument("--max_val_samples", type=int, default=1000, help="Limit the number of validation samples")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="Number of epochs to freeze LoRA and train Adapter only")
    parser.add_argument("--unfreeze_steps", type=int, default=-1, help="Number of steps to freeze LoRA. Overrides warmup_epochs if provided.")
    parser.add_argument("--scheduler_warmup_steps", type=int, default=1000, help="Number of steps for the learning rate scheduler warmup.")
    parser.add_argument("--condition_dropout_prob", type=float, default=0.1, help="Probability of dropping the condition (unconditional training)")
    parser.add_argument("--dataset", type=str, default="vg", help="Dataset to use: vg or vg_clevr")
    args = parser.parse_args()
    
    # Set Seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
            # Ensure deterministic behavior for CuDNN (Convolution operations)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    # Setup logging
    os.makedirs(args.save_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.save_dir, 'training.log'),
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Tensorboard setup
    if args.tensorboard_dir:
        log_dir = args.tensorboard_dir
    else:
        # Add timestamp to log_dir to distinguish experiments
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(args.save_dir, 'tensorboard', timestamp)
    writer = SummaryWriter(log_dir=log_dir)
    
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
    
    # Pre-compute null embedding for condition dropout
    null_input = pipe.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt")
    with torch.no_grad():
        null_embedding = pipe.text_encoder(null_input.input_ids.to(device))[0]
    
    # Inject Dual LoRA
    lora_params = inject_dual_lora(pipe.unet)
    print(f"Injected Dual LoRA. Trainable LoRA params: {len(lora_params)}")
    
    # GCN (Frozen)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    gcn = GCNWrapper(
        vocab_file=os.path.join(project_root, f"datasets/{args.dataset}/vocab.json"),
        checkpoint_path=os.path.join(project_root, "pretrained/sip_vg.pt") 
    ).to(device)
    
    # Adapter (Trainable)
    adapter = SceneGraphEmbedder().to(device)
    
    # Optimizer: Adapter (Higher LR) + LoRA (Lower LR)
    # Adapter needs to learn mapping from scratch -> args.adapter_lr
    # LoRA fine-tunes pre-trained weights -> args.lora_lr
    optimizer = torch.optim.Adam([
        {'params': adapter.parameters(), 'lr': args.adapter_lr},
        {'params': lora_params, 'lr': args.lora_lr}
    ], lr=args.lora_lr)
    
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
        vocab_path=os.path.join(project_root, f"datasets/{args.dataset}/vocab.json"),
        h5_path=os.path.join(project_root, f"datasets/{args.dataset}/train.h5"),
        image_dir=os.path.join(project_root, f"datasets/{args.dataset}/images"),
        image_size=(512, 512),
        normalize_images=False,
        max_samples=args.max_train_samples
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=vg_collate_fn)
    
    # Scheduler
    num_update_steps_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.epochs * num_update_steps_per_epoch
    
    # LoRA Freezing Steps
    if args.unfreeze_steps >= 0:
        num_freeze_steps = args.unfreeze_steps
    else:
        num_freeze_steps = args.warmup_epochs * num_update_steps_per_epoch
    
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.scheduler_warmup_steps,
        num_training_steps=max_train_steps
    )
    
    # Validation Data
    val_dataset = VGDataset(
        vocab_path=os.path.join(project_root, f"datasets/{args.dataset}/vocab.json"),
        h5_path=os.path.join(project_root, f"datasets/{args.dataset}/val.h5"),
        image_dir=os.path.join(project_root, f"datasets/{args.dataset}/images"),
        image_size=(512, 512),
        normalize_images=False,
        max_samples=args.max_val_samples
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=vg_collate_fn)
    
    # 3. Training Loop
    print("Starting training...")
    logging.info("Starting training...")
    
    # Initial Warmup Check
    current_update_step = global_step // args.gradient_accumulation_steps
    is_frozen = current_update_step < num_freeze_steps
    
    if is_frozen:
        print(f"Starting in Frozen Phase (Step {current_update_step}/{num_freeze_steps}). Freezing LoRA.")
        for p in lora_params:
            p.requires_grad = False
    else:
        print(f"Starting in Normal Training Phase. LoRA is trainable.")
        for p in lora_params:
            p.requires_grad = True

    for epoch in range(start_epoch, args.epochs):
        adapter.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        accumulated_loss = 0.0
        accumulated_loss_cond = 0.0
        accumulated_loss_uncond = 0.0
        steps_cond = 0
        steps_uncond = 0
        
        for batch in pbar:
            # Update Freeze Status
            current_update_step = global_step // args.gradient_accumulation_steps
            
            # Check for transition from Frozen -> Normal
            if is_frozen and current_update_step >= num_freeze_steps:
                is_frozen = False
                print(f"\nFreezing completed at step {current_update_step}. Unfreezing LoRA.")
                for p in lora_params:
                    p.requires_grad = True

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
            x_mixed = adapter(gcn_vectors, token_types, obj_idx, sub_ptr, obj_ptr)
            
            # 5. Single Input Strategy
            # Condition Dropout
            # Only apply dropout if NOT in frozen phase (because in frozen phase LoRA is frozen, so dropping condition disconnects the graph)
            use_null = False
            if not is_frozen and random.random() < args.condition_dropout_prob:
                use_null = True
                # Use null embedding (Unconditional)
                context_embedding = null_embedding.repeat(images.shape[0], 1, 1)
            else:
                # Use adapter output
                context_embedding = x_mixed
            
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
            
            # Accumulate loss for logging
            accumulated_loss += loss.item()
            
            if use_null:
                accumulated_loss_uncond += loss.item() * args.gradient_accumulation_steps
                steps_uncond += 1
            else:
                accumulated_loss_cond += loss.item() * args.gradient_accumulation_steps
                steps_cond += 1
            
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                # Log Gradient Norms (Before optimizer step)
                total_adapter_norm = 0.0
                for p in adapter.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_adapter_norm += param_norm.item() ** 2
                total_adapter_norm = total_adapter_norm ** 0.5
                writer.add_scalar("Gradients/Adapter_Norm", total_adapter_norm, global_step)

                total_lora_norm = 0.0
                for p in lora_params:
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_lora_norm += param_norm.item() ** 2
                total_lora_norm = total_lora_norm ** 0.5
                writer.add_scalar("Gradients/LoRA_Norm", total_lora_norm, global_step)

                # Log Embedding Gradients (PosEmb)
                emb_layers = {
                    "Type": adapter.type_emb,
                    "SelfIdx": adapter.self_idx_emb,
                    "SubPtr": adapter.sub_ptr_emb,
                    "ObjPtr": adapter.obj_ptr_emb
                }
                
                for name, layer in emb_layers.items():
                    if layer.weight.grad is not None:
                        grad_norm = layer.weight.grad.data.norm(2).item()
                        writer.add_scalar(f"Gradients/Emb_{name}", grad_norm, global_step)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                # Log LR
                current_lr_adapter = optimizer.param_groups[0]['lr']
                current_lr_lora = optimizer.param_groups[1]['lr']
                writer.add_scalar("LR/Adapter", current_lr_adapter, global_step)
                writer.add_scalar("LR/LoRA", current_lr_lora, global_step)
                
                # Log average loss over accumulation steps
                writer.add_scalar("Loss/train", accumulated_loss, global_step)
                
                if steps_cond > 0:
                    writer.add_scalar("Loss/train_conditional", accumulated_loss_cond / steps_cond, global_step)
                if steps_uncond > 0:
                    writer.add_scalar("Loss/train_unconditional", accumulated_loss_uncond / steps_uncond, global_step)
                
                accumulated_loss = 0.0
                accumulated_loss_cond = 0.0
                accumulated_loss_uncond = 0.0
                steps_cond = 0
                steps_uncond = 0
            
            # --- Sampling every 2000 steps ---
            if global_step % 5000 == 0 and global_step > 0:
                print(f"\n[Sampling] Generating samples at step {global_step}...")
                sample_save_dir = os.path.join(args.save_dir, "sample", str(global_step))
                os.makedirs(sample_save_dir, exist_ok=True)
                
                # Switch to eval
                adapter.eval()
                pipe.unet.eval()
                
                try:
                    # Use the first item in the current batch
                    idx = 0
                    
                    # 1. Prepare Condition (Use x_mixed from training step)
                    # x_mixed is (Batch, Seq, Dim). Take first one.
                    sample_cond_embedding = x_mixed[idx:idx+1] # (1, Seq, Dim)
                    
                    # 2. Prepare Uncondition
                    # Resize null_embedding to match sample_cond_embedding length
                    cond_seq_len = sample_cond_embedding.shape[1]
                    if cond_seq_len <= null_embedding.shape[1]:
                        sample_uncond_embedding = null_embedding[:, :cond_seq_len, :]
                    else:
                        # Pad null_embedding if condition is longer
                        pad_len = cond_seq_len - null_embedding.shape[1]
                        # Use the last token (padding) to extend
                        pad_tensor = null_embedding[:, -1:, :].repeat(1, pad_len, 1)
                        sample_uncond_embedding = torch.cat([null_embedding, pad_tensor], dim=1)
                    
                    # 3. Setup Scheduler (DDIM)
                    original_scheduler = pipe.scheduler
                    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
                    
                    # 4. Setup Attention Saver
                    old_procs = pipe.unet.attn_processors
                    
                    # 5. Generate (CFG 7.5) - No Attention Map Saving
                    with torch.no_grad():
                        gen_image = pipe(
                            prompt_embeds=sample_cond_embedding,
                            negative_prompt_embeds=sample_uncond_embedding,
                            num_inference_steps=50,
                            guidance_scale=7.5,
                            output_type="pil"
                        ).images[0]
                        
                    # 5-2. Generate (No CFG / Guidance Scale 1.0) - Save Attention Map
                    # Register Saver ONLY for this generation
                    attn_store = AttentionStore(save_dir=sample_save_dir, res=32)
                    register_attention_map_saver(pipe, attn_store)
                    
                    with torch.no_grad():
                        gen_image_no_cfg = pipe(
                            prompt_embeds=sample_cond_embedding,
                            negative_prompt_embeds=sample_uncond_embedding,
                            num_inference_steps=50,
                            guidance_scale=1.0,
                            output_type="pil"
                        ).images[0]
                    
                    # 6. Restore
                    pipe.scheduler = original_scheduler
                    pipe.unet.set_attn_processor(old_procs)
                    
                    # 7. Save Generated Image
                    gen_image.save(os.path.join(sample_save_dir, "sample_cfg7.5.png"))
                    gen_image_no_cfg.save(os.path.join(sample_save_dir, "sample_cfg1.0.png"))
                    
                    # 8. Save Original Image
                    # images is [-1, 1]. Unnormalize -> [0, 1] -> [0, 255]
                    orig_tensor = images[idx].cpu()
                    orig_np = (orig_tensor.permute(1, 2, 0).numpy() + 1) / 2
                    orig_np = (orig_np * 255).clip(0, 255).astype(np.uint8)
                    Image.fromarray(orig_np).save(os.path.join(sample_save_dir, "original.png"))
                    
                    # 9. Save Attention Map & Graph
                    # Reconstruct labels for the first image
                    mask_obj = (obj_to_img == idx)
                    mask_triple = (triple_to_img == idx)
                    
                    curr_objects = objects[mask_obj]
                    curr_triples = triples[mask_triple]
                    
                    # Filter out __image__ token (last object)
                    if curr_objects.shape[0] > 0:
                        curr_objects = curr_objects[:-1]
                        # Filter triples connected to last object
                        last_obj_idx = objects[mask_obj].shape[0] - 1
                        # Note: triples indices are global. We need local indices for filtering logic if we used local.
                        # But here triples are global indices.
                        # Let's just use the names from dataset.vocab
                        
                        obj_names = []
                        for o_idx in curr_objects:
                            name = dataset.vocab['object_idx_to_name'][o_idx.item()]
                            obj_names.append(name)
                            
                        # For triples, we need to map global indices to local to find names
                        # This is complicated because triples store global indices.
                        # Let's simplify: Just save the object names + predicate names.
                        # The attention map saver expects a list of strings matching the token sequence.
                        # The token sequence is [Obj1, Obj2, ..., Rel1, Rel2, ...]
                        
                        labels = obj_names[:]
                        
                        # Filter triples for visualization and labels
                        # We need to know which triples correspond to the filtered objects.
                        # The model input `x_mixed` already filtered out the __image__ node and its relations.
                        # So we just need to find the relations that were kept.
                        # In `prepare_batch_for_embedder`, we filtered:
                        # valid_pred_mask = (curr_triples[:, 2] != sep_node_global_idx)
                        
                        global_obj_indices = torch.where(mask_obj)[0]
                        if len(global_obj_indices) > 0:
                            sep_node_global_idx = global_obj_indices[-1]
                            valid_pred_mask = (curr_triples[:, 2] != sep_node_global_idx)
                            valid_triples = curr_triples[valid_pred_mask]
                            
                            for t in valid_triples:
                                p_idx = t[1].item()
                                p_name = dataset.vocab['pred_idx_to_name'][p_idx]
                                labels.append(p_name)
                            
                            attn_store.save_attention_maps(labels, save_name="attn_map")
                            
                            # Save Graph Viz
                            # draw_scene_graph_matplotlib expects local indices or raw data?
                            # It expects (objects, triples, vocab).
                            # We should pass the filtered ones.
                            draw_scene_graph_matplotlib(curr_objects, valid_triples, dataset.vocab, os.path.join(sample_save_dir, "graph.png"))

                except Exception as e:
                    print(f"Sampling failed: {e}")
                    # Restore just in case
                    if 'old_procs' in locals():
                        pipe.unet.set_attn_processor(old_procs)
                    if 'original_scheduler' in locals():
                        pipe.scheduler = original_scheduler

                # Switch back to train
                adapter.train()
                if not is_frozen:
                    pipe.unet.train()

            global_step += 1
            pbar.set_postfix(loss=loss.item() * args.gradient_accumulation_steps)
            
        # Validation Loop
        adapter.eval()
        val_loss = 0.0
        val_steps = 0
        print("Running validation...")
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                images, objects, boxes, triples, obj_to_img, triple_to_img = batch
                
                images = images.to(device)
                objects = objects.to(device)
                boxes = boxes.to(device)
                triples = triples.to(device)
                obj_to_img = obj_to_img.to(device)
                triple_to_img = triple_to_img.to(device)
                
                graphs = [objects, boxes, triples, obj_to_img, triple_to_img]
                
                # Normalize images
                images = images * 2.0 - 1.0
                
                # Encode Image
                latents = pipe.vae.encode(images).latent_dist.sample()
                latents = latents * pipe.vae.config.scaling_factor
                
                # Encode Graph
                obj_vecs, pred_vecs = gcn.get_raw_features(graphs)
                
                # Prepare Batch
                gcn_vectors, token_types, obj_idx, sub_ptr, obj_ptr = prepare_batch_for_embedder(
                    obj_vecs, pred_vecs, triples, obj_to_img, triple_to_img, device
                )
                
                # Adapter Forward
                x_mixed = adapter(gcn_vectors, token_types, obj_idx, sub_ptr, obj_ptr)
                context_embedding = x_mixed
                
                # Diffusion Loss
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                
                noise_pred = pipe.unet(noisy_latents, timesteps, encoder_hidden_states=context_embedding).sample
                
                loss = F.mse_loss(noise_pred, noise)
                val_loss += loss.item()
                val_steps += 1
                
        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0.0
        print(f"Validation Loss: {avg_val_loss:.4f}")
        writer.add_scalar("Loss/validation", avg_val_loss, epoch + 1)
        
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
