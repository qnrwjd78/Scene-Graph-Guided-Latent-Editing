import torch
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class AttentionStore:
    def __init__(self, save_dir=None, res=32):
        self.store = {} # Key: "down", "up", "mid" -> List of maps
        self.save_dir = save_dir
        self.res = res # Target resolution to store (e.g., 32 for 32x32)
        self.step_count = 0
        
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        # attn shape: (Batch*Head, H*W, Seq_Len)
        # We only care about Cross Attention for object localization
        if not is_cross:
            return
        
        # Check resolution
        # H*W = res*res
        hw = attn.shape[1]
        h = int(hw ** 0.5)
        
        # Filter: Only store 32x32 maps (Up-block 1 usually)
        if h != self.res:
            return
            
        # Store
        # We average over heads
        # attn: (B*Head, H*W, Seq) -> (B, Head, H*W, Seq)
        # Assuming Batch=2 (Uncond, Cond) or Batch=1
        # We usually want the Conditional part.
        
        # Reshape to (Batch, Head, H*W, Seq)
        # But we don't know Head count easily here without config.
        # Usually Head=8 for SD v1.5
        # Let's just average over the first dimension for now, 
        # but we need to separate Uncond/Cond if CFG is used.
        # If CFG is used, Batch is 2 * Actual_Batch.
        
        # For simplicity in visualization, we just store the raw tensor and process later.
        # Or we can accumulate here.
        
        key = f"{place_in_unet}_{h}"
        if key not in self.store:
            self.store[key] = []
            
        self.store[key].append(attn.detach().cpu())

    def reset(self):
        self.store = {}
        self.step_count = 0
        
    def get_average_attention(self):
        """
        Returns the average attention map across all steps and heads.
        Target: 32x32 maps.
        """
        # Collect all 32x32 maps
        maps = []
        for key, val_list in self.store.items():
            if str(self.res) in key:
                # val_list is a list of tensors (Steps)
                # Each tensor: (Batch*Head, H*W, Seq)
                for t in val_list:
                    maps.append(t)
        
        if not maps:
            return None
            
        # Stack: (Total_Steps, B*Head, H*W, Seq)
        stack = torch.stack(maps, dim=0)
        
        # Average over Steps and Heads
        # (B*Head, H*W, Seq)
        avg_map = torch.mean(stack, dim=0)
        
        # Split Batch (Uncond, Cond) if CFG used (Batch size usually 2)
        # We assume the second half is Conditional (standard in Diffusers)
        # If Batch=2 (1 image), then index 0 is Uncond, 1 is Cond.
        # If Batch=2*N, we need to be careful.
        # Let's assume Batch=2 for single image generation.
        
        if avg_map.shape[0] % 2 == 0:
            half = avg_map.shape[0] // 2
            cond_map = avg_map[half:] # (Head, H*W, Seq)
        else:
            cond_map = avg_map # (Head, H*W, Seq)
            
        # Average over Heads
        # (Head, H*W, Seq) -> (H*W, Seq)
        final_map = torch.mean(cond_map, dim=0)
        
        # Reshape to (H, W, Seq)
        seq_len = final_map.shape[-1]
        final_map = final_map.view(self.res, self.res, seq_len)
        
        return final_map

    def save_attention_maps(self, tokens, save_name="attn_map"):
        """
        Saves attention maps for each token.
        tokens: List of token names or indices.
        """
        if self.save_dir is None:
            return
            
        avg_map = self.get_average_attention() # (32, 32, Seq)
        if avg_map is None:
            print("No attention maps found for resolution", self.res)
            return
            
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Normalize to 0-1
        # avg_map = (avg_map - avg_map.min()) / (avg_map.max() - avg_map.min())
        
        num_tokens = min(len(tokens), avg_map.shape[-1])
        
        # Plot
        fig, axes = plt.subplots(1, num_tokens, figsize=(num_tokens * 3, 3))
        if num_tokens == 1:
            axes = [axes]
            
        for i in range(num_tokens):
            # Get map for token i
            # Note: Token 0 is usually Start-of-Sentence (SoS) in CLIP, 
            # but in our Graph Adapter, it depends on how we constructed the input.
            # Our input is [Obj1, Obj2, ..., Rel1, Rel2...]
            # So index i corresponds to i-th element in our graph sequence.
            
            attn_img = avg_map[:, :, i].numpy()
            
            # Resize to 256x256 for better view
            attn_img = Image.fromarray((attn_img * 255).astype(np.uint8))
            attn_img = attn_img.resize((256, 256), resample=Image.BICUBIC)
            
            axes[i].imshow(attn_img, cmap='jet')
            axes[i].set_title(f"{tokens[i]}")
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f"{save_name}.png"))
        plt.close()
        print(f"Saved attention map to {os.path.join(self.save_dir, f'{save_name}.png')}")

# Processor to hook into UNet
class AttentionMapSaverProcessor:
    def __init__(self, store, place_in_unet):
        self.store = store
        self.place_in_unet = place_in_unet # "down", "mid", "up"

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        # Standard Attention Forward
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross = encoder_hidden_states is not None
        
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif hasattr(attn, "norm_cross") and attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # --- Hook: Save Attention ---
        # attention_probs: (Batch*Head, H*W, Seq)
        if self.store is not None:
            self.store(attention_probs, is_cross, self.place_in_unet)
        # ----------------------------

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

def register_attention_map_saver(pipe, store):
    """
    Replaces all attention processors with AttentionMapSaverProcessor.
    """
    attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        # Determine place
        if "down_blocks" in name:
            place = "down"
        elif "mid_block" in name:
            place = "mid"
        elif "up_blocks" in name:
            place = "up"
        else:
            place = "unknown"
            
        attn_procs[name] = AttentionMapSaverProcessor(store, place)
        
    pipe.unet.set_attn_processor(attn_procs)
