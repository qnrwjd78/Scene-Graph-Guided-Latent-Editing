import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from einops import rearrange
from diffusers.models.attention_processor import Attention

# ============================================================================
# Attention Store and Processor
# ============================================================================

class AttentionStore:
    def __init__(self):
        self.step_store = {}  # Stores maps for the current step
        self.all_maps = []    # List of step_stores
        
    def store(self, attn, layer_name):
        if layer_name not in self.step_store:
            self.step_store[layer_name] = []
        # attn shape: (Batch*Heads, Query, Key)
        # We detach and move to CPU to save memory
        self.step_store[layer_name].append(attn.detach().cpu())
        
    def next_step(self):
        self.all_maps.append(self.step_store)
        self.step_store = {}
        
    def reset(self):
        self.step_store = {}
        self.all_maps = []

GLOBAL_ATTENTION_STORE = AttentionStore()

class DAAMCrossAttnProcessor:
    def __init__(self, layer_name):
        self.layer_name = layer_name

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None, **kwargs):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        
        # --- DAAM Storage ---
        # Only store if it's cross attention (encoder_hidden_states is not None)
        GLOBAL_ATTENTION_STORE.store(attention_probs, self.layer_name)
        # --------------------

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

def register_daam_processors(pipe):
    """
    Registers DAAM processors for all Cross-Attention layers in the UNet.
    """
    attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        if name.endswith("attn2.processor"): # Cross Attention
            attn_procs[name] = DAAMCrossAttnProcessor(layer_name=name)
        else:
            attn_procs[name] = pipe.unet.attn_processors[name]
    pipe.unet.set_attn_processor(attn_procs)

# ============================================================================
# Visualization Utils
# ============================================================================

def aggregate_attention_maps(target_size=(64, 64)):
    """
    Aggregates attention maps across all steps and layers.
    Returns:
        final_maps: (K, H, W) tensor where K is the number of tokens (context length)
    """
    final_maps = None
    count = 0
    
    for step_data in GLOBAL_ATTENTION_STORE.all_maps:
        for layer_name, attn_list in step_data.items():
            for attn in attn_list:
                # attn: (B*H, Q, K)
                # Average over heads
                attn = attn.mean(dim=0) # (Q, K)
                
                Q, K = attn.shape
                h = int(np.sqrt(Q))
                
                # Reshape Q -> (h, w)
                # Check if Q is square
                if h * h != Q:
                    continue

                attn_2d = rearrange(attn, '(h w) k -> k h w', h=h, w=h)
                
                # Resize to target size
                attn_2d = F.interpolate(attn_2d.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
                
                if final_maps is None:
                    final_maps = torch.zeros_like(attn_2d)
                
                # Handle padding if K differs (shouldn't happen for same image)
                if final_maps.shape[0] != K:
                    # If K is different, we might have issues. But for one image, K is fixed (seq len).
                    pass
                    
                final_maps += attn_2d
                count += 1
                
    if final_maps is not None and count > 0:
        final_maps /= count
        
    return final_maps

def visualize_and_save_maps(image, final_maps, labels, output_path):
    """
    Visualizes attention maps overlaid on the image and saves the plot.
    """
    if final_maps is None:
        print("No attention maps to visualize.")
        return

    num_tokens = len(labels)
    
    # Plot
    cols = 5
    rows = (num_tokens + 1 + cols - 1) // cols
    plt.figure(figsize=(20, 4 * rows))
    
    # Original Generated Image
    plt.subplot(rows, cols, 1)
    plt.imshow(image)
    plt.title("Generated")
    plt.axis('off')
    
    for i in range(num_tokens):
        if i >= final_maps.shape[0]: break
        
        plt.subplot(rows, cols, i + 2)
        
        # Upscale map to image size
        map_resized = F.interpolate(final_maps[i].unsqueeze(0).unsqueeze(0), size=image.size[::-1], mode='bilinear').squeeze()
        
        plt.imshow(image)
        plt.imshow(map_resized, cmap='jet', alpha=0.5)
        plt.title(labels[i], fontsize=8)
        plt.axis('off')
        
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved attention map to {output_path}")
