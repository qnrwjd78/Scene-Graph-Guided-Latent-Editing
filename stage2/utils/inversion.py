import torch
from tqdm import tqdm

def null_text_inversion(pipe, latents, prompt_embeds, num_inference_steps=50, num_inner_steps=10, guidance_scale=7.5):
    """
    Perform Null-text Inversion.
    
    Args:
        pipe: StableDiffusionPipeline
        latents: z_T (starting noise)
        prompt_embeds: Condition embeddings (Scene Graph embeddings)
        num_inference_steps: DDIM steps
        num_inner_steps: Optimization steps per timestamp
        guidance_scale: CFG scale
        
    Returns:
        z_T: Inverted latent
        uncond_embeddings_list: List of optimized unconditional embeddings
    """
    # 1. DDIM Inversion (Forward) to get z_T
    # We assume 'latents' passed here is already z_T or we need to invert x_0 to z_T.
    # Let's assume the user provides x_0 and we do inversion here, OR user provides z_T.
    # The prompt says "Input: z_T". But usually Inversion starts from Image.
    # Let's assume input is 'latents' corresponding to z_T for simplicity, 
    # or we can implement the full forward pass if needed.
    # But wait, Null-text inversion needs the trajectory z_t -> z_{t-1}.
    # So we usually need to run the inversion process first.
    
    # For this snippet, let's assume we are doing the optimization loop.
    # We need the intermediate latents z_t from the inversion process.
    
    # Placeholder for full implementation
    # In a real scenario, we would:
    # 1. Encode image to z_0
    # 2. DDIM Inversion z_0 -> z_T, saving all z_t
    # 3. Loop backwards t from T to 0:
    #    Optimize null_text_embedding to minimize || pred_z_{t-1} - actual_z_{t-1} ||
    
    print("Null-text inversion logic placeholder.")
    return latents, []

class AttentionStore:
    def __init__(self):
        self.store = []
        
    def append(self, k, v):
        self.store.append((k, v))
        
    def pop(self):
        return self.store.pop(0) # Pop from start (FIFO)
        
    def reset(self):
        self.store = []
