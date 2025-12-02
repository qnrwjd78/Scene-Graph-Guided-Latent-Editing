import torch
from tqdm import tqdm

from typing import Union, List, Optional, Tuple
from torch.optim import Adam

@torch.no_grad()
def ddim_inversion(
    pipe, 
    x0: torch.Tensor, 
    prompt_embeds: torch.Tensor, 
    num_inference_steps: int = 50, 
    guidance_scale: float = 7.5
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    Invert a real image x0 to noise z_T using DDIM inversion.
    """
    # 1. Encode image to latent
    # x0 should be (B, 3, H, W) in [-1, 1]
    if x0.min() < -1.5 or x0.max() > 1.5:
        print("Warning: x0 seems not normalized to [-1, 1]")
        
    latents = pipe.vae.encode(x0).latent_dist.sample()
    latents = latents * pipe.vae.config.scaling_factor
    
    # 2. Setup scheduler
    pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps = pipe.scheduler.timesteps
    
    # Collect all latents
    all_latents = [latents.detach().cpu()]
    
    reversed_timesteps = list(reversed(timesteps))
    
    for i, t in enumerate(reversed_timesteps):
        # t is current time. next_t is t+1 (more noise)
        
        # 0. Scale model input
        latent_input = pipe.scheduler.scale_model_input(latents, t)

        # 1. Predict noise
        # We use the conditional embedding
        # Note: ddim_inversion usually uses guidance_scale=1.0 (no CFG)
        noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=prompt_embeds).sample
        
        # 2. Compute next latent
        # z_{t+1}
        # Ensure t is int for indexing
        t_idx = t.item() if isinstance(t, torch.Tensor) else t
        alpha_prod_t = pipe.scheduler.alphas_cumprod[t_idx]
        
        # Handle last step
        if i < len(reversed_timesteps) - 1:
            next_t = reversed_timesteps[i+1]
            next_t_idx = next_t.item() if isinstance(next_t, torch.Tensor) else next_t
            alpha_prod_t_next = pipe.scheduler.alphas_cumprod[next_t_idx]
        else:
            # For the last step, we are going to T_max. 
            # Ideally alpha_prod -> 0. But we can just use the current alpha if we stop here.
            alpha_prod_t_next = pipe.scheduler.alphas_cumprod[t_idx]
            
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_next = 1 - alpha_prod_t_next
        
        # f_theta = (z_t - sqrt(beta_t) * eps) / sqrt(alpha_t)
        f_theta = (latents - beta_prod_t ** 0.5 * noise_pred) / (alpha_prod_t ** 0.5)
        
        # z_{t+1} = sqrt(alpha_{t+1}) * f_theta + sqrt(beta_{t+1}) * eps
        latents = alpha_prod_t_next ** 0.5 * f_theta + beta_prod_t_next ** 0.5 * noise_pred
        
        if i % 20 == 0:
            print(f"Step {i}, Latent mean: {latents.mean().item():.4f}, std: {latents.std().item():.4f}")
            
        all_latents.append(latents.detach().cpu())
        
    return latents, all_latents

def null_text_inversion(pipe, x0, prompt_embeds, num_inference_steps=50, num_inner_steps=10, guidance_scale=7.5):
    """
    Perform Null-text Inversion.
    """
    # 1. DDIM Inversion (Forward) to get trajectory
    print("Running DDIM Inversion for Null-text...")
    z_T, all_latents = ddim_inversion(pipe, x0, prompt_embeds, num_inference_steps, guidance_scale=1.0)
    
    # 2. Null-text Optimization (Backward)
    pipe.scheduler.set_timesteps(num_inference_steps)
    timesteps = pipe.scheduler.timesteps
    
    # Initial uncond embedding (empty text)
    uncond_input = pipe.tokenizer([""], padding="max_length", max_length=pipe.tokenizer.model_max_length, return_tensors="pt")
    with torch.no_grad():
        uncond_embeddings_init = pipe.text_encoder(uncond_input.input_ids.to(pipe.device))[0]
        
    uncond_embeddings_list = []
    
    bar = tqdm(total=num_inference_steps, desc="Null-text Optimization")
    
    for i, t in enumerate(timesteps):
        # Target is z_{t-1} (less noisy)
        # all_latents indices: z_T is at -1, z_{T-1} is at -2
        target_latent = all_latents[-(i+2)].to(pipe.device)
        latent_prev = all_latents[-(i+1)].to(pipe.device) # z_t
        
        # Optimize uncond_embedding
        uncond_embedding = uncond_embeddings_init.clone().detach().requires_grad_(True)
        optimizer = Adam([uncond_embedding], lr=1e-2 * (1. - i / num_inference_steps))
        
        for _ in range(num_inner_steps):
            optimizer.zero_grad()
            
            # Predict noise with CFG
            latent_input = torch.cat([latent_prev] * 2)
            concat_embeds = torch.cat([uncond_embedding, prompt_embeds])
            
            noise_pred = pipe.unet(latent_input, t, encoder_hidden_states=concat_embeds).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred_cfg = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Step (z_t -> z_{t-1})
            alpha_prod_t = pipe.scheduler.alphas_cumprod[t]
            prev_timestep = timesteps[i+1] if i < len(timesteps) - 1 else 0
            alpha_prod_t_prev = pipe.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else pipe.scheduler.final_alpha_cumprod
            
            beta_prod_t = 1 - alpha_prod_t
            beta_prod_t_prev = 1 - alpha_prod_t_prev
            
            # Predicted x0
            pred_original_sample = (latent_prev - beta_prod_t ** 0.5 * noise_pred_cfg) / alpha_prod_t ** 0.5
            
            # Direction
            pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * noise_pred_cfg
            
            # z_{t-1}
            prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
            
            loss = torch.nn.functional.mse_loss(prev_sample, target_latent)
            loss.backward()
            optimizer.step()
            
        uncond_embeddings_list.append(uncond_embedding.detach())
        bar.update(1)
        
    bar.close()
    return z_T, uncond_embeddings_list

class AttentionStore:
    def __init__(self):
        self.store = []
        
    def append(self, k, v):
        self.store.append((k, v))
        
    def pop(self):
        if len(self.store) > 0:
            return self.store.pop(0)
        return None, None
    
    def clear(self):
        self.store = []
        
    def reset(self):
        self.store = []
