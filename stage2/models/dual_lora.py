import torch
import torch.nn as nn
import torch.nn.functional as F

class DualInputLoRALinear(nn.Module):
    """
    Dual Input Strategy를 위한 Custom LoRA Layer.
    Frozen UNet에는 Clean Input을, LoRA에는 Mixed Input을 전달합니다.
    """
    def __init__(self, original_linear, rank=4, alpha=4):
        super().__init__()
        self.original_linear = original_linear # Frozen Linear Layer
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = rank
        self.scale = alpha / rank
        
        # LoRA Layers (Trainable)
        self.lora_down = nn.Linear(self.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, self.out_features, bias=False)
        
        # Initialize LoRA
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)
        
        # Move to same device/dtype as original linear
        self.lora_down.to(device=original_linear.weight.device, dtype=original_linear.weight.dtype)
        self.lora_up.to(device=original_linear.weight.device, dtype=original_linear.weight.dtype)
        
        # Freeze original linear
        self.original_linear.requires_grad_(False)

    def forward(self, x):
        # x shape: (Batch, Seq, Dim * 2) 
        # We expect x to be concatenated [x_clean, x_mixed] along the last dimension
        
        dim = x.shape[-1] // 2
        x_clean = x[..., :dim]
        x_mixed = x[..., dim:]
        
        # 1. Frozen Path (Clean Input)
        frozen_out = self.original_linear(x_clean)
        
        # 2. LoRA Path (Mixed Input)
        lora_out = self.lora_up(self.lora_down(x_mixed)) * self.scale
        
        # 3. Combine
        return frozen_out + lora_out
