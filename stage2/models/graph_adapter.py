import torch
import torch.nn as nn

class GraphAdapter(nn.Module):
    def __init__(self, input_dim=512, output_dim=768):
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )
        
    def forward(self, x):
        return self.adapter(x)
