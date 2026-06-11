import torch
import torch.nn as nn
from typing import List

class DeepTabularMLP(nn.Module):
    """
    Lab 10 (Phase 3) - Custom Deep Architecture for Tabular Data (UNSW-NB15).
    
    Architecture Justification:
    - Standard MLPs suffer from vanishing gradients when very deep.
    - We implement a 5-layer deep MLP with BatchNorm and Residual/Skip connections
      to allow deeper representation learning without degradation.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_blocks: int = 4, num_classes: int = 1, dropout_p: float = 0.3):
        super().__init__()
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            # A residual block
            block = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout_p),
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim)
            )
            self.blocks.append(block)
            
        self.final_relu = nn.GELU()
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, input_dim)
        x = self.input_layer(x) # Shape: (B, hidden_dim)
        
        # Residual blocks
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual # Skip connection
            x = self.final_relu(x)
            
        logits = self.output_layer(x) # Shape: (B, num_classes)
        return logits
