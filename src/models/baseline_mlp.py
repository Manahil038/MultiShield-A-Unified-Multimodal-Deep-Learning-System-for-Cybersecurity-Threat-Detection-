import torch
import torch.nn as nn
from typing import List

class BaselineMLP(nn.Module):
    """
    Baseline Multi-Layer Perceptron for MultiShield.
    Configurable hidden layers, dropout, and initialized with He Normal for GELU.
    """
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, dropout_p: float = 0.3):
        super().__init__()
        
        # Build the layers dynamically
        layers = []
        in_features = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            # Standard: GELU activation (solves dying ReLU)
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=dropout_p))
            in_features = hidden_dim
            
        # Final classification layer (No activation here; handled by Loss Function e.g., BCEWithLogits)
        layers.append(nn.Linear(in_features, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Apply He/Kaiming initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """
        Kaiming/He Normal Initialization.
        Mathematically sound for networks using ReLU/GELU activations to prevent vanishing gradients.
        """
        if isinstance(module, nn.Linear):
            # nonlinearity='relu' is standard for GELU as well in Kaiming
            nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Logits of shape (batch_size, output_dim)
        """
        return self.network(x)
