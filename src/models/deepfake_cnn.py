import torch
import torch.nn as nn
from typing import Tuple

class DeepfakeCNN(nn.Module):
    """
    Lab 10 (Phase 3) - CNN Track Implementation for Deepfake Images.
    
    Architecture Justification & Receptive Field:
    - Input: 3x256x256 images.
    - We use 5 convolutional layers. Since it's < 6 layers, BatchNorm is technically optional by rubric, 
      but we include it to ensure stable training and fulfill the 'Normalized' ablation config.
    - MaxPool2d(2) reduces spatial dimensions by 2 at each block.
    - Receptive Field Calculation at final feature map:
      * Block 1 (Conv3x3 + Pool2x2): RF=3, Output=128x128
      * Block 2 (Conv3x3 + Pool2x2): RF=7, Output=64x64
      * Block 3 (Conv3x3 + Pool2x2): RF=15, Output=32x32
      * Block 4 (Conv3x3 + Pool2x2): RF=31, Output=16x16
      * Block 5 (Conv3x3 + Pool2x2): RF=63, Output=8x8
      The receptive field is 63x63. While smaller than 256x256, the deepfake artifacts 
      are usually localized high-frequency glitches (eyes, edges), so seeing the whole face 
      at once is less critical than capturing local textures.
    """
    def __init__(self, num_classes: int = 1, dropout_p: float = 0.3, base_channels: int = 32):
        super().__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Block 2
        self.conv2 = nn.Conv2d(base_channels, base_channels*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels*2)
        
        # Block 3
        self.conv3 = nn.Conv2d(base_channels*2, base_channels*4, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(base_channels*4)
        
        # Block 4
        self.conv4 = nn.Conv2d(base_channels*4, base_channels*8, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(base_channels*8)
        
        # Block 5
        self.conv5 = nn.Conv2d(base_channels*8, base_channels*8, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(base_channels*8)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_p)
        
        # Classifier (Input is 256x256 -> 5 pooling layers -> 8x8 spatial size)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear((base_channels*8) * 8 * 8, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (B, 3, 256, 256)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))     # Shape: (B, C, 128, 128)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))     # Shape: (B, C*2, 64, 64)
        x = self.pool(self.relu(self.bn3(self.conv3(x))))     # Shape: (B, C*4, 32, 32)
        x = self.pool(self.relu(self.bn4(self.conv4(x))))     # Shape: (B, C*8, 16, 16)
        x = self.pool(self.relu(self.bn5(self.conv5(x))))     # Shape: (B, C*8, 8, 8)
        
        x = self.flatten(x)                                   # Shape: (B, C*8*8*8)
        x = self.dropout(self.relu(self.fc1(x)))              # Shape: (B, 512)
        logits = self.fc2(x)                                  # Shape: (B, num_classes)
        
        return logits
