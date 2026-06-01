import torch
import torch.nn as nn

class TabularAutoencoder(nn.Module):
    """
    A multi-layer dense Autoencoder for Tabular Anomaly Detection.
    Maps high-dimensional normalized features down to a 10D bottleneck latent space,
    then symmetrically reconstructs them.
    """
    def __init__(self, input_dim, latent_dim=10):
        super().__init__()
        # Encoder: compresses input features down to latent bottleneck representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Linear(32, latent_dim)
        )
        
        # Decoder: reconstructs original input features from bottleneck representation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Linear(64, input_dim)
        )

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
