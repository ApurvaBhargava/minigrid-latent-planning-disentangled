"""Flexible encoder that auto-adapts to observation size."""

import torch
import torch.nn as nn


class FlexibleEncoder(nn.Module):
    """Encoder that auto-adapts to any observation size."""
    
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Conv layers (preserve spatial dimensions)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsample
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc = None  # Will be initialized on first forward pass
    
    def _init_fc(self, conv_out_size):
        """Initialize FC layers based on actual conv output."""
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.latent_dim)
        ).to(next(self.conv.parameters()).device)
    
    def forward(self, x):
        if x.dim() == 5:  # Sequence: (batch, seq, C, H, W)
            B, T = x.shape[:2]
            x = x.reshape(B * T, *x.shape[2:])
            conv_out = self.conv(x)
            
            if self.fc is None:
                self._init_fc(conv_out.shape[1])
            
            z = self.fc(conv_out)
            return z.reshape(B, T, self.latent_dim)
        else:  # Single: (batch, C, H, W)
            conv_out = self.conv(x)
            
            if self.fc is None:
                self._init_fc(conv_out.shape[1])
            
            return self.fc(conv_out)

