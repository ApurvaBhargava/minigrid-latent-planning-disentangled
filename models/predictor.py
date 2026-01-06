"""Dynamics predictor for latent space transitions."""

import torch
import torch.nn as nn


class Predictor(nn.Module):
    """Dynamics predictor: (z_t, action) -> z_{t+1}"""
    
    def __init__(self, latent_dim=128, action_dim=7):
        super().__init__()
        self.latent_dim = latent_dim
        self.action_embed = nn.Embedding(action_dim, latent_dim)
        
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, latent_dim)
        )
    
    def forward(self, z, action):
        if z.dim() == 3:  # Sequence
            B, T, _ = z.shape
            z_flat = z.reshape(B * T, self.latent_dim)
            action_flat = action.reshape(B * T)
            action_emb = self.action_embed(action_flat)
            x = torch.cat([z_flat, action_emb], dim=-1)
            z_next = z_flat + self.dynamics(x)  # Residual connection
            return z_next.reshape(B, T, self.latent_dim)
        else:  # Single
            action_emb = self.action_embed(action)
            x = torch.cat([z, action_emb], dim=-1)
            return z + self.dynamics(x)  # Residual connection

