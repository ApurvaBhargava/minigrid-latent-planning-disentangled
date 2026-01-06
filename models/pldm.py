"""Complete PLDM model combining encoder and predictor."""

import torch
import torch.nn as nn

from .encoder import FlexibleEncoder
from .predictor import Predictor


class PLDM(nn.Module):
    """Complete PLDM: Encoder + Predictor."""
    
    def __init__(self, latent_dim=128, action_dim=7):
        super().__init__()
        self.encoder = FlexibleEncoder(latent_dim)
        self.predictor = Predictor(latent_dim, action_dim)
        self.latent_dim = latent_dim
    
    def forward(self, obs, actions, next_obs):
        """Forward pass for training.
        
        Args:
            obs: Current observations (batch, seq, C, H, W)
            actions: Actions taken (batch, seq)
            next_obs: Next observations (batch, seq, C, H, W)
            
        Returns:
            z: Encoded current states
            z_next: Encoded next states (target)
            z_next_pred: Predicted next states
        """
        z = self.encoder(obs)
        z_next = self.encoder(next_obs)
        z_next_pred = self.predictor(z, actions)
        return z, z_next, z_next_pred
    
    def encode(self, obs):
        """Encode observations to latent space."""
        return self.encoder(obs)
    
    def predict_step(self, z, action):
        """Predict next latent state given current state and action."""
        return self.predictor(z, action)
    
    def init_encoder_fc(self, obs_shape, device):
        """Initialize encoder FC layers with a dummy forward pass.
        
        This is useful when loading a saved model to ensure
        the FC layers are properly initialized before loading state dict.
        
        Args:
            obs_shape: Tuple of (C, H, W) for observation shape
            device: Device to use for initialization
        """
        dummy_obs = torch.zeros(1, *obs_shape).to(device)
        _ = self.encode(dummy_obs)

