"""Loss functions for PLDM training."""

import torch
import torch.nn.functional as F


def vicreg_loss(z_pred, z_target, sim_coeff=1.0, std_coeff=1.0, cov_coeff=0.04):
    """VICReg loss for self-supervised representation learning.
    
    Combines three terms:
    - Invariance (similarity): MSE between predicted and target
    - Variance: Encourages each dimension to have unit variance
    - Covariance: Decorrelates dimensions to avoid collapse
    
    Args:
        z_pred: Predicted latent representations
        z_target: Target latent representations (will be detached)
        sim_coeff: Weight for similarity loss
        std_coeff: Weight for standard deviation (variance) loss
        cov_coeff: Weight for covariance loss
        
    Returns:
        total_loss: Combined weighted loss
        sim_loss: Similarity (MSE) component
        std_loss: Variance regularization component
        cov_loss: Covariance regularization component
    """
    z_target = z_target.detach()
    
    # Flatten if sequence
    z_flat = z_pred.reshape(-1, z_pred.shape[-1]) if z_pred.dim() == 3 else z_pred
    
    # Similarity loss (invariance term)
    sim_loss = F.mse_loss(z_pred, z_target)
    
    B, D = z_flat.shape
    
    # Center the representations
    z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
    
    # Variance loss - encourage each dimension to have variance >= 1
    std = torch.sqrt(z_centered.var(dim=0) + 1e-4)
    std_loss = F.relu(1.0 - std).mean()
    
    # Covariance loss - decorrelate dimensions
    cov = (z_centered.T @ z_centered) / (B - 1)
    cov_loss = (cov ** 2).sum() - (torch.diagonal(cov) ** 2).sum()
    cov_loss = cov_loss / (D * (D - 1))
    
    # Combined loss
    total_loss = sim_coeff * sim_loss + std_coeff * std_loss + cov_coeff * cov_loss
    
    return total_loss, sim_loss, std_loss, cov_loss

