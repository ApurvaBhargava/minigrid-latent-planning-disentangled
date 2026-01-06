#!/usr/bin/env python3
"""Train PLDM model on MiniGrid environments.

This script handles data collection, model training, and checkpoint saving.
Supports both training from scratch and resuming from checkpoints.

Example usage:
    # Train with default settings
    python train.py --output_dir outputs/run1
    
    # Train with custom parameters
    python train.py --output_dir outputs/run2 --num_trajectories 2000 --epochs 150 --lr 1e-4
    
    # Resume training from checkpoint
    python train.py --output_dir outputs/run1 --resume
"""

import argparse
import json
import os
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import PLDM
from utils.data import collect_dataset, TrajectoryDataset
from utils.losses import vicreg_loss


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, train_loader, optimizer, device, sim_coeff, std_coeff, cov_coeff):
    """Train for one epoch.
    
    Returns:
        Dictionary of average losses for the epoch
    """
    model.train()
    losses = {'total': [], 'sim': [], 'std': [], 'cov': []}
    
    pbar = tqdm(train_loader, desc="Training")
    for obs, actions, next_obs in pbar:
        obs = obs.to(device)
        actions = actions.to(device)
        next_obs = next_obs.to(device)
        
        # Forward pass
        z, z_next, z_next_pred = model(obs, actions, next_obs)
        
        # Compute loss
        total_loss, sim_loss, std_loss, cov_loss = vicreg_loss(
            z_next_pred, z_next, sim_coeff, std_coeff, cov_coeff
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Record losses
        losses['total'].append(total_loss.item())
        losses['sim'].append(sim_loss.item())
        losses['std'].append(std_loss.item())
        losses['cov'].append(cov_loss.item())
        
        pbar.set_postfix({'loss': total_loss.item(), 'sim': sim_loss.item()})
    
    return {k: np.mean(v) for k, v in losses.items()}


def validate(model, val_loader, device, sim_coeff, std_coeff, cov_coeff):
    """Validate the model.
    
    Returns:
        Dictionary of average losses for validation
    """
    model.eval()
    losses = {'total': [], 'sim': [], 'std': [], 'cov': []}
    
    with torch.no_grad():
        for obs, actions, next_obs in val_loader:
            obs = obs.to(device)
            actions = actions.to(device)
            next_obs = next_obs.to(device)
            
            z, z_next, z_next_pred = model(obs, actions, next_obs)
            total_loss, sim_loss, std_loss, cov_loss = vicreg_loss(
                z_next_pred, z_next, sim_coeff, std_coeff, cov_coeff
            )
            
            losses['total'].append(total_loss.item())
            losses['sim'].append(sim_loss.item())
            losses['std'].append(std_loss.item())
            losses['cov'].append(cov_loss.item())
    
    return {k: np.mean(v) for k, v in losses.items()}


def save_checkpoint(model, optimizer, epoch, history, args, filepath):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history,
        'args': vars(args)
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath, model, optimizer, device):
    """Load training checkpoint."""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['history']


def main():
    parser = argparse.ArgumentParser(
        description='Train PLDM model on MiniGrid environments',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='outputs/default',
                        help='Directory to save model checkpoints and logs')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    
    # Environment settings
    parser.add_argument('--env_name', type=str, default='MiniGrid-DoorKey-5x5-v0',
                        help='MiniGrid environment name')
    
    # Data collection settings
    parser.add_argument('--num_trajectories', type=int, default=1200,
                        help='Number of trajectories to collect')
    parser.add_argument('--bfs_ratio', type=float, default=0.8,
                        help='Fraction of BFS (optimal) trajectories')
    parser.add_argument('--sequence_length', type=int, default=8,
                        help='Sequence length for training')
    parser.add_argument('--max_steps', type=int, default=100,
                        help='Max steps for random trajectories')
    
    # Model settings
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent space dimension')
    parser.add_argument('--action_dim', type=int, default=7,
                        help='Number of actions')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation set ratio')
    parser.add_argument('--val_every', type=int, default=5,
                        help='Validate every N epochs')
    
    # Loss coefficients
    parser.add_argument('--sim_coeff', type=float, default=1.0,
                        help='Similarity loss coefficient')
    parser.add_argument('--std_coeff', type=float, default=1.0,
                        help='Standard deviation loss coefficient')
    parser.add_argument('--cov_coeff', type=float, default=0.04,
                        help='Covariance loss coefficient')
    
    # Other settings
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.output_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Collect data
    print("\n" + "="*60)
    print("Collecting training data...")
    print("="*60)
    trajectories = collect_dataset(
        env_name=args.env_name,
        num_trajectories=args.num_trajectories,
        bfs_ratio=args.bfs_ratio,
        max_steps=args.max_steps
    )
    
    # Create dataset
    dataset = TrajectoryDataset(trajectories, sequence_length=args.sequence_length)
    
    # Split train/val
    train_size = int((1 - args.val_ratio) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train: {len(train_dataset)} sequences")
    print(f"Val: {len(val_dataset)} sequences")
    
    # Create model
    model = PLDM(latent_dim=args.latent_dim, action_dim=args.action_dim).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training history
    history = {
        'train_loss': [], 'train_sim': [], 'train_std': [], 'train_cov': [],
        'val_loss': [], 'val_sim': [], 'val_std': [], 'val_cov': [],
        'val_epochs': []
    }
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Resume from checkpoint if requested
    if args.resume:
        checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
        if os.path.exists(checkpoint_path):
            print(f"\nResuming from {checkpoint_path}")
            start_epoch, history = load_checkpoint(checkpoint_path, model, optimizer, device)
            start_epoch += 1
            best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
            print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "="*60)
    print("Training PLDM...")
    print("="*60)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, device,
            args.sim_coeff, args.std_coeff, args.cov_coeff
        )
        
        history['train_loss'].append(train_losses['total'])
        history['train_sim'].append(train_losses['sim'])
        history['train_std'].append(train_losses['std'])
        history['train_cov'].append(train_losses['cov'])
        
        # Validate
        if (epoch + 1) % args.val_every == 0:
            val_losses = validate(
                model, val_loader, device,
                args.sim_coeff, args.std_coeff, args.cov_coeff
            )
            
            history['val_loss'].append(val_losses['total'])
            history['val_sim'].append(val_losses['sim'])
            history['val_std'].append(val_losses['std'])
            history['val_cov'].append(val_losses['cov'])
            history['val_epochs'].append(epoch + 1)
            
            print(f"\nEpoch {epoch+1}: Train={train_losses['total']:.4f}, Val={val_losses['total']:.4f}, "
                  f"Sim={val_losses['sim']:.4f}, Std={val_losses['std']:.4f}, Cov={val_losses['cov']:.4f}")
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pt'))
                print(f"  Saved best model (val_loss={val_losses['total']:.4f})")
        
        # Save latest checkpoint
        save_checkpoint(model, optimizer, epoch, history, args,
                       os.path.join(checkpoint_dir, 'latest_checkpoint.pt'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'final_model.pt'))
    
    # Save training history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Final training loss: {history['train_loss'][-1]:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    # Check if model learned
    initial_sim = history['train_sim'][0]
    final_sim = history['train_sim'][-1]
    improvement = initial_sim - final_sim
    
    print(f"\nPrediction (MSE): {initial_sim:.4f} -> {final_sim:.4f} (improvement: {improvement:.4f})")
    
    if improvement > 0.05:
        print("Model is learning - prediction loss decreased significantly.")
    else:
        print("Warning: Prediction loss did not decrease much.")


if __name__ == '__main__':
    main()

