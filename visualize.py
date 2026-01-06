#!/usr/bin/env python3
"""Generate visualizations for PLDM model and training results.

This script generates various visualizations including:
- Training loss curves
- Latent space visualization (PCA)
- BFS trajectory visualization
- Planning episode visualization
- Replanning frequency comparison plots
- Environment comparison (different MiniGrid environments)
- Episode execution with distance trajectory plots

Example usage:
    # Generate training curves from history
    python visualize.py --mode training_curves --history_path outputs/run1/training_history.json
    
    # Visualize latent space with PCA
    python visualize.py --mode latent_space --model_path outputs/run1/checkpoints/best_model.pt
    
    # Visualize BFS trajectories
    python visualize.py --mode bfs_trajectories --seeds 1 2 4
    
    # Visualize planning episodes
    python visualize.py --mode planning --model_path outputs/run1/checkpoints/best_model.pt
    
    # Compare different MiniGrid environments
    python visualize.py --mode env_comparison --output_dir outputs/run1/visualizations
    
    # Visualize episode execution with distance plot
    python visualize.py --mode episode_distances --model_path outputs/run1/checkpoints/best_model.pt
    
    # Generate all visualizations
    python visualize.py --mode all --model_path outputs/run1/checkpoints/best_model.pt \\
        --history_path outputs/run1/training_history.json --output_dir outputs/run1/visualizations
"""

import argparse
import json
import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from models import PLDM
from utils.data import bfs_solve, collect_dataset, collect_trajectory
from utils.environment import make_env, get_full_obs
from utils.planner import CEMPlanner


def get_device():
    """Get the best available device."""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


def load_model(model_path, latent_dim, action_dim, device):
    """Load trained model from checkpoint."""
    model = PLDM(latent_dim=latent_dim, action_dim=action_dim).to(device)
    model.init_encoder_fc((3, 40, 40), device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def plot_training_curves(history, output_path=None):
    """Plot training loss curves.
    
    Args:
        history: Dictionary with training history
        output_path: Path to save figure (optional)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    val_epochs = history.get('val_epochs', 
                             list(range(4, len(history['train_loss']), 5)))
    
    # Total Loss
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2, color='blue')
    if history.get('val_loss'):
        axes[0, 0].plot(val_epochs, history['val_loss'], 'o-', label='Val', 
                        linewidth=2, markersize=6, color='orange')
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Similarity Loss
    axes[0, 1].plot(history['train_sim'], label='Train MSE', linewidth=2, color='green')
    if history.get('val_sim'):
        axes[0, 1].plot(val_epochs, history['val_sim'], 'o-', label='Val MSE', 
                        linewidth=2, markersize=6, color='red')
    axes[0, 1].set_title('Similarity Loss (MSE)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Std Loss
    axes[1, 0].plot(history['train_std'], label='Train Std', linewidth=2, color='purple')
    if history.get('val_std'):
        axes[1, 0].plot(val_epochs, history['val_std'], 'o-', label='Val Std', 
                        linewidth=2, markersize=6, color='magenta')
    axes[1, 0].set_title('Std Loss (Variance Reg)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Std Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Cov Loss
    axes[1, 1].plot(history['train_cov'], label='Train Cov', linewidth=2, color='brown')
    if history.get('val_cov'):
        axes[1, 1].plot(val_epochs, history['val_cov'], 'o-', label='Val Cov', 
                        linewidth=2, markersize=6, color='pink')
    axes[1, 1].set_title('Cov Loss (Covariance Reg)', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Cov Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {output_path}")
    
    plt.show()
    
    # Print summary
    initial_sim = history['train_sim'][0]
    final_sim = history['train_sim'][-1]
    improvement = initial_sim - final_sim
    
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Prediction (MSE): {initial_sim:.4f} -> {final_sim:.4f} (improvement: {improvement:.4f})")
    print(f"Std Loss: {history['train_std'][0]:.4f} -> {history['train_std'][-1]:.4f}")
    print(f"Cov Loss: {history['train_cov'][0]:.4f} -> {history['train_cov'][-1]:.4f}")
    print(f"Total Loss: {history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f}")


def plot_latent_space(model, trajectories, output_path=None):
    """Visualize latent space with PCA.
    
    Shows separation between early and late states in episodes.
    
    Args:
        model: Trained PLDM model
        trajectories: List of trajectories
        output_path: Path to save figure (optional)
    """
    device = next(model.parameters()).device
    latents = []
    labels = []  # 0: early, 1: late
    
    model.eval()
    with torch.no_grad():
        for traj in trajectories[:100]:
            if len(traj) < 5:
                continue
            
            # Early state
            early_obs = traj[0]['obs']
            early_tensor = torch.FloatTensor(early_obs).permute(2, 0, 1).unsqueeze(0) / 255.0
            z_early = model.encode(early_tensor.to(device)).squeeze(0).cpu().numpy()
            latents.append(z_early)
            labels.append(0)
            
            # Late state
            late_idx = min(len(traj) - 1, 20)
            late_obs = traj[late_idx]['obs']
            late_tensor = torch.FloatTensor(late_obs).permute(2, 0, 1).unsqueeze(0) / 255.0
            z_late = model.encode(late_tensor.to(device)).squeeze(0).cpu().numpy()
            latents.append(z_late)
            labels.append(1)
    
    latents = np.array(latents)
    labels = np.array(labels)
    
    # PCA to 2D
    pca = PCA(n_components=2)
    latents_2d = pca.fit_transform(latents)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['blue', 'red']
    labels_text = ['Early States', 'Late States']
    
    for i in range(2):
        mask = labels == i
        ax.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                   c=colors[i], label=labels_text[i], alpha=0.5, s=50,
                   edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title('Latent Space Structure (PCA)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved latent space visualization to {output_path}")
    
    plt.show()
    
    print(f"\nExplained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, "
          f"PC2={pca.explained_variance_ratio_[1]:.3f}")
    
    # Check separation
    from scipy.spatial.distance import cdist
    early_latents = latents[labels == 0]
    late_latents = latents[labels == 1]
    distances = cdist(early_latents, late_latents, 'euclidean')
    print(f"Average distance between early/late: {distances.mean():.4f}")


def plot_bfs_trajectory(env_name, seed, output_path=None):
    """Visualize a BFS-solved trajectory.
    
    Args:
        env_name: Environment name
        seed: Random seed
        output_path: Path to save figure (optional)
    """
    actions = bfs_solve(env_name, seed)
    
    if actions is None:
        print(f"Seed {seed}: BFS failed to find solution")
        return None
    
    print(f"\n{'='*70}")
    print(f"Seed {seed}: BFS Solution with {len(actions)} actions")
    print(f"{'='*70}")
    
    env = make_env(env_name)
    env.reset(seed=seed)
    
    images = []
    positions = []
    action_taken = []
    rewards = []
    dones = []
    
    # Initial state
    images.append(get_full_obs(env).copy())
    positions.append(tuple(env.unwrapped.agent_pos))
    
    # Execute actions
    action_names = ['Left', 'Right', 'Fwd', 'Pick', 'Drop', 'Tog', 'Done']
    for i, action in enumerate(actions):
        obs, reward, done, truncated, _ = env.step(action)
        
        images.append(get_full_obs(env).copy())
        positions.append(tuple(env.unwrapped.agent_pos))
        action_taken.append(action)
        rewards.append(reward)
        dones.append(done)
        
        print(f"  Step {i+1}: {action_names[action]:5s} -> pos={env.unwrapped.agent_pos}, "
              f"reward={reward:.1f}, done={done}")
        
        if done:
            print(f"\nTask completed in {i+1} steps!")
            break
    
    env.close()
    
    # Visualize
    num_images = len(images)
    cols = min(8, num_images)
    rows = (num_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(num_images):
        row, col = idx // cols, idx % cols
        axes[row, col].imshow(images[idx])
        
        if idx == 0:
            title = f'S0: START\n{positions[idx]}'
        else:
            title = f'S{idx}: {action_names[action_taken[idx-1]]}\n{positions[idx]}'
            if dones[idx-1]:
                title += '\nDONE'
        
        axes[row, col].set_title(title, fontsize=9, 
                                 fontweight='bold' if idx == num_images-1 else 'normal')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for idx in range(num_images, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis('off')
    
    fig.suptitle(f'Seed {seed}: BFS Optimal Trajectory ({len(actions)} actions)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved BFS trajectory to {output_path}")
    
    plt.show()
    
    return images, actions


def plot_planning_episode(model, trajectory, output_path=None):
    """Visualize planning progress for one episode.
    
    Args:
        model: Trained PLDM model
        trajectory: Trajectory with start and goal states
        output_path: Path to save figure (optional)
    """
    device = next(model.parameters()).device
    
    planner = CEMPlanner(
        model, action_dim=7, horizon=15,
        num_iterations=15, num_samples=500, num_elites=50
    )
    
    start_obs = trajectory[0]['obs']
    goal_obs = trajectory[-1]['obs']
    
    # Encode
    start_tensor = torch.FloatTensor(start_obs).permute(2, 0, 1).unsqueeze(0) / 255.0
    goal_tensor = torch.FloatTensor(goal_obs).permute(2, 0, 1).unsqueeze(0) / 255.0
    
    with torch.no_grad():
        z_start = model.encode(start_tensor.to(device)).squeeze(0)
        z_goal = model.encode(goal_tensor.to(device)).squeeze(0)
    
    # Plan
    print("Planning...")
    action_sequence = planner.plan(z_start, z_goal, verbose=True)
    
    # Roll out latent trajectory
    latent_trajectory = [z_start.cpu().numpy()]
    with torch.no_grad():
        z_current = z_start
        for action in action_sequence:
            z_current = model.predict_step(z_current.unsqueeze(0), action.unsqueeze(0)).squeeze(0)
            latent_trajectory.append(z_current.cpu().numpy())
    
    latent_trajectory = np.array(latent_trajectory)
    z_goal_np = z_goal.cpu().numpy()
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].imshow(start_obs)
    axes[0].set_title('Start State', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(goal_obs)
    axes[1].set_title('Goal State', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    distances = [np.linalg.norm(z - z_goal_np) for z in latent_trajectory]
    axes[2].plot(distances, marker='o', linewidth=2, markersize=8, color='blue')
    axes[2].set_xlabel('Planning Step', fontsize=12)
    axes[2].set_ylabel('Distance to Goal (Latent)', fontsize=12)
    axes[2].set_title('Planning Progress', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].axhline(y=0, color='green', linestyle='--', label='Goal', linewidth=2)
    axes[2].legend()
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved planning visualization to {output_path}")
    
    plt.show()
    
    print(f"\nPlanned actions: {action_sequence.cpu().numpy()}")
    print(f"Initial distance: {distances[0]:.4f}")
    print(f"Final distance: {distances[-1]:.4f}")
    print(f"Improvement: {distances[0] - distances[-1]:.4f}")


def plot_trajectory_steps(trajectory, max_steps=16, output_path=None):
    """Visualize step-by-step trajectory.
    
    Args:
        trajectory: Trajectory to visualize
        max_steps: Maximum number of steps to show
        output_path: Path to save figure (optional)
    """
    num_steps = min(len(trajectory), max_steps)
    cols = 4
    rows = (num_steps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*4))
    axes = axes.flatten()
    
    action_names = ['Right ->', 'Down v', 'Left <-', 'Up ^', 'Pick', 'Drop', 'Toggle']
    
    for i in range(num_steps):
        trans = trajectory[i]
        obs = trans['obs']
        action = trans['action']
        reward = trans['reward']
        
        axes[i].imshow(obs)
        action_str = action_names[action] if action >= 0 else 'None'
        axes[i].set_title(f"Step {i}\nAction: {action_str}\nReward: {reward:.1f}", 
                         fontsize=10, fontweight='bold')
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(num_steps, len(axes)):
        axes[i].axis('off')
    
    success = trajectory[-1]['terminated'] and trajectory[-1]['reward'] > 0
    status = "SUCCESS" if success else "INCOMPLETE"
    plt.suptitle(f"Episode Visualization ({len(trajectory)} steps) - {status}", 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory visualization to {output_path}")
    
    plt.show()


def plot_replan_comparison(results, output_path=None):
    """Plot replanning frequency comparison.
    
    Args:
        results: Dictionary mapping replan frequency to results
        output_path: Path to save figure (optional)
    """
    replan_values = sorted(results.keys())
    success_rates = [results[f]['success_rate'] for f in replan_values]
    avg_steps = [results[f]['avg_steps'] for f in replan_values]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Success rate
    axes[0].plot(replan_values, success_rates, 'o-', linewidth=2, markersize=10, color='blue')
    axes[0].set_xlabel('Replan Every N Steps', fontsize=12)
    axes[0].set_ylabel('Success Rate (%)', fontsize=12)
    axes[0].set_title('Success Rate vs Replan Frequency', fontsize=14, fontweight='bold')
    axes[0].set_xticks(replan_values)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 105])
    
    # Average steps
    axes[1].plot(replan_values, avg_steps, 'o-', linewidth=2, markersize=10, color='green')
    axes[1].set_xlabel('Replan Every N Steps', fontsize=12)
    axes[1].set_ylabel('Average Steps', fontsize=12)
    axes[1].set_title('Average Steps vs Replan Frequency', fontsize=14, fontweight='bold')
    axes[1].set_xticks(replan_values)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved replan comparison to {output_path}")
    
    plt.show()


def plot_env_comparison(env_names=None, output_path=None):
    """Visualize and compare different MiniGrid environments.
    
    Args:
        env_names: List of environment names to visualize
        output_path: Path to save figure (optional)
    """
    if env_names is None:
        env_names = [
            "MiniGrid-Empty-5x5-v0",
            "MiniGrid-Empty-Random-5x5-v0",
            "MiniGrid-Dynamic-Obstacles-5x5-v0",
            "MiniGrid-DoorKey-5x5-v0",
        ]
    
    n_envs = len(env_names)
    cols = min(4, n_envs)
    rows = (n_envs + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if n_envs == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i, env_name in enumerate(env_names):
        row, col = i // cols, i % cols
        
        print(f"Rendering: {env_name}")
        
        try:
            env = gym.make(env_name, render_mode="rgb_array")
            env.reset()
            frame = env.render()
            env.close()
            
            axes[row, col].imshow(frame)
            # Extract short name from full env name
            short_name = env_name.replace("MiniGrid-", "").replace("-v0", "")
            axes[row, col].set_title(short_name, fontsize=12, fontweight='bold')
        except Exception as e:
            axes[row, col].text(0.5, 0.5, f"Error:\n{str(e)[:30]}", 
                               ha='center', va='center', transform=axes[row, col].transAxes)
            axes[row, col].set_title(env_name.split("-")[1], fontsize=12)
        
        axes[row, col].axis('off')
    
    # Hide unused axes
    for i in range(n_envs, rows * cols):
        row, col = i // cols, i % cols
        axes[row, col].axis('off')
    
    plt.suptitle('MiniGrid Environment Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved environment comparison to {output_path}")
    
    plt.show()


def plot_episode_with_distances(model, trajectories, planner, num_episodes=3, 
                                 max_steps=30, replan_every=1, output_dir=None):
    """Visualize episode execution with distance trajectory plots.
    
    This generates two plots per episode:
    1. Step-by-step trajectory visualization
    2. Distance to goal over time
    
    Args:
        model: Trained PLDM model
        trajectories: List of trajectories with goal states
        planner: CEM planner instance
        num_episodes: Number of episodes to visualize
        max_steps: Maximum steps per episode
        replan_every: Replan every N steps
        output_dir: Directory to save figures (optional)
    """
    device = next(model.parameters()).device
    action_names = ['L', 'R', 'F', 'Pick', 'Drop', 'Tog', 'Done']
    
    for ep in range(min(num_episodes, len(trajectories))):
        traj = trajectories[ep]
        goal_obs = traj[-1]['obs']
        
        # Get goal encoding
        goal_tensor = torch.FloatTensor(goal_obs).permute(2, 0, 1).unsqueeze(0) / 255.0
        z_goal = model.encode(goal_tensor.to(device)).squeeze(0)
        
        # Create environment
        env = make_env("MiniGrid-DoorKey-5x5-v0")
        env.reset(seed=ep)
        
        # Collect episode data
        episode_images = []
        episode_distances = []
        episode_actions = []
        
        # Initial state
        current_obs = get_full_obs(env)
        episode_images.append(current_obs.copy())
        current_tensor = torch.FloatTensor(current_obs).permute(2, 0, 1).unsqueeze(0) / 255.0
        with torch.no_grad():
            z_current = model.encode(current_tensor.to(device)).squeeze(0)
            distance = torch.norm(z_current - z_goal).item()
            episode_distances.append(distance)
        
        total_steps = 0
        done = False
        success = False
        
        while total_steps < max_steps and not done:
            with torch.no_grad():
                z_current = model.encode(current_tensor.to(device)).squeeze(0)
                actions = planner.plan(z_current, z_goal, verbose=False)
            
            for a_idx in range(min(replan_every, len(actions))):
                action = int(actions[a_idx].cpu().numpy())
                episode_actions.append(action)
                
                obs, reward, done, truncated, _ = env.step(action)
                total_steps += 1
                
                current_obs = get_full_obs(env)
                episode_images.append(current_obs.copy())
                current_tensor = torch.FloatTensor(current_obs).permute(2, 0, 1).unsqueeze(0) / 255.0
                
                with torch.no_grad():
                    z_current = model.encode(current_tensor.to(device)).squeeze(0)
                    distance = torch.norm(z_current - z_goal).item()
                    episode_distances.append(distance)
                
                if done:
                    success = True
                    break
                
                if total_steps >= max_steps:
                    break
        
        env.close()
        
        status = "SUCCESS" if success else "FAILED"
        print(f"\nEpisode {ep+1}: {status} in {total_steps} steps")
        
        # Plot 1: Trajectory visualization
        num_images = len(episode_images)
        cols = min(10, num_images)
        rows = (num_images + cols - 1) // cols
        
        fig1, axes1 = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
        if rows == 1 and cols == 1:
            axes1 = np.array([[axes1]])
        elif rows == 1:
            axes1 = axes1.reshape(1, -1)
        elif cols == 1:
            axes1 = axes1.reshape(-1, 1)
        
        for idx in range(min(num_images, rows * cols)):
            row, col = idx // cols, idx % cols
            axes1[row, col].imshow(episode_images[idx])
            if idx == 0:
                title = f'S0\nD={episode_distances[idx]:.1f}'
            else:
                title = f'S{idx}\n{action_names[episode_actions[idx-1]]}\nD={episode_distances[idx]:.1f}'
            axes1[row, col].set_title(title, fontsize=8)
            axes1[row, col].axis('off')
        
        for idx in range(num_images, rows * cols):
            row, col = idx // cols, idx % cols
            axes1[row, col].axis('off')
        
        status_marker = "[OK]" if success else "[X]"
        fig1.suptitle(f'Episode {ep+1} {status_marker} - {len(episode_images)} frames', 
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if output_dir:
            fig1.savefig(os.path.join(output_dir, f'episode_{ep+1}_trajectory.png'), 
                        dpi=150, bbox_inches='tight')
        
        plt.show()
        
        # Plot 2: Distance trajectory
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        ax2.plot(episode_distances, 'o-', linewidth=2, markersize=5, color='blue')
        ax2.axhline(0, color='green', linestyle='--', label='Goal (distance=0)', linewidth=2)
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('Distance to Goal (Latent)', fontsize=12)
        ax2.set_title(f'Episode {ep+1}: Distance Trajectory ({status})', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Add markers for start and end
        ax2.scatter([0], [episode_distances[0]], color='green', s=100, zorder=5, label='Start')
        ax2.scatter([len(episode_distances)-1], [episode_distances[-1]], 
                   color='red' if not success else 'green', s=100, zorder=5, label='End')
        
        plt.tight_layout()
        
        if output_dir:
            fig2.savefig(os.path.join(output_dir, f'episode_{ep+1}_distances.png'), 
                        dpi=150, bbox_inches='tight')
        
        plt.show()
        
        print(f"  Initial distance: {episode_distances[0]:.2f}")
        print(f"  Final distance: {episode_distances[-1]:.2f}")
        print(f"  Improvement: {episode_distances[0] - episode_distances[-1]:.2f}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate PLDM visualizations',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Mode selection
    parser.add_argument('--mode', type=str, required=True,
                        choices=['training_curves', 'latent_space', 'bfs_trajectories',
                                 'planning', 'trajectory', 'replan_comparison', 
                                 'env_comparison', 'episode_distances', 'all'],
                        help='Visualization mode')
    
    # Input paths
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--history_path', type=str, default=None,
                        help='Path to training history JSON')
    parser.add_argument('--results_path', type=str, default=None,
                        help='Path to evaluation results JSON')
    
    # Model settings
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent space dimension')
    parser.add_argument('--action_dim', type=int, default=7,
                        help='Number of actions')
    
    # Environment settings
    parser.add_argument('--env_name', type=str, default='MiniGrid-DoorKey-5x5-v0',
                        help='MiniGrid environment name')
    
    # Visualization settings
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 4],
                        help='Seeds for BFS trajectory visualization')
    parser.add_argument('--num_trajectories', type=int, default=100,
                        help='Number of trajectories for latent space viz')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Setup
    device = get_device()
    
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    def get_output_path(name):
        if args.output_dir:
            return os.path.join(args.output_dir, name)
        return None
    
    # Execute based on mode
    if args.mode in ['training_curves', 'all']:
        if args.history_path:
            with open(args.history_path, 'r') as f:
                history = json.load(f)
            plot_training_curves(history, get_output_path('training_curves.png'))
        elif args.mode == 'training_curves':
            print("Error: --history_path required for training_curves mode")
            return
    
    if args.mode in ['latent_space', 'all']:
        if args.model_path:
            model = load_model(args.model_path, args.latent_dim, args.action_dim, device)
            trajectories = collect_dataset(
                env_name=args.env_name,
                num_trajectories=args.num_trajectories,
                bfs_ratio=0.8
            )
            plot_latent_space(model, trajectories, get_output_path('latent_space.png'))
        elif args.mode == 'latent_space':
            print("Error: --model_path required for latent_space mode")
            return
    
    if args.mode in ['bfs_trajectories', 'all']:
        for seed in args.seeds:
            output_path = get_output_path(f'bfs_trajectory_seed{seed}.png') if args.output_dir else None
            plot_bfs_trajectory(args.env_name, seed, output_path)
    
    if args.mode in ['planning', 'all']:
        if args.model_path:
            model = load_model(args.model_path, args.latent_dim, args.action_dim, device)
            trajectories = collect_dataset(
                env_name=args.env_name,
                num_trajectories=10,
                bfs_ratio=1.0
            )
            if trajectories:
                plot_planning_episode(model, trajectories[0], 
                                     get_output_path('planning_episode.png'))
        elif args.mode == 'planning':
            print("Error: --model_path required for planning mode")
            return
    
    if args.mode in ['trajectory', 'all']:
        trajectories = collect_dataset(
            env_name=args.env_name,
            num_trajectories=5,
            bfs_ratio=1.0
        )
        if trajectories:
            plot_trajectory_steps(trajectories[0], 
                                 output_path=get_output_path('trajectory_steps.png'))
    
    if args.mode == 'replan_comparison':
        if args.results_path:
            with open(args.results_path, 'r') as f:
                data = json.load(f)
            if 'replan_analysis' in data:
                results = {int(k): v for k, v in data['replan_analysis'].items()}
                plot_replan_comparison(results, get_output_path('replan_comparison.png'))
            else:
                print("Error: Results file does not contain replan_analysis data")
        else:
            print("Error: --results_path required for replan_comparison mode")
            return
    
    if args.mode in ['env_comparison', 'all']:
        # Default set of 5x5 environments for comparison
        env_names = [
            "MiniGrid-Empty-5x5-v0",
            "MiniGrid-Empty-Random-5x5-v0",
            "MiniGrid-Dynamic-Obstacles-5x5-v0",
            "MiniGrid-DoorKey-5x5-v0",
        ]
        plot_env_comparison(env_names, get_output_path('env_comparison.png'))
    
    if args.mode in ['episode_distances', 'all']:
        if args.model_path:
            model = load_model(args.model_path, args.latent_dim, args.action_dim, device)
            trajectories = collect_dataset(
                env_name=args.env_name,
                num_trajectories=10,
                bfs_ratio=1.0
            )
            planner = CEMPlanner(
                model, action_dim=args.action_dim, horizon=15,
                num_iterations=10, num_samples=200, num_elites=30
            )
            plot_episode_with_distances(
                model, trajectories, planner, 
                num_episodes=3, max_steps=30, replan_every=1,
                output_dir=args.output_dir
            )
        elif args.mode == 'episode_distances':
            print("Error: --model_path required for episode_distances mode")
            return
    
    print("\nVisualization complete!")
    if args.output_dir:
        print(f"Outputs saved to: {args.output_dir}")


if __name__ == '__main__':
    main()

