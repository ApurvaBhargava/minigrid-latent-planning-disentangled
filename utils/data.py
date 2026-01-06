"""Data collection and dataset utilities."""

import numpy as np
import torch
from torch.utils.data import Dataset
from collections import deque
from tqdm import tqdm
import gymnasium as gym

from .environment import make_env, get_full_obs


def bfs_solve(env_name, seed):
    """Optimal BFS solver for DoorKey environment.
    
    Solves DoorKey by exploring grid positions and directions.
    
    Args:
        env_name: Name of the MiniGrid environment
        seed: Random seed for environment reset
        
    Returns:
        List of actions to reach goal, or None if unsolvable
    """
    env = gym.make(env_name, render_mode=None)
    env.reset(seed=seed)
    unwrapped = env.unwrapped
    
    # Direction vectors: 0=Right, 1=Down, 2=Left, 3=Up
    DIR_TO_VEC = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    
    start_pos = unwrapped.agent_pos
    start_dir = unwrapped.agent_dir
    grid = unwrapped.grid
    
    # Queue: (x, y, dir, actions, carrying)
    queue = deque([(start_pos[0], start_pos[1], start_dir, [], None)])
    visited = set([(start_pos[0], start_pos[1], start_dir, None)])
    
    while queue:
        x, y, d, actions, carrying = queue.popleft()
        
        # Direction vector
        dx, dy = DIR_TO_VEC[d]
        fx, fy = x + dx, y + dy
        
        if 0 <= fx < grid.width and 0 <= fy < grid.height:
            cell = grid.get(fx, fy)
            
            # Check if we can reach goal
            if cell and cell.type == 'goal':
                env.close()
                return actions + [unwrapped.actions.forward]
            
            # Check if we can pick up key
            if cell and cell.type == 'key' and carrying is None:
                new_carrying = 'key'
                new_state = (x, y, d, new_carrying)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append((x, y, d, actions + [unwrapped.actions.pickup], new_carrying))
            
            # Check if we can open door
            if cell and cell.type == 'door' and not cell.is_open:
                if carrying == 'key':
                    new_state = (x, y, d, carrying)
                    queue.append((fx, fy, d, actions + [unwrapped.actions.toggle, unwrapped.actions.forward], carrying))
            
            # Check if we can move forward
            can_move = (cell is None) or \
                       (cell and cell.type == 'door' and cell.is_open) or \
                       (cell and cell.type == 'goal')
            
            if can_move:
                state = (fx, fy, d, carrying)
                if state not in visited:
                    visited.add(state)
                    queue.append((fx, fy, d, actions + [unwrapped.actions.forward], carrying))
        
        # Try turning left and right
        for turn in [-1, 1]:
            td = (d + turn) % 4
            state = (x, y, td, carrying)
            if state not in visited:
                visited.add(state)
                action_type = unwrapped.actions.left if turn == -1 else unwrapped.actions.right
                queue.append((x, y, td, actions + [action_type], carrying))
    
    env.close()
    return None


def collect_trajectory(env_name, seed, actions=None, max_steps=100):
    """Collect a single trajectory with full RGB observations.
    
    Args:
        env_name: Name of the MiniGrid environment
        seed: Random seed for environment reset
        actions: If provided, follow these actions (BFS trajectory).
                 Otherwise, use random actions.
        max_steps: Maximum steps for random trajectories
        
    Returns:
        List of transition dictionaries, or None if empty
    """
    env = make_env(env_name)
    obs, _ = env.reset(seed=seed)
    
    trajectory = []
    
    if actions is None:
        # Random trajectory
        for step in range(max_steps):
            rgb_obs = get_full_obs(env)
            
            action = env.action_space.sample()
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            trajectory.append({
                'obs': rgb_obs.copy(),
                'action': action,
                'reward': reward,
                'terminated': terminated
            })
            
            obs = next_obs
            
            if terminated or truncated:
                final_obs = get_full_obs(env)
                trajectory.append({
                    'obs': final_obs.copy(),
                    'action': -1,
                    'reward': 0.0,
                    'terminated': True
                })
                break
    else:
        # Follow provided actions (BFS)
        for action in actions:
            rgb_obs = get_full_obs(env)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            trajectory.append({
                'obs': rgb_obs.copy(),
                'action': action,
                'reward': reward,
                'terminated': terminated
            })
            
            obs = next_obs
            
            if terminated:
                final_obs = get_full_obs(env)
                trajectory.append({
                    'obs': final_obs.copy(),
                    'action': -1,
                    'reward': 0.0,
                    'terminated': True
                })
                break
    
    env.close()
    return trajectory if len(trajectory) > 0 else None


def collect_dataset(env_name="MiniGrid-DoorKey-5x5-v0", 
                   num_trajectories=100,
                   bfs_ratio=0.8,
                   max_steps=100,
                   verbose=True):
    """Collect dataset with mix of BFS (optimal) and random trajectories.
    
    Args:
        env_name: Name of the MiniGrid environment
        num_trajectories: Total number of trajectories to collect
        bfs_ratio: Fraction of trajectories that should be BFS-solved (optimal)
        max_steps: Maximum steps for random trajectories
        verbose: Whether to print progress
        
    Returns:
        List of trajectory lists
    """
    num_bfs = int(num_trajectories * bfs_ratio)
    num_random = num_trajectories - num_bfs
    bfs_seeds = [0 + i for i in range(num_bfs * 3)]  # Extra for retries
    random_seeds = [4000 + i for i in range(num_random)]
    
    trajectories = []
    
    # Collect BFS trajectories
    if verbose:
        print(f"Collecting {num_bfs} BFS trajectories...")
    attempts = 0
    bfs_index = 0
    pbar = tqdm(total=num_bfs, disable=not verbose)
    while len([t for t in trajectories if t]) < num_bfs and attempts < num_bfs * 3:
        actions = bfs_solve(env_name, bfs_seeds[bfs_index])
        
        if actions is not None:
            traj = collect_trajectory(env_name, bfs_seeds[bfs_index], actions=actions)
            if traj and len(traj) > 5:
                trajectories.append(traj)
                pbar.update(1)
        
        attempts += 1
        bfs_index += 1
    pbar.close()
    
    # Collect random trajectories
    if num_random > 0:
        if verbose:
            print(f"\nCollecting {num_random} random trajectories...")
        random_index = 0
        for _ in tqdm(range(num_random), disable=not verbose):
            traj = collect_trajectory(env_name, random_seeds[random_index], actions=None, max_steps=max_steps)
            random_index += 1
            if traj and len(traj) > 5:
                trajectories.append(traj)
    
    if verbose:
        print(f"\nCollected {len(trajectories)} trajectories")
        print(f"  Total transitions: {sum(len(t) for t in trajectories)}")
        print(f"  Avg trajectory length: {np.mean([len(t) for t in trajectories]):.1f}")
    
    return trajectories


class TrajectoryDataset(Dataset):
    """Extract observation sequences from trajectories for dynamics learning.
    
    Returns: obs[0:T], actions[0:T], obs[1:T+1]
    """
    
    def __init__(self, trajectories, sequence_length=8):
        """Initialize dataset.
        
        Args:
            trajectories: List of trajectory lists from collect_dataset
            sequence_length: Length of sequences to extract
        """
        self.trajectories = trajectories
        self.sequence_length = sequence_length
        
        # Find valid starting points
        self.valid_starts = []
        for traj_idx, traj in enumerate(trajectories):
            # Need sequence_length+1 observations (obs[0] through obs[sequence_length])
            for start_idx in range(len(traj) - sequence_length):
                self.valid_starts.append((traj_idx, start_idx))
        
        print(f"Dataset: {len(self.valid_starts)} sequences from {len(trajectories)} trajectories")
    
    def __len__(self):
        return len(self.valid_starts)
    
    def __getitem__(self, idx):
        traj_idx, start_idx = self.valid_starts[idx]
        trajectory = self.trajectories[traj_idx]
        
        # Extract sequence
        observations = []
        actions = []
        
        for i in range(self.sequence_length):
            trans = trajectory[start_idx + i]
            observations.append(trans['obs'])
            actions.append(trans['action'])
        
        # Next observations (shifted by 1)
        next_observations = []
        for i in range(self.sequence_length):
            trans_next = trajectory[start_idx + i + 1]
            next_observations.append(trans_next['obs'])
        
        # Convert to tensors
        # RGB observations: (seq, H, W, 3) -> (seq, 3, H, W), normalize to [0, 1]
        observations = torch.FloatTensor(np.array(observations)).permute(0, 3, 1, 2) / 255.0
        next_observations = torch.FloatTensor(np.array(next_observations)).permute(0, 3, 1, 2) / 255.0
        actions = torch.LongTensor(actions)
        
        return observations, actions, next_observations

