"""Environment utilities for MiniGrid."""

import gymnasium as gym
import minigrid
from PIL import Image


def make_env(env_name="MiniGrid-DoorKey-5x5-v0"):
    """Create environment with FULL RGB observation.
    
    Args:
        env_name: Name of the MiniGrid environment
        
    Returns:
        Gymnasium environment instance
    """
    env = gym.make(env_name, render_mode='rgb_array')
    return env


def get_full_obs(env, tile_size=8):
    """Get full RGB observation - downsampled for efficiency.
    
    Args:
        env: Gymnasium environment instance
        tile_size: Size of each tile in rendered image (smaller = faster)
        
    Returns:
        RGB numpy array of the full grid observation
    """
    if hasattr(env.unwrapped, 'grid'):
        grid = env.unwrapped.grid
        
        img = grid.render(
            tile_size,
            agent_pos=env.unwrapped.agent_pos,
            agent_dir=env.unwrapped.agent_dir,
            highlight_mask=None
        )
        
        return img
    else:
        return env.render()

