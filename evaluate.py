#!/usr/bin/env python3
"""Evaluate trained PLDM model on MiniGrid environments.

This script evaluates model performance on various tasks including:
- Standard DoorKey-5x5 evaluation
- Generalization to other environments (Empty, Dynamic Obstacles)
- Custom DoorKey configurations
- Replanning frequency analysis

Example usage:
    # Basic evaluation on DoorKey-5x5
    python evaluate.py --model_path outputs/run1/checkpoints/best_model.pt
    
    # Evaluation with custom settings
    python evaluate.py --model_path outputs/run1/checkpoints/best_model.pt \\
        --num_episodes 50 --replan_every 3 --output_dir outputs/run1/evaluation
    
    # Test generalization to other environments
    python evaluate.py --model_path outputs/run1/checkpoints/best_model.pt \\
        --env_name MiniGrid-Empty-5x5-v0 --output_dir outputs/run1/generalization
    
    # Evaluate on custom DoorKey configurations
    python evaluate.py --model_path outputs/run1/checkpoints/best_model.pt \\
        --custom_configs --output_dir outputs/run1/custom_eval
"""

import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm

from models import PLDM
from utils.data import bfs_solve, collect_dataset
from utils.environment import make_env, get_full_obs
from utils.planner import CEMPlanner
from utils.custom_env import (
    make_custom_doorkey,
    bfs_solve_custom_env,
    generate_custom_configs
)


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
    
    # Initialize encoder FC layers
    model.init_encoder_fc((3, 40, 40), device)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    return model


def evaluate_standard(model, planner, env_name, trajectories, num_episodes, 
                      max_steps, replan_every, verbose=True):
    """Evaluate on standard environment with hindsight goals.
    
    Args:
        model: Trained PLDM model
        planner: CEM planner
        env_name: Environment name
        trajectories: Pre-collected trajectories for goal states
        num_episodes: Number of episodes to evaluate
        max_steps: Maximum steps per episode
        replan_every: Replan every N steps
        verbose: Print progress
        
    Returns:
        Dictionary of evaluation results
    """
    device = next(model.parameters()).device
    successes = []
    steps_to_goal = []
    
    for ep in range(min(num_episodes, len(trajectories))):
        traj = trajectories[ep]
        goal_obs = traj[-1]['obs']
        
        env = make_env(env_name)
        env.reset(seed=ep)
        
        goal_tensor = torch.FloatTensor(goal_obs).permute(2, 0, 1).unsqueeze(0) / 255.0
        z_goal = model.encode(goal_tensor.to(device)).squeeze(0)
        
        total_steps = 0
        done = False
        
        while total_steps < max_steps and not done:
            current_obs = get_full_obs(env)
            current_tensor = torch.FloatTensor(current_obs).permute(2, 0, 1).unsqueeze(0) / 255.0
            
            with torch.no_grad():
                z_current = model.encode(current_tensor.to(device)).squeeze(0)
                actions = planner.plan(z_current, z_goal, verbose=False)
            
            for a_idx in range(min(replan_every, len(actions))):
                action = int(actions[a_idx].cpu().numpy())
                obs, reward, done, truncated, _ = env.step(action)
                total_steps += 1
                
                if done:
                    successes.append(1)
                    steps_to_goal.append(total_steps)
                    if verbose:
                        print(f"Episode {ep+1}: Success in {total_steps} steps")
                    break
                
                if total_steps >= max_steps:
                    break
        
        if not done:
            successes.append(0)
            if verbose:
                print(f"Episode {ep+1}: Failed")
        
        env.close()
    
    success_rate = np.mean(successes) * 100
    avg_steps = np.mean(steps_to_goal) if steps_to_goal else 0
    
    return {
        'success_rate': success_rate,
        'avg_steps': avg_steps,
        'num_successes': sum(successes),
        'num_episodes': len(successes)
    }


def evaluate_replan_frequencies(model, env_name, trajectories, num_episodes,
                                max_steps, replan_values, verbose=True):
    """Evaluate across different replanning frequencies.
    
    Returns:
        Dictionary mapping replan frequency to results
    """
    results = {}
    
    for replan_every in replan_values:
        if verbose:
            print(f"\nTesting replan_every = {replan_every}")
        
        planner = CEMPlanner(
            model, action_dim=7, horizon=15,
            num_iterations=15, num_samples=500, num_elites=50
        )
        
        result = evaluate_standard(
            model, planner, env_name, trajectories,
            num_episodes, max_steps, replan_every, verbose=False
        )
        
        results[replan_every] = result
        
        if verbose:
            print(f"  Success: {result['success_rate']:.1f}% ({result['num_successes']}/{result['num_episodes']})")
            print(f"  Avg steps: {result['avg_steps']:.1f}")
    
    return results


def evaluate_custom_configs(model, custom_configs, num_trials_per_config,
                            max_steps, replan_every, verbose=True):
    """Evaluate on custom DoorKey configurations.
    
    Returns:
        Dictionary of results per configuration
    """
    device = next(model.parameters()).device
    results = {}
    
    for config_idx, config in enumerate(custom_configs):
        if verbose:
            print(f"\nConfig {config_idx+1}/{len(custom_configs)}: "
                  f"Key={config['key_pos']}, Door={config['door_pos']}, Goal={config['goal_pos']}")
        
        # Generate agent variations
        valid_starts = [(1, y) for y in [1, 2, 3] if (1, y) != config['key_pos']]
        agent_dirs = [0, 1, 2, 3]
        agent_variations = [(s, d) for s in valid_starts for d in agent_dirs]
        
        planner = CEMPlanner(
            model, action_dim=7, horizon=15,
            num_iterations=15, num_samples=500, num_elites=50
        )
        
        config_successes = []
        config_steps = []
        
        for trial in range(num_trials_per_config):
            agent_start, agent_dir = agent_variations[trial % len(agent_variations)]
            
            env = make_custom_doorkey(
                key_pos=config['key_pos'],
                door_pos=config['door_pos'],
                goal_pos=config['goal_pos'],
                agent_start=agent_start,
                agent_dir=agent_dir
            )
            env.reset()
            
            # Get goal from BFS
            bfs_actions = bfs_solve_custom_env(env)
            if bfs_actions is None:
                env.close()
                continue
            
            # Execute BFS to get goal state
            env.reset()
            for a in bfs_actions:
                env.step(a)
            goal_obs = get_full_obs(env)
            
            # Reset for planning
            env.reset()
            goal_tensor = torch.FloatTensor(goal_obs).permute(2, 0, 1).unsqueeze(0) / 255.0
            z_goal = model.encode(goal_tensor.to(device)).squeeze(0)
            
            total_steps = 0
            done = False
            success = False
            
            while total_steps < max_steps and not done:
                current_obs = get_full_obs(env)
                current_tensor = torch.FloatTensor(current_obs).permute(2, 0, 1).unsqueeze(0) / 255.0
                
                with torch.no_grad():
                    z_current = model.encode(current_tensor.to(device)).squeeze(0)
                    actions = planner.plan(z_current, z_goal, verbose=False)
                
                for a_idx in range(min(replan_every, len(actions))):
                    action = int(actions[a_idx].cpu().numpy())
                    obs, reward, done, truncated, _ = env.step(action)
                    total_steps += 1
                    
                    if done:
                        success = True
                        config_successes.append(1)
                        config_steps.append(total_steps)
                        break
                    
                    if total_steps >= max_steps:
                        break
            
            if not success:
                config_successes.append(0)
            
            env.close()
        
        success_rate = np.mean(config_successes) * 100 if config_successes else 0
        avg_steps = np.mean(config_steps) if config_steps else 0
        
        results[config_idx] = {
            'config': config,
            'success_rate': success_rate,
            'avg_steps': avg_steps,
            'num_successes': sum(config_successes),
            'num_trials': len(config_successes)
        }
        
        if verbose:
            print(f"  Success: {success_rate:.1f}% ({sum(config_successes)}/{len(config_successes)})")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained PLDM model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model settings
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--latent_dim', type=int, default=128,
                        help='Latent space dimension')
    parser.add_argument('--action_dim', type=int, default=7,
                        help='Number of actions')
    
    # Environment settings
    parser.add_argument('--env_name', type=str, default='MiniGrid-DoorKey-5x5-v0',
                        help='MiniGrid environment name')
    
    # Evaluation settings
    parser.add_argument('--num_episodes', type=int, default=50,
                        help='Number of episodes for evaluation')
    parser.add_argument('--max_steps', type=int, default=30,
                        help='Maximum steps per episode')
    parser.add_argument('--replan_every', type=int, default=1,
                        help='Replan every N steps')
    
    # Planner settings
    parser.add_argument('--horizon', type=int, default=15,
                        help='Planning horizon')
    parser.add_argument('--num_iterations', type=int, default=15,
                        help='CEM iterations')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='CEM samples per iteration')
    parser.add_argument('--num_elites', type=int, default=50,
                        help='CEM elite samples')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation results')
    
    # Evaluation modes
    parser.add_argument('--replan_analysis', action='store_true',
                        help='Run replanning frequency analysis')
    parser.add_argument('--replan_values', type=int, nargs='+', default=[1, 3, 6, 9, 12, 15],
                        help='Replan frequency values to test')
    parser.add_argument('--custom_configs', action='store_true',
                        help='Evaluate on custom DoorKey configurations')
    parser.add_argument('--num_trials_per_config', type=int, default=4,
                        help='Number of trials per custom config')
    
    args = parser.parse_args()
    
    # Setup
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading model from {args.model_path}")
    model = load_model(args.model_path, args.latent_dim, args.action_dim, device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Collect test trajectories
    print("\nCollecting test trajectories...")
    trajectories = collect_dataset(
        env_name=args.env_name,
        num_trajectories=args.num_episodes,
        bfs_ratio=1.0,  # All BFS trajectories for evaluation
        max_steps=100
    )
    
    results = {}
    
    if args.replan_analysis:
        # Replanning frequency analysis
        print("\n" + "="*60)
        print("REPLANNING FREQUENCY ANALYSIS")
        print("="*60)
        
        replan_results = evaluate_replan_frequencies(
            model, args.env_name, trajectories, args.num_episodes,
            args.max_steps, args.replan_values, verbose=True
        )
        
        results['replan_analysis'] = {str(k): v for k, v in replan_results.items()}
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY: Replan Frequency Analysis")
        print("="*60)
        print(f"{'Replan Every':<15} {'Success Rate':<15} {'Avg Steps':<15}")
        print("-"*50)
        for freq, r in replan_results.items():
            print(f"{freq:<15} {r['success_rate']:<15.1f}% {r['avg_steps']:<15.1f}")
    
    elif args.custom_configs:
        # Custom DoorKey configuration evaluation
        print("\n" + "="*60)
        print("CUSTOM DOORKEY CONFIGURATION EVALUATION")
        print("="*60)
        
        custom_configs = generate_custom_configs(exclude_standard=True)
        print(f"Testing {len(custom_configs)} custom configurations")
        
        custom_results = evaluate_custom_configs(
            model, custom_configs, args.num_trials_per_config,
            args.max_steps, args.replan_every, verbose=True
        )
        
        results['custom_configs'] = {
            str(k): {
                'config': v['config'],
                'success_rate': v['success_rate'],
                'avg_steps': v['avg_steps'],
                'num_successes': v['num_successes'],
                'num_trials': v['num_trials']
            }
            for k, v in custom_results.items()
        }
        
        # Print summary
        total_successes = sum(r['num_successes'] for r in custom_results.values())
        total_trials = sum(r['num_trials'] for r in custom_results.values())
        print("\n" + "="*60)
        print(f"SUMMARY: {total_successes}/{total_trials} ({100*total_successes/total_trials:.1f}%)")
        print("="*60)
    
    else:
        # Standard evaluation
        print("\n" + "="*60)
        print("STANDARD EVALUATION")
        print("="*60)
        
        planner = CEMPlanner(
            model, action_dim=args.action_dim, horizon=args.horizon,
            num_iterations=args.num_iterations, num_samples=args.num_samples,
            num_elites=args.num_elites
        )
        
        result = evaluate_standard(
            model, planner, args.env_name, trajectories,
            args.num_episodes, args.max_steps, args.replan_every, verbose=True
        )
        
        results['standard'] = result
        
        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)
        print(f"Success Rate: {result['success_rate']:.1f}% ({result['num_successes']}/{result['num_episodes']})")
        print(f"Average Steps: {result['avg_steps']:.1f}")
    
    # Save results
    if args.output_dir:
        results_path = os.path.join(args.output_dir, 'evaluation_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()

