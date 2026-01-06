"""Cross-Entropy Method (CEM) planner for latent space planning."""

import torch


class CEMPlanner:
    """Cross-Entropy Method planner - optimizes action sequences in latent space.
    
    Uses iterative sampling and elite selection to find action sequences
    that minimize distance to goal in latent space.
    """
    
    def __init__(self, model, action_dim=7, horizon=15, 
                 num_iterations=10, num_samples=500, num_elites=50):
        """Initialize CEM planner.
        
        Args:
            model: PLDM model with encode and predict_step methods
            action_dim: Number of discrete actions
            horizon: Planning horizon (number of steps to plan)
            num_iterations: Number of CEM optimization iterations
            num_samples: Number of action sequences to sample per iteration
            num_elites: Number of top sequences to keep for distribution update
        """
        self.model = model
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_iterations = num_iterations
        self.num_samples = num_samples
        self.num_elites = num_elites
    
    @torch.no_grad()
    def plan(self, z_current, z_goal, verbose=False):
        """Plan action sequence from current to goal state.
        
        Args:
            z_current: Current latent state (latent_dim,)
            z_goal: Goal latent state (latent_dim,)
            verbose: Print optimization progress
            
        Returns:
            Best action sequence found (horizon,)
        """
        device = z_current.device
        
        # Initialize uniform action distribution
        action_probs = torch.ones(self.horizon, self.action_dim, device=device) / self.action_dim
        
        best_cost = float('inf')
        best_actions = None
        
        for iteration in range(self.num_iterations):
            # Sample action sequences from current distribution
            action_sequences = torch.multinomial(
                action_probs.repeat(self.num_samples, 1, 1).view(-1, self.action_dim),
                num_samples=1
            ).view(self.num_samples, self.horizon)
            
            # Evaluate sequences
            costs = self._evaluate_sequences(z_current, z_goal, action_sequences)
            
            # Select elite sequences
            elite_indices = torch.argsort(costs)[:self.num_elites]
            elite_actions = action_sequences[elite_indices]
            
            # Update distribution from elites
            action_probs = torch.zeros_like(action_probs)
            for h in range(self.horizon):
                for a in range(self.action_dim):
                    action_probs[h, a] = (elite_actions[:, h] == a).float().mean()
            
            # Add small uniform noise for exploration
            action_probs = 0.9 * action_probs + 0.1 / self.action_dim
            
            # Track best
            if costs[elite_indices[0]] < best_cost:
                best_cost = costs[elite_indices[0]]
                best_actions = elite_actions[0].clone()
            
            if verbose:
                print(f"  Iter {iteration+1}: cost={best_cost:.4f}")
        
        return best_actions
    
    def _evaluate_sequences(self, z_current, z_goal, action_sequences):
        """Evaluate cost of action sequences.
        
        Cost is cumulative distance to goal in latent space.
        
        Args:
            z_current: Current latent state (latent_dim,)
            z_goal: Goal latent state (latent_dim,)
            action_sequences: Batch of action sequences (batch, horizon)
            
        Returns:
            Costs for each sequence (batch,)
        """
        batch_size = action_sequences.shape[0]
        
        # Expand current state for batch
        z = z_current.unsqueeze(0).expand(batch_size, -1)
        
        total_cost = torch.zeros(batch_size, device=z.device)
        
        # Roll out each sequence
        for t in range(self.horizon):
            actions = action_sequences[:, t]
            z = self.model.predict_step(z, actions)
            
            # Distance to goal
            cost = torch.norm(z - z_goal.unsqueeze(0), dim=1)
            total_cost += cost
        
        return total_cost

