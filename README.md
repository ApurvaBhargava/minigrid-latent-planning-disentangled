# Latent-space Planning and Disentangled Control in MiniGrid



This project involves first training a reward-free JEPA planning model based on Planning with Latent Dynamics Model (PLDM) paper [(Sobal et al, 2025)](https://latent-planning.github.io/static/paper.pdf) using BFS optimal + noisy trajectories from the MiniGrid DoorKey 5×5 environment and analyzing its learned latent dynamics. Building on this baseline, I then introduce a disentangled PLDM variant to examine how separating latent factors influences representation quality and downstream planning performance.



The write-up on the project is available [here](https://apurvabhargava.github.io/writeups/minigrid-latent-planning-disentangled)





## PLDM for MiniGrid DoorKey



A PyTorch implementation of Predictive Latent Dynamics Models (PLDM) for goal-conditioned planning in MiniGrid environments. The model learns a latent space representation of states and a dynamics model that predicts state transitions, enabling planning via Cross-Entropy Method (CEM) optimization.



## Overview



This repository implements:



- **PLDM Architecture**: Encoder + Predictor for learning latent dynamics

- **VICReg Training**: Self-supervised loss with variance and covariance regularization

- **CEM Planner**: Planning in latent space to reach goal states

- **Custom Environments**: Configurable DoorKey-5x5 variants for testing generalization



## Repository Structure



```
pldm_entangled_repo/
|-- models/
|   |-- __init__.py
|   |-- encoder.py       # FlexibleEncoder that adapts to observation size
|   |-- predictor.py     # Dynamics predictor for latent transitions
|   |-- pldm.py          # Complete PLDM model
|-- utils/
|   |-- __init__.py
|   |-- environment.py   # MiniGrid environment utilities
|   |-- data.py          # Data collection and dataset classes
|   |-- losses.py        # VICReg loss function
|   |-- planner.py       # CEM planner for latent space planning
|   |-- custom_env.py    # Custom DoorKey-5x5 environment
|-- outputs/             # Default output directory (created at runtime)
|-- visualizations/      # Default visualization directory (created at runtime)
|-- train.py             # Training script with CLI
|-- evaluate.py          # Evaluation script with CLI
|-- visualize.py         # Visualization generation script
|-- requirements.txt     # Python dependencies
|-- README.md            # This file
```



## Installation



1. Clone or copy this repository

2. Create a virtual environment (recommended):

  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```

3. Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```



## Usage



### Training



Train a PLDM model on MiniGrid-DoorKey-5x5:



```bash
# Basic training with default parameters
python train.py --output_dir outputs/my_run

# Training with custom parameters
python train.py \
   --output_dir outputs/my_run \
   --num_trajectories 2000 \
   --epochs 150 \
   --lr 1e-4 \
   --batch_size 64 \
   --latent_dim 128

# Resume training from checkpoint
python train.py --output_dir outputs/my_run --resume
```



#### Training Arguments



| Argument | Default | Description |
|----------|---------|-------------|
| `--output_dir` | `outputs/default` | Directory for checkpoints and logs |
| `--env_name` | `MiniGrid-DoorKey-5x5-v0` | MiniGrid environment |
| `--num_trajectories` | 1200 | Number of training trajectories |
| `--bfs_ratio` | 0.8 | Fraction of optimal (BFS) trajectories |
| `--epochs` | 100 | Training epochs |
| `--batch_size` | 64 | Batch size |
| `--lr` | 3e-4 | Learning rate |
| `--latent_dim` | 128 | Latent space dimension |
| `--sim_coeff` | 1.0 | Similarity loss weight |
| `--std_coeff` | 1.0 | Variance loss weight |
| `--cov_coeff` | 0.04 | Covariance loss weight |
| `--resume` | False | Resume from checkpoint |



### Evaluation



Evaluate a trained model:



```bash
# Basic evaluation
python evaluate.py --model_path outputs/my_run/checkpoints/best_model.pt

# Evaluation with custom settings
python evaluate.py \
   --model_path outputs/my_run/checkpoints/best_model.pt \
   --num_episodes 50 \
   --replan_every 3 \
   --output_dir outputs/my_run/evaluation

# Replanning frequency analysis
python evaluate.py \
   --model_path outputs/my_run/checkpoints/best_model.pt \
   --replan_analysis \
   --replan_values 1 3 6 9 12 15 \
   --output_dir outputs/my_run/replan_analysis

# Test generalization to simpler environments
python evaluate.py \
   --model_path outputs/my_run/checkpoints/best_model.pt \
   --env_name MiniGrid-Empty-5x5-v0 \
   --output_dir outputs/my_run/generalization

# Evaluate on custom DoorKey configurations
python evaluate.py \
   --model_path outputs/my_run/checkpoints/best_model.pt \
   --custom_configs \
   --num_trials_per_config 4 \
   --output_dir outputs/my_run/custom_eval
```



#### Evaluation Arguments



| Argument | Default | Description |
|----------|---------|-------------|
| `--model_path` | Required | Path to model checkpoint |
| `--num_episodes` | 50 | Number of evaluation episodes |
| `--max_steps` | 30 | Max steps per episode |
| `--replan_every` | 1 | Replan frequency |
| `--horizon` | 15 | Planning horizon |
| `--replan_analysis` | False | Run replan frequency analysis |
| `--replan_values` | `\[1,3,6,9,12,15]` | Replan values to test |
| `--custom_configs` | False | Evaluate on custom DoorKey configs |



### Visualization



Generate visualizations from training and evaluation:



```bash
# Training loss curves
python visualize.py \
   --mode training_curves \
   --history_path outputs/my_run/training_history.json \
   --output_dir outputs/my_run/visualizations

# Latent space visualization (PCA)
python visualize.py \
   --mode latent_space \
   --model_path outputs/my_run/checkpoints/best_model.pt \
   --output_dir outputs/my_run/visualizations

# BFS trajectory visualization
python visualize.py \
   --mode bfs_trajectories \
   --seeds 1 2 4 \
   --output_dir outputs/my_run/visualizations

# Planning episode visualization
python visualize.py \
   --mode planning \
   --model_path outputs/my_run/checkpoints/best_model.pt \
   --output_dir outputs/my_run/visualizations

# Replan frequency comparison (requires evaluation results)
python visualize.py \
   --mode replan_comparison \
   --results_path outputs/my_run/evaluation/evaluation_results.json \
   --output_dir outputs/my_run/visualizations

# Environment comparison (shows different MiniGrid environments)
python visualize.py \
   --mode env_comparison \
   --output_dir outputs/my_run/visualizations

# Episode execution with distance trajectory plots
python visualize.py \
   --mode episode_distances \
   --model_path outputs/my_run/checkpoints/best_model.pt \
   --output_dir outputs/my_run/visualizations

# Generate all visualizations
python visualize.py \
   --mode all \
   --model_path outputs/my_run/checkpoints/best_model.pt \
   --history_path outputs/my_run/training_history.json \
   --output_dir outputs/my_run/visualizations
```



## Output Files



After training and evaluation, the output directory will contain:



```
outputs/my_run/
|-- config.json                    # Training configuration
|-- training_history.json          # Loss values per epoch
|-- checkpoints/
|   |-- best_model.pt              # Best validation checkpoint
|   |-- final_model.pt             # Final epoch checkpoint
|   |-- latest_checkpoint.pt       # Full checkpoint for resuming
|-- evaluation/
|   |-- evaluation_results.json    # Evaluation metrics
|-- visualizations/
|   |-- training_curves.png        # Loss curves
|   |-- latent_space.png           # PCA visualization
|   |-- planning_episode.png       # Planning progress
|   |-- replan_comparison.png      # Replan frequency analysis
|   |-- env_comparison.png         # MiniGrid environments comparison
|   |-- episode_N_trajectory.png   # Step-by-step episode execution
|   |-- episode_N_distances.png    # Distance to goal over time
```



## Model Architecture



### Encoder (FlexibleEncoder)



- 3 convolutional layers with stride 2 (downsampling)

- Adaptive FC layers initialized on first forward pass

- Handles both single observations and sequences



### Predictor



- Action embedding layer

- 3-layer MLP with residual connection

- Predicts next latent state from current state and action



### VICReg Loss



- Similarity: MSE between predicted and target latent states

- Variance: Encourages each latent dimension to have unit variance

- Covariance: Decorrelates latent dimensions to prevent collapse



## Environments



The code supports all MiniGrid 5x5 environments:



- `MiniGrid-DoorKey-5x5-v0` (default, trained on this)

- `MiniGrid-Empty-5x5-v0`

- `MiniGrid-Empty-Random-5x5-v0`

- `MiniGrid-Dynamic-Obstacles-5x5-v0`



Custom DoorKey configurations are also supported with configurable key, door, and goal positions.



## Notes



1. **GPU Usage**: The code automatically detects MPS (Apple Silicon), CUDA, or falls back to CPU.

2. **Data Collection**: BFS trajectories provide optimal demonstrations. Random trajectories add exploration diversity.

3. **Replanning**: Mid-range replan frequencies (6-9) generally give better success rates but require more computation.

4. **Latent Dimension**: 128 works well for 5x5 environments. Larger environments may need higher dimensions.


## References


This implementation is based on concepts from:

- [PLDM](https://latent-planning.github.io/static/paper.pdf) (Planning With Latent Dynamics Model)

- JEPA (Joint Embedding Predictive Architecture)

- VICReg (Variance-Invariance-Covariance Regularization)

- [MiniGrid environment suite](https://minigrid.farama.org/index.html)



