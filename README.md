# Latent-space Planning and Disentangled Control in MiniGrid



This project involves first training a reward-free JEPA planning model based on Planning with Latent Dynamics Model (PLDM) paper (Sobal et al, 2025) using BFS optimal + noisy trajectories from the MiniGrid DoorKey 5×5 environment and analyzing its learned latent dynamics. Building on this baseline, I then introduce a disentangled PLDM variant to examine how separating latent factors influences representation quality and downstream planning performance.



The write-up on the project is available [here](https://apurvabhargava.github.io/writeups/minigrid-latent-planning-disentangled)





\# PLDM for MiniGrid DoorKey



A PyTorch implementation of Predictive Latent Dynamics Models (PLDM) for goal-conditioned planning in MiniGrid environments. The model learns a latent space representation of states and a dynamics model that predicts state transitions, enabling planning via Cross-Entropy Method (CEM) optimization.



\## Overview



This repository implements:



\- \*\*PLDM Architecture\*\*: Encoder + Predictor for learning latent dynamics

\- \*\*VICReg Training\*\*: Self-supervised loss with variance and covariance regularization

\- \*\*CEM Planner\*\*: Planning in latent space to reach goal states

\- \*\*Custom Environments\*\*: Configurable DoorKey-5x5 variants for testing generalization



\## Repository Structure



```

pldm\_entangled\_repo/

|-- models/

|   |-- \_\_init\_\_.py

|   |-- encoder.py       # FlexibleEncoder that adapts to observation size

|   |-- predictor.py     # Dynamics predictor for latent transitions

|   |-- pldm.py          # Complete PLDM model

|-- utils/

|   |-- \_\_init\_\_.py

|   |-- environment.py   # MiniGrid environment utilities

|   |-- data.py          # Data collection and dataset classes

|   |-- losses.py        # VICReg loss function

|   |-- planner.py       # CEM planner for latent space planning

|   |-- custom\_env.py    # Custom DoorKey-5x5 environment

|-- outputs/             # Default output directory (created at runtime)

|-- visualizations/      # Default visualization directory (created at runtime)

|-- train.py             # Training script with CLI

|-- evaluate.py          # Evaluation script with CLI

|-- visualize.py         # Visualization generation script

|-- requirements.txt     # Python dependencies

|-- README.md            # This file

```



\## Installation



1\. Clone or copy this repository

2\. Create a virtual environment (recommended):

&nbsp;  ```bash

&nbsp;  python -m venv venv

&nbsp;  source venv/bin/activate  # On Windows: venv\\Scripts\\activate

&nbsp;  ```

3\. Install dependencies:

&nbsp;  ```bash

&nbsp;  pip install -r requirements.txt

&nbsp;  ```



\## Usage



\### Training



Train a PLDM model on MiniGrid-DoorKey-5x5:



```bash

\# Basic training with default parameters

python train.py --output\_dir outputs/my\_run



\# Training with custom parameters

python train.py \\

&nbsp;   --output\_dir outputs/my\_run \\

&nbsp;   --num\_trajectories 2000 \\

&nbsp;   --epochs 150 \\

&nbsp;   --lr 1e-4 \\

&nbsp;   --batch\_size 64 \\

&nbsp;   --latent\_dim 128



\# Resume training from checkpoint

python train.py --output\_dir outputs/my\_run --resume

```



\#### Training Arguments



| Argument | Default | Description |

|----------|---------|-------------|

| `--output\_dir` | `outputs/default` | Directory for checkpoints and logs |

| `--env\_name` | `MiniGrid-DoorKey-5x5-v0` | MiniGrid environment |

| `--num\_trajectories` | 1200 | Number of training trajectories |

| `--bfs\_ratio` | 0.8 | Fraction of optimal (BFS) trajectories |

| `--epochs` | 100 | Training epochs |

| `--batch\_size` | 64 | Batch size |

| `--lr` | 3e-4 | Learning rate |

| `--latent\_dim` | 128 | Latent space dimension |

| `--sim\_coeff` | 1.0 | Similarity loss weight |

| `--std\_coeff` | 1.0 | Variance loss weight |

| `--cov\_coeff` | 0.04 | Covariance loss weight |

| `--resume` | False | Resume from checkpoint |



\### Evaluation



Evaluate a trained model:



```bash

\# Basic evaluation

python evaluate.py --model\_path outputs/my\_run/checkpoints/best\_model.pt



\# Evaluation with custom settings

python evaluate.py \\

&nbsp;   --model\_path outputs/my\_run/checkpoints/best\_model.pt \\

&nbsp;   --num\_episodes 50 \\

&nbsp;   --replan\_every 3 \\

&nbsp;   --output\_dir outputs/my\_run/evaluation



\# Replanning frequency analysis

python evaluate.py \\

&nbsp;   --model\_path outputs/my\_run/checkpoints/best\_model.pt \\

&nbsp;   --replan\_analysis \\

&nbsp;   --replan\_values 1 3 6 9 12 15 \\

&nbsp;   --output\_dir outputs/my\_run/replan\_analysis



\# Test generalization to simpler environments

python evaluate.py \\

&nbsp;   --model\_path outputs/my\_run/checkpoints/best\_model.pt \\

&nbsp;   --env\_name MiniGrid-Empty-5x5-v0 \\

&nbsp;   --output\_dir outputs/my\_run/generalization



\# Evaluate on custom DoorKey configurations

python evaluate.py \\

&nbsp;   --model\_path outputs/my\_run/checkpoints/best\_model.pt \\

&nbsp;   --custom\_configs \\

&nbsp;   --num\_trials\_per\_config 4 \\

&nbsp;   --output\_dir outputs/my\_run/custom\_eval

```



\#### Evaluation Arguments



| Argument | Default | Description |

|----------|---------|-------------|

| `--model\_path` | Required | Path to model checkpoint |

| `--num\_episodes` | 50 | Number of evaluation episodes |

| `--max\_steps` | 30 | Max steps per episode |

| `--replan\_every` | 1 | Replan frequency |

| `--horizon` | 15 | Planning horizon |

| `--replan\_analysis` | False | Run replan frequency analysis |

| `--replan\_values` | `\[1,3,6,9,12,15]` | Replan values to test |

| `--custom\_configs` | False | Evaluate on custom DoorKey configs |



\### Visualization



Generate visualizations from training and evaluation:



```bash

\# Training loss curves

python visualize.py \\

&nbsp;   --mode training\_curves \\

&nbsp;   --history\_path outputs/my\_run/training\_history.json \\

&nbsp;   --output\_dir outputs/my\_run/visualizations



\# Latent space visualization (PCA)

python visualize.py \\

&nbsp;   --mode latent\_space \\

&nbsp;   --model\_path outputs/my\_run/checkpoints/best\_model.pt \\

&nbsp;   --output\_dir outputs/my\_run/visualizations



\# BFS trajectory visualization

python visualize.py \\

&nbsp;   --mode bfs\_trajectories \\

&nbsp;   --seeds 1 2 4 \\

&nbsp;   --output\_dir outputs/my\_run/visualizations



\# Planning episode visualization

python visualize.py \\

&nbsp;   --mode planning \\

&nbsp;   --model\_path outputs/my\_run/checkpoints/best\_model.pt \\

&nbsp;   --output\_dir outputs/my\_run/visualizations



\# Replan frequency comparison (requires evaluation results)

python visualize.py \\

&nbsp;   --mode replan\_comparison \\

&nbsp;   --results\_path outputs/my\_run/evaluation/evaluation\_results.json \\

&nbsp;   --output\_dir outputs/my\_run/visualizations



\# Environment comparison (shows different MiniGrid environments)

python visualize.py \\

&nbsp;   --mode env\_comparison \\

&nbsp;   --output\_dir outputs/my\_run/visualizations



\# Episode execution with distance trajectory plots

python visualize.py \\

&nbsp;   --mode episode\_distances \\

&nbsp;   --model\_path outputs/my\_run/checkpoints/best\_model.pt \\

&nbsp;   --output\_dir outputs/my\_run/visualizations



\# Generate all visualizations

python visualize.py \\

&nbsp;   --mode all \\

&nbsp;   --model\_path outputs/my\_run/checkpoints/best\_model.pt \\

&nbsp;   --history\_path outputs/my\_run/training\_history.json \\

&nbsp;   --output\_dir outputs/my\_run/visualizations

```



\## Output Files



After training and evaluation, the output directory will contain:



```

outputs/my\_run/

|-- config.json                    # Training configuration

|-- training\_history.json          # Loss values per epoch

|-- checkpoints/

|   |-- best\_model.pt              # Best validation checkpoint

|   |-- final\_model.pt             # Final epoch checkpoint

|   |-- latest\_checkpoint.pt       # Full checkpoint for resuming

|-- evaluation/

|   |-- evaluation\_results.json    # Evaluation metrics

|-- visualizations/

|   |-- training\_curves.png        # Loss curves

|   |-- latent\_space.png           # PCA visualization

|   |-- planning\_episode.png       # Planning progress

|   |-- replan\_comparison.png      # Replan frequency analysis

|   |-- env\_comparison.png         # MiniGrid environments comparison

|   |-- episode\_N\_trajectory.png   # Step-by-step episode execution

|   |-- episode\_N\_distances.png    # Distance to goal over time

```



\## Model Architecture



\### Encoder (FlexibleEncoder)



\- 3 convolutional layers with stride 2 (downsampling)

\- Adaptive FC layers initialized on first forward pass

\- Handles both single observations and sequences



\### Predictor



\- Action embedding layer

\- 3-layer MLP with residual connection

\- Predicts next latent state from current state and action



\### VICReg Loss



\- Similarity: MSE between predicted and target latent states

\- Variance: Encourages each latent dimension to have unit variance

\- Covariance: Decorrelates latent dimensions to prevent collapse



\## Environments



The code supports all MiniGrid 5x5 environments:



\- `MiniGrid-DoorKey-5x5-v0` (default, trained on this)

\- `MiniGrid-Empty-5x5-v0`

\- `MiniGrid-Empty-Random-5x5-v0`

\- `MiniGrid-Dynamic-Obstacles-5x5-v0`



Custom DoorKey configurations are also supported with configurable key, door, and goal positions.



\## Tips



1\. \*\*GPU Usage\*\*: The code automatically detects MPS (Apple Silicon), CUDA, or falls back to CPU.



2\. \*\*Data Collection\*\*: BFS trajectories provide optimal demonstrations. Random trajectories add exploration diversity.



3\. \*\*Replanning\*\*: Lower replan frequencies (1-3) generally give better success rates but require more computation.



4\. \*\*Latent Dimension\*\*: 128 works well for 5x5 environments. Larger environments may need higher dimensions.



5\. \*\*Training Time\*\*: ~20-30 minutes on Apple M1/M2 for 100 epochs with 1200 trajectories.



\## References



This implementation is based on concepts from:



\- JEPA (Joint Embedding Predictive Architecture)

\- VICReg (Variance-Invariance-Covariance Regularization)

\- MiniGrid environment suite



