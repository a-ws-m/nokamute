# PPO Training for Hive

This directory contains a complete PPO (Proximal Policy Optimization) implementation for training Hive agents using TorchRL.

## Overview

The implementation uses:
- **TorchRL** for RL infrastructure (environment, collectors, loss modules)
- **Graph Neural Networks** for policy and value functions
- **Self-play evaluation** with a 55% win rate threshold for model acceptance
- **PPO algorithm** for stable policy optimization

## Key Components

### 1. Environment (`hive_env.py`)
- Custom TorchRL `EnvBase` implementation wrapping the Hive game
- Graph-based observations (nodes, edges, action masks)
- Discrete action space with masking for legal moves
- Reward: +1 for win, -1 for loss, 0 for draw

### 2. Models (`ppo_models.py`)
- **Actor**: Graph encoder → Policy head → Categorical distribution with action masking
- **Critic**: Graph encoder → Value head → State value estimate
- Both use GAT (Graph Attention) layers for processing board states

### 3. Self-Play Evaluator (`self_play_evaluator.py`)
- Pits two models against each other
- Computes win rates
- Implements 55% acceptance threshold for new models

### 4. Training Script (`train_ppo.py`)
- Main PPO training loop using TorchRL components
- Self-play data collection
- Advantage estimation with GAE (Generalized Advantage Estimation)
- Model gating: only accept new model if it beats previous model at 55%+ win rate
- TensorBoard logging

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Build the Rust extension (if not already done)
cd python
maturin develop --release
```

## Usage

### Basic Training

```bash
python train_ppo.py \
    --total-frames 100000 \
    --frames-per-batch 1000 \
    --eval-interval 5 \
    --eval-games 50 \
    --checkpoint-dir checkpoints_ppo
```

### Key Arguments

**Environment:**
- `--max-actions`: Maximum number of possible actions (default: 500)
- `--max-moves`: Maximum moves per game (default: 200)

**Model:**
- `--hidden-dim`: Hidden dimension for GNN (default: 128)
- `--num-layers`: Number of GAT layers (default: 4)
- `--num-heads`: Number of attention heads (default: 4)

**PPO:**
- `--frames-per-batch`: Frames collected per iteration (default: 1000)
- `--total-frames`: Total training frames (default: 100000)
- `--num-epochs`: Training epochs per batch (default: 10)
- `--sub-batch-size`: Mini-batch size (default: 64)
- `--clip-epsilon`: PPO clipping parameter (default: 0.2)
- `--gamma`: Discount factor (default: 0.99)
- `--lmbda`: GAE lambda (default: 0.95)
- `--lr`: Learning rate (default: 3e-4)

**Evaluation:**
- `--eval-interval`: Evaluate every N iterations (default: 5)
- `--eval-games`: Games per evaluation (default: 50)
- `--acceptance-threshold`: Win rate threshold (default: 0.55)

**System:**
- `--device`: Device to use (default: cuda if available, else cpu)
- `--checkpoint-dir`: Where to save checkpoints (default: checkpoints_ppo)

### Example: Quick Test

```bash
# Small training run for testing
python train_ppo.py \
    --total-frames 10000 \
    --frames-per-batch 500 \
    --eval-interval 3 \
    --eval-games 20 \
    --checkpoint-dir test_run
```

### Example: Full Training

```bash
# Longer training with more evaluation games
python train_ppo.py \
    --total-frames 1000000 \
    --frames-per-batch 2000 \
    --eval-interval 10 \
    --eval-games 100 \
    --hidden-dim 256 \
    --num-layers 6 \
    --checkpoint-dir checkpoints_full
```

## Model Acceptance Gate

The training implements a competitive self-play improvement gate:

1. Every `--eval-interval` iterations, the current model plays against the previous best model
2. If the new model wins ≥ 55% of games (configurable via `--acceptance-threshold`), it becomes the new best model
3. Otherwise, the model is reverted to the previous best
4. This ensures monotonic improvement in self-play strength

## Architecture Details

### Graph Observations

The board state is represented as a graph:
- **Nodes**: Pieces on the board (11 features: color, bug type one-hot, height)
- **Edges**: Adjacency relationships (horizontal and vertical)
- **Action Mask**: Binary mask indicating which actions are legal

### Actor-Critic

- **Shared Encoder**: GAT layers process the graph
- **Actor Head**: MLP → logits → Masked Categorical distribution
- **Critic Head**: MLP → scalar state value

### PPO Loss

The standard PPO loss with three components:
1. **Policy loss**: Clipped surrogate objective
2. **Value loss**: Smooth L1 between predicted and target values
3. **Entropy bonus**: Encourages exploration

## Monitoring

Training progress can be monitored with TensorBoard:

```bash
tensorboard --logdir checkpoints_ppo/logs
```

Metrics logged:
- Training loss
- Learning rate
- Episode rewards
- Model acceptance decisions

## Checkpoints

The training saves:
- `checkpoint_iter_N.pt`: Checkpoint every `--eval-interval` iterations
- `best_model.pt`: Current best model (highest self-play performance)
- `final_model.pt`: Final model at end of training

Each checkpoint contains:
- Actor and critic state dicts
- Optimizer state
- Training arguments
- Best actor state (for gating)

## Comparison with Previous Training

### Previous (`train.py`)
- Simple value-based learning
- MSE loss on game outcomes
- No policy optimization
- Random self-play data generation

### New (`train_ppo.py`)
- PPO policy gradient method
- Actor-critic architecture
- On-policy learning with proper advantage estimation
- Competitive self-play with acceptance threshold
- More stable and sample-efficient

## Testing

Test individual components:

```bash
# Test environment
python hive_env.py

# Test models
python ppo_models.py

# Test self-play evaluator
python self_play_evaluator.py
```

## References

- [PPO Paper](https://arxiv.org/abs/1707.06347)
- [TorchRL Documentation](https://pytorch.org/rl/)
- [TorchRL PPO Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html)
