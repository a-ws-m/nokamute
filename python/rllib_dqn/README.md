# DQN with Custom GNN RLModule for Hive

This directory contains a custom RLlib RLModule implementation for training DQN agents on the Hive board game using Graph Neural Networks.

## Files

- `hive_gnn_rl_module.py` - Custom GNN-based RLModule for DQN
- `__init__.py` - Package initialization
- `README.md` - This file

## Architecture

The `HiveGNNRLModule` implements a custom Graph Neural Network architecture for processing the Hive board state:

### Network Components

1. **Node Embedding Layer**: Projects raw node features (12-dim) to hidden dimension
2. **GAT Layers**: Multiple Graph Attention (GATv2) layers with:
   - Multi-head attention (configurable heads)
   - Batch normalization
   - Residual connections (after first layer)
3. **Global Pooling**: Combines mean and max pooling over graph
4. **Q-Value Head**: MLP that outputs Q-values for all actions

### Input Features

Node features (12-dimensional):
- Color (1): Player color
- Bug type one-hot (9): Queen, Ant, Grasshopper, etc.
- Height (1): Stack height on hex
- Current player (1): Whose turn it is

### Output

Q-values for all 5985 possible actions, with action masking applied to enforce legal moves.

## Usage

### Training

```python
# Basic training command
uv run python train_dqn.py

# With custom hyperparameters
uv run python train_dqn.py \
    --num-iterations 200 \
    --hidden-dim 256 \
    --num-layers 6 \
    --learning-rate 0.0001 \
    --num-workers 8
```

### Testing

```python
# Run test suite
uv run python test_dqn_module.py
```

## Configuration

### Model Hyperparameters

- `node_features`: 12 (fixed by environment)
- `hidden_dim`: 128 (default, configurable)
- `num_layers`: 4 (default, configurable)
- `num_heads`: 4 (default, configurable)
- `dropout`: 0.1

### Training Hyperparameters

- `learning_rate`: 0.0005
- `gamma`: 0.99
- `train_batch_size`: 32
- `replay_buffer_capacity`: 100,000
- `double_q`: True
- `dueling`: True
- `target_network_update_freq`: 500
- `exploration_fraction`: 0.3
- `final_epsilon`: 0.05

## Multi-Agent Setup

The implementation uses parameter sharing:
- Both players (White and Black) use the same policy
- Single RLModule shared across both agents
- Configured via `MultiRLModuleSpec` with single "shared_policy"

## Requirements

- ray[rllib] >= 2.36.1
- torch >= 2.0.0
- torch-geometric >= 2.3.0
- pettingzoo (via nokamute environment)

## Implementation Notes

### Key Design Decisions

1. **Action Masking**: Invalid actions are masked with very negative Q-values (-1e10) rather than removing them from the action space
2. **Graph Representation**: Empty boards handled gracefully with zero-sized tensors
3. **Parameter Sharing**: Single policy for both players simplifies learning in this symmetric game
4. **RLModule API**: Uses new RLlib API stack with proper decorators (`@OverrideToImplementCustomLogic`, `@override`)

### Troubleshooting

If you see "0 parameters" when creating the module:
- Ensure `model_config` parameter is used (not `model_config_dict`)
- Check that `setup()` method has proper decorators
- Verify `try_import_torch()` is working correctly

If you see override decorator errors:
- Use `@OverrideToImplementCustomLogic` for optional overrides
- Use `@override(BaseClass)` with explicit parent class for required overrides
- Order matters: `@OverrideToImplementCustomLogic` before `@override()`

## Future Improvements

- [ ] Add value head for advantage-based learning
- [ ] Implement recurrent version for sequential decision making
- [ ] Add intrinsic curiosity for better exploration
- [ ] Tune hyperparameters via Ray Tune
- [ ] Add multi-agent competitive training (self-play)
