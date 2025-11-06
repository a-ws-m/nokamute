# Evaluation Matching with Branching Markov Chain

## Overview

This implementation uses a **branching Markov chain** approach to generate random positions for evaluation matching pre-training, rather than playing full games with the engine.

## Key Changes

### 1. Random Position Generation (Not Game Playing)

**Old Approach:**
- Played complete games using the engine
- Collected positions from engine's minimax search
- Required games to reach terminal states
- Limited diversity due to game structure

**New Approach:**
- Generates random positions via branching random walks
- No need to reach terminal states
- Better exploration of state space
- More diverse position distribution

### 2. Branching Markov Chain Process

The algorithm works as follows:

```python
1. Start from initial position
2. Apply random legal move
3. With probability p, create branches:
   - Continue main path
   - Create 1-2 branch points for other moves
4. Repeat until N unique positions collected
5. Evaluate each position with Rust engine
```

### 3. Custom Dataset Implementation

Implemented `EvaluationMatchingDataset` that:
- Inherits from `torch_geometric.data.Dataset`
- Generates positions on-the-fly each epoch
- Supports position regeneration for better diversity
- Integrates seamlessly with PyTorch DataLoader

### 4. Training Loop

Each epoch:
1. Generate N unique random positions (optional: regenerate)
2. Evaluate each with Rust engine's `BasicEvaluator`
3. Normalize evaluations using tanh scaling
4. Train GNN using MSE loss with mini-batching
5. Repeat for all epochs

## Usage

### Basic Usage

```python
from pretrain.eval_matching import pretrain_eval_matching
from model import create_model

# Create model
model = create_model({"hidden_dim": 32, "num_layers": 3})

# Pre-train using branching Markov chain
losses = pretrain_eval_matching(
    model=model,
    num_positions_per_epoch=1000,  # Generate 1000 positions per epoch
    num_epochs=50,
    batch_size=64,
    device="cuda",
    aggression=3,
    max_depth=50,
    branch_probability=0.3,
    regenerate_each_epoch=True,  # Fresh positions each epoch
)
```

### Advanced: Using the Dataset Directly

```python
from pretrain.eval_matching import EvaluationMatchingDataset
from torch_geometric.loader import DataLoader

# Create dataset
dataset = EvaluationMatchingDataset(
    num_positions=1000,
    aggression=3,
    max_depth=50,
    branch_probability=0.3,
    scale=0.001,
    regenerate_each_epoch=True,
)

# Create DataLoader
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Training loop
for epoch in range(num_epochs):
    for batch in loader:
        # Training step
        optimizer.zero_grad()
        predictions, _ = model(batch.x, batch.edge_index, batch.batch)
        loss = F.mse_loss(predictions, batch.y)
        loss.backward()
        optimizer.step()
    
    # Regenerate positions for next epoch
    dataset.regenerate()
```

### Command Line

```bash
# Pre-train with evaluation matching
python train.py \
    --pretrain eval-matching \
    --pretrain-games 50 \
    --epochs 100 \
    --batch-size 64 \
    --lr 1e-3 \
    --max-moves 50
```

## Parameters

### `pretrain_eval_matching`

- `num_positions_per_epoch`: Number of unique positions to generate per epoch (default: 1000)
- `num_epochs`: Number of training epochs (default: 50)
- `batch_size`: Mini-batch size for training (default: 64)
- `aggression`: BasicEvaluator aggression level 1-5 (default: 3)
- `max_depth`: Maximum depth for random walks (default: 50)
- `branch_probability`: Probability of branching during generation (default: 0.3)
- `scale`: Evaluation normalization scale factor (default: 0.001)
- `regenerate_each_epoch`: Whether to generate new positions each epoch (default: True)

### `generate_random_positions_branching`

- `num_positions`: Target number of unique positions (default: 1000)
- `aggression`: BasicEvaluator aggression level 1-5 (default: 3)
- `max_depth`: Maximum moves from start position (default: 50)
- `branch_probability`: Probability of creating branch points (default: 0.3)

## Advantages

1. **Efficiency**: No need to play complete games or reach terminal states
2. **Diversity**: Better exploration of position space through branching
3. **Flexibility**: Can target specific regions of state space
4. **Scalability**: Easy to generate millions of unique positions
5. **Fresh Data**: Can regenerate positions each epoch for better generalization

## Loss Function

The loss is the **mean squared error (MSE)** between:
- GNN predictions: `model(graph) → value ∈ [-1, 1]`
- Rust engine evaluations: `tanh(scale * BasicEvaluator.evaluate(board))`

This trains the GNN to match the analytical evaluation function.

## Backward Compatibility

The old `generate_eval_matching_data()` function is kept for backward compatibility but now internally calls the new branching approach with converted parameters.

## Implementation Details

### Position Deduplication

Uses `graph_hash()` from `graph_utils.py` which implements Weisfeiler-Lehman graph hashing for robust position equivalence checking.

### Branching Strategy

- **Branch probability**: Controls exploration width (default: 0.3)
- **Branch factor**: Limited to 2 additional branches per position to prevent explosion
- **Exploration**: Breadth-first-like using FIFO queue for branch points

### Evaluation Normalization

Raw scores from `BasicEvaluator` are normalized using:
```python
normalized = tanh(scale * raw_evaluation)
```
where `scale=0.001` maps typical evaluations to [-1, 1] range.

## Performance

Typical performance on modern hardware:
- **Position generation**: ~100-200 positions/second
- **Training throughput**: ~500-1000 positions/second (batch_size=64, GPU)
- **Memory usage**: O(num_positions) for dataset storage

## Future Improvements

1. **Adaptive branching**: Adjust probability based on position complexity
2. **Targeted generation**: Focus on specific position types (early/mid/endgame)
3. **Importance sampling**: Weight positions by learning value
4. **Parallel generation**: Multi-threaded position generation
5. **Curriculum learning**: Start with simple positions, increase complexity
