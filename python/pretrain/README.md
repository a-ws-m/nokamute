# Pre-training Modules

This directory contains pre-training methods that can be used to initialize the GNN model before self-play training. Pre-training gives the model a better starting point and can significantly speed up convergence.

## Available Methods

### 1. Evaluation Matching (`eval_matching.py`)

Pre-trains the model to match the analytical evaluation function from the Rust engine's `BasicEvaluator`.

**How it works:**
1. The Rust engine plays games against itself using iterative deepening minimax (default depth: 7)
2. **Efficiently captures ALL positions explored during search** - not just root positions
   - For a depth-7 search, this captures thousands of positions per move
   - Dramatically increases training data without additional computation
3. Random moves are occasionally injected to create diverse positions (default: 15% randomness)
4. **Automatic deduplication** using zobrist hashing - each unique position stored only once
5. For each unique position, we record the board state and the analytical evaluation score
6. The GNN is trained to predict these evaluation scores using MSE loss
7. Scores are normalized to [-1, 1] range using tanh scaling

**Benefits:**
- Teaches the model basic positional understanding (mobility, queen safety, piece value, etc.)
- Provides a strong foundation before self-play begins
- Can significantly reduce the number of self-play iterations needed
- **Extremely data-efficient**: A single depth-7 search can yield thousands of training positions
- **Automatic deduplication**: Each unique position (by zobrist hash) stored only once
- **No wasted computation**: Captures all positions already evaluated by the engine

**Usage:**

Pre-training is a standalone mode - it runs to completion and then exits. After pre-training, you can resume with self-play training using the generated checkpoint.

```bash
# Step 1: Pre-train the model
python train.py --pretrain eval-matching \
    --pretrain-games 1000 \
    --pretrain-depth 7 \
    --pretrain-epochs 50

# Step 2: Continue with self-play training
python train.py --resume checkpoints/model_pretrained.pt \
    --iterations 100 \
    --games 100

# Use custom data file path
python train.py --pretrain eval-matching \
    --pretrain-data-path my_custom_data.pkl \
    --pretrain-epochs 100

# Data is automatically saved and reused on subsequent runs with same settings
# File is named: pretrain_eval_d{depth}_g{games}_r{randomness}.pkl
```

**Arguments:**
- `--pretrain eval-matching`: Enable evaluation matching pre-training (standalone mode)
- `--pretrain-games N`: Number of games to generate (default: 1000)
- `--pretrain-depth N`: Engine search depth for game generation (default: 7)
- `--pretrain-epochs N`: Number of training epochs (default: 50)
- `--pretrain-randomness R`: Probability of random moves (default: 0.15)
- `--pretrain-data-path PATH`: Custom path for data file (auto-generated if not provided)

**Data Caching:**
Generated pre-training data is automatically saved to disk with a filename based on the generation parameters. On subsequent runs with the same parameters, the data is loaded from disk instead of being regenerated. This saves significant time when experimenting with different epoch counts or other training parameters.

**Expected Data Sizes:**
With the improved data collection that captures all positions from the search tree:
- **100 games at depth 5**: ~50,000-100,000 unique positions
- **1000 games at depth 7**: ~500,000-1,000,000 unique positions
- **File sizes**: Typically 50-500 MB depending on game count and depth

The actual number varies based on:
- Search depth (exponential growth)
- Randomness rate (more randomness = more unique positions)
- Game length (longer games explore more of the tree)

**Standalone usage:**

```python
from pretrain.eval_matching import (
    generate_eval_matching_data,
    pretrain_eval_matching,
    save_eval_data,
    load_eval_data,
    normalize_evaluation
)
from model import create_model
import torch

# Generate data
data = generate_eval_matching_data(
    num_games=1000,
    depth=7,
    aggression=3,
    randomness_rate=0.15,
    verbose=True
)

# Save for later use
save_eval_data(data, "my_pretrain_data.pkl")

# Create model
model = create_model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Pre-train
losses = pretrain_eval_matching(
    model=model,
    training_data=data,
    optimizer=optimizer,
    num_epochs=50,
    batch_size=64,
    device="cuda"
)

# Normalization uses torch.tanh for smooth scaling
# normalize_evaluation(score, scale=0.001) maps raw scores to [-1, 1]
```

## Adding New Pre-training Methods

To add a new pre-training method:

1. Create a new file in this directory (e.g., `my_method.py`)
2. Implement data generation and training functions
3. Add imports to `__init__.py`
4. Add the method to `train.py`'s `--pretrain` choices
5. Add conditional logic in `train.py` to handle the new method
6. Document it in this README

**Template:**

```python
"""
My new pre-training method.
"""

def generate_my_data(num_samples, **kwargs):
    """
    Generate training data.
    
    Returns:
        List of (node_features, edge_index, target) tuples
    """
    # Your data generation logic
    pass

def pretrain_my_method(model, training_data, optimizer, **kwargs):
    """
    Pre-train the model.
    
    Returns:
        List of losses per epoch
    """
    # Your training logic
    pass
```

## Design Philosophy

Pre-training modules should:
- Be **optional** - training should work without them
- Be **self-contained** - each method in its own file
- **Generate diverse data** - use randomness or exploration
- **Match the target domain** - positions should be realistic
- **Be efficient** - leverage Rust engine for fast data generation
- **Be reproducible** - support saving/loading data

## Performance Tips

1. **Data Generation:**
   - **Depth matters most**: Depth 7 yields 10-100x more positions than depth 3
   - Fewer games at higher depth is more efficient than many games at low depth
   - Add randomness (10-20%) to increase diversity and prevent over-representation of main lines
   - The deduplication means you get diminishing returns from more games
   - Start with 100-200 games at depth 7, which can yield 100k+ unique positions

2. **Training:**
   - Start with lower learning rate (1e-4) for pre-training
   - Use larger batch sizes (64-128) for stability
   - Monitor loss curves - should converge smoothly
   - 50-100 epochs is usually sufficient

3. **Evaluation:**
   - Test pre-trained model against engine before self-play
   - Compare with untrained baseline
   - Track ELO ratings throughout training
   - A good pre-trained model should score 30-40% against depth 3 engine

## Future Ideas

Potential pre-training methods to implement:

- **Endgame database**: Train on known won/lost/drawn positions
- **Opening book**: Learn common opening patterns
- **Tactical puzzles**: Mate-in-N and winning sequences
- **Position classification**: Categorize positions (opening/midgame/endgame)
- **Move prediction**: Predict engine's move from position
- **Value + policy co-training**: Train both heads simultaneously

## References

The evaluation matching approach is inspired by:
- AlphaGo's supervised learning phase (learning from human games)
- Leela Chess Zero's knowledge distillation from Stockfish
- The general principle of imitation learning / behavioral cloning
