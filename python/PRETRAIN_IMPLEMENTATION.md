# Pre-training Implementation Summary

## Overview

I've implemented an optional pre-training system for the Hive GNN model that allows the model to learn from the analytical evaluation function (`BasicEvaluator`) before starting self-play training.

## Key Changes

### 1. New Pre-training Module (`python/pretrain/`)

Created a new package with the following structure:
- `__init__.py` - Package initialization
- `eval_matching.py` - Core pre-training implementation
- `README.md` - Comprehensive documentation
- `example_workflow.sh` - Example usage script

### 2. Rust Python Bindings (`src/python.rs`)

Added new method to the `Board` class:
```rust
fn get_evaluation(&self, aggression: Option<u8>) -> PyResult<i16>
```
This exposes the `BasicEvaluator::evaluate()` function to Python, allowing the GNN to learn from the analytical evaluation scores.

### 3. Training Script Updates (`python/train.py`)

**Pre-training as Standalone Mode:**
- Pre-training now runs independently and exits after completion
- After pre-training, use `--resume model_pretrained.pt` to continue with self-play
- This allows experimentation with different pre-training configurations

**Automatic Data Caching:**
- Generated training data is automatically saved to disk
- Filename based on generation parameters: `pretrain_eval_d{depth}_g{games}_r{randomness}.pkl`
- Subsequent runs with same parameters load from cache instead of regenerating
- Saves significant time when experimenting with training hyperparameters

**Default Parameters:**
- `--pretrain-games`: Increased from 200 to 1000 (more diverse positions)
- `--pretrain-epochs`: Increased from 20 to 50 (better convergence)
- Removed `--save-pretrain-data` flag (now automatic)

### 4. Normalization

**Changed from manual tanh to torch-based:**
```python
def normalize_evaluation(eval_score: float, scale: float = 0.001) -> float:
    import torch
    return torch.tanh(torch.tensor(eval_score * scale)).item()
```
- Uses PyTorch's `torch.tanh()` for consistency with the rest of the codebase
- Scale parameter (default 0.001) maps typical evaluation scores to [-1, 1] range
- Handles extreme values gracefully

## How It Works

### Data Generation
1. Rust engine plays games against itself using iterative deepening minimax
2. At each position, records:
   - Board state (node features + edge index for GNN)
   - Analytical evaluation score from `BasicEvaluator`
3. Random moves (15% by default) injected for diversity
4. Positions saved to disk for reuse

### Training
1. Model learns to predict normalized evaluation scores
2. MSE loss between GNN predictions and analytical evaluations
3. Multiple epochs over the dataset with shuffling
4. Progress tracked via TensorBoard

### Evaluation
1. Pre-trained model tested against engine at various depths
2. ELO rating calculated and tracked
3. Results saved to checkpoint file

## Usage Example

```bash
# Step 1: Pre-train (standalone mode)
python train.py --pretrain eval-matching

# Step 2: Continue with self-play
python train.py --resume checkpoints/model_pretrained.pt --iterations 100

# Re-run with more epochs (uses cached data)
python train.py --pretrain eval-matching --pretrain-epochs 100
```

## Benefits

1. **Better Starting Point:** Model learns basic positional concepts before self-play
2. **Faster Convergence:** Reduces self-play iterations needed
3. **Knowledge Transfer:** Incorporates decades of game AI research (evaluation functions)
4. **Reusable Data:** Cache expensive data generation, experiment with training
5. **Modular Design:** Easy to add new pre-training methods

## Future Extensions

The modular design makes it easy to add new pre-training methods:
- **Endgame databases:** Learn known won/lost positions
- **Opening books:** Common opening patterns
- **Tactical puzzles:** Mate-in-N sequences
- **Move prediction:** Predict engine's chosen move
- **Multi-task learning:** Train value + policy heads together

## File Changes

- **New files:**
  - `python/pretrain/__init__.py`
  - `python/pretrain/eval_matching.py`
  - `python/pretrain/README.md`
  - `python/pretrain/example_workflow.sh`

- **Modified files:**
  - `src/python.rs` - Added `get_evaluation()` method
  - `python/train.py` - Integrated pre-training as standalone mode

## Testing

To test the implementation:

```bash
# Quick test with small dataset
python pretrain/eval_matching.py

# Full pre-training test
python train.py --pretrain eval-matching --pretrain-games 100 --pretrain-epochs 10
```

The implementation is ready to use and can significantly improve the initial performance of the GNN model before self-play training begins.
