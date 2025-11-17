# Quick Start Guide

## Installation

### Option 1: Automated Setup (Linux/macOS)

```bash
cd python
./setup.sh
```

This will:
- Create a Python virtual environment
- Install all dependencies
- Build the Rust bindings
- Test the installation

### Option 2: Manual Setup

```bash
cd python

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Build Rust bindings
maturin develop --release
```

## Usage Examples

### 1. Basic Board Interaction

```python
import nokamute

# Create a board
board = nokamute.Board()

# Get legal moves
moves = board.legal_moves()
print(f"Legal moves: {len(moves)}")

# Apply a move
board.apply(moves[0])

# Check game status
winner = board.get_winner()
if winner:
    print(f"Winner: {winner}")
```

### 2. Graph Conversion

```python
import nokamute

board = nokamute.Board()

# Play a few moves
for move in board.legal_moves()[:5]:
    board.apply(move)

# Convert to graph
node_features, edge_index = board.to_graph()
print(f"Nodes: {len(node_features)}, Edges: {len(edge_index[0])}")
```

### 3. Run Example

```bash
python example.py
```

## Training

### Quick Training Run

```bash
python train.py --games 50 --iterations 5 --epochs 5
```

### Full Training Run

```bash
python train.py \
  --games 1000 \
  --iterations 50 \
  --epochs 10 \
  --batch-size 32 \
  --lr 0.001 \
  --hidden-dim 128 \
  --num-layers 4 \
  --model-path checkpoints
```

Training parameters:
- `--games`: Number of self-play games per iteration (higher = more data but slower)
- `--iterations`: Number of training iterations (epochs of self-play + training)
- `--epochs`: Training epochs per iteration
- `--batch-size`: Batch size for training
- `--lr`: Learning rate
- `--hidden-dim`: Hidden dimension of GNN layers
- `--num-layers`: Number of GNN layers
- `--temperature`: Temperature for move selection (1.0 = random, 0.0 = greedy)

### Monitor Training

Training logs are saved in TensorBoard format:

```bash
tensorboard --logdir checkpoints/logs
```

## Evaluation

### Evaluate Against Random Play

```bash
python evaluate.py \
  --model checkpoints/model_latest.pt \
  --mode vs-random \
  --games 100
```

### Play Interactively

```bash
python evaluate.py \
  --model checkpoints/model_latest.pt \
  --mode interactive
```

## Project Structure

```
python/
├── model_policy_hetero.py           # GNN model architecture (heterogeneous)
├── self_play.py       # Self-play game generation
├── train.py           # Main training script
├── evaluate.py        # Evaluation utilities
├── utils.py           # Helper functions
├── example.py         # Basic usage example
├── requirements.txt   # Python dependencies
├── pyproject.toml     # Maturin build configuration
└── setup.sh          # Automated setup script
```

## Tips

### Development Build

For faster iteration during development:

```bash
maturin develop  # Without --release flag
```

### GPU Training

If you have CUDA available:

```bash
python train.py --device cuda --games 1000 --iterations 50
```

### Resume Training

```bash
python train.py \
  --resume checkpoints/model_iter_10.pt \
  --iterations 50
```

### Adjusting Model Size

For faster training (smaller model):

```bash
python train.py --hidden-dim 64 --num-layers 2
```

For better performance (larger model):

```bash
python train.py --hidden-dim 256 --num-layers 6
```

## Troubleshooting

### Import Error: nokamute module not found

Make sure you've built the Rust bindings:

```bash
maturin develop --release
```

### PyTorch Geometric Installation Issues

On some systems, you may need to install PyTorch Geometric dependencies manually:

```bash
pip install torch
pip install torch-geometric
```

Or follow the [PyTorch Geometric installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html).

### Memory Issues During Training

Reduce batch size or number of games:

```bash
python train.py --games 50 --batch-size 16
```

## Next Steps

1. Run the example to verify installation: `python example.py`
2. Start with a small training run: `python train.py --games 50 --iterations 5`
3. Monitor training with TensorBoard
4. Evaluate your model against random play
5. Experiment with different architectures and hyperparameters

For more details, see `README.md` in this directory.
