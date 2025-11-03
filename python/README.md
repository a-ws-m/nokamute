# Nokamute Python Bindings & Self-Play Learning

This directory contains Python bindings for the Nokamute Hive AI engine and a self-play learning system using Graph Neural Networks (GNN).

## Installation

### Prerequisites

- Python 3.8 or higher
- Rust toolchain (install from [rustup.rs](https://rustup.rs))
- PyTorch and PyTorch Geometric

### Build and Install

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Build and install the Rust bindings:
   ```bash
   maturin develop --release
   ```

   For development with debug symbols:
   ```bash
   maturin develop
   ```

## Usage

### Basic Board Interaction

```python
import nokamute

# Create a new board
board = nokamute.Board()

# Get legal moves
moves = board.legal_moves()

# Apply a move
board.apply(moves[0])

# Check game state
winner = board.get_winner()
if winner:
    print(f"Game over! Winner: {winner}")
```

### Self-Play Training

Run the self-play training loop:

```bash
python train.py --games 1000 --epochs 10 --batch-size 32
```

Options:
- `--games`: Number of self-play games to generate per iteration
- `--epochs`: Number of training epochs per iteration
- `--batch-size`: Batch size for training
- `--iterations`: Number of self-play + training iterations
- `--model-path`: Path to save/load model checkpoints

### Evaluate the Model

```bash
python evaluate.py --model checkpoints/model_latest.pt --games 100
```

## Architecture

### Rust Bindings

The `nokamute` Python module provides:
- `Board`: Game state management
- `Turn`: Move representation (place, move, pass)
- `Bug`: Bug type enum
- `Color`: Player color enum

Key methods:
- `board.legal_moves()`: Generate all legal moves
- `board.apply(turn)`: Apply a move
- `board.undo(turn)`: Undo a move
- `board.to_graph()`: Convert board to graph representation
- `board.get_pieces()`: Get all pieces on the board

### GNN Model

The model uses Graph Neural Networks to evaluate board positions:
- Node features: color, bug type, height, stacking status
- Edges: adjacency between hexagonal tiles
- Architecture: GAT (Graph Attention Network) layers + MLP head
- Output: Position evaluation score

### Self-Play Loop

1. Generate games using MCTS + current model for position evaluation
2. Collect (board state, move probabilities, game outcome) tuples
3. Train GNN on collected data
4. Evaluate new model vs old model
5. Keep better model and repeat

## Files

- `train.py`: Main self-play training script
- `model.py`: GNN model architecture
- `self_play.py`: Self-play game generation
- `evaluate.py`: Model evaluation utilities
- `utils.py`: Helper functions
