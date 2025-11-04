# New ML Training Features

This document describes three new features added to the Nokamute ML training system:

## 1. ELO Rating System

An ELO tracking system that monitors the performance of ML models over time.

### Features

- **Standard ELO calculation**: Uses the traditional ELO formula with configurable K-factor
- **Persistent history**: Saves all ratings and game results to JSON
- **Leaderboard**: Track rankings of models and opponents
- **Statistics**: Detailed win/loss/draw records for each player

### Usage

```python
from elo_tracker import EloTracker

# Initialize tracker
tracker = EloTracker(save_path="checkpoints/elo_history.json", k_factor=32)

# Update ratings after a game
# score: 1.0 = player_a wins, 0.5 = draw, 0.0 = player_b wins
new_rating_a, new_rating_b = tracker.update_ratings(
    player_a="model_iter_5",
    player_b="engine_depth_3",
    score_a=1.0,  # model won
    game_metadata={"num_moves": 42}
)

# Get current rating
rating = tracker.get_rating("model_iter_5")

# Get best model
best = tracker.get_best_model(prefix="model_iter_")

# View leaderboard
leaderboard = tracker.get_leaderboard(top_n=10)

# Save history
tracker.save()
```

### Default Ratings

The system initializes with baseline ratings:
- `random`: 800 ELO
- `engine_depth_1`: 1200 ELO
- `engine_depth_2`: 1400 ELO
- `engine_depth_3`: 1600 ELO
- `engine_depth_4`: 1800 ELO
- `engine_depth_5`: 2000 ELO

New models start at 1500 ELO by default.

## 2. Rust Engine Move Selection

A Python binding to get the best move according to the Nokamute Rust engine.

### API

```python
import nokamute

board = nokamute.Board()

# Get best move with depth-limited search
engine_move = board.get_engine_move(
    depth=3,              # Search depth (default: 3)
    time_limit_ms=None,   # Time limit in ms (overrides depth)
    aggression=3          # Aggression level 1-5 (default: 3)
)

# With time limit
engine_move = board.get_engine_move(time_limit_ms=5000)

# More aggressive play
engine_move = board.get_engine_move(depth=4, aggression=5)
```

### Parameters

- **depth** (int): Minimax search depth. Higher = stronger but slower.
- **time_limit_ms** (int, optional): Time limit in milliseconds. If set, overrides depth.
- **aggression** (int): Aggression level from 1-5:
  - 1: Very defensive, values opponent mobility highly
  - 3: Balanced (default)
  - 5: Very aggressive, focuses on attacking queen

### Returns

- `Turn` object with the best move, or `None` if game is over

## 3. Model vs Engine Evaluation

Comprehensive evaluation system for testing ML models against the Rust engine.

### Features

- **Multi-game evaluation**: Play multiple games at different depths
- **Balanced color play**: Automatically plays half games as each color
- **ELO integration**: Updates ratings based on results
- **Detailed statistics**: Win/loss/draw rates, average game length, time per game

### Usage

#### Simple Evaluation

```python
from evaluate_vs_engine import evaluate_model
from model import create_model

model = create_model()

results = evaluate_model(
    model=model,
    engine_depth=3,
    num_games=20,
    device="cuda",
    verbose=True
)

print(f"Win rate: {results['win_rate']:.1%}")
print(f"Record: {results['wins']}W-{results['losses']}L-{results['draws']}D")
```

#### Evaluation with ELO Updates

```python
from evaluate_vs_engine import evaluate_and_update_elo
from elo_tracker import EloTracker

tracker = EloTracker()

# Evaluate against multiple depths and update ELO
results = evaluate_and_update_elo(
    model=model,
    model_name="model_iter_10",
    elo_tracker=tracker,
    engine_depths=[2, 3, 4],
    games_per_depth=20,
    device="cuda",
    verbose=True
)

# Check updated rating
print(f"New ELO: {tracker.get_rating('model_iter_10'):.1f}")
```

### Components

#### EngineOpponent

```python
from evaluate_vs_engine import EngineOpponent

# Create engine opponent
engine = EngineOpponent(depth=3, aggression=3)

# Select a move
move = engine.select_move(board)
```

#### MLPlayer

```python
from evaluate_vs_engine import MLPlayer

# Create ML player
player = MLPlayer(model, temperature=0.1, device="cuda")

# Select a move
move = player.select_move(board)
```

## Integration with Training

All three features are integrated into the main training script:

```bash
python train.py \
    --iterations 50 \
    --eval-interval 5 \
    --eval-depths 2 3 4 \
    --eval-games 20 \
    --elo-k-factor 32
```

### New Training Arguments

- `--eval-interval N`: Evaluate against engine every N iterations (default: 5)
- `--eval-games N`: Number of games per evaluation depth (default: 20)
- `--eval-depths D1 D2 ...`: Engine depths to evaluate against (default: [3])
- `--elo-k-factor K`: K-factor for ELO rating system (default: 32)

### What Happens During Training

1. **Self-play generation**: Model plays against itself to generate training data
2. **Model training**: GNN is trained on self-play positions
3. **Self-play evaluation**: Model plays against itself for basic stats
4. **Engine evaluation** (periodic): 
   - Model plays against Rust engine at specified depths
   - ELO ratings are updated based on results
   - Statistics logged to TensorBoard
5. **Checkpoint saving**: Model and ELO history saved

### TensorBoard Logs

New metrics logged:
- `ELO/model_rating`: Current ELO rating
- `EngineEval/depth_N_win_rate`: Win rate vs depth N
- `EngineEval/depth_N_avg_moves`: Average game length vs depth N

View with:
```bash
tensorboard --logdir checkpoints/logs
```

## File Structure

```
python/
├── elo_tracker.py          # ELO rating system
├── evaluate_vs_engine.py   # Model vs engine evaluation
├── model.py                # GNN model (updated node features)
├── train.py                # Training script (with ELO integration)
├── self_play.py            # Self-play game generation
├── demo_features.py        # Demo of new features
└── README_NEW_FEATURES.md  # This file
```

## Example Workflow

```python
# 1. Create model and tracker
from model import create_model
from elo_tracker import EloTracker

model = create_model()
tracker = EloTracker(save_path="checkpoints/elo_history.json")

# 2. Train with self-play
# ... training code ...

# 3. Evaluate against engine
from evaluate_vs_engine import evaluate_and_update_elo

results = evaluate_and_update_elo(
    model=model,
    model_name="model_iter_10",
    elo_tracker=tracker,
    engine_depths=[3, 4],
    games_per_depth=20
)

# 4. Check progress
print(f"Current ELO: {tracker.get_rating('model_iter_10'):.1f}")
best_model = tracker.get_best_model()
print(f"Best model so far: {best_model}")

# 5. Save everything
tracker.save()
```

## Demo

Run the demo to see all features in action:

```bash
cd python
python demo_features.py
```

This will demonstrate:
1. Engine move selection with different depths and time limits
2. ELO tracking with simulated games
3. Model evaluation against the engine
4. Full integration of ELO + evaluation

## Performance Notes

- **Engine depth**: Depth 3-4 is reasonable for training evaluation. Depth 5+ can be very slow.
- **Evaluation frequency**: Every 5-10 iterations is a good balance between feedback and training speed.
- **Games per depth**: 20 games provides stable estimates while keeping evaluation time reasonable.
- **GPU usage**: Use `--device cuda` for faster model inference during evaluation.

## Troubleshooting

### "Import nokamute could not be resolved"

Make sure the Rust Python bindings are built:
```bash
cd python
pip install maturin
maturin develop --release
```

### Slow evaluation

- Reduce `--eval-games`
- Use lower `--eval-depths`
- Increase `--eval-interval`
- Enable GPU with `--device cuda`

### ELO history not persisting

Check that the checkpoint directory exists and is writable:
```bash
mkdir -p checkpoints
```

## Future Enhancements

Potential improvements:
- [ ] Parallel game evaluation for faster results
- [ ] Move-level policy evaluation (not just value)
- [ ] Opening book integration
- [ ] Tournament-style evaluation (round-robin)
- [ ] Automatic difficulty adjustment based on win rate
