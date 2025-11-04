# Quick Reference: New ML Features

## 1. Get Best Move from Engine

```python
import nokamute

board = nokamute.Board()

# Basic usage (depth 3, balanced aggression)
move = board.get_engine_move()

# Custom depth
move = board.get_engine_move(depth=4)

# Time limit (overrides depth)
move = board.get_engine_move(time_limit_ms=5000)

# Aggressive play
move = board.get_engine_move(depth=3, aggression=5)

# Defensive play
move = board.get_engine_move(depth=3, aggression=1)
```

## 2. Track ELO Ratings

```python
from elo_tracker import EloTracker

# Create tracker
tracker = EloTracker(save_path="elo.json", k_factor=32)

# Record game result
# score_a: 1.0=win, 0.5=draw, 0.0=loss
tracker.update_ratings("model_v1", "engine_depth_3", score_a=0.5)

# Get rating
rating = tracker.get_rating("model_v1")

# Find best model
best = tracker.get_best_model(prefix="model_")

# View leaderboard
for rank, (player, elo) in enumerate(tracker.get_leaderboard(10), 1):
    print(f"{rank}. {player}: {elo:.1f}")

# Save progress
tracker.save()
```

## 3. Evaluate Model vs Engine

```python
from model import create_model
from evaluate_vs_engine import evaluate_model

model = create_model()

# Simple evaluation
results = evaluate_model(
    model, 
    engine_depth=3, 
    num_games=20,
    device="cuda"
)

print(f"Win rate: {results['win_rate']:.1%}")
print(f"Record: {results['wins']}W-{results['losses']}L-{results['draws']}D")
```

## 4. Full Evaluation with ELO

```python
from evaluate_vs_engine import evaluate_and_update_elo
from elo_tracker import EloTracker

tracker = EloTracker()

# Evaluate and update ELO
results = evaluate_and_update_elo(
    model=model,
    model_name="my_model_v5",
    elo_tracker=tracker,
    engine_depths=[2, 3, 4],
    games_per_depth=20,
    device="cuda"
)

# Check new rating
elo = tracker.get_rating("my_model_v5")
print(f"New ELO: {elo:.1f}")
```

## 5. Training with Engine Evaluation

```bash
# Basic - evaluate every 5 iterations against depth 3
python train.py --iterations 50 --eval-interval 5

# Multiple depths
python train.py \
    --iterations 50 \
    --eval-interval 5 \
    --eval-depths 2 3 4

# More evaluation games
python train.py \
    --iterations 50 \
    --eval-interval 3 \
    --eval-games 30

# Custom ELO K-factor
python train.py \
    --iterations 50 \
    --elo-k-factor 24
```

## 6. View Training Progress

```bash
# TensorBoard
tensorboard --logdir checkpoints/logs

# Check ELO history
python -c "
from elo_tracker import EloTracker
t = EloTracker(save_path='checkpoints/elo_history.json')
t.load()
for player, elo in t.get_leaderboard(10):
    print(f'{player}: {elo:.1f}')
"
```

## Command Line Arguments

### New Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--eval-interval` | 5 | Evaluate vs engine every N iterations |
| `--eval-games` | 20 | Games per evaluation depth |
| `--eval-depths` | [3] | Engine depths to test (space-separated) |
| `--elo-k-factor` | 32 | ELO K-factor (higher = more volatile) |

### Example Configurations

**Fast iteration (fewer evaluations)**:
```bash
python train.py --eval-interval 10 --eval-games 10 --eval-depths 3
```

**Thorough evaluation**:
```bash
python train.py --eval-interval 3 --eval-games 30 --eval-depths 2 3 4 5
```

**Competition-ready**:
```bash
python train.py --eval-interval 5 --eval-games 50 --eval-depths 4 --elo-k-factor 24
```

## File Locations

```
python/
├── elo_tracker.py              # ELO system
├── evaluate_vs_engine.py       # Evaluation module
├── demo_features.py            # Demo script
└── README_NEW_FEATURES.md      # Full documentation

checkpoints/
├── elo_history.json            # ELO ratings and game history
├── model_iter_*.pt             # Model checkpoints
└── logs/                       # TensorBoard logs
```

## Quick Demo

```bash
cd python
python demo_features.py
```

Demonstrates:
1. Engine move selection
2. ELO tracking
3. Model evaluation
4. Full integration

## Common Patterns

### Training Loop with Periodic Evaluation

```python
from elo_tracker import EloTracker
from evaluate_vs_engine import evaluate_and_update_elo

tracker = EloTracker()

for iteration in range(num_iterations):
    # Train model...
    
    # Periodic evaluation
    if iteration % 5 == 0:
        evaluate_and_update_elo(
            model, f"model_iter_{iteration}", tracker,
            engine_depths=[3], games_per_depth=20
        )
        
        print(f"ELO: {tracker.get_rating(f'model_iter_{iteration}'):.1f}")
        tracker.save()
```

### Compare Two Models

```python
from evaluate_vs_engine import play_game, MLPlayer

player1 = MLPlayer(model1, temperature=0.1)
player2 = MLPlayer(model2, temperature=0.1)

wins = 0
for _ in range(20):
    result, _, _ = play_game(player1, player2)
    if result == "player1":
        wins += 1

print(f"Model 1 win rate: {wins/20:.1%}")
```

### Progressive Difficulty Training

```python
depths = [2, 3, 4, 5]
for iteration in range(num_iterations):
    # Train model...
    
    # Gradually increase difficulty
    current_depth = depths[min(iteration // 10, len(depths) - 1)]
    
    if iteration % 5 == 0:
        evaluate_and_update_elo(
            model, f"model_iter_{iteration}", tracker,
            engine_depths=[current_depth], games_per_depth=20
        )
```

## Tips

1. **Start with depth 2-3** for faster iterations
2. **Increase evaluation frequency** when model is learning fast
3. **Use GPU** for faster model inference during evaluation
4. **Monitor TensorBoard** for ELO trends over time
5. **Save regularly** - both model checkpoints and ELO history
6. **Compare against multiple depths** to understand model strength range
