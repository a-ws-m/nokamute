# League Training - Quick Reference

## Installation & Setup

```bash
cd python
source .venv/bin/activate
```

## Quick Commands

### Run Tests
```bash
python test_league.py
```

### Basic Training
```bash
python train_league.py --iterations 1000 --device cuda
```

### With Pretrained Model
```bash
python train_league.py \
    --pretrained-model checkpoints/model_pretrained.pt \
    --iterations 1000 \
    --device cuda
```

### Resume Training
```bash
python train_league.py --resume league_checkpoints --iterations 2000
```

### Monitor Training
```bash
tensorboard --logdir league_checkpoints/logs
```

## Key Parameters

```bash
--iterations 1000          # Total training iterations
--main-games 10            # Games per Main Agent update
--exploiter-games 20       # Games per Exploiter update
--hidden-dim 128           # Model hidden dimension
--num-layers 4             # GNN layers
--eval-interval 50         # Eval vs engine every N iters
--eval-depths 2 3          # Engine depths to test
--device cuda              # cuda or cpu
--save-dir <path>          # Where to save models
```

## Configuration

Edit `league/config.py` to adjust:
- PFSP sampling parameters
- Minimax reward weight
- Convergence thresholds
- Spawning schedules

## Key Metrics (TensorBoard)

- `agents/{name}/win_rate` - Agent performance
- `exploiters/{name}/win_rate_vs_target` - Convergence
- `league/num_converged_exploiters` - Progress
- `pfsp/opponent_selection/*` - PFSP behavior

## File Structure

```
league/
├── config.py      # Hyperparameters
├── manager.py     # League lifecycle
├── exploiter.py   # Minimax reward
└── tracker.py     # Metrics & logging

train_league.py    # Main script
test_league.py     # Tests
```

## Documentation

- `LEAGUE_TRAINING.md` - Complete guide
- `LEAGUE_IMPLEMENTATION_SUMMARY.md` - Implementation details
- Paper: AAMAS_smartexploiter.tex

## Troubleshooting

**Exploiters not converging?**
- Increase `exploiter_games_per_iter`
- Adjust `minimax_reward_weight`

**Main Agent not improving?**
- Check PFSP opponent distribution
- Increase `main_agent_games_per_iter`

**Out of memory?**
- Reduce batch sizes
- Use smaller `hidden_dim`
- Disable branching MCMC

## Help

```bash
python train_league.py --help
```
