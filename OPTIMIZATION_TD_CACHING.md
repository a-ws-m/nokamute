# TD Learning Optimization: Caching Move Evaluations

## Problem

The previous implementation was evaluating positions twice during TD learning:

1. **During self-play** (`select_move`): The model evaluated all legal moves to compute move probabilities for selection
2. **During training data preparation** (`prepare_td_training_data`): The model re-evaluated the same positions to compute TD targets

This caused significant redundant computation, especially since:
- Self-play already computes the exact values we need for TD targets
- Each position evaluation requires a forward pass through the GNN
- With branching MCMC, many positions are shared across games

## Solution

Store the model's evaluation of the selected move during self-play and reuse it during TD target computation.

### Changes Made

#### 1. `self_play.py` - Enhanced `select_move` method

**Before:**
```python
def select_move(self, board, legal_moves, return_probs=False):
    # ... evaluate moves, compute probabilities, select move
    return selected_move  # or (selected_move, move_probs)
```

**After:**
```python
def select_move(self, board, legal_moves, return_probs=False, return_value=False):
    # ... evaluate moves ONCE, compute probabilities, select move
    # Optionally return the selected move's evaluation
    return selected_move  # or (selected_move, move_probs, move_value)
```

**Key improvements:**
- Added `return_value` parameter to optionally return the evaluated value
- Evaluates moves only once and reuses results for both probabilities and value return
- Updated `_compute_move_probabilities` to accept pre-computed `move_values` to avoid redundant evaluation

#### 2. `self_play.py` - Updated game data format

**Before:**
```python
game_data.append((nx_graph, legal_moves, selected_move, current_player, pos_hash))
```

**After:**
```python
game_data.append((nx_graph, legal_moves, selected_move, current_player, pos_hash, move_value))
```

Where `move_value` is the model's evaluation of the position **after** applying the selected move (from the opponent's perspective).

#### 3. `self_play.py` - Modified `play_game` method

Now requests both probabilities and move value during branching mode, and just the move value during standard mode:

```python
# Branching mode
selected_move, move_probs, move_value = self.select_move(
    board, legal_moves, return_probs=True, return_value=True
)

# Standard mode
selected_move, move_value = self.select_move(
    board, legal_moves, return_value=True
)
```

#### 4. `train.py` - Simplified `prepare_td_training_data`

**Before:**
- Phase 1: Collect all unique positions
- Phase 2: **Batch evaluate all positions** (expensive!)
- Phase 3: Compute TD targets using evaluations
- Phase 4: Average targets for duplicate positions

**After:**
- Phase 1: Extract stored move values from game data (cheap!)
- Phase 2: Compute TD targets using stored values (no re-evaluation!)
- Phase 3: Average targets for duplicate positions

**Removed code:**
```python
# No longer needed - we use stored move values instead!
for i in range(0, len(data_list), batch_size):
    batch_data = data_list[i:i + batch_size]
    batch = Batch.from_data_list(batch_data).to(device)
    with torch.no_grad():
        predictions, _ = model(batch.x, batch.edge_index, batch.batch)
        # ... process predictions
```

**New code:**
```python
# Simply extract the stored move_value from game data
position_move_values[(game_idx, position_idx)] = move_value

# Later, use it directly for TD target
next_move_value = position_move_values.get((game_idx, position_idx))
if next_move_value is not None:
    next_value = -next_move_value  # Flip sign for perspective
    td_target = gamma * next_value
```

#### 5. Backward compatibility

Both `prepare_training_data` and `prepare_td_training_data` support the old 5-element format for backward compatibility:

```python
if len(item) == 6:
    # New format with move_value
    nx_graph, legal_moves, selected_move, player, pos_hash, move_value = item
elif len(item) == 5:
    # Old format without move_value (fallback)
    nx_graph, legal_moves, selected_move, player, pos_hash = item
    move_value = None  # or 0.0
```

## Performance Impact

### Computational savings

For each training iteration with N games and M average positions per game:

**Before:**
- Self-play: N × M forward passes (one per position)
- TD preparation: N × M forward passes (re-evaluate all positions)
- **Total: 2 × N × M forward passes**

**After:**
- Self-play: N × M forward passes (one per position)
- TD preparation: 0 forward passes (use cached values)
- **Total: N × M forward passes**

**Result: 50% reduction in forward passes during TD learning!**

### Memory impact

Minimal - only one additional float per position in game data:
- Before: ~5 references per position
- After: ~6 references per position (5 + 1 float)

For typical game lengths (~50-100 moves), this adds ~200-400 bytes per game.

## Correctness

The stored `move_value` is exactly what we need for TD learning:

1. During self-play at position `s_t`, we evaluate the position `s_{t+1}` that results from the selected move
2. This evaluation is `V(s_{t+1})` from the opponent's perspective (since they play next)
3. For TD target computation, we need this same `V(s_{t+1})` value
4. We negate it to get it from current player's perspective: `-V(s_{t+1})`
5. TD target: `gamma * (-V(s_{t+1}))`

This is mathematically identical to the previous approach, just without redundant computation.

## Testing

Run the existing training script with TD learning enabled:

```bash
python python/train.py --use-td --games 1 --epochs 1 --iterations 5
```

Expected output should now show:
```
Using X stored move evaluations (no re-evaluation needed)
```

Instead of:
```
Evaluating X unique positions (from Y total)...
```

## Future Improvements

1. **Store all move values**: Currently we only store the selected move's value. We could store values for all legal moves to enable richer analysis or multi-step lookahead.

2. **Value interpolation**: For positions that appear multiple times with different outcomes, we could use more sophisticated aggregation than simple averaging.

3. **Selective re-evaluation**: For critical positions (e.g., near endgame), we might want to re-evaluate with higher precision even if we have cached values.
