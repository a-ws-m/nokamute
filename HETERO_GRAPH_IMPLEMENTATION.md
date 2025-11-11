# Heterogeneous Graph Implementation Summary

## Overview
Successfully implemented a heterogeneous graph representation for the Hive board game, as specified in `policy_model_change.md`.

## Changes Made

### 1. Rust Implementation (`src/python.rs`)

#### Graph Structure
The `to_graph()` method now returns a heterogeneous graph with:

**Node Types:**
- `in_play`: Pieces currently on the board
  - Features: [color (0/1), bug_onehot (9 values)]
- `out_of_play`: Pieces not yet placed 
  - Features: [color (0/1), bug_onehot (9 values)]
- `destination`: Empty spaces where pieces can be placed/moved
  - Features: [on_top (0/1), padding (9 zeros)]

**Edge Types:**
- `neighbour`: Adjacency/stacking relationships
  - Connects: in_play↔in_play, in_play↔destination, destination↔destination
  - Features: [0.0] (no meaningful features)
- `move`: Legal moves for both players
  - Connects: in_play→destination, out_of_play→destination
  - Features: [1.0] for current player, [0.0] for opponent

#### Output Format
Returns a Python dictionary with separate variables for each node/edge type:
```python
{
    'x_in_play': [[features], ...],
    'x_out_of_play': [[features], ...],
    'x_destination': [[features], ...],
    'edge_index_in_play_neighbour_in_play': [[from_indices], [to_indices]],
    'edge_index_in_play_move_destination': [[from_indices], [to_indices]],
    # ... more edge types ...
    'edge_attr_in_play_neighbour_in_play': [[attr], ...],
    'edge_attr_in_play_move_destination': [[attr], ...],
    # ... more edge attributes ...
    'move_to_action': ['move_str1', 'move_str2', ...]  # Current player's legal moves
}
```

### 2. Python Helper (`python/hetero_graph_utils.py`)

Created `board_to_hetero_data()` function that converts the Rust dictionary format to PyTorch Geometric's `HeteroData` format:
- Parses node type keys (`x_{node_type}`)
- Parses edge type keys (`edge_index_{src}_{edge}_{dst}`)
- Converts move strings to action space indices
- Returns `(HeteroData, move_to_action_indices)` tuple

### 3. Model Architecture Changes

**Removed Features:**
- Height feature from all nodes
- Current player feature from all nodes  
- Global pooling for action prediction

**New Architecture (to be completed):**
- Use `HeteroConv` with `GATv2Conv` layers
- Increased default layers from 4 to 6
- Move edge features directly produce action logits
- Action vector assembled by reading move edge features
- Move-to-action mapping used to map edge logits to action space indices

### 4. Minor Changes

Added `Hash` trait to `Bug` and `Color` enums in Rust to support HashMap usage.

## Testing

Created test scripts:
- `test_heterogeneous_graph.py`: Tests Rust graph generation
- `hetero_graph_utils.py`: Tests Python conversion utilities

All tests pass successfully, including:
- Empty board case (1 destination node with 7 legal first moves)
- After first move (1 in_play, 7 destinations, 90 move edges)
- After 6 moves (4 in_play, 14 destinations, 65 move edges)
- Conversion to PyTorch Geometric HeteroData format

## Next Steps

1. Complete the heterogeneous GNN model implementation in `model_policy_hetero.py`
2. Update `self_play.py` to use the new graph format
3. Update training scripts to work with the new model
4. Test with league training system

## Benefits

1. **More expressive representation**: Explicitly models different types of nodes and edges
2. **Legal moves for both players**: Enables better strategic reasoning
3. **Efficient action prediction**: No pooling bottleneck, direct edge-to-action mapping
4. **Cleaner separation**: Move edges are separate from neighbor edges
5. **PyTorch Geometric compatible**: Standard format for heterogeneous GNNs
