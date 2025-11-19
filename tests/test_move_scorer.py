import torch

import nokamute
from nokamute import Board

try:
    from nokamute import MoveScorer  # if model is installed as part of package
    from nokamute import BoardHeteroBuilder
except Exception:
    import os
    import sys

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    from python.graph import BoardHeteroBuilder
    from python.hetero_models import MoveScorer


def test_move_scorer_on_start():
    board = Board()
    builder = BoardHeteroBuilder(board)
    data = builder.to_model_input()

    model = MoveScorer(1, hidden_dim=32)
    # The model will prepare its hetero encoder based on data in `forward`.
    probs, critic = model(data)
    # If model returns a list for batched inputs, unpack for single graph
    if isinstance(probs, list):
        assert len(probs) == 1
        probs = probs[0]
    # Critic should be a 1-element tensor for single graph
    assert critic is not None
    assert critic.numel() == 1

    # Starting position must have seven current-move edges (asserted in other tests)
    assert probs.numel() == 7
    # action scores should be within (-1, 1) from tanh activation
    assert probs.max().item() <= 1.0
    assert probs.min().item() >= -1.0
    # mapping from edges to moves should be present for the current_move edge type
    cur_rel = ("out_of_play_piece", "current_move", "destination")
    assert cur_rel in data.edge_index_dict
    # ensure there is a move_str list attached for that type and it aligns
    assert hasattr(data[cur_rel], "move_str")
    assert len(data[cur_rel].move_str) == 7


def test_move_scorer_batch_prediction():
    # Create two starting boards and batch them
    b1 = Board()
    b2 = Board()
    builder = BoardHeteroBuilder(b1)
    d1 = builder.to_model_input()
    builder2 = BoardHeteroBuilder(b2)
    d2 = builder2.to_model_input()

    from torch_geometric.loader import DataLoader

    loader = DataLoader([d1, d2], batch_size=2)
    batch = next(iter(loader))

    model = MoveScorer(1, hidden_dim=32)
    out, critic = model(batch)
    # We expect a list of results (one per graph)
    assert isinstance(out, list)
    assert len(out) == 2
    for i, p in enumerate(out):
        assert p.numel() == 7
        assert p.max().item() <= 1.0
        assert p.min().item() >= -1.0
        # ensure mapping exists for each graph
        cur_rel = ("out_of_play_piece", "current_move", "destination")
        assert cur_rel in batch.edge_index_dict
        assert hasattr(batch[cur_rel], "move_str")
        move_str = batch[cur_rel].move_str
        # Depending on how PyG collates, we may get a flat list (single graph)
        # or a list of lists for a batched tensor; support both.
        if isinstance(move_str[0], list):
            assert len(move_str) == 2
            assert all(len(m) == 7 for m in move_str)
        else:
            # Single flat list: ensure it has 7 entries
            assert len(move_str) == 7
    assert isinstance(critic, torch.Tensor)
    assert critic.numel() == 2


def test_action_scores_to_move_mapping_batch_varied():
    # Initial position
    b1 = Board()
    builder1 = BoardHeteroBuilder(b1)
    d1 = builder1.to_model_input()

    # Sequence: wG1; bG1 -wG1; then white moves
    s = "Base+M;InProgress;turn;" "wG1;" "bG1 -wG1"
    b2 = Board.from_game_string(s)
    builder2 = BoardHeteroBuilder(b2)
    d2 = builder2.to_model_input()

    from torch_geometric.loader import DataLoader

    batch = next(iter(DataLoader([d1, d2], batch_size=2)))

    model = MoveScorer(1, hidden_dim=32)
    action_scores, critic = model(batch)
    assert isinstance(action_scores, list)

    # Map scores to moves
    maps = model.action_scores_to_move_dicts(batch, action_scores)
    assert isinstance(maps, list)
    # First graph: 7 unique actions
    assert len(maps[0]) == 7
    # Second graph: white has 16 unique moves (some symmetric), check >= 16
    assert len(maps[1]) >= 16
    # Highest-scoring move is first
    for m in maps:
        if len(m) == 0:
            continue
        first_score = list(m.values())[0]
        for sc in m.values():
            assert first_score >= sc
    # Critic should be a 2-element tensor for this batch
    assert isinstance(critic, torch.Tensor)
    assert critic.numel() == 2
