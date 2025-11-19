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
    probs = model(data)
    # If model returns a list for batched inputs, unpack for single graph
    if isinstance(probs, list):
        assert len(probs) == 1
        probs = probs[0]

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
    out = model(batch)
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
