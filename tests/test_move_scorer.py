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

    # Starting position must have seven current-move edges (asserted in other tests)
    assert probs.numel() == 7
    # probabilities should sum to 1
    assert abs(probs.sum().item() - 1.0) < 1e-6
