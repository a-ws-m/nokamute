import sys

import nokamute

sys.path.insert(0, "./python")
from action_space import string_to_action
from hetero_graph_utils import board_to_hetero_data, ordered_unique_action_indices


def test_rust_move_order_present():
    b = nokamute.Board()
    graph_dict = b.to_graph()

    # We prefer the numeric move_order provided by Rust; fall back to strings
    # for backward compatibility with older Rust versions.
    assert (
        "move_order_idx" in graph_dict or "move_order" in graph_dict
    ), "Rust must include move_order or move_order_idx in graph dict"

    if "move_order_idx" in graph_dict:
        rust_move_order = list(graph_dict["move_order_idx"])
    else:
        rust_move_order = [string_to_action(m) for m in graph_dict["move_order"]]

    # Now compute the Python-side unique list from move_to_action indices (should match)
    data, move_to_action_indices = board_to_hetero_data(graph_dict)
    python_order = ordered_unique_action_indices(move_to_action_indices)

    # It should at least be non-empty and contain same set
    assert len(rust_move_order) > 0
    assert set(rust_move_order).issubset(
        set(python_order)
    ), "Rust move_order should match Pythonunique ordering"
