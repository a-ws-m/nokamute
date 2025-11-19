import nokamute
from nokamute import Board, Bug, Turn

try:
    from nokamute import BoardHeteroBuilder
except Exception:
    # Fall back to local python module for builder when running tests from the workspace
    # without installation. Insert project root into sys.path and import python.graph.
    import os
    import sys

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    from python.graph import BoardHeteroBuilder


def test_basic_graph_conversion():
    board = Board()

    # There should be at least a few place moves at the start.
    moves = board.legal_moves()
    place_move = next((m for m in moves if m.is_place()), None)
    assert place_move is not None, "No place move found"

    board.apply(place_move)
    builder = BoardHeteroBuilder(board)
    data = builder.to_heterodata()

    # Basic checks on node counts
    assert data["in_play_piece"].num_nodes >= 1
    assert data["destination"].num_nodes >= 6

    # We should have adjacency edges from the in-play node to the destination
    assert ("in_play_piece", "adj", "destination") in data.edge_index_dict

    # Check that current_move edges exist (some legal moves for the player after the placement)
    assert ("in_play_piece", "current_move", "destination") in data.edge_index_dict or (
        "out_of_play_piece",
        "current_move",
        "destination",
    ) in data.edge_index_dict


def test_starting_position_counts():
    board = Board()
    builder = BoardHeteroBuilder(board)
    data = builder.to_heterodata()

    # Single destination node for starting position
    assert data["destination"].num_nodes == 1

    # 16 out-of-play piece nodes (8 white, 8 black)
    assert data["out_of_play_piece"].num_nodes == 16

    # There should be 7 current-player move edges (queen forbidden first move)
    cur_rel = ("out_of_play_piece", "current_move", "destination")
    assert cur_rel in data.edge_index_dict
    assert data[cur_rel].edge_index.shape[1] == 7

    # Next-player move edges after passing
    next_rel = ("out_of_play_piece", "next_move", "destination")
    assert next_rel in data.edge_index_dict
    assert data[next_rel].edge_index.shape[1] == 7
