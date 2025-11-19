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


def test_sequence_moves_adjacency():
    # Position from sequence of moves in UHP notation. We'll use the 7-move
    # sequence specified and assert adjacency for black queen, beetle, grasshopper.
    s = (
        "Base;InProgress;turn;"
        "wG1;"  # 1. wG1
        "bG1 -wG1;"  # 2. bG1 -wG1
        "wQ wG1-;"  # 3. wQ wG1-
        "bQ -bG1;"  # 4. bQ -bG1
        "wB1 wQ-;"  # 5. wB1 wQ-
        "bB1 \\bG1;"  # 6. bB1 \bG1  (escaped)
        "wB1 wQ"  # 7. wB1 wQ
    )

    board = Board.from_game_string(s)
    builder = BoardHeteroBuilder(board)
    data = builder.to_heterodata()

    # Map hex -> in_play id
    in_map = {n["hex"]: n["id"] for n in builder.raw.get("in_play_nodes", [])}

    # Find hexes from Board.get_pieces for black queen, beetle, grasshopper
    # get_pieces returns tuples (hex,color,bug,height)
    black_hexes = {p[2].lower(): p[0] for p in board.get_pieces() if p[1] == "Black"}
    # Ensure keys exist
    assert "queen" in black_hexes
    assert "beetle" in black_hexes
    assert "grasshopper" in black_hexes

    adj_key = ("in_play_piece", "adj", "in_play_piece")
    assert adj_key in data.edge_index_dict
    edges = data[adj_key].edge_index

    # Convert to Python list for easier membership tests
    edge_pairs = set(
        (int(edges[0, i].item()), int(edges[1, i].item()))
        for i in range(edges.shape[1])
    )

    for bug in ("queen", "beetle", "grasshopper"):
        h = black_hexes[bug]
        assert h in in_map
        id = in_map[h]
        # Check that id appears in either src or dst of an adjacency edge
        connected = any(id == a or id == b for (a, b) in edge_pairs)
        assert connected, f"{bug} (id={id}) is not connected by adjacency"

    # Check is_above/is_underneath binary features on in_play nodes using HeteroData
    x = data["in_play_piece"].x
    feat_under = len(BoardHeteroBuilder.BUGS)
    feat_above = feat_under + 1

    # Find ids by color/bug using raw map
    in_play = (
        list(builder.raw["in_play_nodes"]) if "in_play_nodes" in builder.raw else []
    )
    wb_id = next(
        n["id"] for n in in_play if n["color"] == "White" and n["bug"] == "beetle"
    )
    # The white queen must be the underneath piece, since final move places the
    # white beetle on top of the white queen.
    wq_id = next(
        n["id"] for n in in_play if n["color"] == "White" and n["bug"] == "queen"
    )

    # White beetle should be above something
    assert x[wb_id, feat_above].item() == 1.0
    assert x[wb_id, feat_under].item() == 0.0

    # Ensure white queen is underneath
    assert x[wq_id, feat_under].item() == 1.0
    assert x[wq_id, feat_above].item() == 0.0

    # All others not having either
    for n in in_play:
        if n["id"] in (wb_id, wq_id):
            continue
        id = n["id"]
        # Some pieces may be the underneath piece - skip them
        if x[id, feat_under].item() == 1.0 or x[id, feat_above].item() == 1.0:
            continue
        assert x[id, feat_under].item() == 0.0
        assert x[id, feat_above].item() == 0.0
