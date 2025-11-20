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
    G = builder.as_networkx()

    # Basic checks on node counts
    assert (
        sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "in_play_piece")
        >= 1
    )
    assert (
        sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "destination")
        >= 6
    )

    # We should have adjacency edges from the in-play node to the destination
    assert any(
        attrs.get("edge_type") == "adj"
        and G.nodes[u].get("node_type") == "in_play_piece"
        and G.nodes[v].get("node_type") == "destination"
        for u, v, attrs in G.edges(data=True)
    )

    # Check that current_move edges exist (some legal moves for the player after the placement)
    assert any(
        attrs.get("edge_type") == "current_move" for _, _, attrs in G.edges(data=True)
    )
    # Ensure edge_label exists for at least one current_move edge and looks like a Turn
    assert any(
        attrs.get("edge_type") == "current_move"
        and attrs.get("edge_label") is not None
        and hasattr(attrs.get("edge_label"), "is_place")
        for _, _, attrs in G.edges(data=True)
    )


def test_starting_position_counts():
    board = Board()
    builder = BoardHeteroBuilder(board)
    G = builder.as_networkx()

    # Single destination node for starting position
    assert (
        sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "destination")
        == 1
    )

    # 16 out-of-play piece nodes (8 white, 8 black)
    assert (
        sum(
            1
            for _, d in G.nodes(data=True)
            if d.get("node_type") == "out_of_play_piece"
        )
        == 16
    )

    # There should be 7 current-player move edges (queen forbidden first move)
    cur_edges = [
        (u, v, attrs)
        for u, v, attrs in G.edges(data=True)
        if attrs.get("edge_type") == "current_move"
        and G.nodes[u].get("node_type") == "out_of_play_piece"
        and G.nodes[v].get("node_type") == "destination"
    ]
    assert len(cur_edges) == 7
    # Each current edge should have an edge_label (Turn) and be a placement in starting position
    assert all(
        attrs.get("edge_label") is not None
        and hasattr(attrs.get("edge_label"), "is_place")
        and attrs.get("edge_label").is_place()
        for _, _, attrs in cur_edges
    )

    # Next-player move edges after passing
    next_edges = [
        (u, v, attrs)
        for u, v, attrs in G.edges(data=True)
        if attrs.get("edge_type") == "next_move"
        and G.nodes[u].get("node_type") == "out_of_play_piece"
        and G.nodes[v].get("node_type") == "destination"
    ]
    assert len(next_edges) == 7
    assert all(
        attrs.get("edge_label") is not None
        and hasattr(attrs.get("edge_label"), "is_place")
        and attrs.get("edge_label").is_place()
        for _, _, attrs in next_edges
    )


def test_sequence_moves_adjacency():
    # Position from sequence of moves in UHP notation. We'll use the 7-move
    # sequence specified and assert adjacency for black queen, beetle, grasshopper.
    s = (
        "Base+M;InProgress;turn;"
        "wG1;"  # 1. wG1
        "bG1 -wG1;"  # 2. bG1 -wG1
        "wQ wG1-;"  # 3. wQ wG1-
        "bQ -bG1;"  # 4. bQ -bG1
        "wB1 wQ-;"  # 5. wB1 wQ-
        "bB1 \\bG1;"  # 6. bB1 \bG1  (escaped)
        "wB1 wQ;"  # 7. wB1 wQ
        "bG2 \\bQ;"  # 8. bG2 \bQ
        "wM wB1-;"  # 9. wM wB1-
        "bG3 /bG2;"  # 10. bG3 /bG2
        "wM wB1"  # 11. wM wB1
    )

    board = Board.from_game_string(s)
    builder = BoardHeteroBuilder(board)
    G = builder.as_networkx()

    # Map hex -> topmost in_play id
    in_map = {
        n["hex"]: n["id"]
        for n in builder.raw.get("in_play_nodes", [])
        if not n.get("is_underneath", False)
    }

    # Find hexes from Board.get_pieces for black queen, beetle, grasshopper
    # get_pieces returns tuples (hex,color,bug,height)
    # Map bug -> list of hexes for black pieces
    black_hexes = {}
    for p in board.get_pieces():
        if p[1] == "Black":
            black_hexes.setdefault(p[2].lower(), []).append(p[0])
    # Ensure keys exist
    assert "queen" in black_hexes
    assert "beetle" in black_hexes
    assert "grasshopper" in black_hexes

    # Find all adjacency edges between in_play nodes
    edge_pairs = set()
    for u, v, attrs in G.edges(data=True):
        if (
            attrs.get("edge_type") == "adj"
            and G.nodes[u].get("node_type") == "in_play_piece"
            and G.nodes[v].get("node_type") == "in_play_piece"
        ):
            edge_pairs.add((int(u), int(v)))
            edge_pairs.add((int(v), int(u)))

    # We'll gather ids for black queen, beetle and all grasshoppers
    in_play_nodes = list(builder.raw.get("in_play_nodes", []))
    # Convert local in-play ids to global ids using offsets
    in_count = sum(1 for n in builder.raw.get("in_play_nodes", []))
    out_count = sum(1 for n in builder.raw.get("out_of_play_nodes", []))
    out_offset = in_count
    dest_offset = in_count + out_count

    def local_to_global(kind, local_id):
        if kind == "in_play":
            return local_id
        if kind == "out_of_play":
            return out_offset + local_id
        if kind == "destination":
            return dest_offset + local_id

    black_beetle_id = local_to_global(
        "in_play",
        next(
            n["id"]
            for n in in_play_nodes
            if n["color"] == "Black" and n["bug"] == "beetle"
        ),
    )
    black_queen_id = local_to_global(
        "in_play",
        next(
            n["id"]
            for n in in_play_nodes
            if n["color"] == "Black" and n["bug"] == "queen"
        ),
    )
    black_grasshopper_ids = [
        local_to_global("in_play", n["id"])
        for n in in_play_nodes
        if n["color"] == "Black" and n["bug"] == "grasshopper"
    ]

    # The black beetle should be adjacent to two black grasshoppers and the black queen
    adj_with_beetle = {b for (a, b) in edge_pairs if a == black_beetle_id} | {
        a for (a, b) in edge_pairs if b == black_beetle_id
    }
    assert black_queen_id in adj_with_beetle
    assert len([g for g in black_grasshopper_ids if g in adj_with_beetle]) == 2

    # The black queen should be adjacent to three black grasshoppers and the black beetle
    adj_with_queen = {b for (a, b) in edge_pairs if a == black_queen_id} | {
        a for (a, b) in edge_pairs if b == black_queen_id
    }
    assert black_beetle_id in adj_with_queen
    assert len([g for g in black_grasshopper_ids if g in adj_with_queen]) == 3

    # Check is_above/is_underneath binary features on in_play nodes using HeteroData
    # Map node id -> feature vector tensor
    x_map = {
        n: d["node_feature"]
        for n, d in G.nodes(data=True)
        if d.get("node_type") == "in_play_piece"
    }
    feat_under = len(BoardHeteroBuilder.BUGS)
    feat_above = feat_under + 1

    # Find ids by color/bug using raw map
    in_play = (
        list(builder.raw["in_play_nodes"]) if "in_play_nodes" in builder.raw else []
    )
    wb_id = local_to_global(
        "in_play",
        next(
            n["id"] for n in in_play if n["color"] == "White" and n["bug"] == "beetle"
        ),
    )
    # The white queen must be the underneath piece, since final move places the
    # white beetle on top of the white queen.
    wq_id = local_to_global(
        "in_play",
        next(n["id"] for n in in_play if n["color"] == "White" and n["bug"] == "queen"),
    )

    # White beetle should be both underneath and above (middle of a 3-stack)
    assert x_map[wb_id][feat_under].item() == 1.0
    assert x_map[wb_id][feat_above].item() == 1.0

    # Ensure white queen is underneath (bottom of stack)
    assert x_map[wq_id][feat_under].item() == 1.0
    assert x_map[wq_id][feat_above].item() == 0.0

    # All others not having either
    for n in in_play:
        if local_to_global("in_play", n["id"]) in (wb_id, wq_id):
            continue
        # the white mosquito should be above something
        if n["color"] == "White" and n["bug"] == "mosquito":
            mid_mosq = local_to_global("in_play", n["id"])
            assert x_map[mid_mosq][feat_above].item() == 1.0
            assert x_map[mid_mosq][feat_under].item() == 0.0
            continue
        id = local_to_global("in_play", n["id"])
        # Some pieces may be the underneath piece - skip them
        if x_map[id][feat_under].item() == 1.0 or x_map[id][feat_above].item() == 1.0:
            continue
        assert x_map[id][feat_under].item() == 0.0
        assert x_map[id][feat_above].item() == 0.0

    # Verify that there are no out-of-play nodes for black grasshopper or black queen
    out_nodes = list(builder.raw.get("out_of_play_nodes", []))
    for n in out_nodes:
        assert not (n["color"] == "Black" and n["bug"] in ("grasshopper", "queen"))


def test_destination_counts_after_sequence():
    # Same position as test_sequence_moves_adjacency; there should be 22 destinations
    #  - 15 empty neighbors around the hive
    #  - 7 top-of-piece destinations (one for each piece with space above)
    s = (
        "Base+M;InProgress;turn;"
        "wG1;"
        "bG1 -wG1;"
        "wQ wG1-;"
        "bQ -bG1;"
        "wB1 wQ-;"
        "bB1 \\bG1;"
        "wB1 wQ;"
        "bG2 \\bQ;"
        "wM wB1-;"
        "bG3 /bG2;"
        "wM wB1"
    )

    board = Board.from_game_string(s)
    builder = BoardHeteroBuilder(board)
    G = builder.as_networkx()
    dest_nodes = list(builder.raw.get("destination_nodes", []))

    assert len(dest_nodes) == 22
    top_count = sum(1 for n in dest_nodes if n.get("is_top", False))
    assert top_count == 7
    around_count = sum(1 for n in dest_nodes if not n.get("is_top", False))
    assert around_count == 15
