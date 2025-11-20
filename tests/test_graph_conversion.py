import nokamute
from nokamute import Board

try:
    from nokamute import BoardHeteroBuilder
except Exception:
    import os
    import sys

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    from python.graph import BoardHeteroBuilder


def test_basic_graph_conversion_networkx():
    board = Board()
    moves = board.legal_moves()
    place_move = next((m for m in moves if m.is_place()), None)
    assert place_move is not None
    board.apply(place_move)
    G = BoardHeteroBuilder(board).as_networkx()
    assert any(d.get("node_type") == "in_play_piece" for _, d in G.nodes(data=True))
    assert (
        sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "destination")
        >= 6
    )
    assert any(
        attrs.get("edge_type") == "adj"
        and G.nodes[u].get("node_type") == "in_play_piece"
        and G.nodes[v].get("node_type") == "destination"
        for u, v, attrs in G.edges(data=True)
    )
    assert any(
        attrs.get("edge_type") == "current_move" and attrs.get("edge_label") is not None
        for _, _, attrs in G.edges(data=True)
    )


def test_starting_position_counts_networkx():
    G = BoardHeteroBuilder(Board()).as_networkx()
    assert (
        sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "destination")
        == 1
    )
    assert (
        sum(
            1
            for _, d in G.nodes(data=True)
            if d.get("node_type") == "out_of_play_piece"
        )
        == 16
    )
    cur_edges = [
        (u, v, attrs)
        for u, v, attrs in G.edges(data=True)
        if attrs.get("edge_type") == "current_move"
        and G.nodes[u].get("node_type") == "out_of_play_piece"
        and G.nodes[v].get("node_type") == "destination"
    ]
    assert len(cur_edges) == 7
    assert all(
        attrs.get("edge_label") is not None
        and hasattr(attrs.get("edge_label"), "is_place")
        and attrs.get("edge_label").is_place()
        for _, _, attrs in cur_edges
    )
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


def test_sequence_moves_adjacency_networkx():
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
    G = BoardHeteroBuilder(Board.from_game_string(s)).as_networkx()
    in_nodes = [
        (gid, d)
        for gid, d in G.nodes(data=True)
        if d.get("node_type") == "in_play_piece"
    ]
    black_beetle_id = next(
        gid
        for gid, d in in_nodes
        if d.get("color") == "Black" and d.get("bug") == "beetle"
    )
    black_queen_id = next(
        gid
        for gid, d in in_nodes
        if d.get("color") == "Black" and d.get("bug") == "queen"
    )
    black_grasshopper_ids = [
        gid
        for gid, d in in_nodes
        if d.get("color") == "Black" and d.get("bug") == "grasshopper"
    ]
    edge_pairs = set()
    for u, v, attrs in G.edges(data=True):
        if (
            attrs.get("edge_type") == "adj"
            and G.nodes[u].get("node_type") == "in_play_piece"
            and G.nodes[v].get("node_type") == "in_play_piece"
        ):
            edge_pairs.add((int(u), int(v)))
            edge_pairs.add((int(v), int(u)))
    adj_with_beetle = {b for (a, b) in edge_pairs if a == black_beetle_id} | {
        a for (a, b) in edge_pairs if b == black_beetle_id
    }
    assert black_queen_id in adj_with_beetle
    assert len([g for g in black_grasshopper_ids if g in adj_with_beetle]) == 2
    adj_with_queen = {b for (a, b) in edge_pairs if a == black_queen_id} | {
        a for (a, b) in edge_pairs if b == black_queen_id
    }
    assert black_beetle_id in adj_with_queen
    assert len([g for g in black_grasshopper_ids if g in adj_with_queen]) == 3
    x_map = {gid: d["node_feature"] for gid, d in in_nodes}
    feat_under = len(BoardHeteroBuilder.BUGS)
    feat_above = feat_under + 1
    wb_id = next(
        gid
        for gid, d in in_nodes
        if d.get("color") == "White" and d.get("bug") == "beetle"
    )
    wq_id = next(
        gid
        for gid, d in in_nodes
        if d.get("color") == "White" and d.get("bug") == "queen"
    )
    assert x_map[wb_id][feat_under].item() == 1.0
    assert x_map[wb_id][feat_above].item() == 1.0
    assert x_map[wq_id][feat_under].item() == 1.0
    assert x_map[wq_id][feat_above].item() == 0.0
    out_nodes = [
        d for _, d in G.nodes(data=True) if d.get("node_type") == "out_of_play_piece"
    ]
    for n in out_nodes:
        assert not (
            n.get("color") == "Black" and n.get("bug") in ("grasshopper", "queen")
        )


def test_destination_counts_after_sequence_networkx():
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
    G = BoardHeteroBuilder(Board.from_game_string(s)).as_networkx()
    dest_nodes = [
        d for _, d in G.nodes(data=True) if d.get("node_type") == "destination"
    ]
    assert len(dest_nodes) == 22
    top_count = sum(1 for n in dest_nodes if n.get("is_top", False))
    assert top_count == 7
    around_count = sum(1 for n in dest_nodes if not n.get("is_top", False))
    assert around_count == 15
