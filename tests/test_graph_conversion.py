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


def test_transposition_hash_equivalence():
    """Two different move sequences that result in the same board position
    should produce identical Weisfeiler-Lehman hashes when using node
    features and edge types.
    """
    s1 = (
        "Base+M;InProgress;turn;"
        "wG1;"
        "bG1 -wG1;"
        "wQ wG1\\;"
        "bQ /bG1;"
        "wB1 wG1/;"
        "bB1 \\bG1"
    )

    s2 = (
        "Base+M;InProgress;turn;"
        "wG1;"
        "bG1 -wG1;"
        "wB1 wG1/;"
        "bB1 \\bG1;"
        "wQ wG1\\;"
        "bQ /bG1"
    )

    b1 = Board.from_game_string(s1)
    b2 = Board.from_game_string(s2)

    from python.graph import BoardHeteroBuilder

    h1 = BoardHeteroBuilder(b1)
    h2 = BoardHeteroBuilder(b2)

    assert h1.weisfeiler_lehman_hash() == h2.weisfeiler_lehman_hash()


def test_transposition_hash_equivalence_permutation():
    """A permutation of the same moves produces an identical board; verify
    WL hash equality.
    """
    s1 = (
        "Base+M;InProgress;turn;"
        "wG1;"
        "bG1 -wG1;"
        "wQ wG1\\;"
        "bQ /bG1;"
        "wB1 wG1/;"
        "bB1 \\bG1"
    )

    s3 = (
        "Base+M;InProgress;turn;"
        "wG1;"
        "bG1 -wG1;"
        "wB1 wG1/;"
        "bB1 \\bG1;"
        "wQ wG1\\;"
        "bQ /bG1"
    )

    b1 = Board.from_game_string(s1)
    b3 = Board.from_game_string(s3)

    from python.graph import BoardHeteroBuilder

    h1 = BoardHeteroBuilder(b1)
    h3 = BoardHeteroBuilder(b3)

    assert h1.weisfeiler_lehman_hash() == h3.weisfeiler_lehman_hash()


def test_transposition_hash_symmetry_debug():
    """Check WL hash for a symmetric variant (rotation/reflection). If the
    hash differs, dump human-friendly diagnostics so we can see why the graphs
    are not considered equivalent.
    """
    s1 = (
        "Base+M;InProgress;turn;"
        "wG1;"
        "bG1 -wG1;"
        "wQ wG1\\;"
        "bQ /bG1;"
        "wB1 wG1/;"
        "bB1 \\bG1"
    )

    s_sym = (
        "Base+M;InProgress;turn;"
        "wG1;"
        "bG1 -wG1;"
        "wB1 wG1\\;"
        "bB1 /bG1;"
        "wQ wG1/;"
        "bQ \\bG1"
    )

    s3 = (
        "Base+M;InProgress;turn;"
        "wG1;"
        "bG1 -wG1;"
        "wB1 wG1/;"
        "bB1 \\bG1;"
        "wQ wG1\\;"
        "bQ /bG1"
    )

    from collections import Counter

    from python.graph import BoardHeteroBuilder

    b1 = Board.from_game_string(s1)
    bsym = Board.from_game_string(s_sym)
    b3 = Board.from_game_string(s3)

    h1 = BoardHeteroBuilder(b1).weisfeiler_lehman_hash()
    hs = BoardHeteroBuilder(bsym).weisfeiler_lehman_hash()
    h3 = BoardHeteroBuilder(b3).weisfeiler_lehman_hash()

    print(f"hash s1={h1}\nhash sym={hs}\nhash s3={h3}")

    if not (h1 == hs == h3):
        # Dump node feature counts and edge type counts for each graph
        for name, board in [("s1", b1), ("sym", bsym), ("s3", b3)]:
            G = BoardHeteroBuilder(board).as_networkx()
            feat_cnt = Counter()
            for _, d in G.nodes(data=True):
                nf = d.get("node_feature")
                if nf is None:
                    feat_cnt[None] += 1
                else:
                    try:
                        import torch

                        if isinstance(nf, torch.Tensor):
                            key = tuple(nf.detach().cpu().tolist())
                        else:
                            key = tuple(nf)
                    except Exception:
                        key = str(nf)
                    feat_cnt[key] += 1
            edge_cnt = Counter(
                [attrs.get("edge_type") for _, _, attrs in G.edges(data=True)]
            )
            print("")
            print(name)
            print(" node feature counts:")
            for k, v in feat_cnt.most_common():
                print(k, v)
            print(" edge type counts:")
            print(edge_cnt)

            # list the current move and next_move edges to compare
            cur = [
                (u, v, attrs.get("edge_label"))
                for u, v, attrs in G.edges(data=True)
                if attrs.get("edge_type") == "current_move"
            ]
            nxt = [
                (u, v, attrs.get("edge_label"))
                for u, v, attrs in G.edges(data=True)
                if attrs.get("edge_type") == "next_move"
            ]
            print(name, "current_move count", len(cur))
            print(name, "next_move count", len(nxt))
            print(name, "sample current_move labels:", [c[2] for c in cur][:10])
        # Fail test so we can iterate and fix code
        assert (
            False
        ), "WL hashes differ between symmetric transpositions; see debug output"
    # If they match, assert equality to the canonical sample
    assert h1 == h3


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
    import nokamute as _nm

    assert all(
        isinstance(attrs.get("edge_label"), str)
        and _nm.Turn.from_string(attrs.get("edge_label")).is_place()
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
        isinstance(attrs.get("edge_label"), str)
        and _nm.Turn.from_string(attrs.get("edge_label")).is_place()
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
