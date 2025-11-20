import os
import sys

import torch

import nokamute
from nokamute import Board

try:
    from nokamute import BoardHeteroBuilder
except Exception:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    from python.graph import BoardHeteroBuilder

from deepsnap.batch import Batch
from deepsnap.dataset import GraphDataset
from torch.utils.data import DataLoader

from python.hive_actor_critic import HiveActorCritic


def test_hive_actor_critic_empty_board_shapes():
    """Ensure actor-critic returns policy logits for all current_move edges
    and a single critic value for the graph.
    """
    board = Board()
    ones = BoardHeteroBuilder(board).as_heterograph()
    ds = GraphDataset([ones], task="graph", minimum_node_per_graph=0)
    loader = DataLoader(ds, collate_fn=Batch.collate(), batch_size=1)
    batch = next(iter(loader))

    # debug: inspect heterograph tensors
    print("message_types:", ones.message_types)
    print("node_feature keys:", list(ones.node_feature.keys()))
    for nt, feat in ones.node_feature.items():
        print(nt, feat.shape)
    for k, idx in ones.edge_index.items():
        print("edge_index", k, idx.shape)

    model = HiveActorCritic(ones, hidden_size=32, policy_dim=7)
    model.eval()  # ensure batchnorms don't fail on tiny batches
    policy_logits, critic = model(batch)

    # The empty (starting) board has 7 current_move edges
    assert policy_logits.shape[0] == 7
    assert policy_logits.shape[1] == 7

    assert critic.shape[0] == 1
    assert critic.shape[1] == 1

    # Also assert the helper returns logits in the same order as edge_label_index
    edge_message_type = next(
        mt for mt in batch.edge_label_index.keys() if mt[1] == "current_move"
    )
    logits_map = model.policy_logits_edge_index(batch)
    assert edge_message_type in logits_map
    logits = logits_map[edge_message_type]
    # Expect the DeepSNAP edge_label_index width for this message type
    e_count = batch.edge_label_index[edge_message_type].shape[1]
    assert logits.shape == (e_count, 7)


def test_transposition_value_and_policy_equivalence():
    """Two different move sequences that lead to the same position should have
    identical network predictions (value) and identical policy logits for a
    canonical action: place a White ant to the right of the starting hex.
    """
    import nokamute as _nm

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
        "wB1 wG1\\;"
        "bB1 \\bG1;"
        "wQ wG1/;"
        "bQ /bG1"
    )

    b1 = Board.from_game_string(s1)
    b2 = Board.from_game_string(s2)

    H1 = BoardHeteroBuilder(b1).as_heterograph()
    H2 = BoardHeteroBuilder(b2).as_heterograph()

    # Use a link_pred dataset so batch.edge_label_index contains the current_move
    # message type ordering; this helps us find the exact column index for the
    # chosen White-ant placement across both graphs.
    ds1 = GraphDataset(
        [H1], task="link_pred", minimum_node_per_graph=0, edge_train_mode="all"
    )
    ds2 = GraphDataset(
        [H2], task="link_pred", minimum_node_per_graph=0, edge_train_mode="all"
    )

    batch1 = next(iter(DataLoader(ds1, collate_fn=Batch.collate(), batch_size=1)))
    batch2 = next(iter(DataLoader(ds2, collate_fn=Batch.collate(), batch_size=1)))

    model = HiveActorCritic(H1, hidden_size=32, policy_dim=7)
    model.eval()

    # Compare values
    _, critic1 = model(batch1)
    _, critic2 = model(batch2)
    # The model is random-initialized; due to floating-point differences and
    # graph ordering the results may vary by tiny amounts — accept a small
    # numerical tolerance for equivalence.
    assert torch.allclose(critic1, critic2)

    # Find a canonical action: place a WHITE ant at the same destination hex in
    # both graphs. We'll find any destination hex that is reachable by placing a
    # White ant in both positions.
    G1 = BoardHeteroBuilder(b1).as_networkx()
    G2 = BoardHeteroBuilder(b2).as_networkx()

    def white_ant_destinations(G):
        ds = set()
        for u, v, attrs in G.edges(data=True):
            if attrs.get("edge_type") == "current_move":
                # start node is an out_of_play_piece
                start = G.nodes[u]
                if start.get("color") == "White" and start.get("bug") == "ant":
                    ds.add(G.nodes[v].get("hex"))
        return ds

    dests1 = white_ant_destinations(G1)
    dests2 = white_ant_destinations(G2)
    common = dests1 & dests2
    assert len(common) > 0
    chosen_hex = next(iter(common))

    # Determine the position in model output by checking the label ordering
    # returned by `BoardHeteroBuilder.as_networkx()` — these labels appear
    # in the same order as the policy logits returned by `model(batch)`.

    # Now get the model policy predictions from forward(), which follows the
    # ordering of edges returned by the networkx graph. We can therefore
    # map the white-ant placement string to the returned index.
    pols1, _ = model(batch1)
    pols2, _ = model(batch2)

    # Build a list of current_move edge labels in the forward() ordering.
    labels1 = [
        attrs.get("edge_label")
        for u, v, attrs in BoardHeteroBuilder(b1).as_networkx().edges(data=True)
        if attrs.get("edge_type") == "current_move"
    ]
    labels2 = [
        attrs.get("edge_label")
        for u, v, attrs in BoardHeteroBuilder(b2).as_networkx().edges(data=True)
        if attrs.get("edge_type") == "current_move"
    ]

    # Build the desired Place(...) string for the chosen_hex and White ant.
    # Turn format: Place(<hex>, <bugname>)
    desired_turn = f"Place({chosen_hex}, ant)"
    assert desired_turn in labels1 and desired_turn in labels2
    idx1 = labels1.index(desired_turn)
    idx2 = labels2.index(desired_turn)
    logits1 = pols1[idx1]
    logits2 = pols2[idx2]
    assert torch.allclose(logits1, logits2)


def test_symmetry_value_and_policy_equivalence():
    """Verify value and policy invariance between s1 and its symmetric move
    ordering `s_sym` for two representative moves.
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

    b1 = Board.from_game_string(s1)
    bs = Board.from_game_string(s_sym)

    H1 = BoardHeteroBuilder(b1).as_heterograph()
    Hs = BoardHeteroBuilder(bs).as_heterograph()

    ds1 = GraphDataset(
        [H1], task="link_pred", minimum_node_per_graph=0, edge_train_mode="all"
    )
    dss = GraphDataset(
        [Hs], task="link_pred", minimum_node_per_graph=0, edge_train_mode="all"
    )
    batch1 = next(iter(DataLoader(ds1, collate_fn=Batch.collate(), batch_size=1)))
    batchs = next(iter(DataLoader(dss, collate_fn=Batch.collate(), batch_size=1)))

    model = HiveActorCritic(H1, hidden_size=32, policy_dim=7)
    model.eval()

    _, critic1 = model(batch1)
    _, critics = model(batchs)
    assert torch.allclose(critic1, critics, atol=1e-2)

    # Build networkx graphs to find canonical actions
    G1 = BoardHeteroBuilder(b1).as_networkx()
    Gs = BoardHeteroBuilder(bs).as_networkx()

    # 1) Find a destination node adjacent to all White in-play pieces
    white_in_play = [
        n
        for n, d in G1.nodes(data=True)
        if d.get("node_type") == "in_play_piece" and d.get("color") == "White"
    ]
    candidate_dests = []
    for n, d in G1.nodes(data=True):
        if d.get("node_type") != "destination" or d.get("is_top"):
            continue
        # check adjacency to all white-in-play nodes
        ok = True
        for wid in white_in_play:
            if not any(
                (
                    attrs.get("edge_type") == "adj"
                    and ((u == wid and v == n) or (u == n and v == wid))
                )
                for u, v, attrs in G1.edges(data=True)
            ):
                ok = False
                break
        if ok:
            candidate_dests.append(n)
    assert candidate_dests, "No destination adjacent to all white pieces found"
    chosen_dest = candidate_dests[0]

    # Find a place edge label in both graphs for chosen_dest and white color
    def find_place_label_for_dest(G, dest):
        for u, v, attrs in G.edges(data=True):
            if (
                attrs.get("edge_type") == "current_move"
                and G.nodes[u].get("node_type") == "out_of_play_piece"
                and G.nodes[u].get("color") == "White"
                and v == dest
            ):
                if attrs.get("edge_label") and attrs.get("edge_label").startswith(
                    "Place("
                ):
                    return attrs.get("edge_label")
        return None

    p1 = find_place_label_for_dest(G1, chosen_dest)
    ps = find_place_label_for_dest(Gs, chosen_dest)
    assert p1 is not None and ps is not None

    pols1, _ = model(batch1)
    polss, _ = model(batchs)
    labels1 = [
        attrs.get("edge_label")
        for u, v, attrs in G1.edges(data=True)
        if attrs.get("edge_type") == "current_move"
    ]
    labelss = [
        attrs.get("edge_label")
        for u, v, attrs in Gs.edges(data=True)
        if attrs.get("edge_type") == "current_move"
    ]
    assert p1 in labels1 and ps in labelss
    idx1 = labels1.index(p1)
    idxs = labelss.index(ps)
    assert torch.allclose(pols1[idx1], polss[idxs], atol=1e-2)

    # 2) Find a Move of White beetle onto the White grasshopper
    def find_move_to_grasshopper(G):
        gh = next(
            (
                n
                for n, d in G.nodes(data=True)
                if d.get("node_type") == "in_play_piece"
                and d.get("color") == "White"
                and d.get("bug") == "grasshopper"
            ),
            None,
        )
        if gh is None:
            return None
        gh_hex = G.nodes[gh].get("hex")
        for u, v, attrs in G.edges(data=True):
            if (
                attrs.get("edge_type") == "current_move"
                and attrs.get("edge_label")
                and attrs.get("edge_label").startswith("Move(")
            ):
                inner = attrs.get("edge_label")[len("Move(") : -1]
                parts = [p.strip() for p in inner.split(",")]
                if len(parts) != 2:
                    continue
                to_hex = int(parts[1])
                if (
                    to_hex == gh_hex
                    and G.nodes[u].get("color") == "White"
                    and G.nodes[u].get("bug") == "beetle"
                ):
                    return attrs.get("edge_label")
        return None

    m1 = find_move_to_grasshopper(G1)
    ms = find_move_to_grasshopper(Gs)
    assert m1 is not None and ms is not None
    idx1 = labels1.index(m1)
    idxs = labelss.index(ms)
    assert torch.allclose(pols1[idx1], polss[idxs], atol=1e-2)
