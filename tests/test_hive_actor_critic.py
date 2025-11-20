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
