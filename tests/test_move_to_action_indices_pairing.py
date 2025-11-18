import os
import sys

import torch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
)

import torch_geometric.transforms as T
from hetero_graph_utils import board_to_hetero_data
from model_policy_hetero import create_policy_model
from torch_geometric.data import HeteroData

import nokamute


def test_reverse_edges_pair_with_forward_edges(monkeypatch):
    """Tests robust forward-reverse pairing for move_to_action_indices.

    We monkeypatch the ToUndirected transform so reverse edges are inserted
    before forward edges and then check that `move_to_action_indices` maps
    each reverse edge to the same action index as the matching forward edge
    (where appropriate) by matching node pair (src,dst) -> action.
    """

    # Build a board and its graph dict
    board = nokamute.Board()
    graph_dict = board.to_graph()

    # Replace ToUndirected with a transform that inserts reverse edges first
    class ReverseFirstToUndirected:
        def __call__(self, data: HeteroData):
            new = HeteroData()
            # Copy node features
            for nt in data.node_types:
                new[nt].x = data[nt].x.clone()
                if hasattr(data[nt], "batch"):
                    new[nt].batch = data[nt].batch.clone()

            # Iterate and insert reverse edges before a forward move type
            for et in data.edge_types:
                src, etype, dst = et
                # For move edges: insert the reverse first
                if "move" in etype and "rev_" not in etype:
                    # Create reverse tuple
                    rev = (dst, "rev_move", src)
                    if data[et].edge_index.shape[1] > 0:
                        rev_idx = torch.stack(
                            [data[et].edge_index[1], data[et].edge_index[0]], dim=0
                        )
                        new[rev].edge_index = rev_idx
                        if hasattr(data[et], "edge_attr"):
                            new[rev].edge_attr = data[et].edge_attr.clone()

                # Then insert forward edge as normal
                if data[et].edge_index.shape[1] > 0:
                    new[et].edge_index = data[et].edge_index.clone()
                    if hasattr(data[et], "edge_attr"):
                        new[et].edge_attr = data[et].edge_attr.clone()

            return new

    monkeypatch.setattr(T, "ToUndirected", ReverseFirstToUndirected)

    # Now call board_to_hetero_data with patched transform
    data, move_to_action_indices = board_to_hetero_data(graph_dict)

    # Collect mapping segments for move/rev_move
    segments = []
    offset = 0
    for et in data.edge_types:
        if "move" in et[1] or "rev_move" in et[1]:
            count = data[et].edge_index.shape[1]
            seg = move_to_action_indices[offset : offset + count]
            segments.append((et, data[et].edge_index, seg))
            offset += count

    # Build forward mapping: (src,dst) -> action index
    forward_map = {}
    for et, idx, seg in segments:
        if "move" == et[1]:
            for j in range(idx.shape[1]):
                action_idx = int(seg[j].item())
                if action_idx >= 0:
                    src = int(idx[0, j].item())
                    dst = int(idx[1, j].item())
                    forward_map[(src, dst)] = action_idx

    # Check reverse edges map to corresponding forward index
    for et, idx, seg in segments:
        if "rev_move" == et[1]:
            for j in range(idx.shape[1]):
                action_idx = int(seg[j].item())
                if action_idx >= 0:
                    src = int(idx[0, j].item())
                    dst = int(idx[1, j].item())
                    expected = forward_map.get((dst, src), None)
                    assert (
                        expected == action_idx
                    ), f"Reverse edge ({src}->{dst}) mapped to {action_idx}, expected {expected}"
