import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
)

import torch
from action_space import string_to_action
from hetero_graph_utils import board_to_hetero_data, prepare_model_inputs
from model_policy_hetero import create_policy_model
from torch_geometric.loader import DataLoader

import nokamute


def test_move_to_action_indices_batched_order_preserved():
    """Batched conversion preserves per-example move ordering.

    For each example in a batched `HeteroData`, the ordered unique positive
    action indices derived from `move_to_action_indices` should match the
    original `graph_dict['move_to_action']` list (converted to action ids).
    """

    model = create_policy_model({"hidden_dim": 32, "num_layers": 1, "num_heads": 1})

    boards = [nokamute.Board(), nokamute.Board()]
    examples = []
    for board in boards:
        graph = board.to_graph()
        data, move_to_action_indices = board_to_hetero_data(graph)
        examples.append((data, move_to_action_indices, graph))

    # Build a DataLoader with the HeteroData examples
    dataset = []
    for d, m, g in examples:
        d.move_to_action_indices = m
        dataset.append(d)

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    for batch in loader:
        x_dict, edge_index_dict, edge_attr_dict, mta = prepare_model_inputs(
            batch, batch.move_to_action_indices
        )

        # Reconstruct per-example ordered unique action IDs from batch.move_to_action_indices
        batch_dict = {
            nt: x_dict[nt].batch for nt in x_dict if hasattr(x_dict[nt], "batch")
        }
        move_edge_batch = []
        for et in edge_index_dict.keys():
            if "move" in et[1] or "rev_move" in et[1]:
                edge_idx = edge_index_dict[et]
                if edge_idx.shape[1] > 0 and et[0] in batch_dict:
                    move_edge_batch.append(batch_dict[et[0]][edge_idx[0]])

        all_move_batch = (
            torch.cat(move_edge_batch)
            if move_edge_batch
            else torch.empty(0, dtype=torch.long)
        )

        for i, (d, m, g) in enumerate(examples):
            # The original `move_to_action` list from the graph contains a unique
            # set of action ids for the current player in order. Convert the
            # strings to action ids here as the expected ordering.
            positive = [string_to_action(s) for s in g["move_to_action"]]
            # unique ordered from batch
            legal_idxs = batch.move_to_action_indices[all_move_batch == i]
            seen = set()
            unique_ordered = []
            for v in legal_idxs.tolist():
                if v >= 0 and v not in seen:
                    unique_ordered.append(v)
                    seen.add(v)

            if unique_ordered != positive:
                print(f"unique_ordered({len(unique_ordered)}): {unique_ordered}")
                print(f"positive({len(positive)}): {positive}")
            assert unique_ordered == positive
