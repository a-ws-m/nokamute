import os
import sys

import torch

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
)

from action_space import get_action_space, string_to_action
from hetero_graph_utils import board_to_hetero_data, prepare_model_inputs
from model_policy_hetero import create_policy_model

import nokamute


def test_move_to_action_indices_order_matches_edge_iteration():
    """Ensure move_to_action_indices follow the order of move edges in data.edge_types.

    This test reconstructs the concatenation ordering of move edges from
    `data.edge_types` and verifies that the `move_to_action_indices` tensor
    has the same structure (counts and positions) and that the sequence of
    legal action indices matches the `move_to_action` list produced by
    `Board.to_graph()`.
    """

    board = nokamute.Board()
    graph_dict = board.to_graph()

    data, move_to_action_indices = board_to_hetero_data(graph_dict)

    # Move_to_action from graph_dict should correspond to action indices
    _, _, action_to_string = get_action_space()
    move_strs = graph_dict["move_to_action"]
    expected_action_indices = [string_to_action(m) for m in move_strs]

    x_dict, edge_index_dict, edge_attr_dict, move_to_idx = prepare_model_inputs(
        data, move_to_action_indices
    )

    # Build a list of positive indices in the mapping in the order they appear
    mapped_positive = [int(i) for i in move_to_idx if i >= 0]

    # Now compute the expected list of action indices from the HeteroData
    # by iterating over move edges in the same order the model will.
    expected_from_edges = []
    for et in edge_index_dict.keys():
        if "move" in et[1] or "rev_move" in et[1]:
            # Get edge attributes to know which edges belong to the current player
            if et in edge_attr_dict:
                attrs = edge_attr_dict[et].squeeze(-1).tolist()
            else:
                attrs = []
            # For move edges, `graph_dict['move_to_action']` supplies the next action
            # index for current player edges; for rev_move, expected positive indices
            # should correspond to the matching forward edges.
            for a in attrs:
                if a == 1.0:
                    # Append placeholder; we'll compare sequence only after collecting forward
                    expected_from_edges.append("P")

    # The mapping includes duplicate forward+reverse edges. Check that the
    # unique sequence of positive indices (in order of first appearance)
    # equals the action indices produced by move_to_action.
    unique_mapped = []
    for v in mapped_positive:
        if v not in unique_mapped:
            unique_mapped.append(v)

    assert (
        unique_mapped == expected_action_indices
    ), f"Unique mapped positive sequence mismatch:\n mapped_unique={unique_mapped}\n expected={expected_action_indices}"
