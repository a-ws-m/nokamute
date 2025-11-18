"""
Helper functions to convert board graph representation to PyTorch Geometric format.
"""

import logging
import os

import torch
import torch_geometric.transforms as T
from custom_hetero_data import StackedHeteroData
from torch_geometric.data import HeteroData


def board_to_hetero_data(graph_dict):
    """
    Convert board graph dictionary from Rust to PyTorch Geometric HeteroData format.

    This function also applies the ToUndirected transform, which adds reverse edges
    for all edge types. This ensures bidirectional message passing in the GNN,
    allowing information to flow in both directions along each edge.

    For example:
    - ('in_play', 'neighbour', 'destination') gets reverse edge ('destination', 'rev_neighbour', 'in_play')
    - ('out_of_play', 'move', 'destination') gets reverse edge ('destination', 'rev_move', 'out_of_play')

    Args:
        graph_dict: Dictionary returned by board.to_graph() with keys like:
                   - x_in_play, x_out_of_play, x_destination (node features)
                   - edge_index_{src}_{edge}_{dst} (edge indices)
                   - edge_attr_{src}_{edge}_{dst} (edge attributes)
                   - move_to_action (list of move strings)

    Returns:
        data: HeteroData object with node and edge information (with reverse edges added)
        move_to_action_indices: Tensor of action space indices for legal moves
    """
    import logging
    import os

    from action_space import string_to_action

    # Create HeteroData which stacks certain attributes along a new axis
    # rather than concatenating them. This is necessary for per-edge
    # predictions (action values) that we later index per-example.
    data = StackedHeteroData()

    # Add node features for each node type
    if len(graph_dict["x_in_play"]) > 0:
        data["in_play"].x = torch.tensor(graph_dict["x_in_play"], dtype=torch.float32)
    else:
        data["in_play"].x = torch.empty((0, 10), dtype=torch.float32)

    if len(graph_dict["x_out_of_play"]) > 0:
        data["out_of_play"].x = torch.tensor(
            graph_dict["x_out_of_play"], dtype=torch.float32
        )
    else:
        data["out_of_play"].x = torch.empty((0, 10), dtype=torch.float32)

    if len(graph_dict["x_destination"]) > 0:
        data["destination"].x = torch.tensor(
            graph_dict["x_destination"], dtype=torch.float32
        )
    else:
        data["destination"].x = torch.empty((0, 10), dtype=torch.float32)

    # Add edge indices and attributes for each edge type
    for key in graph_dict.keys():
        if key.startswith("edge_index_"):
            # Parse edge type from key: edge_index_{src}_{edge}_{dst}
            # Node types can have underscores: in_play, out_of_play, destination
            # Edge types: neighbour, move
            remainder = key.replace("edge_index_", "")

            # Try to parse by known edge types
            if "_neighbour_" in remainder:
                parts = remainder.split("_neighbour_")
                src_type = parts[0]
                dst_type = parts[1]
                edge_type = "neighbour"
            elif "_move_" in remainder:
                parts = remainder.split("_move_")
                src_type = parts[0]
                dst_type = parts[1]
                edge_type = "move"
            else:
                continue

            edge_index = graph_dict[key]
            if len(edge_index) == 2 and len(edge_index[0]) > 0:
                data[src_type, edge_type, dst_type].edge_index = torch.tensor(
                    edge_index, dtype=torch.long
                )

                # Add edge attributes if they exist
                attr_key = f"edge_attr_{src_type}_{edge_type}_{dst_type}"
                if attr_key in graph_dict:
                    edge_attr = graph_dict[attr_key]
                    if len(edge_attr) > 0:
                        data[src_type, edge_type, dst_type].edge_attr = torch.tensor(
                            edge_attr, dtype=torch.float32
                        )

    # Convert move strings to action indices
    move_to_action_list = graph_dict["move_to_action"]
    current_player_action_indices = torch.tensor(
        [string_to_action(move_str) for move_str in move_to_action_list],
        dtype=torch.long,
    )

    # (will be generated after applying ToUndirected so that order aligns with
    # `data.edge_types` used by `compute_move_edge_values`.)

    # Apply ToUndirected transform to add reverse edges
    # This ensures message passing works in both directions
    transform = T.ToUndirected()
    data = transform(data)

    # After ToUndirected, move edges are duplicated (forward + reverse)
    # Rebuild `move_to_action_indices` in the same order as the final
    # `data.edge_types`, so it's aligned with the model's iteration order.
    # We'll pair reverse edges with their forward counterparts by matching
    # node index pairs instead of relying on FIFO ordering. This is robust
    # regardless of how ToUndirected inserts reverse edges into `data.edge_types`.
    move_to_action_indices_list = []
    current_idx = 0
    # Map forward edge pairs (src, dst) -> action_idx for pairing
    forward_pair_to_action = {}

    for edge_type in data.edge_types:
        if "move" in edge_type[1] or "rev_move" in edge_type[1]:
            # Get edge indices and attributes safely
            if (
                hasattr(data[edge_type], "edge_index")
                and data[edge_type].edge_index.numel() > 0
            ):
                e_idx = data[edge_type].edge_index
            else:
                e_idx = torch.empty((2, 0), dtype=torch.long)

            if (
                hasattr(data[edge_type], "edge_attr")
                and data[edge_type].edge_attr.numel() > 0
            ):
                e_attr = data[edge_type].edge_attr
            else:
                e_attr = torch.zeros((e_idx.shape[1], 1))

            for j in range(e_idx.shape[1]):
                src = int(e_idx[0, j].item())
                dst = int(e_idx[1, j].item())
                is_current = bool(e_attr[j, 0].item() == 1.0)

                if "move" == edge_type[1]:
                    # Forward move: if current player's edge, assign next action index
                    if is_current:
                        if current_idx >= len(current_player_action_indices):
                            # Something went wrong — the number of current-player edges
                            # does not match the number of moves reported in the graph
                            # (move_to_action). Raise early for visibility.
                            raise ValueError(
                                f"Mismatch: current action indices shorter than edges: {current_idx} >= {len(current_player_action_indices)}"
                            )

                        idx_val = int(current_player_action_indices[current_idx].item())
                        current_idx += 1
                    else:
                        idx_val = -1

                    move_to_action_indices_list.append(idx_val)
                    # Record forward pair -> action for later pairing with reverse
                    forward_pair_to_action[(src, dst)] = idx_val

                else:
                    # Reverse move: lookup corresponding forward edge (dst, src)
                    if is_current:
                        idx_val = forward_pair_to_action.get((dst, src), -1)
                    else:
                        idx_val = -1

                    move_to_action_indices_list.append(idx_val)

    move_to_action_indices = torch.tensor(move_to_action_indices_list, dtype=torch.long)

    # Debugging: log the move_to_action_indices ordering and length so we can
    # verify alignment with model's `compute_move_edge_values` concatenation order.
    try:
        logger = logging.getLogger(__name__)
        if os.getenv("NKAMUTE_DEBUG", os.getenv("NK_DEBUG", "")).lower() in (
            "1",
            "true",
            "yes",
            "y",
        ):
            if not logger.handlers:
                ch = logging.StreamHandler()
                ch.setLevel(logging.DEBUG)
                formatter = logging.Formatter("[%(levelname)s:%(name)s] %(message)s")
                ch.setFormatter(formatter)
                logger.addHandler(ch)
            logger.setLevel(logging.DEBUG)

        logger.debug(
            "board_to_hetero_data: move_to_action_indices len=%d; head=%s",
            len(move_to_action_indices),
            move_to_action_indices[:20].tolist(),
        )

        # Prefer a canonical move order returned by the Rust engine if present.
        # This reduces drift between the order used during selection and the
        # order used by the model at training time. `graph_dict` may include
        # `move_order` as a list of UHP strings; convert those to action indices
        # using the same `string_to_action` helper used for `move_to_action`.
        try:
            if "move_order" in graph_dict:
                py_move_order = graph_dict["move_order"]
                # Convert UHP move strings into action indices
                data.move_order = [string_to_action(mv) for mv in py_move_order]
                logger.debug(
                    "board_to_hetero_data: move_order(from rust)=%s", data.move_order
                )
            else:
                unique_ordered = ordered_unique_action_indices(move_to_action_indices)
                data.move_order = unique_ordered
                logger.debug("board_to_hetero_data: move_order=%s", unique_ordered)
        except Exception:
            pass
    except Exception:
        pass

    return data, move_to_action_indices


def prepare_model_inputs(data, move_to_action_indices):
    """
    Prepare inputs for the heterogeneous GNN model from HeteroData.

    Args:
        data: HeteroData object from board_to_hetero_data (can be batched)
        move_to_action_indices: Tensor of action indices for legal moves

    Returns:
        x_dict: Dictionary of node features by node type (with batch attr if batched)
        edge_index_dict: Dictionary of edge indices by edge type tuple
        edge_attr_dict: Dictionary of edge attributes by edge type tuple
        move_to_action_indices: Passed through unchanged
    """
    # Extract x_dict - preserve batch information if present
    x_dict = {}
    for node_type in data.node_types:
        node_data = data[node_type]
        x_dict[node_type] = node_data.x

        # Attach batch information if this is a batched graph
        if hasattr(node_data, "batch"):
            x_dict[node_type].batch = node_data.batch

    # Extract edge_index_dict and edge_attr_dict
    edge_index_dict = {}
    edge_attr_dict = {}
    for edge_type in data.edge_types:
        edge_index_dict[edge_type] = data[edge_type].edge_index
        if hasattr(data[edge_type], "edge_attr"):
            edge_attr_dict[edge_type] = data[edge_type].edge_attr

    # Debugging block: ensure move_to_action_indices aligns with move edges
    try:
        total_move_edges = 0
        for et, idx in edge_index_dict.items():
            if "move" in et[1] or "rev_move" in et[1]:
                total_move_edges += idx.shape[1]
        if len(move_to_action_indices) != total_move_edges:
            raise ValueError(
                f"move_to_action_indices must match total_move_edges; got "
                f"{len(move_to_action_indices)} vs {total_move_edges}"
            )
    except Exception:
        pass

    return x_dict, edge_index_dict, edge_attr_dict, move_to_action_indices


def get_move_edge_mask(data):
    """
    Get a mask indicating which move edges belong to the current player.

    Args:
        data: HeteroData object

    Returns:
        Tensor of booleans, True for current player's legal moves
    """
    masks = []

    # Check both types of move edges
    for edge_type in [
        ("in_play", "move", "destination"),
        ("out_of_play", "move", "destination"),
    ]:
        if edge_type in data.edge_types:
            if hasattr(data[edge_type], "edge_attr"):
                # edge_attr has shape [num_edges, 1] where 1.0 = current player, 0.0 = opponent
                mask = data[edge_type].edge_attr.squeeze(-1) == 1.0
                masks.append(mask)

    if masks:
        return torch.cat(masks, dim=0)
    else:
        return torch.empty(0, dtype=torch.bool)


def ordered_unique_action_indices(move_to_action_indices):
    """Return ordered, deduplicated list of non-negative action indices.

    This preserves the order of the move edges as they appear in the
    `move_to_action_indices` sequence. It is intentionally *not* sorted; the
    model order depends on the PyG `data.edge_types` ordering and we must keep
    the first-appearance ordering for a correct mapping between local and
    global action indices.

    Args:
        move_to_action_indices: Iterable of ints (torch.Tensor or list)

    Returns:
        list of int: ordered unique non-negative action indices
    """
    seen = set()
    uniq = []
    # Support both torch.Tensor and Python list
    if hasattr(move_to_action_indices, "tolist"):
        seq = move_to_action_indices.tolist()
    else:
        seq = list(move_to_action_indices)

    for v in seq:
        if v >= 0 and v not in seen:
            uniq.append(int(v))
            seen.add(v)

    return uniq


if __name__ == "__main__":
    # Test the conversion
    logger = logging.getLogger(__name__)
    logger.info("Testing board_to_hetero_data...")
    import nokamute

    board = nokamute.Board()

    # Make a few moves
    for _ in range(3):
        moves = board.legal_moves()
        if moves:
            board.apply(moves[0])

    # Convert to HeteroData
    graph_dict = board.to_graph()
    data, move_to_action_indices = board_to_hetero_data(graph_dict)

    print(f"\nHeteroData structure:")
    print(f"  Node types: {data.node_types}")
    print(f"  Edge types: {data.edge_types}")
    print(f"\nNode counts:")
    for node_type in data.node_types:
        print(f"  {node_type}: {data[node_type].x.shape[0]} nodes")
    print(f"\nEdge counts:")
    for edge_type in data.edge_types:
        print(f"  {edge_type}: {data[edge_type].edge_index.shape[1]} edges")

    print(f"\nLegal moves: {len(move_to_action_indices)}")

    # Test move edge mask
    mask = get_move_edge_mask(data)
    print(f"Current player move edges: {mask.sum().item()}")

    print("\n✓ Conversion test passed!")
