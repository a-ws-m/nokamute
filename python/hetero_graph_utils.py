"""
Helper functions to convert board graph representation to PyTorch Geometric format.
"""

import torch
import torch_geometric.transforms as T
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
    from action_space import string_to_action

    data = HeteroData()

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

    # Convert move strings to action indices BEFORE ToUndirected
    move_to_action_list = graph_dict["move_to_action"]
    current_player_action_indices = torch.tensor(
        [string_to_action(move_str) for move_str in move_to_action_list],
        dtype=torch.long,
    )

    # Build full action index tensor that matches ALL move edges (current + opponent)
    # We need to match the order of edges in the graph with their action indices
    # Current player moves have action indices, opponent moves get -1 (will be masked)
    move_to_action_indices_list = []
    current_idx = 0

    for edge_type in data.edge_types:
        if "move" in edge_type:
            # Get edge attributes for this edge type
            attr_key = f"edge_attr_{edge_type[0]}_{edge_type[1]}_{edge_type[2]}"
            if attr_key in graph_dict:
                edge_attrs = graph_dict[attr_key]
                for attr in edge_attrs:
                    if attr[0] == 1.0:  # Current player's move
                        move_to_action_indices_list.append(
                            current_player_action_indices[current_idx].item()
                        )
                        current_idx += 1
                    else:  # Opponent's move
                        move_to_action_indices_list.append(-1)  # Will be masked out

    move_to_action_indices = torch.tensor(move_to_action_indices_list, dtype=torch.long)

    # Apply ToUndirected transform to add reverse edges
    # This ensures message passing works in both directions
    transform = T.ToUndirected()
    data = transform(data)

    # After ToUndirected, move edges are duplicated (forward + reverse)
    # We need to duplicate move_to_action_indices to match
    # The order is: [original edges, then reverse edges]
    move_to_action_indices = torch.cat([move_to_action_indices, move_to_action_indices])

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


if __name__ == "__main__":
    # Test the conversion
    print("Testing board_to_hetero_data...")
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
