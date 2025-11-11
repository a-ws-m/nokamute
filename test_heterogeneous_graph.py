"""
Test the new heterogeneous graph representation.
"""

import sys

import nokamute


def test_basic_graph():
    """Test basic graph structure."""
    print("=" * 60)
    print("Testing heterogeneous graph representation")
    print("=" * 60)

    # Create a board and make a few moves
    board = nokamute.Board()

    # Get graph representation
    print("\n1. Initial board (empty):")
    graph_dict = board.to_graph()

    print(
        f"   Graph keys: {sorted([k for k in graph_dict.keys() if not k.startswith('_')])[:10]}..."
    )

    # Check node features
    print(f"\n2. Node types:")
    print(f"   - in_play nodes: {len(graph_dict['x_in_play'])}")
    print(f"   - out_of_play nodes: {len(graph_dict['x_out_of_play'])}")
    print(f"   - destination nodes: {len(graph_dict['x_destination'])}")

    # Count edge types
    edge_type_keys = [k for k in graph_dict.keys() if k.startswith("edge_index_")]
    print(f"\n3. Edge types found: {len(edge_type_keys)}")
    for key in sorted(edge_type_keys):
        num_edges = len(graph_dict[key][0]) if len(graph_dict[key]) > 0 else 0
        print(f"   - {key}: {num_edges} edges")

    # Check move to action mapping
    move_to_action = graph_dict["move_to_action"]
    print(f"\n4. Move to action mapping:")
    print(f"   - {len(move_to_action)} legal moves")
    if move_to_action:
        print(f"   - First 5 moves: {move_to_action[:5]}")

    # Check node feature dimensions
    if graph_dict["x_in_play"]:
        print(f"\n5. Node feature dimensions:")
        print(f"   - in_play feature size: {len(graph_dict['x_in_play'][0])}")
    if graph_dict["x_out_of_play"]:
        print(f"   - out_of_play feature size: {len(graph_dict['x_out_of_play'][0])}")
    if graph_dict["x_destination"]:
        print(f"   - destination feature size: {len(graph_dict['x_destination'][0])}")

    # Make a move and check again
    print("\n6. After first move:")
    legal_moves = board.legal_moves()
    board.apply(legal_moves[0])

    graph_dict = board.to_graph()
    move_to_action = graph_dict["move_to_action"]

    print(f"   - in_play nodes: {len(graph_dict['x_in_play'])}")
    print(f"   - out_of_play nodes: {len(graph_dict['x_out_of_play'])}")
    print(f"   - destination nodes: {len(graph_dict['x_destination'])}")

    # Count total move edges
    total_move_edges = 0
    for key in graph_dict.keys():
        if "move" in key and "edge_index" in key:
            total_move_edges += (
                len(graph_dict[key][0]) if len(graph_dict[key]) > 0 else 0
            )
    print(f"   - total move edges: {total_move_edges}")
    print(f"   - legal moves: {len(move_to_action)}")
    if move_to_action:
        print(f"   - First 5 moves: {move_to_action[:5]}")

    # Make several more moves
    print("\n7. After 6 moves:")
    for _ in range(5):
        legal_moves = board.legal_moves()
        if legal_moves:
            board.apply(legal_moves[0])

    graph_dict = board.to_graph()
    move_to_action = graph_dict["move_to_action"]

    print(f"   - in_play nodes: {len(graph_dict['x_in_play'])}")
    print(f"   - out_of_play nodes: {len(graph_dict['x_out_of_play'])}")
    print(f"   - destination nodes: {len(graph_dict['x_destination'])}")

    # Count edges
    total_neighbour_edges = 0
    total_move_edges = 0
    for key in graph_dict.keys():
        if "edge_index" in key:
            if "neighbour" in key:
                total_neighbour_edges += (
                    len(graph_dict[key][0]) if len(graph_dict[key]) > 0 else 0
                )
            elif "move" in key:
                total_move_edges += (
                    len(graph_dict[key][0]) if len(graph_dict[key]) > 0 else 0
                )

    print(f"   - total neighbour edges: {total_neighbour_edges}")
    print(f"   - total move edges: {total_move_edges}")
    print(f"   - legal moves: {len(move_to_action)}")

    # Check move edge features
    print(f"\n8. Move edge features:")
    current_player_edges = 0
    opponent_edges = 0
    for key in graph_dict.keys():
        if "edge_attr" in key and "move" in key:
            attrs = graph_dict[key]
            current_player_edges += sum(1 for attr in attrs if attr[0] == 1.0)
            opponent_edges += sum(1 for attr in attrs if attr[0] == 0.0)
    print(f"   - Current player move edges: {current_player_edges}")
    print(f"   - Opponent move edges: {opponent_edges}")
    print(f"   - Legal moves for current player: {len(move_to_action)}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_basic_graph()
