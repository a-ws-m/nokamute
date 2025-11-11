"""
Test the heterogeneous GNN model with real board data.
"""

import sys

import torch

sys.path.insert(0, "python")

from hetero_graph_utils import board_to_hetero_data, prepare_model_inputs
from model_policy_hetero import create_policy_model

import nokamute


def test_model_forward():
    """Test model forward pass with real board data."""
    print("=" * 60)
    print("Testing HiveGNNPolicyHetero model")
    print("=" * 60)

    # Create model (will be initialized on first forward pass)
    model = create_policy_model(
        config={
            "hidden_dim": 128,  # Smaller for testing
            "num_layers": 3,
            "num_heads": 4,
            "dropout": 0.1,
        }
    )
    model.eval()

    print(f"\n1. Model created")
    print(f"   Action space size: {model.num_actions}")
    print(
        f"   Note: Model uses lazy initialization - parameters will be created on first forward pass"
    )

    # Test with empty board
    print("\n2. Testing with empty board...")
    board = nokamute.Board()
    graph_dict = board.to_graph()
    data, move_to_action_indices = board_to_hetero_data(graph_dict)

    print(f"   Node types: {data.node_types}")
    print(f"   Edge types: {data.edge_types}")
    print(f"   Legal moves: {len(move_to_action_indices)}")

    # Prepare inputs
    x_dict, edge_index_dict, edge_attr_dict, move_indices = prepare_model_inputs(
        data, move_to_action_indices
    )

    # Forward pass
    with torch.no_grad():
        policy_logits, value = model(
            x_dict, edge_index_dict, edge_attr_dict, move_indices
        )

    # Print parameter count after initialization (skip uninitialized params)
    try:
        param_count = sum(p.numel() for p in model.parameters() if p.numel() > 0)
        print(f"   Model initialized with {param_count:,} parameters")
    except:
        print(f"   Model has lazy parameters (will initialize fully on GPU usage)")

    print(f"   Policy logits shape: {policy_logits.shape}")
    print(f"   Value shape: {value.shape}")
    print(f"   Value: {value.item():.3f}")

    # Check that legal moves have finite logits
    legal_actions_mask = policy_logits[0] != float("-inf")
    num_legal = legal_actions_mask.sum().item()
    print(f"   Legal actions (finite logits): {num_legal}")
    print(f"   Expected legal moves: {len(move_to_action_indices)}")

    assert num_legal == len(
        move_to_action_indices
    ), f"Mismatch: {num_legal} != {len(move_to_action_indices)}"

    # Test action selection
    action_idx, action_prob = model.select_action(
        x_dict, edge_index_dict, edge_attr_dict, move_indices, deterministic=True
    )
    print(f"   Selected action: {action_idx} with probability {action_prob:.4f}")

    # Test with board after a few moves
    print("\n3. Testing with board after 6 moves...")
    board = nokamute.Board()
    for _ in range(6):
        moves = board.legal_moves()
        if moves:
            board.apply(moves[0])

    graph_dict = board.to_graph()
    data, move_to_action_indices = board_to_hetero_data(graph_dict)

    print(f"   in_play nodes: {data['in_play'].x.shape[0]}")
    print(f"   out_of_play nodes: {data['out_of_play'].x.shape[0]}")
    print(f"   destination nodes: {data['destination'].x.shape[0]}")
    print(f"   Legal moves: {len(move_to_action_indices)}")

    # Count edges
    total_neighbour = 0
    total_move = 0
    for edge_type in data.edge_types:
        num_edges = data[edge_type].edge_index.shape[1]
        if "neighbour" in edge_type:
            total_neighbour += num_edges
        elif "move" in edge_type:
            total_move += num_edges

    print(f"   Total neighbour edges: {total_neighbour}")
    print(f"   Total move edges: {total_move}")

    # Prepare inputs
    x_dict, edge_index_dict, edge_attr_dict, move_indices = prepare_model_inputs(
        data, move_to_action_indices
    )

    # Forward pass
    with torch.no_grad():
        policy_logits, value = model(
            x_dict, edge_index_dict, edge_attr_dict, move_indices
        )

    print(f"   Policy logits shape: {policy_logits.shape}")
    print(f"   Value: {value.item():.3f}")

    legal_actions_mask = policy_logits[0] != float("-inf")
    num_legal = legal_actions_mask.sum().item()
    print(f"   Legal actions: {num_legal}")

    # Note: num_legal might be less than move_to_action_indices if some moves map to -1
    valid_move_indices = (move_to_action_indices >= 0).sum().item()
    print(f"   Valid move indices: {valid_move_indices}")

    assert (
        num_legal == valid_move_indices
    ), f"Mismatch: {num_legal} != {valid_move_indices}"

    # Test action probabilities
    probs = model.predict_action_probs(
        x_dict, edge_index_dict, edge_attr_dict, move_indices, temperature=1.0
    )
    print(
        f"   Probability sum over legal actions: {probs[0][legal_actions_mask].sum().item():.4f}"
    )
    print(f"   Max probability: {probs[0].max().item():.4f}")
    print(f"   Min legal probability: {probs[0][legal_actions_mask].min().item():.6f}")

    # Test sampling
    action_idx, action_prob = model.select_action(
        x_dict, edge_index_dict, edge_attr_dict, move_indices, deterministic=False
    )
    print(f"   Sampled action: {action_idx} with probability {action_prob:.4f}")

    print("\n" + "=" * 60)
    print("✓ All model tests passed!")
    print("=" * 60)


def test_batch_processing():
    """Test that the model can handle different board positions."""
    print("\n" + "=" * 60)
    print("Testing model with various board positions")
    print("=" * 60)

    model = create_policy_model(
        config={"hidden_dim": 64, "num_layers": 2, "num_heads": 2}
    )
    model.eval()

    test_cases = [
        ("Empty board", 0),
        ("After 1 move", 1),
        ("After 3 moves", 3),
        ("After 10 moves", 10),
    ]

    for name, num_moves in test_cases:
        board = nokamute.Board()
        for _ in range(num_moves):
            moves = board.legal_moves()
            if not moves:
                break
            board.apply(moves[0])

        graph_dict = board.to_graph()
        data, move_to_action_indices = board_to_hetero_data(graph_dict)
        x_dict, edge_index_dict, edge_attr_dict, move_indices = prepare_model_inputs(
            data, move_to_action_indices
        )

        with torch.no_grad():
            policy_logits, value = model(
                x_dict, edge_index_dict, edge_attr_dict, move_indices
            )

        legal_count = (policy_logits[0] != float("-inf")).sum().item()
        print(f"   {name}: {legal_count} legal actions, value={value.item():.3f}")

    print("\n✓ Batch processing test passed!")


if __name__ == "__main__":
    test_model_forward()
    test_batch_processing()
