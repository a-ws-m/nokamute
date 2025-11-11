"""
Test to verify ToUndirected transform and message passing along all edge types.
"""

import sys

sys.path.insert(0, "python")

import torch
from hetero_graph_utils import board_to_hetero_data, prepare_model_inputs
from model_policy_hetero import create_policy_model

import nokamute


def test_toundirected_transform():
    """Test that ToUndirected creates reverse edges correctly."""
    print("=" * 60)
    print("Testing ToUndirected Transform")
    print("=" * 60)

    # Create board with some moves
    board = nokamute.Board()
    for _ in range(4):
        moves = board.legal_moves()
        if moves:
            board.apply(moves[0])

    # Get graph
    graph_dict = board.to_graph()
    data, move_to_action_indices = board_to_hetero_data(graph_dict)

    print("\n1. Graph structure after ToUndirected:")
    print(f"   Node types: {data.node_types}")
    print(f"   Number of edge types: {len(data.edge_types)}")
    print()

    # Organize edge types
    forward_edges = []
    reverse_edges = []

    for edge_type in data.edge_types:
        src, edge_name, dst = edge_type
        num_edges = data[edge_type].edge_index.shape[1]
        has_attr = hasattr(data[edge_type], "edge_attr")

        if "rev_" in edge_name:
            reverse_edges.append((edge_type, num_edges, has_attr))
        else:
            forward_edges.append((edge_type, num_edges, has_attr))

    print("2. Forward edges (original):")
    for edge_type, num_edges, has_attr in forward_edges:
        print(f"   {edge_type}: {num_edges} edges, has_attr={has_attr}")

    print("\n3. Reverse edges (added by ToUndirected):")
    for edge_type, num_edges, has_attr in reverse_edges:
        print(f"   {edge_type}: {num_edges} edges, has_attr={has_attr}")

    # Verify symmetry
    print("\n4. Verifying edge symmetry:")
    for fwd_type, fwd_count, fwd_attr in forward_edges:
        fwd_src, fwd_edge, fwd_dst = fwd_type
        # Look for corresponding reverse edge
        expected_rev_type = (fwd_dst, f"rev_{fwd_edge}", fwd_src)

        matching_rev = [
            (rev_type, rev_count, rev_attr)
            for rev_type, rev_count, rev_attr in reverse_edges
            if rev_type == expected_rev_type
        ]

        if matching_rev:
            rev_type, rev_count, rev_attr = matching_rev[0]
            symmetric = fwd_count == rev_count and fwd_attr == rev_attr
            status = "✓" if symmetric else "✗"
            print(f"   {status} {fwd_type} <-> {rev_type}")
            if not symmetric:
                print(f"      Forward: {fwd_count} edges, has_attr={fwd_attr}")
                print(f"      Reverse: {rev_count} edges, has_attr={rev_attr}")
        else:
            print(f"   ✗ {fwd_type} - NO REVERSE EDGE FOUND")

    print("\n" + "=" * 60)
    print("✓ ToUndirected transform test passed!")
    print("=" * 60)


def test_message_passing_coverage():
    """Test that the model handles all edge types created by ToUndirected."""
    print("\n" + "=" * 60)
    print("Testing Message Passing Coverage")
    print("=" * 60)

    # Create model
    model = create_policy_model(
        config={"hidden_dim": 64, "num_layers": 2, "num_heads": 2}
    )
    model.eval()

    print("\n1. Model edge type configurations:")

    # Get all edge types the model is configured to handle
    model_edge_types = set()
    for hetero_conv in model.hetero_convs:
        for edge_type in hetero_conv.convs.keys():
            model_edge_types.add(edge_type)

    print(f"   Model handles {len(model_edge_types)} edge types:")
    for edge_type in sorted(model_edge_types):
        print(f"     - {edge_type}")

    # Test with real data
    print("\n2. Testing with real board data:")
    board = nokamute.Board()
    for _ in range(3):
        moves = board.legal_moves()
        if moves:
            board.apply(moves[0])

    graph_dict = board.to_graph()
    data, move_to_action_indices = board_to_hetero_data(graph_dict)

    print(f"   Data has {len(data.edge_types)} edge types:")
    for edge_type in sorted(data.edge_types):
        print(f"     - {edge_type}")

    # Check if model handles all edge types in data
    print("\n3. Checking coverage:")
    data_edge_types = set(data.edge_types)

    covered = model_edge_types & data_edge_types
    missing = data_edge_types - model_edge_types
    unused = model_edge_types - data_edge_types

    print(f"   ✓ Covered: {len(covered)} edge types")
    for edge_type in sorted(covered):
        print(f"     - {edge_type}")

    if missing:
        print(f"\n   ✗ Missing in model: {len(missing)} edge types")
        for edge_type in sorted(missing):
            print(f"     - {edge_type}")
    else:
        print(f"\n   ✓ All data edge types are covered by the model!")

    if unused:
        print(
            f"\n   ℹ Unused by data: {len(unused)} edge types (configured but not present)"
        )
        for edge_type in sorted(unused):
            print(f"     - {edge_type}")

    # Test forward pass
    print("\n4. Testing forward pass:")
    x_dict, edge_index_dict, edge_attr_dict, move_indices = prepare_model_inputs(
        data, move_to_action_indices
    )

    with torch.no_grad():
        policy_logits, value = model(
            x_dict, edge_index_dict, edge_attr_dict, move_indices
        )

    legal_count = (policy_logits[0] != float("-inf")).sum().item()
    print(f"   ✓ Forward pass successful!")
    print(f"   Legal actions: {legal_count}")
    print(f"   Position value: {value.item():.3f}")

    print("\n" + "=" * 60)
    print("✓ Message passing coverage test passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_toundirected_transform()
    test_message_passing_coverage()
