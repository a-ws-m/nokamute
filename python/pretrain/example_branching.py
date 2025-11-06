#!/usr/bin/env python3
"""
Example script demonstrating evaluation matching with branching Markov chain.

This script shows how to:
1. Generate random positions using branching
2. Use the custom Dataset
3. Train a model to match Rust engine evaluations
"""

import sys
import os
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval_matching import (
    generate_random_positions_branching,
    EvaluationMatchingDataset,
    pretrain_eval_matching,
    normalize_evaluation,
)
from model import create_model
from torch_geometric.loader import DataLoader


def example_1_generate_positions():
    """Example 1: Generate random positions using branching."""
    print("\n" + "=" * 70)
    print("Example 1: Generating Random Positions with Branching Markov Chain")
    print("=" * 70)
    
    positions = generate_random_positions_branching(
        num_positions=100,
        aggression=3,
        max_depth=40,
        branch_probability=0.3,
        verbose=True,
    )
    
    print(f"\nGenerated {len(positions)} unique positions")
    
    # Analyze the positions
    evaluations = [eval_score for _, _, eval_score in positions]
    print(f"\nEvaluation statistics:")
    print(f"  Min: {min(evaluations):.2f}")
    print(f"  Max: {max(evaluations):.2f}")
    print(f"  Mean: {sum(evaluations) / len(evaluations):.2f}")
    print(f"  Std: {torch.tensor(evaluations).std().item():.2f}")
    
    # Show normalized versions
    normalized = [normalize_evaluation(e) for e in evaluations]
    print(f"\nNormalized evaluations (tanh scaling):")
    print(f"  Min: {min(normalized):.4f}")
    print(f"  Max: {max(normalized):.4f}")
    print(f"  Mean: {sum(normalized) / len(normalized):.4f}")
    
    # Sample positions
    print(f"\nSample positions:")
    for i in range(min(3, len(positions))):
        node_features, edge_index, eval_score = positions[i]
        print(f"  Position {i+1}:")
        print(f"    Nodes: {len(node_features)}")
        print(f"    Edges: {len(edge_index[0]) if edge_index else 0}")
        print(f"    Raw eval: {eval_score:.2f}")
        print(f"    Normalized: {normalize_evaluation(eval_score):.4f}")


def example_2_use_dataset():
    """Example 2: Use the custom Dataset with DataLoader."""
    print("\n" + "=" * 70)
    print("Example 2: Using EvaluationMatchingDataset with DataLoader")
    print("=" * 70)
    
    # Create dataset
    dataset = EvaluationMatchingDataset(
        num_positions=200,
        aggression=3,
        max_depth=40,
        branch_probability=0.3,
        scale=0.001,
        regenerate_each_epoch=True,
    )
    
    print(f"\nDataset created with {len(dataset)} positions")
    
    # Create DataLoader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Iterate through batches
    print(f"\nIterating through batches (batch_size=32):")
    for i, batch in enumerate(loader):
        print(f"  Batch {i+1}:")
        print(f"    Graphs in batch: {batch.num_graphs}")
        print(f"    Total nodes: {batch.x.shape[0]}")
        print(f"    Total edges: {batch.edge_index.shape[1]}")
        print(f"    Target values range: [{batch.y.min():.4f}, {batch.y.max():.4f}]")
        
        if i >= 2:  # Show only first 3 batches
            break
    
    # Test regeneration
    print(f"\nTesting position regeneration...")
    old_size = len(dataset)
    dataset.regenerate()
    new_size = len(dataset)
    print(f"  Dataset size before: {old_size}")
    print(f"  Dataset size after: {new_size}")
    print(f"  Regeneration successful: {new_size > 0}")


def example_3_train_model():
    """Example 3: Train a model using evaluation matching."""
    print("\n" + "=" * 70)
    print("Example 3: Training Model with Evaluation Matching")
    print("=" * 70)
    
    # Create model
    model_config = {
        "hidden_dim": 32,
        "num_layers": 3,
    }
    model = create_model(model_config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    print(f"\nModel configuration:")
    print(f"  Hidden dim: {model_config['hidden_dim']}")
    print(f"  Num layers: {model_config['num_layers']}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Device: {device}")
    
    # Train with evaluation matching
    print(f"\nStarting pre-training...")
    losses = pretrain_eval_matching(
        model=model,
        num_positions_per_epoch=100,  # Small for demo
        num_epochs=10,
        batch_size=32,
        device=device,
        aggression=3,
        max_depth=40,
        branch_probability=0.3,
        scale=0.001,
        regenerate_each_epoch=True,
        verbose=True,
    )
    
    print(f"\nTraining completed!")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    
    # Test model predictions
    print(f"\nTesting model predictions...")
    import nokamute
    
    model.eval()
    with torch.no_grad():
        # Test on a few random positions
        test_positions = generate_random_positions_branching(
            num_positions=5,
            aggression=3,
            max_depth=30,
            branch_probability=0.3,
            verbose=False,
        )
        
        print(f"\nModel predictions vs Rust engine evaluations:")
        print(f"{'Position':<12} {'Rust (raw)':<15} {'Rust (norm)':<15} {'GNN (norm)':<15} {'Diff':<10}")
        print("-" * 70)
        
        for i, (node_features, edge_index, rust_eval) in enumerate(test_positions):
            x = torch.tensor(node_features, dtype=torch.float32).to(device)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).to(device)
            
            # Get GNN prediction
            pred, _ = model(x, edge_index_tensor)
            pred_value = pred.item()
            
            # Normalized Rust evaluation
            rust_norm = normalize_evaluation(rust_eval)
            
            # Difference
            diff = abs(pred_value - rust_norm)
            
            print(f"Position {i+1:<5} {rust_eval:>10.2f}     {rust_norm:>10.4f}     {pred_value:>10.4f}     {diff:>8.4f}")


def example_4_compare_parameters():
    """Example 4: Compare different hyperparameters."""
    print("\n" + "=" * 70)
    print("Example 4: Comparing Branching Parameters")
    print("=" * 70)
    
    configurations = [
        {"branch_probability": 0.1, "max_depth": 30, "name": "Low branching"},
        {"branch_probability": 0.3, "max_depth": 40, "name": "Medium branching"},
        {"branch_probability": 0.5, "max_depth": 50, "name": "High branching"},
    ]
    
    print(f"\nGenerating positions with different configurations:")
    print(f"(All with num_positions=100, aggression=3)")
    print()
    
    for config in configurations:
        positions = generate_random_positions_branching(
            num_positions=100,
            aggression=3,
            max_depth=config["max_depth"],
            branch_probability=config["branch_probability"],
            verbose=False,
        )
        
        evaluations = [eval_score for _, _, eval_score in positions]
        depths = [len(node_features) for node_features, _, _ in positions]
        
        print(f"{config['name']} (p={config['branch_probability']}, max_depth={config['max_depth']}):")
        print(f"  Positions generated: {len(positions)}")
        print(f"  Avg nodes per position: {sum(depths) / len(depths):.1f}")
        print(f"  Eval std dev: {torch.tensor(evaluations).std().item():.2f}")
        print()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Evaluation Matching with Branching Markov Chain - Examples")
    print("=" * 70)
    
    try:
        # Run examples
        example_1_generate_positions()
        example_2_use_dataset()
        example_3_train_model()
        example_4_compare_parameters()
        
        print("\n" + "=" * 70)
        print("All examples completed successfully!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
