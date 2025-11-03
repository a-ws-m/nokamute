"""
Utility functions for the Hive GNN training pipeline.
"""

import json
import os

import torch

import nokamute


def save_model(model, optimizer, config, path, metadata=None):
    """
    Save model checkpoint.

    Args:
        model: PyTorch model
        optimizer: Optimizer
        config: Model configuration dictionary
        path: Path to save checkpoint
        metadata: Additional metadata to save
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
    }

    if metadata is not None:
        checkpoint["metadata"] = metadata

    torch.save(checkpoint, path)
    print(f"Model saved to {path}")


def load_model(model_class, path, device="cpu"):
    """
    Load model checkpoint.

    Args:
        model_class: Model class to instantiate
        path: Path to checkpoint
        device: Device to load model on

    Returns:
        model, optimizer, config, metadata
    """
    checkpoint = torch.load(path, map_location=device)

    config = checkpoint.get("config", {})
    model = model_class(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer = None
    if "optimizer_state_dict" in checkpoint:
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    metadata = checkpoint.get("metadata", {})

    return model, optimizer, config, metadata


def print_board_state(board):
    """
    Print a human-readable representation of the board.

    Args:
        board: Nokamute board object
    """
    pieces = board.get_pieces()

    print(f"\nTurn {board.turn_num()}: {board.to_move().name()} to move")
    print(f"Pieces on board: {len(pieces)}")

    if len(pieces) > 0:
        print("\nPieces:")
        for hex_pos, color, bug, height in pieces:
            print(f"  {color} {bug} at hex {hex_pos} (height {height})")

    winner = board.get_winner()
    if winner:
        print(f"\nGame over! Winner: {winner}")


def board_to_feature_vector(board, max_pieces=28):
    """
    Convert board to a fixed-size feature vector (alternative to graph).

    Args:
        board: Nokamute board object
        max_pieces: Maximum number of pieces

    Returns:
        Feature vector
    """
    pieces = board.get_pieces()

    # Initialize feature vector
    # For each piece: [hex_x, hex_y, color, bug_type, height]
    features = []

    bug_to_idx = {
        "queen": 0,
        "grasshopper": 1,
        "spider": 2,
        "ant": 3,
        "beetle": 4,
        "mosquito": 5,
        "ladybug": 6,
        "pillbug": 7,
    }

    for hex_pos, color, bug, height in pieces[:max_pieces]:
        # Simplified hex coordinates (just use hex value)
        hex_x = hex_pos % 32
        hex_y = hex_pos // 32
        color_val = 0 if color == "White" else 1
        bug_val = bug_to_idx.get(bug.lower(), 0)

        features.extend([hex_x, hex_y, color_val, bug_val, height])

    # Pad if fewer pieces than max
    while len(features) < max_pieces * 5:
        features.extend([0, 0, 0, 0, 0])

    return features[: max_pieces * 5]


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_training_dirs(base_dir="experiments"):
    """
    Create directory structure for training.

    Args:
        base_dir: Base directory for experiments

    Returns:
        Dictionary with directory paths
    """
    import time

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"exp_{timestamp}")

    dirs = {
        "exp_dir": exp_dir,
        "checkpoints": os.path.join(exp_dir, "checkpoints"),
        "logs": os.path.join(exp_dir, "logs"),
        "games": os.path.join(exp_dir, "games"),
    }

    for path in dirs.values():
        os.makedirs(path, exist_ok=True)

    return dirs


def save_config(config, path):
    """
    Save configuration to JSON file.

    Args:
        config: Configuration dictionary
        path: Path to save JSON file
    """
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def load_config(path):
    """
    Load configuration from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Configuration dictionary
    """
    with open(path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")

    # Test board state
    board = nokamute.Board()
    print_board_state(board)

    # Test feature vector conversion
    features = board_to_feature_vector(board)
    print(f"\nFeature vector length: {len(features)}")

    # Test directory setup
    dirs = setup_training_dirs("test_exp")
    print(f"\nCreated directories: {dirs}")

    print("\nUtility tests passed!")
