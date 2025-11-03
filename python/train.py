"""
Main training script for self-play learning with GNN.
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from model import create_model
from self_play import SelfPlayGame, prepare_training_data
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Batch, Data


def train_epoch(model, training_data, optimizer, batch_size=32, device="cpu"):
    """
    Train the model for one epoch.

    Args:
        model: GNN model
        training_data: List of (node_features, edge_index, target_value)
        optimizer: Optimizer
        batch_size: Batch size
        device: Device to train on

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    # Shuffle training data
    import random

    random.shuffle(training_data)

    # Create batches
    for i in range(0, len(training_data), batch_size):
        batch_data = training_data[i : i + batch_size]

        # Convert to PyG Data objects
        data_list = []
        targets = []

        for node_features, edge_index, target_value in batch_data:
            if len(node_features) == 0:
                continue

            x = torch.tensor(node_features, dtype=torch.float32)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)

            data = Data(x=x, edge_index=edge_index_tensor)
            data_list.append(data)
            targets.append(target_value)

        if len(data_list) == 0:
            continue

        # Create batch
        batch = Batch.from_data_list(data_list).to(device)
        targets = torch.tensor(targets, dtype=torch.float32).to(device).unsqueeze(1)

        # Forward pass
        optimizer.zero_grad()
        predictions, _ = model(batch.x, batch.edge_index, batch.batch)

        # Compute loss (MSE between predicted and target values)
        loss = F.mse_loss(predictions, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate_model(model, num_games=50, device="cpu"):
    """
    Evaluate model performance through self-play.

    Args:
        model: GNN model to evaluate
        num_games: Number of games to play
        device: Device to run on

    Returns:
        Statistics dictionary
    """
    player = SelfPlayGame(model=model, temperature=0.1, device=device)

    results = {"wins": 0, "losses": 0, "draws": 0, "avg_game_length": 0}
    total_moves = 0

    for _ in range(num_games):
        game_data, result = player.play_game()
        total_moves += len(game_data)

        if result > 0.5:
            results["wins"] += 1
        elif result < -0.5:
            results["losses"] += 1
        else:
            results["draws"] += 1

    results["avg_game_length"] = total_moves / num_games
    return results


def main():
    parser = argparse.ArgumentParser(description="Train Hive GNN with self-play")
    parser.add_argument("--games", type=int, default=100, help="Games per iteration")
    parser.add_argument(
        "--iterations", type=int, default=10, help="Training iterations"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per iteration")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument(
        "--num-layers", type=int, default=4, help="Number of GNN layers"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Move selection temperature"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="checkpoints",
        help="Model checkpoint directory",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Resume from checkpoint"
    )

    args = parser.parse_args()

    # Create checkpoint directory
    Path(args.model_path).mkdir(parents=True, exist_ok=True)

    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.model_path, "logs"))

    # Create model
    print("Creating model...")
    model_config = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
    }
    model = create_model(model_config).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_iteration = 0

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iteration = checkpoint.get("iteration", 0)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {args.device}")

    # Training loop
    for iteration in range(start_iteration, args.iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{args.iterations}")
        print(f"{'='*60}")

        # Generate self-play games
        print(f"\nGenerating {args.games} self-play games...")
        start_time = time.time()

        player = SelfPlayGame(
            model=model if iteration > 0 else None,
            temperature=args.temperature,
            device=args.device,
        )
        games = player.generate_games(args.games)

        gen_time = time.time() - start_time
        print(f"Generated {len(games)} games in {gen_time:.2f}s")

        # Prepare training data
        print("Preparing training data...")
        training_data = prepare_training_data(games)
        print(f"Training examples: {len(training_data)}")

        # Train for multiple epochs
        print(f"\nTraining for {args.epochs} epochs...")
        for epoch in range(args.epochs):
            loss = train_epoch(
                model,
                training_data,
                optimizer,
                batch_size=args.batch_size,
                device=args.device,
            )

            print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss:.6f}")

            # Log to tensorboard
            global_step = iteration * args.epochs + epoch
            writer.add_scalar("Loss/train", loss, global_step)

        # Evaluate model
        print("\nEvaluating model...")
        eval_stats = evaluate_model(model, num_games=20, device=args.device)
        print(f"Evaluation results: {eval_stats}")

        writer.add_scalar(
            "Eval/avg_game_length", eval_stats["avg_game_length"], iteration
        )
        writer.add_scalar("Eval/wins", eval_stats["wins"], iteration)
        writer.add_scalar("Eval/draws", eval_stats["draws"], iteration)

        # Save checkpoint
        checkpoint_path = os.path.join(args.model_path, f"model_iter_{iteration}.pt")
        torch.save(
            {
                "iteration": iteration + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model_config,
                "eval_stats": eval_stats,
            },
            checkpoint_path,
        )
        print(f"Saved checkpoint: {checkpoint_path}")

        # Save latest model
        latest_path = os.path.join(args.model_path, "model_latest.pt")
        torch.save(
            {
                "iteration": iteration + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model_config,
                "eval_stats": eval_stats,
            },
            latest_path,
        )

    writer.close()
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
