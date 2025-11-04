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
import nokamute
from torch_geometric.data import Batch, Data
from elo_tracker import EloTracker
from evaluate_vs_engine import evaluate_and_update_elo


def train_epoch(model, training_data, optimizer, batch_size=32, device="cpu", gamma=0.99):
    """
    Train the model for one epoch using temporal difference learning.

    Args:
        model: GNN model
        training_data: List of (node_features, edge_index, td_target)
        optimizer: Optimizer
        batch_size: Batch size
        device: Device to train on
        gamma: TD discount factor (default: 0.99)

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

        for node_features, edge_index, td_target in batch_data:
            if len(node_features) == 0:
                continue

            x = torch.tensor(node_features, dtype=torch.float32)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)

            data = Data(x=x, edge_index=edge_index_tensor)
            data_list.append(data)
            targets.append(td_target)

        if len(data_list) == 0:
            continue

        # Create batch
        batch = Batch.from_data_list(data_list).to(device)
        targets = torch.tensor(targets, dtype=torch.float32).to(device).unsqueeze(1)

        # Forward pass
        optimizer.zero_grad()
        predictions, _ = model(batch.x, batch.edge_index, batch.batch)

        # Compute TD loss (MSE between predicted value and TD target)
        loss = F.mse_loss(predictions, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def prepare_td_training_data(games, model, device="cpu", gamma=0.99):
    """
    Prepare training data using temporal difference learning.
    
    TD target: V(s_t) = r_t + gamma * V(s_{t+1})
    Where:
    - V(s_t) is the value of current state
    - r_t is the immediate reward (0 for non-terminal states)
    - V(s_{t+1}) is the value of next state (from model)
    
    For terminal states: V(s_terminal) = final_result
    
    Args:
        games: List of (game_data, result) tuples from self-play
        model: Current model to estimate future values
        device: Device to run model on
        gamma: Discount factor for future rewards
        
    Returns:
        training_examples: List of (node_features, edge_index, td_target)
    """
    training_examples = []
    model.eval()
    
    for game_data, final_result in games:
        if len(game_data) == 0:
            continue
            
        # Process positions in reverse (from end to start)
        # This allows us to compute TD targets efficiently
        next_value = None
        
        for position_idx in reversed(range(len(game_data))):
            board_graph, legal_moves, selected_move, player = game_data[position_idx]
            node_features, edge_index = board_graph
            
            # Skip empty boards
            if len(node_features) == 0:
                continue
            
            # Determine if this is the last position
            is_terminal = (position_idx == len(game_data) - 1)
            
            if is_terminal:
                # Terminal state: use final game result
                if player.name == "White":
                    td_target = final_result
                else:
                    td_target = -final_result
            else:
                # Non-terminal state: use TD target
                # If we haven't computed next value yet, get it from model
                if next_value is None:
                    # Get next position
                    next_board_graph, _, _, next_player = game_data[position_idx + 1]
                    next_node_features, next_edge_index = next_board_graph
                    
                    if len(next_node_features) > 0:
                        # Evaluate next position with model
                        with torch.no_grad():
                            x = torch.tensor(next_node_features, dtype=torch.float32).to(device)
                            edge_index_tensor = torch.tensor(next_edge_index, dtype=torch.long).to(device)
                            
                            # Create temporary batch with single item
                            data = Data(x=x, edge_index=edge_index_tensor)
                            batch = Batch.from_data_list([data]).to(device)
                            
                            prediction, _ = model(batch.x, batch.edge_index, batch.batch)
                            next_value_raw = prediction.item()
                            
                            # Flip sign if different player
                            if player.name == next_player.name:
                                # Same player (shouldn't happen in normal gameplay)
                                next_value = next_value_raw
                            else:
                                # Different player - negate value
                                next_value = -next_value_raw
                    else:
                        next_value = 0.0
                
                # TD target: immediate reward (0) + discounted next value
                # We use gamma * next_value because opponent's good position is bad for us
                td_target = gamma * next_value
            
            # Store training example
            training_examples.append((node_features, edge_index, td_target))
            
            # Current position becomes "next position" for previous iteration
            # Evaluate current position to use as next_value in next iteration
            if position_idx > 0:
                with torch.no_grad():
                    x = torch.tensor(node_features, dtype=torch.float32).to(device)
                    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).to(device)
                    
                    data = Data(x=x, edge_index=edge_index_tensor)
                    batch = Batch.from_data_list([data]).to(device)
                    
                    prediction, _ = model(batch.x, batch.edge_index, batch.batch)
                    next_value = prediction.item()
            else:
                next_value = None
    
    return training_examples


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
        "--num-layers", type=int, default=3, help="Number of GNN layers"
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
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=5,
        help="Evaluate against engine every N iterations",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=20,
        help="Number of games per evaluation",
    )
    parser.add_argument(
        "--eval-depths",
        type=int,
        nargs="+",
        default=[3],
        help="Engine depths to evaluate against",
    )
    parser.add_argument(
        "--elo-k-factor",
        type=int,
        default=32,
        help="K-factor for ELO rating system",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="TD learning discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--use-td",
        action="store_true",
        help="Use temporal difference learning instead of Monte Carlo returns",
    )
    parser.add_argument(
        "--enable-branching",
        action="store_true",
        help="Enable branching MCMC for parallel game generation",
    )
    parser.add_argument(
        "--branch-ratio",
        type=float,
        default=0.5,
        help="Fraction of games to generate from branch points (default: 0.5)",
    )

    args = parser.parse_args()

    # Create checkpoint directory
    Path(args.model_path).mkdir(parents=True, exist_ok=True)

    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.model_path, "logs"))

    # Initialize ELO tracker
    elo_path = os.path.join(args.model_path, "elo_history.json")
    elo_tracker = EloTracker(save_path=elo_path, k_factor=args.elo_k_factor)
    print(f"ELO tracker initialized at {elo_path}")

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
        if args.enable_branching:
            print(f"Using branching MCMC with branch ratio: {args.branch_ratio}")
        start_time = time.time()

        player = SelfPlayGame(
            model=model if iteration > 0 else None,
            temperature=args.temperature,
            device=args.device,
            enable_branching=args.enable_branching,
        )
        
        if args.enable_branching:
            games = player.generate_games_with_branching(
                num_games=args.games,
                branch_ratio=args.branch_ratio
            )
        else:
            games = player.generate_games(args.games)

        gen_time = time.time() - start_time
        print(f"Generated {len(games)} games in {gen_time:.2f}s")
        
        # Clear branch cache for next iteration (fresh start)
        if args.enable_branching:
            num_branches = len(player.branch_points)
            num_nodes = len(player.game_tree)
            print(f"Collected {num_branches} branch points from {num_nodes} unique positions")
            player.clear_branch_cache()

        # Prepare training data
        print("Preparing training data...")
        if args.use_td:
            print(f"Using TD learning with gamma={args.gamma}")
            training_data = prepare_td_training_data(
                games, model, device=args.device, gamma=args.gamma
            )
        else:
            print("Using Monte Carlo returns")
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
                gamma=args.gamma,
            )

            print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss:.6f}")

            # Log to tensorboard
            global_step = iteration * args.epochs + epoch
            writer.add_scalar("Loss/train", loss, global_step)

        # Evaluate model (self-play)
        print("\nEvaluating model (self-play)...")
        eval_stats = evaluate_model(model, num_games=20, device=args.device)
        print(f"Self-play evaluation results: {eval_stats}")

        writer.add_scalar(
            "Eval/avg_game_length", eval_stats["avg_game_length"], iteration
        )
        writer.add_scalar("Eval/wins", eval_stats["wins"], iteration)
        writer.add_scalar("Eval/draws", eval_stats["draws"], iteration)

        # Periodic evaluation against engine
        model_name = f"model_iter_{iteration}"
        if (iteration + 1) % args.eval_interval == 0 or iteration == 0:
            print("\n" + "=" * 60)
            print("EVALUATING AGAINST ENGINE")
            print("=" * 60)
            
            engine_eval_results = evaluate_and_update_elo(
                model=model,
                model_name=model_name,
                elo_tracker=elo_tracker,
                engine_depths=args.eval_depths,
                games_per_depth=args.eval_games,
                device=args.device,
                verbose=True,
            )
            
            # Log engine evaluation results
            for depth_key, results in engine_eval_results.items():
                depth = int(depth_key.split("_")[1])
                writer.add_scalar(
                    f"EngineEval/depth_{depth}_win_rate",
                    results.get("win_rate", 0),
                    iteration,
                )
                writer.add_scalar(
                    f"EngineEval/depth_{depth}_avg_moves",
                    results.get("avg_moves", 0),
                    iteration,
                )
            
            # Log current ELO rating
            current_elo = elo_tracker.get_rating(model_name)
            writer.add_scalar("ELO/model_rating", current_elo, iteration)
            
            print(f"\nCurrent ELO: {current_elo:.1f}")
            
            # Print leaderboard
            print("\nTop 10 Leaderboard:")
            for rank, (player, rating) in enumerate(elo_tracker.get_leaderboard(10), 1):
                print(f"  {rank}. {player}: {rating:.1f}")
            
            # Update eval_stats with engine results
            eval_stats["engine_eval"] = engine_eval_results
            eval_stats["elo_rating"] = current_elo

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
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    
    best_model = elo_tracker.get_best_model(prefix="model_iter_")
    if best_model:
        print(f"\nBest model: {best_model}")
        print(f"Best ELO: {elo_tracker.get_rating(best_model):.1f}")
        elo_tracker.print_stats(best_model)
    
    print("\nFinal Leaderboard:")
    for rank, (player, rating) in enumerate(elo_tracker.get_leaderboard(15), 1):
        print(f"  {rank}. {player}: {rating:.1f}")
    
    print(f"\nELO history saved to: {elo_path}")
    print(f"Model checkpoints saved to: {args.model_path}")


if __name__ == "__main__":
    main()
