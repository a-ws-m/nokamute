"""
Main training script for self-play learning with GNN.
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from model import create_model
from self_play import SelfPlayGame, prepare_training_data
from torch.utils.tensorboard import SummaryWriter
import nokamute
from torch_geometric.data import Batch, Data
from elo_tracker import EloTracker
from evaluate_vs_engine import evaluate_and_update_elo
from graph_utils import networkx_to_pyg, graph_hash

# Import pre-training modules
try:
    from pretrain.eval_matching import (
        generate_eval_matching_data,
        pretrain_eval_matching,
        save_eval_data,
        load_eval_data,
    )
    PRETRAIN_AVAILABLE = True
except ImportError:
    PRETRAIN_AVAILABLE = False
    print("Warning: Pre-training modules not available")


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


def prepare_td_training_data(games, model, device="cpu", gamma=0.99, batch_size=32):
    """
    Prepare training data using semi-gradient TD(0) learning.
    
    This implements the semi-gradient form where we update parameters based on:
    TD target: V(s_t) = r_t + gamma * V(s_{t+1})
    
    Where:
    - V(s_t) is the value of current state (trainable)
    - r_t is the immediate reward (0 for non-terminal states)
    - V(s_{t+1}) is the value of next state (from stored move evaluation or model)
    
    The "semi-gradient" aspect means we do NOT backpropagate through V(s_{t+1}),
    treating it as a fixed target. This is the standard approach in TD learning.
    
    For terminal states: V(s_terminal) = final_result
    
    This implementation uses stored move evaluations from self-play to avoid
    redundant computation. During self-play, we already evaluated the position
    resulting from each selected move. We reuse those cached values here.
    
    Optimization steps:
    1. Extract stored move values from game data
    2. Use stored values for TD target computation (no re-evaluation needed)
    3. Handle multiple results for the same position by averaging
    
    Args:
        games: List of (game_data, result, branch_id) tuples from self-play
               Each game_data contains (nx_graph, legal_moves, selected_move, player, pos_hash, move_value)
        model: Current model (kept for compatibility, may be used for fallback)
        device: Device to run model on (kept for compatibility)
        gamma: Discount factor for future rewards
        batch_size: Batch size (kept for compatibility)
        
    Returns:
        training_examples: List of (node_features, edge_index, td_target)
    """
    from graph_utils import networkx_to_pyg
    
    # Group games by branch_id for efficient processing
    branch_groups = {}  # branch_id -> list of (game_data, result)
    for game_data, final_result, branch_id in games:
        if branch_id not in branch_groups:
            branch_groups[branch_id] = []
        branch_groups[branch_id].append((game_data, final_result))
    
    print(f"  Games grouped into {len(branch_groups)} branches")
    
    # Phase 1: Extract stored move values and build index
    # Map (game_idx, position_idx) -> stored move_value (evaluation of position AFTER move)
    position_move_values = {}
    # Track all position data: pos_hash -> (node_features, edge_index)
    position_data = {}
    
    for game_idx, (game_data, final_result, branch_id) in enumerate(games):
        for position_idx, item in enumerate(game_data):
            # New format with move_value: (nx_graph, legal_moves, selected_move, player, pos_hash, move_value)
            if len(item) == 6:
                nx_graph, legal_moves, selected_move, player, pos_hash, move_value = item
            elif len(item) == 5:
                # Old format without move_value (fallback to None)
                nx_graph, legal_moves, selected_move, player, pos_hash = item
                move_value = None
            else:
                continue
            
            # Convert NetworkX graph to PyG format
            node_features, edge_index = networkx_to_pyg(nx_graph)
            
            # Skip empty boards
            if len(node_features) == 0:
                continue
            
            # Store position data if not seen before
            if pos_hash not in position_data:
                position_data[pos_hash] = (node_features, edge_index)
            
            # Store the move value for this specific game position
            position_move_values[(game_idx, position_idx)] = move_value
    
    print(f"  Using {len(position_move_values)} stored move evaluations (no re-evaluation needed)")
    
    # Phase 2: Compute TD targets using stored move values
    # Collect all (pos_hash, td_target) pairs to handle duplicates
    position_targets = {}  # pos_hash -> list of TD targets
    
    for game_idx, (game_data, final_result, branch_id) in enumerate(games):
        if len(game_data) == 0:
            continue
        
        # Process each position in the game
        for position_idx, item in enumerate(game_data):
            # Extract position info
            if len(item) == 6:
                nx_graph, legal_moves, selected_move, player, pos_hash, move_value = item
            elif len(item) == 5:
                nx_graph, legal_moves, selected_move, player, pos_hash = item
                move_value = None
            else:
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
                # Non-terminal state: compute TD target using stored move value
                # The move_value is the model's evaluation of the position AFTER applying the move
                # This is equivalent to V(s_{t+1}) from the next position's perspective
                
                next_move_value = position_move_values.get((game_idx, position_idx))
                
                if next_move_value is not None:
                    # move_value is from opponent's perspective (they evaluate the position they see)
                    # We need it from current player's perspective, so negate it
                    next_value = -next_move_value
                else:
                    # Fallback: no stored value available
                    next_value = 0.0
                
                # TD target: gamma * V(s_{t+1})
                td_target = gamma * next_value
            
            # Collect target for this position
            if pos_hash not in position_targets:
                position_targets[pos_hash] = []
            position_targets[pos_hash].append(td_target)
    
    # Phase 3: Average targets for positions with multiple outcomes
    training_examples = []
    for pos_hash, (node_features, edge_index) in position_data.items():
        if pos_hash in position_targets:
            targets = position_targets[pos_hash]
            avg_target = sum(targets) / len(targets)
            training_examples.append((node_features, edge_index, avg_target))
    
    return training_examples


def evaluate_model(model, num_games=50, device="cpu", max_moves=400):
    """
    Evaluate model performance through self-play.

    Args:
        model: GNN model to evaluate
        num_games: Number of games to play
        device: Device to run on
        max_moves: Maximum number of moves per game

    Returns:
        Statistics dictionary
    """
    player = SelfPlayGame(model=model, temperature=0.1, device=device, max_moves=max_moves)

    results = {"wins": 0, "losses": 0, "draws": 0, "avg_game_length": 0}
    total_moves = 0

    for _ in range(num_games):
        game_data, result, _ = player.play_game()
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
        "--iterations", type=int, default=1000, help="Training iterations"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per iteration")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden dimension")
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
        default=100,
        help="Evaluate against engine every N iterations",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Save model checkpoint every N iterations (default: 100)",
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
        help="Use semi-gradient TD(0) learning instead of Monte Carlo returns (defaults to 1 game, 1 epoch per iteration)",
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
    parser.add_argument(
        "--max-moves",
        type=int,
        default=400,
        help="Maximum number of moves per game before declaring a draw (default: 400)",
    )
    
    # Pre-training arguments
    parser.add_argument(
        "--pretrain",
        type=str,
        choices=["eval-matching", "none"],
        default="none",
        help="Pre-training method to use (runs pre-training only, then exits)",
    )
    parser.add_argument(
        "--pretrain-games",
        type=int,
        default=5,
        help="Number of games to generate for pre-training (default: 10)",
    )
    parser.add_argument(
        "--pretrain-depth",
        type=int,
        default=3,
        help="Engine search depth for pre-training data generation (default: 3)",
    )
    parser.add_argument(
        "--pretrain-epochs",
        type=int,
        default=100,
        help="Number of epochs for pre-training (default: 100)",
    )
    parser.add_argument(
        "--pretrain-randomness",
        type=float,
        default=0.1,
        help="Randomness rate for pre-training data generation (default: 0.1)",
    )
    parser.add_argument(
        "--pretrain-data-path",
        type=str,
        default=None,
        help="Path to saved pre-training data (auto-generated if not provided)",
    )

    args = parser.parse_args()

    # When using TD learning, default to 1 game and 1 epoch per iteration
    # unless explicitly overridden by user
    if args.use_td:
        # Check if user explicitly set these values
        import sys
        games_explicitly_set = '--games' in sys.argv
        epochs_explicitly_set = '--epochs' in sys.argv
        
        if not games_explicitly_set and args.games == 100:  # 100 is the default
            args.games = 1
            print("TD learning mode: defaulting to 1 game per iteration")
        
        if not epochs_explicitly_set and args.epochs == 10:  # 10 is the default
            args.epochs = 1
            print("TD learning mode: defaulting to 1 epoch per iteration")

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

    # Pre-training phase (standalone mode - exits after completion)
    if args.pretrain != "none":
        if not PRETRAIN_AVAILABLE:
            print("\nError: Pre-training requested but modules not available.")
            print("Make sure the pretrain package is installed correctly.")
            return
        
        print("\n" + "=" * 60)
        print("PRE-TRAINING MODE")
        print("=" * 60)
        
        if args.pretrain == "eval-matching":
            print(f"\nPre-training method: Evaluation Matching")
            print(f"Target: Match analytical BasicEvaluator scores")
            
            # Determine data file path
            if args.pretrain_data_path:
                data_file_path = args.pretrain_data_path
            else:
                # Auto-generate path based on settings
                data_filename = f"pretrain_eval_d{args.pretrain_depth}_g{args.pretrain_games}_r{int(args.pretrain_randomness*100)}.pkl"
                data_file_path = os.path.join(args.model_path, data_filename)
            
            # Load or generate pre-training data
            if os.path.exists(data_file_path):
                print(f"\nLoading existing pre-training data from: {data_file_path}")
                pretrain_data = load_eval_data(data_file_path)
            else:
                print(f"\nGenerating new pre-training data...")
                print(f"  Games: {args.pretrain_games}")
                print(f"  Engine depth: {args.pretrain_depth}")
                print(f"  Randomness rate: {args.pretrain_randomness}")
                print(f"  Max moves per game: {args.max_moves}")
                
                pretrain_data = generate_eval_matching_data(
                    num_games=args.pretrain_games,
                    depth=args.pretrain_depth,
                    aggression=3,
                    randomness_rate=args.pretrain_randomness,
                    max_moves=args.max_moves,
                    verbose=True,
                )
                
                # Always save generated data for future use
                print(f"\nSaving pre-training data to: {data_file_path}")
                save_eval_data(pretrain_data, data_file_path)
            
            # Pre-train the model
            print(f"\nPre-training for {args.pretrain_epochs} epochs...")
            print(f"Batch size: {args.batch_size}")
            print(f"Learning rate: {args.lr}")
            
            pretrain_losses = pretrain_eval_matching(
                model=model,
                training_data=pretrain_data,
                optimizer=optimizer,
                num_epochs=args.pretrain_epochs,
                batch_size=args.batch_size,
                device=args.device,
                verbose=True,
            )
            
            # Log to tensorboard
            for epoch, loss in enumerate(pretrain_losses):
                writer.add_scalar("Pretrain/Loss", loss, epoch)
            
            # Save pre-trained model
            pretrain_model_path = os.path.join(args.model_path, "model_pretrained.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "pretrain_method": args.pretrain,
                "pretrain_loss": pretrain_losses[-1],
                "config": model_config,
                "pretrain_games": args.pretrain_games,
                "pretrain_depth": args.pretrain_depth,
                "pretrain_epochs": args.pretrain_epochs,
            }, pretrain_model_path)
            print(f"\nPre-trained model saved to: {pretrain_model_path}")
            print(f"Final pre-training loss: {pretrain_losses[-1]:.6f}")
            
            # Evaluate pre-trained model against engine
            print("\nEvaluating pre-trained model against engine...")
            pretrain_eval_results = evaluate_and_update_elo(
                model=model,
                elo_tracker=elo_tracker,
                model_name="model_pretrained",
                engine_depths=args.eval_depths,
                games_per_depth=args.eval_games,
                device=args.device,
                verbose=True,
            )
            
            current_elo = elo_tracker.get_rating("model_pretrained")
            print(f"\nPre-trained model ELO: {current_elo:.1f}")
            writer.add_scalar("Pretrain/ELO", current_elo, 0)
            
            # Print results summary
            print("\n" + "=" * 60)
            print("PRE-TRAINING COMPLETE!")
            print("=" * 60)
            print(f"\nPre-trained model: {pretrain_model_path}")
            print(f"Training data: {data_file_path}")
            print(f"Final loss: {pretrain_losses[-1]:.6f}")
            print(f"ELO rating: {current_elo:.1f}")
            print("\nTo continue with self-play training, run:")
            print(f"  python train.py --resume {pretrain_model_path} --iterations N")
        
        writer.close()
        return  # Exit after pre-training

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
            max_moves=args.max_moves,
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
            print(f"Using semi-gradient TD(0) with gamma={args.gamma}")
            training_data = prepare_td_training_data(
                games, model, device=args.device, gamma=args.gamma, batch_size=args.batch_size
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

        # Initialize eval_stats for checkpoint saving
        eval_stats = {}

        # Periodic evaluation against engine and previous models
        # Only save named checkpoints and evaluate at eval_interval
        model_name = f"model_iter_{iteration}"
        should_evaluate = (iteration + 1) % args.eval_interval == 0
        
        if should_evaluate:
            print("\n" + "=" * 60)
            print("EVALUATION AT ITERATION {}".format(iteration))
            print("=" * 60)
            
            # Evaluate self-play performance
            print("\nEvaluating model (self-play)...")
            selfplay_stats = evaluate_model(model, num_games=args.eval_games, device=args.device, max_moves=args.max_moves)
            print(f"Self-play evaluation results: {selfplay_stats}")
            
            writer.add_scalar(
                "Eval/avg_game_length", selfplay_stats["avg_game_length"], iteration
            )
            writer.add_scalar("Eval/wins", selfplay_stats["wins"], iteration)
            writer.add_scalar("Eval/draws", selfplay_stats["draws"], iteration)
            
            # Evaluate against engine
            print("\n" + "-" * 60)
            print("EVALUATING AGAINST ENGINE")
            print("-" * 60)
            
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
            
            # Evaluate against previous model versions (if they exist)
            print("\n" + "-" * 60)
            print("EVALUATING AGAINST PREVIOUS MODELS")
            print("-" * 60)
            
            # Get all previous model iterations from ratings
            # Only include models that were saved at eval intervals (have named checkpoints)
            previous_models = []
            for name in elo_tracker.ratings.keys():
                if name.startswith("model_iter_") and name != model_name:
                    prev_iteration = int(name.split("_")[-1])
                    # Only include if it was saved at an eval interval
                    if (prev_iteration + 1) % args.eval_interval == 0:
                        previous_models.append(name)
            
            if previous_models:
                # Sort by iteration number and take the most recent ones
                previous_models.sort(key=lambda x: int(x.split("_")[-1]))
                
                # Evaluate against up to 3 most recent models
                recent_models = previous_models[-3:] if len(previous_models) > 3 else previous_models
                
                for prev_model_name in recent_models:
                    prev_iteration = int(prev_model_name.split("_")[-1])
                    prev_checkpoint = os.path.join(args.model_path, f"{prev_model_name}.pt")
                    
                    if os.path.exists(prev_checkpoint):
                        print(f"\nEvaluating against {prev_model_name} (iteration {prev_iteration})...")
                        
                        # Load previous model
                        prev_checkpoint_data = torch.load(prev_checkpoint)
                        prev_model = create_model(prev_checkpoint_data["config"]).to(args.device)
                        prev_model.load_state_dict(prev_checkpoint_data["model_state_dict"])
                        
                        # Play games between current and previous model
                        wins = 0
                        losses = 0
                        draws = 0
                        total_moves = 0
                        
                        from evaluate_vs_engine import MLPlayer, play_game
                        
                        current_player = MLPlayer(model, temperature=0.1, device=args.device)
                        prev_player = MLPlayer(prev_model, temperature=0.1, device=args.device)
                        
                        games_per_side = args.eval_games // 2
                        
                        # Play as player1 (Black)
                        for _ in range(games_per_side):
                            winner, num_moves, _ = play_game(
                                current_player, prev_player, max_moves=args.max_moves, verbose=False
                            )
                            total_moves += num_moves
                            
                            if winner == "player1":
                                wins += 1
                            elif winner == "player2":
                                losses += 1
                            else:
                                draws += 1
                        
                        # Play as player2 (White)
                        for _ in range(games_per_side):
                            winner, num_moves, _ = play_game(
                                prev_player, current_player, max_moves=args.max_moves, verbose=False
                            )
                            total_moves += num_moves
                            
                            if winner == "player2":
                                wins += 1
                            elif winner == "player1":
                                losses += 1
                            else:
                                draws += 1
                        
                        # Update ELO based on match results
                        total_games = wins + losses + draws
                        score = (wins + 0.5 * draws) / total_games
                        
                        elo_tracker.update_ratings(
                            model_name,
                            prev_model_name,
                            score,
                            game_metadata={
                                "wins": wins,
                                "losses": losses,
                                "draws": draws,
                                "avg_moves": total_moves / total_games,
                            },
                        )
                        
                        print(f"  vs {prev_model_name}: W={wins} L={losses} D={draws} "
                              f"(win rate: {wins/total_games:.1%})")
                        
                        # Log to tensorboard
                        writer.add_scalar(
                            f"PrevModelEval/vs_iter_{prev_iteration}_win_rate",
                            wins / total_games,
                            iteration,
                        )
                        writer.add_scalar(
                            f"PrevModelEval/vs_iter_{prev_iteration}_avg_moves",
                            total_moves / total_games,
                            iteration,
                        )
            else:
                print("No previous models to evaluate against.")
            
            # Log current ELO rating
            current_elo = elo_tracker.get_rating(model_name)
            writer.add_scalar("ELO/model_rating", current_elo, iteration)
            
            print("\n" + "-" * 60)
            print(f"Current ELO: {current_elo:.1f}")
            
            # Print leaderboard
            print("\nTop 10 Leaderboard:")
            for rank, (player, rating) in enumerate(elo_tracker.get_leaderboard(10), 1):
                print(f"  {rank}. {player}: {rating:.1f}")
            
            # Update eval_stats with results
            eval_stats["selfplay"] = selfplay_stats
            eval_stats["engine_eval"] = engine_eval_results
            eval_stats["elo_rating"] = current_elo
            
            # Save named checkpoint (only at eval intervals)
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

        # Save latest model at specified intervals (overwriting each time)
        if (iteration + 1) % args.save_interval == 0:
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
            print(f"Saved latest model: {latest_path}")

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
