"""
League-based competitive self-play training for Hive.

Implements the approach from:
"Minimax Exploiter: A Data Efficient Approach for Competitive Self-Play"

This script orchestrates training of three agent archetypes:
1. Main Agent: Learns robust strategies via PFSP
2. Main Exploiter: Learns counter-strategies against Main Agent
3. League Exploiter: Learns counter-strategies against entire league
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from elo_tracker import EloTracker
from evaluate_vs_engine import evaluate_and_update_elo
from league import (
    AgentArchetype,
    ExploiterAgent,
    LeagueConfig,
    LeagueManager,
    LeagueTracker,
    prepare_exploiter_training_data,
)
from model import create_model
from model_policy import create_policy_model
from model_policy_hetero import create_policy_model as create_policy_model_hetero
from self_play import SelfPlayGame, prepare_training_data
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def create_model_by_type(model_type, model_config):
    """
    Create model based on type selection.

    Args:
        model_type: "value", "policy", or "policy_hetero"
        model_config: Model configuration dict

    Returns:
        Model instance
    """
    if model_type == "policy":
        return create_policy_model(model_config)
    elif model_type == "policy_hetero":
        return create_policy_model_hetero(model_config)
    else:
        return create_model(model_config)


def train_epoch_standard(model, training_data, optimizer, batch_size=32, device="cpu"):
    """
    Train model for one epoch using standard supervised learning on game outcomes.

    Args:
        model: GNN model (can be heterogeneous policy model)
        training_data: List of (hetero_data, move_to_action_indices, target_value) for heterogeneous models
        optimizer: Optimizer
        batch_size: Batch size
        device: Device

    Returns:
        Average loss
    """
    from hetero_graph_utils import prepare_model_inputs
    from torch_geometric.data import HeteroData

    model.train()
    total_loss = 0
    num_batches = 0

    # Check if this is a heterogeneous model
    is_hetero = hasattr(model, "node_embedding")

    if is_hetero:
        # Handle heterogeneous data - process one at a time (batching hetero graphs is complex)
        for hetero_data, move_to_action_indices, target in tqdm(
            training_data, desc="Training", leave=False, unit="sample"
        ):
            if not isinstance(hetero_data, HeteroData):
                continue

            # Move to device
            hetero_data = hetero_data.to(device)
            move_to_action_indices = move_to_action_indices.to(device)

            # Prepare inputs
            x_dict, edge_index_dict, edge_attr_dict, _ = prepare_model_inputs(
                hetero_data, move_to_action_indices
            )

            optimizer.zero_grad()

            # Forward pass - heterogeneous policy model
            _, value = model(
                x_dict, edge_index_dict, edge_attr_dict, move_to_action_indices
            )

            prediction = value.squeeze()
            target_tensor = torch.tensor([target], dtype=torch.float32, device=device)

            # MSE loss
            loss = F.mse_loss(prediction, target_tensor)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
    else:
        # Handle homogeneous data (old format - shouldn't be used anymore)
        data_list = []
        for item in training_data:
            if len(item) == 3:
                node_features, edge_index, target = item
                if len(node_features) == 0:
                    continue

                x = torch.tensor(node_features, dtype=torch.float32)
                edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)

                data = Data(x=x, edge_index=edge_index_tensor)
                data.y = torch.tensor([target], dtype=torch.float32)
                data_list.append(data)

        if len(data_list) == 0:
            return 0.0

        loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

        for batch in tqdm(loader, desc="Training batches", leave=False, unit="batch"):
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass
            output1, output2 = model(batch.x, batch.edge_index, batch.batch)

            # Check output shapes to determine which is the value
            if output1.shape[-1] == 1 or output1.dim() == 1:
                predictions = output1.squeeze()
            else:
                predictions = output2.squeeze()

            targets = batch.y

            # MSE loss
            loss = F.mse_loss(predictions, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def train_main_agent(
    league_manager: LeagueManager,
    league_tracker: LeagueTracker,
    config: LeagueConfig,
    iteration: int,
    args=None,
):
    """
    Train the Main Agent for one iteration using PFSP.

    Args:
        league_manager: League manager
        league_tracker: Performance tracker
        config: League configuration
        iteration: Current training iteration
    """
    print(f"\n{'='*60}")
    print(f"Training Main Agent (Iteration {iteration})")
    print(f"{'='*60}")

    # Load current Main Agent model
    agent = league_manager.current_main_agent
    checkpoint = torch.load(agent.model_path, map_location=config.device)
    model_type = checkpoint.get(
        "model_type", "policy"
    )  # Default to policy (11x faster than value model)
    model = create_model_by_type(model_type, checkpoint.get("config", {})).to(
        config.device
    )

    # Handle torch.compile() state dict keys (strip _orig_mod. prefix)
    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.main_agent_lr)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Generate self-play games using PFSP opponent selection
    print(f"\nGenerating {config.main_agent_games_per_iter} self-play games...")
    games = []
    opponent_sample_counts = {}

    player = SelfPlayGame(
        model=model,
        temperature=config.main_agent_temperature,
        device=config.device,
        enable_branching=config.enable_branching,
        max_moves=config.max_moves,
        use_amp=args.use_amp if args else False,
        cache_graphs=args.cache_graphs if args else True,
        inference_batch_size=config.inference_batch_size,
    )

    for game_idx in tqdm(
        range(config.main_agent_games_per_iter), desc="Generating games", unit="game"
    ):
        # Sample opponent using PFSP
        opponent_agent = league_manager.sample_opponent_for_main_agent()
        opponent_sample_counts[opponent_agent.name] = (
            opponent_sample_counts.get(opponent_agent.name, 0) + 1
        )

        # Load opponent model
        opponent_checkpoint = torch.load(
            opponent_agent.model_path, map_location=config.device
        )
        opponent_model_type = opponent_checkpoint.get("model_type", "policy")
        opponent_model = create_model_by_type(
            opponent_model_type, opponent_checkpoint.get("config", {})
        ).to(config.device)

        # Handle torch.compile() state dict keys
        opponent_state_dict = opponent_checkpoint["model_state_dict"]
        if any(key.startswith("_orig_mod.") for key in opponent_state_dict.keys()):
            opponent_state_dict = {
                k.replace("_orig_mod.", ""): v for k, v in opponent_state_dict.items()
            }
        opponent_model.load_state_dict(opponent_state_dict)
        opponent_model.eval()

        # Play game (alternating who plays as White)
        main_is_white = game_idx % 2 == 0
        game_data, result, branch_id = player.play_game()

        # Record result
        if main_is_white:
            main_result = result
        else:
            main_result = -result

        agent.record_game_result(opponent_agent.name, main_result)
        opponent_agent.record_game_result(agent.name, -main_result)

        # Log game
        league_tracker.log_game_result(
            iteration, agent, opponent_agent, main_result, len(game_data)
        )

        games.append((game_data, result, branch_id))

    # Log PFSP statistics
    league_tracker.log_pfsp_stats(iteration, opponent_sample_counts)

    # Prepare training data
    print("Preparing training data...")
    training_data = prepare_training_data(games)
    print(f"Training examples: {len(training_data)}")

    # Train for multiple epochs
    print(f"\nTraining for {config.main_agent_epochs} epochs...")
    for epoch in tqdm(
        range(config.main_agent_epochs), desc="Training epochs", unit="epoch"
    ):
        loss = train_epoch_standard(
            model,
            training_data,
            optimizer,
            batch_size=config.main_agent_batch_size,
            device=config.device,
        )

        # Log metrics
        league_tracker.log_training_metrics(
            iteration * config.main_agent_epochs + epoch,
            agent.name,
            loss,
            optimizer.param_groups[0]["lr"],
            epoch=epoch,
        )

        if (epoch + 1) % 5 == 0 or epoch == 0:
            tqdm.write(
                f"  Epoch {epoch+1}/{config.main_agent_epochs}: Loss = {loss:.6f}"
            )

    # Update Main Agent
    model_config = checkpoint.get("config", {})
    updated_agent = league_manager.update_main_agent(model, optimizer, model_config)

    # Log performance
    league_tracker.log_agent_performance(iteration, updated_agent)

    return updated_agent


def train_main_exploiter(
    league_manager: LeagueManager,
    league_tracker: LeagueTracker,
    config: LeagueConfig,
    iteration: int,
):
    """
    Train the Main Exploiter against the current Main Agent.

    Uses minimax reward shaping for efficient exploitation.

    Args:
        league_manager: League manager
        league_tracker: Performance tracker
        config: League configuration
        iteration: Current training iteration
    """
    print(f"\n{'='*60}")
    print(f"Training Main Exploiter (Iteration {iteration})")
    print(f"{'='*60}")

    exploiter = league_manager.current_main_exploiter
    if exploiter is None:
        print("No Main Exploiter to train")
        return None

    # Load exploiter model
    checkpoint = torch.load(exploiter.model_path, map_location=config.device)
    exploiter_model_type = checkpoint.get("model_type", "policy")
    model = create_model_by_type(exploiter_model_type, checkpoint.get("config", {})).to(
        config.device
    )

    # Handle torch.compile() state dict keys
    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.exploiter_lr)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load target (Main Agent) model
    main_agent = league_manager.current_main_agent
    main_checkpoint = torch.load(main_agent.model_path, map_location=config.device)
    main_model_type = main_checkpoint.get("model_type", "policy")
    main_model = create_model_by_type(
        main_model_type, main_checkpoint.get("config", {})
    ).to(config.device)

    # Handle torch.compile() state dict keys
    main_state_dict = main_checkpoint["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in main_state_dict.keys()):
        main_state_dict = {
            k.replace("_orig_mod.", ""): v for k, v in main_state_dict.items()
        }
    main_model.load_state_dict(main_state_dict)
    main_model.eval()

    # Create exploiter agent
    exploiter_player = ExploiterAgent(
        model=model,
        opponent_model=main_model,
        device=config.device,
        temperature=config.exploiter_temperature,
        minimax_reward_weight=config.minimax_reward_weight,
        gamma=config.minimax_gamma,
        enable_branching=config.enable_branching,
        max_moves=config.max_moves,
        inference_batch_size=config.inference_batch_size,
    )

    # Generate training games with minimax rewards
    print(
        f"\nGenerating {config.exploiter_games_per_iter} exploiter games with minimax rewards..."
    )
    games = exploiter_player.generate_training_data(
        num_games=config.exploiter_games_per_iter,
        exploiter_is_white=True,  # Alternate in production
    )

    # Record results
    for game_data, result, _ in games:
        # Result is from White's perspective
        exploiter_result = result  # Exploiter is White
        main_result = -result

        exploiter.record_game_result(main_agent.name, exploiter_result)
        main_agent.record_game_result(exploiter.name, main_result)

        league_tracker.log_game_result(
            iteration, exploiter, main_agent, exploiter_result, len(game_data)
        )

    # Prepare training data with minimax rewards
    print("Preparing exploiter training data...")
    training_data = prepare_exploiter_training_data(
        games, minimax_reward_weight=config.minimax_reward_weight
    )
    print(f"Training examples: {len(training_data)}")

    # Train for multiple epochs
    print(f"\nTraining for {config.exploiter_epochs} epochs...")
    for epoch in tqdm(
        range(config.exploiter_epochs), desc="Training epochs", unit="epoch"
    ):
        loss = train_epoch_standard(
            model,
            training_data,
            optimizer,
            batch_size=config.exploiter_batch_size,
            device=config.device,
        )

        exploiter.training_epochs += 1

        # Log metrics
        league_tracker.log_training_metrics(
            iteration * config.exploiter_epochs + epoch,
            exploiter.name,
            loss,
            optimizer.param_groups[0]["lr"],
            epoch=epoch,
        )

        if (epoch + 1) % 5 == 0 or epoch == 0:
            tqdm.write(
                f"  Epoch {epoch+1}/{config.exploiter_epochs}: Loss = {loss:.6f}"
            )

    # Save updated exploiter
    model_config = checkpoint.get("config", {})

    # Get state dict and strip torch.compile() prefix if present
    state_dict = model.state_dict()
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    torch.save(
        {
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model_config,
            "model_type": model_config.get(
                "model_type", "policy"
            ),  # Store at top level for easy access
            "iteration": iteration,
            "target_agent": main_agent.name,
        },
        exploiter.model_path,
    )  # Check convergence
    win_rate_vs_main = exploiter.get_win_rate_vs(main_agent.name)
    print(f"\nExploiter win rate vs {main_agent.name}: {win_rate_vs_main:.2%}")

    # Get recent games for convergence check
    recent_games = []
    for opp_name, wins in exploiter.matchup_wins.items():
        losses = exploiter.matchup_losses[opp_name]
        draws = exploiter.matchup_draws[opp_name]
        total = wins + losses + draws

        # Approximate recent games (simplified)
        for _ in range(wins):
            recent_games.append((opp_name, 1.0))
        for _ in range(losses):
            recent_games.append((opp_name, -1.0))
        for _ in range(draws):
            recent_games.append((opp_name, 0.0))

    is_converged = league_manager.check_exploiter_convergence(exploiter, recent_games)
    exploiter.is_converged = is_converged

    # Log convergence
    league_tracker.log_exploiter_convergence(
        iteration,
        exploiter,
        win_rate_vs_main,
        config.main_exploiter_convergence_threshold,
    )

    if is_converged:
        print(
            f"✓ Main Exploiter CONVERGED! (WR: {win_rate_vs_main:.2%} >= {config.main_exploiter_convergence_threshold:.2%})"
        )

    # Log performance
    league_tracker.log_agent_performance(iteration, exploiter)

    return exploiter


def main():
    parser = argparse.ArgumentParser(
        description="League-based competitive self-play training"
    )

    # League configuration
    parser.add_argument(
        "--config", type=str, help="Path to league config file (optional)"
    )

    # Training schedule
    parser.add_argument(
        "--iterations", type=int, default=1000, help="Total training iterations"
    )
    parser.add_argument(
        "--main-games", type=int, default=10, help="Games per Main Agent iteration"
    )
    parser.add_argument(
        "--exploiter-games", type=int, default=20, help="Games per Exploiter iteration"
    )

    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument(
        "--num-layers", type=int, default=4, help="Number of GNN layers"
    )

    # Paths
    parser.add_argument(
        "--save-dir",
        type=str,
        default="league_checkpoints",
        help="Directory for saving league state",
    )
    parser.add_argument(
        "--resume", type=str, help="Resume from league checkpoint directory"
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        help="Path to pretrained model for initialization",
    )

    # Evaluation
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=50,
        help="Evaluate against engine every N iterations",
    )
    parser.add_argument(
        "--eval-games", type=int, default=20, help="Games per engine evaluation"
    )
    parser.add_argument(
        "--eval-depths",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Engine depths to evaluate against",
    )

    # Device
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Model type
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["value", "policy"],
        default="policy",
        help="Model architecture: 'value' (evaluates each move) or 'policy' (fixed action space). "
        "Benchmark shows policy is ~11x faster for game generation.",
    )

    # Performance optimization arguments
    parser.add_argument(
        "--use-compile",
        action="store_true",
        default=True,
        help="Use torch.compile() for model inference speedup (default: True, GPU recommended)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_false",
        dest="use_compile",
        help="Disable torch.compile() (useful for debugging or CPU-only systems)",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use Automatic Mixed Precision for faster GPU inference",
    )
    parser.add_argument(
        "--no-cache-graphs",
        action="store_false",
        dest="cache_graphs",
        default=True,
        help="Disable graph caching",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=None,
        help="Batch size for position evaluation during game generation. "
        "None = evaluate all positions in single batch. "
        "Set to your GPU's max capacity (e.g., 256, 512, 1024) for optimal performance.",
    )

    args = parser.parse_args()

    # Create or load league configuration
    if args.config:
        # Load from file (implement as needed)
        config = LeagueConfig()
    else:
        config = LeagueConfig(
            main_agent_games_per_iter=args.main_games,
            exploiter_games_per_iter=args.exploiter_games,
            device=args.device,
            inference_batch_size=args.inference_batch_size,
        )

    print("\n" + "=" * 60)
    print("LEAGUE TRAINING SYSTEM")
    print("=" * 60)
    print(f"Device: {config.device}")
    print(f"Model type: {args.model_type}")
    print(f"Main Agent games/iter: {config.main_agent_games_per_iter}")
    print(f"Exploiter games/iter: {config.exploiter_games_per_iter}")
    print(f"Minimax reward weight: {config.minimax_reward_weight}")
    print(f"PFSP exponent: {config.pfsp_exponent}")
    if config.inference_batch_size is not None:
        print(f"Inference batch size: {config.inference_batch_size}")
    else:
        print(f"Inference batch size: Dynamic (all positions in single batch)")

    # Setup directories
    save_dir = Path(args.resume) if args.resume else Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize league manager and tracker
    league_manager = LeagueManager(config, str(save_dir))
    league_tracker = LeagueTracker(
        str(save_dir / "logs"), elo_save_path=str(save_dir / "elo_ratings.json")
    )

    # Initialize or resume Main Agent
    model_config = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "model_type": args.model_type,  # Store model type for future loading
    }

    if league_manager.current_main_agent is None:
        print("\nInitializing Main Agent...")
        model = create_model_by_type(args.model_type, model_config).to(config.device)

        # Load pretrained weights if specified
        if args.pretrained_model:
            print(f"Loading pretrained model from {args.pretrained_model}...")
            pretrained = torch.load(args.pretrained_model, map_location=config.device)
            model.load_state_dict(pretrained["model_state_dict"])

        # Create optimizer before compiling (optimizer needs to see original parameters)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.main_agent_lr)

        # Initialize agent with uncompiled model first (saves clean state dict)
        league_manager.initialize_main_agent(model, optimizer, model_config)

        # Apply torch.compile() for faster inference AFTER saving initial checkpoint
        if args.use_compile:
            if config.device == "cpu":
                print("Note: torch.compile() on CPU may not provide speedup")
            else:
                try:
                    print("Compiling model with torch.compile()...")
                    model = torch.compile(model, mode="reduce-overhead")
                    print("✓ Model compiled successfully!")
                except Exception as e:
                    print(f"Warning: Could not compile model: {type(e).__name__}")
    else:
        print(f"\nResuming from iteration {league_manager.iteration}")

    # Main training loop
    start_iteration = league_manager.iteration

    for iteration in tqdm(
        range(start_iteration, args.iterations), desc="League iterations", unit="iter"
    ):
        league_manager.iteration = iteration

        tqdm.write(f"\n{'='*80}")
        tqdm.write(f"LEAGUE ITERATION {iteration + 1}/{args.iterations}")
        tqdm.write(f"{'='*80}")

        # 1. Train Main Agent
        if iteration % config.main_agent_update_interval == 0:
            main_agent = train_main_agent(
                league_manager, league_tracker, config, iteration, args
            )

        # 2. Spawn new Main Exploiter if needed
        if iteration > 0 and iteration % config.main_exploiter_spawn_interval == 0:
            print(f"\n{'='*60}")
            print(f"Spawning new Main Exploiter")
            print(f"{'='*60}")

            # Create fresh model for exploiter
            model = create_model_by_type(args.model_type, model_config).to(
                config.device
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=config.exploiter_lr)
            league_manager.spawn_main_exploiter(model, optimizer, model_config)

        # 3. Train Main Exploiter if one exists
        if league_manager.current_main_exploiter is not None:
            if not league_manager.current_main_exploiter.is_converged:
                train_main_exploiter(league_manager, league_tracker, config, iteration)

        # 4. Spawn League Exploiter (less frequently)
        if iteration > 0 and iteration % config.league_exploiter_spawn_interval == 0:
            print(f"\n{'='*60}")
            print(f"Spawning new League Exploiter")
            print(f"{'='*60}")

            model = create_model_by_type(args.model_type, model_config).to(
                config.device
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=config.exploiter_lr)
            league_manager.spawn_league_exploiter(model, optimizer, model_config)

        # 5. Log league summary
        num_converged = sum(1 for a in league_manager.main_exploiters if a.is_converged)
        num_converged += sum(
            1 for a in league_manager.league_exploiters if a.is_converged
        )

        league_tracker.log_league_summary(
            iteration,
            len(league_manager.main_agents),
            len(league_manager.main_exploiters),
            len(league_manager.league_exploiters),
            num_converged,
        )

        # Log ELO leaderboard periodically
        if (iteration + 1) % 10 == 0:
            league_tracker.log_elo_leaderboard(iteration)

        # 6. Periodic evaluation against engine
        if (iteration + 1) % args.eval_interval == 0:
            print(f"\n{'='*60}")
            print(f"Engine Evaluation (Iteration {iteration + 1})")
            print(f"{'='*60}")

            main_agent = league_manager.current_main_agent
            checkpoint = torch.load(main_agent.model_path, map_location=config.device)
            eval_model_type = checkpoint.get("model_type", "policy")
            model = create_model_by_type(
                eval_model_type, checkpoint.get("config", {})
            ).to(config.device)

            # Handle torch.compile() state dict keys
            state_dict = checkpoint["model_state_dict"]
            if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
                state_dict = {
                    k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
                }
            model.load_state_dict(state_dict)

            evaluate_and_update_elo(
                model,
                main_agent.name,
                league_tracker.elo_tracker,
                engine_depths=args.eval_depths,
                games_per_depth=args.eval_games,
                device=config.device,
                tensorboard_writer=league_tracker.writer,
                iteration=iteration,
            )

            # Log ELO leaderboard to TensorBoard
            league_tracker.log_elo_leaderboard(iteration)

            # Show current leaderboard
            print("\nCurrent ELO Leaderboard:")
            for rank, (player, rating) in enumerate(
                league_tracker.elo_tracker.get_leaderboard(10), 1
            ):
                print(f"  {rank}. {player}: {rating:.1f}")

    # Final summary
    print(f"\n{'='*80}")
    print("LEAGUE TRAINING COMPLETE!")
    print(f"{'='*80}")

    print(f"\nFinal League Composition:")
    print(f"  Main Agents: {len(league_manager.main_agents)}")
    print(f"  Main Exploiters: {len(league_manager.main_exploiters)}")
    print(f"  League Exploiters: {len(league_manager.league_exploiters)}")
    print(
        f"  Converged Exploiters: {sum(1 for a in league_manager.main_exploiters if a.is_converged)}"
    )

    print(f"\nBest Main Agent: {league_manager.current_main_agent.name}")
    print(f"  Games played: {league_manager.current_main_agent.games_played}")
    print(f"  Win rate: {league_manager.current_main_agent.win_rate:.2%}")

    # Close trackers
    league_tracker.close()

    print(f"\nLeague saved to: {save_dir}")
    print(f"TensorBoard logs: {save_dir / 'logs'}")


if __name__ == "__main__":
    main()
