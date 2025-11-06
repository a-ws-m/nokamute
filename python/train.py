"""
Main training script for self-play learning with GNN.
"""

import argparse
import os
import time
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from model import create_model
from self_play import SelfPlayGame, prepare_training_data
from torch.utils.tensorboard import SummaryWriter
import nokamute
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
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
    from pretrain.human_games import (
        generate_human_games_data,
        save_human_games_data,
        load_human_games_data,
    )
    PRETRAIN_AVAILABLE = True
except ImportError:
    PRETRAIN_AVAILABLE = False
    print("Warning: Pre-training modules not available")


def train_epoch(model, training_data, optimizer, batch_size=32, device="cpu", gamma=0.99, use_mc=False):
    """
    Train the model for one epoch using temporal difference learning or Monte Carlo returns.

    Args:
        model: GNN model
        training_data: List of tuples:
            - For TD: (node_features, edge_index, next_node_features, next_edge_index, is_terminal, final_result)
            - For MC: (node_features, edge_index, mc_return)
        optimizer: Optimizer
        batch_size: Batch size
        device: Device to train on
        gamma: TD discount factor (default: 0.99)
        use_mc: If True, use Monte Carlo returns; if False, use TD learning (default: False)

    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    num_batches = 0

    if use_mc:
        # Monte Carlo mode: training_data = [(node_features, edge_index, mc_return), ...]
        data_list = []
        for node_features, edge_index, mc_return in training_data:
            if len(node_features) == 0:
                continue

            x = torch.tensor(node_features, dtype=torch.float32)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)

            data = Data(x=x, edge_index=edge_index_tensor)
            data.y = torch.tensor([mc_return], dtype=torch.float32)
            data_list.append(data)

        if len(data_list) == 0:
            return 0.0

        loader = DataLoader(data_list, batch_size=batch_size, shuffle=True)

        for batch in loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            predictions, _ = model(batch.x, batch.edge_index, batch.batch)

            targets = batch.y.unsqueeze(1) if batch.y.dim() == 1 else batch.y
            loss = F.mse_loss(predictions, targets)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1
    else:
        # TD mode: compute targets on-the-fly using current model
        # training_data = [(curr_features, curr_edge_index, next_features, next_edge_index, is_terminal, final_result), ...]
        
        # We can't easily batch this with DataLoader since we need to evaluate next states
        # Instead, we'll process in mini-batches manually
        
        # Shuffle the data
        indices = list(range(len(training_data)))
        import random
        random.shuffle(indices)
        
        for batch_start in range(0, len(training_data), batch_size):
            batch_end = min(batch_start + batch_size, len(training_data))
            batch_indices = indices[batch_start:batch_end]
            
            curr_states = []
            td_targets_list = []
            
            for idx in batch_indices:
                item = training_data[idx]
                if len(item) != 6:
                    continue
                    
                curr_features, curr_edge_index, next_features, next_edge_index, is_terminal, final_result = item
                
                if len(curr_features) == 0:
                    continue
                
                # Add current state
                x_curr = torch.tensor(curr_features, dtype=torch.float32, device=device)
                edge_index_curr = torch.tensor(curr_edge_index, dtype=torch.long, device=device)
                curr_states.append(Data(x=x_curr, edge_index=edge_index_curr))
                
                # Compute TD target
                if is_terminal:
                    # Terminal state: use final result
                    td_targets_list.append(final_result)
                else:
                    # Non-terminal: evaluate next state with current model (no grad)
                    if next_features is not None and len(next_features) > 0:
                        with torch.no_grad():
                            x_next = torch.tensor(next_features, dtype=torch.float32, device=device)
                            edge_index_next = torch.tensor(next_edge_index, dtype=torch.long, device=device)
                            next_batch = torch.zeros(len(x_next), dtype=torch.long, device=device)
                            
                            next_value, _ = model(x_next, edge_index_next, next_batch)
                            td_targets_list.append(gamma * next_value[0].item())
                    else:
                        td_targets_list.append(0.0)
            
            if len(curr_states) == 0:
                continue
            
            # Batch current states
            batch = Batch.from_data_list(curr_states).to(device)
            td_targets = torch.tensor(td_targets_list, dtype=torch.float32, device=device).unsqueeze(1)
            
            # Forward pass on current states
            optimizer.zero_grad()
            curr_values, _ = model(batch.x, batch.edge_index, batch.batch)
            
            # Compute TD loss
            loss = F.mse_loss(curr_values, td_targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def prepare_td_training_data(games):
    """
    Prepare training data for on-the-fly TD learning.
    
    This function extracts state transitions from self-play games. During training,
    the TD targets will be computed on-the-fly using the current model's evaluation
    of the next state: TD_target = gamma * V(s_{t+1}) for non-terminal states.
    
    This allows the model to continuously update its value estimates based on its
    current understanding, implementing true temporal difference learning.
    
    Args:
        games: List of (game_data, result, branch_id) tuples from self-play
               Each game_data contains (nx_graph, legal_moves, selected_move, player, pos_hash, [move_value])
        
    Returns:
        training_examples: List of (curr_features, curr_edge_index, next_features, next_edge_index, is_terminal, final_result)
    """
    from graph_utils import networkx_to_pyg
    
    training_examples = []
    
    for game_data, final_result, branch_id in games:
        if len(game_data) == 0:
            continue
        
        # Process each position in the game
        for position_idx, item in enumerate(game_data):
            # Extract position info (handle both old and new formats)
            if len(item) >= 5:
                nx_graph = item[0]
                pos_hash = item[4]
            else:
                continue
            
            # Convert NetworkX graph to PyG format
            curr_features, curr_edge_index = networkx_to_pyg(nx_graph)
            
            # Skip empty boards
            if len(curr_features) == 0:
                continue
            
            # Determine if this is the last position (terminal state)
            is_terminal = (position_idx == len(game_data) - 1)
            
            if is_terminal:
                # Terminal state: no next state
                training_examples.append((
                    curr_features,
                    curr_edge_index,
                    None,  # next_features
                    None,  # next_edge_index
                    True,  # is_terminal
                    final_result  # final game result
                ))
            else:
                # Non-terminal state: get next state
                next_item = game_data[position_idx + 1]
                if len(next_item) >= 5:
                    next_nx_graph = next_item[0]
                    next_features, next_edge_index = networkx_to_pyg(next_nx_graph)
                    
                    training_examples.append((
                        curr_features,
                        curr_edge_index,
                        next_features,
                        next_edge_index,
                        False,  # is_terminal
                        final_result  # for reference (not used in non-terminal TD)
                    ))
    
    return training_examples


class ReplayBuffer:
    """
    Stores self-play games on disk for experience replay.
    
    For TD learning, we store raw game trajectories and recompute TD targets
    on-the-fly during training, so that earlier games benefit from the improved
    model's value estimates.
    """
    
    def __init__(self, save_dir, max_size=None, use_mc=False):
        """
        Args:
            save_dir: Directory to save replay buffer files
            max_size: Maximum number of games to store (None = unlimited)
            use_mc: Whether using Monte Carlo returns (affects data format)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_size = max_size
        self.use_mc = use_mc
        
        # Track stored games
        self.game_files = []
        self.total_games = 0
        
        # Load existing buffer state if available
        self._load_buffer_state()
    
    def _get_buffer_state_path(self):
        """Path to buffer metadata file."""
        return self.save_dir / "buffer_state.pkl"
    
    def _load_buffer_state(self):
        """Load buffer state from disk."""
        state_path = self._get_buffer_state_path()
        if state_path.exists():
            with open(state_path, "rb") as f:
                state = pickle.load(f)
                self.game_files = state.get("game_files", [])
                self.total_games = state.get("total_games", 0)
                self.use_mc = state.get("use_mc", self.use_mc)
    
    def _save_buffer_state(self):
        """Save buffer state to disk."""
        state = {
            "game_files": self.game_files,
            "total_games": self.total_games,
            "use_mc": self.use_mc,
        }
        with open(self._get_buffer_state_path(), "wb") as f:
            pickle.dump(state, f)
    
    def add_games(self, games, iteration):
        """
        Add new games to the replay buffer.
        
        Args:
            games: List of game tuples from self-play
            iteration: Current training iteration (used for filename)
        """
        if len(games) == 0:
            return
        
        # Save games to disk
        filename = f"games_iter_{iteration}.pkl"
        filepath = self.save_dir / filename
        
        with open(filepath, "wb") as f:
            pickle.dump(games, f)
        
        self.game_files.append(str(filepath))
        self.total_games += len(games)
        
        # Enforce max size if specified
        if self.max_size is not None:
            while self.total_games > self.max_size and len(self.game_files) > 0:
                # Remove oldest file
                old_file = Path(self.game_files[0])
                if old_file.exists():
                    # Load to count games before deleting
                    with open(old_file, "rb") as f:
                        old_games = pickle.load(f)
                    self.total_games -= len(old_games)
                    old_file.unlink()
                
                self.game_files.pop(0)
        
        # Save updated state
        self._save_buffer_state()
    
    def load_all_games(self):
        """
        Load all games from the replay buffer.
        
        Returns:
            List of all game tuples
        """
        all_games = []
        
        for filepath in self.game_files:
            path = Path(filepath)
            if path.exists():
                with open(path, "rb") as f:
                    games = pickle.load(f)
                    all_games.extend(games)
        
        return all_games
    
    def sample_games(self, num_games):
        """
        Sample a random subset of games from the buffer.
        
        Args:
            num_games: Number of games to sample
            
        Returns:
            List of sampled game tuples
        """
        all_games = self.load_all_games()
        
        if len(all_games) <= num_games:
            return all_games
        
        # Random sampling without replacement
        import random
        return random.sample(all_games, num_games)
    
    def get_stats(self):
        """Get buffer statistics."""
        return {
            "total_games": self.total_games,
            "num_files": len(self.game_files),
            "use_mc": self.use_mc,
        }
    
    def clear(self):
        """Clear all games from the buffer."""
        for filepath in self.game_files:
            path = Path(filepath)
            if path.exists():
                path.unlink()
        
        self.game_files = []
        self.total_games = 0
        self._save_buffer_state()


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
    parser.add_argument("--games", type=int, default=1, help="Games per iteration (default: 1 for TD learning)")
    parser.add_argument(
        "--iterations", type=int, default=10000, help="Training iterations"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per iteration (default: 1 for TD learning)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension")
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
        default=1000,
        help="Evaluate against engine every N iterations",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=10,
        help="Save model checkpoint every N iterations (default: 100)",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=10,
        help="Number of games per evaluation",
    )
    parser.add_argument(
        "--eval-depths",
        type=int,
        nargs="+",
        default=[1, 2, 3],
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
        "--use-mc",
        action="store_true",
        help="Use Monte Carlo returns instead of TD learning (all moves in a game get same final result)",
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
    
    # Replay buffer arguments
    parser.add_argument(
        "--replay-buffer",
        action="store_true",
        help="Enable experience replay: store games on disk and train on all accumulated games",
    )
    parser.add_argument(
        "--replay-buffer-dir",
        type=str,
        default=None,
        help="Directory to store replay buffer (default: <model-path>/replay_buffer)",
    )
    parser.add_argument(
        "--replay-buffer-size",
        type=int,
        default=10000,
        help="Maximum number of games to keep in replay buffer (default: 10000)",
    )
    parser.add_argument(
        "--replay-sample-size",
        type=int,
        default=5000,
        help="Number of games to sample from buffer for training each iteration (default: 5000, or all if fewer)",
    )
    
    # Pre-training arguments
    parser.add_argument(
        "--pretrain",
        type=str,
        choices=["eval-matching", "human-games", "none"],
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
    parser.add_argument(
        "--pretrain-sheet-urls",
        type=str,
        nargs="+",
        default=[
            "https://docs.google.com/spreadsheets/d/1d4YW6PDdYYSH8sGDQ-FB1zmdNJhNU-f2811o5JxMqsc/edit?gid=1437601135#gid=1437601135",
            # Note: Second spreadsheet has poor data quality (~26% parseable)
            "https://docs.google.com/spreadsheets/d/1JGc6hbmQITjqF-MLiW4AGZ24VpAFgN802EY179RrPII/edit?gid=194203898#gid=194203898",
        ],
        help="Google Sheets URLs for human games pre-training (default: high-quality tournament games)",
    )
    parser.add_argument(
        "--pretrain-max-games",
        type=int,
        default=None,
        help="Maximum number of human games to use for pre-training (default: all available)",
    )

    args = parser.parse_args()

    # When using Monte Carlo, suggest more games and epochs per iteration
    # unless explicitly overridden by user
    if args.use_mc:
        # Check if user explicitly set these values
        import sys
        games_explicitly_set = '--games' in sys.argv
        epochs_explicitly_set = '--epochs' in sys.argv
        
        if not games_explicitly_set and args.games == 1:  # 1 is the default
            args.games = 100
            print("Monte Carlo mode: defaulting to 100 games per iteration")
        
        if not epochs_explicitly_set and args.epochs == 1:  # 1 is the default
            args.epochs = 10
            print("Monte Carlo mode: defaulting to 10 epochs per iteration")

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
            
            print("\nPre-training evaluation results:")
            for depth, results in pretrain_eval_results.items():
                print(f"  vs Depth {depth}: {results}")
        
        elif args.pretrain == "human-games":
            print(f"\nPre-training method: Human Games TD Learning")
            print(f"Target: Learn from human game positions using TD learning")
            
            # Determine data file path
            if args.pretrain_data_path:
                data_file_path = args.pretrain_data_path
            else:
                # Auto-generate path based on settings
                max_games_str = f"_max{args.pretrain_max_games}" if args.pretrain_max_games else "_all"
                data_filename = f"pretrain_human{max_games_str}.pkl"
                data_file_path = os.path.join(args.model_path, data_filename)
            
            # Load or generate pre-training data
            if os.path.exists(data_file_path):
                print(f"\nLoading existing pre-training data from: {data_file_path}")
                pretrain_data = load_human_games_data(data_file_path)
            else:
                print(f"\nDownloading and parsing human games...")
                print(f"  Sheet URLs: {len(args.pretrain_sheet_urls)} sheets")
                if args.pretrain_max_games:
                    print(f"  Max games: {args.pretrain_max_games}")
                else:
                    print(f"  Max games: all available")
                
                pretrain_data = generate_human_games_data(
                    sheet_urls=args.pretrain_sheet_urls,
                    game_type="Base+MLP",
                    max_games=args.pretrain_max_games,
                    verbose=True,
                )
                
                if len(pretrain_data) == 0:
                    print("\nError: No valid training data generated from human games.")
                    print("Please check the sheet URLs and game log format.")
                    return
                
                # Save generated data for future use
                print(f"\nSaving pre-training data to: {data_file_path}")
                save_human_games_data(pretrain_data, data_file_path)
            
            # Pre-train the model using TD learning
            # We use the same train_epoch function with use_mc=False
            print(f"\nPre-training with TD learning for {args.pretrain_epochs} epochs...")
            print(f"Training transitions: {len(pretrain_data)}")
            print(f"Batch size: {args.batch_size}")
            print(f"Learning rate: {args.lr}")
            print(f"Gamma (TD discount): {args.gamma}")
            
            epoch_losses = []
            for epoch in range(args.pretrain_epochs):
                epoch_loss = train_epoch(
                    model=model,
                    training_data=pretrain_data,
                    optimizer=optimizer,
                    batch_size=args.batch_size,
                    device=args.device,
                    gamma=args.gamma,
                    use_mc=False,  # Use TD learning
                )
                
                epoch_losses.append(epoch_loss)
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    print(f"Epoch {epoch + 1}/{args.pretrain_epochs}: Loss = {epoch_loss:.6f}")
                
                # Log to tensorboard
                writer.add_scalar("Pretrain/Loss", epoch_loss, epoch)
            
            # Save pre-trained model
            pretrain_model_path = os.path.join(args.model_path, "model_pretrained.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "pretrain_method": args.pretrain,
                "pretrain_loss": epoch_losses[-1],
                "config": model_config,
                "pretrain_num_transitions": len(pretrain_data),
                "pretrain_epochs": args.pretrain_epochs,
            }, pretrain_model_path)
            print(f"\nPre-trained model saved to: {pretrain_model_path}")
            print(f"Final pre-training loss: {epoch_losses[-1]:.6f}")
            
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
            
            print("\nPre-training evaluation results:")
            for depth, results in pretrain_eval_results.items():
                print(f"  vs Depth {depth}: {results}")
            
            current_elo = elo_tracker.get_rating("model_pretrained")
            print(f"\nPre-trained model ELO: {current_elo:.1f}")
            writer.add_scalar("Pretrain/ELO", current_elo, 0)
            
            # Print results summary
            print("\n" + "=" * 60)
            print("PRE-TRAINING COMPLETE!")
            print("=" * 60)
            print(f"\nPre-trained model: {pretrain_model_path}")
            print(f"Training data: {data_file_path}")
            print(f"Final loss: {epoch_losses[-1]:.6f}")
            print(f"ELO rating: {current_elo:.1f}")
            print("\nTo continue with self-play training, run:")
            print(f"  python train.py --resume {pretrain_model_path} --iterations N")
        
        writer.close()
        return  # Exit after pre-training

    # Initialize replay buffer if enabled
    replay_buffer = None
    if args.replay_buffer:
        buffer_dir = args.replay_buffer_dir or os.path.join(args.model_path, "replay_buffer")
        replay_buffer = ReplayBuffer(
            save_dir=buffer_dir,
            max_size=args.replay_buffer_size,
            use_mc=args.use_mc,
        )
        print(f"\nReplay buffer enabled:")
        print(f"  Directory: {buffer_dir}")
        print(f"  Max size: {args.replay_buffer_size:,} games")
        print(f"  Sample size: {args.replay_sample_size:,} games per iteration")
        
        # Load existing buffer stats
        stats = replay_buffer.get_stats()
        if stats["total_games"] > 0:
            print(f"  Loaded existing buffer: {stats['total_games']:,} games in {stats['num_files']} files")

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

        # Add games to replay buffer if enabled
        if replay_buffer is not None:
            replay_buffer.add_games(games, iteration)
            buffer_stats = replay_buffer.get_stats()
            print(f"Replay buffer: {buffer_stats['total_games']} total games stored")

        # Prepare training data
        print("Preparing training data...")
        
        # Determine which games to use for training
        if replay_buffer is not None:
            # Use games from replay buffer
            buffer_stats = replay_buffer.get_stats()
            total_available = buffer_stats['total_games']
            
            # Sample if we have more games than sample size, otherwise use all
            if total_available > args.replay_sample_size:
                training_games = replay_buffer.sample_games(args.replay_sample_size)
                print(f"Sampled {len(training_games)} games from replay buffer ({total_available} available)")
            else:
                training_games = replay_buffer.load_all_games()
                print(f"Using all {len(training_games)} games from replay buffer")
        else:
            # Use only current iteration's games (original behavior)
            training_games = games
        
        # Convert games to training data
        if args.use_mc:
            print("Using Monte Carlo returns")
            training_data = prepare_training_data(training_games)
        else:
            print(f"Using on-the-fly TD learning with gamma={args.gamma}")
            training_data = prepare_td_training_data(training_games)
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
                use_mc=args.use_mc,
            )

            print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {loss:.6f}")

            # Log to tensorboard
            global_step = iteration * args.epochs + epoch
            writer.add_scalar("Loss/train", loss, global_step)
        
        # Log replay buffer stats if enabled
        if replay_buffer is not None:
            buffer_stats = replay_buffer.get_stats()
            writer.add_scalar("ReplayBuffer/total_games", buffer_stats["total_games"], iteration)
            writer.add_scalar("ReplayBuffer/num_files", buffer_stats["num_files"], iteration)

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
            checkpoint_data = {
                "iteration": iteration + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model_config,
                "eval_stats": eval_stats,
            }
            if replay_buffer is not None:
                checkpoint_data["replay_buffer_stats"] = replay_buffer.get_stats()
            
            torch.save(checkpoint_data, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Save latest model at specified intervals (overwriting each time)
        if (iteration + 1) % args.save_interval == 0:
            latest_path = os.path.join(args.model_path, "model_latest.pt")
            latest_data = {
                "iteration": iteration + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model_config,
                "eval_stats": eval_stats,
            }
            if replay_buffer is not None:
                latest_data["replay_buffer_stats"] = replay_buffer.get_stats()
            
            torch.save(latest_data, latest_path)
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
