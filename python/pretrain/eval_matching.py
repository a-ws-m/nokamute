"""
Pre-training by matching analytical evaluation function.

This module generates random positions using a branching Markov chain process
and trains the GNN model to match the Rust engine's analytical evaluation scores.

Rather than playing full games, we generate diverse positions by:
1. Starting from the initial position
2. Applying random legal moves with branching
3. Evaluating each unique position with the Rust engine
4. Training the GNN to match these evaluations using MSE loss

This gives the model a good initial understanding of position evaluation
before transitioning to self-play training.
"""

import random
import pickle
import torch
import torch.nn.functional as F
import nokamute
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from typing import List, Tuple, Optional, Set
from tqdm import tqdm
import argparse
import hashlib
from pathlib import Path
from multiprocessing import Pool, cpu_count

import sys
import os
# Add parent directory to path to import graph_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph_utils import graph_hash


def generate_random_positions_branching(
    num_positions: int = 1000,
    aggression: int = 3,
    max_depth: int = 50,
    branch_probability: float = 0.3,
    eval_depth: int = 0,
    verbose: bool = True,
) -> List[Tuple[List[float], List[List[int]], float]]:
    """
    Generate random positions using a branching Markov chain process.
    
    This creates diverse positions by:
    1. Starting from initial position (or random branch points)
    2. Applying random legal moves to explore the state space
    3. Occasionally branching to create parallel exploration paths
    4. Evaluating each unique position with the Rust engine
    
    This is more efficient than playing full games, as we don't need to
    reach terminal states and can explore a wider variety of positions.
    
    Args:
        num_positions: Target number of unique positions to generate
        aggression: Aggression level 1-5 for BasicEvaluator (default: 3)
        max_depth: Maximum number of moves from start position (default: 50)
        branch_probability: Probability of creating a branch point (default: 0.3)
        eval_depth: Search depth for minimax evaluation (default: 0 for static eval)
                   If > 0, uses minimax to evaluate position N moves ahead
        verbose: Print progress information
        
    Returns:
        List of (node_features, edge_index, eval_score) tuples
        eval_score is the raw evaluation from BasicEvaluator (absolute scale)
    """
    unique_positions = {}  # Map: graph_hash -> (node_features, edge_index, eval_score)
    branch_points = []  # List of (board_state, depth) tuples for branching
    
    if verbose:
        print(f"Generating {num_positions} unique positions using branching Markov chain...")
        pbar = tqdm(total=num_positions, desc="Generating positions")
    
    # Start with initial position
    initial_board = nokamute.Board()
    branch_points.append((initial_board, 0))
    
    while len(unique_positions) < num_positions and branch_points:
        # Select a branch point (FIFO for breadth-first-like exploration)
        board, current_depth = branch_points.pop(0)
        
        # Add current position if unique
        _add_position_if_unique(board, aggression, eval_depth, unique_positions)
        if verbose:
            pbar.update(1)
        
        # Check if we've reached our target
        if len(unique_positions) >= num_positions:
            break
        
        # Check if game is over or max depth reached
        if board.get_winner() is not None or current_depth >= max_depth:
            continue
        
        # Get legal moves
        legal_moves = board.legal_moves()
        if not legal_moves:
            continue
        
        # Apply a random move
        move = random.choice(legal_moves)
        board.apply(move)
        
        # Add the new position
        _add_position_if_unique(board, aggression, eval_depth, unique_positions)
        if verbose:
            pbar.update(1)
        
        # Decide whether to branch
        if random.random() < branch_probability and len(legal_moves) > 1:
            # Create branches for other legal moves
            num_branches = min(2, len(legal_moves) - 1)  # Limit branching factor
            other_moves = [m for m in legal_moves if m != move]
            for branch_move in random.sample(other_moves, num_branches):
                # Clone the board state before the move
                board.undo(move)
                branch_board = board.clone()
                branch_board.apply(branch_move)
                branch_points.append((branch_board, current_depth + 1))
                board.apply(move)  # Restore state
        
        # Continue on the main path
        branch_points.append((board, current_depth + 1))
    
    if verbose:
        pbar.close()
        print(f"\nGenerated {len(unique_positions)} unique positions")
        
        # Show evaluation statistics
        evals = [score for _, _, score in unique_positions.values()]
        print(f"Evaluation stats:")
        print(f"  Min: {min(evals):.2f}")
        print(f"  Max: {max(evals):.2f}")
        print(f"  Mean: {sum(evals) / len(evals):.2f}")
        print(f"  Median: {sorted(evals)[len(evals) // 2]:.2f}")
    
    return list(unique_positions.values())


def _add_position_if_unique(board, aggression, eval_depth, unique_positions):
    """
    Add a position to the training data if it hasn't been seen before.
    
    Args:
        board: Current board state
        aggression: Aggression level for evaluation
        eval_depth: Search depth for minimax evaluation (0 for static eval)
        unique_positions: Dictionary of unique positions (modified in-place)
    """
    # Get graph representation and hash
    node_features, edge_index = board.to_graph()
    pos_hash = graph_hash(node_features, edge_index)
    
    # Only add if we haven't seen it before
    if pos_hash not in unique_positions:
        eval_score = board.get_evaluation(aggression, eval_depth)
        unique_positions[pos_hash] = (node_features, edge_index, eval_score)


def _generate_position_batch_worker(args):
    """
    Worker function for generating a batch of positions.
    
    This generates positions in small batches (e.g., 10 at a time) to allow
    frequent progress bar updates while preserving the branching behavior.
    
    Args:
        args: Tuple of (batch_size, aggression, max_depth, branch_probability, eval_depth, seed)
        
    Returns:
        List of (node_features, edge_index, eval_score) tuples
    """
    batch_size, aggression, max_depth, branch_probability, eval_depth, seed = args
    
    # Set random seed for this batch
    random.seed(seed)
    
    # Generate a batch of positions using the branching process
    result = generate_random_positions_branching(
        num_positions=batch_size,
        aggression=aggression,
        max_depth=max_depth,
        branch_probability=branch_probability,
        eval_depth=eval_depth,
        verbose=False,
    )
    
    return result


def _get_cache_key(num_positions: int, aggression: int, max_depth: int, 
                    branch_probability: float, eval_depth: int) -> str:
    """
    Generate a unique cache key for a set of generation parameters.
    
    Args:
        num_positions: Number of positions
        aggression: Aggression level
        max_depth: Maximum depth
        branch_probability: Branch probability
        eval_depth: Evaluation depth
        
    Returns:
        MD5 hash of the configuration
    """
    config_str = f"{num_positions}|{aggression}|{max_depth}|{branch_probability}|{eval_depth}"
    return hashlib.md5(config_str.encode()).hexdigest()


def _get_cache_path(cache_dir: str, cache_key: str) -> Path:
    """
    Get the path to a cached data directory.
    
    Args:
        cache_dir: Base directory to store cache files
        cache_key: Unique cache key
        
    Returns:
        Path to cache directory
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path / f"eval_matching_{cache_key}"


def generate_and_save_positions(
    num_positions: int = 1000,
    aggression: int = 3,
    max_depth: int = 200,
    branch_probability: float = 0.3,
    eval_depth: int = 0,
    cache_dir: str = "pretrain_cache",
    num_workers: Optional[int] = None,
    force_refresh: bool = False,
    verbose: bool = True,
) -> str:
    """
    Generate random positions with evaluations and save to disk using multiprocessing.
    
    Similar to human_games.py's generate_human_games_data, this function generates
    positions and saves them in a format suitable for streaming during training.
    
    Args:
        num_positions: Total number of unique positions to generate
        aggression: Aggression level 1-5 for BasicEvaluator (default: 3)
        max_depth: Maximum number of moves from start position (default: 200)
        branch_probability: Probability of creating a branch point (default: 0.3)
        eval_depth: Search depth for minimax evaluation (default: 0 for static eval)
        cache_dir: Directory to store cached data (default: "pretrain_cache")
        num_workers: Number of parallel workers (default: cpu_count())
        force_refresh: If True, ignore cache and regenerate data
        verbose: Print progress information
        
    Returns:
        Path to the cache directory containing the generated positions
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    # Check cache first
    cache_key = _get_cache_key(num_positions, aggression, max_depth, branch_probability, eval_depth)
    cache_data_dir = _get_cache_path(cache_dir, cache_key)
    cache_metadata_path = cache_data_dir / "metadata.pkl"
    
    # Check if we have cached data
    if not force_refresh and cache_metadata_path.exists():
        if verbose:
            print(f"Found cached data at {cache_data_dir}")
        # Load metadata to get statistics
        with open(cache_metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        if verbose:
            print(f"Loaded cached data:")
            print(f"  Total positions: {metadata['total_positions']}")
            print(f"  Evaluation depth: {metadata['eval_depth']}")
            print(f"  Number of files: {metadata['num_files']}")
        
        return str(cache_data_dir)
    
    # No cache or force refresh - generate positions
    if verbose:
        if force_refresh:
            print("Force refresh enabled - regenerating data")
        print(f"\nGenerating {num_positions} positions using {num_workers} workers...")
        print(f"Parameters:")
        print(f"  Aggression: {aggression}")
        print(f"  Max depth: {max_depth}")
        print(f"  Branch probability: {branch_probability}")
        print(f"  Eval depth: {eval_depth}")
    
    # Create cache directory
    cache_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Create work items - batch positions into groups for progress updates
    batch_size = 10  # Generate this many positions per worker task
    num_batches = (num_positions + batch_size - 1) // batch_size  # Ceiling division
    
    work_items = []
    remaining = num_positions
    for i in range(num_batches):
        # Last batch might be smaller
        current_batch_size = min(batch_size, remaining)
        remaining -= current_batch_size
        seed = random.randint(0, 2**32 - 1)
        work_items.append((current_batch_size, aggression, max_depth, branch_probability, eval_depth, seed))
    
    # Run workers in parallel
    if verbose:
        print(f"\nGenerating {num_positions} positions using {num_workers} workers...")
        print(f"Processing in batches of {batch_size} positions for progress updates...")
    
    positions_list = []
    
    with Pool(num_workers) as pool:
        if verbose:
            # Use imap to process batches, updating progress bar for each batch
            pbar = tqdm(total=num_positions, desc="Positions generated", unit="pos")
            for batch in pool.imap(_generate_position_batch_worker, work_items):
                positions_list.extend(batch)
                pbar.update(len(batch))
            pbar.close()
        else:
            for batch in pool.map(_generate_position_batch_worker, work_items):
                positions_list.extend(batch)
    
    # Save positions to disk
    if verbose:
        print("\nSaving positions to disk...")
    
    total_positions = 0
    positions_per_file = 1000  # Save every 1000 positions to a file
    file_index = 0
    current_batch = []
    
    for position in positions_list:
        current_batch.append(position)
        total_positions += 1
        
        # Save batch when it reaches the target size
        if len(current_batch) >= positions_per_file:
            batch_file = cache_data_dir / f"positions_{file_index:06d}.pkl"
            with open(batch_file, 'wb') as f:
                pickle.dump(current_batch, f)
            file_index += 1
            current_batch = []
    
    # Save any remaining positions
    if len(current_batch) > 0:
        batch_file = cache_data_dir / f"positions_{file_index:06d}.pkl"
        with open(batch_file, 'wb') as f:
            pickle.dump(current_batch, f)
        file_index += 1
    
    # Save metadata
    metadata = {
        'total_positions': total_positions,
        'num_files': file_index,
        'aggression': aggression,
        'max_depth': max_depth,
        'branch_probability': branch_probability,
        'eval_depth': eval_depth,
        'cache_key': cache_key,
    }
    with open(cache_metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Data Generation Complete")
        print(f"{'='*60}")
        print(f"Total positions: {total_positions}")
        print(f"Saved to: {cache_data_dir}")
        print(f"Number of files: {file_index}")
        
        # Show evaluation statistics
        if total_positions > 0:
            all_evals = [eval_score for _, _, eval_score in positions_list]
            
            print(f"\nEvaluation statistics:")
            print(f"  Min: {min(all_evals):.2f}")
            print(f"  Max: {max(all_evals):.2f}")
            print(f"  Mean: {sum(all_evals) / len(all_evals):.2f}")
            print(f"  Median: {sorted(all_evals)[len(all_evals) // 2]:.2f}")
    
    return str(cache_data_dir)


class EvaluationMatchingDataset(Dataset):
    """
    PyTorch Geometric Dataset for evaluation matching training.
    
    This dataset generates random positions on-the-fly each epoch,
    ensuring the model sees diverse training data throughout training.
    """
    
    def __init__(
        self,
        num_positions: int,
        aggression: int = 3,
        max_depth: int = 50,
        branch_probability: float = 0.3,
        eval_depth: int = 0,
        scale: float = 0.001,
        regenerate_each_epoch: bool = True,
    ):
        """
        Initialize the evaluation matching dataset.
        
        Args:
            num_positions: Number of unique positions to generate per epoch
            aggression: Aggression level for BasicEvaluator (1-5)
            max_depth: Maximum depth for random walk
            branch_probability: Probability of branching during generation
            eval_depth: Search depth for minimax evaluation (0 for static eval)
            scale: Scaling factor for normalizing evaluations
            regenerate_each_epoch: If True, generate new positions each epoch
        """
        super().__init__()
        self.num_positions = num_positions
        self.aggression = aggression
        self.max_depth = max_depth
        self.branch_probability = branch_probability
        self.eval_depth = eval_depth
        self.scale = scale
        self.regenerate_each_epoch = regenerate_each_epoch
        
        # Generate initial data
        self._regenerate_positions()
    
    def _regenerate_positions(self):
        """Generate new random positions."""
        raw_data = generate_random_positions_branching(
            num_positions=self.num_positions,
            aggression=self.aggression,
            max_depth=self.max_depth,
            branch_probability=self.branch_probability,
            eval_depth=self.eval_depth,
            verbose=False,
        )
        
        # Convert to PyG Data objects
        self.data_list = []
        for node_features, edge_index, eval_score in raw_data:
            # Skip empty graphs
            if len(node_features) == 0:
                continue
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float32)
            edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
            
            # Normalize evaluation score
            normalized_score = normalize_evaluation(eval_score, self.scale)
            y = torch.tensor([normalized_score], dtype=torch.float32)
            
            data = Data(x=x, edge_index=edge_index_tensor, y=y)
            self.data_list.append(data)
    
    def len(self):
        """Return the number of positions in the dataset."""
        return len(self.data_list)
    
    def get(self, idx):
        """Get a single position."""
        return self.data_list[idx]
    
    def regenerate(self):
        """Regenerate positions for a new epoch."""
        if self.regenerate_each_epoch:
            self._regenerate_positions()


class CachedEvaluationDataset(Dataset):
    """
    PyTorch Geometric Dataset for loading pre-generated evaluation data from disk.
    
    This dataset loads positions and evaluations that were previously generated
    using generate_and_save_positions(). The data is loaded into memory and
    used for multiple epochs with shuffling.
    
    Similar to HumanGamesDataset but simpler since we don't need TD learning.
    """
    
    def __init__(
        self,
        cache_dir: str,
        scale: float = 0.001,
        verbose: bool = True,
    ):
        """
        Initialize the cached evaluation dataset.
        
        Args:
            cache_dir: Directory containing cached position files
            scale: Scaling factor for normalizing evaluations
            verbose: Print loading information
        """
        super().__init__()
        self.cache_dir = Path(cache_dir)
        self.scale = scale
        self.verbose = verbose
        
        # Load metadata
        metadata_path = self.cache_dir / "metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"No metadata found at {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        if verbose:
            print(f"Loading cached evaluation data from {cache_dir}")
            print(f"  Total positions: {self.metadata['total_positions']}")
            print(f"  Evaluation depth: {self.metadata['eval_depth']}")
            print(f"  Number of files: {self.metadata['num_files']}")
        
        # Load all positions into memory
        self.data_list = []
        position_files = sorted(self.cache_dir.glob("positions_*.pkl"))
        
        if verbose:
            print(f"Loading {len(position_files)} position files...")
            position_files = tqdm(position_files, desc="Loading files")
        
        for filepath in position_files:
            with open(filepath, 'rb') as f:
                positions = pickle.load(f)
            
            # Convert to PyG Data objects
            for node_features, edge_index, eval_score in positions:
                # Skip empty graphs
                if len(node_features) == 0:
                    continue
                
                # Convert to tensors
                x = torch.tensor(node_features, dtype=torch.float32)
                edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
                
                # Normalize evaluation score
                normalized_score = normalize_evaluation(eval_score, self.scale)
                y = torch.tensor([normalized_score], dtype=torch.float32)
                
                data = Data(x=x, edge_index=edge_index_tensor, y=y)
                self.data_list.append(data)
        
        if verbose:
            print(f"Loaded {len(self.data_list)} positions into memory")
    
    def len(self):
        """Return the number of positions in the dataset."""
        return len(self.data_list)
    
    def get(self, idx):
        """Get a single position."""
        return self.data_list[idx]
    
    def get_metadata(self):
        """Return dataset metadata."""
        return self.metadata.copy()


def normalize_evaluation(eval_score: float, scale: float = 0.001) -> float:
    """
    Normalize evaluation scores to [-1, 1] range using tanh-like scaling.
    
    Args:
        eval_score: Raw evaluation score from BasicEvaluator (absolute scale:
                   positive = White advantage, negative = Black advantage)
        scale: Scaling factor (default: 0.001, which maps 1000 -> ~0.76)
        
    Returns:
        Normalized score in approximately [-1, 1] range (absolute scale preserved)
    """
    # Use tanh to smoothly map to [-1, 1]
    # This handles extreme values gracefully while preserving the sign
    import torch
    return torch.tanh(torch.tensor(eval_score * scale)).item()


def pretrain_eval_matching(
    model,
    num_positions_per_epoch: int = 1000,
    optimizer=None,
    num_epochs: int = 50,
    batch_size: int = 64,
    device: str = "cpu",
    aggression: int = 3,
    max_depth: int = 50,
    branch_probability: float = 0.3,
    eval_depth: int = 0,
    scale: float = 0.001,
    regenerate_each_epoch: bool = True,
    verbose: bool = True,
    epoch_callback=None,
) -> List[float]:
    """
    Pre-train model to match analytical evaluation scores using random position generation.
    
    Each epoch:
    1. Generate N unique random positions using branching Markov chain
    2. Evaluate each position with the Rust engine
    3. Train the GNN to match these evaluations using MSE loss with mini-batching
    
    Args:
        model: GNN model to train
        num_positions_per_epoch: Number of unique positions to generate per epoch
        optimizer: PyTorch optimizer (if None, uses Adam with lr=1e-3)
        num_epochs: Number of training epochs (default: 50)
        batch_size: Batch size for mini-batch training
        device: Device to train on
        aggression: Aggression level 1-5 for BasicEvaluator (default: 3)
        max_depth: Maximum depth for random position generation (default: 50)
        branch_probability: Branching probability during position generation (default: 0.3)
        eval_depth: Search depth for minimax evaluation (default: 0 for static eval)
                   If > 0, uses minimax to evaluate position N moves ahead
        scale: Scaling factor for normalizing evaluations (default: 0.001)
        regenerate_each_epoch: If True, generate new positions each epoch (default: True)
        verbose: Print progress information
        epoch_callback: Optional callback function called after each epoch with (epoch, loss)
        
    Returns:
        List of average losses per epoch
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    epoch_losses = []
    
    # Create dataset
    dataset = EvaluationMatchingDataset(
        num_positions=num_positions_per_epoch,
        aggression=aggression,
        max_depth=max_depth,
        branch_probability=branch_probability,
        eval_depth=eval_depth,
        scale=scale,
        regenerate_each_epoch=regenerate_each_epoch,
    )
    
    if verbose:
        print(f"Training on {len(dataset)} positions per epoch...")
        print(f"Regenerating positions each epoch: {regenerate_each_epoch}")
    
    for epoch in tqdm(range(num_epochs), desc="Training epochs", disable=not verbose):
        # Regenerate positions for this epoch if enabled
        if epoch > 0:
            dataset.regenerate()
        
        total_loss = 0
        num_batches = 0
        
        # Create DataLoader for mini-batch training
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Single-threaded for simplicity
        )
        
        for batch in loader:
            batch = batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions, _ = model(batch.x, batch.edge_index, batch.batch)
            
            # Extract targets from batch
            targets = batch.y.unsqueeze(1) if batch.y.dim() == 1 else batch.y
            
            # Ensure predictions and targets have the same shape
            if predictions.shape != targets.shape:
                raise ValueError(f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}")
            
            # Compute MSE loss between GNN predictions and Rust engine evaluations
            loss = F.mse_loss(predictions, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        epoch_losses.append(avg_loss)
        
        if verbose and (epoch + 1) % 10 == 0:
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.6f}")
        
        # Call epoch callback if provided
        if epoch_callback is not None:
            epoch_callback(epoch, avg_loss)
    
    return epoch_losses



def save_eval_data(training_data: List[Tuple[List[float], List[List[int]], float]], filepath: str):
    """
    Save evaluation matching training data to disk.
    
    Args:
        training_data: List of (node_features, edge_index, eval_score) tuples
        filepath: Path to save the data
    """
    import pickle
    with open(filepath, 'wb') as f:
        pickle.dump(training_data, f)
    print(f"Saved {len(training_data)} positions to {filepath}")


def load_eval_data(filepath: str) -> List[Tuple[List[float], List[List[int]], float]]:
    """
    Load evaluation matching training data from disk.
    
    Args:
        filepath: Path to load the data from
        
    Returns:
        List of (node_features, edge_index, eval_score) tuples
    """
    import pickle
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} positions from {filepath}")
    return data


# Backward compatibility: keep old function name as alias
def generate_eval_matching_data(
    num_games: int = 100,
    depth: int = 7,
    aggression: int = 3,
    randomness_rate: float = 0.0,
    max_moves: int = 400,
    eval_depth: int = 0,
    verbose: bool = True,
) -> List[Tuple[List[float], List[List[int]], float]]:
    """
    DEPRECATED: Use generate_random_positions_branching instead.
    
    This function is kept for backward compatibility but now uses the new
    branching Markov chain approach instead of playing full games.
    
    Args:
        num_games: Ignored (kept for compatibility)
        depth: Used as max_depth for random walks
        aggression: Aggression level 1-5 for BasicEvaluator
        randomness_rate: Ignored (kept for compatibility)
        max_moves: Used as max_depth for random walks
        eval_depth: Search depth for minimax evaluation (default: 0 for static eval)
        verbose: Print progress information
        
    Returns:
        List of (node_features, edge_index, eval_score) tuples
    """
    if verbose:
        print("WARNING: generate_eval_matching_data is deprecated. Use generate_random_positions_branching instead.")
        print("Converting parameters to new format...")
    
    # Use max_moves as max_depth since it represents similar concept
    max_depth = min(max_moves, 50)  # Cap at reasonable value
    
    # Generate roughly similar number of positions
    # Old approach: num_games * average_positions_per_game
    # New approach: directly specify num_positions
    num_positions = num_games * 20  # Rough estimate
    
    return generate_random_positions_branching(
        num_positions=num_positions,
        aggression=aggression,
        max_depth=max_depth,
        branch_probability=0.3,
        eval_depth=eval_depth,
        verbose=verbose,
    )




if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate random positions with evaluations for pre-training"
    )
    
    # Generation parameters
    parser.add_argument(
        "--num-positions",
        type=int,
        default=1000,
        help="Number of unique positions to generate (default: 1000)"
    )
    parser.add_argument(
        "--aggression",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="Aggression level for BasicEvaluator (default: 3)"
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=200,
        help="Maximum number of moves from start position (default: 200)"
    )
    parser.add_argument(
        "--branch-probability",
        type=float,
        default=0.3,
        help="Probability of creating a branch point (default: 0.3)"
    )
    parser.add_argument(
        "--eval-depth",
        type=int,
        default=0,
        help="Search depth for minimax evaluation (default: 0 for static eval)"
    )
    
    # Output parameters
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="pretrain_cache",
        help="Directory to store cached data (default: pretrain_cache)"
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force regeneration even if cache exists"
    )
    
    # Performance parameters
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: number of CPU cores)"
    )
    
    # Testing mode
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run tests instead of generating data"
    )
    
    args = parser.parse_args()
    
    if args.test:
        # Test random position generation
        print("Testing evaluation matching pre-training with branching Markov chain...")
        
        # Generate small dataset
        print("\n1. Testing position generation:")
        data = generate_random_positions_branching(
            num_positions=100,
            aggression=3,
            max_depth=30,
            branch_probability=0.3,
            eval_depth=0,  # Static evaluation
            verbose=True
        )
        
        print(f"\nGenerated {len(data)} training examples")
        
        # Test normalization
        if data:
            sample_node_features, sample_edge_index, sample_eval = data[0]
            print(f"\nSample position:")
            print(f"  Nodes: {len(sample_node_features)}")
            print(f"  Edges: {len(sample_edge_index[0]) if sample_edge_index else 0}")
            print(f"  Raw evaluation: {sample_eval}")
            print(f"  Normalized: {normalize_evaluation(sample_eval):.4f}")
        
        # Test with depth > 0
        print("\n2. Testing with minimax evaluation (depth=2):")
        data_depth = generate_random_positions_branching(
            num_positions=20,
            aggression=3,
            max_depth=30,
            branch_probability=0.3,
            eval_depth=2,  # Minimax evaluation at depth 2
            verbose=True
        )
        
        print(f"\nGenerated {len(data_depth)} training examples with minimax eval")
        
        # Test dataset
        print("\n3. Testing EvaluationMatchingDataset:")
        dataset = EvaluationMatchingDataset(
            num_positions=50,
            aggression=3,
            max_depth=30,
            branch_probability=0.3,
            eval_depth=0,
            regenerate_each_epoch=True,
        )
        print(f"Dataset length: {len(dataset)}")
        sample = dataset[0]
        print(f"Sample data shape: x={sample.x.shape}, edge_index={sample.edge_index.shape}, y={sample.y.shape}")
        
        print("\n4. Testing DataLoader:")
        loader = DataLoader(dataset, batch_size=8, shuffle=True)
        for batch in loader:
            print(f"Batch: x={batch.x.shape}, edge_index={batch.edge_index.shape}, y={batch.y.shape}")
            print(f"Batch has {batch.num_graphs} graphs")
            break
        
        print("\n5. Testing multiprocessing generation and caching:")
        cache_dir = generate_and_save_positions(
            num_positions=100,
            aggression=3,
            max_depth=30,
            branch_probability=0.3,
            eval_depth=0,
            cache_dir="test_cache",
            num_workers=2,
            force_refresh=True,
            verbose=True,
        )
        
        print("\n6. Testing CachedEvaluationDataset:")
        cached_dataset = CachedEvaluationDataset(cache_dir, verbose=True)
        print(f"Cached dataset length: {len(cached_dataset)}")
        sample = cached_dataset[0]
        print(f"Sample data shape: x={sample.x.shape}, edge_index={sample.edge_index.shape}, y={sample.y.shape}")
        
        print("\nAll tests passed!")
    else:
        # Generate and save positions
        cache_dir = generate_and_save_positions(
            num_positions=args.num_positions,
            aggression=args.aggression,
            max_depth=args.max_depth,
            branch_probability=args.branch_probability,
            eval_depth=args.eval_depth,
            cache_dir=args.cache_dir,
            num_workers=args.num_workers,
            force_refresh=args.force_refresh,
            verbose=True,
        )
        
        print(f"\n{'='*60}")
        print("SUCCESS!")
        print(f"{'='*60}")
        print(f"Generated positions saved to: {cache_dir}")
        print(f"\nTo use this data for pre-training, run:")
        print(f"  python train.py --pretrain eval-matching --pretrain-data-path {cache_dir}")
