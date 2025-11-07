"""
Pre-training by matching analytical evaluation function.

This module generates positions using engine self-play with controlled stochasticity
and trains the GNN model to match the Rust engine's analytical evaluation scores.

Rather than using purely random moves, we generate more realistic positions by:
1. Starting from the initial position
2. Using engine self-play to select moves, with controlled randomness:
   - Sometimes choosing completely random moves
   - Sometimes choosing suboptimal (but good) moves from top-N
   - Otherwise choosing the best engine move
3. Applying branching to create parallel exploration paths
4. Evaluating each unique position with the Rust engine
5. Training the GNN to match these evaluations using MSE loss

This gives the model a good initial understanding of position evaluation
with more game-realistic positions compared to pure random walks, while
still maintaining diversity through stochasticity and branching.
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
    engine_depth: int = 3,
    random_move_probability: float = 0.1,
    suboptimal_move_probability: float = 0.2,
    top_n_moves: int = 3,
    verbose: bool = True,
) -> List[Tuple[List[float], List[List[int]], float]]:
    """
    Generate positions using engine self-play with stochasticity and branching.
    
    This creates diverse positions by:
    1. Starting from initial position (or random branch points)
    2. Using engine self-play to select moves, with controlled randomness:
       - Sometimes choosing completely random moves
       - Sometimes choosing suboptimal (but good) moves from top-N
       - Otherwise choosing the best engine move
    3. Occasionally branching to create parallel exploration paths
    4. Evaluating each unique position with the Rust engine
    
    This gives more realistic game-like positions compared to pure random walks,
    while still maintaining diversity through stochasticity and branching.
    
    Args:
        num_positions: Target number of unique positions to generate
        aggression: Aggression level 1-5 for BasicEvaluator (default: 3)
        max_depth: Maximum number of moves from start position (default: 50)
        branch_probability: Probability of creating a branch point (default: 0.3)
        eval_depth: Search depth for minimax evaluation (default: 0 for static eval)
                   If > 0, uses minimax to evaluate position N moves ahead
        engine_depth: Search depth for engine move selection (default: 3)
        random_move_probability: Probability of choosing a completely random move (default: 0.1)
        suboptimal_move_probability: Probability of choosing a suboptimal move from top-N (default: 0.2)
        top_n_moves: Number of top moves to consider for suboptimal selection (default: 3)
        verbose: Print progress information
        
    Returns:
        List of (node_features, edge_index, eval_score) tuples
        eval_score is the raw evaluation from BasicEvaluator (absolute scale)
    """
    unique_positions = {}  # Map: graph_hash -> (node_features, edge_index, eval_score)
    branch_points = []  # List of (board_state, depth) tuples for branching
    
    if verbose:
        print(f"Generating {num_positions} unique positions using engine self-play with branching...")
        print(f"  Engine depth: {engine_depth}")
        print(f"  Random move probability: {random_move_probability}")
        print(f"  Suboptimal move probability: {suboptimal_move_probability}")
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
        
        # Choose move based on stochastic policy
        rand = random.random()
        if rand < random_move_probability:
            # Completely random move
            move = random.choice(legal_moves)
        elif rand < random_move_probability + suboptimal_move_probability:
            # Choose from top-N moves
            move = _choose_suboptimal_move(board, legal_moves, aggression, engine_depth, top_n_moves)
        else:
            # Choose best engine move
            engine_move = board.get_engine_move(depth=engine_depth, aggression=aggression)
            move = engine_move if engine_move is not None else random.choice(legal_moves)
        
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
            for branch_move in random.sample(other_moves, min(num_branches, len(other_moves))):
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


def _choose_suboptimal_move(board, legal_moves, aggression, engine_depth, top_n):
    """
    Choose a suboptimal move from the top-N best moves.
    
    This evaluates all legal moves, ranks them by evaluation score,
    and randomly selects from the top-N moves.
    
    Args:
        board: Current board state
        legal_moves: List of legal moves
        aggression: Aggression level for evaluation
        engine_depth: Search depth for move evaluation
        top_n: Number of top moves to consider
        
    Returns:
        A randomly selected move from the top-N moves
    """
    if len(legal_moves) <= 1:
        return legal_moves[0] if legal_moves else None
    
    # Evaluate each move
    move_scores = []
    for move in legal_moves:
        board.apply(move)
        # Get evaluation from the perspective of the player who just moved
        # (we want to maximize this score)
        score = -board.get_evaluation(aggression, engine_depth)
        board.undo(move)
        move_scores.append((move, score))
    
    # Sort by score (descending)
    move_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select from top-N
    top_moves = move_scores[:min(top_n, len(move_scores))]
    selected_move, _ = random.choice(top_moves)
    
    return selected_move


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
        args: Tuple of (batch_size, aggression, max_depth, branch_probability, eval_depth, 
                        engine_depth, random_move_probability, suboptimal_move_probability, 
                        top_n_moves, seed)
        
    Returns:
        List of (node_features, edge_index, eval_score) tuples
    """
    (batch_size, aggression, max_depth, branch_probability, eval_depth, 
     engine_depth, random_move_probability, suboptimal_move_probability, 
     top_n_moves, seed) = args
    
    # Set random seed for this batch
    random.seed(seed)
    
    # Generate a batch of positions using the branching process
    result = generate_random_positions_branching(
        num_positions=batch_size,
        aggression=aggression,
        max_depth=max_depth,
        branch_probability=branch_probability,
        eval_depth=eval_depth,
        engine_depth=engine_depth,
        random_move_probability=random_move_probability,
        suboptimal_move_probability=suboptimal_move_probability,
        top_n_moves=top_n_moves,
        verbose=False,
    )
    
    return result


def _get_cache_key(num_positions: int, aggression: int, max_depth: int, 
                    branch_probability: float, eval_depth: int, engine_depth: int,
                    random_move_probability: float, suboptimal_move_probability: float,
                    top_n_moves: int) -> str:
    """
    Generate a unique cache key for a set of generation parameters.
    
    Args:
        num_positions: Number of positions
        aggression: Aggression level
        max_depth: Maximum depth
        branch_probability: Branch probability
        eval_depth: Evaluation depth
        engine_depth: Engine search depth for move selection
        random_move_probability: Probability of random moves
        suboptimal_move_probability: Probability of suboptimal moves
        top_n_moves: Number of top moves to consider
        
    Returns:
        MD5 hash of the configuration
    """
    config_str = (f"{num_positions}|{aggression}|{max_depth}|{branch_probability}|"
                  f"{eval_depth}|{engine_depth}|{random_move_probability}|"
                  f"{suboptimal_move_probability}|{top_n_moves}")
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
    engine_depth: int = 3,
    random_move_probability: float = 0.1,
    suboptimal_move_probability: float = 0.2,
    top_n_moves: int = 3,
    cache_dir: str = "pretrain_cache",
    num_workers: Optional[int] = None,
    force_refresh: bool = False,
    verbose: bool = True,
) -> str:
    """
    Generate positions using engine self-play and save to disk using multiprocessing.
    
    Similar to human_games.py's generate_human_games_data, this function generates
    positions and saves them in a format suitable for streaming during training.
    
    Args:
        num_positions: Total number of unique positions to generate
        aggression: Aggression level 1-5 for BasicEvaluator (default: 3)
        max_depth: Maximum number of moves from start position (default: 200)
        branch_probability: Probability of creating a branch point (default: 0.3)
        eval_depth: Search depth for minimax evaluation (default: 0 for static eval)
        engine_depth: Search depth for engine move selection (default: 3)
        random_move_probability: Probability of choosing random moves (default: 0.1)
        suboptimal_move_probability: Probability of choosing suboptimal moves (default: 0.2)
        top_n_moves: Number of top moves to consider for suboptimal selection (default: 3)
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
    cache_key = _get_cache_key(num_positions, aggression, max_depth, branch_probability, 
                                eval_depth, engine_depth, random_move_probability, 
                                suboptimal_move_probability, top_n_moves)
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
            print(f"  Engine depth: {metadata['engine_depth']}")
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
        print(f"  Engine depth: {engine_depth}")
        print(f"  Random move probability: {random_move_probability}")
        print(f"  Suboptimal move probability: {suboptimal_move_probability}")
    
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
        work_items.append((current_batch_size, aggression, max_depth, branch_probability, 
                          eval_depth, engine_depth, random_move_probability, 
                          suboptimal_move_probability, top_n_moves, seed))
    
    # Run workers in parallel
    if verbose:
        print(f"\nGenerating {num_positions} positions using {num_workers} workers...")
        print(f"Processing in batches of {batch_size} positions for progress updates...")
    
    # Initialize counters for incremental saving
    total_positions = 0
    positions_per_file = 1000  # Save every 1000 positions to a file
    file_index = 0
    current_batch = []
    
    # Track unique positions across all batches to prevent duplicates
    seen_hashes = set()
    duplicates_skipped = 0
    
    # Track statistics without keeping all positions in memory
    eval_min = float('inf')
    eval_max = float('-inf')
    eval_sum = 0.0
    eval_count = 0
    # For median, we use a sampling approach to avoid storing all values
    # Sample up to 10,000 values for median calculation
    eval_samples_for_median = []
    median_sample_size = 10000
    median_sampling_probability = 1.0  # Start at 100%, will decrease as we get more samples
    
    with Pool(num_workers) as pool:
        if verbose:
            # Use imap to process batches, updating progress bar for each batch
            pbar = tqdm(total=num_positions, desc="Positions generated", unit="pos")
            for batch in pool.imap(_generate_position_batch_worker, work_items):
                # Process each position in the batch
                for position in batch:
                    # Check for duplicates across all batches
                    node_features, edge_index, eval_score = position
                    pos_hash = graph_hash(node_features, edge_index)
                    
                    if pos_hash in seen_hashes:
                        # Skip duplicate position
                        duplicates_skipped += 1
                        continue
                    
                    # Add to seen hashes
                    seen_hashes.add(pos_hash)
                    
                    current_batch.append(position)
                    total_positions += 1
                    
                    # Update statistics
                    eval_min = min(eval_min, eval_score)
                    eval_max = max(eval_max, eval_score)
                    eval_sum += eval_score
                    eval_count += 1
                    
                    # Reservoir sampling for median approximation
                    if len(eval_samples_for_median) < median_sample_size:
                        eval_samples_for_median.append(eval_score)
                    else:
                        # Dynamically adjust sampling probability
                        median_sampling_probability = median_sample_size / eval_count
                        if random.random() < median_sampling_probability:
                            # Replace a random sample
                            eval_samples_for_median[random.randint(0, median_sample_size - 1)] = eval_score
                    
                    # Save batch when it reaches the target size
                    if len(current_batch) >= positions_per_file:
                        batch_file = cache_data_dir / f"positions_{file_index:06d}.pkl"
                        with open(batch_file, 'wb') as f:
                            pickle.dump(current_batch, f)
                        file_index += 1
                        current_batch = []
                
                pbar.update(len(batch))
            pbar.close()
        else:
            for batch in pool.map(_generate_position_batch_worker, work_items):
                # Process each position in the batch
                for position in batch:
                    # Check for duplicates across all batches
                    node_features, edge_index, eval_score = position
                    pos_hash = graph_hash(node_features, edge_index)
                    
                    if pos_hash in seen_hashes:
                        # Skip duplicate position
                        duplicates_skipped += 1
                        continue
                    
                    # Add to seen hashes
                    seen_hashes.add(pos_hash)
                    
                    current_batch.append(position)
                    total_positions += 1
                    
                    # Update statistics
                    eval_min = min(eval_min, eval_score)
                    eval_max = max(eval_max, eval_score)
                    eval_sum += eval_score
                    eval_count += 1
                    
                    # Reservoir sampling for median approximation
                    if len(eval_samples_for_median) < median_sample_size:
                        eval_samples_for_median.append(eval_score)
                    else:
                        # Dynamically adjust sampling probability
                        median_sampling_probability = median_sample_size / eval_count
                        if random.random() < median_sampling_probability:
                            # Replace a random sample
                            eval_samples_for_median[random.randint(0, median_sample_size - 1)] = eval_score
                    
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
        'engine_depth': engine_depth,
        'random_move_probability': random_move_probability,
        'suboptimal_move_probability': suboptimal_move_probability,
        'top_n_moves': top_n_moves,
        'cache_key': cache_key,
    }
    with open(cache_metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Data Generation Complete")
        print(f"{'='*60}")
        print(f"Total unique positions: {total_positions}")
        print(f"Duplicates skipped: {duplicates_skipped}")
        print(f"Saved to: {cache_data_dir}")
        print(f"Number of files: {file_index}")
        
        # Show evaluation statistics
        if total_positions > 0:
            print(f"\nEvaluation statistics:")
            print(f"  Min: {eval_min:.2f}")
            print(f"  Max: {eval_max:.2f}")
            print(f"  Mean: {eval_sum / eval_count:.2f}")
            if eval_samples_for_median:
                median_val = sorted(eval_samples_for_median)[len(eval_samples_for_median) // 2]
                print(f"  Median (approx): {median_val:.2f}")
    
    return str(cache_data_dir)


class EvaluationMatchingDataset(Dataset):
    """
    PyTorch Geometric Dataset for evaluation matching training.
    
    This dataset generates positions on-the-fly each epoch using engine self-play,
    ensuring the model sees diverse training data throughout training.
    """
    
    def __init__(
        self,
        num_positions: int,
        aggression: int = 3,
        max_depth: int = 50,
        branch_probability: float = 0.3,
        eval_depth: int = 0,
        engine_depth: int = 3,
        random_move_probability: float = 0.1,
        suboptimal_move_probability: float = 0.2,
        top_n_moves: int = 3,
        scale: float = 0.001,
        regenerate_each_epoch: bool = True,
    ):
        """
        Initialize the evaluation matching dataset.
        
        Args:
            num_positions: Number of unique positions to generate per epoch
            aggression: Aggression level for BasicEvaluator (1-5)
            max_depth: Maximum depth for position generation
            branch_probability: Probability of branching during generation
            eval_depth: Search depth for minimax evaluation (0 for static eval)
            engine_depth: Search depth for engine move selection
            random_move_probability: Probability of choosing random moves
            suboptimal_move_probability: Probability of choosing suboptimal moves
            top_n_moves: Number of top moves to consider for suboptimal selection
            scale: Scaling factor for normalizing evaluations
            regenerate_each_epoch: If True, generate new positions each epoch
        """
        super().__init__()
        self.num_positions = num_positions
        self.aggression = aggression
        self.max_depth = max_depth
        self.branch_probability = branch_probability
        self.eval_depth = eval_depth
        self.engine_depth = engine_depth
        self.random_move_probability = random_move_probability
        self.suboptimal_move_probability = suboptimal_move_probability
        self.top_n_moves = top_n_moves
        self.scale = scale
        self.regenerate_each_epoch = regenerate_each_epoch
        
        # Generate initial data
        self._regenerate_positions()
    
    def _regenerate_positions(self):
        """Generate new positions using engine self-play."""
        raw_data = generate_random_positions_branching(
            num_positions=self.num_positions,
            aggression=self.aggression,
            max_depth=self.max_depth,
            branch_probability=self.branch_probability,
            eval_depth=self.eval_depth,
            engine_depth=self.engine_depth,
            random_move_probability=self.random_move_probability,
            suboptimal_move_probability=self.suboptimal_move_probability,
            top_n_moves=self.top_n_moves,
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


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split a dataset into train/validation/test sets.
    
    Args:
        dataset: PyTorch Geometric Dataset
        train_ratio: Fraction for training (default: 0.8)
        val_ratio: Fraction for validation (default: 0.1)
        test_ratio: Fraction for testing (default: 0.1)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Set random seed
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    
    # Get total size
    total_size = len(dataset)
    
    # Calculate split sizes
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size  # Ensure all samples are used
    
    # Create random indices
    indices = list(range(total_size))
    random.shuffle(indices)
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset


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
    engine_depth: int = 3,
    random_move_probability: float = 0.1,
    suboptimal_move_probability: float = 0.2,
    top_n_moves: int = 3,
    scale: float = 0.001,
    regenerate_each_epoch: bool = True,
    verbose: bool = True,
    epoch_callback=None,
) -> List[float]:
    """
    Pre-train model to match analytical evaluation scores using engine self-play.
    
    Each epoch:
    1. Generate N unique positions using engine self-play with stochasticity
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
        max_depth: Maximum depth for position generation (default: 50)
        branch_probability: Branching probability during position generation (default: 0.3)
        eval_depth: Search depth for minimax evaluation (default: 0 for static eval)
                   If > 0, uses minimax to evaluate position N moves ahead
        engine_depth: Search depth for engine move selection (default: 3)
        random_move_probability: Probability of choosing random moves (default: 0.1)
        suboptimal_move_probability: Probability of choosing suboptimal moves (default: 0.2)
        top_n_moves: Number of top moves to consider for suboptimal selection (default: 3)
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
        engine_depth=engine_depth,
        random_move_probability=random_move_probability,
        suboptimal_move_probability=suboptimal_move_probability,
        top_n_moves=top_n_moves,
        scale=scale,
        regenerate_each_epoch=regenerate_each_epoch,
    )
    
    if verbose:
        print(f"Training on {len(dataset)} positions per epoch...")
        print(f"Regenerating positions each epoch: {regenerate_each_epoch}")
        print(f"Using engine self-play with:")
        print(f"  Engine depth: {engine_depth}")
        print(f"  Random move probability: {random_move_probability}")
        print(f"  Suboptimal move probability: {suboptimal_move_probability}")
    
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


def pretrain_eval_matching_with_validation(
    model,
    train_dataset,
    val_dataset,
    test_dataset,
    optimizer=None,
    num_epochs: int = 50,
    batch_size: int = 64,
    device: str = "cpu",
    scale: float = 0.001,
    verbose: bool = True,
    checkpoint_callback=None,
    writer=None,
) -> dict:
    """
    Pre-train model with proper train/validation/test split.
    
    This function trains on the training set, validates on the validation set,
    and reports final test performance.
    
    Args:
        model: GNN model to train
        train_dataset: Training dataset (PyTorch Dataset or Subset)
        val_dataset: Validation dataset (PyTorch Dataset or Subset)
        test_dataset: Test dataset (PyTorch Dataset or Subset)
        optimizer: PyTorch optimizer (if None, uses Adam with lr=1e-3)
        num_epochs: Number of training epochs (default: 50)
        batch_size: Batch size for mini-batch training
        device: Device to train on
        scale: Scaling factor for normalizing evaluations (default: 0.001)
        verbose: Print progress information
        checkpoint_callback: Optional callback(epoch, train_loss, val_loss, is_best) -> None
        writer: Optional TensorBoard SummaryWriter for logging
        
    Returns:
        Dictionary with training history and test results:
        {
            'train_losses': List of training losses per epoch,
            'val_losses': List of validation losses per epoch,
            'test_loss': Final test loss,
            'best_epoch': Epoch with best validation loss,
            'best_val_loss': Best validation loss achieved
        }
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    if verbose:
        print(f"\nDataset sizes:")
        print(f"  Training:   {len(train_dataset):,} positions")
        print(f"  Validation: {len(val_dataset):,} positions")
        print(f"  Test:       {len(test_dataset):,} positions")
        print(f"\nTraining for {num_epochs} epochs...")
        print(f"  Batch size: {batch_size}")
        print(f"  Device: {device}")
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in tqdm(range(num_epochs), desc="Training epochs", disable=not verbose):
        # Training phase
        model.train()
        train_total_loss = 0
        train_num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions, _ = model(batch.x, batch.edge_index, batch.batch)
            
            # Extract targets from batch
            targets = batch.y.unsqueeze(1) if batch.y.dim() == 1 else batch.y
            
            # Compute MSE loss
            loss = F.mse_loss(predictions, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_total_loss += loss.item()
            train_num_batches += 1
        
        train_loss = train_total_loss / max(train_num_batches, 1)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_total_loss = 0
        val_num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                
                # Forward pass
                predictions, _ = model(batch.x, batch.edge_index, batch.batch)
                
                # Extract targets from batch
                targets = batch.y.unsqueeze(1) if batch.y.dim() == 1 else batch.y
                
                # Compute MSE loss
                loss = F.mse_loss(predictions, targets)
                
                val_total_loss += loss.item()
                val_num_batches += 1
        
        val_loss = val_total_loss / max(val_num_batches, 1)
        val_losses.append(val_loss)
        
        # Check if this is the best model so far
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
        
        # Log to tensorboard if provided
        if writer is not None:
            writer.add_scalar("Pretrain/TrainLoss", train_loss, epoch)
            writer.add_scalar("Pretrain/ValLoss", val_loss, epoch)
            writer.add_scalar("Pretrain/BestValLoss", best_val_loss, epoch)
        
        # Print progress
        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs}: "
                      f"Train Loss = {train_loss:.6f}, "
                      f"Val Loss = {val_loss:.6f}"
                      f"{' (BEST)' if is_best else ''}")
        
        # Call checkpoint callback if provided
        if checkpoint_callback is not None:
            checkpoint_callback(epoch, train_loss, val_loss, is_best)
    
    # Final evaluation on test set
    if verbose:
        print(f"\nBest validation loss: {best_val_loss:.6f} at epoch {best_epoch + 1}")
        print(f"\nEvaluating on test set...")
    
    model.eval()
    test_total_loss = 0
    test_num_batches = 0
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            
            # Forward pass
            predictions, _ = model(batch.x, batch.edge_index, batch.batch)
            
            # Extract targets from batch
            targets = batch.y.unsqueeze(1) if batch.y.dim() == 1 else batch.y
            
            # Compute MSE loss
            loss = F.mse_loss(predictions, targets)
            
            test_total_loss += loss.item()
            test_num_batches += 1
    
    test_loss = test_total_loss / max(test_num_batches, 1)
    
    if verbose:
        print(f"Test loss: {test_loss:.6f}")
    
    # Log final test loss
    if writer is not None:
        writer.add_scalar("Pretrain/TestLoss", test_loss, num_epochs)
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
    }



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
        description="Generate positions using engine self-play for pre-training"
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
        default=500,
        help="Maximum number of moves from start position (default: 500)"
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
    parser.add_argument(
        "--engine-depth",
        type=int,
        default=3,
        help="Search depth for engine move selection (default: 3)"
    )
    parser.add_argument(
        "--random-move-probability",
        type=float,
        default=0.1,
        help="Probability of choosing completely random moves (default: 0.1)"
    )
    parser.add_argument(
        "--suboptimal-move-probability",
        type=float,
        default=0.2,
        help="Probability of choosing suboptimal moves from top-N (default: 0.2)"
    )
    parser.add_argument(
        "--top-n-moves",
        type=int,
        default=3,
        help="Number of top moves to consider for suboptimal selection (default: 3)"
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
        # Test engine self-play position generation
        print("Testing evaluation matching pre-training with engine self-play...")
        
        # Generate small dataset
        print("\n1. Testing position generation with engine self-play:")
        data = generate_random_positions_branching(
            num_positions=100,
            aggression=3,
            max_depth=30,
            branch_probability=0.3,
            eval_depth=0,  # Static evaluation
            engine_depth=2,  # Shallow engine search for testing
            random_move_probability=0.1,
            suboptimal_move_probability=0.2,
            top_n_moves=3,
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
            engine_depth=2,
            random_move_probability=0.1,
            suboptimal_move_probability=0.2,
            top_n_moves=3,
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
            engine_depth=2,
            random_move_probability=0.1,
            suboptimal_move_probability=0.2,
            top_n_moves=3,
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
            engine_depth=2,
            random_move_probability=0.1,
            suboptimal_move_probability=0.2,
            top_n_moves=3,
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
            engine_depth=args.engine_depth,
            random_move_probability=args.random_move_probability,
            suboptimal_move_probability=args.suboptimal_move_probability,
            top_n_moves=args.top_n_moves,
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
