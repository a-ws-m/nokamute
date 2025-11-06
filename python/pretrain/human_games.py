"""
Pre-training by learning from human games.

This module downloads human game data from Google Sheets, parses the game logs,
and generates training data using TD learning. The model learns to evaluate
positions from actual human games, which provides a different learning signal
compared to engine self-play.

The TD learning approach means we extract state transitions from human games
and compute TD targets on-the-fly during training, allowing the model to
continuously improve its value estimates.
"""

import random
import pickle
import torch
import torch.nn.functional as F
import nokamute
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from typing import List, Tuple, Optional
import csv
import io
import requests
import hashlib
from pathlib import Path
from tqdm import tqdm

import sys
import os
# Add parent directory to path to import graph_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph_utils import networkx_to_pyg, graph_hash


def _get_cache_key(sheet_urls: List[str], game_type: str) -> str:
    """
    Generate a unique cache key for a set of sheet URLs and game type.
    
    Args:
        sheet_urls: List of Google Sheets URLs
        game_type: Game type string
        
    Returns:
        MD5 hash of the configuration
    """
    # Sort URLs for consistent hashing
    sorted_urls = sorted(sheet_urls)
    config_str = f"{game_type}||{'||'.join(sorted_urls)}"
    return hashlib.md5(config_str.encode()).hexdigest()


def _get_cache_path(cache_dir: str, cache_key: str) -> Path:
    """
    Get the path to a cached data file.
    
    Args:
        cache_dir: Directory to store cache files
        cache_key: Unique cache key
        
    Returns:
        Path to cache file
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return cache_path / f"human_games_{cache_key}.pkl"


def download_games_from_google_sheets(
    sheet_url: str,
    verbose: bool = True,
) -> List[Tuple[str, str]]:
    """
    Download game data from a Google Sheets URL.
    
    The function converts the Google Sheets URL to a CSV export URL
    and downloads the game data.
    
    Args:
        sheet_url: URL to the Google Sheets document
        verbose: Print progress information
        
    Returns:
        List of (game_log, result) tuples where:
        - game_log is the UHP format game string
        - result is "Black" or "White" indicating the winner
    """
    # Convert Google Sheets URL to CSV export URL
    # Example: https://docs.google.com/spreadsheets/d/SHEET_ID/edit?gid=GID#gid=GID
    # Becomes: https://docs.google.com/spreadsheets/d/SHEET_ID/export?format=csv&gid=GID
    
    if "/edit" in sheet_url:
        # Extract sheet ID and gid
        parts = sheet_url.split("/")
        sheet_id = None
        for i, part in enumerate(parts):
            if part == "d" and i + 1 < len(parts):
                sheet_id = parts[i + 1]
                break
        
        gid = None
        if "gid=" in sheet_url:
            gid_part = sheet_url.split("gid=")[1].split("#")[0].split("&")[0]
            gid = gid_part
        
        if sheet_id:
            csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            if gid:
                csv_url += f"&gid={gid}"
        else:
            raise ValueError(f"Could not extract sheet ID from URL: {sheet_url}")
    else:
        csv_url = sheet_url
    
    if verbose:
        print(f"Downloading from: {csv_url}")
    
    # Download CSV data
    response = requests.get(csv_url)
    response.raise_for_status()
    
    # Parse CSV
    games = []
    csv_reader = csv.DictReader(io.StringIO(response.text))
    
    for row in csv_reader:
        # Try different possible column names for game log and result
        game_log = None
        result = None
        
        # Common column names for game log
        for col_name in ["moves", "Moves", "gamelog", "GameLog", "Game Log", "game_log",
                        "GameString", "Game String", "gamestring", "game_string", 
                        "MoveString", "Move String"]:
            if col_name in row and row[col_name]:
                game_log = row[col_name].strip()
                break
        
        # Common column names for result/winner
        for col_name in ["result", "Result", "Winner", "winner", 
                        "WinningColor", "Winning Color", "winning_color"]:
            if col_name in row and row[col_name]:
                result = row[col_name].strip()
                break
        
        if game_log and result:
            # Normalize result to "Black" or "White"
            if "black" in result.lower() or result.lower() in ["b", "1"]:
                result = "Black"
            elif "white" in result.lower() or result.lower() in ["w", "2", "0"]:
                result = "White"
            else:
                # Skip draws or unclear results
                continue
            
            games.append((game_log, result))
    
    if verbose:
        print(f"Downloaded {len(games)} games")
        if games:
            print(f"Sample game log: {games[0][0][:100]}...")
            print(f"Sample result: {games[0][1]}")
    
    return games


def normalize_move_string(move_str: str) -> str:
    """
    Normalize a move string to handle variations in notation.
    
    Some game logs include bug numbers (e.g., 'wL1') even for bugs that only
    have one piece (Queen, Ladybug, Mosquito, Pillbug). This function removes
    the '1' suffix for single-piece bugs.
    
    Args:
        move_str: Move string to normalize
        
    Returns:
        Normalized move string
    """
    # Bugs that only have one piece don't need a number
    single_piece_bugs = ['Q', 'L', 'M', 'P']
    
    # Split the move into parts (e.g., "wB2 bM1" -> ["wB2", "bM1"])
    parts = move_str.split()
    normalized_parts = []
    
    for part in parts:
        # Check if this is a piece reference (starts with color)
        if len(part) >= 3 and part[0] in ['w', 'b']:
            bug_type = part[1]
            
            # If it's a single-piece bug and has '1' as the third character, remove it
            if bug_type in single_piece_bugs and len(part) > 2 and part[2] == '1':
                # Remove the '1' and keep the rest (e.g., 'wL1' -> 'wL', 'bM1' -> 'bM')
                # But preserve any direction markers or other suffixes
                normalized_parts.append(part[0:2] + part[3:])
            else:
                normalized_parts.append(part)
        else:
            # Not a piece reference, keep as-is (could be a direction marker or standalone piece)
            normalized_parts.append(part)
    
    return ' '.join(normalized_parts)


def parse_game_log(
    game_log: str,
    winner: str,
    game_type: str = "Base+MLP",
    verbose: bool = False,
) -> Optional[List[Tuple]]:
    """
    Parse a game log string and extract state transitions.
    
    Args:
        game_log: Move string (can be comma-separated or semicolon-separated)
        winner: "Black" or "White" indicating who won
        game_type: Game type string (default: "Base+MLP")
        verbose: Print parsing errors
        
    Returns:
        List of (board_state, next_board_state, is_terminal, final_result) tuples
        or None if the game could not be parsed
    """
    try:
        # Create initial board
        board = nokamute.Board.from_game_string(f"{game_type};NotStarted;White[1]")
        
        # Split moves - handle both comma and semicolon separators
        if ',' in game_log:
            moves = [m.strip() for m in game_log.split(",") if m.strip()]
        else:
            moves = [m.strip() for m in game_log.split(";") if m.strip()]
        
        if len(moves) == 0:
            return None
        
        # Play through the game and record transitions
        transitions = []
        
        for i, move_str in enumerate(moves):
            # Get current state
            curr_features, curr_edge_index = board.to_graph()
            
            # Normalize and parse move
            normalized_move = move_str
            try:
                normalized_move = normalize_move_string(move_str)
                move = board.parse_move(normalized_move)
                board.apply(move)
            except Exception as e:
                if verbose:
                    print(f"Error parsing move '{move_str}' (normalized: '{normalized_move}'): {e}")
                return None
            
            # Skip if current state was empty (first move)
            if len(curr_features) == 0:
                continue
            
            # Get next state
            next_features, next_edge_index = board.to_graph()
            
            # Determine if terminal
            is_terminal = (i == len(moves) - 1)
            
            # Compute final result on ABSOLUTE scale:
            # Positive = White won, Negative = Black won, Zero = Draw
            # This is independent of whose turn it is
            if winner == "White":
                final_result = 1.0
            elif winner == "Black":
                final_result = -1.0
            else:
                # Draw (shouldn't happen in this data, but handle it)
                final_result = 0.0
            
            transitions.append((
                curr_features,
                curr_edge_index,
                next_features,
                next_edge_index,
                is_terminal,
                final_result
            ))
        
        return transitions
        
    except Exception as e:
        if verbose:
            print(f"Error parsing game: {e}")
        return None


def generate_human_games_data(
    sheet_urls: List[str],
    game_type: str = "Base+MLP",
    max_games: Optional[int] = None,
    verbose: bool = True,
    cache_dir: Optional[str] = None,
    force_refresh: bool = False,
) -> str:
    """
    Download and parse human games from Google Sheets to create training data.
    
    This function downloads games from one or more Google Sheets, parses the
    game logs, and extracts state transitions for TD learning. Transitions are
    saved to disk immediately (streaming mode) to avoid memory issues.
    
    Results are automatically cached to disk to avoid re-downloading and re-parsing.
    
    Args:
        sheet_urls: List of Google Sheets URLs to download from
        game_type: Game type string (default: "Base+MLP")
        max_games: Maximum number of games to process (None = all)
        verbose: Print progress information
        cache_dir: Directory to store cached data (default: ./pretrain_cache)
        force_refresh: If True, ignore cache and re-download data
        
    Returns:
        Path to the cache directory containing transition files
    """
    # Set default cache directory
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "pretrain_cache")
    
    # Check cache first
    cache_key = _get_cache_key(sheet_urls, game_type)
    cache_base_path = _get_cache_path(cache_dir, cache_key)
    
    # For streaming, we use a directory instead of a single file
    cache_data_dir = cache_base_path.parent / f"human_games_{cache_key}"
    cache_metadata_path = cache_data_dir / "metadata.pkl"
    
    if not force_refresh and cache_metadata_path.exists():
        if verbose:
            print(f"Found cached data in {cache_data_dir}")
        # Load metadata to get statistics
        with open(cache_metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        if verbose:
            print(f"Loaded metadata: {metadata['total_games']} games, {metadata['total_transitions']} transitions")
        
        return str(cache_data_dir)
    
    # No cache or force refresh - download and parse
    if verbose:
        if force_refresh:
            print("Force refresh enabled - ignoring cache")
        print("Downloading and parsing human games (streaming to disk)...")
    
    # Create cache directory
    cache_data_dir.mkdir(parents=True, exist_ok=True)
    
    total_games_processed = 0
    total_games_failed = 0
    total_transitions = 0
    transition_file_index = 0
    transitions_per_file = 1000  # Save every 1000 transitions to a file
    current_batch = []
    
    for sheet_url in tqdm(sheet_urls, desc="Processing sheets", disable=not verbose):
        if verbose:
            print(f"\nProcessing sheet: {sheet_url}")
        
        # Download games
        games = download_games_from_google_sheets(sheet_url, verbose=verbose)
        
        # Limit number of games if specified
        if max_games and total_games_processed >= max_games:
            if verbose:
                print(f"Reached max_games limit ({max_games})")
            break
        
        if max_games:
            remaining = max_games - total_games_processed
            games = games[:remaining]
        
        # Parse each game with progress bar
        failed_games = []
        for i, (game_log, winner) in enumerate(tqdm(games, desc="Parsing games", disable=not verbose)):
            transitions = parse_game_log(game_log, winner, game_type, verbose=False)
            
            if transitions:
                current_batch.extend(transitions)
                total_games_processed += 1
                total_transitions += len(transitions)
                
                # Write batch to disk when it reaches threshold
                if len(current_batch) >= transitions_per_file:
                    batch_file = cache_data_dir / f"transitions_{transition_file_index:06d}.pkl"
                    with open(batch_file, 'wb') as f:
                        pickle.dump(current_batch, f)
                    transition_file_index += 1
                    current_batch = []
            else:
                total_games_failed += 1
                if verbose and total_games_failed <= 5:  # Show first 5 failures
                    failed_games.append((i + 1, game_log[:80]))
        
        # Print failed games at the end
        if verbose and failed_games:
            print(f"\nFailed to parse {len(failed_games)} games (showing first 5):")
            for game_idx, log_preview in failed_games[:5]:
                print(f"  Game {game_idx}: {log_preview}...")
    
    # Write any remaining transitions
    if len(current_batch) > 0:
        batch_file = cache_data_dir / f"transitions_{transition_file_index:06d}.pkl"
        with open(batch_file, 'wb') as f:
            pickle.dump(current_batch, f)
        transition_file_index += 1
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Data Generation Complete")
        print(f"{'='*60}")
        print(f"Successfully processed: {total_games_processed} games")
        print(f"Failed to parse: {total_games_failed} games")
        print(f"Total transitions: {total_transitions}")
        print(f"Avg transitions per game: {total_transitions / max(total_games_processed, 1):.1f}")
        print(f"Saved to {transition_file_index} files in {cache_data_dir}")
    
    # Save metadata
    metadata = {
        'total_games': total_games_processed,
        'total_transitions': total_transitions,
        'num_files': transition_file_index,
        'sheet_urls': sheet_urls,
        'game_type': game_type,
        'cache_key': cache_key,
    }
    with open(cache_metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    if verbose:
        print(f"Saved metadata to {cache_metadata_path}")
    
    return str(cache_data_dir)


def save_human_games_data(training_data: List[Tuple], filepath: str):
    """
    Save human games training data to disk.
    
    DEPRECATED: Use generate_human_games_data() which now streams to disk automatically.
    This function is kept for backward compatibility.
    
    Args:
        training_data: List of transition tuples
        filepath: Path to save the data
    """
    with open(filepath, 'wb') as f:
        pickle.dump(training_data, f)
    print(f"Saved {len(training_data)} transitions to {filepath}")


def load_human_games_data(filepath: str) -> List[Tuple]:
    """
    Load human games training data from disk.
    
    DEPRECATED: Use HumanGamesDataset for streaming data instead.
    This function is kept for backward compatibility.
    
    Args:
        filepath: Path to load the data from
        
    Returns:
        List of transition tuples
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} transitions from {filepath}")
    return data


class HumanGamesDataset:
    """
    Dataset class that streams human games transitions from disk.
    
    This class reads transition files on-demand, avoiding loading all data into memory.
    It can be used with PyTorch DataLoader or iterated directly.
    """
    
    def __init__(self, cache_dir: str, shuffle: bool = True, verbose: bool = True):
        """
        Initialize the dataset.
        
        Args:
            cache_dir: Directory containing transition files and metadata
            shuffle: Whether to shuffle the data
            verbose: Print loading information
        """
        self.cache_dir = Path(cache_dir)
        self.shuffle = shuffle
        self.verbose = verbose
        
        # Load metadata
        metadata_path = self.cache_dir / "metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Find all transition files
        self.transition_files = sorted(self.cache_dir.glob("transitions_*.pkl"))
        
        if len(self.transition_files) == 0:
            raise FileNotFoundError(f"No transition files found in {cache_dir}")
        
        if verbose:
            print(f"HumanGamesDataset initialized:")
            print(f"  Cache dir: {cache_dir}")
            print(f"  Total games: {self.metadata['total_games']}")
            print(f"  Total transitions: {self.metadata['total_transitions']}")
            print(f"  Files: {len(self.transition_files)}")
        
        # Optionally shuffle file order
        if self.shuffle:
            random.shuffle(self.transition_files)
    
    def __len__(self):
        """Return total number of transitions."""
        return self.metadata['total_transitions']
    
    def __iter__(self):
        """
        Iterate over all transitions, streaming from disk.
        
        Yields:
            Transition tuples: (curr_features, curr_edge_index, next_features, 
                               next_edge_index, is_terminal, final_result)
        """
        files_to_process = list(self.transition_files)
        if self.shuffle:
            random.shuffle(files_to_process)
        
        for file_path in files_to_process:
            # Load one file at a time
            with open(file_path, 'rb') as f:
                transitions = pickle.load(f)
            
            # Optionally shuffle transitions within file
            if self.shuffle:
                random.shuffle(transitions)
            
            # Yield each transition
            for transition in transitions:
                yield transition
    
    def iter_batches(self, batch_size: int):
        """
        Iterate over transitions in batches, streaming from disk.
        
        Args:
            batch_size: Number of transitions per batch
            
        Yields:
            List of transition tuples (batch)
        """
        batch = []
        for transition in self:
            batch.append(transition)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        # Yield remaining items
        if len(batch) > 0:
            yield batch
    
    def get_metadata(self):
        """Return dataset metadata."""
        return self.metadata.copy()


def train_epoch_streaming(
    model,
    dataset: HumanGamesDataset,
    optimizer,
    batch_size: int = 32,
    device: str = "cpu",
    gamma: float = 0.99,
    verbose: bool = True,
):
    """
    Train the model for one epoch using streaming TD learning from disk.
    
    This function streams transitions from disk rather than loading everything
    into memory, making it suitable for large datasets.
    
    Args:
        model: GNN model
        dataset: HumanGamesDataset instance
        optimizer: Optimizer
        batch_size: Batch size
        device: Device to train on
        gamma: TD discount factor
        verbose: Print progress
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Create progress bar if verbose
    batch_iter = dataset.iter_batches(batch_size)
    if verbose:
        batch_iter = tqdm(
            batch_iter,
            desc="Training",
            total=len(dataset) // batch_size
        )
    
    for batch_transitions in batch_iter:
        curr_states = []
        td_targets_list = []
        
        for transition in batch_transitions:
            curr_features, curr_edge_index, next_features, next_edge_index, is_terminal, final_result = transition
            
            # Skip empty states
            if len(curr_features) == 0:
                continue
            
            # Create PyG Data object for current state
            curr_x = torch.tensor(curr_features, dtype=torch.float32)
            curr_edge_index_tensor = torch.tensor(curr_edge_index, dtype=torch.long)
            curr_data = Data(x=curr_x, edge_index=curr_edge_index_tensor)
            curr_states.append(curr_data)
            
            # Compute TD target
            if is_terminal:
                # Terminal state: target is the final result (absolute scale)
                td_target = final_result
            else:
                # Non-terminal: compute V(s_{t+1}) using current model
                if len(next_features) > 0:
                    next_x = torch.tensor(next_features, dtype=torch.float32, device=device)
                    next_edge_index_tensor = torch.tensor(next_edge_index, dtype=torch.long, device=device)
                    
                    with torch.no_grad():
                        # Create batch with single item for next state
                        next_batch = torch.zeros(len(next_features), dtype=torch.long, device=device)
                        next_value, _ = model(next_x, next_edge_index_tensor, next_batch)
                        next_value = next_value.item()
                    
                    # TD target: gamma * V(s_{t+1})
                    # Both next_value and final_result are on absolute scale
                    # (positive = White advantage, negative = Black advantage)
                    td_target = gamma * next_value
                else:
                    # Empty next state (shouldn't happen but handle gracefully)
                    td_target = final_result
            
            td_targets_list.append(td_target)
        
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
        
        # Update progress bar
        if verbose and hasattr(batch_iter, 'set_postfix'):
            batch_iter.set_postfix({"loss": f"{loss.item():.6f}"})
    
    return total_loss / max(num_batches, 1)


if __name__ == "__main__":
    # Test data generation
    print("Testing human games pre-training...")
    
    # Example URLs (these are the ones provided by the user)
    sheet_urls = [
        "https://docs.google.com/spreadsheets/d/1d4YW6PDdYYSH8sGDQ-FB1zmdNJhNU-f2811o5JxMqsc/edit?gid=1437601135#gid=1437601135",
        "https://docs.google.com/spreadsheets/d/1JGc6hbmQITjqF-MLiW4AGZ24VpAFgN802EY179RrPII/edit?gid=194203898#gid=194203898",
    ]
    
    # Generate small dataset for testing (streams to disk)
    cache_dir = generate_human_games_data(
        sheet_urls=sheet_urls,
        max_games=5,
        verbose=True
    )
    
    print(f"\nData saved to: {cache_dir}")
    
    # Test streaming dataset
    print("\nTesting HumanGamesDataset...")
    dataset = HumanGamesDataset(cache_dir, shuffle=False, verbose=True)
    
    print(f"\nDataset contains {len(dataset)} transitions")
    print(f"Metadata: {dataset.get_metadata()}")
    
    # Test iteration
    print("\nTesting iteration (first 3 transitions):")
    for i, transition in enumerate(dataset):
        if i >= 3:
            break
        curr_features, curr_edge_index, next_features, next_edge_index, is_terminal, final_result = transition
        print(f"\nTransition {i+1}:")
        print(f"  Current state nodes: {len(curr_features)}")
        print(f"  Next state nodes: {len(next_features)}")
        print(f"  Is terminal: {is_terminal}")
        print(f"  Final result: {final_result}")
    
    # Test batch iteration
    print("\nTesting batch iteration (batch_size=2):")
    for i, batch in enumerate(dataset.iter_batches(batch_size=2)):
        if i >= 2:
            break
        print(f"Batch {i+1}: {len(batch)} transitions")
