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

import sys
import os
# Add parent directory to path to import graph_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph_utils import networkx_to_pyg, graph_hash


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
            
            # Compute final result from perspective of player who just moved
            # (the player who made the move that led to next_features)
            current_player = board.to_move()  # This is the NEXT player to move
            
            # The player who just moved is the opposite color
            if current_player.name == "Black":
                player_who_moved = "White"
            else:
                player_who_moved = "Black"
            
            # Result from perspective of player who moved
            if winner == player_who_moved:
                final_result = 1.0
            else:
                final_result = -1.0
            
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
) -> List[Tuple]:
    """
    Download and parse human games from Google Sheets to create training data.
    
    This function downloads games from one or more Google Sheets, parses the
    game logs, and extracts state transitions for TD learning.
    
    Args:
        sheet_urls: List of Google Sheets URLs to download from
        game_type: Game type string (default: "Base+MLP")
        max_games: Maximum number of games to process (None = all)
        verbose: Print progress information
        
    Returns:
        List of (curr_features, curr_edge_index, next_features, next_edge_index, is_terminal, final_result) tuples
    """
    all_transitions = []
    total_games_processed = 0
    total_games_failed = 0
    
    for sheet_url in sheet_urls:
        print(f"\nProcessing sheet: {sheet_url}")
        
        # Download games
        games = download_games_from_google_sheets(sheet_url, verbose=verbose)
        
        # Limit number of games if specified
        if max_games and total_games_processed >= max_games:
            print(f"Reached max_games limit ({max_games})")
            break
        
        if max_games:
            remaining = max_games - total_games_processed
            games = games[:remaining]
        
        # Parse each game
        for i, (game_log, winner) in enumerate(games):
            if verbose and (i + 1) % 100 == 0:
                print(f"  Parsing game {i + 1}/{len(games)}...")
            
            transitions = parse_game_log(game_log, winner, game_type, verbose=False)
            
            if transitions:
                all_transitions.extend(transitions)
                total_games_processed += 1
            else:
                total_games_failed += 1
                if verbose and total_games_failed <= 5:  # Show first 5 failures
                    print(f"  Failed to parse game {i + 1}: {game_log[:80]}...")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Data Generation Complete")
        print(f"{'='*60}")
        print(f"Successfully processed: {total_games_processed} games")
        print(f"Failed to parse: {total_games_failed} games")
        print(f"Total transitions: {len(all_transitions)}")
        print(f"Avg transitions per game: {len(all_transitions) / max(total_games_processed, 1):.1f}")
    
    return all_transitions


def save_human_games_data(training_data: List[Tuple], filepath: str):
    """
    Save human games training data to disk.
    
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
    
    Args:
        filepath: Path to load the data from
        
    Returns:
        List of transition tuples
    """
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} transitions from {filepath}")
    return data


if __name__ == "__main__":
    # Test data generation
    print("Testing human games pre-training...")
    
    # Example URLs (these are the ones provided by the user)
    sheet_urls = [
        "https://docs.google.com/spreadsheets/d/1d4YW6PDdYYSH8sGDQ-FB1zmdNJhNU-f2811o5JxMqsc/edit?gid=1437601135#gid=1437601135",
        "https://docs.google.com/spreadsheets/d/1JGc6hbmQITjqF-MLiW4AGZ24VpAFgN802EY179RrPII/edit?gid=194203898#gid=194203898",
    ]
    
    # Generate small dataset for testing
    data = generate_human_games_data(
        sheet_urls=sheet_urls,
        max_games=5,
        verbose=True
    )
    
    print(f"\nGenerated {len(data)} training examples")
    
    if data:
        # Show sample transition
        curr_features, curr_edge_index, next_features, next_edge_index, is_terminal, final_result = data[0]
        print(f"\nSample transition:")
        print(f"  Current state nodes: {len(curr_features)}")
        print(f"  Next state nodes: {len(next_features)}")
        print(f"  Is terminal: {is_terminal}")
        print(f"  Final result: {final_result}")
