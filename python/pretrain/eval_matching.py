"""
Pre-training by matching analytical evaluation function.

This module generates training data by having the Rust engine play games
and recording the analytical evaluation scores for each position. The GNN
model is then trained to match these evaluation scores.

This gives the model a good initial understanding of position evaluation
before transitioning to self-play training.
"""

import random
import pickle
import torch
import torch.nn.functional as F
import nokamute
from torch_geometric.data import Batch, Data
from typing import List, Tuple, Optional

import sys
import os
# Add parent directory to path to import graph_utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from graph_utils import graph_hash


def generate_eval_matching_data(
    num_games: int = 100,
    depth: int = 7,
    aggression: int = 3,
    randomness_rate: float = 0.0,
    max_moves: int = 400,
    verbose: bool = True,
) -> List[Tuple[List[float], List[List[int]], float]]:
    """
    Generate training data by having the engine play against itself.
    
    This efficiently captures ALL positions evaluated during the engine's search,
    not just the root positions. This dramatically increases the amount of training
    data without additional computational cost.
    
    At each position, we record:
    - The board state (node_features, edge_index)
    - The analytical evaluation from BasicEvaluator
    
    The engine uses iterative deepening minimax with occasional random moves
    to create diverse positions.
    
    Args:
        num_games: Number of games to generate
        depth: Search depth for the engine (default: 7)
        aggression: Aggression level 1-5 for BasicEvaluator (default: 3)
        randomness_rate: Probability of making a random move instead of best move (default: 0.15)
        max_moves: Maximum moves per game before declaring draw
        verbose: Print progress information
        
    Returns:
        List of (node_features, edge_index, eval_score) tuples
        eval_score is normalized to [-1, 1] range for training
    """
    # Use a dictionary to deduplicate positions by graph hash
    # Map: graph_hash -> (node_features, edge_index, eval_score)
    unique_positions = {}
    position_repetitions = {}  # Track how many times each position was seen
    
    for game_idx in range(num_games):
        if verbose and (game_idx + 1) % 10 == 0:
            print(f"Generating game {game_idx + 1}/{num_games}... ({len(unique_positions)} unique positions so far)")
        
        board = nokamute.Board()
        game_position_hashes = {}  # Track positions for three-fold repetition within this game
        moves_in_game = 0
        
        for move_num in range(max_moves):
            # Check if game is over
            winner = board.get_winner()
            if winner is not None:
                break
            
            # Get legal moves
            legal_moves = board.legal_moves()
            if not legal_moves:
                break
            
            # Track position for repetition detection within the game
            node_features, edge_index = board.to_graph()
            pos_hash = graph_hash(node_features, edge_index)
            game_position_hashes[pos_hash] = game_position_hashes.get(pos_hash, 0) + 1
            
            # Check for three-fold repetition
            if game_position_hashes[pos_hash] >= 3:
                if verbose:
                    print(f"  Game {game_idx + 1}: Draw by repetition after {move_num} moves")
                break
            
            # Choose move: either random or from engine
            if random.random() < randomness_rate:
                # Random move for exploration - still evaluate positions along the way
                move = random.choice(legal_moves)
                
                # Evaluate this root position
                _add_position_if_unique(board, aggression, unique_positions, position_repetitions)
                
                # Also evaluate some candidate positions (sample a few legal moves)
                sample_size = min(5, len(legal_moves))
                for candidate_move in random.sample(legal_moves, sample_size):
                    board.apply(candidate_move)
                    _add_position_if_unique(board, aggression, unique_positions, position_repetitions)
                    board.undo(candidate_move)
            else:
                # Use engine to find best move
                # IMPORTANT: Evaluate all positions the engine explores
                _collect_positions_from_search(
                    board, legal_moves, depth, aggression, unique_positions, position_repetitions
                )
                
                move = board.get_engine_move(depth=depth, aggression=aggression)
                if move is None:
                    break
            
            # Apply move
            board.apply(move)
            moves_in_game += 1
        
        if verbose:
            winner = board.get_winner()
            if winner:
                print(f"  Game {game_idx + 1}: {winner} after {moves_in_game} moves")
            elif moves_in_game >= max_moves:
                print(f"  Game {game_idx + 1}: Draw by move limit after {moves_in_game} moves")
    
    # Convert dictionary to list
    training_data = list(unique_positions.values())
    
    if verbose:
        print(f"\nGenerated {len(training_data)} unique positions from {num_games} games")
        print(f"Average positions per game: {len(training_data) / num_games:.1f}")
        
        # Show evaluation statistics
        evals = [score for _, _, score in training_data]
        print(f"Evaluation stats:")
        print(f"  Min: {min(evals)}")
        print(f"  Max: {max(evals)}")
        print(f"  Mean: {sum(evals) / len(evals):.2f}")
        print(f"  Median: {sorted(evals)[len(evals) // 2]}")
        
        # Show most frequently seen positions
        top_repetitions = sorted(position_repetitions.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\nMost repeated positions:")
        for hash_val, count in top_repetitions:
            print(f"  Hash {hash_val}: seen {count} times")
    
    return training_data


def _add_position_if_unique(board, aggression, unique_positions, position_repetitions):
    """
    Add a position to the training data if it hasn't been seen before.
    
    Args:
        board: Current board state
        aggression: Aggression level for evaluation
        unique_positions: Dictionary of unique positions (modified in-place)
        position_repetitions: Dictionary tracking repetition counts (modified in-place)
    """
    # Get graph representation and hash
    node_features, edge_index = board.to_graph()
    pos_hash = graph_hash(node_features, edge_index)
    
    # Track how many times we've seen this position
    position_repetitions[pos_hash] = position_repetitions.get(pos_hash, 0) + 1
    
    # Only add if we haven't seen it before
    if pos_hash not in unique_positions:
        eval_score = board.get_evaluation(aggression)
        unique_positions[pos_hash] = (node_features, edge_index, eval_score)


def _collect_positions_from_search(board, legal_moves, depth, aggression, unique_positions, position_repetitions):
    """
    Collect all positions that would be evaluated during a minimax search.
    
    This simulates what the engine does internally, exploring the game tree
    up to the specified depth and recording all evaluated positions.
    
    Args:
        board: Current board state
        legal_moves: List of legal moves from current position
        depth: Search depth
        aggression: Aggression level for evaluation
        unique_positions: Dictionary of unique positions (modified in-place)
        position_repetitions: Dictionary tracking repetition counts (modified in-place)
    """
    # Add the root position
    _add_position_if_unique(board, aggression, unique_positions, position_repetitions)
    
    if depth <= 0:
        return
    
    # Explore all legal moves
    for move in legal_moves:
        board.apply(move)
        
        # Check if this position ends the game
        winner = board.get_winner()
        if winner is None:
            # Recursively collect positions from this branch
            next_legal_moves = board.legal_moves()
            _collect_positions_from_search(
                board, next_legal_moves, depth - 1, aggression, unique_positions, position_repetitions
            )
        else:
            # Terminal position - still add it
            _add_position_if_unique(board, aggression, unique_positions, position_repetitions)
        
        board.undo(move)



def normalize_evaluation(eval_score: float, scale: float = 0.001) -> float:
    """
    Normalize evaluation scores to [-1, 1] range using tanh-like scaling.
    
    Args:
        eval_score: Raw evaluation score from BasicEvaluator
        scale: Scaling factor (default: 0.001, which maps 1000 -> ~0.76)
        
    Returns:
        Normalized score in approximately [-1, 1] range
    """
    # Use tanh to smoothly map to [-1, 1]
    # This handles extreme values gracefully
    import torch
    return torch.tanh(torch.tensor(eval_score * scale)).item()


def pretrain_eval_matching(
    model,
    training_data: List[Tuple[List[float], List[List[int]], float]],
    optimizer,
    num_epochs: int = 50,
    batch_size: int = 64,
    device: str = "cpu",
    scale: float = 0.001,
    verbose: bool = True,
) -> List[float]:
    """
    Pre-train model to match analytical evaluation scores.
    
    Args:
        model: GNN model to train
        training_data: List of (node_features, edge_index, eval_score) tuples
        optimizer: PyTorch optimizer
        num_epochs: Number of training epochs (default: 50)
        batch_size: Batch size for training
        device: Device to train on
        scale: Scaling factor for normalizing evaluations (default: 0.001)
        verbose: Print progress information
        
    Returns:
        List of average losses per epoch
    """
    model.train()
    epoch_losses = []
    
    for epoch in range(num_epochs):
        # Shuffle training data
        shuffled_data = training_data.copy()
        random.shuffle(shuffled_data)
        
        total_loss = 0
        num_batches = 0
        
        # Train in batches
        for i in range(0, len(shuffled_data), batch_size):
            batch_data = shuffled_data[i : i + batch_size]
            
            # Convert to PyG Data objects
            data_list = []
            targets = []
            
            for node_features, edge_index, eval_score in batch_data:
                # Skip empty graphs
                if len(node_features) == 0:
                    continue
                    
                # Convert to tensors
                x = torch.tensor(node_features, dtype=torch.float32)
                edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
                data = Data(x=x, edge_index=edge_index_tensor)
                data_list.append(data)
                
                # Normalize evaluation score
                normalized_score = normalize_evaluation(eval_score, scale)
                targets.append(normalized_score)
            
            if len(data_list) == 0:
                continue
            
            # Create batch
            batch = Batch.from_data_list(data_list).to(device)
            targets = torch.tensor(targets, dtype=torch.float32).to(device).unsqueeze(1)
            
            # Forward pass
            optimizer.zero_grad()
            predictions, _ = model(batch.x, batch.edge_index, batch.batch)
            
            # Ensure predictions and targets have the same shape
            # This handles cases where the batch might be smaller than batch_size
            assert predictions.shape == targets.shape, f"Shape mismatch: predictions {predictions.shape} vs targets {targets.shape}"
            
            # Compute MSE loss
            loss = F.mse_loss(predictions, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        epoch_losses.append(avg_loss)
        
        if verbose:
            print(f"Epoch {epoch + 1}/{num_epochs}: Loss = {avg_loss:.6f}")
    
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


if __name__ == "__main__":
    # Test data generation
    print("Testing evaluation matching pre-training...")
    
    # Generate small dataset
    data = generate_eval_matching_data(
        num_games=5,
        depth=3,  # Shallow for testing
        randomness_rate=0.2,
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
