"""
Evaluation module for testing ML models against the Rust engine.
"""

import time
from typing import Dict, List, Optional, Tuple

import torch

import nokamute
from elo_tracker import EloTracker
from model import HiveGNN


class EngineOpponent:
    """
    Opponent that uses the Rust minimax engine.
    """

    def __init__(self, depth: int = 3, time_limit_ms: Optional[int] = None, aggression: int = 3):
        """
        Args:
            depth: Search depth for minimax
            time_limit_ms: Time limit in milliseconds (overrides depth if set)
            aggression: Aggression level 1-5
        """
        self.depth = depth
        self.time_limit_ms = time_limit_ms
        self.aggression = aggression
        self.name = f"engine_depth_{depth}" if time_limit_ms is None else f"engine_time_{time_limit_ms}ms"

    def select_move(self, board: nokamute.Board) -> Optional[nokamute.Turn]:
        """
        Select a move using the engine.
        
        Args:
            board: Current board state
            
        Returns:
            Selected move or None if game is over
        """
        return board.get_engine_move(
            depth=self.depth if self.time_limit_ms is None else None,
            time_limit_ms=self.time_limit_ms,
            aggression=self.aggression,
        )


class MLPlayer:
    """
    Player that uses the trained GNN model.
    """

    def __init__(self, model: HiveGNN, temperature: float = 0.1, device: str = "cpu"):
        """
        Args:
            model: Trained GNN model
            temperature: Temperature for move selection (0 = greedy)
            device: Device to run model on
        """
        self.model = model
        self.temperature = temperature
        self.device = device
        self.model.eval()

    def select_move(self, board: nokamute.Board) -> Optional[nokamute.Turn]:
        """
        Select a move using the model.
        
        Args:
            board: Current board state
            
        Returns:
            Selected move or None if no legal moves
        """
        import numpy as np

        legal_moves = board.legal_moves()
        
        if len(legal_moves) == 0:
            return None
        
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Evaluate each move
        move_values = []
        for move in legal_moves:
            board_copy = board.clone()
            board_copy.apply(move)
            
            # Convert to graph
            node_features, edge_index = board_copy.to_graph()
            
            if len(node_features) == 0:
                value = 0.0
            else:
                x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
                edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).to(self.device)
                
                with torch.no_grad():
                    value = self.model.predict_value(x, edge_index_tensor).item()
            
            # Negate for opponent's perspective
            move_values.append(-value)
            
            board_copy.undo(move)

        # Select move based on temperature
        if self.temperature == 0:
            best_idx = np.argmax(move_values)
            return legal_moves[best_idx]
        else:
            values_array = np.array(move_values)
            exp_values = np.exp(values_array / self.temperature)
            probs = exp_values / np.sum(exp_values)
            selected_idx = np.random.choice(len(legal_moves), p=probs)
            return legal_moves[selected_idx]


def play_game(
    player1,
    player2,
    max_moves: int = 200,
    verbose: bool = False,
) -> Tuple[Optional[str], int, Dict]:
    """
    Play a game between two players.
    
    Args:
        player1: First player (plays Black)
        player2: Second player (plays White)
        max_moves: Maximum moves before declaring draw
        verbose: Print game progress
        
    Returns:
        Tuple of (winner, num_moves, metadata)
        winner is "player1", "player2", "draw", or None
    """
    board = nokamute.Board()
    num_moves = 0
    
    start_time = time.time()
    
    for move_num in range(max_moves):
        # Check for game over
        winner = board.get_winner()
        if winner is not None:
            result = None
            if winner == "Draw":
                result = "draw"
            elif winner == "Black":
                result = "player1"
            elif winner == "White":
                result = "player2"
            
            metadata = {
                "num_moves": num_moves,
                "duration": time.time() - start_time,
                "outcome": winner,
            }
            
            if verbose:
                print(f"Game over: {result} ({winner}) after {num_moves} moves")
            
            return result, num_moves, metadata
        
        # Select player
        current_player = player1 if board.to_move().name == "Black" else player2
        
        # Get move
        move = current_player.select_move(board)
        
        if move is None or move.is_pass():
            # No legal moves or pass - draw
            metadata = {
                "num_moves": num_moves,
                "duration": time.time() - start_time,
                "outcome": "Draw (no moves)",
            }
            if verbose:
                print(f"Game over: draw (no moves) after {num_moves} moves")
            return "draw", num_moves, metadata
        
        board.apply(move)
        num_moves += 1
        
        if verbose and move_num % 10 == 0:
            print(f"Move {num_moves}: {board.to_move().name} to move")
    
    # Max moves reached
    metadata = {
        "num_moves": num_moves,
        "duration": time.time() - start_time,
        "outcome": "Draw (max moves)",
    }
    
    if verbose:
        print(f"Game over: draw (max moves) after {num_moves} moves")
    
    return "draw", num_moves, metadata


def evaluate_model(
    model: HiveGNN,
    engine_depth: int = 3,
    num_games: int = 20,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """
    Evaluate a model against the engine.
    
    Args:
        model: GNN model to evaluate
        engine_depth: Depth of engine opponent
        num_games: Number of games to play (half as each color)
        device: Device to run model on
        verbose: Print progress
        
    Returns:
        Dictionary with evaluation statistics
    """
    ml_player = MLPlayer(model, temperature=0.1, device=device)
    engine_player = EngineOpponent(depth=engine_depth)
    
    results = {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "total_moves": 0,
        "total_time": 0,
    }
    
    games_per_side = num_games // 2
    
    if verbose:
        print(f"Evaluating model vs engine (depth {engine_depth})...")
        print(f"Playing {games_per_side} games as Black, {games_per_side} games as White")
    
    # Play as Black (player1)
    for i in range(games_per_side):
        if verbose and i % 5 == 0:
            print(f"  Game {i+1}/{games_per_side} as Black...")
        
        winner, num_moves, metadata = play_game(
            ml_player, engine_player, verbose=False
        )
        
        results["total_moves"] += num_moves
        results["total_time"] += metadata["duration"]
        
        if winner == "player1":
            results["wins"] += 1
        elif winner == "player2":
            results["losses"] += 1
        else:
            results["draws"] += 1
    
    # Play as White (player2)
    for i in range(games_per_side):
        if verbose and i % 5 == 0:
            print(f"  Game {i+1}/{games_per_side} as White...")
        
        winner, num_moves, metadata = play_game(
            engine_player, ml_player, verbose=False
        )
        
        results["total_moves"] += num_moves
        results["total_time"] += metadata["duration"]
        
        if winner == "player2":
            results["wins"] += 1
        elif winner == "player1":
            results["losses"] += 1
        else:
            results["draws"] += 1
    
    results["avg_moves"] = results["total_moves"] / num_games
    results["avg_time"] = results["total_time"] / num_games
    results["win_rate"] = results["wins"] / num_games
    
    if verbose:
        print(f"\nResults:")
        print(f"  Wins: {results['wins']}")
        print(f"  Losses: {results['losses']}")
        print(f"  Draws: {results['draws']}")
        print(f"  Win rate: {results['win_rate']:.1%}")
        print(f"  Avg moves: {results['avg_moves']:.1f}")
        print(f"  Avg time: {results['avg_time']:.2f}s")
    
    return results


def evaluate_and_update_elo(
    model: HiveGNN,
    model_name: str,
    elo_tracker: EloTracker,
    engine_depths: List[int] = [3],
    games_per_depth: int = 20,
    device: str = "cpu",
    verbose: bool = True,
) -> Dict:
    """
    Evaluate model against multiple engine depths and update ELO.
    
    Args:
        model: GNN model to evaluate
        model_name: Name/identifier for this model
        elo_tracker: ELO tracker instance
        engine_depths: List of engine depths to test against
        games_per_depth: Games to play per depth
        device: Device to run on
        verbose: Print progress
        
    Returns:
        Dictionary with all evaluation results
    """
    all_results = {}
    
    for depth in engine_depths:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating against engine depth {depth}")
            print(f"{'='*60}")
        
        results = evaluate_model(
            model, depth, games_per_depth, device, verbose
        )
        
        all_results[f"depth_{depth}"] = results
        
        # Update ELO for each game
        # Simplified: update once with aggregate score
        wins = results["wins"]
        losses = results["losses"]
        draws = results["draws"]
        total_games = wins + losses + draws
        
        # Calculate score (from model's perspective)
        score = (wins + 0.5 * draws) / total_games
        
        engine_name = f"engine_depth_{depth}"
        
        # Update ELO
        new_model_rating, new_engine_rating = elo_tracker.update_ratings(
            model_name,
            engine_name,
            score,
            game_metadata={
                "wins": wins,
                "losses": losses,
                "draws": draws,
                "avg_moves": results["avg_moves"],
            },
        )
        
        if verbose:
            print(f"\nELO Update:")
            print(f"  {model_name}: {new_model_rating:.1f}")
            print(f"  {engine_name}: {new_engine_rating:.1f}")
    
    # Save ELO history
    elo_tracker.save()
    
    return all_results


if __name__ == "__main__":
    # Test evaluation
    print("Testing evaluation module...")
    
    # Create a dummy model
    from model import create_model
    
    model = create_model()
    
    # Test against engine
    results = evaluate_model(
        model, engine_depth=2, num_games=4, verbose=True
    )
    
    print("\nEvaluation test passed!")
