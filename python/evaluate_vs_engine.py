"""
Evaluation module for testing ML models against the Rust engine.
"""

import time
from typing import Dict, List, Optional, Tuple

import torch
from elo_tracker import EloTracker
from model_policy_hetero import HiveGNNPolicyHetero
from self_play import SelfPlayGame

import nokamute


class EngineOpponent:
    """
    Opponent that uses the Rust minimax engine.
    """

    def __init__(
        self, depth: int = 3, time_limit_ms: Optional[int] = None, aggression: int = 3
    ):
        """
        Args:
            depth: Search depth for minimax
            time_limit_ms: Time limit in milliseconds (overrides depth if set)
            aggression: Aggression level 1-5
        """
        self.depth = depth
        self.time_limit_ms = time_limit_ms
        self.aggression = aggression
        self.name = (
            f"engine_depth_{depth}"
            if time_limit_ms is None
            else f"engine_time_{time_limit_ms}ms"
        )

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


def play_game(
    player1,
    player2,
    max_moves: int = 200,
    verbose: bool = False,
) -> Tuple[Optional[str], int, Dict]:
    """
    Play a game between two players.

    Draw conditions in Hive:
    1. Threefold repetition (same position occurs 3 times) - detected by board.get_winner()
    2. Both Queen bees are surrounded simultaneously - detected by board.get_winner()
    3. Max moves reached (to prevent infinite games)

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
        if isinstance(current_player, SelfPlayGame):
            # SelfPlayGame needs legal_moves explicitly
            legal_moves = board.legal_moves()
            move = current_player.select_move(board, legal_moves)
        else:
            # EngineOpponent handles legal moves internally
            move = current_player.select_move(board)

        if move is None:
            # No legal moves - this shouldn't happen as get_winner() should catch it
            metadata = {
                "num_moves": num_moves,
                "duration": time.time() - start_time,
                "outcome": "Draw (no legal moves)",
            }
            if verbose:
                print(f"Game over: draw (no legal moves) after {num_moves} moves")
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
    model: HiveGNNPolicyHetero,
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
    # Use SelfPlayGame instead of MLPlayer (same functionality, optimized batch eval)
    # Temperature=0 ensures greedy (best) move selection during evaluation
    ml_player = SelfPlayGame(model=model, epsilon=0.0, device=device)
    engine_player = EngineOpponent(depth=engine_depth)

    results = {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "total_moves": 0,
        "total_time": 0,
    }

    games_per_side = -(-num_games // 2)  # Ceiling division

    if verbose:
        print(f"Evaluating model vs engine (depth {engine_depth})...")
        print(
            f"Playing {games_per_side} games as Black, {games_per_side} games as White"
        )

    # Play as Black (player1)
    for i in range(games_per_side):
        if verbose and i % 5 == 0:
            print(f"  Game {i+1}/{games_per_side} as Black...")

        winner, num_moves, metadata = play_game(ml_player, engine_player, verbose=False)

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

        winner, num_moves, metadata = play_game(engine_player, ml_player, verbose=False)

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
    model: HiveGNNPolicyHetero,
    model_name: str,
    elo_tracker: EloTracker,
    engine_depths: List[int] = [3],
    games_per_depth: int = 20,
    device: str = "cpu",
    verbose: bool = True,
    tensorboard_writer=None,
    iteration: Optional[int] = None,
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
        tensorboard_writer: Optional TensorBoard SummaryWriter for logging
        iteration: Current training iteration (for TensorBoard logging)

    Returns:
        Dictionary with all evaluation results
    """
    all_results = {}

    for depth in engine_depths:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating against engine depth {depth}")
            print(f"{'='*60}")

        results = evaluate_model(model, depth, games_per_depth, device, verbose)

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

        # Log to TensorBoard if provided
        if tensorboard_writer is not None and iteration is not None:
            prefix = f"engine_evaluation/depth_{depth}"
            tensorboard_writer.add_scalar(f"{prefix}/wins", wins, iteration)
            tensorboard_writer.add_scalar(f"{prefix}/losses", losses, iteration)
            tensorboard_writer.add_scalar(f"{prefix}/draws", draws, iteration)
            tensorboard_writer.add_scalar(
                f"{prefix}/win_rate", results["win_rate"], iteration
            )
            tensorboard_writer.add_scalar(
                f"{prefix}/avg_moves", results["avg_moves"], iteration
            )

            # Log ELO ratings - both individual and unified view
            tensorboard_writer.add_scalar(
                f"elo_individual/{model_name}", new_model_rating, iteration
            )
            tensorboard_writer.add_scalar(
                f"elo_individual/{engine_name}", new_engine_rating, iteration
            )

            # Log to unified ELO graph (all agents on same plot)
            tensorboard_writer.add_scalar(
                f"elo_unified/{model_name}", new_model_rating, iteration
            )
            tensorboard_writer.add_scalar(
                f"elo_unified/{engine_name}", new_engine_rating, iteration
            )

    # Save ELO history
    elo_tracker.save()

    return all_results


if __name__ == "__main__":
    # Test evaluation
    print("Testing evaluation module...")

    # Create a dummy model
    from model_policy_hetero import create_policy_model

    model = create_policy_model()

    # Test against engine
    results = evaluate_model(model, engine_depth=2, num_games=4, verbose=True)

    print("\nEvaluation test passed!")
