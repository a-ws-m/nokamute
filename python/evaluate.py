"""
Evaluation utilities for trained models.
"""

import argparse

import torch
from model import create_model
from self_play import SelfPlayGame

import nokamute


def play_interactive_game(model, device="cpu", human_first=True):
    """
    Play an interactive game against the trained model.

    Args:
        model: Trained GNN model
        device: Device to run model on
        human_first: Whether human plays first
    """
    board = nokamute.Board()
    # Temperature=0 ensures greedy (best) move selection during evaluation
    ai_player = SelfPlayGame(model=model, temperature=0, device=device)

    print("\n" + "=" * 60)
    print("Interactive Game vs AI")
    print("=" * 60 + "\n")

    human_color = "White" if human_first else "Black"
    ai_color = "Black" if human_first else "White"

    print(f"You are playing as {human_color}")
    print(f"AI is playing as {ai_color}\n")

    while True:
        current_player = board.to_move().name
        legal_moves = board.legal_moves()

        # Check for game over
        winner = board.get_winner()
        if winner is not None:
            print(f"\nGame Over! Winner: {winner}")
            break

        # Check for pass
        if len(legal_moves) == 1 and legal_moves[0].is_pass():
            print("No legal moves - game is a draw")
            break

        # Display board state
        pieces = board.get_pieces()
        print(f"\nTurn {board.turn_num()}: {current_player} to move")
        print(f"Pieces on board: {len(pieces)}")

        if current_player == human_color:
            # Human move
            print(f"\nLegal moves ({len(legal_moves)}):")
            for i, move in enumerate(legal_moves):
                print(f"  {i}: {move}")

            while True:
                try:
                    choice = int(input("\nSelect move number: "))
                    if 0 <= choice < len(legal_moves):
                        selected_move = legal_moves[choice]
                        break
                    else:
                        print(f"Invalid choice. Enter 0-{len(legal_moves)-1}")
                except (ValueError, KeyboardInterrupt):
                    print("\nGame aborted")
                    return

            board.apply(selected_move)
            print(f"You played: {selected_move}")
        else:
            # AI move
            print("\nAI is thinking...")
            selected_move = ai_player.select_move(board, legal_moves)
            board.apply(selected_move)
            print(f"AI played: {selected_move}")


def evaluate_vs_random(model, num_games=100, device="cpu"):
    """
    Evaluate model performance against random play.

    Args:
        model: Trained GNN model
        num_games: Number of games to play
        device: Device to run on

    Returns:
        Win rate statistics
    """
    # Temperature=0 ensures greedy (best) move selection during evaluation
    ai_player = SelfPlayGame(model=model, temperature=0, device=device)
    random_player = SelfPlayGame(model=None, temperature=1.0, device=device)

    results = {
        "ai_white_wins": 0,
        "ai_black_wins": 0,
        "random_wins": 0,
        "draws": 0,
    }

    for game_idx in range(num_games):
        board = nokamute.Board()
        ai_is_white = game_idx % 2 == 0

        while True:
            legal_moves = board.legal_moves()
            winner = board.get_winner()

            if winner is not None:
                if winner == "Draw":
                    results["draws"] += 1
                elif (winner == "White" and ai_is_white) or (
                    winner == "Black" and not ai_is_white
                ):
                    if ai_is_white:
                        results["ai_white_wins"] += 1
                    else:
                        results["ai_black_wins"] += 1
                else:
                    results["random_wins"] += 1
                break

            if len(legal_moves) == 1 and legal_moves[0].is_pass():
                results["draws"] += 1
                break

            # Select move based on current player
            current_is_ai = (board.to_move().name == "White" and ai_is_white) or (
                board.to_move().name == "Black" and not ai_is_white
            )

            if current_is_ai:
                move = ai_player.select_move(board, legal_moves)
            else:
                move = random_player.select_move(board, legal_moves)

            board.apply(move)

        if (game_idx + 1) % 10 == 0:
            print(f"Played {game_idx + 1}/{num_games} games...")

    # Calculate statistics
    total_ai_wins = results["ai_white_wins"] + results["ai_black_wins"]
    win_rate = total_ai_wins / num_games

    print(f"\nResults over {num_games} games:")
    print(f"  AI wins (as White): {results['ai_white_wins']}")
    print(f"  AI wins (as Black): {results['ai_black_wins']}")
    print(f"  Random wins: {results['random_wins']}")
    print(f"  Draws: {results['draws']}")
    print(f"  AI win rate: {win_rate:.1%}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained Hive GNN model")
    parser.add_argument(
        "--model", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="vs-random",
        choices=["vs-random", "interactive"],
        help="Evaluation mode",
    )
    parser.add_argument(
        "--games", type=int, default=100, help="Number of games (for vs-random)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    checkpoint = torch.load(args.model, map_location=args.device)

    model_config = checkpoint.get("config", {})
    model = create_model(model_config).to(args.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded (iteration {checkpoint.get('iteration', 'unknown')})")
    print(f"Device: {args.device}")

    if args.mode == "vs-random":
        print(f"\nEvaluating against random play ({args.games} games)...")
        evaluate_vs_random(model, num_games=args.games, device=args.device)

    elif args.mode == "interactive":
        print("\nStarting interactive game...")
        play_interactive_game(model, device=args.device)


if __name__ == "__main__":
    main()
