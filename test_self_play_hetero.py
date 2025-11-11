"""
Test self-play with the new heterogeneous GNN model.
"""

import sys

sys.path.insert(0, "python")

import torch
from model_policy_hetero import create_policy_model
from self_play import SelfPlayGame

import nokamute


def test_self_play_with_hetero_model():
    """Test that self-play works with the new heterogeneous model."""
    print("=" * 60)
    print("Testing self-play with heterogeneous GNN model")
    print("=" * 60)

    # Create model
    model = create_policy_model(
        config={"hidden_dim": 64, "num_layers": 2, "num_heads": 2, "dropout": 0.1}
    )
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    print("\n1. Model created")

    # Create self-play game
    game = SelfPlayGame(
        model=model,
        device=device,
        temperature=1.0,
    )

    print("2. Self-play game created")

    # Test move selection on empty board
    print("\n3. Testing move selection on empty board...")
    board = nokamute.Board()
    legal_moves = board.legal_moves()
    print(f"   Legal moves: {len(legal_moves)}")

    selected_move, move_probs, value = game.select_move_policy(
        board, legal_moves, return_probs=True, return_value=True
    )

    print(f"   Selected move: {selected_move}")
    print(f"   Position value: {value:.3f}")
    print(
        f"   Move probabilities ({len(move_probs)}): {list(move_probs.values())[:5]}..."
    )

    # Test move selection after a few moves
    print("\n4. Testing move selection after 3 moves...")
    board = nokamute.Board()
    for _ in range(3):
        moves = board.legal_moves()
        if moves:
            board.apply(moves[0])

    legal_moves = board.legal_moves()
    print(f"   Legal moves: {len(legal_moves)}")

    selected_move, move_probs, value = game.select_move_policy(
        board, legal_moves, return_probs=True, return_value=True
    )

    print(f"   Selected move: {selected_move}")
    print(f"   Position value: {value:.3f}")
    print(f"   Probability sum: {sum(move_probs.values()):.4f}")

    # Test full self-play game
    print("\n5. Testing full self-play game...")
    board = nokamute.Board()
    move_count = 0
    max_moves = 20

    while move_count < max_moves:
        winner = board.get_winner()
        if winner is not None:
            break

        legal_moves = board.legal_moves()
        if not legal_moves:
            break

        selected_move = game.select_move_policy(board, legal_moves)
        board.apply(selected_move)
        move_count += 1

        if move_count % 5 == 0:
            print(f"   Completed {move_count} moves")

    print(f"   Game completed after {move_count} moves")

    winner = board.get_winner()
    if winner is not None:
        print(f"   Game over: True")
        print(f"   Winner: {winner}")
    else:
        print(f"   Game over: False")
        print(f"   Stopped at move limit")

    print("\n" + "=" * 60)
    print("✓ Self-play test passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_self_play_with_hetero_model()
