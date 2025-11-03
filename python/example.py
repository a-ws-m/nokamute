"""
Quick example demonstrating the nokamute Python bindings.
"""

import nokamute


def main():
    print("Nokamute Python Bindings - Quick Example")
    print("=" * 60)

    # Create a new board
    print("\n1. Creating a new board...")
    board = nokamute.Board()
    print(f"   {board}")

    # Get legal moves
    print("\n2. Getting legal moves...")
    moves = board.legal_moves()
    print(f"   Found {len(moves)} legal moves")
    print(f"   First move: {moves[0]}")

    # Apply a few moves
    print("\n3. Playing first few moves...")
    for i in range(min(6, len(moves))):
        move = moves[0]
        print(f"   Turn {board.turn_num()}: {board.to_move()} plays {move}")
        board.apply(move)
        moves = board.legal_moves()

    # Get board state
    print("\n4. Current board state...")
    pieces = board.get_pieces()
    print(f"   Pieces on board: {len(pieces)}")
    for hex_pos, color, bug, height in pieces[:5]:
        print(f"     {color} {bug} at position {hex_pos} (height {height})")
    if len(pieces) > 5:
        print(f"     ... and {len(pieces) - 5} more")

    # Convert to graph
    print("\n5. Converting board to graph representation...")
    node_features, edge_index = board.to_graph()
    print(f"   Nodes: {len(node_features)}")
    print(f"   Edges: {len(edge_index[0])}")
    if len(node_features) > 0:
        print(f"   First node features: {node_features[0]}")

    # Play a complete game
    print("\n6. Playing a complete random game...")
    board = nokamute.Board()
    move_count = 0
    max_moves = 100

    while move_count < max_moves:
        moves = board.legal_moves()
        winner = board.get_winner()

        if winner is not None:
            print(f"   Game over after {move_count} moves!")
            print(f"   Winner: {winner}")
            break

        # Random move selection
        import random

        move = random.choice(moves)
        board.apply(move)
        move_count += 1

    if move_count >= max_moves:
        print(f"   Game reached max moves ({max_moves})")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("\nNext steps:")
    print("  - Run 'python train.py --help' to see training options")
    print("  - Run 'python evaluate.py --help' to see evaluation options")
    print("  - See python/README.md for detailed documentation")


if __name__ == "__main__":
    main()
