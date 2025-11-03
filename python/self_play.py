"""
Self-play game generation for training the GNN model.
"""

import random
from typing import List, Optional, Tuple

import numpy as np
import torch

import nokamute


class SelfPlayGame:
    """
    Generates self-play games using the current model for evaluation.
    """

    def __init__(self, model=None, temperature=1.0, device="cpu"):
        """
        Args:
            model: GNN model for position evaluation (optional)
            temperature: Temperature for move selection (higher = more exploration)
            device: Device to run model on
        """
        self.model = model
        self.temperature = temperature
        self.device = device

        if self.model is not None:
            self.model.eval()

    def select_move(self, board, legal_moves):
        """
        Select a move based on model evaluation or random selection.

        Args:
            board: Current board state
            legal_moves: List of legal moves

        Returns:
            Selected move
        """
        if len(legal_moves) == 1:
            return legal_moves[0]

        if self.model is None:
            # Random selection
            return random.choice(legal_moves)

        # Evaluate each move
        move_values = []
        for move in legal_moves:
            # Create a copy and apply the move
            board_copy = board.clone()
            board_copy.apply(move)

            # Convert to graph and evaluate
            node_features, edge_index = board_copy.to_graph()

            # Convert to tensors
            if len(node_features) == 0:
                # Empty board
                value = 0.0
            else:
                x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
                edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).to(
                    self.device
                )

                with torch.no_grad():
                    value = self.model.predict_value(x, edge_index_tensor).item()

            # Negate value (we want opponent's perspective after our move)
            move_values.append(-value)

            # Undo the move
            board_copy.undo(move)

        # Apply temperature and select
        if self.temperature == 0:
            # Greedy selection
            best_idx = np.argmax(move_values)
            return legal_moves[best_idx]
        else:
            # Sample proportional to softmax of values
            values_array = np.array(move_values)
            exp_values = np.exp(values_array / self.temperature)
            probs = exp_values / np.sum(exp_values)

            selected_idx = np.random.choice(len(legal_moves), p=probs)
            return legal_moves[selected_idx]

    def play_game(self, max_moves=200):
        """
        Play a single self-play game.

        Args:
            max_moves: Maximum number of moves before declaring draw

        Returns:
            game_data: List of (board_state, legal_moves, selected_move)
            result: Game result (1.0 for player 1 win, -1.0 for player 2 win, 0.0 for draw)
        """
        board = nokamute.Board()
        game_data = []

        for _ in range(max_moves):
            legal_moves = board.legal_moves()

            # Check for game over
            winner = board.get_winner()
            if winner is not None:
                # Determine result from perspective of each position
                if winner == "Draw":
                    result = 0.0
                elif winner == "White":
                    result = 1.0  # Positive for white win
                else:  # Black
                    result = -1.0  # Negative for black win

                return game_data, result

            # Check for pass (no legal moves except pass)
            if len(legal_moves) == 1 and legal_moves[0].is_pass():
                return game_data, 0.0  # Draw

            # Store board state before move
            current_player = board.to_move()
            board_graph = board.to_graph()

            # Select and apply move
            selected_move = self.select_move(board, legal_moves)
            game_data.append((board_graph, legal_moves, selected_move, current_player))

            board.apply(selected_move)

        # Max moves reached - draw
        return game_data, 0.0

    def generate_games(self, num_games=100):
        """
        Generate multiple self-play games.

        Args:
            num_games: Number of games to generate

        Returns:
            List of (game_data, result) tuples
        """
        games = []

        for i in range(num_games):
            game_data, result = self.play_game()
            games.append((game_data, result))

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_games} games...")

        return games


def prepare_training_data(games):
    """
    Convert self-play games to training data.

    Args:
        games: List of (game_data, result) tuples from self-play

    Returns:
        training_examples: List of (node_features, edge_index, target_value)
    """
    training_examples = []

    for game_data, final_result in games:
        # Assign values to each position based on final result
        for position_idx, (
            board_graph,
            legal_moves,
            selected_move,
            player,
        ) in enumerate(game_data):
            node_features, edge_index = board_graph

            # Skip empty boards
            if len(node_features) == 0:
                continue

            # Flip result based on player perspective
            # White = positive, Black = negative
            if player.name() == "White":
                target_value = final_result
            else:
                target_value = -final_result

            training_examples.append((node_features, edge_index, target_value))

    return training_examples


if __name__ == "__main__":
    # Test self-play
    print("Testing self-play game generation...")

    player = SelfPlayGame()
    game_data, result = player.play_game()

    print(f"Game finished with result: {result}")
    print(f"Number of positions: {len(game_data)}")

    if len(game_data) > 0:
        board_graph, legal_moves, selected_move, player_color = game_data[0]
        node_features, edge_index = board_graph
        print(f"First position: {len(node_features)} nodes, {len(edge_index[0])} edges")

    print("Self-play test passed!")
