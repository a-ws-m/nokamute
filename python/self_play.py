"""
Self-play game generation for training the GNN model.
"""

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

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
        Checks for immediate winning moves first.
        Uses batch evaluation for all legal moves when model is available.

        Args:
            board: Current board state
            legal_moves: List of legal moves

        Returns:
            Selected move
        """
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Check for immediate winning moves
        current_player = board.to_move().name
        for move in legal_moves:
            board_copy = board.clone()
            board_copy.apply(move)
            winner = board_copy.get_winner()
            
            if winner == current_player:
                # This move wins immediately - play it!
                return move
            
            # Clean up the board copy (though it will be garbage collected anyway)
            del board_copy

        if self.model is None:
            # Random selection
            return random.choice(legal_moves)

        # Batch evaluate all moves
        move_values = self._batch_evaluate_moves(board, legal_moves)

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

    def _batch_evaluate_moves(self, board, legal_moves):
        """
        Evaluate all legal moves in a single batch using DataLoader.

        Args:
            board: Current board state
            legal_moves: List of legal moves

        Returns:
            List of values for each move (from current player's perspective)
        """
        # Prepare all board states after each move
        data_list = []
        
        for move in legal_moves:
            # Create a copy and apply the move
            board_copy = board.clone()
            board_copy.apply(move)

            # Convert to graph
            node_features, edge_index = board_copy.to_graph()

            # Skip if empty board
            if len(node_features) == 0:
                data_list.append(None)
            else:
                # Convert to PyG Data object
                x = torch.tensor(node_features, dtype=torch.float32)
                edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
                
                data = Data(x=x, edge_index=edge_index_tensor)
                data_list.append(data)

            # Undo the move
            board_copy.undo(move)

        # Batch evaluate all valid positions
        move_values = []
        valid_data = [d for d in data_list if d is not None]
        
        if len(valid_data) == 0:
            # All moves lead to empty boards (shouldn't happen in practice)
            return [0.0] * len(legal_moves)

        # Create batch from all data
        batch = Batch.from_data_list(valid_data).to(self.device)
        
        # Evaluate all positions in a single forward pass
        with torch.no_grad():
            predictions, _ = self.model(batch.x, batch.edge_index, batch.batch)
            values = predictions.squeeze().cpu().numpy()
        
        # Handle single value case (only one valid move)
        if len(valid_data) == 1:
            values = [values.item()]
        else:
            values = values.tolist()

        # Map values back to all moves (including None positions)
        value_idx = 0
        for data in data_list:
            if data is None:
                move_values.append(0.0)
            else:
                # Negate value (we want opponent's perspective after our move)
                move_values.append(-values[value_idx])
                value_idx += 1

        return move_values

    def play_game(self, max_moves=200):
        """
        Play a single self-play game.
        
        Draw conditions in Hive:
        1. Threefold repetition (same position occurs 3 times)
        2. Both Queen bees are surrounded simultaneously
        3. Max moves reached (to prevent infinite games)

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
            if player.name == "White":
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
