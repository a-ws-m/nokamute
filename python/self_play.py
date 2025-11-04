"""
Self-play game generation for training the GNN model.

Implements branching MCMC for efficient parallel game generation:
- Maintains a tree of game states with decision probabilities
- Branches from intermediate positions to create diverse trajectories
- Reuses early-game computation across multiple game instances
"""

import random
from typing import List, Optional, Tuple, Dict
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

import nokamute


@dataclass
class GameNode:
    """
    A node in the game tree representing a board state.
    Used for branching MCMC to track alternative paths.
    """
    board_state: str  # Game state representation for deduplication
    move_probs: Dict[str, float]  # Move -> probability mapping
    parent: Optional['GameNode'] = None
    children: Optional[Dict[str, 'GameNode']] = None  # Move -> child node
    visit_count: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = {}


class SelfPlayGame:
    """
    Generates self-play games using the current model for evaluation.
    Supports branching MCMC for efficient parallel game generation.
    """

    def __init__(self, model=None, temperature=1.0, device="cpu", enable_branching=False):
        """
        Args:
            model: GNN model for position evaluation (optional)
            temperature: Temperature for move selection (higher = more exploration)
            device: Device to run model on
            enable_branching: Enable branching MCMC for game generation
        """
        self.model = model
        self.temperature = temperature
        self.device = device
        self.enable_branching = enable_branching
        
        # Branching MCMC state
        self.game_tree: Dict[str, GameNode] = {}  # board_state -> GameNode
        self.branch_points: List[Tuple[nokamute.Board, GameNode, int]] = []  # (board, node, depth)

        if self.model is not None:
            self.model.eval()

    def _get_board_state_key(self, board):
        """
        Get a hashable representation of the board state.
        Uses the zobrist hash for deduplication.
        """
        return str(board.zobrist_hash())

    def _compute_move_probabilities(self, board, legal_moves):
        """
        Compute probability distribution over legal moves.
        
        Args:
            board: Current board state
            legal_moves: List of legal moves
            
        Returns:
            Dictionary mapping move string to probability
        """
        if len(legal_moves) == 1:
            return {str(legal_moves[0]): 1.0}
        
        # Check for immediate winning moves first
        current_player = board.to_move().name
        for move in legal_moves:
            board_copy = board.clone()
            board_copy.apply(move)
            winner = board_copy.get_winner()
            
            if winner == current_player:
                # Winning move gets probability 1
                return {str(move): 1.0}
            
            del board_copy
        
        if self.model is None:
            # Uniform distribution for random play
            prob = 1.0 / len(legal_moves)
            return {str(move): prob for move in legal_moves}
        
        # Batch evaluate all moves
        move_values = self._batch_evaluate_moves(board, legal_moves)
        
        # Compute softmax probabilities
        if self.temperature == 0:
            # Greedy: assign probability 1 to best move
            best_idx = np.argmax(move_values)
            probs = np.zeros(len(legal_moves))
            probs[best_idx] = 1.0
        else:
            # Softmax with temperature
            values_array = np.array(move_values)
            exp_values = np.exp(values_array / self.temperature)
            probs = exp_values / np.sum(exp_values)
        
        return {str(legal_moves[i]): float(probs[i]) for i in range(len(legal_moves))}

    def select_move(self, board, legal_moves, return_probs=False):
        """
        Select a move based on model evaluation or random selection.
        Checks for immediate winning moves first.
        Uses batch evaluation for all legal moves when model is available.

        Args:
            board: Current board state
            legal_moves: List of legal moves
            return_probs: If True, also return move probabilities

        Returns:
            Selected move (and optionally move probabilities dict)
        """
        if len(legal_moves) == 1:
            move = legal_moves[0]
            if return_probs:
                return move, {str(move): 1.0}
            return move

        # Compute probabilities for all moves
        move_probs = self._compute_move_probabilities(board, legal_moves)
        
        # Sample move according to probabilities
        move_strs = list(move_probs.keys())
        probs = list(move_probs.values())
        selected_move_str = np.random.choice(move_strs, p=probs)
        
        # Find the actual move object
        selected_move = None
        for move in legal_moves:
            if str(move) == selected_move_str:
                selected_move = move
                break
        
        if return_probs:
            return selected_move, move_probs
        return selected_move

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

    def play_game(self, max_moves=400, start_board=None, start_depth=0):
        """
        Play a single self-play game.
        
        Draw conditions in Hive:
        1. Threefold repetition (same position occurs 3 times)
        2. Both Queen bees are surrounded simultaneously
        3. Max moves reached (to prevent infinite games)

        Args:
            max_moves: Maximum number of moves before declaring draw
            start_board: Optional starting board position (for branching)
            start_depth: Starting depth (for branching)

        Returns:
            game_data: List of (board_state, legal_moves, selected_move)
            result: Game result (1.0 for player 1 win, -1.0 for player 2 win, 0.0 for draw)
        """
        board = start_board.clone() if start_board is not None else nokamute.Board()
        game_data = []

        for move_num in range(max_moves):
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

            # Select move and get probabilities
            if self.enable_branching:
                selected_move, move_probs = self.select_move(board, legal_moves, return_probs=True)
                
                # Track this position in the game tree
                current_depth = start_depth + move_num
                board_key = self._get_board_state_key(board)
                
                if board_key not in self.game_tree:
                    node = GameNode(board_state=board_key, move_probs=move_probs)
                    self.game_tree[board_key] = node
                else:
                    node = self.game_tree[board_key]
                    node.move_probs = move_probs  # Update with latest probabilities
                
                node.visit_count += 1
                
                # Record potential branch point if there are multiple viable moves
                # (probability > threshold indicates a meaningful alternative)
                branch_threshold = 0.15
                num_viable_moves = sum(1 for p in move_probs.values() if p > branch_threshold)
                if num_viable_moves > 1 and current_depth < max_moves // 2:
                    # Only keep branch points from early/mid game
                    self.branch_points.append((board.clone(), node, current_depth))
                
                game_data.append((board_graph, legal_moves, selected_move, current_player))
            else:
                # Standard play without branching
                selected_move = self.select_move(board, legal_moves)
                game_data.append((board_graph, legal_moves, selected_move, current_player))

            board.apply(selected_move)

        # Max moves reached - draw
        return game_data, 0.0

    def generate_games(self, num_games=100):
        """
        Generate multiple self-play games.
        Uses branching MCMC if enabled, otherwise standard sequential generation.

        Args:
            num_games: Number of games to generate

        Returns:
            List of (game_data, result) tuples
        """
        if self.enable_branching:
            return self.generate_games_with_branching(num_games)
        else:
            return self._generate_games_sequential(num_games)

    def _generate_games_sequential(self, num_games):
        """
        Generate games sequentially (standard approach).
        """
        games = []

        for i in range(num_games):
            game_data, result = self.play_game()
            games.append((game_data, result))

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_games} games...")

        return games

    def generate_games_with_branching(self, num_games=100, branch_ratio=0.5):
        """
        Generate games using branching MCMC.
        
        This exploits the fact that self-play is essentially MCMC sampling:
        - Play initial games to build up a tree of positions
        - Branch from promising intermediate positions to explore alternatives
        - Reuse early-game computation across multiple game trajectories
        
        Args:
            num_games: Number of games to generate
            branch_ratio: Fraction of games to start from branch points vs. start
            
        Returns:
            List of (game_data, result) tuples
        """
        games = []
        
        # Calculate how many games to generate from scratch vs. branches
        num_from_start = int(num_games * (1 - branch_ratio))
        num_from_branches = num_games - num_from_start
        
        # Phase 1: Generate initial games from scratch to build branch points
        print(f"Generating {num_from_start} games from start position...")
        for i in range(num_from_start):
            game_data, result = self.play_game()
            games.append((game_data, result))
            
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{num_from_start} initial games, "
                      f"{len(self.branch_points)} branch points collected...")
        
        # Phase 2: Branch from collected positions
        if num_from_branches > 0 and len(self.branch_points) > 0:
            print(f"\nGenerating {num_from_branches} games from {len(self.branch_points)} branch points...")
            
            for i in range(num_from_branches):
                # Select a branch point (weighted by move probabilities)
                board, node, depth = self._select_branch_point()
                
                # Play from this position
                game_data, result = self.play_game(
                    start_board=board,
                    start_depth=depth
                )
                games.append((game_data, result))
                
                if (i + 1) % 10 == 0:
                    print(f"  {i + 1}/{num_from_branches} branched games...")
        
        return games

    def _select_branch_point(self):
        """
        Select a branch point to start a new game from.
        Prefers less-visited positions and those with higher entropy.
        
        Returns:
            (board, node, depth) tuple
        """
        if not self.branch_points:
            # Fallback to empty board
            return nokamute.Board(), GameNode(board_state="", move_probs={}), 0
        
        # Calculate selection weights based on:
        # 1. Move probability entropy (higher = more uncertain/interesting)
        # 2. Visit count (lower = less explored)
        weights = []
        for board, node, depth in self.branch_points:
            # Entropy of move probabilities
            probs = list(node.move_probs.values())
            if len(probs) > 1:
                entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            else:
                entropy = 0.0
            
            # Inverse visit count (explore less-visited positions)
            visit_weight = 1.0 / (node.visit_count + 1)
            
            # Combined weight
            weight = entropy * visit_weight
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            probs = [w / total_weight for w in weights]
        else:
            # Uniform if all weights are zero
            probs = [1.0 / len(weights)] * len(weights)
        
        # Sample a branch point
        idx = np.random.choice(len(self.branch_points), p=probs)
        return self.branch_points[idx]
    
    def clear_branch_cache(self):
        """
        Clear the branching cache. Call this between training iterations.
        """
        self.game_tree.clear()
        self.branch_points.clear()


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
