from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import List, Optional

from nokamute import Board, Turn
from python.graph import BoardHeteroBuilder
from python.hetero_models import MoveScorer


class Agent(ABC):
    """Abstract Agent for playing Hive via the Nokamute python bindings."""

    def choose_winning_move(self, board: Board) -> Optional[Turn]:
        """Return a move that immediately wins for the current player, or None."""
        player_name = board.to_move().name
        for m in board.legal_moves():
            b2 = board.clone()
            b2.apply(m)
            winner = b2.get_winner()
            if winner is not None and winner == player_name:
                return m
        return None

    @abstractmethod
    def decide_move(self, board: Board) -> Turn:
        raise NotImplementedError


class RandomAgent(Agent):
    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)

    def decide_move(self, board: Board) -> Turn:
        # always take a winning move if present
        wm = self.choose_winning_move(board)
        if wm is not None:
            return wm

        moves = board.legal_moves()
        return self.rng.choice(moves)


class NokamuteEngine(Agent):
    """Wrapper around Nokamute's Rust minimax engine exposed via the Board
    bindings. This calls `Board.choose_move(depth)` to get an engine move.
    """

    def __init__(self, depth: int = 1):
        self.depth = depth

    def decide_move(self, board: Board) -> Turn:
        wm = self.choose_winning_move(board)
        if wm is not None:
            return wm

        # Use Rust engine via the `Board.choose_move` binding.
        mv = board.choose_move(self.depth)
        if mv is not None:
            return mv

        # If the engine did not return a move (possible in draw-like positions),
        # fall back to a random legal move as a last resort.
        moves = board.legal_moves()
        if not moves:
            raise RuntimeError("No legal moves available")
        return random.choice(moves)


class GNNAgent(Agent):
    def __init__(self, model: MoveScorer | None = None, device: Optional[str] = None):
        self.model = model if model is not None else MoveScorer(1, hidden_dim=16)
        self.device = device

    def decide_move(self, board: Board) -> Turn:
        wm = self.choose_winning_move(board)
        if wm is not None:
            return wm

        builder = BoardHeteroBuilder(board)
        data = builder.to_model_input()

        action_scores, _ = self.model(data)
        # normalize to single graph representation
        if isinstance(action_scores, list):
            action_scores = action_scores[0]

        # Turn ordered move strings into mapping
        maps = self.model.action_scores_to_move_dicts(data, action_scores)
        if not maps or not maps[0]:
            # fallback to random
            return RandomAgent().decide_move(board)

        top_move_str = next(iter(maps[0].keys()))

        # Find python Turn object with matching str()
        for m in board.legal_moves():
            if str(m) == top_move_str:
                return m

        # fallback
        return RandomAgent().decide_move(board)
