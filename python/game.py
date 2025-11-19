from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from nokamute import Board, Turn


@dataclass
class TrajectoryStep:
    board: Board
    player: str
    action: Turn
    action_str: str
    action_scores: Optional[List[float]] = None
    critic: Optional[float] = None


class Game:
    def __init__(self):
        self.trajectory: List[TrajectoryStep] = []
        self.winner: Optional[str] = None

    def append_step(self, step: TrajectoryStep):
        self.trajectory.append(step)

    def finalize(self, board: Board):
        self.winner = board.get_winner()

    def play(self, agent_white, agent_black, max_steps: int = 1000):
        board = Board()
        for _ in range(max_steps):
            player = board.to_move().name
            agent = agent_white if player == "White" else agent_black
            move = agent.decide_move(board)
            # store step
            step = TrajectoryStep(
                board=board.clone(),
                player=player,
                action=move,
                action_str=str(move),
            )
            self.append_step(step)
            board.apply(move)
            winner = board.get_winner()
            if winner is not None:
                self.finalize(board)
                return self
        # If we hit max steps, consider it a draw
        self.finalize(board)
        return self
