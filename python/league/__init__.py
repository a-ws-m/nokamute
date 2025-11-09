"""
League training system for competitive self-play in Hive.

Implements the league structure from:
"Minimax Exploiter: A Data Efficient Approach for Competitive Self-Play"

This package provides:
- League manager for handling multiple agent archetypes
- Exploiter agents with minimax reward shaping
- Performance tracking and opponent selection (PFSP)
"""

from .config import LeagueConfig
from .exploiter import (
    ExploiterAgent,
    compute_minimax_reward,
    prepare_exploiter_training_data,
)
from .manager import AgentArchetype, LeagueManager
from .tracker import LeagueTracker

__all__ = [
    "LeagueManager",
    "AgentArchetype",
    "ExploiterAgent",
    "compute_minimax_reward",
    "prepare_exploiter_training_data",
    "LeagueTracker",
    "LeagueConfig",
]
