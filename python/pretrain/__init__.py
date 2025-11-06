"""
Pre-training modules for the Hive GNN model.

This package contains different pre-training strategies that can be used
before self-play training to give the model a better starting point.
"""

from .eval_matching import (
    generate_eval_matching_data,
    pretrain_eval_matching,
    save_eval_data,
    load_eval_data,
    generate_random_positions_branching,
)
from .human_games import (
    generate_human_games_data,
    HumanGamesDataset,
    train_epoch_streaming,
)

__all__ = [
    'generate_eval_matching_data',
    'pretrain_eval_matching',
    'save_eval_data',
    'load_eval_data',
    'generate_random_positions_branching',
    'generate_human_games_data',
    'HumanGamesDataset',
    'train_epoch_streaming',
]
