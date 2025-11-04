"""
Pre-training modules for the Hive GNN model.

This package contains different pre-training strategies that can be used
before self-play training to give the model a better starting point.
"""

from .eval_matching import generate_eval_matching_data, pretrain_eval_matching

__all__ = [
    'generate_eval_matching_data',
    'pretrain_eval_matching',
]
