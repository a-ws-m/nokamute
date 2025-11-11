"""
Configuration for league training system.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class LeagueConfig:
    """Configuration for competitive self-play league training."""

    # Main Agent settings
    main_agent_lr: float = 1e-3
    main_agent_batch_size: int = 64
    main_agent_epochs: int = 10
    main_agent_games_per_iter: int = 10
    main_agent_temperature: float = 1.0

    # Exploiter settings
    exploiter_lr: float = 1e-3
    exploiter_batch_size: int = 64
    exploiter_epochs: int = 10
    exploiter_games_per_iter: int = 20
    exploiter_temperature: float = 1.0

    # Minimax reward settings
    minimax_reward_weight: float = 1.0  # Weight of minimax reward vs game outcome
    minimax_gamma: float = 0.99  # Discount factor for minimax reward

    # PFSP (Prioritized Fictitious Self-Play) settings
    pfsp_exponent: float = 2.0  # Exponent for (1 - win_rate)^exponent sampling
    pfsp_epsilon: float = 0.1  # Probability of uniform sampling

    # Convergence criteria
    main_exploiter_convergence_threshold: float = 0.70  # 70% win rate vs Main Agent
    league_exploiter_convergence_threshold: float = 0.60  # 60% win rate vs league
    convergence_window: int = 50  # Number of recent games to consider

    # League management
    max_main_exploiters: int = 3  # Maximum number of Main Exploiters in pool
    max_league_exploiters: int = 3  # Maximum number of League Exploiters in pool
    historical_agents_keep_rate: float = 0.5  # Fraction of historical agents to keep

    # Training schedule
    main_agent_update_interval: int = 1  # Update Main Agent every N iterations
    main_exploiter_spawn_interval: int = (
        5  # Spawn new Main Exploiter every N Main Agent updates
    )
    league_exploiter_spawn_interval: int = (
        10  # Spawn new League Exploiter every N Main Agent updates
    )

    # Evaluation
    eval_games_per_matchup: int = 20  # Games to play per agent matchup for evaluation

    # General
    device: str = "cuda"
    max_moves: int = 400
    gamma: float = 0.99  # TD discount factor
    enable_branching: bool = True

    # Inference settings
    inference_batch_size: Optional[int] = (
        None  # Batch size for position evaluation during game generation
    )
    # None = evaluate all positions in single batch
    # Set to GPU capacity for optimal performance

    def __post_init__(self):
        """Validate configuration."""
        assert 0.0 <= self.main_exploiter_convergence_threshold <= 1.0
        assert 0.0 <= self.league_exploiter_convergence_threshold <= 1.0
        assert 0.0 <= self.pfsp_epsilon <= 1.0
        assert self.pfsp_exponent >= 0.0
        assert self.minimax_reward_weight >= 0.0
