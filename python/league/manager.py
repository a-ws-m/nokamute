"""
League manager for competitive self-play training.

Manages three agent archetypes:
1. Main Agent: Learns robust strategies via PFSP against opponent pool
2. Main Exploiter: Learns counter-strategies against current Main Agent
3. League Exploiter: Learns counter-strategies against entire league history
"""

import os
import pickle
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import LeagueConfig


class AgentArchetype(Enum):
    """Agent role in the league."""

    MAIN = "main"
    MAIN_EXPLOITER = "main_exploiter"
    LEAGUE_EXPLOITER = "league_exploiter"


@dataclass
class LeagueAgent:
    """Represents an agent in the league."""

    name: str  # Unique identifier (e.g., "main_v5", "main_exploiter_3")
    archetype: AgentArchetype
    model_path: str  # Path to model checkpoint
    iteration_created: int  # Training iteration when created

    # Performance tracking
    games_played: int = 0
    total_wins: int = 0
    total_losses: int = 0
    total_draws: int = 0

    # Matchup-specific stats
    matchup_wins: Dict[str, int] = None  # opponent_name -> wins
    matchup_losses: Dict[str, int] = None
    matchup_draws: Dict[str, int] = None

    # For exploiters: convergence tracking
    is_converged: bool = False
    training_epochs: int = 0

    def __post_init__(self):
        if self.matchup_wins is None:
            self.matchup_wins = defaultdict(int)
        if self.matchup_losses is None:
            self.matchup_losses = defaultdict(int)
        if self.matchup_draws is None:
            self.matchup_draws = defaultdict(int)

    @property
    def win_rate(self) -> float:
        """Overall win rate."""
        total = self.games_played
        if total == 0:
            return 0.5
        return self.total_wins / total

    def get_win_rate_vs(self, opponent_name: str) -> float:
        """Win rate against specific opponent."""
        total = (
            self.matchup_wins[opponent_name]
            + self.matchup_losses[opponent_name]
            + self.matchup_draws[opponent_name]
        )
        if total == 0:
            return 0.5
        return self.matchup_wins[opponent_name] / total

    def record_game_result(self, opponent_name: str, result: float):
        """
        Record a game result.

        Args:
            opponent_name: Name of opponent agent
            result: 1.0 for win, -1.0 for loss, 0.0 for draw
        """
        self.games_played += 1

        if result > 0.5:
            self.total_wins += 1
            self.matchup_wins[opponent_name] += 1
        elif result < -0.5:
            self.total_losses += 1
            self.matchup_losses[opponent_name] += 1
        else:
            self.total_draws += 1
            self.matchup_draws[opponent_name] += 1


class LeagueManager:
    """
    Manages the competitive self-play league.

    Implements:
    - Agent lifecycle (creation, storage, retirement)
    - Opponent selection via PFSP (Prioritized Fictitious Self-Play)
    - Performance tracking and convergence detection
    """

    def __init__(self, config: LeagueConfig, save_dir: str):
        """
        Args:
            config: League configuration
            save_dir: Directory for saving league state and models
        """
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Agent pools
        self.main_agents: List[LeagueAgent] = []  # Current + historical Main Agents
        self.main_exploiters: List[LeagueAgent] = []
        self.league_exploiters: List[LeagueAgent] = []

        # Current active agents
        self.current_main_agent: Optional[LeagueAgent] = None
        self.current_main_exploiter: Optional[LeagueAgent] = None
        self.current_league_exploiter: Optional[LeagueAgent] = None

        # Training state
        self.iteration = 0
        self.main_agent_version = 0
        self.main_exploiter_count = 0
        self.league_exploiter_count = 0

        # Try to load existing state
        self._load_state()

    def initialize_main_agent(
        self, model, optimizer, model_config: dict
    ) -> LeagueAgent:
        """
        Initialize the first Main Agent.

        Args:
            model: Initial model (can be pre-trained)
            optimizer: Optimizer for the model
            model_config: Model configuration dict

        Returns:
            Created LeagueAgent
        """
        self.main_agent_version = 0
        agent_name = f"main_v{self.main_agent_version}"
        model_path = self.save_dir / f"{agent_name}.pt"

        # Get state dict and strip torch.compile() prefix if present
        state_dict = model.state_dict()
        if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # Save initial model
        torch.save(
            {
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model_config,
                "model_type": model_config.get(
                    "model_type", "policy"
                ),  # Store at top level for easy access
                "iteration": self.iteration,
            },
            model_path,
        )

        agent = LeagueAgent(
            name=agent_name,
            archetype=AgentArchetype.MAIN,
            model_path=str(model_path),
            iteration_created=self.iteration,
        )

        self.main_agents.append(agent)
        self.current_main_agent = agent

        print(f"Initialized Main Agent: {agent_name}")
        self._save_state()

        return agent

    def update_main_agent(self, model, optimizer, model_config: dict) -> LeagueAgent:
        """
        Update the Main Agent (create new version after training).

        Args:
            model: Updated model
            optimizer: Optimizer state
            model_config: Model configuration

        Returns:
            New Main Agent version
        """
        self.main_agent_version += 1
        agent_name = f"main_v{self.main_agent_version}"
        model_path = self.save_dir / f"{agent_name}.pt"

        # Get state dict and strip torch.compile() prefix if present
        state_dict = model.state_dict()
        if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        # Save updated model
        torch.save(
            {
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model_config,
                "model_type": model_config.get(
                    "model_type", "policy"
                ),  # Store at top level for easy access
                "iteration": self.iteration,
            },
            model_path,
        )

        agent = LeagueAgent(
            name=agent_name,
            archetype=AgentArchetype.MAIN,
            model_path=str(model_path),
            iteration_created=self.iteration,
        )

        self.main_agents.append(agent)
        self.current_main_agent = agent

        # Prune old Main Agents to save space
        self._prune_historical_agents()

        print(f"Updated Main Agent: {agent_name} (version {self.main_agent_version})")
        self._save_state()

        return agent

    def spawn_main_exploiter(self, model, optimizer, model_config: dict) -> LeagueAgent:
        """
        Spawn a new Main Exploiter to exploit current Main Agent.

        Args:
            model: Initial model (typically freshly initialized or cloned from Main Agent)
            optimizer: Optimizer for the model
            model_config: Model configuration

        Returns:
            Created Main Exploiter
        """
        self.main_exploiter_count += 1
        agent_name = f"main_exploiter_{self.main_exploiter_count}"
        model_path = self.save_dir / f"{agent_name}.pt"

        # Get state dict and strip torch.compile() prefix if present
        state_dict = model.state_dict()
        if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        torch.save(
            {
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model_config,
                "model_type": model_config.get(
                    "model_type", "policy"
                ),  # Store at top level for easy access
                "iteration": self.iteration,
                "target_agent": self.current_main_agent.name,  # Track which agent it's exploiting
            },
            model_path,
        )

        agent = LeagueAgent(
            name=agent_name,
            archetype=AgentArchetype.MAIN_EXPLOITER,
            model_path=str(model_path),
            iteration_created=self.iteration,
        )

        self.main_exploiters.append(agent)
        self.current_main_exploiter = agent

        # Prune old exploiters if we exceed max
        if len(self.main_exploiters) > self.config.max_main_exploiters:
            removed = self.main_exploiters.pop(0)
            print(f"Pruning old Main Exploiter: {removed.name}")
            # Optionally delete the model file
            # Path(removed.model_path).unlink(missing_ok=True)

        print(
            f"Spawned Main Exploiter: {agent_name} (targets {self.current_main_agent.name})"
        )
        self._save_state()

        return agent

    def spawn_league_exploiter(
        self, model, optimizer, model_config: dict
    ) -> LeagueAgent:
        """
        Spawn a new League Exploiter to exploit entire league.

        Args:
            model: Initial model
            optimizer: Optimizer for the model
            model_config: Model configuration

        Returns:
            Created League Exploiter
        """
        self.league_exploiter_count += 1
        agent_name = f"league_exploiter_{self.league_exploiter_count}"
        model_path = self.save_dir / f"{agent_name}.pt"

        # Get state dict and strip torch.compile() prefix if present
        state_dict = model.state_dict()
        if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
            state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

        torch.save(
            {
                "model_state_dict": state_dict,
                "optimizer_state_dict": optimizer.state_dict(),
                "config": model_config,
                "model_type": model_config.get(
                    "model_type", "policy"
                ),  # Store at top level for easy access
                "iteration": self.iteration,
            },
            model_path,
        )

        agent = LeagueAgent(
            name=agent_name,
            archetype=AgentArchetype.LEAGUE_EXPLOITER,
            model_path=str(model_path),
            iteration_created=self.iteration,
        )

        self.league_exploiters.append(agent)
        self.current_league_exploiter = agent

        # Prune old exploiters if we exceed max
        if len(self.league_exploiters) > self.config.max_league_exploiters:
            removed = self.league_exploiters.pop(0)
            print(f"Pruning old League Exploiter: {removed.name}")

        print(f"Spawned League Exploiter: {agent_name}")
        self._save_state()

        return agent

    def sample_opponent_for_main_agent(self) -> LeagueAgent:
        """
        Sample an opponent for the Main Agent using PFSP.

        PFSP (Prioritized Fictitious Self-Play) samples opponents proportional
        to (1 - win_rate)^exponent, favoring tougher opponents.

        Returns:
            Sampled opponent agent
        """
        # Build opponent pool: historical main agents + converged exploiters
        opponent_pool = []

        # Add historical main agents (excluding current)
        for agent in self.main_agents:
            if agent != self.current_main_agent:
                opponent_pool.append(agent)

        # Add converged exploiters
        for agent in self.main_exploiters:
            if agent.is_converged:
                opponent_pool.append(agent)

        for agent in self.league_exploiters:
            if agent.is_converged:
                opponent_pool.append(agent)

        if not opponent_pool:
            # No opponents yet, return current main agent (self-play)
            return self.current_main_agent

        # Epsilon-greedy: sometimes sample uniformly
        if np.random.random() < self.config.pfsp_epsilon:
            idx = np.random.randint(len(opponent_pool))
            return opponent_pool[idx]

        # PFSP sampling: (1 - win_rate)^exponent
        win_rates = []
        for opponent in opponent_pool:
            # Get win rate of current main agent vs this opponent
            win_rate = self.current_main_agent.get_win_rate_vs(opponent.name)
            win_rates.append(win_rate)

        # Compute sampling probabilities
        priorities = np.array(
            [(1 - wr) ** self.config.pfsp_exponent for wr in win_rates]
        )

        # Handle case where all priorities are zero (or sum to zero)
        if priorities.sum() == 0 or np.isnan(priorities.sum()):
            # Uniform sampling
            priorities = np.ones(len(opponent_pool)) / len(opponent_pool)
        else:
            priorities = priorities / priorities.sum()

        # Sample
        idx = np.random.choice(len(opponent_pool), p=priorities)
        return opponent_pool[idx]

    def sample_opponent_for_league_exploiter(self) -> LeagueAgent:
        """
        Sample an opponent for League Exploiter using PFSP over entire league.

        Returns:
            Sampled opponent agent
        """
        # Opponent pool: all main agents + all converged exploiters
        opponent_pool = self.main_agents.copy()

        for agent in self.main_exploiters:
            if agent.is_converged:
                opponent_pool.append(agent)

        for agent in self.league_exploiters:
            if agent != self.current_league_exploiter and agent.is_converged:
                opponent_pool.append(agent)

        if not opponent_pool:
            return self.current_main_agent

        # Epsilon-greedy
        if np.random.random() < self.config.pfsp_epsilon:
            idx = np.random.randint(len(opponent_pool))
            return opponent_pool[idx]

        # PFSP sampling
        win_rates = []
        for opponent in opponent_pool:
            if self.current_league_exploiter is None:
                win_rate = 0.5
            else:
                win_rate = self.current_league_exploiter.get_win_rate_vs(opponent.name)
            win_rates.append(win_rate)

        priorities = np.array(
            [(1 - wr) ** self.config.pfsp_exponent for wr in win_rates]
        )

        # Handle case where all priorities are zero
        if priorities.sum() == 0 or np.isnan(priorities.sum()):
            priorities = np.ones(len(opponent_pool)) / len(opponent_pool)
        else:
            priorities = priorities / priorities.sum()

        idx = np.random.choice(len(opponent_pool), p=priorities)
        return opponent_pool[idx]

    def check_exploiter_convergence(
        self, exploiter: LeagueAgent, recent_games: List[Tuple[str, float]]
    ) -> bool:
        """
        Check if an exploiter has converged.

        Args:
            exploiter: The exploiter agent to check
            recent_games: List of (opponent_name, result) from recent games

        Returns:
            True if converged
        """
        if len(recent_games) < self.config.convergence_window:
            return False

        # Check recent performance
        recent_window = recent_games[-self.config.convergence_window :]

        if exploiter.archetype == AgentArchetype.MAIN_EXPLOITER:
            # Main Exploiter: check win rate vs Main Agent
            main_agent_games = [
                r for o, r in recent_window if o == self.current_main_agent.name
            ]
            if len(main_agent_games) < 20:  # Need enough games
                return False

            wins = sum(1 for r in main_agent_games if r > 0.5)
            win_rate = wins / len(main_agent_games)

            return win_rate >= self.config.main_exploiter_convergence_threshold

        elif exploiter.archetype == AgentArchetype.LEAGUE_EXPLOITER:
            # League Exploiter: check win rate vs league
            wins = sum(1 for _, r in recent_window if r > 0.5)
            win_rate = wins / len(recent_window)

            return win_rate >= self.config.league_exploiter_convergence_threshold

        return False

    def _prune_historical_agents(self):
        """Prune old historical Main Agents to save disk space."""
        if len(self.main_agents) <= 3:  # Always keep at least a few
            return

        # Keep current agent + some historical agents
        num_to_keep = max(
            3, int(len(self.main_agents) * self.config.historical_agents_keep_rate)
        )

        if len(self.main_agents) > num_to_keep:
            # Keep current agent and evenly spaced historical agents
            agents_to_keep = [self.current_main_agent]
            historical = [a for a in self.main_agents if a != self.current_main_agent]

            # Sample evenly
            step = max(1, len(historical) // (num_to_keep - 1))
            for i in range(0, len(historical), step):
                if len(agents_to_keep) < num_to_keep:
                    agents_to_keep.append(historical[i])

            # Remove agents not in keep list
            for agent in self.main_agents:
                if agent not in agents_to_keep:
                    print(f"Pruning historical Main Agent: {agent.name}")
                    # Optionally delete model file
                    # Path(agent.model_path).unlink(missing_ok=True)

            self.main_agents = agents_to_keep

    def get_all_agents(self) -> List[LeagueAgent]:
        """Get all agents in the league."""
        return self.main_agents + self.main_exploiters + self.league_exploiters

    def _save_state(self):
        """Save league state to disk."""
        state = {
            "iteration": self.iteration,
            "main_agent_version": self.main_agent_version,
            "main_exploiter_count": self.main_exploiter_count,
            "league_exploiter_count": self.league_exploiter_count,
            "main_agents": self.main_agents,
            "main_exploiters": self.main_exploiters,
            "league_exploiters": self.league_exploiters,
            "current_main_agent": self.current_main_agent,
            "current_main_exploiter": self.current_main_exploiter,
            "current_league_exploiter": self.current_league_exploiter,
        }

        state_path = self.save_dir / "league_state.pkl"
        with open(state_path, "wb") as f:
            pickle.dump(state, f)

    def _load_state(self):
        """Load league state from disk if it exists."""
        state_path = self.save_dir / "league_state.pkl"

        if not state_path.exists():
            return

        try:
            with open(state_path, "rb") as f:
                state = pickle.load(f)

            self.iteration = state["iteration"]
            self.main_agent_version = state["main_agent_version"]
            self.main_exploiter_count = state["main_exploiter_count"]
            self.league_exploiter_count = state["league_exploiter_count"]
            self.main_agents = state["main_agents"]
            self.main_exploiters = state["main_exploiters"]
            self.league_exploiters = state["league_exploiters"]
            self.current_main_agent = state["current_main_agent"]
            self.current_main_exploiter = state["current_main_exploiter"]
            self.current_league_exploiter = state["current_league_exploiter"]

            print(f"Loaded league state from iteration {self.iteration}")
            print(
                f"  Main Agent: {self.current_main_agent.name if self.current_main_agent else 'None'}"
            )
            print(f"  Main Exploiters: {len(self.main_exploiters)}")
            print(f"  League Exploiters: {len(self.league_exploiters)}")

        except Exception as e:
            print(f"Warning: Failed to load league state: {e}")
