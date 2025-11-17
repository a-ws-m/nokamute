"""
Performance tracking and evaluation for league training.

Provides:
- Win rate tracking between agents
- ELO rating tracking for all agents
- Performance visualization via TensorBoard
- Head-to-head matchup evaluation
- League strength metrics
"""

import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from elo_tracker import EloTracker
from league.manager import AgentArchetype, LeagueAgent
from model_policy_hetero import HiveGNNPolicyHetero
from self_play import SelfPlayGame
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import nokamute


class LeagueTracker:
    """
    Tracks performance metrics for the competitive self-play league.

    Integrates with TensorBoard for visualization of:
    - Agent win rates over time
    - Head-to-head matchup results
    - Exploiter convergence progress
    - League diversity metrics
    """

    def __init__(
        self,
        log_dir: str,
        elo_save_path: Optional[str] = None,
        engine_depths: Optional[List[int]] = None,
    ):
        """
        Args:
            log_dir: Directory for TensorBoard logs
            elo_save_path: Path to save ELO ratings (optional, defaults to log_dir/elo_ratings.json)
            engine_depths: List of engine depths to track (optional, defaults to all depths 1-5)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        # ELO tracking
        if elo_save_path is None:
            elo_save_path = str(self.log_dir / "elo_ratings.json")
        self.elo_tracker = EloTracker(
            save_path=elo_save_path, k_factor=32, engine_depths=engine_depths
        )

        # Performance history
        self.matchup_history = defaultdict(
            list
        )  # (agent1, agent2) -> [(iteration, result), ...]
        self.agent_elo_history = defaultdict(
            list
        )  # agent_name -> [(iteration, elo), ...]

        # Convergence tracking for exploiters
        self.exploiter_training_history = defaultdict(
            list
        )  # exploiter_name -> [(epoch, win_rate), ...]

    def log_training_metrics(
        self,
        iteration: int,
        agent_name: str,
        loss: float,
        learning_rate: float,
        **kwargs,
    ):
        """
        Log training metrics to TensorBoard.

        Args:
            iteration: Training iteration
            agent_name: Name of agent being trained
            loss: Training loss
            learning_rate: Current learning rate
            **kwargs: Additional metrics to log
        """
        prefix = f"{agent_name}/training"

        self.writer.add_scalar(f"{prefix}/loss", loss, iteration)
        self.writer.add_scalar(f"{prefix}/lr", learning_rate, iteration)

        for key, value in kwargs.items():
            self.writer.add_scalar(f"{prefix}/{key}", value, iteration)

    def log_game_result(
        self,
        iteration: int,
        agent1: LeagueAgent,
        agent2: LeagueAgent,
        result: float,
        game_length: int,
    ):
        """
        Log a game result and update ELO ratings.

        Args:
            iteration: Training iteration
            agent1: First agent (from their perspective: 1.0=win, -1.0=loss, 0.0=draw)
            agent2: Second agent
            result: Game result from agent1's perspective
            game_length: Number of moves in the game
        """
        # Store in matchup history
        matchup_key = (agent1.name, agent2.name)
        self.matchup_history[matchup_key].append((iteration, result))

        # Update ELO ratings
        # Convert result from [-1, 1] to [0, 1] scale for ELO
        elo_score = (result + 1.0) / 2.0  # 1.0 -> 1.0, 0.0 -> 0.5, -1.0 -> 0.0
        new_elo1, new_elo2 = self.elo_tracker.update_ratings(
            agent1.name,
            agent2.name,
            elo_score,
            game_metadata={
                "iteration": iteration,
                "game_length": game_length,
                "agent1_archetype": agent1.archetype.value,
                "agent2_archetype": agent2.archetype.value,
            },
        )

        # Store ELO history
        self.agent_elo_history[agent1.name].append((iteration, new_elo1))
        self.agent_elo_history[agent2.name].append((iteration, new_elo2))

        # Log to TensorBoard
        self.writer.add_scalar(
            f"games/{agent1.name}_vs_{agent2.name}/result", result, iteration
        )
        self.writer.add_scalar(
            f"games/{agent1.name}_vs_{agent2.name}/length", game_length, iteration
        )

        # Log ELO ratings - both individual and unified view
        self.writer.add_scalar(f"elo_individual/{agent1.name}", new_elo1, iteration)
        self.writer.add_scalar(f"elo_individual/{agent2.name}", new_elo2, iteration)

        # Log to unified ELO graph (all agents on same plot)
        # Use add_scalars for multi-line plots
        self.writer.add_scalar(f"elo_unified/{agent1.name}", new_elo1, iteration)
        self.writer.add_scalar(f"elo_unified/{agent2.name}", new_elo2, iteration)

    def log_agent_performance(
        self,
        iteration: int,
        agent: LeagueAgent,
    ):
        """
        Log agent's overall performance metrics including ELO rating.

        Args:
            iteration: Training iteration
            agent: Agent to log
        """
        prefix = f"agents/{agent.name}"

        # Overall stats
        self.writer.add_scalar(f"{prefix}/total_games", agent.games_played, iteration)
        self.writer.add_scalar(f"{prefix}/win_rate", agent.win_rate, iteration)
        self.writer.add_scalar(f"{prefix}/total_wins", agent.total_wins, iteration)
        self.writer.add_scalar(f"{prefix}/total_losses", agent.total_losses, iteration)
        self.writer.add_scalar(f"{prefix}/total_draws", agent.total_draws, iteration)

        # ELO rating - both individual and unified view
        elo_rating = self.elo_tracker.get_rating(agent.name)
        self.writer.add_scalar(f"{prefix}/elo", elo_rating, iteration)
        self.writer.add_scalar(f"elo_individual/{agent.name}", elo_rating, iteration)
        self.writer.add_scalar(f"elo_unified/{agent.name}", elo_rating, iteration)

        # Convergence status for exploiters
        if agent.archetype in [
            AgentArchetype.MAIN_EXPLOITER,
            AgentArchetype.LEAGUE_EXPLOITER,
        ]:
            self.writer.add_scalar(
                f"{prefix}/is_converged", 1.0 if agent.is_converged else 0.0, iteration
            )
            self.writer.add_scalar(
                f"{prefix}/training_epochs", agent.training_epochs, iteration
            )

    def log_exploiter_convergence(
        self,
        iteration: int,
        exploiter: LeagueAgent,
        win_rate_vs_target: float,
        convergence_threshold: float,
    ):
        """
        Log exploiter's convergence progress.

        Args:
            iteration: Training iteration
            exploiter: Exploiter agent
            win_rate_vs_target: Current win rate vs target opponent(s)
            convergence_threshold: Win rate threshold for convergence
        """
        prefix = f"exploiters/{exploiter.name}"

        self.writer.add_scalar(
            f"{prefix}/win_rate_vs_target", win_rate_vs_target, iteration
        )
        self.writer.add_scalar(f"{prefix}/threshold", convergence_threshold, iteration)
        self.writer.add_scalar(
            f"{prefix}/distance_to_convergence",
            max(0, convergence_threshold - win_rate_vs_target),
            iteration,
        )

        # Store in history
        self.exploiter_training_history[exploiter.name].append(
            (iteration, win_rate_vs_target)
        )

    def log_league_summary(
        self,
        iteration: int,
        num_main_agents: int,
        num_main_exploiters: int,
        num_league_exploiters: int,
        num_converged_exploiters: int,
    ):
        """
        Log summary statistics for the entire league.

        Args:
            iteration: Training iteration
            num_main_agents: Total Main Agents (including historical)
            num_main_exploiters: Total Main Exploiters
            num_league_exploiters: Total League Exploiters
            num_converged_exploiters: Number of converged exploiters
        """
        self.writer.add_scalar("league/num_main_agents", num_main_agents, iteration)
        self.writer.add_scalar(
            "league/num_main_exploiters", num_main_exploiters, iteration
        )
        self.writer.add_scalar(
            "league/num_league_exploiters", num_league_exploiters, iteration
        )
        self.writer.add_scalar(
            "league/num_converged_exploiters", num_converged_exploiters, iteration
        )
        self.writer.add_scalar(
            "league/total_agents",
            num_main_agents + num_main_exploiters + num_league_exploiters,
            iteration,
        )

    def log_pfsp_stats(
        self,
        iteration: int,
        sampled_opponents: Dict[str, int],
    ):
        """
        Log PFSP opponent selection statistics.

        Args:
            iteration: Training iteration
            sampled_opponents: Dict mapping opponent name to number of times sampled
        """
        for opponent_name, count in sampled_opponents.items():
            self.writer.add_scalar(
                f"pfsp/opponent_selection/{opponent_name}", count, iteration
            )

    def evaluate_head_to_head(
        self,
        agent1: LeagueAgent,
        agent2: LeagueAgent,
        num_games: int,
        device: str = "cpu",
        iteration: Optional[int] = None,
        inference_batch_size: Optional[int] = None,
    ) -> Dict:
        """
        Evaluate two agents head-to-head.

        Args:
            agent1: First agent
            agent2: Second agent
            num_games: Number of games to play
            device: Device for models
            iteration: Training iteration (for logging)
            inference_batch_size: Batch size for position evaluation during inference

        Returns:
            Results dictionary with win rates and game statistics
        """
        print(f"\n  Evaluating {agent1.name} vs {agent2.name} ({num_games} games)...")

        # Load models
        model1 = self._load_model(agent1, device)
        model2 = self._load_model(agent2, device)

        # Create players (greedy evaluation)
        player1 = SelfPlayGame(
            model=model1,
            epsilon=0.0,
            device=device,
            inference_batch_size=inference_batch_size,
        )
        player2 = SelfPlayGame(
            model=model2,
            epsilon=0.0,
            device=device,
            inference_batch_size=inference_batch_size,
        )

        results = {
            "agent1_wins": 0,
            "agent2_wins": 0,
            "draws": 0,
            "avg_game_length": 0,
        }

        total_moves = 0
        start_time = time.time()

        for game_idx in tqdm(
            range(num_games),
            desc=f"Evaluating {agent1.name} vs {agent2.name}",
            unit="game",
        ):
            board = nokamute.Board()
            agent1_is_white = game_idx % 2 == 0
            move_count = 0

            while move_count < 400:  # Max moves
                legal_moves = board.legal_moves()
                winner = board.get_winner()

                if winner is not None:
                    if winner == "Draw":
                        results["draws"] += 1
                        result_from_agent1_perspective = 0.0
                    elif winner == "White":
                        if agent1_is_white:
                            results["agent1_wins"] += 1
                            result_from_agent1_perspective = 1.0
                        else:
                            results["agent2_wins"] += 1
                            result_from_agent1_perspective = -1.0
                    else:  # Black wins
                        if agent1_is_white:
                            results["agent2_wins"] += 1
                            result_from_agent1_perspective = -1.0
                        else:
                            results["agent1_wins"] += 1
                            result_from_agent1_perspective = 1.0

                    total_moves += move_count

                    # Log this game
                    if iteration is not None:
                        self.log_game_result(
                            iteration,
                            agent1,
                            agent2,
                            result_from_agent1_perspective,
                            move_count,
                        )

                    break

                # Select move
                current_player = board.to_move()
                is_white_turn = current_player.name == "White"

                if is_white_turn == agent1_is_white:
                    move = player1.select_move(board, legal_moves)
                else:
                    move = player2.select_move(board, legal_moves)

                board.apply(move)
                move_count += 1
            else:
                # Max moves reached - draw
                results["draws"] += 1
                total_moves += move_count

        elapsed = time.time() - start_time
        results["avg_game_length"] = total_moves / num_games
        results["agent1_win_rate"] = results["agent1_wins"] / num_games
        results["agent2_win_rate"] = results["agent2_wins"] / num_games
        results["draw_rate"] = results["draws"] / num_games
        results["time_elapsed"] = elapsed

        print(
            f"    {agent1.name}: {results['agent1_wins']}W-{results['agent2_wins']}L-{results['draws']}D "
            f"(WR: {results['agent1_win_rate']:.2%}) in {elapsed:.1f}s"
        )

        return results

    def _load_model(self, agent: LeagueAgent, device: str):
        checkpoint = torch.load(agent.model_path, map_location=device)
        model_type = checkpoint.get("model_type", "policy")
        # All legacy models have been consolidated into the heterogeneous policy
        # model. If a non-policy type is found, log a warning and instantiate
        # the heterogenous model anyway.
        from model_policy_hetero import create_policy_model

        if model_type != "policy":
            print(
                f"Warning: legacy model_type '{model_type}' found in checkpoint; using heterogeneous policy model instead"
            )

        model = create_policy_model(checkpoint.get("config", {})).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        return model

    def get_recent_win_rate(
        self,
        agent: LeagueAgent,
        opponent_name: Optional[str] = None,
        window: int = 50,
    ) -> float:
        """
        Get recent win rate for an agent.

        Args:
            agent: Agent to check
            opponent_name: If specified, get win rate vs this opponent only
            window: Number of recent games to consider

        Returns:
            Win rate in [0, 1]
        """
        if opponent_name:
            matchup_key = (agent.name, opponent_name)
            if matchup_key not in self.matchup_history:
                return 0.5

            recent_games = self.matchup_history[matchup_key][-window:]
            if not recent_games:
                return 0.5

            wins = sum(1 for _, result in recent_games if result > 0.5)
            return wins / len(recent_games)
        else:
            # Overall recent win rate
            all_games = []
            for matchup_key, games in self.matchup_history.items():
                if matchup_key[0] == agent.name:
                    all_games.extend(games)

            if not all_games:
                return 0.5

            recent_games = sorted(all_games, key=lambda x: x[0])[-window:]
            wins = sum(1 for _, result in recent_games if result > 0.5)
            return wins / len(recent_games)

    def log_elo_leaderboard(self, iteration: int, top_n: int = 10):
        """
        Log the top ELO-rated agents to TensorBoard.

        Args:
            iteration: Training iteration
            top_n: Number of top agents to log
        """
        leaderboard = self.elo_tracker.get_leaderboard(top_n)

        # Only include agents that have played at least one evaluation game
        leaderboard = [
            (agent_name, elo_rating)
            for agent_name, elo_rating in leaderboard
            if agent_name in self.agent_elo_history
            and len(self.agent_elo_history[agent_name]) > 0
        ]

        for rank, (agent_name, elo_rating) in enumerate(leaderboard, 1):
            self.writer.add_scalar(f"leaderboard/rank_{rank}", elo_rating, iteration)
            self.writer.add_text(
                f"leaderboard/rank_{rank}_name",
                agent_name,
                iteration,
            )

            # Also log to unified ELO graph for complete view
            self.writer.add_scalar(f"elo_unified/{agent_name}", elo_rating, iteration)

    def get_agent_elo(self, agent_name: str) -> float:
        """
        Get current ELO rating for an agent.

        Args:
            agent_name: Name of the agent

        Returns:
            Current ELO rating
        """
        return self.elo_tracker.get_rating(agent_name)

    def close(self):
        """Close TensorBoard writer and save ELO ratings."""
        self.elo_tracker.save()
        self.writer.close()
