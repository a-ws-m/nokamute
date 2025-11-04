"""
Self-play evaluation module for comparing two Hive agents.

This module provides functionality to pit two models against each other
and compute win rates, which is used to gate model updates in training.
"""

import torch
import numpy as np
from tensordict import TensorDict
from hive_env import HiveEnv


class SelfPlayEvaluator:
    """
    Evaluates two agents by having them play against each other.
    
    This is used to implement the 55% win rate threshold for accepting
    new models during training.
    """
    
    def __init__(self, device="cpu"):
        """
        Args:
            device: Device to run evaluation on
        """
        self.device = device
    
    def evaluate(
        self,
        agent1,
        agent2,
        num_games=100,
        max_moves=200,
        verbose=False,
    ):
        """
        Evaluate two agents by playing them against each other.
        
        Args:
            agent1: First agent (ProbabilisticActor or policy module)
            agent2: Second agent (ProbabilisticActor or policy module)
            num_games: Number of games to play
            max_moves: Maximum moves per game
            verbose: Whether to print progress
            
        Returns:
            Dictionary with evaluation statistics:
                - agent1_wins: Number of games won by agent1
                - agent2_wins: Number of games won by agent2
                - draws: Number of draws
                - agent1_win_rate: Win rate for agent1
                - avg_game_length: Average number of moves per game
        """
        agent1.eval()
        agent2.eval()
        
        results = {
            "agent1_wins": 0,
            "agent2_wins": 0,
            "draws": 0,
            "total_moves": 0,
        }
        
        for game_idx in range(num_games):
            # Alternate who plays first
            if game_idx % 2 == 0:
                white_agent = agent1
                black_agent = agent2
                white_is_agent1 = True
            else:
                white_agent = agent2
                black_agent = agent1
                white_is_agent1 = False
            
            # Play one game
            winner, num_moves = self._play_game(
                white_agent,
                black_agent,
                max_moves=max_moves,
            )
            
            results["total_moves"] += num_moves
            
            # Record result
            if winner == "Draw":
                results["draws"] += 1
            elif winner == "White":
                if white_is_agent1:
                    results["agent1_wins"] += 1
                else:
                    results["agent2_wins"] += 1
            else:  # Black
                if white_is_agent1:
                    results["agent2_wins"] += 1
                else:
                    results["agent1_wins"] += 1
            
            if verbose and (game_idx + 1) % 10 == 0:
                print(f"Completed {game_idx + 1}/{num_games} games...")
        
        # Compute statistics
        results["agent1_win_rate"] = results["agent1_wins"] / num_games
        results["agent2_win_rate"] = results["agent2_wins"] / num_games
        results["draw_rate"] = results["draws"] / num_games
        results["avg_game_length"] = results["total_moves"] / num_games
        
        return results
    
    def _play_game(self, white_agent, black_agent, max_moves=200):
        """
        Play a single game between two agents.
        
        Args:
            white_agent: Agent playing as White
            black_agent: Agent playing as Black
            max_moves: Maximum number of moves
            
        Returns:
            winner: "White", "Black", or "Draw"
            num_moves: Number of moves played
        """
        env = HiveEnv(max_moves=max_moves, device=self.device)
        td = env.reset()
        
        current_agent = white_agent
        move_count = 0
        
        with torch.no_grad():
            while move_count < max_moves:
                # Select action using current agent
                td_with_action = current_agent(td)
                
                # Step environment
                td_step = env.step(td_with_action)
                move_count += 1
                
                # Check if game is done (from the step result)
                if td_step["next"]["done"].item():
                    break
                
                # Update td to next state for next iteration
                # Extract next state observations from nested "next" key
                td = TensorDict(
                    {
                        "node_features": td_step["next"]["node_features"],
                        "edge_index": td_step["next"]["edge_index"],
                        "action_mask": td_step["next"]["action_mask"],
                        "num_nodes": td_step["next"]["num_nodes"],
                        "num_edges": td_step["next"]["num_edges"],
                        "done": td_step["next"]["done"],
                        "terminated": td_step["next"]["terminated"],
                        "truncated": td_step["next"]["truncated"],
                    },
                    batch_size=td_step.batch_size,
                    device=self.device,
                )
                
                # Switch agents
                current_agent = black_agent if current_agent == white_agent else white_agent
        
        # Determine winner
        import nokamute
        winner = env.board.get_winner()
        
        if winner is None:
            winner = "Draw"
        
        return winner, move_count
    
    def check_improvement_threshold(
        self,
        new_agent,
        old_agent,
        num_games=100,
        threshold=0.55,
        verbose=True,
    ):
        """
        Check if new agent beats old agent at the threshold rate.
        
        Args:
            new_agent: New agent to evaluate
            old_agent: Baseline agent to compare against
            num_games: Number of games to play
            threshold: Win rate threshold (default 0.55 = 55%)
            verbose: Whether to print results
            
        Returns:
            bool: True if new agent meets threshold, False otherwise
        """
        results = self.evaluate(
            agent1=new_agent,
            agent2=old_agent,
            num_games=num_games,
            verbose=verbose,
        )
        
        new_agent_win_rate = results["agent1_win_rate"]
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Self-Play Evaluation Results")
            print(f"{'='*60}")
            print(f"New agent wins: {results['agent1_wins']}/{num_games} ({new_agent_win_rate:.1%})")
            print(f"Old agent wins: {results['agent2_wins']}/{num_games} ({results['agent2_win_rate']:.1%})")
            print(f"Draws: {results['draws']}/{num_games} ({results['draw_rate']:.1%})")
            print(f"Avg game length: {results['avg_game_length']:.1f} moves")
            print(f"\nThreshold: {threshold:.1%}")
            print(f"Result: {'ACCEPT' if new_agent_win_rate >= threshold else 'REJECT'}")
            print(f"{'='*60}\n")
        
        return new_agent_win_rate >= threshold


def play_single_game_demonstration(agent, max_moves=200, device="cpu"):
    """
    Play and display a single game for demonstration purposes.
    
    Args:
        agent: Agent to play the game
        max_moves: Maximum number of moves
        device: Device to run on
        
    Returns:
        Game information dictionary
    """
    env = HiveEnv(max_moves=max_moves, device=device)
    td = env.reset()
    
    moves = []
    agent.eval()
    
    with torch.no_grad():
        for move_idx in range(max_moves):
            # Get action from agent
            td_with_action = agent(td)
            action = td_with_action["action"].item()
            
            # Record move
            moves.append({
                "move_num": move_idx + 1,
                "player": env.board.to_move().name,
                "action": action,
                "num_legal_moves": td["action_mask"].sum().item(),
            })
            
            # Step environment
            td = env.step(td_with_action)
            
            # Check if game is done
            if td["done"].item():
                break
    
    winner = env.board.get_winner()
    if winner is None:
        winner = "Draw"
    
    return {
        "winner": winner,
        "num_moves": len(moves),
        "moves": moves,
    }


if __name__ == "__main__":
    # Test the evaluator
    print("Testing SelfPlayEvaluator...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create dummy agents (random policy)
    from ppo_models import make_ppo_models
    
    actor1, _ = make_ppo_models(device=device)
    actor2, _ = make_ppo_models(device=device)
    
    # Evaluate
    evaluator = SelfPlayEvaluator(device=device)
    
    print("\nRunning evaluation with 10 games...")
    results = evaluator.evaluate(
        agent1=actor1,
        agent2=actor2,
        num_games=10,
        verbose=True,
    )
    
    print(f"\nResults:")
    print(f"  Agent 1 wins: {results['agent1_wins']}")
    print(f"  Agent 2 wins: {results['agent2_wins']}")
    print(f"  Draws: {results['draws']}")
    print(f"  Agent 1 win rate: {results['agent1_win_rate']:.1%}")
    print(f"  Avg game length: {results['avg_game_length']:.1f}")
    
    # Test threshold check
    print("\nTesting improvement threshold...")
    accepts = evaluator.check_improvement_threshold(
        new_agent=actor1,
        old_agent=actor2,
        num_games=10,
        threshold=0.55,
        verbose=True,
    )
    
    print(f"Accepts new agent: {accepts}")
    
    print("\nSelfPlayEvaluator test passed!")
