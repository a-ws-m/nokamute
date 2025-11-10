"""
Example usage of the Hive PettingZoo AEC environment.

This script demonstrates:
1. Creating and resetting the environment
2. Playing a game with random legal moves
3. Accessing observations (graph structure)
4. Using action masks for legal moves
5. Different render modes
"""

import numpy as np
from pettingzoo_env import env, raw_env


def example_basic_usage():
    """Basic example of playing a game."""
    print("=" * 70)
    print("EXAMPLE 1: Basic Usage")
    print("=" * 70)
    
    # Create environment (with wrappers for safety)
    game_env = env(render_mode="human")
    
    # Reset environment
    game_env.reset(seed=42)
    
    # Play game with random legal moves
    for agent in game_env.agent_iter(max_iter=100):
        observation, reward, termination, truncation, info = game_env.last()
        
        if termination or truncation:
            print(f"\nGame Over!")
            print(f"Final Rewards: {game_env.rewards}")
            action = None
        else:
            # Get action mask to see legal moves
            mask = observation["action_mask"]
            legal_actions = np.where(mask == 1)[0]
            
            # Sample a random legal action
            action = np.random.choice(legal_actions)
        
        game_env.step(action)
    
    game_env.close()
    print("\n")


def example_observation_structure():
    """Example showing how to access the graph observation."""
    print("=" * 70)
    print("EXAMPLE 2: Observation Structure")
    print("=" * 70)
    
    game_env = raw_env()
    obs, _ = game_env.reset()
    
    print(f"\nObservation is a dictionary with keys: {obs.keys()}")
    print(f"\n1. Graph Observation (game state):")
    graph_obs = obs["observation"]
    print(f"   Type: {type(graph_obs)}")
    print(f"   Nodes shape: {graph_obs.nodes.shape}")
    print(f"   Edges shape: {graph_obs.edges.shape}")
    print(f"   Edge links shape: {graph_obs.edge_links.shape}")
    
    print(f"\n2. Action Mask (legal moves):")
    action_mask = obs["action_mask"]
    print(f"   Type: {type(action_mask)}")
    print(f"   Shape: {action_mask.shape}")
    print(f"   Number of legal actions: {action_mask.sum()}")
    
    # Play a few moves to see how the graph grows
    print(f"\n3. Graph growth over time:")
    for i in range(5):
        mask = game_env.action_mask(game_env.agent_selection)
        legal_actions = np.where(mask == 1)[0]
        action = np.random.choice(legal_actions)
        
        game_env.step(action)
        obs = game_env.observe(game_env.agent_selection)
        
        print(f"   After move {i+1}: {obs['observation'].nodes.shape[0]} nodes, "
              f"{obs['observation'].edge_links.shape[0]} edges")
    
    game_env.close()
    print("\n")


def example_action_masking():
    """Example showing how to use action masks."""
    print("=" * 70)
    print("EXAMPLE 3: Action Masking")
    print("=" * 70)
    
    game_env = raw_env()
    game_env.reset()
    
    # Get legal actions for first move
    agent = game_env.agent_selection
    mask = game_env.action_mask(agent)
    legal_actions = np.where(mask == 1)[0]
    
    print(f"\nAgent: {agent}")
    print(f"Total action space size: {len(mask)}")
    print(f"Number of legal actions: {len(legal_actions)}")
    print(f"\nFirst 10 legal move strings:")
    
    for i, action_idx in enumerate(legal_actions[:10]):
        move_string = game_env._action_to_string[action_idx]
        print(f"   {i+1}. Action {action_idx}: '{move_string}'")
    
    game_env.close()
    print("\n")


def example_game_variants():
    """Example showing different game variants."""
    print("=" * 70)
    print("EXAMPLE 4: Game Variants")
    print("=" * 70)
    
    variants = ["Base", "Base+M", "Base+L", "Base+P", "Base+MLP"]
    
    for variant in variants:
        game_env = raw_env(game_type=variant)
        game_env.reset()
        
        # Play a few moves
        for _ in range(10):
            mask = game_env.action_mask(game_env.agent_selection)
            legal_actions = np.where(mask == 1)[0]
            
            if len(legal_actions) == 0:
                break
            
            action = np.random.choice(legal_actions)
            game_env.step(action)
        
        print(f"\n{variant}: Played {game_env.num_moves} moves")
        game_env.close()
    
    print("\n")


def example_with_custom_policy():
    """Example with a simple custom policy."""
    print("=" * 70)
    print("EXAMPLE 5: Custom Policy")
    print("=" * 70)
    
    def simple_policy(env, agent):
        """
        Simple policy that prefers:
        1. Placing the Queen early (within first 4 moves)
        2. Random otherwise
        """
        mask = env.action_mask(agent)
        legal_actions = np.where(mask == 1)[0]
        
        if len(legal_actions) == 0:
            return None
        
        # If early in game, try to place queen
        if env.num_moves < 4:
            # Look for queen placement moves
            queen_piece = "wQ" if agent == "player_0" else "bQ"
            for action_idx in legal_actions:
                move_str = env._action_to_string[action_idx]
                if queen_piece in move_str:
                    return action_idx
        
        # Otherwise random
        return np.random.choice(legal_actions)
    
    game_env = raw_env(render_mode="ansi")
    game_env.reset(seed=42)
    
    move_count = 0
    for agent in game_env.agent_iter(max_iter=30):
        observation, reward, termination, truncation, info = game_env.last()
        
        if termination or truncation:
            print(f"\nGame ended after {move_count} moves")
            print(game_env.render())
            break
        
        action = simple_policy(game_env, agent)
        game_env.step(action)
        move_count += 1
    
    game_env.close()
    print("\n")


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_observation_structure()
    example_action_masking()
    example_game_variants()
    example_with_custom_policy()
    
    print("=" * 70)
    print("All examples completed!")
    print("=" * 70)
