"""
Test script for the DQN RLModule with Hive environment.

This script tests that the custom RLModule can be instantiated and
that it can process observations from the Hive environment.
"""

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.spaces.graph import GraphInstance
from pettingzoo_env.hive_env import env as hive_env
from ray.rllib.core.columns import Columns
from rllib_dqn import HiveGNNRLModule, get_default_model_config


def test_rl_module_creation():
    """Test that we can create the RLModule."""
    print("Testing RLModule creation...")
    
    # Create a dummy environment to get spaces
    from ray.rllib.env import PettingZooEnv
    
    def env_creator():
        return hive_env(game_type="Base+MLP", max_moves=400)
    
    # Wrap in PettingZooEnv for RLlib
    test_env = PettingZooEnv(env_creator())
    observation_space = test_env.observation_space
    action_space = test_env.action_space
    
    print(f"Observation space: {observation_space}")
    print(f"Action space: {action_space}")
    
    # Extract single-agent spaces (they're the same for both players)
    single_obs_space = observation_space["player_0"]
    single_action_space = action_space["player_0"]
    print(f"Single agent action space size: {single_action_space.n}")
    
    # Create RLModule with new API
    model_config = {
        "node_features": 12,
        "hidden_dim": 64,
        "num_layers": 2,
        "num_heads": 4,
        "dropout": 0.1,
    }
    
    # Create RLModule using new API (direct parameters, not config object)
    rl_module = HiveGNNRLModule(
        observation_space=single_obs_space,
        action_space=single_action_space,
        model_config=model_config,
    )
    print(f"✓ RLModule created successfully")
    print(f"  Model parameters: {sum(p.numel() for p in rl_module.parameters()):,}")
    
    return rl_module, test_env


def test_forward_pass(rl_module, env):
    """Test forward pass with a real observation."""
    print("\nTesting forward pass...")
    
    # Reset environment (PettingZooEnv wrapper)
    obs, info = env.reset()
    
    # PettingZooEnv wraps observations with agent IDs
    print(f"Observation keys: {obs.keys()}")
    
    # Get observation for the current agent
    agent_id = list(obs.keys())[0]  # Get first agent
    agent_obs = obs[agent_id]
    print(f"Agent {agent_id} observation keys: {agent_obs.keys()}")
    
    # Extract graph observation
    graph_obs = agent_obs["observation"]
    action_mask = agent_obs["action_mask"]
    
    print(f"Graph observation type: {type(graph_obs)}")
    print(f"  Nodes shape: {graph_obs.nodes.shape}")
    print(f"  Edges shape: {graph_obs.edges.shape}")
    print(f"  Edge links shape: {graph_obs.edge_links.shape}")
    print(f"Action mask shape: {action_mask.shape}")
    print(f"Legal actions: {np.sum(action_mask)}/{len(action_mask)}")
    
    # Prepare batch for RLModule
    # Convert numpy arrays to torch tensors
    batch = {
        Columns.OBS: {
            "observation": {
                "nodes": torch.from_numpy(graph_obs.nodes).float(),
                "edges": torch.from_numpy(graph_obs.edges).float(),
                "edge_links": torch.from_numpy(graph_obs.edge_links).long(),
            },
            "action_mask": torch.from_numpy(action_mask).float(),
        }
    }
    
    # Forward pass
    with torch.no_grad():
        output = rl_module._forward_inference(batch)
    
    q_values = output[Columns.ACTION_DIST_INPUTS]
    print(f"\n✓ Forward pass successful")
    print(f"  Q-values shape: {q_values.shape}")
    print(f"  Q-values range: [{q_values.min().item():.3f}, {q_values.max().item():.3f}]")
    
    # Check that masked actions have very negative Q-values
    masked_actions = action_mask == 0
    if masked_actions.any():
        masked_q = q_values[0, masked_actions]
        legal_q = q_values[0, action_mask == 1]
        print(f"  Masked Q-values range: [{masked_q.min().item():.3f}, {masked_q.max().item():.3f}]")
        print(f"  Legal Q-values range: [{legal_q.min().item():.3f}, {legal_q.max().item():.3f}]")
    
    # Select best legal action
    legal_q_values = torch.where(
        torch.from_numpy(action_mask).bool(),
        q_values[0],
        torch.tensor(float('-inf'))
    )
    best_action = torch.argmax(legal_q_values).item()
    print(f"  Best action (greedy): {best_action}")
    
    return q_values


def test_multiple_steps(rl_module, env):
    """Test multiple environment steps."""
    print("\nTesting multiple environment steps...")
    
    # Reset environment
    obs, info = env.reset()
    done = False
    step_count = 0
    max_steps = 10
    
    while not done and step_count < max_steps:
        # Get observation for current agent (PettingZooEnv wraps with agent IDs)
        if not isinstance(obs, dict) or not obs:
            break
        
        agent_id = list(obs.keys())[0]
        agent_obs = obs[agent_id]
        graph_obs = agent_obs["observation"]
        action_mask = agent_obs["action_mask"]
        
        # Prepare batch
        batch = {
            Columns.OBS: {
                "observation": {
                    "nodes": torch.from_numpy(graph_obs.nodes).float(),
                    "edges": torch.from_numpy(graph_obs.edges).float(),
                    "edge_links": torch.from_numpy(graph_obs.edge_links).long(),
                },
                "action_mask": torch.from_numpy(action_mask).float(),
            }
        }
        
        # Get Q-values
        with torch.no_grad():
            output = rl_module._forward_inference(batch)
        q_values = output[Columns.ACTION_DIST_INPUTS]
        
        # Select best legal action
        legal_q_values = torch.where(
            torch.from_numpy(action_mask).bool(),
            q_values[0],
            torch.tensor(float('-inf'))
        )
        action = torch.argmax(legal_q_values).item()
        
        # Take action (PettingZooEnv handles step differently)
        obs, reward, terminated, truncated, info = env.step({agent_id: action})
        step_count += 1
        
        # Check if done
        done = terminated or truncated
        
        print(f"  Step {step_count}: took action {action}, done={done}")
    
    print(f"✓ Completed {step_count} steps successfully")


def main():
    print("="*60)
    print("DQN RLModule Test Suite")
    print("="*60)
    
    try:
        # Test 1: Create RLModule
        rl_module, env = test_rl_module_creation()
        
        # Test 2: Forward pass
        test_forward_pass(rl_module, env)
        
        # Test 3: Multiple steps
        test_multiple_steps(rl_module, env)
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"✗ Test failed with error:")
        print(f"  {type(e).__name__}: {e}")
        print("="*60)
        raise


if __name__ == "__main__":
    main()
