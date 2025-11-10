"""
Tests for the Hive PettingZoo AEC environment.

This module tests the Hive environment using PettingZoo's test suite and custom tests.
"""

import pytest
import numpy as np
from pettingzoo.test import api_test, parallel_api_test

from pettingzoo_env import env, raw_env


class TestHiveEnvironment:
    """Test suite for Hive AEC environment."""
    
    @pytest.mark.skip(reason="PettingZoo's api_test has issues with Graph observation spaces (dtype check fails)")
    def test_api_compliance(self):
        """Test that the environment complies with PettingZoo API."""
        # Create environment
        test_env = env()
        
        # Run PettingZoo API test
        # Note: This test fails due to PettingZoo's test suite not fully supporting Graph spaces
        api_test(test_env, num_cycles=100, verbose_progress=False)
    
    def test_environment_initialization(self):
        """Test that environment initializes correctly."""
        test_env = raw_env()
        
        # Check agents
        assert len(test_env.possible_agents) == 2
        assert "player_0" in test_env.possible_agents
        assert "player_1" in test_env.possible_agents
        
        # Check metadata
        assert "render_modes" in test_env.metadata
        assert "name" in test_env.metadata
        assert test_env.metadata["name"] == "hive_v0"
    
    def test_reset(self):
        """Test environment reset functionality."""
        test_env = raw_env()
        obs, info = test_env.reset()
        
        # Check observation structure
        assert isinstance(obs, dict)
        assert "observation" in obs
        assert "action_mask" in obs
        
        # Check graph observation
        graph_obs = obs["observation"]
        assert hasattr(graph_obs, "nodes")
        assert hasattr(graph_obs, "edges")
        assert hasattr(graph_obs, "edge_links")
        
        # Check agents are reset
        assert len(test_env.agents) == 2
        assert test_env.agent_selection in test_env.agents
        
        # Check episode state
        assert test_env.num_moves == 0
        assert all(not term for term in test_env.terminations.values())
        assert all(not trunc for trunc in test_env.truncations.values())
    
    def test_action_space(self):
        """Test action space properties."""
        test_env = raw_env()
        test_env.reset()
        
        for agent in test_env.possible_agents:
            action_space = test_env.action_space(agent)
            
            # Check it's a Discrete space
            from gymnasium.spaces import Discrete
            assert isinstance(action_space, Discrete)
            
            # Check action space is large enough to include various moves
            assert action_space.n > 100  # Should have many possible moves
    
    def test_observation_space(self):
        """Test observation space properties."""
        test_env = raw_env()
        test_env.reset()
        
        for agent in test_env.possible_agents:
            obs_space = test_env.observation_space(agent)
            
            # Check it's a Dict space containing a Graph
            from gymnasium.spaces import Dict, Graph
            assert isinstance(obs_space, Dict)
            assert "observation" in obs_space.spaces
            assert "action_mask" in obs_space.spaces
            
            # Check the graph space
            graph_space = obs_space.spaces["observation"]
            assert isinstance(graph_space, Graph)
            
            # Check node space
            assert graph_space.node_space.shape == (12,)  # 12 features per node
            assert graph_space.node_space.dtype == np.float32
    
    def test_observation_format(self):
        """Test that observations are in correct format."""
        test_env = raw_env()
        obs, _ = test_env.reset()
        
        # Check keys
        assert "observation" in obs
        assert "action_mask" in obs
        
        # Check graph observation
        graph_obs = obs["observation"]
        from gymnasium.spaces.graph import GraphInstance
        assert isinstance(graph_obs, GraphInstance)
        
        # Check types
        assert isinstance(graph_obs.nodes, np.ndarray)
        assert isinstance(graph_obs.edges, np.ndarray)
        assert isinstance(graph_obs.edge_links, np.ndarray)
        
        # Check shapes
        if graph_obs.nodes.shape[0] > 0:  # If there are nodes
            assert graph_obs.nodes.ndim == 2
            assert graph_obs.nodes.shape[1] == 12  # 12 features per node
        
        if graph_obs.edge_links.shape[0] > 0:  # If there are edges
            assert graph_obs.edge_links.ndim == 2
            assert graph_obs.edge_links.shape[1] == 2  # Pairs of node indices
    
    def test_action_mask(self):
        """Test action masking functionality."""
        test_env = raw_env()
        test_env.reset()
        
        mask = test_env.action_mask(test_env.agent_selection)
        
        # Check mask properties
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == np.int8
        assert mask.shape == (test_env.action_space(test_env.agent_selection).n,)
        
        # Check that at least one action is legal
        assert np.sum(mask) > 0
        
        # Check that mask contains only 0s and 1s
        assert np.all((mask == 0) | (mask == 1))
    
    def test_step_with_legal_action(self):
        """Test stepping with a legal action."""
        test_env = raw_env()
        test_env.reset()
        
        # Get a legal action
        mask = test_env.action_mask(test_env.agent_selection)
        legal_actions = np.where(mask == 1)[0]
        assert len(legal_actions) > 0
        
        action = legal_actions[0]
        initial_agent = test_env.agent_selection
        
        # Take action
        test_env.step(action)
        
        # Check that agent selection changed
        assert test_env.agent_selection != initial_agent or \
               test_env.terminations[initial_agent] or \
               test_env.truncations[initial_agent]
        
        # Check move counter increased
        assert test_env.num_moves == 1
    
    def test_full_game_simulation(self):
        """Test a complete game simulation."""
        test_env = raw_env()
        test_env.reset()
        
        max_steps = 50  # Limit steps for testing
        step_count = 0
        
        for agent in test_env.agent_iter(max_iter=max_steps):
            obs, reward, termination, truncation, info = test_env.last()
            
            if termination or truncation:
                action = None
            else:
                # Sample from legal actions
                mask = test_env.action_mask(agent)
                legal_actions = np.where(mask == 1)[0]
                
                if len(legal_actions) > 0:
                    action = np.random.choice(legal_actions)
                else:
                    action = None
            
            test_env.step(action)
            step_count += 1
            
            # If game ended, break
            if termination or truncation:
                break
        
        # Check that we took some steps
        assert step_count > 0
    
    def test_game_termination(self):
        """Test that games can terminate correctly."""
        test_env = raw_env(max_moves=20)  # Short game for testing
        test_env.reset()
        
        # Play random moves until termination or max steps
        for agent in test_env.agent_iter(max_iter=100):
            obs, reward, termination, truncation, info = test_env.last()
            
            if termination or truncation:
                break
            
            # Sample legal action
            mask = test_env.action_mask(agent)
            legal_actions = np.where(mask == 1)[0]
            
            if len(legal_actions) > 0:
                action = np.random.choice(legal_actions)
                test_env.step(action)
        
        # Check that game either terminated or truncated
        assert any(test_env.terminations.values()) or any(test_env.truncations.values())
    
    def test_render_modes(self):
        """Test different render modes."""
        # Test ansi mode
        test_env = raw_env(render_mode="ansi")
        test_env.reset()
        
        render_output = test_env.render()
        assert isinstance(render_output, str)
        assert len(render_output) > 0
        
        # Test human mode (just check it doesn't error)
        test_env_human = raw_env(render_mode="human")
        test_env_human.reset()
        test_env_human.render()  # Should print to console
    
    def test_game_types(self):
        """Test different game type variants."""
        game_types = ["Base", "Base+M", "Base+L", "Base+P", "Base+MLP"]
        
        for game_type in game_types:
            test_env = raw_env(game_type=game_type)
            obs, _ = test_env.reset()
            
            # Check environment initialized correctly
            assert test_env.game_type == game_type
            assert isinstance(obs, dict)
            
            # Take a few steps to verify it works
            for _ in range(3):
                mask = test_env.action_mask(test_env.agent_selection)
                legal_actions = np.where(mask == 1)[0]
                
                if len(legal_actions) > 0:
                    action = np.random.choice(legal_actions)
                    test_env.step(action)
    
    def test_seeded_reset(self):
        """Test that seeded resets produce consistent results."""
        seed = 42
        
        # Create two environments with same seed
        env1 = raw_env()
        env2 = raw_env()
        
        obs1, _ = env1.reset(seed=seed)
        obs2, _ = env2.reset(seed=seed)
        
        # Check that observations are identical
        # (Initially empty board, so this mainly checks structure)
        assert obs1.keys() == obs2.keys()


def test_basic_functionality():
    """Quick smoke test for basic functionality."""
    test_env = env()
    test_env.reset()
    
    # Play a few moves
    for agent in test_env.agent_iter(max_iter=10):
        observation, reward, termination, truncation, info = test_env.last()
        
        if termination or truncation:
            action = None
        else:
            # Get action mask if available
            if hasattr(test_env, 'action_mask'):
                mask = test_env.action_mask(agent)
                legal_actions = np.where(mask == 1)[0]
                action = np.random.choice(legal_actions) if len(legal_actions) > 0 else test_env.action_space(agent).sample()
            else:
                action = test_env.action_space(agent).sample()
        
        test_env.step(action)
    
    test_env.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
