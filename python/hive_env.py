"""
TorchRL Environment for the Hive Board Game.

This module implements a custom TorchRL environment that wraps the Hive game,
providing a standardized RL interface for training agents.
"""

import torch
import numpy as np
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data.tensor_specs import (
    Composite,
    Unbounded,
    Binary,
)
from torchrl.data import Categorical
import nokamute


class HiveEnv(EnvBase):
    """
    TorchRL environment for the Hive board game.
    
    The environment represents the Hive game state as a graph, where nodes are pieces
    and edges represent adjacency relationships. Actions are discrete indices into
    the list of legal moves.
    
    Observations:
        - node_features: [num_nodes, 11] tensor with node features
        - edge_index: [2, num_edges] tensor with graph connectivity
        - action_mask: [max_actions] binary tensor indicating legal actions
        - num_nodes: scalar indicating number of nodes in current state
        - num_edges: scalar indicating number of edges in current state
    
    Actions:
        - Categorical (discrete) action space from 0 to max_actions-1
        - Only actions with action_mask[i] == 1 are legal
    
    Rewards:
        - +1 for winning
        - -1 for losing
        - 0 for draw or ongoing game
    """
    
    def __init__(
        self,
        max_actions=500,
        max_nodes=200,
        max_edges=1000,
        max_moves=200,
        device="cpu",
        batch_size=None,
    ):
        """
        Args:
            max_actions: Maximum number of possible actions (for spec)
            max_nodes: Maximum number of nodes in graph (for spec)
            max_edges: Maximum number of edges in graph (for spec)
            max_moves: Maximum moves before declaring draw
            device: Device to place tensors on
            batch_size: Batch size for environment (None for single env)
        """
        super().__init__(device=device, batch_size=batch_size or torch.Size([]))
        
        self.max_actions = max_actions
        self.max_nodes = max_nodes
        self.max_edges = max_edges
        self.max_moves = max_moves
        
        # Initialize board
        self.board = None
        self.current_legal_moves = []
        self.move_count = 0
        
        # Define specs
        self._make_specs()
        
    def _make_specs(self):
        """Create environment specifications."""
        # Observation spec
        self.observation_spec = Composite(
            node_features=Unbounded(
                shape=(self.max_nodes, 11),
                dtype=torch.float32,
            ),
            edge_index=Unbounded(
                shape=(2, self.max_edges),
                dtype=torch.long,
            ),
            action_mask=Binary(
                n=self.max_actions,
                shape=(self.max_actions,),
                dtype=torch.bool,
            ),
            num_nodes=Unbounded(
                shape=(1,),
                dtype=torch.long,
            ),
            num_edges=Unbounded(
                shape=(1,),
                dtype=torch.long,
            ),
            shape=(),
        )
        
        # Action spec - Categorical for discrete actions
        self.action_spec = Categorical(n=self.max_actions, shape=(), dtype=torch.long)
        
        # Reward spec
        self.reward_spec = Unbounded(shape=(1,), dtype=torch.float32)
        
        # Done spec
        self.done_spec = Composite(
            done=Binary(n=1, shape=(1,), dtype=torch.bool),
            terminated=Binary(n=1, shape=(1,), dtype=torch.bool),
            truncated=Binary(n=1, shape=(1,), dtype=torch.bool),
            shape=(),
        )
        
    def _reset(self, tensordict=None):
        """Reset the environment to initial state."""
        self.board = nokamute.Board()
        self.current_legal_moves = self.board.legal_moves()
        self.move_count = 0
        
        # Get observation
        obs_dict = self._get_observation()
        
        # Create output tensordict
        out = TensorDict(
            {
                "node_features": obs_dict["node_features"],
                "edge_index": obs_dict["edge_index"],
                "action_mask": obs_dict["action_mask"],
                "num_nodes": obs_dict["num_nodes"],
                "num_edges": obs_dict["num_edges"],
                "done": torch.tensor([False], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([False], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([False], dtype=torch.bool, device=self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        
        return out
    
    def _step(self, tensordict):
        """Execute one step in the environment."""
        action_idx = tensordict["action"].item()
        
        # Validate action
        if action_idx >= len(self.current_legal_moves):
            raise ValueError(
                f"Invalid action index {action_idx}. "
                f"Only {len(self.current_legal_moves)} legal moves available."
            )
        
        # Apply the move
        move = self.current_legal_moves[action_idx]
        self.board.apply(move)
        self.move_count += 1
        
        # Check for game over
        winner = self.board.get_winner()
        current_player = self.board.to_move()
        
        # Determine reward and done flags
        done = False
        terminated = False
        truncated = False
        reward = 0.0
        
        if winner is not None:
            done = True
            terminated = True
            if winner == "Draw":
                reward = 0.0
            else:
                # Reward from perspective of player who just moved
                # (opposite of current player since we already applied the move)
                if winner == "White":
                    reward = 1.0 if current_player.name == "Black" else -1.0
                else:  # Black won
                    reward = 1.0 if current_player.name == "White" else -1.0
        elif self.move_count >= self.max_moves:
            done = True
            truncated = True
            reward = 0.0
        
        # Get new legal moves
        if not done:
            self.current_legal_moves = self.board.legal_moves()
        else:
            self.current_legal_moves = []
        
        # Get observation
        obs_dict = self._get_observation()
        
        # Create output tensordict
        # Return observations at root level along with reward/done
        # TorchRL will handle the "next" key wrapping automatically
        out = TensorDict(
            {
                "node_features": obs_dict["node_features"],
                "edge_index": obs_dict["edge_index"],
                "action_mask": obs_dict["action_mask"],
                "num_nodes": obs_dict["num_nodes"],
                "num_edges": obs_dict["num_edges"],
                "reward": torch.tensor([reward], dtype=torch.float32, device=self.device),
                "done": torch.tensor([done], dtype=torch.bool, device=self.device),
                "terminated": torch.tensor([terminated], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([truncated], dtype=torch.bool, device=self.device),
            },
            batch_size=self.batch_size,
            device=self.device,
        )
        
        return out
    
    def _get_observation(self):
        """Convert board state to observation dictionary."""
        # Get graph representation from board
        node_features_list, edge_index_list = self.board.to_graph()
        
        # Convert to tensors
        if len(node_features_list) > 0:
            node_features = torch.tensor(
                node_features_list, dtype=torch.float32, device=self.device
            )
            edge_index = torch.tensor(
                edge_index_list, dtype=torch.long, device=self.device
            )
        else:
            # Empty board (shouldn't happen in practice)
            node_features = torch.zeros(
                (1, 11), dtype=torch.float32, device=self.device
            )
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=self.device)
        
        num_nodes = node_features.shape[0]
        num_edges = edge_index.shape[1]
        
        # Pad to max sizes
        node_features_padded = torch.zeros(
            (self.max_nodes, 11), dtype=torch.float32, device=self.device
        )
        node_features_padded[:num_nodes] = node_features
        
        edge_index_padded = torch.zeros(
            (2, self.max_edges), dtype=torch.long, device=self.device
        )
        edge_index_padded[:, :num_edges] = edge_index
        
        # Create action mask
        action_mask = torch.zeros(
            self.max_actions, dtype=torch.bool, device=self.device
        )
        num_legal_moves = min(len(self.current_legal_moves), self.max_actions)
        action_mask[:num_legal_moves] = True
        
        return {
            "node_features": node_features_padded,
            "edge_index": edge_index_padded,
            "action_mask": action_mask,
            "num_nodes": torch.tensor([num_nodes], dtype=torch.long, device=self.device),
            "num_edges": torch.tensor([num_edges], dtype=torch.long, device=self.device),
        }
    
    def _set_seed(self, seed):
        """Set random seed (not used in deterministic Hive)."""
        # Hive is deterministic, but we implement this for completeness
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        return seed


def make_env(device="cpu", **kwargs):
    """Factory function to create HiveEnv."""
    return HiveEnv(device=device, **kwargs)


if __name__ == "__main__":
    # Test the environment
    print("Testing HiveEnv...")
    
    env = HiveEnv(device="cpu")
    
    # Check specs
    print(f"Observation spec: {env.observation_spec}")
    print(f"Action spec: {env.action_spec}")
    print(f"Reward spec: {env.reward_spec}")
    print(f"Done spec: {env.done_spec}")
    
    # Test reset
    print("\nTesting reset...")
    td = env.reset()
    print(f"Reset output keys: {td.keys()}")
    print(f"Node features shape: {td['node_features'].shape}")
    print(f"Edge index shape: {td['edge_index'].shape}")
    print(f"Action mask sum: {td['action_mask'].sum().item()}")
    print(f"Num nodes: {td['num_nodes'].item()}")
    print(f"Num edges: {td['num_edges'].item()}")
    
    # Test step
    print("\nTesting step...")
    action = torch.tensor(0, dtype=torch.long)
    td["action"] = action
    td_next = env.step(td)
    print(f"Step output keys: {td_next.keys()}")
    print(f"Next keys: {td_next['next'].keys()}")
    print(f"Reward: {td_next['reward'].item()}")
    print(f"Done: {td_next['done'].item()}")
    
    # Test rollout
    print("\nTesting rollout...")
    env.reset()
    rollout_td = env.rollout(max_steps=10)
    print(f"Rollout shape: {rollout_td.batch_size}")
    print(f"Rollout keys: {rollout_td.keys()}")
    
    print("\nHiveEnv test passed!")
