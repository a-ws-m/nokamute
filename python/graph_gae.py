"""
Custom Generalized Advantage Estimation (GAE) for Graph Neural Networks.

This module implements GAE without using vmap, making it compatible with
Graph Neural Networks that have variable-sized inputs.
"""

import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Optional


class GraphGAE(nn.Module):
    """
    Generalized Advantage Estimation for graph-based environments.
    
    This implementation processes samples sequentially instead of using vmap,
    which allows it to work with Graph Neural Networks that can't handle
    the extra batch dimension added by vmap.
    
    Args:
        value_network: Critic network that outputs state values
        gamma: Discount factor for future rewards (default: 0.99)
        lmbda: GAE lambda parameter for bias-variance tradeoff (default: 0.95)
        device: Device to run computations on
    """
    
    def __init__(
        self,
        value_network: nn.Module,
        gamma: float = 0.99,
        lmbda: float = 0.95,
        device: str = "cpu",
    ):
        super().__init__()
        self.value_network = value_network
        self.gamma = gamma
        self.lmbda = lmbda
        self.device = device
        
    def forward(self, tensordict: TensorDict) -> TensorDict:
        """
        Compute advantages and value targets using GAE.
        
        Adds the following keys to tensordict:
            - "advantage": GAE advantage estimates
            - "value_target": Value function targets (rewards-to-go)
            
        Args:
            tensordict: TensorDict with keys:
                - "next", "reward": Rewards for each step
                - "next", "done": Done flags
                - Observation keys for value network
                
        Returns:
            tensordict: Updated with advantage and value_target keys
        """
        with torch.no_grad():
            # Get batch dimensions
            batch_size = tensordict.batch_size
            if len(batch_size) == 0:
                # Single sample
                num_steps = 1
            else:
                num_steps = batch_size[0]
            
            # Extract rewards and dones
            rewards = tensordict["next", "reward"].squeeze(-1)  # [num_steps]
            dones = tensordict["next", "done"].squeeze(-1)      # [num_steps]
            
            # Compute values for current states (one at a time to avoid vmap)
            values = []
            for i in range(num_steps):
                # Extract single sample without vmap
                sample = TensorDict(
                    {
                        "node_features": tensordict["node_features"][i].unsqueeze(0),
                        "edge_index": tensordict["edge_index"][i].unsqueeze(0),
                        "num_nodes": tensordict["num_nodes"][i].unsqueeze(0),
                    },
                    batch_size=torch.Size([1]),
                    device=self.device,
                )
                val = self.value_network(sample)["state_value"]
                values.append(val.squeeze())
            values = torch.stack(values)  # [num_steps]
            
            # Compute next state values
            next_values = []
            for i in range(num_steps):
                if dones[i]:
                    # Terminal state has value 0
                    next_values.append(torch.tensor(0.0, device=self.device))
                else:
                    # Extract next state observations from nested "next" key
                    next_sample = TensorDict(
                        {
                            "node_features": tensordict["next"]["node_features"][i].unsqueeze(0),
                            "edge_index": tensordict["next"]["edge_index"][i].unsqueeze(0),
                            "num_nodes": tensordict["next"]["num_nodes"][i].unsqueeze(0),
                        },
                        batch_size=torch.Size([1]),
                        device=self.device,
                    )
                    next_val = self.value_network(next_sample)["state_value"]
                    next_values.append(next_val.squeeze())
            next_values = torch.stack(next_values)  # [num_steps]
            
            # Compute TD errors (delta_t = r_t + gamma * V(s_{t+1}) - V(s_t))
            td_errors = rewards + self.gamma * next_values * (~dones).float() - values
            
            # Compute GAE advantages using reverse iteration
            advantages = torch.zeros_like(rewards)
            gae = 0.0
            
            for t in reversed(range(num_steps)):
                if dones[t]:
                    # Reset GAE at episode boundaries
                    gae = 0.0
                
                # GAE formula: A_t = δ_t + (γλ) * A_{t+1}
                gae = td_errors[t] + self.gamma * self.lmbda * gae * (~dones[t]).float()
                advantages[t] = gae
            
            # Compute value targets (for critic loss)
            value_targets = advantages + values
            
            # Add to tensordict
            tensordict["advantage"] = advantages.unsqueeze(-1)  # [num_steps, 1]
            tensordict["value_target"] = value_targets.unsqueeze(-1)  # [num_steps, 1]
            
        return tensordict


class GraphGAEWithBootstrap(GraphGAE):
    """
    GAE with bootstrap value for incomplete episodes.
    
    Useful when collecting fixed-length rollouts that may not end at
    episode boundaries.
    """
    
    def forward(self, tensordict: TensorDict) -> TensorDict:
        """
        Compute advantages with bootstrap value for last state if not done.
        """
        with torch.no_grad():
            # Get batch dimensions
            batch_size = tensordict.batch_size
            if len(batch_size) == 0:
                num_steps = 1
            else:
                num_steps = batch_size[0]
            
            # Extract rewards and dones
            rewards = tensordict["next", "reward"].squeeze(-1)
            dones = tensordict["next", "done"].squeeze(-1)
            
            # Compute values for current states
            values = []
            for i in range(num_steps):
                sample = TensorDict(
                    {
                        "node_features": tensordict["node_features"][i].unsqueeze(0),
                        "edge_index": tensordict["edge_index"][i].unsqueeze(0),
                        "num_nodes": tensordict["num_nodes"][i].unsqueeze(0),
                    },
                    batch_size=torch.Size([1]),
                    device=self.device,
                )
                val = self.value_network(sample)["state_value"]
                values.append(val.squeeze())
            values = torch.stack(values)
            
            # Compute next state values
            next_values = []
            for i in range(num_steps):
                # Always compute next value for bootstrapping
                next_sample = TensorDict(
                    {
                        "node_features": tensordict["next"]["node_features"][i].unsqueeze(0),
                        "edge_index": tensordict["next"]["edge_index"][i].unsqueeze(0),
                        "num_nodes": tensordict["next"]["num_nodes"][i].unsqueeze(0),
                    },
                    batch_size=torch.Size([1]),
                    device=self.device,
                )
                next_val = self.value_network(next_sample)["state_value"]
                next_values.append(next_val.squeeze())
            next_values = torch.stack(next_values)
            
            # Mask out next values for done states
            next_values = next_values * (~dones).float()
            
            # Compute TD errors
            td_errors = rewards + self.gamma * next_values - values
            
            # Compute GAE advantages
            advantages = torch.zeros_like(rewards)
            gae = 0.0
            
            for t in reversed(range(num_steps)):
                # Don't reset GAE at boundaries (use bootstrap)
                gae = td_errors[t] + self.gamma * self.lmbda * gae
                advantages[t] = gae
            
            # Compute value targets
            value_targets = advantages + values
            
            # Add to tensordict
            tensordict["advantage"] = advantages.unsqueeze(-1)
            tensordict["value_target"] = value_targets.unsqueeze(-1)
            
        return tensordict


def normalize_advantages(tensordict: TensorDict, eps: float = 1e-8) -> TensorDict:
    """
    Normalize advantages to have zero mean and unit variance.
    
    This is a common technique in PPO to improve training stability.
    
    Args:
        tensordict: TensorDict with "advantage" key
        eps: Small constant for numerical stability
        
    Returns:
        tensordict: Updated with normalized advantages
    """
    advantages = tensordict["advantage"]
    
    # Normalize
    mean = advantages.mean()
    std = advantages.std()
    normalized = (advantages - mean) / (std + eps)
    
    tensordict["advantage"] = normalized
    return tensordict
