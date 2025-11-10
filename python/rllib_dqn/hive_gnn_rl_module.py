"""
Custom RLModule for Hive using Graph Neural Networks with RLlib.

This module implements a custom RLModule that wraps the HiveGNN architecture
for use with RLlib's DQN algorithm.
"""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import OverrideToImplementCustomLogic, override
from ray.rllib.utils.framework import try_import_torch
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool

torch, nn = try_import_torch()


class HiveGNNRLModule(TorchRLModule):
    """
    Custom RLModule for Hive using Graph Neural Networks.
    
    This module adapts the HiveGNN architecture for use with RLlib's DQN algorithm.
    It processes graph observations and outputs Q-values for each possible action.
    
    The architecture:
    - Processes node features (color, bug type, height, current player)
    - Uses multiple GAT (Graph Attention) layers for message passing
    - Global pooling (mean + max) for graph-level representation
    - Outputs Q-values for all actions in the action space
    """
        
    @OverrideToImplementCustomLogic
    @override(RLModule)
    def setup(self):
        """
        Initialize the GNN architecture.
        """
        # Get model config
        model_config = self.model_config
        
        # Architecture hyperparameters
        self.node_features = model_config.get("node_features", 12)  # color + bug_onehot(9) + height + current_player
        self.hidden_dim = model_config.get("hidden_dim", 128)
        self.num_layers = model_config.get("num_layers", 4)
        self.num_heads = model_config.get("num_heads", 4)
        self.dropout = model_config.get("dropout", 0.1)
        
        # Get action space size (number of possible actions)
        self.num_actions = self.action_space.n
        
        # Initial embedding layer
        self.node_embedding = nn.Linear(self.node_features, self.hidden_dim)
        
        # GAT layers with multi-head attention
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(self.num_layers):
            in_channels = self.hidden_dim
            out_channels = self.hidden_dim // self.num_heads
            
            self.gat_layers.append(
                GATv2Conv(
                    in_channels,
                    out_channels,
                    heads=self.num_heads,
                    dropout=self.dropout,
                    concat=True,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(self.hidden_dim))
        
        # Q-value head - outputs Q-value for each action
        self.q_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, self.num_actions),
        )
    
    def _process_graph_observation(self, obs_dict):
        """
        Process graph observation into format suitable for GNN.
        
        Args:
            obs_dict: Dictionary with keys:
                - 'node_features': [batch_size?, max_nodes, feature_dim] or [max_nodes, feature_dim]
                - 'adjacency_matrix': [batch_size?, max_nodes, max_nodes] or [max_nodes, max_nodes]
                - 'node_mask': [batch_size?, max_nodes] or [max_nodes]
            
        Returns:
            Tuple of (node_features, edge_index, batch_indices)
        """
        node_features = obs_dict["node_features"]
        adjacency = obs_dict["adjacency_matrix"]
        node_mask = obs_dict["node_mask"]
        
        # Check if we have a batch dimension
        if len(node_features.shape) == 3:
            # Batched case: [batch_size, max_nodes, features]
            batch_size = node_features.shape[0]
            max_nodes = node_features.shape[1]
            
            # Flatten nodes and create batch indices
            nodes_flat = node_features.reshape(-1, node_features.shape[-1])
            batch_idx = torch.arange(batch_size, device=node_features.device).repeat_interleave(max_nodes)
            
            # Convert adjacency matrix to edge_index
            # For each graph in batch, extract edges and offset node indices
            edge_index_list = []
            for i in range(batch_size):
                # Get edges for this graph (upper triangle to avoid duplicates)
                adj_i = adjacency[i]
                edges_i = torch.nonzero(adj_i, as_tuple=False)  # [num_edges, 2]
                
                if edges_i.shape[0] > 0:
                    # Add offset based on which graph in batch
                    edges_i = edges_i + i * max_nodes
                    edge_index_list.append(edges_i)
            
            if edge_index_list:
                edge_links = torch.cat(edge_index_list, dim=0)
                edge_index = edge_links.t().contiguous()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=node_features.device)
            
            return nodes_flat, edge_index, batch_idx
        else:
            # Single graph case: [max_nodes, features]
            nodes = node_features
            
            # Convert adjacency matrix to edge_index
            edge_links = torch.nonzero(adjacency, as_tuple=False)  # [num_edges, 2]
            
            if edge_links.shape[0] > 0:
                edge_index = edge_links.t().contiguous()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long, device=nodes.device)
            
            return nodes, edge_index, None
    
    def _gnn_forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the GNN.
        
        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for each node [num_nodes] (optional)
            
        Returns:
            Graph-level embedding
        """
        # Initial embedding
        x = self.node_embedding(x)
        x = torch.relu(x)
        
        # Handle empty graph case
        if x.shape[0] == 0:
            # Return zero embedding
            return torch.zeros((1, self.hidden_dim * 2), device=x.device)
        
        # GAT layers with residual connections
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            x_residual = x
            x = gat(x, edge_index)
            x = bn(x)
            x = torch.relu(x)
            
            # Residual connection (skip first layer)
            if i > 0:
                x = x + x_residual
        
        # Global pooling (combine mean and max)
        if batch is None:
            # Single graph
            graph_mean = torch.mean(x, dim=0, keepdim=True)
            graph_max = torch.max(x, dim=0, keepdim=True)[0]
        else:
            # Batch of graphs
            graph_mean = global_mean_pool(x, batch)
            graph_max = global_max_pool(x, batch)
        
        graph_embedding = torch.cat([graph_mean, graph_max], dim=-1)
        return graph_embedding
    
    @OverrideToImplementCustomLogic
    def _forward(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Generic forward pass.
        
        Args:
            batch: Batch containing observations and other data
            
        Returns:
            Dictionary with Q-values
        """
        # Extract observations
        obs = batch[Columns.OBS]
        
        # Process graph observation (now expects node_features, adjacency_matrix, node_mask)
        nodes, edge_index, batch_idx = self._process_graph_observation(obs)
        
        # Forward through GNN
        graph_embedding = self._gnn_forward(nodes, edge_index, batch_idx)
        
        # Compute Q-values
        q_values = self.q_head(graph_embedding)
        
        # Apply action mask if available
        if "action_mask" in obs:
            action_mask = obs["action_mask"]
            # Convert mask to float and replace invalid actions with large negative value
            mask_float = action_mask.float()
            # Where mask is 0 (invalid), set Q-value to very negative number
            q_values = torch.where(
                mask_float.bool(),
                q_values,
                torch.full_like(q_values, -1e10)
            )
        
        return {Columns.ACTION_DIST_INPUTS: q_values}
    
    @OverrideToImplementCustomLogic
    @override(TorchRLModule)
    def _forward_inference(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Forward pass for inference (action selection).
        
        Args:
            batch: Batch containing observations
            
        Returns:
            Dictionary with Q-values for action selection
        """
        return self._forward(batch, **kwargs)
    
    @OverrideToImplementCustomLogic
    @override(TorchRLModule)
    def _forward_exploration(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Forward pass for exploration (training with epsilon-greedy).
        
        Args:
            batch: Batch containing observations
            
        Returns:
            Dictionary with Q-values for exploration
        """
        return self._forward(batch, **kwargs)
    
    @OverrideToImplementCustomLogic
    @override(RLModule)
    def _forward_train(self, batch: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Forward pass for training (computing Q-values for loss calculation).
        
        Args:
            batch: Batch containing observations
            
        Returns:
            Dictionary with Q-values for training
        """
        return self._forward(batch, **kwargs)


def get_default_model_config() -> Dict[str, Any]:
    """
    Get default model configuration for the HiveGNNRLModule.
    
    Returns:
        Default model configuration dictionary
    """
    return {
        "node_features": 12,
        "hidden_dim": 128,
        "num_layers": 4,
        "num_heads": 4,
        "dropout": 0.1,
    }
