"""
Custom GNN model for Hive using RLlib's old API (TorchModelV2).
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


class HiveGNNModel(TorchModelV2, nn.Module):
    """
    Custom GNN model for processing Hive board states.
    
    Uses Graph Attention Networks (GAT) to process the graph-structured
    board representation with node features and adjacency information.
    """
    
    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: dict,
        name: str,
        **kwargs,
    ):
        """
        Initialize the HiveGNNModel.
        
        Args:
            obs_space: Observation space (Dict with node_features, adjacency_matrix, node_mask)
            action_space: Action space
            num_outputs: Number of outputs (action space size)
            model_config: Model configuration dictionary
            name: Model name
        """
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        
        # Extract custom model config
        custom_config = model_config.get("custom_model_config", {})
        self.node_features = custom_config.get("node_features", 12)
        self.hidden_dim = custom_config.get("hidden_dim", 128)
        self.num_layers = custom_config.get("num_layers", 4)
        self.num_heads = custom_config.get("num_heads", 4)
        self.dropout = custom_config.get("dropout", 0.1)
        
        # Build GNN layers
        self.gnn_layers = nn.ModuleList()
        
        # First layer
        self.gnn_layers.append(
            GATv2Conv(
                in_channels=self.node_features,
                out_channels=self.hidden_dim // self.num_heads,
                heads=self.num_heads,
                concat=True,
                dropout=self.dropout,
            )
        )
        
        # Hidden layers
        for _ in range(self.num_layers - 1):
            self.gnn_layers.append(
                GATv2Conv(
                    in_channels=self.hidden_dim,
                    out_channels=self.hidden_dim // self.num_heads,
                    heads=self.num_heads,
                    concat=True,
                    dropout=self.dropout,
                )
            )
        
        # Q-value head: graph embedding -> Q-values
        self.q_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, num_outputs),
        )
        
        # Value head for dueling DQN (optional)
        self.value_head = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1),
        )
        
        # Store last computed values
        self._last_value = None
    
    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, torch.Tensor],
        state: List[torch.Tensor],
        seq_lens: torch.Tensor,
    ) -> tuple:
        """
        Forward pass through the model.
        
        Args:
            input_dict: Dictionary with 'obs' containing observations
            state: RNN state (not used for this model)
            seq_lens: Sequence lengths (not used for this model)
            
        Returns:
            Tuple of (Q-values, state)
        """
        obs = input_dict["obs"]
        
        # Process graph observation
        nodes, edge_index, batch_idx = self._process_graph_observation(obs)
        
        # Forward through GNN
        graph_embedding = self._gnn_forward(nodes, edge_index, batch_idx)
        
        # Compute Q-values using dueling architecture
        value = self.value_head(graph_embedding)
        advantages = self.q_head(graph_embedding)
        
        # Dueling DQN: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=-1, keepdim=True))
        
        # Apply action mask if available and shapes match
        # Following Ray RLlib's parametric actions pattern:
        # Convert binary mask to logit mask and add to Q-values
        if "action_mask" in obs:
            action_mask = obs["action_mask"]
            # Only apply mask if shapes are compatible
            if action_mask.shape[-1] == q_values.shape[-1]:
                # Convert binary mask (0/1) to log-space mask (-inf/0)
                # torch.log will give -inf for 0s and 0 for 1s
                # Clamp to -1e10 to avoid actual -inf which can cause numerical issues
                inf_mask = torch.clamp(torch.log(action_mask.float() + 1e-8), -1e10, 0.0)
                # Add the mask - this sets illegal actions to very negative values
                q_values = q_values + inf_mask
        
        # Store value for value_function()
        self._last_value = value
        
        return q_values, state
    
    @override(TorchModelV2)
    def value_function(self) -> torch.Tensor:
        """
        Return the value function estimate for the most recent forward pass.
        """
        if self._last_value is None:
            return torch.tensor(0.0)
        return self._last_value.squeeze(-1)
    
    def _process_graph_observation(self, obs_dict):
        """
        Process graph observation into format suitable for GNN.
        
        Args:
            obs_dict: Dictionary with keys:
                - 'node_features': [batch_size?, max_nodes, feature_dim]
                - 'adjacency_matrix': [batch_size?, max_nodes, max_nodes]
                - 'node_mask': [batch_size?, max_nodes]
            
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
                # Get edges for this graph
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
            Graph-level embedding [batch_size, hidden_dim * 2]
        """
        # Apply GNN layers
        for i, layer in enumerate(self.gnn_layers):
            x = layer(x, edge_index)
            x = F.relu(x)
            if i < len(self.gnn_layers) - 1:  # Don't apply dropout after last layer
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling to get graph-level representation
        if batch is None:
            # Single graph - use simple pooling
            if x.shape[0] == 0:
                # Empty graph case
                return torch.zeros((1, self.hidden_dim * 2), device=x.device)
            graph_mean = torch.mean(x, dim=0, keepdim=True)
            graph_max = torch.max(x, dim=0, keepdim=True)[0]
        else:
            # Batch of graphs - use PyG's batch-aware pooling
            # But first check for empty graphs
            if x.shape[0] == 0:
                batch_size = 1 if batch is None else (batch.max().item() + 1)
                return torch.zeros((batch_size, self.hidden_dim * 2), device=x.device)
            
            graph_mean = global_mean_pool(x, batch)
            graph_max = global_max_pool(x, batch)
        
        # Concatenate mean and max pooling
        graph_embedding = torch.cat([graph_mean, graph_max], dim=-1)
        
        return graph_embedding
