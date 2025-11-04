"""
Actor-Critic models for PPO training with graph-based Hive observations.

This module provides neural network architectures for the policy (actor) and
value function (critic) using graph neural networks to process Hive board states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.modules import MaskedCategorical, ProbabilisticActor, ValueOperator


class GraphEncoder(nn.Module):
    """
    Graph encoder using GAT layers to process Hive board states.
    
    This is adapted from the existing HiveGNN model but structured to work
    with the batched graph inputs from the environment.
    """
    
    def __init__(
        self,
        node_features=11,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Initial embedding
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_channels = hidden_dim
            out_channels = hidden_dim // num_heads
            
            self.gat_layers.append(
                GATv2Conv(
                    in_channels,
                    out_channels,
                    heads=num_heads,
                    dropout=dropout,
                    concat=True,
                )
            )
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
    def forward(self, node_features, edge_index, num_nodes):
        """
        Encode graph into a fixed-size embedding.
        
        Args:
            node_features: [total_nodes, 11] node features across batch
            edge_index: [2, total_edges] edge indices across batch
            num_nodes: [batch_size] number of nodes per graph
            
        Returns:
            graph_embedding: [batch_size, hidden_dim * 2] or [hidden_dim * 2] graph-level embeddings  
            node_embeddings: [batch_size, num_nodes, hidden_dim] or [num_nodes, hidden_dim] node-level embeddings
        """
        # Check if we have a batch dimension
        if node_features.dim() == 3:
            # Batched input: [batch_size, max_nodes, node_dim]
            batch_size = node_features.shape[0]
            embeddings = []
            node_embs = []
            
            for i in range(batch_size):
                # Process each graph separately
                emb, nodes = self._forward_single_graph(
                    node_features[i],  # [max_nodes, node_dim]
                    edge_index[i],     # [2, max_edges]
                    num_nodes[i] if num_nodes.dim() > 1 else num_nodes
                )
                embeddings.append(emb)
                node_embs.append(nodes)
            
            graph_embedding = torch.stack(embeddings)  # [batch_size, hidden_dim * 2]
            node_embeddings = torch.stack(node_embs)   # [batch_size, max_nodes, hidden_dim]
            
            return graph_embedding, node_embeddings
        else:
            # Single graph: [max_nodes, node_dim]
            return self._forward_single_graph(node_features, edge_index, num_nodes)
    
    def _forward_single_graph(self, node_features, edge_index, num_nodes):
        """
        Process a single graph.
        
        Args:
            node_features: [max_nodes, node_dim]
            edge_index: [2, max_edges]
            num_nodes: [1] or scalar
            
        Returns:
            graph_embedding: [hidden_dim * 2]
            node_embeddings: [max_nodes, hidden_dim]
        """
        # Initial embedding
        x = self.node_embedding(node_features)
        x = F.relu(x)
        
        # GAT layers with residual connections
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            x_residual = x
            x = gat(x, edge_index)
            # Skip batch norm to avoid issues with single-sample batches
            x = F.relu(x)
            
            # Residual connection (skip first layer)
            if i > 0:
                x = x + x_residual
        
        # Global pooling - mean/max over nodes
        graph_mean = torch.mean(x, dim=0)  # [hidden_dim]
        graph_max, _ = torch.max(x, dim=0)  # [hidden_dim]
        graph_embedding = torch.cat([graph_mean, graph_max], dim=-1)  # [hidden_dim * 2]
        
        return graph_embedding, x


class ActorNetwork(nn.Module):
    """
    Policy network (actor) that outputs action logits.
    
    Uses a graph encoder followed by an MLP to produce logits over actions.
    Supports action masking for invalid moves.
    """
    
    def __init__(
        self,
        node_features=11,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        max_actions=500,
    ):
        super().__init__()
        
        self.encoder = GraphEncoder(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, max_actions),
        )
    
    def forward(self, node_features, edge_index, num_nodes):
        """
        Forward pass to compute action logits.
        
        Args:
            node_features: [total_nodes, 11] node features
            edge_index: [2, total_edges] edge indices
            num_nodes: [batch_size] number of nodes per graph
            
        Returns:
            logits: [batch_size, max_actions] action logits
        """
        graph_embedding, _ = self.encoder(node_features, edge_index, num_nodes)
        logits = self.policy_head(graph_embedding)
        return logits


class CriticNetwork(nn.Module):
    """
    Value network (critic) that estimates state values.
    
    Uses a graph encoder followed by an MLP to produce a scalar value estimate.
    """
    
    def __init__(
        self,
        node_features=11,
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        
        self.encoder = GraphEncoder(
            node_features=node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, node_features, edge_index, num_nodes):
        """
        Forward pass to compute state value.
        
        Args:
            node_features: [total_nodes, 11] node features
            edge_index: [2, total_edges] edge indices
            num_nodes: [batch_size] number of nodes per graph
            
        Returns:
            value: [batch_size, 1] state value estimates
        """
        graph_embedding, _ = self.encoder(node_features, edge_index, num_nodes)
        value = self.value_head(graph_embedding)
        return value


def make_ppo_models(
    node_features=11,
    hidden_dim=128,
    num_layers=4,
    num_heads=4,
    dropout=0.1,
    max_actions=500,
    device="cpu",
):
    """
    Create actor and critic modules for PPO training.
    
    Returns TensorDictModule wrapped models that can be used with TorchRL's
    PPO loss and data collectors.
    
    Args:
        node_features: Number of node features
        hidden_dim: Hidden dimension size
        num_layers: Number of GAT layers
        num_heads: Number of attention heads
        dropout: Dropout rate
        max_actions: Maximum number of actions
        device: Device to place models on
        
    Returns:
        actor: ProbabilisticActor for sampling actions
        critic: ValueOperator for value estimation
    """
    # Actor network
    actor_net = ActorNetwork(
        node_features=node_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
        max_actions=max_actions,
    ).to(device)
    
    # Wrap actor in TensorDictModule
    actor_module = TensorDictModule(
        actor_net,
        in_keys=["node_features", "edge_index", "num_nodes"],
        out_keys=["logits"],
    )
    
    # Create probabilistic actor with masked categorical distribution
    actor = ProbabilisticActor(
        module=actor_module,
        in_keys={"logits": "logits", "mask": "action_mask"},
        out_keys=["action"],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
        log_prob_key="sample_log_prob",
    )
    
    # Critic network
    critic_net = CriticNetwork(
        node_features=node_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    ).to(device)
    
    # Wrap critic in ValueOperator
    critic = ValueOperator(
        module=critic_net,
        in_keys=["node_features", "edge_index", "num_nodes"],
        out_keys=["state_value"],
    )
    
    return actor, critic


if __name__ == "__main__":
    # Test the models
    print("Testing PPO models...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Create models
    actor, critic = make_ppo_models(device=device)
    
    print(f"Actor parameters: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")
    
    # Create dummy input
    batch_size = 2
    num_nodes_per_graph = [10, 15]
    total_nodes = sum(num_nodes_per_graph)
    num_edges = 20
    
    node_features = torch.randn(total_nodes, 11, device=device)
    edge_index = torch.randint(0, total_nodes, (2, num_edges), device=device)
    num_nodes = torch.tensor(num_nodes_per_graph, dtype=torch.long, device=device)
    action_mask = torch.ones(batch_size, 500, dtype=torch.bool, device=device)
    action_mask[:, 100:] = False  # Mask actions beyond 100
    
    # Create TensorDict input
    td = TensorDict(
        {
            "node_features": node_features,
            "edge_index": edge_index,
            "num_nodes": num_nodes,
            "action_mask": action_mask,
        },
        batch_size=[batch_size],
        device=device,
    )
    
    # Test actor
    print("\nTesting actor...")
    with torch.no_grad():
        td_actor = actor(td)
    print(f"Actor output keys: {td_actor.keys()}")
    print(f"Action shape: {td_actor['action'].shape}")
    print(f"Log prob shape: {td_actor['sample_log_prob'].shape}")
    print(f"Sampled actions: {td_actor['action']}")
    
    # Test critic
    print("\nTesting critic...")
    with torch.no_grad():
        td_critic = critic(td)
    print(f"Critic output keys: {td_critic.keys()}")
    print(f"State value shape: {td_critic['state_value'].shape}")
    print(f"State values: {td_critic['state_value']}")
    
    print("\nPPO models test passed!")
