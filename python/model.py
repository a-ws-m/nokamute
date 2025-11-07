"""
Graph Neural Network model for Hive board position evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


class HiveGNN(nn.Module):
    """
    Graph Neural Network for evaluating Hive board positions.

    Architecture:
    - Input: Graph with node features (color, bug type, height, current player)
    - Multiple GAT (Graph Attention) layers for message passing
    - Global pooling (mean + max)
    - MLP head for position evaluation

    Output: Single scalar value representing position evaluation on absolute scale:
            Positive values favor White, negative values favor Black
            +1.0 = White is winning, -1.0 = Black is winning, 0.0 = neutral/drawn
    """

    def __init__(
        self,
        node_features=12,  # color (1) + bug onehot (9) + height (1) + current_player (1)
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
    ):
        super(HiveGNN, self).__init__()

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Initial embedding layer
        self.node_embedding = nn.Linear(node_features, hidden_dim)

        # GAT layers with multi-head attention
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

        # Value head - predicts position evaluation
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),  # Output in [-1, 1]
        )

        # Policy head - predicts move probabilities (for future use)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for each node [num_nodes] (optional)

        Returns:
            value: Position evaluation in [-1, 1] on ABSOLUTE scale:
                   +1 = White is winning (regardless of whose turn it is)
                   -1 = Black is winning (regardless of whose turn it is)
                    0 = neutral/drawn position
            node_embeddings: Node-level embeddings for policy head
        """
        # Initial embedding
        x = self.node_embedding(x)
        x = F.relu(x)

        # GAT layers with residual connections
        for i, (gat, bn) in enumerate(zip(self.gat_layers, self.batch_norms)):
            x_residual = x
            x = gat(x, edge_index)
            x = bn(x)
            x = F.relu(x)

            # Residual connection (skip first layer)
            if i > 0:
                x = x + x_residual

        # Store node embeddings for policy head
        node_embeddings = x

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

        # Value prediction
        value = self.value_head(graph_embedding)

        return value, node_embeddings

    def predict_value(self, x, edge_index, batch=None):
        """
        Predict only the position value.
        """
        value, _ = self.forward(x, edge_index, batch)
        return value

    def get_node_embeddings(self, x, edge_index):
        """
        Get node embeddings (useful for move policy).
        """
        _, node_embeddings = self.forward(x, edge_index, batch=None)
        return node_embeddings


def create_model(config=None):
    """
    Create a HiveGNN model with default or custom configuration.

    Args:
        config: Dictionary with model configuration (optional)

    Returns:
        HiveGNN model
    """
    if config is None:
        config = {}

    return HiveGNN(
        node_features=config.get("node_features", 12),  # color + bug_onehot + height + current_player
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 4),
        num_heads=config.get("num_heads", 4),
        dropout=config.get("dropout", 0.1),
    )


if __name__ == "__main__":
    # Test the model
    print("Testing HiveGNN model...")

    model = create_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create dummy data with correct feature size (12)
    num_nodes = 10
    node_features = 12  # color + bug_onehot (9) + height + current_player
    num_edges = 20

    x = torch.randn(num_nodes, node_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Forward pass
    value, embeddings = model(x, edge_index)

    print(f"Input shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Output value shape: {value.shape}")
    print(f"Output value: {value.item():.4f}")
    print(f"Node embeddings shape: {embeddings.shape}")

    print("Model test passed!")
