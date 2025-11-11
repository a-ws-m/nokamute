"""
Graph Neural Network model with fixed action space and policy head for Hive.

This model uses a fixed action space with action masking, outputting logits
for all possible actions and using a mask to select only legal moves.

Compared to the value-based model (model.py) which evaluates each legal move
sequentially, this model:
1. Evaluates the position once to get graph embeddings
2. Outputs logits for all possible actions (5985 actions)
3. Applies action mask to filter illegal moves
4. Uses softmax over masked logits to get move probabilities

This approach may be faster when there are many legal moves since it avoids
repeatedly evaluating similar positions.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from action_space import get_action_space_size
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


class HiveGNNPolicy(nn.Module):
    """
    Graph Neural Network for Hive with fixed action space and policy head.

    Architecture:
    - Input: Graph with node features (color, bug type, height, current player)
    - Multiple GAT (Graph Attention) layers for message passing
    - Global pooling (mean + max)
    - Policy head: outputs logits for all possible actions (5985)
    - Action masking: filters illegal moves
    - Value head: outputs position evaluation (for auxiliary learning)

    Output:
    - Policy logits: [batch_size, num_actions] unnormalized logits for each action
    - Value: [batch_size, 1] position evaluation on absolute scale
            (+1 = White winning, -1 = Black winning, 0 = neutral)
    """

    def __init__(
        self,
        node_features=12,  # color (1) + bug onehot (9) + height (1) + current_player (1)
        hidden_dim=128,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
    ):
        super(HiveGNNPolicy, self).__init__()

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_actions = get_action_space_size()

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

        # Policy head - outputs logits for all actions
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for mean+max pooling
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_actions),  # Output logits for each action
        )

        # Value head - predicts position evaluation (auxiliary task)
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

    def forward(self, x, edge_index, batch=None, action_mask=None):
        """
        Forward pass.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch assignment for each node [num_nodes] (optional)
            action_mask: Binary mask of legal actions [batch_size, num_actions] (optional)
                        1 = legal, 0 = illegal

        Returns:
            policy_logits: Action logits [batch_size, num_actions]
                          If action_mask is provided, illegal actions have -inf logits
            value: Position evaluation in [-1, 1] on ABSOLUTE scale
                   +1 = White winning, -1 = Black winning, 0 = neutral
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

        # Policy logits
        policy_logits = self.policy_head(graph_embedding)

        # Apply action mask if provided
        if action_mask is not None:
            # Convert binary mask (0/1) to logit mask (-inf/0)
            # Illegal actions get -inf, legal actions get 0 (no change)
            inf_mask = torch.where(
                action_mask.bool(),
                torch.zeros_like(policy_logits),
                torch.full_like(policy_logits, float("-inf")),
            )
            policy_logits = policy_logits + inf_mask

        # Value prediction
        value = self.value_head(graph_embedding)

        return policy_logits, value

    def predict_action_probs(
        self, x, edge_index, batch=None, action_mask=None, temperature=1.0
    ):
        """
        Predict action probabilities with temperature scaling.

        Args:
            x: Node features
            edge_index: Graph connectivity
            batch: Batch assignment (optional)
            action_mask: Binary mask of legal actions (optional)
            temperature: Temperature for softmax (higher = more uniform)

        Returns:
            probs: Action probabilities [batch_size, num_actions]
                   Sums to 1 over legal actions, 0 for illegal actions
        """
        policy_logits, _ = self.forward(x, edge_index, batch, action_mask)

        # Apply temperature
        if temperature != 1.0:
            policy_logits = policy_logits / temperature

        # Softmax over logits (illegal actions have -inf logits, so prob = 0)
        probs = F.softmax(policy_logits, dim=-1)

        return probs

    def predict_value(self, x, edge_index, batch=None):
        """
        Predict only the position value (for compatibility with value-based model).
        """
        _, value = self.forward(x, edge_index, batch, action_mask=None)
        return value

    def select_action(
        self, x, edge_index, action_mask=None, temperature=1.0, deterministic=False
    ):
        """
        Select an action given a board state.

        Args:
            x: Node features [num_nodes, node_features]
            edge_index: Graph connectivity [2, num_edges]
            action_mask: Binary mask of legal actions [num_actions]
            temperature: Temperature for sampling (higher = more exploration)
            deterministic: If True, select argmax instead of sampling

        Returns:
            action_idx: Selected action index
            action_prob: Probability of selected action
        """
        with torch.no_grad():
            # Ensure action_mask has batch dimension
            if action_mask is not None and action_mask.dim() == 1:
                action_mask = action_mask.unsqueeze(0)

            probs = self.predict_action_probs(
                x,
                edge_index,
                batch=None,
                action_mask=action_mask,
                temperature=temperature,
            )
            probs = probs.squeeze(0)  # Remove batch dimension

            if deterministic:
                action_idx_tensor = torch.argmax(probs)
                action_idx = action_idx_tensor.item()
            else:
                # Sample from categorical distribution
                action_idx_tensor = torch.multinomial(probs, 1)
                action_idx = action_idx_tensor.item()

            action_prob = probs[int(action_idx)].item()

        return action_idx, action_prob


def create_policy_model(config=None):
    """
    Create a HiveGNNPolicy model with default or custom configuration.

    Args:
        config: Dictionary with model configuration (optional)

    Returns:
        HiveGNNPolicy model
    """
    if config is None:
        config = {}

    return HiveGNNPolicy(
        node_features=config.get("node_features", 12),
        hidden_dim=config.get("hidden_dim", 128),
        num_layers=config.get("num_layers", 4),
        num_heads=config.get("num_heads", 4),
        dropout=config.get("dropout", 0.1),
    )


if __name__ == "__main__":
    # Test the model
    print("Testing HiveGNNPolicy model...")

    model = create_policy_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Action space size: {model.num_actions}")

    # Create dummy data with correct feature size (12)
    num_nodes = 10
    node_features = 12
    num_edges = 20

    x = torch.randn(num_nodes, node_features)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))

    # Create action mask (mark first 100 actions as legal)
    action_mask = torch.zeros(model.num_actions)
    action_mask[:100] = 1

    # Forward pass
    policy_logits, value = model(x, edge_index, action_mask=action_mask.unsqueeze(0))

    print(f"\nInput shape: {x.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Action mask shape: {action_mask.shape}")
    print(f"Output policy logits shape: {policy_logits.shape}")
    print(f"Output value shape: {value.shape}")
    print(f"Output value: {value.item():.4f}")

    # Test action probabilities
    probs = model.predict_action_probs(
        x, edge_index, action_mask=action_mask.unsqueeze(0)
    )
    print(f"\nAction probabilities shape: {probs.shape}")
    print(f"Sum of probabilities: {probs.sum().item():.6f}")
    print(f"Number of non-zero probabilities: {(probs > 0).sum().item()}")
    print(f"Max probability: {probs.max().item():.6f}")

    # Test action selection
    action_idx, action_prob = model.select_action(
        x, edge_index, action_mask=action_mask, temperature=1.0
    )
    print(f"\nSelected action: {action_idx}")
    print(f"Action probability: {action_prob:.6f}")

    # Test that illegal actions have zero probability
    print(f"\nProbability of illegal action (index 150): {probs[0, 150].item():.10f}")
    print(f"Probability of legal action (index 50): {probs[0, 50].item():.10f}")

    print("\nModel test passed!")
