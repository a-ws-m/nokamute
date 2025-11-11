"""
Heterogeneous Graph Neural Network model with fixed action space and policy head for Hive.

This model uses a heterogeneous graph representation with:
- Node types: in_play (pieces on board), out_of_play (pieces in hand), destination (empty spaces)
- Edge types: neighbour (adjacency), move (legal moves)

The model processes the heterogeneous graph through multiple GNN layers and outputs action logits
by reading move edge features (for edges representing current player's legal moves).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from action_space import get_action_space_size, string_to_action
from torch_geometric.nn import GATv2Conv, HeteroConv, Linear


class HiveGNNPolicyHetero(nn.Module):
    """
    Heterogeneous Graph Neural Network for Hive with fixed action space and policy head.

    Architecture:
    - Input: Heterogeneous graph with 3 node types and 2 edge types
    - Multiple GAT layers for message passing on neighbour edges
    - Move edge features are updated through the layers
    - Policy head: reads move edge features to generate action logits
    - Value head: pools graph to predict position evaluation

    Output:
    - Policy logits: [batch_size, num_actions] unnormalized logits for each action
    - Value: [batch_size, 1] position evaluation on absolute scale
            (+1 = White winning, -1 = Black winning, 0 = neutral)
    """

    def __init__(
        self,
        node_features=10,  # color (1) + bug onehot (9) for in_play/out_of_play
        # on_top (1) + padding (9) for destination
        hidden_dim=256,
        num_layers=6,
        num_heads=4,
        dropout=0.1,
    ):
        super(HiveGNNPolicyHetero, self).__init__()

        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_actions = get_action_space_size()

        # Initial embedding layers for each node type
        self.node_embedding = nn.ModuleDict(
            {
                "in_play": Linear(node_features, hidden_dim),
                "out_of_play": Linear(node_features, hidden_dim),
                "destination": Linear(node_features, hidden_dim),
            }
        )

        # Edge embedding for move edges (from binary feature to hidden_dim)
        self.move_edge_embedding = Linear(1, hidden_dim)

        # Heterogeneous GAT layers
        self.hetero_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleDict()

        for layer_idx in range(num_layers):
            # Define convolutions for each edge type
            conv_dict = {}

            # Neighbour edges (bidirectional between different node types)
            neighbour_conv_configs = [
                ("in_play", "neighbour", "in_play"),
                ("in_play", "neighbour", "destination"),
                ("destination", "neighbour", "in_play"),
                ("destination", "neighbour", "destination"),
            ]

            for src_type, edge_type, dst_type in neighbour_conv_configs:
                conv_dict[(src_type, edge_type, dst_type)] = GATv2Conv(
                    (-1, -1),
                    hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    add_self_loops=False,
                    edge_dim=None,  # Neighbour edges don't have meaningful features
                )

            # Move edges with edge features
            move_conv_configs = [
                ("in_play", "move", "destination"),
                ("out_of_play", "move", "destination"),
            ]

            for src_type, edge_type, dst_type in move_conv_configs:
                conv_dict[(src_type, edge_type, dst_type)] = GATv2Conv(
                    (-1, -1),
                    hidden_dim // num_heads,
                    heads=num_heads,
                    concat=True,
                    add_self_loops=False,
                    edge_dim=hidden_dim,  # Move edges have learned features
                )

            self.hetero_convs.append(HeteroConv(conv_dict, aggr="sum"))

            # Batch normalization for each node type
            self.batch_norms[f"layer_{layer_idx}"] = nn.ModuleDict(
                {
                    "in_play": nn.BatchNorm1d(hidden_dim),
                    "out_of_play": nn.BatchNorm1d(hidden_dim),
                    "destination": nn.BatchNorm1d(hidden_dim),
                }
            )

        # Policy head: MLP to transform move edge features to action logits
        # Each move edge feature will become a single logit
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),  # Single logit per edge
        )

        # Value head - predicts position evaluation (auxiliary task)
        # Uses global pooling over all node types
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),  # 3 node types pooled together
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),  # Clip to [-1, 1]
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, move_to_action_indices):
        """
        Forward pass.

        Args:
            x_dict: Dictionary of node features for each node type
                    {'in_play': [num_in_play, node_features],
                     'out_of_play': [num_out_of_play, node_features],
                     'destination': [num_dest, node_features]}
            edge_index_dict: Dictionary of edge indices for each edge type
                    {('src_type', 'edge_type', 'dst_type'): [2, num_edges]}
            edge_attr_dict: Dictionary of edge attributes for each edge type
                    {('src_type', 'edge_type', 'dst_type'): [num_edges, 1]}
            move_to_action_indices: Tensor mapping move edge indices to action space indices
                    [num_legal_moves] where each entry is the action space index

        Returns:
            policy_logits: Action logits [1, num_actions]
                          Legal actions have computed logits, others have -inf
            value: Position evaluation in [-1, 1] on ABSOLUTE scale
                   +1 = White winning, -1 = Black winning, 0 = neutral
        """
        # Initial embedding for each node type
        x_dict_embedded = {}
        for node_type in x_dict.keys():
            x_dict_embedded[node_type] = F.relu(
                self.node_embedding[node_type](x_dict[node_type])
            )

        # Prepare edge attributes with embedded move edge features
        edge_attr_embedded = {}
        for edge_type_tuple, edge_attr in edge_attr_dict.items():
            if "move" in edge_type_tuple:
                # Embed move edge features from binary to hidden_dim
                edge_attr_embedded[edge_type_tuple] = self.move_edge_embedding(
                    edge_attr
                )
            # Neighbour edges don't have meaningful features, so we skip them
            # GATv2Conv will work without edge_attr for neighbour edges

        # Heterogeneous GAT layers with residual connections
        x_dict_current = x_dict_embedded
        for layer_idx, hetero_conv in enumerate(self.hetero_convs):
            # Apply heterogeneous convolution
            x_dict_new = hetero_conv(
                x_dict_current, edge_index_dict, edge_attr_dict=edge_attr_embedded
            )

            # Apply activation, dropout, residual connection, and batch norm
            bn_dict = self.batch_norms[f"layer_{layer_idx}"]
            for node_type in x_dict_current.keys():
                if node_type in x_dict_new:
                    h = x_dict_new[node_type]
                    h = F.relu(h)
                    h = F.dropout(h, p=0.1, training=self.training)

                    # Residual connection
                    if x_dict_current[node_type].shape == h.shape:
                        h = h + x_dict_current[node_type]

                    # Batch norm (skip if only 1 sample)
                    if h.shape[0] > 1:
                        h = bn_dict[node_type](h)

                    x_dict_current[node_type] = h

        # Generate policy logits from move edge features
        # Strategy: For each move edge, compute edge representation by combining
        # source and destination node features, then apply policy head MLP

        all_move_edge_features = []
        all_move_edge_attrs = []

        # Collect move edges from all edge types
        for edge_type_tuple in edge_index_dict.keys():
            if "move" in edge_type_tuple:
                src_type, _, dst_type = edge_type_tuple
                edge_idx = edge_index_dict[edge_type_tuple]

                if edge_idx.shape[1] > 0:  # Has edges of this type
                    # Get source and destination node features
                    src_features = x_dict_current[src_type][
                        edge_idx[0]
                    ]  # [num_edges, hidden_dim]
                    dst_features = x_dict_current[dst_type][
                        edge_idx[1]
                    ]  # [num_edges, hidden_dim]

                    # Combine source and destination (average)
                    edge_features = (src_features + dst_features) / 2.0
                    all_move_edge_features.append(edge_features)

                    # Keep track of which edges belong to current player
                    all_move_edge_attrs.append(edge_attr_dict[edge_type_tuple])

        if all_move_edge_features:
            # Concatenate all move edge features
            all_move_features = torch.cat(
                all_move_edge_features, dim=0
            )  # [total_move_edges, hidden_dim]
            all_move_attrs = torch.cat(
                all_move_edge_attrs, dim=0
            )  # [total_move_edges, 1]

            # Apply policy head to get logit for each move edge
            all_move_logits = self.policy_head(all_move_features).squeeze(
                -1
            )  # [total_move_edges]

            # Filter to current player's legal moves (edge_attr == 1.0)
            current_player_mask = all_move_attrs.squeeze(-1) == 1.0
            legal_move_logits = all_move_logits[current_player_mask]
        else:
            # No move edges at all (should not happen in a valid position)
            legal_move_logits = torch.empty(
                0, device=list(x_dict_current.values())[0].device
            )

        # Initialize action logits with -inf (illegal actions)
        device = list(x_dict_current.values())[0].device
        action_logits = torch.full((self.num_actions,), float("-inf"), device=device)

        # Map legal move logits to action space using move_to_action_indices
        # Filter out any indices that are -1 (moves not in action space)
        if len(move_to_action_indices) > 0 and len(legal_move_logits) > 0:
            valid_mask = move_to_action_indices >= 0
            valid_indices = move_to_action_indices[valid_mask]
            valid_logits = legal_move_logits[valid_mask]
            if len(valid_indices) > 0:
                action_logits[valid_indices] = valid_logits

        # Add batch dimension
        action_logits = action_logits.unsqueeze(0)  # [1, num_actions]

        # Compute value through global pooling over all node types
        pooled_features = []
        for node_type in ["in_play", "out_of_play", "destination"]:
            if node_type in x_dict_current and x_dict_current[node_type].shape[0] > 0:
                # Mean pool over nodes of this type
                pooled = torch.mean(x_dict_current[node_type], dim=0, keepdim=True)
            else:
                # No nodes of this type - use zeros
                pooled = torch.zeros(1, self.hidden_dim, device=device)
            pooled_features.append(pooled)

        # Concatenate pooled features from all node types
        graph_embedding = torch.cat(pooled_features, dim=-1)  # [1, hidden_dim * 3]
        value = self.value_head(graph_embedding)  # [1, 1]

        return action_logits, value

    def predict_action_probs(
        self,
        x_dict,
        edge_index_dict,
        edge_attr_dict,
        move_to_action_indices,
        temperature=1.0,
    ):
        """
        Predict action probabilities with temperature scaling.

        Args:
            x_dict, edge_index_dict, edge_attr_dict, move_to_action_indices: Same as forward()
            temperature: Temperature for softmax (higher = more uniform)

        Returns:
            probs: Action probabilities [1, num_actions]
                   Sums to 1 over legal actions, 0 for illegal actions
        """
        policy_logits, _ = self.forward(
            x_dict, edge_index_dict, edge_attr_dict, move_to_action_indices
        )

        # Apply temperature
        if temperature != 1.0:
            policy_logits = policy_logits / temperature

        # Softmax over logits (illegal actions have -inf logits, so prob = 0)
        probs = F.softmax(policy_logits, dim=-1)

        return probs

    def predict_value(
        self, x_dict, edge_index_dict, edge_attr_dict, move_to_action_indices
    ):
        """
        Predict only the position value.
        """
        _, value = self.forward(
            x_dict, edge_index_dict, edge_attr_dict, move_to_action_indices
        )
        return value

    def select_action(
        self,
        x_dict,
        edge_index_dict,
        edge_attr_dict,
        move_to_action_indices,
        temperature=1.0,
        deterministic=False,
    ):
        """
        Select an action given a board state.

        Args:
            x_dict, edge_index_dict, edge_attr_dict, move_to_action_indices: Same as forward()
            temperature: Temperature for sampling (higher = more exploration)
            deterministic: If True, select argmax instead of sampling

        Returns:
            action_idx: Selected action index
            action_prob: Probability of selected action
        """
        with torch.no_grad():
            probs = self.predict_action_probs(
                x_dict,
                edge_index_dict,
                edge_attr_dict,
                move_to_action_indices,
                temperature,
            )
            probs = probs.squeeze(0)  # [num_actions]

            if deterministic:
                action_idx = torch.argmax(probs).item()
            else:
                # Sample from the distribution
                action_idx = torch.multinomial(probs, 1).item()

            action_prob = probs[action_idx].item()

        return action_idx, action_prob


def create_policy_model(config=None):
    """
    Create a HiveGNNPolicyHetero model with default or custom configuration.

    Args:
        config: Dictionary with model configuration (optional)

    Returns:
        HiveGNNPolicyHetero model
    """
    if config is None:
        config = {}

    return HiveGNNPolicyHetero(
        node_features=config.get("node_features", 10),
        hidden_dim=config.get("hidden_dim", 256),
        num_layers=config.get("num_layers", 6),
        num_heads=config.get("num_heads", 4),
        dropout=config.get("dropout", 0.1),
    )


if __name__ == "__main__":
    # Test the model
    print("Testing HiveGNNPolicyHetero model...")
    import nokamute

    model = create_policy_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Action space size: {model.num_actions}")

    # Create a real board and get its graph representation
    board = nokamute.Board()

    # Make a few moves to have a non-trivial graph
    for _ in range(3):
        moves = board.legal_moves()
        if moves:
            board.apply(moves[0])

    # Get heterogeneous graph representation
    graph_dict = board.to_graph()

    # Convert to tensors
    x_dict = {
        "in_play": torch.tensor(
            graph_dict["node_features"]["in_play"], dtype=torch.float32
        ),
        "out_of_play": torch.tensor(
            graph_dict["node_features"]["out_of_play"], dtype=torch.float32
        ),
        "destination": torch.tensor(
            graph_dict["node_features"]["destination"], dtype=torch.float32
        ),
    }

    edge_index_dict = {}
    for src_type in ["in_play", "out_of_play", "destination"]:
        for edge_type in ["neighbour", "move"]:
            for dst_type in ["in_play", "out_of_play", "destination"]:
                key = (src_type, edge_type, dst_type)
                # Check if this edge type exists in the graph
                if (
                    edge_type in graph_dict["edge_index"]
                    and len(graph_dict["edge_index"][edge_type][0]) > 0
                ):
                    # We need to filter edges by source and destination type
                    # For now, include all - the actual filtering happens in the Rust code
                    pass

    # Simplified: just use the edge indices as provided
    edge_index_dict = {
        ("in_play", "neighbour", "in_play"): torch.tensor([[], []], dtype=torch.long),
        ("in_play", "neighbour", "destination"): torch.tensor(
            [[], []], dtype=torch.long
        ),
        ("destination", "neighbour", "in_play"): torch.tensor(
            [[], []], dtype=torch.long
        ),
        ("destination", "neighbour", "destination"): torch.tensor(
            [[], []], dtype=torch.long
        ),
        ("in_play", "move", "destination"): torch.tensor([[], []], dtype=torch.long),
        ("out_of_play", "move", "destination"): torch.tensor(
            [[], []], dtype=torch.long
        ),
    }

    edge_attr_dict = {
        "neighbour": torch.tensor(
            graph_dict["edge_attr"]["neighbour"], dtype=torch.float32
        ),
        "move": torch.tensor(graph_dict["edge_attr"]["move"], dtype=torch.float32),
    }

    # Get move to action mapping
    move_to_action_list = graph_dict["move_to_action"]
    move_to_action_indices = torch.tensor(
        [string_to_action(move_str) for move_str in move_to_action_list],
        dtype=torch.long,
    )

    print(f"\nGraph structure:")
    print(f"  in_play nodes: {x_dict['in_play'].shape[0]}")
    print(f"  out_of_play nodes: {x_dict['out_of_play'].shape[0]}")
    print(f"  destination nodes: {x_dict['destination'].shape[0]}")
    print(f"  move edges: {edge_attr_dict['move'].shape[0]}")
    print(f"  legal moves: {len(move_to_action_list)}")

    # TODO: Complete implementation - need to properly parse the heterogeneous edge structure
    print("\n⚠ Note: Full model test requires completing the edge index parsing")
    print("The model architecture is ready, but we need to properly map the Rust")
    print("heterogeneous graph edges to PyTorch Geometric format.")
