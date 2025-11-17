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
    Heterogeneous Graph Neural Network for Hive with separate policy and value networks.

    Architecture:
    - Input: Heterogeneous graph with 3 node types and 2 edge types
    - Multiple GAT layers for message passing on neighbour edges
    - Move edge features are updated through the layers
    - Policy head: reads move edge features to generate action logits (ONE logit per legal move)
    - Value head: pools entire graph to predict position evaluation (scalar output)

    Output:
    - Policy logits: [batch_size, num_actions] unnormalized logits for each action
                     OR [num_legal_moves] for single position
    - Value: [batch_size, 1] position evaluation on ABSOLUTE scale
            (+1 = White winning, -1 = Black winning, 0 = neutral)
            Value is always from White's perspective, regardless of current player
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
            # The ToUndirected transform (applied in hetero_graph_utils.py) adds reverse edges
            # with 'rev_' prefix, so we need to define convolutions for both directions.
            # This ensures message passing works bidirectionally for all edge types.
            conv_dict = {}

            # Neighbour edges (all possible combinations after ToUndirected)
            # Original: in_play <-> in_play, in_play <-> destination
            # After ToUndirected: both directions are explicit
            neighbour_conv_configs = [
                ("in_play", "neighbour", "in_play"),
                ("in_play", "neighbour", "destination"),
                ("destination", "neighbour", "in_play"),
                ("destination", "neighbour", "destination"),
                # Reverse edges added by ToUndirected (with 'rev_' prefix)
                ("in_play", "rev_neighbour", "in_play"),
                ("destination", "rev_neighbour", "in_play"),
                ("in_play", "rev_neighbour", "destination"),
                ("destination", "rev_neighbour", "destination"),
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
            # Original: in_play -> destination, out_of_play -> destination
            # After ToUndirected: reverse edges are added
            move_conv_configs = [
                ("in_play", "move", "destination"),
                ("out_of_play", "move", "destination"),
                # Reverse edges added by ToUndirected
                ("destination", "rev_move", "in_play"),
                ("destination", "rev_move", "out_of_play"),
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

        # Action value head: MLP to transform move edge features to action values
        # Each move edge feature becomes a single action value in [-1, 1]
        # These action values are used for both:
        #   - Policy: softmax(action_values) -> probability distribution
        #   - Value: max(action_values) -> best available move value
        self.action_value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),  # Single value per move edge
            nn.Tanh(),  # Output in [-1, 1] on ABSOLUTE scale
        )

    def compute_move_edge_values(self, x_dict, edge_index_dict, edge_attr_dict):
        """
        Compute action values for all move edges.

        This extracts the forward pass logic for computing move edge values into a separate method.
        The heterogeneous graph is undirected (via ToUndirected transform), so there are 2x as many
        move edges as legal moves. Each legal move has both forward and reverse edges with current_player=1.

        Args:
            x_dict: Dictionary of processed node features (after GNN layers)
            edge_index_dict: Dictionary of edge indices
            edge_attr_dict: Dictionary of edge attributes (original, not embedded)

        Returns:
            action_values: Tensor [num_actions] with values in [-1, 1] on ABSOLUTE scale
                          (+1 = White winning, -1 = Black winning, -inf for illegal actions)
            all_move_values: Tensor [total_move_edges] with raw values for all move edges
            current_player_mask: Tensor [total_move_edges] boolean mask for current player's moves
        """
        device = list(x_dict.values())[0].device

        all_move_edge_features = []
        all_move_edge_attrs = []

        # Collect move edges from all edge types (including reverse edges from ToUndirected)
        for edge_type_tuple in edge_index_dict.keys():
            src_type, edge_type, dst_type = edge_type_tuple
            if "move" in edge_type or "rev_move" in edge_type:
                edge_idx = edge_index_dict[edge_type_tuple]

                if edge_idx.shape[1] > 0:  # Has edges of this type
                    # Get source and destination node features
                    src_features = x_dict[src_type][
                        edge_idx[0]
                    ]  # [num_edges, hidden_dim]
                    dst_features = x_dict[dst_type][
                        edge_idx[1]
                    ]  # [num_edges, hidden_dim]

                    # Combine source and destination (average)
                    edge_features = (src_features + dst_features) / 2.0
                    all_move_edge_features.append(edge_features)

                    # Keep track of which edges belong to current player
                    all_move_edge_attrs.append(edge_attr_dict[edge_type_tuple])

        if not all_move_edge_features:
            # No move edges at all - this is an error (should have at least one legal move)
            raise ValueError(
                "No move edges found in the graph. This indicates either a terminal position "
                "or an error in graph construction."
            )

        # Concatenate all move edge features
        all_move_features = torch.cat(
            all_move_edge_features, dim=0
        )  # [total_move_edges, hidden_dim]
        all_move_attrs = torch.cat(all_move_edge_attrs, dim=0)  # [total_move_edges, 1]

        # Apply action_value_head to get value for each move edge
        all_move_values = self.action_value_head(all_move_features).squeeze(
            -1
        )  # [total_move_edges]

        # Mask: edges with attr == 1.0 are current player's legal moves
        current_player_mask = all_move_attrs.squeeze(-1) == 1.0  # [total_move_edges]

        # Check that we have at least one current player move
        if not current_player_mask.any():
            raise ValueError(
                "No current player move edges found (all edge_attr != 1.0). "
                "This indicates an error in graph construction."
            )

        # Set non-current-player moves to -inf
        masked_values = torch.where(
            current_player_mask,
            all_move_values,
            torch.tensor(
                float("-inf"),
                device=all_move_values.device,
                dtype=all_move_values.dtype,
            ),
        )  # [total_move_edges]

        return masked_values, all_move_values, current_player_mask

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, move_to_action_indices):
        """
        Forward pass to compute action values, then derive policy and value.

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
                    [total_move_edges] where each entry is the action space index
                    Note: Due to ToUndirected, there are 2x as many edges as legal moves

        Returns:
            action_values: Action values [1, num_actions] in ABSOLUTE scale
                          +1 = White winning, -1 = Black winning, -inf for illegal actions
            white_value: Best value for White (max of action values) [1, 1] or [batch_size, 1]
            black_value: Best value for Black (min of action values) [1, 1] or [batch_size, 1]
            white_action_idx: Action index corresponding to white_value [1] or [batch_size]
            black_action_idx: Action index corresponding to black_value [1] or [batch_size]
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
            # Check if this is a move edge (original or reverse)
            if (
                "move" in edge_type_tuple[1]
            ):  # edge_type_tuple[1] is the edge type string
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

        # Detect batch size early
        batch_size_detected = 1
        batch_dict = {}
        for node_type in x_dict.keys():
            # Check the original input x_dict for batch information
            if hasattr(x_dict[node_type], "batch"):
                batch_dict[node_type] = x_dict[node_type].batch
                # Only compute max if there are nodes of this type
                if x_dict[node_type].batch.numel() > 0:
                    batch_size_detected = max(
                        batch_size_detected, int(batch_dict[node_type].max().item()) + 1
                    )

        is_batched = batch_size_detected > 1

        # Compute action values from move edges (works for both batched and single)
        masked_values, all_move_values, current_player_mask = (
            self.compute_move_edge_values(
                x_dict_current, edge_index_dict, edge_attr_dict
            )
        )

        # Initialize action values with -inf (illegal actions)
        device = list(x_dict_current.values())[0].device
        # Infer dtype from computed values to handle mixed precision correctly
        dtype = masked_values.dtype if len(masked_values) > 0 else torch.float32
        action_values = torch.full(
            (self.num_actions,), float("-inf"), device=device, dtype=dtype
        )

        # Map move edge values to action space using move_to_action_indices
        # move_to_action_indices maps each move edge to its action index
        # Edges with -1 index are not in the action space (shouldn't happen)
        # NOTE: For batched graphs, move_to_action_indices may not align with masked_values
        # due to concatenation during batching. In that case, skip action space mapping
        # and only compute white_value/black_value directly from masked_values.
        use_action_space = (
            len(move_to_action_indices) > 0
            and len(masked_values) > 0
            and len(move_to_action_indices) == len(masked_values)
        )

        if use_action_space:
            # Replace -1 indices with a dummy index (0) and mask them out
            # This avoids dynamic control flow
            valid_mask = move_to_action_indices >= 0
            safe_indices = torch.where(
                valid_mask,
                move_to_action_indices,
                torch.zeros_like(move_to_action_indices),
            )

            # Mask values for invalid indices to -inf (won't affect scatter max)
            safe_values = torch.where(
                valid_mask,
                masked_values,
                torch.tensor(
                    float("-inf"),
                    device=masked_values.device,
                    dtype=masked_values.dtype,
                ),
            )

            # Use scatter_reduce with 'amax' to handle potential duplicates
            # This is compile-friendly and avoids loops
            action_values.scatter_reduce_(
                0, safe_indices, safe_values, reduce="amax", include_self=True
            )

        # Use the batch_size and is_batched detected earlier
        batch_size = batch_size_detected

        if not is_batched:
            # Single graph - add batch dimension
            action_values = action_values.unsqueeze(0)  # [1, num_actions]

            # Compute value assuming optimal play
            # Action values are absolute: +1 = White winning, -1 = Black winning
            # White wants to maximize (pick best move) -> white_value = max
            # Black wants to minimize (pick best move) -> black_value = min

            if use_action_space:
                # Use action space values
                legal_mask = torch.isfinite(action_values)
                if legal_mask.any():
                    legal_values = action_values[legal_mask]

                    # White value: best outcome for White (maximum)
                    white_value = legal_values.max().unsqueeze(0).unsqueeze(0)  # [1, 1]
                    # Find index of maximum among legal actions only
                    white_masked = torch.where(
                        legal_mask,
                        action_values,
                        torch.tensor(float("-inf"), device=device, dtype=dtype),
                    )
                    white_action_idx = torch.argmax(white_masked).unsqueeze(0)  # [1]

                    # Black value: best outcome for Black (minimum)
                    black_value = legal_values.min().unsqueeze(0).unsqueeze(0)  # [1, 1]
                    # Find index of minimum among legal actions only
                    black_masked = torch.where(
                        legal_mask,
                        action_values,
                        torch.tensor(float("inf"), device=device, dtype=dtype),
                    )
                    black_action_idx = torch.argmin(black_masked).unsqueeze(0)  # [1]
                else:
                    # No legal moves - neutral value
                    white_value = torch.zeros(1, 1, device=device, dtype=dtype)
                    black_value = torch.zeros(1, 1, device=device, dtype=dtype)
                    white_action_idx = torch.zeros(1, device=device, dtype=torch.long)
                    black_action_idx = torch.zeros(1, device=device, dtype=torch.long)
            else:
                # No action space mapping - compute directly from masked_values
                # This can happen if move_to_action_indices doesn't align (e.g., batching issue)
                if len(masked_values) > 0:
                    legal_values = masked_values[torch.isfinite(masked_values)]
                    if len(legal_values) > 0:
                        white_value = legal_values.max().unsqueeze(0).unsqueeze(0)
                        black_value = legal_values.min().unsqueeze(0).unsqueeze(0)
                    else:
                        white_value = torch.zeros(1, 1, device=device, dtype=dtype)
                        black_value = torch.zeros(1, 1, device=device, dtype=dtype)
                else:
                    white_value = torch.zeros(1, 1, device=device, dtype=dtype)
                    black_value = torch.zeros(1, 1, device=device, dtype=dtype)
                # No action indices available
                white_action_idx = torch.zeros(1, device=device, dtype=torch.long)
                black_action_idx = torch.zeros(1, device=device, dtype=torch.long)
        else:
            # Batched graphs - need to compute per-graph values
            # The key insight: move edges have a batch assignment we can use
            # We need to track which graph each move edge belongs to

            # For batched case, we need to compute value per graph from the move edge features
            # masked_values contains the action values for all move edges across all graphs
            # We'll take the max per graph using batch assignments

            if len(masked_values) > 0:
                # Get batch assignment for move edges
                # IMPORTANT: Must collect in the same order as compute_move_edge_values
                # which iterates over edge_index_dict.keys()
                move_edge_batch = []

                # Iterate in the same order as compute_move_edge_values
                for edge_type_tuple in edge_index_dict.keys():
                    src_type, edge_type, dst_type = edge_type_tuple
                    if "move" in edge_type or "rev_move" in edge_type:
                        edge_idx = edge_index_dict[edge_type_tuple]

                        if edge_idx.shape[1] > 0:  # Has edges of this type
                            # Get batch assignment from source nodes
                            src_indices = edge_idx[0]
                            if src_type in batch_dict:
                                move_edge_batch.append(
                                    batch_dict[src_type][src_indices]
                                )

                if move_edge_batch:
                    # Concatenate batch assignments for all move edges
                    all_move_batch = torch.cat(move_edge_batch)  # [num_move_edges]

                    # Compute white_value (max) and black_value (min) per graph
                    # Action values are absolute: +1 = White winning, -1 = Black winning
                    # IMPORTANT: Only consider finite values (current player's moves)
                    # masked_values has -inf for opponent moves, which should be excluded
                    from torch_geometric.utils import scatter

                    # Filter out -inf values (opponent moves) before scatter
                    finite_mask = torch.isfinite(masked_values)
                    if finite_mask.any():
                        finite_values = masked_values[finite_mask]
                        finite_batch = all_move_batch[finite_mask]

                        white_value = scatter(
                            finite_values.unsqueeze(-1),
                            finite_batch,
                            dim=0,
                            reduce="max",
                            dim_size=batch_size,
                        )  # [batch_size, 1]

                        black_value = scatter(
                            finite_values.unsqueeze(-1),
                            finite_batch,
                            dim=0,
                            reduce="min",
                            dim_size=batch_size,
                        )  # [batch_size, 1]
                    else:
                        # No finite values - all opponent moves (shouldn't happen)
                        white_value = torch.zeros(
                            batch_size, 1, device=device, dtype=dtype
                        )
                        black_value = torch.zeros(
                            batch_size, 1, device=device, dtype=dtype
                        )

                    # Find action indices corresponding to best values
                    # For batched case, we need to find argmax/argmin per graph
                    # This is more complex - for now return placeholder (-1 indicates not computed)
                    white_action_idx = torch.full(
                        (batch_size,), -1, device=device, dtype=torch.long
                    )
                    black_action_idx = torch.full(
                        (batch_size,), -1, device=device, dtype=torch.long
                    )
                else:
                    # No move edges - return zeros (should not happen in valid positions)
                    white_value = torch.zeros(
                        batch_size, 1, device=device, dtype=dtype, requires_grad=True
                    )
                    black_value = torch.zeros(
                        batch_size, 1, device=device, dtype=dtype, requires_grad=True
                    )
                    white_action_idx = torch.zeros(
                        batch_size, device=device, dtype=torch.long
                    )
                    black_action_idx = torch.zeros(
                        batch_size, device=device, dtype=torch.long
                    )
            else:
                # No legal moves at all
                white_value = torch.zeros(
                    batch_size, 1, device=device, dtype=dtype, requires_grad=True
                )
                black_value = torch.zeros(
                    batch_size, 1, device=device, dtype=dtype, requires_grad=True
                )
                white_action_idx = torch.zeros(
                    batch_size, device=device, dtype=torch.long
                )
                black_action_idx = torch.zeros(
                    batch_size, device=device, dtype=torch.long
                )

            # For batched training, action_logits is not used (return placeholder)
            action_values = action_values.unsqueeze(0)  # [1, num_actions]

        return (
            action_values,
            white_value,
            black_value,
            white_action_idx,
            black_action_idx,
        )

    def predict_action_probs(
        self,
        x_dict,
        edge_index_dict,
        edge_attr_dict,
        move_to_action_indices,
        temperature=1.0,
        current_player=None,
    ):
        """
        Predict action probabilities with temperature scaling.

        Action values from the model are on an ABSOLUTE scale (+1 = White winning, -1 = Black winning).
        For move selection, we need to convert to player-relative utilities:
        - White wants to maximize absolute value (keep as-is)
        - Black wants to minimize absolute value (negate before softmax)

        Args:
            x_dict, edge_index_dict, edge_attr_dict, move_to_action_indices: Same as forward()
            temperature: Temperature for softmax (higher = more uniform)
            current_player: "White" or "Black" - needed to convert absolute values to utilities

        Returns:
            probs: Action probabilities [1, num_actions]
                   Sums to 1 over legal actions, 0 for illegal actions
        """
        action_values, _, _, _, _ = self.forward(
            x_dict, edge_index_dict, edge_attr_dict, move_to_action_indices
        )

        # Convert absolute action values to player-relative utilities
        # Action values are on absolute scale: +1 = White winning, -1 = Black winning
        # For move selection:
        #   White wants to maximize (keep as-is)
        #   Black wants to minimize (negate)
        if current_player == "Black":
            # Negate for Black player
            # Preserve -inf for illegal moves (they stay -inf after negation)
            action_values = torch.where(
                torch.isfinite(action_values),
                -action_values,
                action_values,  # Keep -inf as -inf
            )

        # Apply temperature
        if temperature != 1.0:
            action_values = action_values / temperature

        # Softmax over action values (illegal actions have -inf, so prob = 0)
        probs = F.softmax(action_values, dim=-1)

        return probs

    def predict_value(
        self,
        x_dict,
        edge_index_dict,
        edge_attr_dict,
        move_to_action_indices,
        current_player=None,
    ):
        """
        Predict position value from current player's perspective.

        Args:
            current_player: "White" or "Black" - if None, returns white_value

        Returns:
            value: [batch_size, 1] - best achievable value for current player on ABSOLUTE scale
                   Returns white_value (max) if current_player="White"
                   Returns black_value (min) if current_player="Black"
        """
        _, white_value, black_value, _, _ = self.forward(
            x_dict, edge_index_dict, edge_attr_dict, move_to_action_indices
        )

        if current_player == "Black":
            return black_value
        else:
            # Default to White or when current_player is None
            return white_value

    def select_action(
        self,
        x_dict,
        edge_index_dict,
        edge_attr_dict,
        move_to_action_indices,
        deterministic=False,
    ):
        """
        Select an action given a board state using greedy selection.

        Args:
            x_dict, edge_index_dict, edge_attr_dict, move_to_action_indices: Same as forward()
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
            )
            probs = probs.squeeze(0)  # [num_actions]

            if deterministic:
                action_idx = torch.argmax(probs).item()
            else:
                # Sample from the distribution
                action_idx = torch.multinomial(probs, 1).item()

            action_prob = probs[int(action_idx)].item()

        return action_idx, action_prob

    def predict_action_info(
        self,
        x_dict,
        edge_index_dict,
        edge_attr_dict,
        move_to_action_indices,
        current_player=None,
    ):
        """
        Compute action_values, value (best achievable for current_player), and action_idx in one pass.

        Args:
            x_dict, edge_index_dict, edge_attr_dict, move_to_action_indices: Model inputs
            current_player: "White" or "Black"

        Returns:
            action_values: [1, num_actions] absolute values (+1=White win, -1=Black win, -inf illegal)
            action_probs: [1, num_actions] softmax probabilities for current_player
            value: [1, 1] best achievable value for current_player
            action_idx: [1] index of best action for current_player
        """
        action_values, white_value, black_value, white_action_idx, black_action_idx = (
            self.forward(
                x_dict, edge_index_dict, edge_attr_dict, move_to_action_indices
            )
        )

        # Convert absolute action values to player-relative utilities
        if current_player == "Black":
            # Negate for Black player
            action_values = torch.where(
                torch.isfinite(action_values),
                -action_values,
                action_values,
            )
            value = black_value
            action_idx = black_action_idx
        else:
            # Default to White or when current_player is None
            value = white_value
            action_idx = white_action_idx

        action_probs = F.softmax(action_values, dim=-1)

        return action_values, action_probs, value, action_idx


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
