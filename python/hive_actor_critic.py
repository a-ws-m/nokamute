from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from deepsnap.hetero_gnn import forward_op
from deepsnap.hetero_graph import HeteroGraph
from torch_scatter import scatter_mean


def generate_2convs(hete: HeteroGraph, conv, hidden_size: int):
    convs1 = {}
    convs2 = {}
    # Only build convs for message types where edges exist; this avoids
    # PyG/HeteroSAGE errors when a message type exists conceptually but has
    # no edges or incompatible node counts in a particular graph.
    for message_type, edge_index in hete.edge_index.items():
        # edge_index is a tensor [2, E]; if E == 0 skip
        if edge_index.numel() == 0:
            continue
        n_type = message_type[0]
        s_type = message_type[2]
        n_feat_dim = hete.num_node_features(n_type)
        s_feat_dim = hete.num_node_features(s_type)
        convs1[message_type] = conv(n_feat_dim, hidden_size, s_feat_dim)
        convs2[message_type] = conv(hidden_size, hidden_size, hidden_size)
    return convs1, convs2


class HiveActorCritic(nn.Module):
    """Heterogeneous actor-critic for Hive using DeepSNAP.

    Trunk: two HeteroSAGE layers producing node embeddings for every node_type.

    Policy head: For each `current_move` message type, produces a per-edge
    policy vector of size `policy_dim` (default 7). All `current_move` edges across
    message types are concatenated and returned as a single tensor.

    Critic head: A pooled graph embedding followed by an MLP returning a scalar
    per graph in the batch.
    """

    def __init__(
        self,
        hetero: HeteroGraph,
        hidden_size: int = 64,
        policy_dim: int = 7,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hetero = hetero
        self.hidden_size = hidden_size
        self.policy_dim = policy_dim

        # Simple per-node-type input encoders instead of full heterogeneous
        # convs. This keeps the model stable when message types are present
        # but there are zero nodes for some types (e.g., empty board).
        self.input_encoders = nn.ModuleDict()
        for node_type in hetero.node_types:
            in_dim = hetero.num_node_features(node_type)
            # If a node type has 0 feature dims (unlikely), map to hidden with 1-dim input
            in_dim = max(in_dim, 1)
            self.input_encoders[node_type] = nn.Linear(in_dim, hidden_size)

        self.bns1 = nn.ModuleDict()
        self.bns2 = nn.ModuleDict()
        self.relus1 = nn.ModuleDict()
        self.relus2 = nn.ModuleDict()
        for node_type in hetero.node_types:
            # Use LayerNorm for small node counts; BatchNorm fails on size=1
            self.bns1[node_type] = nn.LayerNorm(hidden_size)
            # bns2 not used with our simplified trunk (only bns1 for encoder)
            self.bns2[node_type] = nn.LayerNorm(hidden_size)
            self.relus1[node_type] = nn.LeakyReLU()
            self.relus2[node_type] = nn.LeakyReLU()

        # policy head: map pairwise (elementwise product) to policy dim
        self.policy_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, policy_dim),
        )

        # critic head: pool across node types and map to scalar
        self.critic_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2 or 1),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 2 or 1, 1),
        )

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, data) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            data: a deepsnap `Batch` generated from a `HeteroGraph`

        Returns:
            policy_logits: Tensor [total_edges_current_move, policy_dim]
            critic: Tensor [batch_size, 1]
        """
        x = {}
        edge_index = data.edge_index
        # encode each node type independently
        for node_type, feat in data.node_feature.items():
            if feat is None or feat.numel() == 0:
                x[node_type] = feat
                continue
            h = self.input_encoders[node_type](feat)
            h = self.bns1[node_type](h)
            h = self.relus1[node_type](h)
            x[node_type] = h

        # Policy: iterate networkx graphs in the batch and find 'current_move' edges.
        # Map the nodes in each graph to their indices inside the aggregated
        # `data.node_feature[node_type]` arrays using `data.node_to_graph_mapping`.
        policies = []
        for g_idx, G in enumerate(data.G):
            # Precompute offsets for per-type indexing into a global node tensor
            # and a concatenated x_all used for pooling and edge lookup.
            node_types = list(x.keys())
            offsets = {}
            running = 0
            for nt in node_types:
                offsets[nt] = running
                running += x[nt].size(0)

            x_all = (
                torch.cat([x[nt] for nt in node_types], dim=0)
                if len(x) > 0
                else torch.zeros(0, self.hidden_size, device=device)
            )

            # Iterate edges in this graph
            for u, v, attrs in G.edges(data=True):
                if attrs.get("edge_type") != "current_move":
                    continue
                src_type = G.nodes[u]["node_type"]
                dst_type = G.nodes[v]["node_type"]
                # node_label holds the local id inside the original graph builder
                src_label = G.nodes[u].get("node_label")
                dst_label = G.nodes[v].get("node_label")
                src_global_index = offsets[src_type] + int(src_label)
                dst_global_index = offsets[dst_type] + int(dst_label)
                src_feat = x_all[src_global_index]
                dst_feat = x_all[dst_global_index]
                pair = src_feat * dst_feat
                logits = self.policy_mlp(self.dropout(pair))
                # ensure shape [1, policy_dim] so concatenation yields [E, policy_dim]
                policies.append(logits.unsqueeze(0))

        if len(policies) == 0:
            policy_logits = torch.zeros(
                0, self.policy_dim, device=next(self.parameters()).device
            )
        else:
            policy_logits = torch.cat(policies, dim=0)

        # Critic: pool node embeddings across all node types per graph
        # data.node_to_graph_mapping is a dict mapping node_type -> tensor
        # of size num_nodes_of_type -> graph_id
        batch_size = len(data.G)
        device = next(self.parameters()).device
        pooled = torch.zeros(batch_size, self.hidden_size, device=device)
        # Build a single concatenated node feature tensor and pool over the
        # PyG-style `batch` index map which maps each node to a graph id.
        if len(x) > 0:
            node_types = list(x.keys())
            x_all = torch.cat([x[nt] for nt in node_types], dim=0)
            pooled = scatter_mean(x_all, data.batch.long(), dim=0, dim_size=batch_size)
        else:
            pooled = pooled

        critic = self.critic_mlp(self.dropout(pooled))

        return policy_logits, critic


__all__ = ["HiveActorCritic"]
