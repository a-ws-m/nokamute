from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, SAGEConv, to_hetero


class HeteroNodeEncoder(nn.Module):
    """Encode heterogeneous nodes using a small GNN and to_hetero().

    This builds a small GraphSAGE encoder and then uses `to_hetero`
    to adapt it to the heterogeneous types used by our `BoardHeteroBuilder`.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 32):
        super().__init__()
        # Graph convs expect node features as `x` on each node type.
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)

        # We create one message passing conv and an MLP applied to its outputs.
        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

        # Output feature size that subsequent layers expect
        self.out_dim = hidden_channels

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # This forward will be called by the to_hetero adapter for each node type
        # Linear layers applied per-node-type. Use functional ReLU because
        # `to_hetero` has trouble when modules introduce intermediate graph
        # nodes that do not get updated by all types.
        h = F.relu(self.lin1(x))
        h = F.relu(self.conv1(h, edge_index))
        h = F.relu(self.lin2(h))
        h = F.relu(self.conv2(h, edge_index))
        return h


class MoveScorer(nn.Module):
    """Model that encodes nodes and scores `current_move` edges.

    It expects `HeteroData` with node types `in_play_piece`, `out_of_play_piece`,
    `destination`, and edge type `('in_play_piece','current_move','destination')` and
    `('out_of_play_piece','current_move','destination')`.
    """

    def __init__(self, node_in_dim: int, hidden_dim: int = 64):
        super().__init__()
        # Create an encoder and wrap with to_hetero later in `prepare_hetero`.
        self.node_encoder = HeteroNodeEncoder(node_in_dim, hidden_channels=hidden_dim)

        # Simple MLP scoring edges: takes concatenated (src_feat, dst_feat)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Softmax over edges will be applied externally in forward

    def prepare_hetero(self, data: HeteroData):
        # Determine the maximum node feature dimension across types and
        # re-create the node encoder to accept that dimensionality.
        max_dim = 0
        for x in data.x_dict.values():
            if x is not None:
                max_dim = max(max_dim, x.size(1))

        # Re-initialize the node encoder with the required input dimension
        # if it differs from the current encoder setting.
        if max_dim <= 0:
            raise ValueError("No node features found in HeteroData")

        self.node_encoder = HeteroNodeEncoder(
            max_dim, hidden_channels=self.node_encoder.out_dim
        )

        # Pad features of all node types to `max_dim` so the shared encoder
        # can be applied uniformly.
        for k in list(data.x_dict.keys()):
            x = data.x_dict[k]
            if x is None:
                continue
            if x.size(1) < max_dim:
                pad = torch.zeros(
                    (x.size(0), max_dim - x.size(1)), dtype=x.dtype, device=x.device
                )
                data[k].x = torch.cat([x, pad], dim=1)

        self.hetero_encoder = to_hetero(self.node_encoder, data.metadata())

    def forward(self, data: HeteroData) -> torch.Tensor | list:
        if not hasattr(self, "hetero_encoder"):
            # Prepare using the graph metadata derived from the data
            self.prepare_hetero(data)

        # Run the hetero encoder to get per-node embeddings
        # It returns a dict keyed by node type.
        node_emb = self.hetero_encoder(data.x_dict, data.edge_index_dict)

        # Collect all edges of relation 'current_move' -> 'destination'
        edges = []
        for etype, edge_index in data.edge_index_dict.items():
            src_type, rel, dst_type = etype
            if rel != "current_move" or dst_type != "destination":
                continue

            # Only accept source types in our two source buckets
            if src_type not in ("in_play_piece", "out_of_play_piece"):
                continue

            src, dst = edge_index

            # Some nodes might be missing; make sure they exist in node_emb
            if src_type not in node_emb or "destination" not in node_emb:
                continue

            src_feat = node_emb[src_type][src]
            dst_feat = node_emb["destination"][dst]
            cat = torch.cat([src_feat, dst_feat], dim=1)
            scores = self.edge_mlp(cat).view(-1)
            edges.append(scores)

        if not edges:
            # No moves
            return torch.tensor([])

        logits = torch.cat(edges, dim=0)

        # If this is a batched graph, `destination` will carry a `batch`
        # vector indicating which graph each destination node belongs to.
        # Group logits by that batch index and apply softmax per graph.
        if "batch" in data["destination"]:
            # We collected edges in the same order as logits; collate the
            # per-edge batch index using the destination node batch vector.
            batch_idx_list = []
            for etype, edge_index in data.edge_index_dict.items():
                src_type, rel, dst_type = etype
                if rel != "current_move" or dst_type != "destination":
                    continue
                if src_type not in ("in_play_piece", "out_of_play_piece"):
                    continue
                dst = edge_index[1]
                batch_idx_list.append(data["destination"].batch[dst])

            batch_idx = torch.cat(batch_idx_list, dim=0)

            num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0

            out = []
            for g in range(num_graphs):
                mask = batch_idx == g
                if mask.sum().item() == 0:
                    out.append(torch.tensor([]))
                    continue
                logits_g = logits[mask]
                scores_g = torch.tanh(logits_g)
                out.append(scores_g)
            return out

        # Single graph case: just return tanh-activated scores
        action_scores = torch.tanh(logits)
        return action_scores
