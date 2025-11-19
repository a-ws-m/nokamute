from collections import OrderedDict
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GCNConv, SAGEConv, global_mean_pool, to_hetero


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

        # Critic head: pool node-level embeddings into graph-level embedding
        # and predict a single scalar per graph.
        self.critic_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

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

    def forward(self, data: HeteroData) -> tuple:
        # Always prepare the hetero encoder to account for graphs with
        # different node feature sizes or types; this ensures we will
        # correctly pad inputs when nodes change between positions.
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
            # No moves; return empty actions and empty critic
            return torch.tensor([]), torch.tensor([])

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
                # Softmax normalization applied per-graph
                scores_g = F.softmax(logits_g, dim=0)
                out.append(scores_g)
            # compute critic per graph
            # build pooled graph embedding by concatenating all node types
            x_all = []
            batch_all = []
            for ntype in data.node_types:
                emb = node_emb.get(ntype, None)
                if emb is None:
                    continue
                x_all.append(emb)
                batch_vec = (
                    data[ntype].batch
                    if "batch" in data[ntype]
                    else torch.zeros(emb.size(0), dtype=torch.long, device=emb.device)
                )
                batch_all.append(batch_vec)

            if x_all:
                x_all = torch.cat(x_all, dim=0)
                batch_all = torch.cat(batch_all, dim=0)
                pooled = global_mean_pool(x_all, batch_all)
                critic_vals = self.critic_mlp(pooled).view(-1)
            else:
                critic_vals = torch.tensor([], dtype=torch.float32)

            return out, critic_vals

        # Single graph case: just return tanh-activated scores
        # Single graph case: compute critic
        x_all = []
        batch_all = []
        for ntype in data.node_types:
            emb = node_emb.get(ntype, None)
            if emb is None:
                continue
            x_all.append(emb)
            # No `batch` in single graphs: zeros
            batch_all.append(
                torch.zeros(emb.size(0), dtype=torch.long, device=emb.device)
            )

        if x_all:
            x_all = torch.cat(x_all, dim=0)
            batch_all = torch.cat(batch_all, dim=0)
            pooled = global_mean_pool(x_all, batch_all)
            critic_vals = self.critic_mlp(pooled).view(-1)
        else:
            critic_vals = torch.tensor([], dtype=torch.float32)

        action_scores = F.softmax(logits, dim=0)
        return action_scores, critic_vals

    def action_scores_to_move_dicts(
        self, data: HeteroData, action_scores: torch.Tensor | list
    ) -> list:
        """Convert action scores back to {move_string: score} OrderedDicts per graph.

        This matches the ordering used in `forward` and groups batched scores
        by the `destination.batch` vector. For each graph we return an
        OrderedDict with moves ordered by descending score.
        """

        # Collect move strings in the same order we iterate in forward
        all_moves = []
        for etype, edge_index in data.edge_index_dict.items():
            src_type, rel, dst_type = etype
            if rel != "current_move" or dst_type != "destination":
                continue
            if src_type not in ("in_play_piece", "out_of_play_piece"):
                continue

            # move_str may be stored on the edge storage
            ms = getattr(data[etype], "move_str", None)
            if ms is None:
                # create placeholders
                ms = [None] * edge_index.shape[1]
            else:
                # flatten if collated as list-of-lists
                if isinstance(ms, list) and len(ms) > 0 and isinstance(ms[0], list):
                    ms = [s for inner in ms for s in inner]

            # extend as strings
            all_moves.extend(ms)

        # Single graph
        if isinstance(action_scores, torch.Tensor):
            assert (
                len(all_moves) == action_scores.numel()
            ), "Mismatch between moves and scores"
            scores = action_scores.detach().cpu()
            # Pair and sort
            pairs = list(zip(all_moves, scores.tolist()))
            pairs.sort(key=lambda p: p[1], reverse=True)
            return [OrderedDict(pairs)]

        # Batched: action_scores is list of tensors; gather batch mapping
        # Build batch_idx per edge by using destination index mapping
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
        # Now gather per-graph moves and scores
        num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() > 0 else 0
        out = []
        offset = 0
        for g in range(num_graphs):
            mask = batch_idx == g
            n = mask.sum().item()
            if n == 0:
                out.append(OrderedDict())
                continue
            # take next n moves from all_moves
            moves_g = all_moves[offset : offset + n]
            scores_g = action_scores[g].detach().cpu().tolist()
            assert len(moves_g) == len(scores_g)
            pairs = list(zip(moves_g, scores_g))
            pairs.sort(key=lambda p: p[1], reverse=True)
            out.append(OrderedDict(pairs))
            offset += n

        return out
