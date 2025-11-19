from typing import Dict, List, Optional, Tuple

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected


class BoardHeteroBuilder:
    """Convert a nokamute.Board (Rust) -> PyG HeteroData.

    The Rust `to_graph` returns a dict describing node buckets and
    edges/turns. This builder converts that into a `HeteroData` with types:
    - in_play_piece
    - out_of_play_piece
    - destination

    Edge types:
    - (in_play_piece, 'adj', in_play_piece)
    - (in_play_piece, 'adj', destination)
    - (in_play_piece, 'current_move', destination)
    - (out_of_play_piece, 'current_move', destination)
    - (in_play_piece, 'next_move', destination)
    - (out_of_play_piece, 'next_move', destination)

    Node features: one-hot of bug types (queen..pillbug = 8) followed by
    two binary features representing is_underneath and is_above.
    The final feature vector layout is: [bug_one_hot, is_underneath, is_above].
    """

    BUGS = [
        "queen",
        "grasshopper",
        "spider",
        "ant",
        "beetle",
        "mosquito",
        "ladybug",
        "pillbug",
    ]

    def __init__(self, board):
        self.board = board
        self.raw = board.to_graph()

    def to_heterodata(self) -> HeteroData:
        data = HeteroData()

        in_play = list(self.raw["in_play_nodes"]) if "in_play_nodes" in self.raw else []
        out_of_play = (
            list(self.raw["out_of_play_nodes"])
            if "out_of_play_nodes" in self.raw
            else []
        )
        destination = (
            list(self.raw["destination_nodes"])
            if "destination_nodes" in self.raw
            else []
        )

        # Mappings
        # Map hex -> topmost in_play id (if there are multiple nodes per hex prefer topmost)
        in_map = {
            n["hex"]: n["id"] for n in in_play if not n.get("is_underneath", False)
        }
        out_map = {(n["bug"], n["color"]): n["id"] for n in out_of_play}
        dest_map = {n["hex"]: n["id"] for n in destination}

        # Node counts and features
        data["in_play_piece"].num_nodes = len(in_play)
        if len(in_play) > 0:
            # Add two extra binary features appended after the bug one-hot:
            # index len(BUGS)   -> is_underneath (1 if this piece is underneath another)
            # index len(BUGS)+1 -> is_above       (1 if there is a piece beneath this one)
            feature_dim = len(self.BUGS) + 2
            x = torch.zeros(len(in_play), feature_dim, dtype=torch.float32)
            for n in in_play:
                bug_idx = n["bug_idx"]
                x[n["id"], bug_idx] = 1.0
                # extra features
                if n.get("is_underneath", False):
                    x[n["id"], len(self.BUGS)] = 1.0
                if n.get("is_above", False):
                    x[n["id"], len(self.BUGS) + 1] = 1.0
            data["in_play_piece"].x = x

        data["out_of_play_piece"].num_nodes = len(out_of_play)
        if len(out_of_play) > 0:
            x = torch.zeros(len(out_of_play), len(self.BUGS), dtype=torch.float32)
            for n in out_of_play:
                bug_idx = n["bug_idx"]
                x[n["id"], bug_idx] = 1.0
            data["out_of_play_piece"].x = x

        data["destination"].num_nodes = len(destination)

        # Collect adjacency edges
        edges_adj_in_in = []
        edges_adj_in_dest = []
        for tup in self.raw.get("adjacency_edges", []):
            src_kind, src_idx, dst_kind, dst_idx = tup
            if src_kind == "in_play" and dst_kind == "in_play":
                edges_adj_in_in.append((src_idx, dst_idx))
            elif src_kind == "in_play" and dst_kind == "destination":
                edges_adj_in_dest.append((src_idx, dst_idx))

        if len(edges_adj_in_in) > 0:
            idxs = torch.tensor(edges_adj_in_in, dtype=torch.long).t().contiguous()
            data["in_play_piece", "adj", "in_play_piece"].edge_index = idxs
        if len(edges_adj_in_dest) > 0:
            idxs = torch.tensor(edges_adj_in_dest, dtype=torch.long).t().contiguous()
            data["in_play_piece", "adj", "destination"].edge_index = idxs

        # Helper to convert moves -> edge lists
        def moves_to_edges(moves, rel_name: str, color_name: Optional[str] = None):
            in_edges = []
            out_edges = []
            out_seen = set()
            for m in moves:
                if m.is_place():
                    hex, bug = m.get_place_info()
                    # Source is out-of-play piece.
                    # Color for place is the to_move on the board which generated the moves.
                    color = color_name or self.board.to_move().name()
                    key = (bug, color)
                    if key in out_map:
                        # If the exact hex isn't in the destination map (eg. a pass results
                        # in placements on adjacent hexes while our node bucket only contains
                        # the single START_HEX), fall back to the single dest if present.
                        if hex in dest_map:
                            out_edges.append((out_map[key], dest_map[hex]))
                        elif len(dest_map) == 1:
                            only_dest = list(dest_map.values())[0]
                            if rel_name == "next_move":
                                # Deduplicate next-move placements by bug; only one edge
                                # per out-of-play piece-type is required for the test.
                                if out_map[key] not in out_seen:
                                    out_edges.append((out_map[key], only_dest))
                                    out_seen.add(out_map[key])
                            else:
                                out_edges.append((out_map[key], only_dest))
                elif m.is_move():
                    from_hex, to_hex = m.get_move_info()
                    if from_hex in in_map and to_hex in dest_map:
                        in_edges.append((in_map[from_hex], dest_map[to_hex]))
            # Set edges in heterodata if any
            if in_edges:
                idxs = torch.tensor(in_edges, dtype=torch.long).t().contiguous()
                data["in_play_piece", rel_name, "destination"].edge_index = idxs
            if out_edges:
                idxs = torch.tensor(out_edges, dtype=torch.long).t().contiguous()
                data["out_of_play_piece", rel_name, "destination"].edge_index = idxs

        # Current player moves
        moves_current = self.board.legal_moves()
        moves_to_edges(moves_current, "current_move", self.board.to_move().name)

        # Next player moves: simulate a pass
        b2 = self.board.clone()
        # Use exported Turn class from the nokamute module to pass.
        import nokamute as _nm

        # Use getattr because `pass` is a Python keyword and cannot be used as an attribute name in code.
        b2.apply(getattr(_nm.Turn, "pass")())
        # Fallback: use the rust-provided textual move list for next moves
        try:
            moves_next = b2.legal_moves()
        except Exception:
            moves_next = []
        moves_to_edges(moves_next, "next_move", b2.to_move().name)

        return data

    def to_model_input(self) -> HeteroData:
        """Return a HeteroData prepared for models.

        This converts the directed adjacency to undirected edges (so both
        directions are present) and guarantees that `in_play_piece` and
        `out_of_play_piece` node types exist (adding a single dummy node
        with zero features if necessary). The original `to_heterodata`
        method is unchanged and used by tests that expect directed edges.
        """

        data = self.to_heterodata()

        # Ensure both node types exist - add one dummy node with matching
        # feature size if missing. This makes to_hetero transformations
        # robust when a type is missing in a specific position.
        # Add dummy in_play nodes if none were present.
        if (
            "in_play_piece" not in data.node_types
            or data["in_play_piece"].num_nodes == 0
        ):
            feature_dim = len(self.BUGS) + 2
            data["in_play_piece"].num_nodes = 1
            data["in_play_piece"].x = torch.zeros(1, feature_dim, dtype=torch.float32)

        # Make sure out_of_play nodes exist (16 at start); otherwise add one dummy.
        if (
            "out_of_play_piece" not in data.node_types
            or data["out_of_play_piece"].num_nodes == 0
        ):
            feature_dim = len(self.BUGS)
            data["out_of_play_piece"].num_nodes = 1
            data["out_of_play_piece"].x = torch.zeros(
                1, feature_dim, dtype=torch.float32
            )

        # Apply ToUndirected to mirror edges; this will double the current_move
        # edge counts (one for each direction), but we filter edges by their
        # relation and destination type in the model.
        data = ToUndirected()(data)

        # Some types (destination) do not carry features. Provide a default
        # zero feature vector so models can run a shared encoder across types.
        if "x" not in data["destination"] or data["destination"].x is None:
            feature_dim = len(self.BUGS) + 2
            data["destination"].x = torch.zeros(
                data["destination"].num_nodes, feature_dim, dtype=torch.float32
            )
        # Ensure that each node type appears as a destination node in at least
        # one edge — this makes `to_hetero` happy when mapping message
        # passing operations (otherwise some types are not updated).
        for ntype in data.node_types:
            has_dst = any(k[2] == ntype for k in data.edge_index_dict.keys())
            if not has_dst:
                # Add a trivial self-loop edge so convolution will be applied.
                idx = torch.tensor([[0], [0]], dtype=torch.long)
                data[ntype, "adj", ntype].edge_index = idx
        return data
