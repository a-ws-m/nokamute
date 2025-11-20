from typing import Dict, List, Optional, Tuple

import networkx as nx
import torch
from deepsnap.hetero_graph import HeteroGraph


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

    def as_networkx(self) -> nx.MultiGraph:
        """Return a networkx MultiGraph for the current board state.

        Nodes have attributes: node_type, node_label, node_feature, hex, color, bug, etc.
        Edges have attributes: edge_type and for move edges an edge_label (Turn object).
        """
        # Build a networkx multi-graph with node/edge types compatible with deepsnap
        G = nx.MultiGraph()

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
        # Assign offsets for each node bucket so all node ids are unique in the NX graph
        in_count = len(in_play)
        out_count = len(out_of_play)
        dest_count = len(destination)
        in_offset = 0
        out_offset = in_count
        dest_offset = in_count + out_count

        # Add in-play nodes to graph
        if len(in_play) > 0:
            # Add two extra binary features appended after the bug one-hot:
            # index len(BUGS)   -> is_underneath (1 if this piece is underneath another)
            # index len(BUGS)+1 -> is_above       (1 if there is a piece beneath this one)
            feature_dim = len(self.BUGS) + 2
            x = torch.zeros(len(in_play), feature_dim, dtype=torch.float32)
            for n in in_play:
                bug_idx = n["bug_idx"]
                local_id = n["id"]
                x[local_id, bug_idx] = 1.0
                # extra features
                if n.get("is_underneath", False):
                    x[local_id, len(self.BUGS)] = 1.0
                if n.get("is_above", False):
                    x[local_id, len(self.BUGS) + 1] = 1.0
                # global id
                gid = in_offset + local_id
                # add node with attributes including more human-readable fields
                G.add_node(
                    gid,
                    node_type="in_play_piece",
                    node_feature=x[local_id],
                    node_label=local_id,
                    hex=n.get("hex"),
                    color=n.get("color"),
                    bug=n.get("bug"),
                    is_underneath=n.get("is_underneath", False),
                    is_above=n.get("is_above", False),
                )

        # Out-of-play nodes
        if len(out_of_play) > 0:
            x_out = torch.zeros(len(out_of_play), len(self.BUGS), dtype=torch.float32)
            for n in out_of_play:
                bug_idx = n["bug_idx"]
                local_id = n["id"]
                x_out[local_id, bug_idx] = 1.0
                gid = out_offset + local_id
                G.add_node(
                    gid,
                    node_type="out_of_play_piece",
                    node_feature=x_out[local_id],
                    node_label=local_id,
                    bug=n.get("bug"),
                    color=n.get("color"),
                    num_left=n.get("num_left", 0),
                )

        # Destination nodes
        for n in destination:
            local_id = n["id"]
            gid = dest_offset + local_id
            G.add_node(
                gid,
                node_type="destination",
                node_feature=torch.zeros(1),
                node_label=local_id,
                hex=n.get("hex"),
                is_top=n.get("is_top", False),
            )

        # Collect adjacency edges and add to NX graph
        for tup in self.raw.get("adjacency_edges", []):
            src_kind, src_idx, dst_kind, dst_idx = tup
            # compute global ids
            if src_kind == "in_play":
                src_gid = in_offset + src_idx
            elif src_kind == "out_of_play":
                src_gid = out_offset + src_idx
            elif src_kind == "destination":
                src_gid = dest_offset + src_idx
            else:
                continue
            if dst_kind == "in_play":
                dst_gid = in_offset + dst_idx
            elif dst_kind == "out_of_play":
                dst_gid = out_offset + dst_idx
            elif dst_kind == "destination":
                dst_gid = dest_offset + dst_idx
            else:
                continue
            # adjacency edges are undirected
            G.add_edge(src_gid, dst_gid, edge_type="adj")

        # Helper to convert stringified moves -> edge lists
        def moves_to_edges(moves, rel_name: str, color_name: Optional[str] = None):
            """Convert stringified moves (from Rust `format!("{:?}", turn)`) into edges.

            Expected formats:
            - "Place(<hex>, <BugVariant>)"
            - "Move(<from>, <to>)"
            - "Pass"
            """
            out_seen = set()
            for m in moves:
                # only accept strings from the Rust side
                if not isinstance(m, str):
                    raise ValueError(
                        f"Expected move string from Rust, got {type(m)!r}: {m!r}"
                    )

                ms = m.strip()
                # Place
                if ms.startswith("Place("):
                    try:
                        inner = ms[len("Place(") : -1]
                        parts = [p.strip() for p in inner.split(",")]
                        if len(parts) == 2:
                            hex_val = int(parts[0])
                            bug_variant = parts[1]
                            bug_name = bug_variant.strip().lower()
                            color = color_name or self.board.to_move().name
                            key = (bug_name, color)
                            if key in out_map:
                                if hex_val in dest_map:
                                    src_gid = out_offset + out_map[key]
                                    dst_gid = dest_offset + dest_map[hex_val]
                                    G.add_edge(
                                        src_gid,
                                        dst_gid,
                                        edge_type=rel_name,
                                        edge_label=m,
                                    )
                                elif len(dest_map) == 1:
                                    only_dest = list(dest_map.values())[0]
                                    if rel_name == "next_move":
                                        if out_map[key] not in out_seen:
                                            src_gid = out_offset + out_map[key]
                                            dst_gid = dest_offset + only_dest
                                            G.add_edge(
                                                src_gid,
                                                dst_gid,
                                                edge_type=rel_name,
                                                edge_label=m,
                                            )
                                            out_seen.add(out_map[key])
                                    else:
                                        src_gid = out_offset + out_map[key]
                                        dst_gid = dest_offset + only_dest
                                        G.add_edge(
                                            src_gid,
                                            dst_gid,
                                            edge_type=rel_name,
                                            edge_label=m,
                                        )
                    except Exception as e:
                        raise ValueError(f"Malformed Place move string: {m!r}: {e}")

                # Move
                elif ms.startswith("Move("):
                    try:
                        inner = ms[len("Move(") : -1]
                        parts = [p.strip() for p in inner.split(",")]
                        if len(parts) == 2:
                            from_hex = int(parts[0])
                            to_hex = int(parts[1])
                            if from_hex in in_map and to_hex in dest_map:
                                src_gid = in_offset + in_map[from_hex]
                                dst_gid = dest_offset + dest_map[to_hex]
                                G.add_edge(
                                    src_gid, dst_gid, edge_type=rel_name, edge_label=m
                                )
                    except Exception as e:
                        raise ValueError(f"Malformed Move move string: {m!r}: {e}")
                else:
                    # Pass is allowed and produces no edges; otherwise it's an error
                    if ms.lower() in ("pass", "turn::pass"):
                        continue
                    raise ValueError(f"Unrecognized Turn string: {m!r}")

        # Current player moves: use the Rust-produced string list in the `to_graph` output
        moves_current = [m for m in self.raw.get("moves_current", [])]
        moves_to_edges(moves_current, "current_move", self.board.to_move().name)

        # Next player moves: use the Rust-produced `moves_next` strings (these are produced
        # from a cloned board with a Pass applied). We still create a clone and apply Pass
        # only to obtain the correct `to_move().name` for labeling.
        moves_next = [m for m in self.raw.get("moves_next", [])]
        b2 = self.board.clone()
        import nokamute as _nm

        b2.apply(getattr(_nm.Turn, "pass")())
        moves_to_edges(moves_next, "next_move", b2.to_move().name)

        return G

    def as_heterograph(self) -> HeteroGraph:
        """Return a deepsnap.HeteroGraph converted from the networkx graph."""
        G = self.as_networkx()
        # DeepSNAP's HeteroGraph treats any 'edge_label' attribute on edges as
        # a numeric or tensor label. BoardHeteroBuilder attaches string labels
        # (e.g. "Place(...)"), which raises a TypeError during HeteroGraph
        # construction. To preserve the networkx graph but avoid HeteroGraph
        # errors, make a shallow copy and remove the string labels only for
        # the deepSNAP representation.
        import copy

        G_copy = copy.deepcopy(G)
        for u, v, attrs in list(G_copy.edges(data=True)):
            attrs.pop("edge_label", None)

        H = HeteroGraph(G_copy)
        return H

    # Backwards compatibility: to_heterodata previously returned a PyG HeteroData. Now return DeepSNAP HeteroGraph
    def to_heterodata(self) -> HeteroGraph:
        return self.as_heterograph()
