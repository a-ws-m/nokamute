from torch_geometric.data import HeteroData


class StackedHeteroData(HeteroData):
    """
    Thin wrapper around PyG's HeteroData that allows certain scalar attributes
    to be stacked along a new batch dimension instead of being concatenated.

    This is useful when we want per-example attributes like `y` or
    `selected_action_idx` to appear with shape [batch_size, ...] when
    collated by DataLoader (instead of being concatenated along the
    existing dimension). Use cases in this repo include per-example
    labels and selections attached to HeteroData objects for training.

    Notes:
    - We intentionally keep the default behavior for node/edge features and
      edge_index so PyG will correctly offset indices during batching.
    - Only override stacking for attributes that are scalar or small
      vectors that should be stacked as separate examples.
    """

    def __cat_dim__(self, key, value, *args, **kwargs):
        # Attributes listed here will be stacked into a new leading batch
        # dimension rather than concatenated (which is PyG's default).
        stack_keys = {
            "y",
            "selected_action_idx",
            "selected_action_local_idx",
            "current_player",
            "has_next_state",
        }

        if key in stack_keys:
            return None

        return super().__cat_dim__(key, value, *args, **kwargs)
