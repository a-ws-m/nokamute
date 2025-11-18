"""
League-based competitive self-play training for Hive.

Implements the approach from:
"Minimax Exploiter: A Data Efficient Approach for Competitive Self-Play"

This script orchestrates training of three agent archetypes:
1. Main Agent: Learns robust strategies via PFSP
2. Main Exploiter: Learns counter-strategies against Main Agent
3. League Exploiter: Learns counter-strategies against entire league
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from elo_tracker import EloTracker
from evaluate_vs_engine import evaluate_and_update_elo
from league import (
    AgentArchetype,
    ExploiterAgent,
    LeagueConfig,
    LeagueManager,
    LeagueTracker,
    prepare_exploiter_training_data,
)
from model_policy_hetero import create_policy_model
from self_play import SelfPlayGame, prepare_training_data
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm


def create_model_by_type(model_type, model_config):
    """
    Create model based on type selection.

    Args:
        model_type: "value", "policy", or "policy_hetero"
        model_config: Model configuration dict

    Returns:
        Model instance
    """
    # The homogeneous `model.py` has been removed; all model types now use
    # the heterogeneous policy model implemented in `model_policy_hetero`.
    if model_type in (None, "policy", "policy_hetero", "hetero"):
        return create_policy_model(model_config)
    # For backward compatibility, if model_type was set to "value/old" we
    # still instantiate the hetero model but emit a warning.
    else:
        print(
            f"Warning: legacy model_type '{model_type}' is unsupported — using heterogeneous policy model"
        )
        return create_policy_model(model_config)


def train_epoch_selfplay(model, training_data, optimizer, batch_size=32, device="cpu"):
    """
    Train model for one epoch using TD loss only (self-play format).
    Expects training_data as 7-element tuples.

    With the new architecture, both policy and value heads derive from the same action values:
    - Value = max(action_values)
    - Policy = softmax(action_values)

    Since they share the same parameters (action_value_head), we only need to train on the value target.
    The policy automatically learns to assign higher probabilities to better moves.

    Loss: TD Loss (value network) = MSE between predicted value V(S_t) and target value

    Args:
        model: GNN model (heterogeneous policy model with action_value_head)
        training_data: List of (hetero_data, move_to_action_indices, target_value)
        optimizer: Optimizer
        batch_size: Batch size
        device: Device

    Returns:
        Average TD loss
    """
    from hetero_graph_utils import prepare_model_inputs
    from torch_geometric.data import HeteroData

    model = model.to(device)
    model.train()
    total_loss = 0
    num_batches = 0

    from torch_geometric.loader import DataLoader as PyGDataLoader

    data_list = []
    for item in training_data:
        if len(item) == 9:
            (
                hetero_data,
                move_to_action_indices,
                target,
                selected_action_idx,
                selected_action_local_idx,
                next_hetero_data,
                next_move_to_action_indices,
                current_player,
            ) = item
        else:
            continue

        if not isinstance(hetero_data, HeteroData):
            continue

        hetero_data = hetero_data.cpu()
        hetero_data.y = torch.tensor([target], dtype=torch.float32)
        hetero_data.move_to_action_indices = move_to_action_indices.cpu()
        hetero_data.selected_action_idx = torch.tensor(
            [selected_action_idx], dtype=torch.long
        )
        hetero_data.selected_action_local_idx = torch.tensor(
            [selected_action_local_idx], dtype=torch.long
        )
        # No `move_order` stored on HeteroData anymore - we compute local->global
        # mapping from `move_to_action_indices` on demand during training so
        # move ordering is always derived from the final HeteroData structure.
        hetero_data.has_next_state = torch.tensor(
            [next_hetero_data is not None], dtype=torch.bool
        )
        player_int = 0 if current_player.name == "White" else 1
        hetero_data.current_player = torch.tensor([player_int], dtype=torch.long)
        data_list.append(hetero_data)

    if len(data_list) == 0:
        return 0.0

    loader = PyGDataLoader(data_list, batch_size=batch_size, shuffle=False)
    for batch in tqdm(loader, desc="Training batches", leave=False, unit="batch"):
        batch = batch.to(device)

        # Explicitly move custom tensor attributes
        def recursive_to(obj, device):
            if torch.is_tensor(obj):
                return obj.to(device)
            elif isinstance(obj, list):
                return [recursive_to(x, device) for x in obj]
            elif isinstance(obj, dict):
                return {k: recursive_to(v, device) for k, v in obj.items()}
            else:
                return obj

        if hasattr(batch, "move_to_action_indices"):
            batch.move_to_action_indices = recursive_to(
                batch.move_to_action_indices, device
            )
        if hasattr(batch, "selected_action_idx"):
            batch.selected_action_idx = batch.selected_action_idx.to(device)
        if hasattr(batch, "current_player"):
            batch.current_player = batch.current_player.to(device)
        if hasattr(batch, "has_next_state"):
            batch.has_next_state = batch.has_next_state.to(device)
        # Move all node and edge tensors inside batch to device
        for node_type in batch.node_types:
            if hasattr(batch[node_type], "x"):
                batch[node_type].x = batch[node_type].x.to(device)
        for edge_type in batch.edge_types:
            if hasattr(batch[edge_type], "edge_index"):
                batch[edge_type].edge_index = batch[edge_type].edge_index.to(device)
            if hasattr(batch[edge_type], "edge_attr"):
                batch[edge_type].edge_attr = batch[edge_type].edge_attr.to(device)

        optimizer.zero_grad()

        # Prepare inputs from batch (preserves batch information)
        x_dict, edge_index_dict, edge_attr_dict, _ = prepare_model_inputs(
            batch, batch.move_to_action_indices
        )

        # Move all x_dict, edge_index_dict, edge_attr_dict to device
        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
        edge_attr_dict = {k: v.to(device) for k, v in edge_attr_dict.items()}

        # Forward pass - heterogeneous policy model
        # For batched training, `action_values` may be a [batch_size, num_actions]
        # tensor. If available, we will prefer the `selected_action_idx` entry
        # per-example (action-level supervision) instead of using the
        # aggregated white/black values.
        action_values, white_value, black_value, _, _ = model(
            x_dict, edge_index_dict, edge_attr_dict, batch.move_to_action_indices
        )

        # Do NOT replace non-finite action_values with finite sentinels here.
        # If non-finite values are present for legal moves then the graph
        # construction / masking is incorrect and the issue should be fixed.

        # If selected_action_idx is available, index the per-example action
        # value and use it as training target. Otherwise, fallback to the
        # previously used white/black value prediction.
        sel_idx = None
        if hasattr(batch, "selected_action_idx") and action_values is not None:
            # DEBUG: log shapes first time through
            # NOTE: This will show up in test output for debugging purposes.
            if num_batches == 0:
                try:
                    import os

                    os.makedirs("python/tmp", exist_ok=True)
                    with open("python/tmp/nokamute_debug_train.txt", "a") as f:
                        f.write(
                            f"DEBUG action_values.shape={getattr(action_values,'shape',None)}\n"
                        )
                        f.write(
                            f"DEBUG selected_action_idx={batch.selected_action_idx}\n"
                        )
                except Exception:
                    pass
            # Normalize selected idx to 1D LongTensor (available for all batches)
            sel_idx = batch.selected_action_idx.view(-1).long()
            sel_local_idx = (
                batch.selected_action_local_idx.view(-1).long()
                if hasattr(batch, "selected_action_local_idx")
                else torch.full_like(sel_idx, -1)
            )

            # action_values may be [1, num_actions] (not batched) or
            # [batch_size, num_actions] (batched). Handle both.
            # action_values may be [batch_size, num_actions] when batched.
            # We want to index the per-example selected action and use that
            # as the predicted value. If selected index is invalid (-1) or
            # maps to an illegal action (resulting in -inf), fall back to the
            # per-graph white/black value.
            if action_values.dim() == 2 and action_values.shape[0] == sel_idx.shape[0]:
                batch_indices = torch.arange(sel_idx.shape[0], device=sel_idx.device)

                # Mask out invalid selections for both global and local
                valid_sel = sel_idx >= 0
                valid_local = sel_local_idx >= 0

                # Also mask out selections that are out-of-range for action_values
                max_idx = action_values.shape[-1]
                out_of_range = sel_idx >= max_idx
                if out_of_range.any():
                    try:
                        import sys

                        print(
                            f"[DEBUG] train_epoch_selfplay: selected indices out-of-range: {sel_idx[out_of_range]}",
                            file=sys.stderr,
                        )
                    except Exception:
                        pass
                valid_sel = valid_sel & (~out_of_range)

                # Initialize predictions with fallback per-player values
                fallback_vals = torch.where(
                    batch.current_player == 0,
                    white_value.squeeze(-1),
                    black_value.squeeze(-1),
                )

                # If no valid selections, keep fallback
                if not valid_sel.any():
                    predictions = fallback_vals
                else:
                    # Collect selected action values for valid entries
                    valid_batch_idx = batch_indices[valid_sel]
                    valid_sel_idx_global = sel_idx[valid_sel]
                    valid_sel_idx_local = sel_local_idx[valid_sel]

                    # Build per-batch legal action lists to map local -> global
                    batch_size_local = sel_idx.shape[0]
                    # Reconstruct per-edge batch assignment
                    move_edge_batch = []
                    batch_dict = {
                        nt: x_dict[nt].batch
                        for nt in x_dict
                        if hasattr(x_dict[nt], "batch")
                    }
                    for et in edge_index_dict.keys():
                        if "move" in et[1] or "rev_move" in et[1]:
                            edge_idx = edge_index_dict[et]
                            if edge_idx.shape[1] > 0 and et[0] in batch_dict:
                                move_edge_batch.append(batch_dict[et[0]][edge_idx[0]])
                    all_move_batch = (
                        torch.cat(move_edge_batch)
                        if move_edge_batch
                        else torch.empty(0, dtype=torch.long)
                    )

                    # Map local indices to global action indices using the per-example
                    # `move_order` list produced in `prepare_training_data` (preferred),
                    # otherwise fall back to scanning `batch.move_to_action_indices`.
                    selected_global_for_valid = []
                    use_move_order = False
                    move_order_list = None
                    for b_idx, l_idx in zip(
                        valid_batch_idx.tolist(), valid_sel_idx_local.tolist()
                    ):
                        if l_idx < 0:
                            selected_global_for_valid.append(-1)
                            continue
                        if use_move_order and move_order_list is not None:
                            local_list = move_order_list[b_idx]
                            if 0 <= l_idx < len(local_list):
                                selected_global_for_valid.append(local_list[l_idx])
                                continue

                        if all_move_batch.numel() > 0:
                            legal_idxs = batch.move_to_action_indices[
                                all_move_batch == b_idx
                            ]
                            # preserve order of first appearance
                            seen = set()
                            unique_ordered = []
                            for x in legal_idxs.tolist():
                                if x >= 0 and x not in seen:
                                    unique_ordered.append(x)
                                    seen.add(x)
                        else:
                            unique_ordered = []

                        if 0 <= l_idx < len(unique_ordered):
                            selected_global_for_valid.append(unique_ordered[l_idx])
                        else:
                            selected_global_for_valid.append(-1)

                    selected_global_for_valid = torch.tensor(
                        selected_global_for_valid,
                        device=action_values.device,
                        dtype=torch.long,
                    )

                    # Now we can fetch action values (global indexing) for these selected moves
                    selected_vals_valid = action_values[
                        valid_batch_idx, selected_global_for_valid
                    ]

                    # DEBUG ASSERT: selected action values must be finite for current player
                    if not torch.isfinite(selected_vals_valid).all():
                        try:
                            import sys

                            print(
                                f"[DEBUG] train_epoch_selfplay: Found non-finite selected action values: {selected_vals_valid}",
                                file=sys.stderr,
                            )
                            # Log the legal action mask per example
                            for b_idx, sel_i in zip(
                                valid_batch_idx.tolist(),
                                selected_global_for_valid.tolist(),
                            ):
                                legal_mask = torch.isfinite(action_values[b_idx])
                                legal_idxs = (
                                    torch.nonzero(legal_mask).squeeze(-1).tolist()
                                )
                                print(
                                    f"[DEBUG] batch={b_idx}, selected={sel_i}, legal_idxs_sample={legal_idxs[:10]}",
                                    file=sys.stderr,
                                )
                        except Exception:
                            pass
                    # If any selected value is -inf (illegal), fall back on that example
                    finite_mask = torch.isfinite(selected_vals_valid)
                    # Always log the raw values for the first batch (helps diagnose mixture)
                    if num_batches == 0:
                        try:
                            import numpy as _np

                            with open("python/tmp/nokamute_debug_train.txt", "a") as f:
                                f.write(
                                    f"DEBUG selected_vals_valid_raw={_np.round(selected_vals_valid.cpu().numpy(),3)}\n"
                                )
                                f.write(
                                    f"DEBUG finite_mask_raw={finite_mask.cpu().numpy()}\n"
                                )
                        except Exception:
                            pass

                    # Start with fallback, then fill in valid finite selections
                    predictions = fallback_vals.clone()
                    if finite_mask.any():
                        # Assign only finite values
                        predictions[valid_sel] = torch.where(
                            finite_mask, selected_vals_valid, fallback_vals[valid_sel]
                        )
                    # More detailed debug: log the selected action values and predictions
                    if num_batches == 0:
                        try:
                            import numpy as _np

                            with open("python/tmp/nokamute_debug_train.txt", "a") as f:
                                f.write(
                                    f"DEBUG valid_batch_idx={_np.array(valid_batch_idx.cpu())}\n"
                                )
                                f.write(
                                    f"DEBUG valid_selected_global_idx={_np.array(selected_global_for_valid.cpu())}\n"
                                )
                                f.write(
                                    f"DEBUG selected_vals_valid={_np.round(selected_vals_valid.cpu().numpy(),3)}\n"
                                )
                                f.write(
                                    f"DEBUG predictions_after={_np.round(predictions.cpu().numpy(),3)}\n"
                                )
                        except Exception:
                            pass
                    # Debug info: log selected and fallback values
                    if num_batches == 0:
                        try:
                            import numpy as _np

                            valid_batch_idx = batch_indices[valid_sel]
                            with open("python/tmp/nokamute_debug_train.txt", "a") as f:
                                f.write(
                                    f"DEBUG selected_vals_valid={_np.round(selected_vals_valid.cpu().numpy(),3)}\n"
                                )
                                f.write(
                                    f"DEBUG fallback_vals={_np.round(fallback_vals.cpu().numpy(),3)}\n"
                                )
                                f.write(
                                    f"DEBUG finite_mask={finite_mask.cpu().numpy()}\n"
                                )
                        except Exception:
                            pass
            elif action_values.dim() == 2 and action_values.shape[0] == 1:
                # Non-batched action_values - use same action values for all
                # examples in this mini-batch (rare). Index the leading row.
                # Mask invalid indices and fall back similarly.
                sel_idx_1d = sel_idx
                valid_sel = sel_idx_1d >= 0

                fallback_vals = torch.where(
                    batch.current_player == 0,
                    white_value.squeeze(-1),
                    black_value.squeeze(-1),
                )

                if valid_sel.any():
                    all_selected = action_values.squeeze(0)[sel_idx_1d.clamp(min=0)]
                    finite_mask = torch.isfinite(all_selected)
                    predictions = torch.where(
                        valid_sel & finite_mask, all_selected, fallback_vals
                    )
                else:
                    predictions = fallback_vals
            else:
                # Fallback
                predictions = torch.where(
                    batch.current_player == 0,
                    white_value.squeeze(-1),
                    black_value.squeeze(-1),
                )  # [batch_size]
        else:
            # Select appropriate value based on current player
            predictions = torch.where(
                batch.current_player == 0,
                white_value.squeeze(-1),
                black_value.squeeze(-1),
            )  # [batch_size]
        targets = batch.y.squeeze()

        # Ensure predictions are finite and 1D (one value per example). Sometimes a
        # mis-shaped predictions tensor can occur; prefer indexing by
        # `sel_idx` when available.
        if predictions.dim() > 1:
            # Only attempt to salvage using action_values if `sel_idx` is in
            # scope and has appropriate length.
            if (
                "sel_idx" in locals()
                and action_values is not None
                and action_values.dim() == 2
                and sel_idx.numel() == action_values.shape[0]
            ):
                idx = torch.arange(action_values.shape[0], device=sel_idx.device)
                predictions = action_values[idx, sel_idx]
            else:
                predictions = predictions.max(dim=-1)[0]

        # Replace non-finite predictions (e.g., -inf from illegal actions)
        if not torch.isfinite(predictions).all():
            try:
                predictions = torch.where(
                    torch.isfinite(predictions),
                    predictions,
                    torch.where(
                        batch.current_player == 0,
                        white_value.squeeze(-1),
                        black_value.squeeze(-1),
                    ),
                )
            except Exception:
                # If batch attrs are missing, fallback to zeros
                predictions = torch.zeros_like(predictions)

        # TD Loss (value network supervised by game outcome)
        loss = F.mse_loss(predictions, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if num_batches == 0:
            try:
                with open("python/tmp/nokamute_debug_train.txt", "a") as f:
                    f.write(
                        f"DEBUG final predictions={predictions[:10].cpu().numpy()}\n"
                    )
            except Exception:
                pass
        num_batches += 1

    # Return average TD loss
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def train_epoch_exploiter(model, training_data, optimizer, batch_size=32, device="cpu"):
    """
    Train model for one epoch using TD loss only (exploiter format).
    Expects training_data as 4-element tuples: (hetero_data, move_to_action_indices, target_value, exploiter_colour)

    With the new architecture, both policy and value heads derive from the same action values:
    - Value = max(action_values)
    - Policy = softmax(action_values)

    Since they share the same parameters (action_value_head), we only need to train on the value target.
    The policy automatically learns to assign higher probabilities to better moves.

    Loss: TD Loss (value network) = MSE between predicted value V(S_t) and target value

    Args:
        model: GNN model (heterogeneous policy model with action_value_head)
        training_data: List of (hetero_data, move_to_action_indices, target_value, exploiter_colour)
        optimizer: Optimizer
        batch_size: Batch size
        device: Device

    Returns:
        Average TD loss
    """
    from hetero_graph_utils import prepare_model_inputs
    from torch_geometric.data import HeteroData

    model = model.to(device)
    model.train()
    total_loss = 0
    num_batches = 0

    from torch_geometric.loader import DataLoader as PyGDataLoader

    data_list = []
    exploiter_colours = []
    for item in training_data:
        # Expecting (hetero_data, move_to_action_indices, target, exploiter_colour)
        if len(item) == 4:
            hetero_data, move_to_action_indices, target, exploiter_colour = item
        else:
            continue

        if not isinstance(hetero_data, HeteroData):
            continue

        hetero_data = hetero_data.cpu()
        hetero_data.y = torch.tensor([target], dtype=torch.float32)
        hetero_data.move_to_action_indices = move_to_action_indices.cpu()
        data_list.append(hetero_data)
        exploiter_colours.append(exploiter_colour)

    if len(data_list) == 0:
        return 0.0

    loader = PyGDataLoader(data_list, batch_size=batch_size, shuffle=False)

    # Split exploiter_colours into batches
    def batch_exploiter_colours(colours, batch_size):
        for i in range(0, len(colours), batch_size):
            yield colours[i : i + batch_size]

    for batch, batch_colours in zip(
        tqdm(loader, desc="Training batches", leave=False, unit="batch"),
        batch_exploiter_colours(exploiter_colours, batch_size),
    ):
        batch = batch.to(device)

        # Explicitly move custom tensor attributes
        def recursive_to(obj, device):
            if torch.is_tensor(obj):
                return obj.to(device)
            elif isinstance(obj, list):
                return [recursive_to(x, device) for x in obj]
            elif isinstance(obj, dict):
                return {k: recursive_to(v, device) for k, v in obj.items()}
            else:
                return obj

        if hasattr(batch, "move_to_action_indices"):
            batch.move_to_action_indices = recursive_to(
                batch.move_to_action_indices, device
            )
        for node_type in batch.node_types:
            if hasattr(batch[node_type], "x"):
                batch[node_type].x = batch[node_type].x.to(device)
        for edge_type in batch.edge_types:
            if hasattr(batch[edge_type], "edge_index"):
                batch[edge_type].edge_index = batch[edge_type].edge_index.to(device)
            if hasattr(batch[edge_type], "edge_attr"):
                batch[edge_type].edge_attr = batch[edge_type].edge_attr.to(device)

        optimizer.zero_grad()

        # Prepare inputs from batch (preserves batch information)
        x_dict, edge_index_dict, edge_attr_dict, _ = prepare_model_inputs(
            batch, batch.move_to_action_indices
        )

        # Move all x_dict, edge_index_dict, edge_attr_dict to device
        x_dict = {k: v.to(device) for k, v in x_dict.items()}
        edge_index_dict = {k: v.to(device) for k, v in edge_index_dict.items()}
        edge_attr_dict = {k: v.to(device) for k, v in edge_attr_dict.items()}

        # Forward pass - heterogeneous policy model
        _, white_value, black_value, _, _ = model(
            x_dict, edge_index_dict, edge_attr_dict, batch.move_to_action_indices
        )

        # Select appropriate value based on exploiter's colour
        # batch_colours: list of "White" or "Black"
        batch_colours_tensor = torch.tensor(
            [0 if c == "White" else 1 for c in batch_colours], device=device
        )
        value_current = torch.where(
            batch_colours_tensor == 0,
            white_value.squeeze(-1),
            black_value.squeeze(-1),
        )  # [batch_size]

        predictions = value_current
        targets = batch.y.squeeze()

        # TD Loss (value network supervised by game outcome)
        loss = F.mse_loss(predictions, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    # Return average TD loss
    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def train_main_agent(
    league_manager: LeagueManager,
    league_tracker: LeagueTracker,
    config: LeagueConfig,
    iteration: int,
    args=None,
):
    """
    Train the Main Agent for one iteration using PFSP.

    Args:
        league_manager: League manager
        league_tracker: Performance tracker
        config: League configuration
        iteration: Current training iteration
    """
    print(f"\n{'='*60}")
    print(f"Training Main Agent (Iteration {iteration})")
    print(f"{'='*60}")

    # Load current Main Agent model
    agent = league_manager.current_main_agent
    checkpoint = torch.load(
        agent.model_path, map_location=config.train_device, weights_only=False
    )
    model_type = checkpoint.get("model_type", "policy")
    model = create_model_by_type(model_type, checkpoint.get("config", {})).to(
        config.train_device
    )

    # Handle torch.compile() state dict keys (strip _orig_mod. prefix)
    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(config.train_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.main_agent_lr)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Move model to gen_device for game generation
    model_for_gen = model.to(config.gen_device)

    # Generate self-play games using PFSP opponent selection
    print(f"\nGenerating {config.main_agent_games_per_iter} self-play games...")
    games = []
    opponent_sample_counts = {}

    player = SelfPlayGame(
        model=model_for_gen,  # <-- model on gen_device
        epsilon=config.epsilon,  # Epsilon-greedy sampling
        device=config.gen_device,
        enable_branching=config.enable_branching,
        max_moves=config.max_moves,
        use_amp=args.use_amp if args else False,
        cache_graphs=args.cache_graphs if args else True,
        inference_batch_size=config.inference_batch_size,
    )

    for game_idx in tqdm(
        range(config.main_agent_games_per_iter), desc="Generating games", unit="game"
    ):
        # Sample opponent using PFSP
        opponent_agent = league_manager.sample_opponent_for_main_agent()
        opponent_sample_counts[opponent_agent.name] = (
            opponent_sample_counts.get(opponent_agent.name, 0) + 1
        )

        # Load opponent model
        opponent_checkpoint = torch.load(
            opponent_agent.model_path,
            map_location=config.gen_device,
            weights_only=False,
        )
        opponent_model_type = opponent_checkpoint.get("model_type", "policy")
        opponent_model = create_model_by_type(
            opponent_model_type, opponent_checkpoint.get("config", {})
        ).to(config.gen_device)

        # Handle torch.compile() state dict keys
        opponent_state_dict = opponent_checkpoint["model_state_dict"]
        if any(key.startswith("_orig_mod.") for key in opponent_state_dict.keys()):
            opponent_state_dict = {
                k.replace("_orig_mod.", ""): v for k, v in opponent_state_dict.items()
            }
        opponent_model.load_state_dict(opponent_state_dict)
        opponent_model = opponent_model.to(config.gen_device)
        opponent_model.eval()

        # Play game (alternating who plays as White)
        main_is_white = game_idx % 2 == 0
        game_data, result, branch_id = player.play_game()

        # Record result
        if main_is_white:
            main_result = result
        else:
            main_result = -result

        agent.record_game_result(opponent_agent.name, main_result)
        opponent_agent.record_game_result(agent.name, -main_result)

        # Log game
        league_tracker.log_game_result(
            iteration, agent, opponent_agent, main_result, len(game_data)
        )

        games.append((game_data, result, branch_id))

    # Log PFSP statistics
    league_tracker.log_pfsp_stats(iteration, opponent_sample_counts)

    # Prepare training data
    print("Preparing training data...")
    training_data = prepare_training_data(games)
    print(f"Training examples: {len(training_data)}")

    # Train for multiple epochs
    print(f"\nTraining for {config.main_agent_epochs} epochs...")
    for epoch in tqdm(
        range(config.main_agent_epochs), desc="Training epochs", unit="epoch"
    ):
        loss = train_epoch_selfplay(
            model,
            training_data,
            optimizer,
            batch_size=config.main_agent_batch_size,
            device=config.train_device,  # <-- use train_device for training
        )

        # Log metrics
        league_tracker.log_training_metrics(
            iteration * config.main_agent_epochs + epoch,
            agent.name,
            loss,
            optimizer.param_groups[0]["lr"],
            epoch=epoch,
        )

        if (epoch + 1) % 5 == 0 or epoch == 0:
            tqdm.write(
                f"  Epoch {epoch+1}/{config.main_agent_epochs}: TD Loss = {loss:.6f}"
            )

    # Update Main Agent
    model_config = checkpoint.get("config", {})
    updated_agent = league_manager.update_main_agent(model, optimizer, model_config)

    # Log performance
    league_tracker.log_agent_performance(iteration, updated_agent)

    return updated_agent


def train_main_exploiter(
    league_manager: LeagueManager,
    league_tracker: LeagueTracker,
    config: LeagueConfig,
    iteration: int,
):
    """
    Train the Main Exploiter against the current Main Agent.

    Uses minimax reward shaping for efficient exploitation.

    Args:
        league_manager: League manager
        league_tracker: Performance tracker
        config: League configuration
        iteration: Current training iteration
    """
    print(f"\n{'='*60}")
    print(f"Training Main Exploiter (Iteration {iteration})")
    print(f"{'='*60}")

    exploiter = league_manager.current_main_exploiter
    if exploiter is None:
        print("No Main Exploiter to train")
        return None

    # Load exploiter model
    checkpoint = torch.load(
        exploiter.model_path, map_location=config.train_device, weights_only=False
    )
    exploiter_model_type = checkpoint.get("model_type", "policy")
    model = create_model_by_type(exploiter_model_type, checkpoint.get("config", {})).to(
        config.train_device
    )

    # Handle torch.compile() state dict keys
    state_dict = checkpoint["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(config.train_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.exploiter_lr)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Move exploiter model to gen_device for game generation
    model_for_gen = model.to(config.gen_device)

    # Load target (Main Agent) model
    main_agent = league_manager.current_main_agent
    main_checkpoint = torch.load(
        main_agent.model_path, map_location=config.gen_device, weights_only=False
    )
    main_model_type = main_checkpoint.get("model_type", "policy")
    main_model = create_model_by_type(
        main_model_type, main_checkpoint.get("config", {})
    ).to(config.gen_device)

    # Handle torch.compile() state dict keys
    main_state_dict = main_checkpoint["model_state_dict"]
    if any(key.startswith("_orig_mod.") for key in main_state_dict.keys()):
        main_state_dict = {
            k.replace("_orig_mod.", ""): v for k, v in main_state_dict.items()
        }
    main_model.load_state_dict(main_state_dict)
    main_model = main_model.to(config.gen_device)
    main_model.eval()

    # Create exploiter agent
    exploiter_player = ExploiterAgent(
        model=model_for_gen,  # <-- model on gen_device
        opponent_model=main_model,
        device=config.gen_device,
        epsilon=config.epsilon,
        minimax_reward_weight=config.minimax_reward_weight,
        gamma=config.minimax_gamma,
        enable_branching=config.enable_branching,
        max_moves=config.max_moves,
        inference_batch_size=config.inference_batch_size,
    )

    # Generate training games with minimax rewards
    print(
        f"\nGenerating {config.exploiter_games_per_iter} exploiter games with minimax rewards..."
    )
    games = []
    greedy_games = []
    for i in range(config.exploiter_games_per_iter):
        # Alternate epsilon: half greedy (epsilon=0), half exploratory (epsilon=config.exploiter_epsilon or 0.1)
        epsilon = 0.0 if i % 2 == 0 else exploiter_player.epsilon
        exploiter_player.epsilon = epsilon
        game_data, result, branch_id = exploiter_player.play_game_with_minimax_rewards(
            exploiter_is_white=(i % 2 == 0)
        )
        games.append((game_data, result, branch_id))
        if epsilon == 0.0:
            greedy_games.append((game_data, result, branch_id))
    # Restore exploiter_player.epsilon to original value
    exploiter_player.epsilon = config.minimax_reward_weight

    # Record results
    for idx, (game_data, result, _) in enumerate(games):
        # Result is from White's perspective
        exploiter_result = result  # Exploiter is White
        main_result = -result

        exploiter.record_game_result(main_agent.name, exploiter_result)
        main_agent.record_game_result(exploiter.name, main_result)

        league_tracker.log_game_result(
            iteration, exploiter, main_agent, exploiter_result, len(game_data)
        )

    # Prepare training data with minimax rewards
    print("Preparing exploiter training data...")
    training_data = prepare_exploiter_training_data(
        games, minimax_reward_weight=config.minimax_reward_weight
    )
    print(f"Training examples: {len(training_data)}")

    # Train for multiple epochs
    print(f"\nTraining for {config.exploiter_epochs} epochs...")
    for epoch in tqdm(
        range(config.exploiter_epochs), desc="Training epochs", unit="epoch"
    ):
        loss = train_epoch_exploiter(
            model,
            training_data,
            optimizer,
            batch_size=config.exploiter_batch_size,
            device=config.train_device,  # <-- use train_device for training
        )

        exploiter.training_epochs += 1

        # Log metrics
        league_tracker.log_training_metrics(
            iteration * config.exploiter_epochs + epoch,
            exploiter.name,
            loss,
            optimizer.param_groups[0]["lr"],
            epoch=epoch,
        )

        if (epoch + 1) % 5 == 0 or epoch == 0:
            tqdm.write(
                f"  Epoch {epoch+1}/{config.exploiter_epochs}: TD Loss = {loss:.6f}"
            )

    # Save updated exploiter
    model_config = checkpoint.get("config", {})

    # Get state dict and strip torch.compile() prefix if present
    state_dict = model.state_dict()
    if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    torch.save(
        {
            "model_state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "config": model_config,
            "model_type": model_config.get(
                "model_type", "policy"
            ),  # Store at top level for easy access
            "iteration": iteration,
            "target_agent": main_agent.name,
        },
        exploiter.model_path,
    )  # Check convergence
    win_rate_vs_main = exploiter.get_win_rate_vs(main_agent.name)
    print(f"\nExploiter win rate vs {main_agent.name}: {win_rate_vs_main:.2%}")

    # Get recent games for convergence check (only greedy games)
    recent_games = []
    for game_data, result, _ in greedy_games:
        # Only count results from greedy games
        # Result is from exploiter's perspective
        # Use main_agent.name as opponent
        if result > 0.5:
            recent_games.append((main_agent.name, 1.0))
        elif result < -0.5:
            recent_games.append((main_agent.name, -1.0))
        else:
            recent_games.append((main_agent.name, 0.0))

    is_converged = league_manager.check_exploiter_convergence(exploiter, recent_games)
    exploiter.is_converged = is_converged

    # Log convergence
    league_tracker.log_exploiter_convergence(
        iteration,
        exploiter,
        win_rate_vs_main,
        config.main_exploiter_convergence_threshold,
    )

    if is_converged:
        print(
            f"✓ Main Exploiter CONVERGED! (WR: {win_rate_vs_main:.2%} >= {config.main_exploiter_convergence_threshold:.2%})"
        )

    # Log performance
    league_tracker.log_agent_performance(iteration, exploiter)

    return exploiter


def main():
    parser = argparse.ArgumentParser(
        description="League-based competitive self-play training"
    )

    # League configuration
    parser.add_argument(
        "--config", type=str, help="Path to league config file (optional)"
    )

    # Training schedule
    parser.add_argument(
        "--iterations", type=int, default=1000, help="Total training iterations"
    )
    parser.add_argument(
        "--main-games", type=int, default=10, help="Games per Main Agent iteration"
    )
    parser.add_argument(
        "--exploiter-games", type=int, default=20, help="Games per Exploiter iteration"
    )

    # Model architecture
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden dimension")
    parser.add_argument(
        "--num-layers", type=int, default=3, help="Number of GNN layers"
    )

    # Paths
    parser.add_argument(
        "--save-dir",
        type=str,
        default="league_checkpoints",
        help="Directory for saving league state",
    )
    parser.add_argument(
        "--resume", type=str, help="Resume from league checkpoint directory"
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        help="Path to pretrained model for initialization",
    )

    # Evaluation
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=50,
        help="Evaluate against engine every N iterations",
    )
    parser.add_argument(
        "--eval-games", type=int, default=20, help="Games per engine evaluation"
    )
    parser.add_argument(
        "--eval-depths",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Engine depths to evaluate against",
    )

    # Device
    parser.add_argument(
        "--gen-device",
        type=str,
        default="cpu",
        help="Device for self-play game generation (default: cpu)",
    )
    parser.add_argument(
        "--train-device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for model training (default: cuda if available, else cpu)",
    )

    # Model type
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["value", "policy"],
        default="policy",
        help="Model architecture: 'value' (evaluates each move) or 'policy' (fixed action space). "
        "Benchmark shows policy is ~11x faster for game generation.",
    )

    # Performance optimization arguments
    parser.add_argument(
        "--use-compile",
        action="store_true",
        default=True,
        help="Use torch.compile() for model inference speedup (default: True, GPU recommended)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_false",
        dest="use_compile",
        help="Disable torch.compile() (useful for debugging or CPU-only systems)",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use Automatic Mixed Precision for faster GPU inference",
    )
    parser.add_argument(
        "--no-cache-graphs",
        action="store_false",
        dest="cache_graphs",
        default=True,
        help="Disable graph caching",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=None,
        help="Batch size for position evaluation during game generation. "
        "None = evaluate all positions in single batch. "
        "Set to your GPU's max capacity (e.g., 256, 512, 1024) for optimal performance.",
    )

    args = parser.parse_args()

    # Create or load league configuration
    if args.config:
        # Load from file (implement as needed)
        config = LeagueConfig()
    else:
        config = LeagueConfig(
            main_agent_games_per_iter=args.main_games,
            exploiter_games_per_iter=args.exploiter_games,
            gen_device=args.gen_device,  # <-- add to config
            train_device=args.train_device,  # <-- add to config
            inference_batch_size=args.inference_batch_size,
        )

    print("\n" + "=" * 60)
    print("LEAGUE TRAINING SYSTEM")
    print("=" * 60)
    print(f"Game generation device: {config.gen_device}")
    print(f"Training device: {config.train_device}")
    print(f"Model type: {args.model_type}")
    print(f"Main Agent games/iter: {config.main_agent_games_per_iter}")
    print(f"Exploiter games/iter: {config.exploiter_games_per_iter}")
    print(f"Minimax reward weight: {config.minimax_reward_weight}")
    print(f"PFSP exponent: {config.pfsp_exponent}")
    if config.inference_batch_size is not None:
        print(f"Inference batch size: {config.inference_batch_size}")
    else:
        print(f"Inference batch size: Dynamic (all positions in single batch)")

    # Setup directories
    save_dir = Path(args.resume) if args.resume else Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Initialize league manager and tracker
    league_manager = LeagueManager(config, str(save_dir))
    league_tracker = LeagueTracker(
        str(save_dir / "logs"),
        elo_save_path=str(save_dir / "elo_ratings.json"),
        engine_depths=args.eval_depths,
    )

    # Initialize or resume Main Agent
    model_config = {
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "model_type": args.model_type,
    }

    if league_manager.current_main_agent is None:
        print("\nInitializing Main Agent...")
        model = create_model_by_type(args.model_type, model_config).to(
            config.train_device
        )

        # Load pretrained weights if specified
        if args.pretrained_model:
            print(f"Loading pretrained model from {args.pretrained_model}...")
            pretrained = torch.load(
                args.pretrained_model,
                map_location=config.train_device,
                weights_only=False,
            )
            model.load_state_dict(pretrained["model_state_dict"])

        optimizer = torch.optim.Adam(model.parameters(), lr=config.main_agent_lr)
        league_manager.initialize_main_agent(model, optimizer, model_config)

        # Apply torch.compile() for faster inference AFTER saving initial checkpoint
        if args.use_compile:
            if config.train_device == "cpu":
                print("Note: torch.compile() on CPU may not provide speedup")
            else:
                try:
                    print("Compiling model with torch.compile(fullgraph=True)...")
                    if torch.cuda.is_available():
                        capability = torch.cuda.get_device_capability()
                        if capability[0] < 7:
                            print(
                                f"GPU CUDA capability {capability[0]}.{capability[1]} < 7.0, using eager backend"
                            )
                            model = torch.compile(
                                model, fullgraph=True, backend="eager"
                            )
                        else:
                            model = torch.compile(
                                model, fullgraph=True, mode="reduce-overhead"
                            )
                    else:
                        model = torch.compile(
                            model, fullgraph=True, mode="reduce-overhead"
                        )
                    print("✓ Model compiled successfully with fullgraph=True!")
                except Exception as e:
                    print(f"Warning: Could not compile model: {type(e).__name__}")
                    print(f"Falling back to default compile without fullgraph")
                    model = torch.compile(model, mode="reduce-overhead")
    else:
        print(f"\nResuming from iteration {league_manager.iteration}")

    # Main training loop
    start_iteration = league_manager.iteration

    for iteration in tqdm(
        range(start_iteration, args.iterations), desc="League iterations", unit="iter"
    ):
        league_manager.iteration = iteration

        tqdm.write(f"\n{'='*80}")
        tqdm.write(f"LEAGUE ITERATION {iteration + 1}/{args.iterations}")
        tqdm.write(f"{'='*80}")

        # 1. Train Main Agent
        if iteration % config.main_agent_update_interval == 0:
            main_agent = train_main_agent(
                league_manager, league_tracker, config, iteration, args
            )

        # 2. Spawn new Main Exploiter if needed
        if iteration > 0 and iteration % config.main_exploiter_spawn_interval == 0:
            print(f"\n{'='*60}")
            print(f"Spawning new Main Exploiter")
            print(f"{'='*60}")

            # Create fresh model for exploiter
            model = create_model_by_type(args.model_type, model_config).to(
                config.train_device
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=config.exploiter_lr)
            league_manager.spawn_main_exploiter(model, optimizer, model_config)

        # 3. Train Main Exploiter if one exists
        if league_manager.current_main_exploiter is not None:
            if not league_manager.current_main_exploiter.is_converged:
                train_main_exploiter(league_manager, league_tracker, config, iteration)

        # 4. Spawn League Exploiter (less frequently)
        if iteration > 0 and iteration % config.league_exploiter_spawn_interval == 0:
            print(f"\n{'='*60}")
            print(f"Spawning new League Exploiter")
            print(f"{'='*60}")

            model = create_model_by_type(args.model_type, model_config).to(
                config.train_device
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=config.exploiter_lr)
            league_manager.spawn_league_exploiter(model, optimizer, model_config)

        # 5. Log league summary
        num_converged = sum(1 for a in league_manager.main_exploiters if a.is_converged)
        num_converged += sum(
            1 for a in league_manager.league_exploiters if a.is_converged
        )

        league_tracker.log_league_summary(
            iteration,
            len(league_manager.main_agents),
            len(league_manager.main_exploiters),
            len(league_manager.league_exploiters),
            num_converged,
        )

        # Log ELO leaderboard periodically
        if (iteration + 1) % 10 == 0:
            league_tracker.log_elo_leaderboard(iteration)

        # 6. Periodic evaluation against engine
        if (iteration + 1) % args.eval_interval == 0:
            print(f"\n{'='*60}")
            print(f"Engine Evaluation (Iteration {iteration + 1})")
            print(f"{'='*60}")

            main_agent = league_manager.current_main_agent
            checkpoint = torch.load(
                main_agent.model_path,
                map_location=config.train_device,
                weights_only=False,
            )
            eval_model_type = checkpoint.get("model_type", "policy")
            model = create_model_by_type(
                eval_model_type, checkpoint.get("config", {})
            ).to(config.train_device)

            # Handle torch.compile() state dict keys
            state_dict = checkpoint["model_state_dict"]
            if any(key.startswith("_orig_mod.") for key in state_dict.keys()):
                state_dict = {
                    k.replace("_orig_mod.", ""): v for k, v in state_dict.items()
                }
            model.load_state_dict(state_dict)

            # Ensure model is on correct device after loading state dict
            model = model.to(config.train_device)

            evaluate_and_update_elo(
                model,
                main_agent.name,
                league_tracker.elo_tracker,
                engine_depths=args.eval_depths,
                games_per_depth=args.eval_games,
                device=config.train_device,
                tensorboard_writer=league_tracker.writer,
                iteration=iteration,
            )

            # Log ELO leaderboard to TensorBoard
            league_tracker.log_elo_leaderboard(iteration)

            # Show current leaderboard
            print("\nCurrent ELO Leaderboard:")
            leaderboard = [
                (player, rating)
                for player, rating in league_tracker.elo_tracker.get_leaderboard(10)
                if player in league_tracker.agent_elo_history
                and len(league_tracker.agent_elo_history[player]) > 0
            ]
            for rank, (player, rating) in enumerate(leaderboard, 1):
                print(f"  {rank}. {player}: {rating:.1f}")

            # Simulate matches between latest batch of models and other leaderboard models
            # Only include agents that have played evaluation games
            leaderboard_agents = [
                agent
                for agent in league_manager.get_all_agents()
                if agent.name in league_tracker.agent_elo_history
                and len(league_tracker.agent_elo_history[agent.name]) > 0
            ]
            # Latest batch: current main agent and any exploiters spawned this iteration
            latest_agents = []
            if (
                league_manager.current_main_agent
                and league_manager.current_main_agent.name
                in league_tracker.agent_elo_history
            ):
                latest_agents.append(league_manager.current_main_agent)
            if (
                league_manager.current_main_exploiter
                and league_manager.current_main_exploiter.name
                in league_tracker.agent_elo_history
            ):
                latest_agents.append(league_manager.current_main_exploiter)
            if (
                league_manager.current_league_exploiter
                and league_manager.current_league_exploiter.name
                in league_tracker.agent_elo_history
            ):
                latest_agents.append(league_manager.current_league_exploiter)

            # For each pair (latest_agent, leaderboard_agent) where names differ, simulate matches
            for agent_a in latest_agents:
                for agent_b in leaderboard_agents:
                    if agent_a.name == agent_b.name:
                        continue
                    print(
                        f"\nSimulating evaluation matches: {agent_a.name} vs {agent_b.name}"
                    )
                    results = league_tracker.evaluate_head_to_head(
                        agent_a,
                        agent_b,
                        num_games=args.eval_games,
                        device=config.train_device,
                        iteration=iteration,
                        inference_batch_size=config.inference_batch_size,
                    )
                    print(
                        f"  {agent_a.name} win rate: {results['agent1_win_rate']:.2%}, "
                        f"{agent_b.name} win rate: {results['agent2_win_rate']:.2%}, "
                        f"draw rate: {results['draw_rate']:.2%}"
                    )

    # Final summary
    print(f"\n{'='*80}")
    print("LEAGUE TRAINING COMPLETE!")
    print(f"{'='*80}")

    print(f"\nFinal League Composition:")
    print(f"  Main Agents: {len(league_manager.main_agents)}")
    print(f"  Main Exploiters: {len(league_manager.main_exploiters)}")
    print(f"  League Exploiters: {len(league_manager.league_exploiters)}")
    print(
        f"  Converged Exploiters: {sum(1 for a in league_manager.main_exploiters if a.is_converged)}"
    )

    print(f"\nBest Main Agent: {league_manager.current_main_agent.name}")
    print(f"  Games played: {league_manager.current_main_agent.games_played}")
    print(f"  Win rate: {league_manager.current_main_agent.win_rate:.2%}")

    # Close trackers
    league_tracker.close()

    print(f"\nLeague saved to: {save_dir}")
    print(f"TensorBoard logs: {save_dir / 'logs'}")


if __name__ == "__main__":
    main()
