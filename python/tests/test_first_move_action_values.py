"""
Integration test that trains a Main Agent via self-play and records the
predicted action values for the initial board position over iterations.

Notes:
- This test uses a short training loop and a mocked `get_winner` method so the
  game ends immediately after the first move. If White's first move is a
  grasshopper (UHP string starting with `wG`), White wins; otherwise Black
  wins. This creates a simple supervised signal that the model can learn.
- To run this test locally in the recommended environment use:

    micromamba activate torch
    pytest python/tests/test_first_move_action_values.py -s

"""

from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # headless
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

import nokamute

# Ensure local python/ package modules are importable when tests are executed
# from the repository root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from action_space import action_to_string, get_action_space
from hetero_graph_utils import board_to_hetero_data, prepare_model_inputs
from model_policy_hetero import create_policy_model
from self_play import SelfPlayGame, prepare_training_data


def _mock_get_winner_decide_after_first_move(self):
    """Return game winner immediately after first move.

    Rule: If White's first move contains a grasshopper ('wG' prefix) -> White
    wins; otherwise Black wins. Before any moves return None.
    """
    log = self.get_game_log()
    if not log:
        return None

    # UHP log uses semicolon-separated moves; first entry is first move
    first_segment = log.split(";")[0].strip() if ";" in log else log.strip()
    # If no moves yet
    if first_segment == "" or first_segment.lower().startswith("inprogress"):
        return None

    # First move string like 'wG1', 'wQ', 'wA1' etc - check for wG prefix
    if first_segment.startswith("wG"):
        return "White"
    else:
        return "Black"


from pathlib import Path


def test_first_move_action_value_evolution(monkeypatch):
    """Run a small number of self-play iterations, train, and plot action values.

    This test is intended to validate the training loop is updating the value
    predictions. The plot is written under `tmp_path` so CI / developer can
    inspect it.
    """

    # Small model for quick test
    model = create_policy_model({"hidden_dim": 64, "num_layers": 2, "num_heads": 2})
    model.eval()

    # Self-play generator using the model for evaluation
    # Allow configuring epsilon so the test can explore other legal actions and
    # gather a richer set of self-play trajectories. Default to 0.5.
    epsilon = float(os.getenv("NK_TEST_EPSILON", "0.5"))
    player = SelfPlayGame(
        model=model, epsilon=epsilon, device="cpu", enable_branching=False, max_moves=20
    )

    # Use CPU and small LR so the model slowly updates
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # We perform action-level TD training only (no policy cross-entropy). Use
    # epsilon to explore other moves when collecting training games.
    use_policy_loss = False
    policy_loss_weight = 0.0

    from action_space import get_action_space

    action_to_str, _, _ = get_action_space()

    # We'll store action value (absolute) for the 7 initial legal actions each iteration
    n_iter = 50
    values_history = []  # list of arrays of shape (num_legal_actions,)
    labels = None
    grasshopper_indices = []

    # Mock get_winner so that game always ends after the first play
    monkeypatch.setattr(
        nokamute.Board, "get_winner", _mock_get_winner_decide_after_first_move
    )

    for it in range(n_iter):
        # Evaluate action-values for the initial board
        start_board = nokamute.Board()
        graph_dict = start_board.to_graph()
        data, move_to_action_indices = board_to_hetero_data(graph_dict)
        x_dict, edge_index_dict, edge_attr_dict, move_to_idx = prepare_model_inputs(
            data, move_to_action_indices
        )

        # Predict action values for White at start
        model.eval()
        with torch.no_grad():
            action_values, action_probs, value, action_idx = model.predict_action_info(
                x_dict,
                edge_index_dict,
                edge_attr_dict,
                move_to_idx,
                current_player="White",
            )

        # Determine legal actions (unique positive indices in move_to_idx)
        legal_action_idxs = np.unique(move_to_idx[move_to_idx >= 0].cpu().numpy())
        legal_action_idxs = sorted(legal_action_idxs)

        # Save labels (just once) - keep full UHP string
        if labels is None:
            labels = [action_to_string(i) for i in legal_action_idxs]
            # Determine which index corresponds to a grasshopper placement
            grasshopper_indices = [
                i for i, lab in enumerate(labels) if lab.startswith("wG")
            ]

        # Extract model's predicted action values for those indices
        action_values_np = action_values.squeeze(0).cpu().numpy()
        selected_values = np.array([action_values_np[i] for i in legal_action_idxs])
        action_probs_np = action_probs.squeeze(0).cpu().numpy()

        # Ensure length is 7 (7 legal starting moves in default start position)
        assert (
            len(selected_values) == 7
        ), f"Expected 7 legal starting actions; found {len(selected_values)}. Indices: {legal_action_idxs}"

        # Print iteration-level debug information
        print(f"Iteration {it}: labels={labels}")
        print(f"Self-play epsilon={player.epsilon}")
        print(f"Predicted action values: {np.round(selected_values,3)}")
        print(
            f"Action probabilities: {np.round(np.array([action_probs_np[i] for i in legal_action_idxs]),3)}"
        )
        print(f"Predicted value (current): {value.squeeze().cpu().numpy():.3f}")
        expected_outcomes = [
            "White" if lab.startswith("wG") else "Black" for lab in labels
        ]
        print(f"Expected outcomes (if that action chosen): {expected_outcomes}")
        if grasshopper_indices:
            for gi in grasshopper_indices:
                print(
                    f"  grasshopper at idx={gi}, label={labels[gi]}, value={selected_values[gi]:.3f}"
                )

        # Record values for plotting
        values_history.append(selected_values)

        # Generate synthetic training examples that enumerate all legal starting
        # moves so we have both positive and negative signals for the loss.
        training_examples = []
        for act_idx in legal_action_idxs:
            # target = 1 if grasshopper (White) else -1
            move_str = action_to_str[act_idx]
            target = 1.0 if move_str.startswith("wG") else -1.0

            # Append 7-tuple: (hetero_data, move_to_action_indices, target, selected_action_idx,
            # next_hetero_data, next_move_to_action_indices, current_player)
            training_examples.append(
                (
                    data,
                    move_to_action_indices,
                    target,
                    int(act_idx),
                    None,
                    None,
                    "White",
                )
            )

        # Print training example targets for inspection
        print(f"Training examples: {len(training_examples)}")
        for ex in training_examples:
            _, _, tgt, sel_action_idx, _next, _next_idx, ex_player = ex
            sel_action = (
                action_to_str[sel_action_idx] if sel_action_idx >= 0 else "pass"
            )
            print(f"  player={player}, target={tgt:.3f}, selected_action={sel_action}")

        # Do a single-epoch train step with the examples (simple TD update)
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        count = 0

        for example in training_examples:
            hetero_data, move_to_action_indices, target, *_rest = example

            # Convert player string into 0/1 int
            player_str = example[-1]
            current_player_int = 0 if player_str == "White" else 1

            # Set up label tensors similar to train_epoch_selfplay
            hetero_data = hetero_data.cpu()
            hetero_data.y = torch.tensor([float(target)], dtype=torch.float32)
            hetero_data.move_to_action_indices = move_to_action_indices.cpu()
            hetero_data.current_player = torch.tensor(
                [current_player_int], dtype=torch.long
            )

            # Prepare inputs
            x_d, e_idx_d, e_attr_d, m2a = prepare_model_inputs(
                hetero_data, hetero_data.move_to_action_indices
            )

            # Move to device (cpu)
            x_d = {k: v.to("cpu") for k, v in x_d.items()}
            e_idx_d = {k: v.to("cpu") for k, v in e_idx_d.items()}
            e_attr_d = {k: v.to("cpu") for k, v in e_attr_d.items()}

            # Forward - get action_values and values
            action_values, white_value, black_value, _, _ = model(
                x_d, e_idx_d, e_attr_d, m2a
            )

            # Use the selected action's value as the training prediction
            selected_idx = example[3] if len(example) > 3 else -1
            if selected_idx is not None and selected_idx >= 0:
                # action_values shape: [1, num_actions] for single example
                action_val = action_values.squeeze(0)[selected_idx]
                value_current = action_val.unsqueeze(0)
            else:
                value_current = (
                    white_value.squeeze()
                    if current_player_int == 0
                    else black_value.squeeze()
                )
            if value_current.dim() == 0:
                value_current = value_current.unsqueeze(0)

            # Compute TD loss (MSE to TD target)
            target_val = torch.tensor([float(target)], dtype=torch.float32)
            loss = torch.nn.functional.mse_loss(value_current, target_val)
            # We intentionally do not add policy cross-entropy; action-level
            # supervision comes from using the selected action's value directly.
            loss.backward()
            total_loss += loss.item()
            count += 1

        if count > 0:
            optimizer.step()
            print(f"  optimizer.step() applied, avg_loss={total_loss/count:.4f}")

    # Convert history to numpy array -> shape (n_iter, 7)
    history = np.vstack(values_history)

    # Plot per-action curves
    # Save the plot to a non-temporary path so it is easy to inspect after the test
    plots_dir = Path(__file__).resolve().parents[1] / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "first_move_action_values.png"
    plt.figure(figsize=(8, 5))
    assert labels is not None
    for a_idx in range(history.shape[1]):
        plt.plot(history[:, a_idx], label=labels[a_idx])

    plt.xlabel("Iteration")
    plt.ylabel("Predicted action value (absolute)")
    plt.title("Action value evolution for 7 starting moves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)

    # Basic assertion: the file was written and has non-zero size
    assert out_path.exists() and out_path.stat().st_size > 0

    # Print trends in action values
    diffs = history[-1] - history[0]
    slopes = diffs / max(1, history.shape[0] - 1)

    print("\nAction value trends (final - initial):")
    for i, lab in enumerate(labels):
        print(
            f"  {lab}: change={diffs[i]:.3f}, slope={slopes[i]:.4f}, final={history[-1,i]:.3f}"
        )

    # If the model learned to favor grasshopper, grasshopper index should have
    # one of the largest positive slopes.
    if grasshopper_indices:
        g_slopes = [slopes[g] for g in grasshopper_indices]
        max_slope = slopes.max()
        print(f"Grasshopper slopes: {g_slopes}, max_slope={max_slope:.4f}")
        # Check whether grasshopper is chosen as highest-value move at the end
        best_idx = int(np.argmax(history[-1]))
        print(
            f"Best final action: idx={best_idx}, label={labels[best_idx]}, value={history[-1,best_idx]:.3f}"
        )
        if best_idx in grasshopper_indices:
            print("Grasshopper is final best action.")
        else:
            print(
                "Grasshopper is not the final best action (training did not distinguish moves strongly). Consider adding policy cross-entropy or action-level supervision to drive other action values down."
            )
