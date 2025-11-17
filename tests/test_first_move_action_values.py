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
    pytest tests/test_first_move_action_values.py -q

"""

from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")  # headless
import os
import sys

# Ensure local python/ package modules are importable when tests are executed
# from the repository root -- add `python` directory to sys.path (not the
# tests dir). This makes imports like `from action_space import ...` work when
# tests live under `tests/` instead of `python/tests/`.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "python"))
)

import matplotlib.pyplot as plt
import numpy as np
import torch
from action_space import action_to_string, get_action_space
from hetero_graph_utils import board_to_hetero_data, prepare_model_inputs
from league.config import LeagueConfig
from league.manager import LeagueManager
from league.tracker import LeagueTracker
from model_policy_hetero import create_policy_model
from self_play import SelfPlayGame, prepare_training_data
from train_league import train_main_agent

import nokamute

# (rest of file unchanged - other helpers and tests)


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
    predictions. The plot is written under `python/plots` so CI / developer can
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
    plots_dir = Path(__file__).resolve().parents[1] / "python" / "plots"
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
    # Explicit assertion: at the end of training, only grasshopper moves should have positive values
    if grasshopper_indices:
        final_vals = history[-1]
        pos_idxs = [i for i, v in enumerate(final_vals) if v > 0]
        assert set(pos_idxs) == set(
            grasshopper_indices
        ), f"Only grasshopper moves should be positive; positives={pos_idxs}, grasshopper={grasshopper_indices}"


# Keep the helper functions unchanged below, but ensure that the plots path is also
# updated to the `python/plots` directory when used from this new top-level
# `tests/` path.


def _initial_board_action_values_for_model(model):
    """Return (selected_values, labels, grasshopper_indices, action_probs, value)

    selected_values: numpy array of action-values (absolute) for the legal starting
      positions (sorted order)
    labels: list of UHP strings for each selected action
    grasshopper_indices: indices of labels that start with 'wG'
    """

    start_board = nokamute.Board()
    graph_dict = start_board.to_graph()
    data, move_to_action_indices = board_to_hetero_data(graph_dict)
    x_dict, edge_index_dict, edge_attr_dict, move_to_idx = prepare_model_inputs(
        data, move_to_action_indices
    )

    model.eval()
    with torch.no_grad():
        action_values, action_probs, value, action_idx = model.predict_action_info(
            x_dict, edge_index_dict, edge_attr_dict, move_to_idx, current_player="White"
        )

    # Determine legal actions
    legal_action_idxs = np.unique(move_to_idx[move_to_idx >= 0].cpu().numpy())
    legal_action_idxs = sorted(legal_action_idxs)

    action_values_np = action_values.squeeze(0).cpu().numpy()
    selected_values = np.array([action_values_np[i] for i in legal_action_idxs])
    action_probs_np = action_probs.squeeze(0).cpu().numpy()

    labels = [action_to_string(i) for i in legal_action_idxs]
    grasshopper_indices = [i for i, lab in enumerate(labels) if lab.startswith("wG")]

    return selected_values, labels, grasshopper_indices, action_probs_np, value


def _plot_action_value_history(history, labels, out_path):
    plt.figure(figsize=(8, 5))
    for a_idx in range(history.shape[1]):
        plt.plot(history[:, a_idx], label=labels[a_idx])

    plt.xlabel("Iteration")
    plt.ylabel("Predicted action value (absolute)")
    plt.title("Action value evolution for 7 starting moves")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)


def test_first_move_action_value_evolution_with_train_main_agent(monkeypatch, tmp_path):
    """Train via `train_main_agent` and track first move values over time.

    Use a custom `get_winner` rule: if White's first move is a grasshopper (wG)
    AND White has played a second move, White wins; otherwise Black wins. This
    creates a two-move dependency that can only be learned when we actually
    train the Main Agent with `train_main_agent()`.
    """

    # Small model for quick test
    model = create_policy_model({"hidden_dim": 64, "num_layers": 2, "num_heads": 2})
    model.eval()

    # Monkeypatch get_winner to require grasshopper then any other move (White) to win
    def _mock_get_winner_grasshopper_first(self):
        log = self.get_game_log()
        if not log:
            return None

        segments = [s.strip() for s in log.split(";") if s.strip() != ""]
        if not segments or len(segments) < 3:
            # Not enough moves yet
            return None

        # If first move isn't a White grasshopper, White cannot win
        first_segment = segments[0]
        if first_segment.startswith("wG"):
            return "White"
        else:
            return "Black"

    monkeypatch.setattr(
        nokamute.Board, "get_winner", _mock_get_winner_grasshopper_first
    )

    # Debug: verify monkeypatch is working
    # Apply moves: wG1, bA1, bQ
    winning_moves = ["wG1", "bA1 -wG1", "wQ wG1-"]
    losing_moves = ["wA1", "bA1 -wA1", "wQ wA1-"]
    for move_set in (winning_moves, losing_moves):
        test_board = nokamute.Board()
        for move_str in move_set:
            turn = test_board.parse_move(move_str)
            test_board.apply(turn)
        log = test_board.get_game_log()
        winner = test_board.get_winner()
        print(f"[DEBUG] Monkeypatch test: log={log}, winner={winner}")

    # Counter for move sequences and results
    from collections import Counter

    move_counter = Counter()
    result_counter = Counter()

    # Build small league and tracker in a temp directory
    config = LeagueConfig()
    # Increase epsilon to encourage exploration so we see non-greedy moves
    config.epsilon = 1.0
    # Small numbers for quicker tests
    config.main_agent_games_per_iter = 20
    config.main_agent_epochs = 5
    config.main_agent_batch_size = 8
    # Ensure CPU for CI / developer machines
    config.gen_device = "cpu"
    config.train_device = "cpu"

    league_dir = tmp_path / "league"
    league_dir.mkdir(parents=True, exist_ok=True)

    league_manager = LeagueManager(config, str(league_dir))
    tracker = LeagueTracker(str(league_dir / "logs"))

    # Initialize main agent
    model_config = {"hidden_dim": 64, "num_layers": 2, "num_heads": 2}
    optimizer = torch.optim.Adam(model.parameters(), lr=config.main_agent_lr)
    league_manager.initialize_main_agent(model, optimizer, model_config)

    n_iter = 5
    values_history = []
    # We'll save full per-iteration action-values for the first moves
    full_values_history = []
    labels = None
    grasshopper_indices = []

    for it in range(n_iter):
        # Evaluate current main agent's predicted start position values
        # Load current main model
        ckpt = torch.load(
            league_manager.current_main_agent.model_path, map_location="cpu"
        )
        # Load the model using LeagueTracker helper so saved config/state match
        # exactly and we avoid size mismatches.
        assert league_manager.current_main_agent is not None
        cur_model = tracker._load_model(league_manager.current_main_agent, device="cpu")
        cur_model.eval()

        # Extract starting action values
        selected_values, labels, grasshopper_indices, action_probs, value = (
            _initial_board_action_values_for_model(cur_model)
        )

        # Save only the grasshopper values for monitoring
        if grasshopper_indices:
            # There may be multiple grasshopper placements; take the maximum
            g_vals = selected_values[grasshopper_indices]
            values_history.append(float(g_vals.max()))
        else:
            values_history.append(0.0)

        # Also record full values for assertion at end
        full_values_history.append(selected_values)

        print(
            f"Iteration {it}: grasshopper_idx={grasshopper_indices}, value={values_history[-1]}"
        )

        # Train one iteration of Main Agent
        train_main_agent(league_manager, tracker, config, iteration=it)

        # After training, inspect the games played in this iteration
        # LeagueManager should have a record of games played
        # Try to find the most recent games played by main agent
        main_agent_games_dir = league_dir / "main_agent_games"
        if main_agent_games_dir.exists():
            for game_file in sorted(main_agent_games_dir.glob("*.txt")):
                try:
                    with open(game_file, "r") as f:
                        lines = f.readlines()
                        # Try to extract move sequence and result
                        moves = None
                        result = None
                        for line in lines:
                            if line.startswith("Moves:"):
                                moves = line.strip().split(":", 1)[-1].strip()
                            if line.startswith("Result:"):
                                result = line.strip().split(":", 1)[-1].strip()
                        if moves:
                            move_counter[moves] += 1
                        if result:
                            result_counter[result] += 1
                except Exception as e:
                    print(f"[DEBUG] Error reading {game_file}: {e}")

        # Also, print a sample of games from the tracker logs if available
        logs_dir = league_dir / "logs"
        if logs_dir.exists():
            for log_file in sorted(logs_dir.glob("*.txt"))[-5:]:
                print(f"[DEBUG] Recent log: {log_file}")
                try:
                    with open(log_file, "r") as f:
                        print(f.read())
                except Exception as e:
                    print(f"[DEBUG] Error reading log {log_file}: {e}")

    # Print summary of move sequences and results
    print("\n[DEBUG] Move sequence counts:")
    for move_seq, count in move_counter.most_common():
        print(f"  {move_seq}: {count}")
    print("\n[DEBUG] Result counts:")
    for result, count in result_counter.most_common():
        print(f"  {result}: {count}")

    # Convert history to numpy array -> shape (n_iter,) or (n_iter,1)
    history = np.array(values_history).squeeze()
    full_history = np.vstack(full_values_history)

    # Plot trend
    plots_dir = Path(__file__).resolve().parents[1] / "python" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_path = plots_dir / "first_move_action_values_main_agent.png"

    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.xlabel("Iteration")
    plt.ylabel("Grasshopper predicted action value (approx)")
    plt.title("Grasshopper value trend when learning by train_main_agent()")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)

    # Basic assertion: the file was written and has non-zero size
    assert out_path.exists() and out_path.stat().st_size > 0

    # Explicit assertion: at the end of training, only grasshopper moves should have positive values
    if grasshopper_indices:
        final_vals = full_history[-1]
        pos_idxs = [i for i, v in enumerate(final_vals) if v > 0]
        assert set(pos_idxs) == set(
            grasshopper_indices
        ), f"Only grasshopper moves should be positive; positives={pos_idxs}, grasshopper={grasshopper_indices}"
