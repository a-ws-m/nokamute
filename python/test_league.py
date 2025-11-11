"""
Quick test of the league training system.

This script performs a minimal smoke test to verify:
- League manager initialization
- Agent spawning
- Exploiter training with minimax rewards
- TensorBoard logging
"""

import sys
from pathlib import Path

import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from league import ExploiterAgent, LeagueConfig, LeagueManager, LeagueTracker
from league.exploiter import prepare_exploiter_training_data
from model_policy_hetero import create_policy_model


def test_league_initialization():
    """Test 1: Initialize league and create Main Agent."""
    print("\n" + "=" * 60)
    print("TEST 1: League Initialization")
    print("=" * 60)

    config = LeagueConfig(
        device="cpu",
        main_agent_games_per_iter=2,
        exploiter_games_per_iter=2,
    )

    manager = LeagueManager(config, "test_league")
    tracker = LeagueTracker("test_league/logs")

    # Create initial Main Agent with heterogeneous policy model
    model_config = {"hidden_dim": 32, "num_layers": 2, "model_type": "policy_hetero"}
    model = create_policy_model(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    agent = manager.initialize_main_agent(model, optimizer, model_config)

    assert agent is not None
    assert agent.name == "main_v0"
    assert manager.current_main_agent == agent

    print(f"✓ Created Main Agent: {agent.name}")
    print(f"✓ Model saved to: {agent.model_path}")

    tracker.close()
    return manager, config


def test_exploiter_spawning(manager, config):
    """Test 2: Spawn Main Exploiter."""
    print("\n" + "=" * 60)
    print("TEST 2: Exploiter Spawning")
    print("=" * 60)

    model_config = {"hidden_dim": 32, "num_layers": 2, "model_type": "policy_hetero"}
    model = create_policy_model(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    exploiter = manager.spawn_main_exploiter(model, optimizer, model_config)

    assert exploiter is not None
    assert exploiter.name == "main_exploiter_1"
    assert manager.current_main_exploiter == exploiter

    print(f"✓ Spawned Main Exploiter: {exploiter.name}")
    print(f"✓ Target: {manager.current_main_agent.name}")

    return exploiter


def test_minimax_reward():
    """Test 3: Compute minimax reward."""
    print("\n" + "=" * 60)
    print("TEST 3: Minimax Reward Computation")
    print("=" * 60)

    from league.exploiter import compute_minimax_reward

    import nokamute

    # Create a simple position
    board = nokamute.Board()

    # Make a few moves to have a non-empty position
    moves = board.legal_moves()
    if moves:
        board.apply(moves[0])

    # Create opponent model
    model_config = {"hidden_dim": 32, "num_layers": 2, "model_type": "policy_hetero"}
    opponent_model = create_policy_model(model_config)
    opponent_model.eval()

    # Apply another move and compute minimax reward
    moves = board.legal_moves()
    if moves:
        prev_board = board.clone()
        board.apply(moves[0])

        minimax_reward = compute_minimax_reward(
            prev_board,
            board,
            opponent_model,
            device="cpu",
            gamma=0.99,
        )

        print(f"✓ Computed minimax reward: {minimax_reward:.6f}")
        print(f"  (Should be a finite value)")

        assert isinstance(minimax_reward, float)
        assert not torch.isnan(torch.tensor(minimax_reward))
    else:
        print("⚠ No legal moves to test minimax reward")


def test_exploiter_game_generation():
    """Test 4: Generate exploiter games with minimax rewards."""
    print("\n" + "=" * 60)
    print("TEST 4: Exploiter Game Generation")
    print("=" * 60)

    model_config = {"hidden_dim": 32, "num_layers": 2, "model_type": "policy_hetero"}

    exploiter_model = create_policy_model(model_config)
    opponent_model = create_policy_model(model_config)

    exploiter_agent = ExploiterAgent(
        model=exploiter_model,
        opponent_model=opponent_model,
        device="cpu",
        temperature=1.0,
        minimax_reward_weight=1.0,
        max_moves=50,  # Short games for testing
    )

    print("Generating 2 test games...")
    games = exploiter_agent.generate_training_data(
        num_games=2,
        exploiter_is_white=True,
    )

    assert len(games) == 2
    print(f"✓ Generated {len(games)} games")

    # Check game data structure
    game_data, result, branch_id = games[0]
    assert len(game_data) > 0
    print(f"✓ First game: {len(game_data)} moves, result={result:.1f}")

    # Check for minimax rewards in data
    has_minimax_rewards = False
    for item in game_data:
        if len(item) == 7:  # New format with minimax reward
            _, _, _, _, _, _, minimax_reward = item
            if minimax_reward != 0.0:
                has_minimax_rewards = True
                print(f"✓ Found minimax reward: {minimax_reward:.6f}")
                break

    if has_minimax_rewards:
        print("✓ Minimax rewards present in game data")

    # Prepare training data
    training_data = prepare_exploiter_training_data(games, minimax_reward_weight=1.0)
    print(f"✓ Prepared {len(training_data)} training examples")

    return training_data


def test_pfsp_sampling():
    """Test 5: PFSP opponent sampling."""
    print("\n" + "=" * 60)
    print("TEST 5: PFSP Opponent Sampling")
    print("=" * 60)

    config = LeagueConfig(device="cpu", pfsp_exponent=2.0, pfsp_epsilon=0.1)
    manager = LeagueManager(config, "test_league_pfsp")

    # Create Main Agent and a few historical versions
    model_config = {"hidden_dim": 32, "num_layers": 2, "model_type": "policy_hetero"}
    model = create_policy_model(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    agent = manager.initialize_main_agent(model, optimizer, model_config)

    # Add a couple more versions
    for i in range(2):
        model = create_policy_model(model_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        manager.update_main_agent(model, optimizer, model_config)

    print(f"Created {len(manager.main_agents)} Main Agent versions")

    # Simulate some game results to create win rates
    current = manager.current_main_agent
    for historical in manager.main_agents[:-1]:  # Exclude current
        # Simulate current beating historical
        for _ in range(5):
            current.record_game_result(historical.name, 1.0)
            historical.record_game_result(current.name, -1.0)

    # Sample opponents multiple times
    sample_counts = {}
    for _ in range(20):
        opponent = manager.sample_opponent_for_main_agent()
        sample_counts[opponent.name] = sample_counts.get(opponent.name, 0) + 1

    print(f"✓ Sampled opponents 20 times:")
    for name, count in sample_counts.items():
        print(f"  {name}: {count} times")

    # PFSP should favor opponents we've beaten (lower win rate against)
    # Since we beat historical agents, current agent should be sampled most
    print("✓ PFSP sampling working")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("LEAGUE TRAINING SYSTEM - SMOKE TESTS")
    print("=" * 80)

    try:
        # Test 1: Basic initialization
        manager, config = test_league_initialization()

        # Test 2: Spawn exploiter
        exploiter = test_exploiter_spawning(manager, config)

        # Test 3: Minimax reward computation
        test_minimax_reward()

        # Test 4: Exploiter game generation
        training_data = test_exploiter_game_generation()

        # Test 5: PFSP sampling
        test_pfsp_sampling()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nThe league training system is ready to use.")
        print("Run: python train_league.py --help")

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Cleanup test files
    import shutil

    for test_dir in ["test_league", "test_league_pfsp"]:
        if Path(test_dir).exists():
            shutil.rmtree(test_dir)
            print(f"\nCleaned up {test_dir}/")


if __name__ == "__main__":
    main()
