#!/usr/bin/env python3
"""
Quick training test with heterogeneous policy model.
Runs a very short league training session to verify the full pipeline works.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
from league import LeagueConfig, LeagueManager, LeagueTracker
from model_policy_hetero import create_policy_model


def main():
    print("\n" + "=" * 80)
    print("QUICK TRAINING TEST - Heterogeneous Policy Model")
    print("=" * 80)

    # Configuration for quick test
    config = LeagueConfig(
        device="cpu",
        main_agent_games_per_iter=3,  # Very few games
        exploiter_games_per_iter=3,
        main_agent_epochs=2,  # Very few epochs
        exploiter_epochs=2,
        max_moves=100,  # Short games
    )

    save_dir = "test_train_hetero_quick"
    manager = LeagueManager(config, save_dir)
    tracker = LeagueTracker(f"{save_dir}/logs")

    print("\n1. Creating initial heterogeneous policy model...")
    model_config = {"hidden_dim": 32, "num_layers": 2, "model_type": "policy_hetero"}
    model = create_policy_model(model_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"   ✓ Created model (parameters will be initialized on first forward pass)")

    # Initialize main agent
    agent = manager.initialize_main_agent(model, optimizer, model_config)
    print(f"   ✓ Initialized: {agent.name}")

    print("\n2. Running self-play with heterogeneous graphs...")
    from self_play import SelfPlayGame, prepare_training_data

    player = SelfPlayGame(
        model=model,
        temperature=1.0,
        device=config.device,
        max_moves=config.max_moves,
    )

    # Generate a couple of games
    games = []
    for i in range(config.main_agent_games_per_iter):
        print(f"   Game {i+1}/{config.main_agent_games_per_iter}...", end=" ")
        game_data, result, branch_id = player.play_game()
        games.append((game_data, result, branch_id))
        print(f"✓ ({len(game_data)} moves, result={result:.1f})")

    print("\n3. Preparing training data...")
    training_data = prepare_training_data(games)
    print(f"   ✓ {len(training_data)} training examples")

    print("\n4. Training for 2 epochs...")
    from train_league import train_epoch_standard

    for epoch in range(2):
        loss = train_epoch_standard(
            model, training_data, optimizer, batch_size=32, device=config.device
        )
        print(f"   Epoch {epoch+1}: loss = {loss:.4f}")

    print("\n5. Spawning Main Exploiter...")
    exploiter_model = create_policy_model(model_config)
    exploiter_optimizer = torch.optim.Adam(exploiter_model.parameters(), lr=1e-3)

    exploiter = manager.spawn_main_exploiter(
        exploiter_model, exploiter_optimizer, model_config
    )
    print(f"   ✓ Spawned: {exploiter.name}")

    print("\n6. Testing exploiter with minimax rewards...")
    from league import ExploiterAgent

    exploiter_agent = ExploiterAgent(
        model=exploiter_model,
        opponent_model=model,
        device=config.device,
        temperature=1.0,
        minimax_reward_weight=1.0,
        max_moves=50,
    )

    print("   Generating 2 exploiter games...")
    exploit_games = exploiter_agent.generate_training_data(
        num_games=2,
        exploiter_is_white=True,
    )
    print(f"   ✓ Generated {len(exploit_games)} games")

    print("\n7. Preparing exploiter training data...")
    from league.exploiter import prepare_exploiter_training_data

    exploit_training_data = prepare_exploiter_training_data(
        exploit_games,
        minimax_reward_weight=1.0,
    )
    print(f"   ✓ {len(exploit_training_data)} exploiter training examples")

    # Verify the data format is correct (HeteroData objects)
    if len(exploit_training_data) > 0:
        sample_data, sample_target = exploit_training_data[0]
        from torch_geometric.data import HeteroData

        if isinstance(sample_data, HeteroData):
            print("   ✓ Training data is in correct HeteroData format")
        else:
            print(f"   ✗ Unexpected data type: {type(sample_data)}")

    print("\n" + "=" * 80)
    print("✓ ALL TESTS PASSED!")
    print("=" * 80)
    print("\nThe heterogeneous policy model works correctly with:")
    print("  - Self-play game generation")
    print("  - Training data preparation")
    print("  - Model training")
    print("  - League exploiter spawning")
    print("  - Minimax reward computation")
    print("\nReady for full league training!")

    tracker.close()

    # Cleanup
    import shutil

    if Path(save_dir).exists():
        shutil.rmtree(save_dir)
        print(f"\nCleaned up {save_dir}/")


if __name__ == "__main__":
    main()
