"""
Demonstration of new features:
1. ELO tracking system
2. Engine move selection from Rust bindings
3. Model evaluation against engine
"""

import nokamute
from elo_tracker import EloTracker
from evaluate_vs_engine import evaluate_model, evaluate_and_update_elo
from model import create_model
import torch


def demo_engine_move():
    """Demonstrate the get_engine_move feature."""
    print("=" * 60)
    print("DEMO 1: Engine Move Selection")
    print("=" * 60)
    
    board = nokamute.Board()
    
    # Play a few moves
    print("\nPlaying some initial moves...")
    board.apply(nokamute.Turn.place(0, nokamute.Bug("Queen")))
    board.apply(nokamute.Turn.place(1, nokamute.Bug("Spider")))
    board.apply(nokamute.Turn.place(255, nokamute.Bug("Ant")))
    
    print(f"Board state: Turn {board.turn_num()}, {board.to_move().name} to move")
    
    # Get engine move with different depths
    for depth in [1, 2, 3]:
        print(f"\nEngine move (depth {depth}):")
        engine_move = board.get_engine_move(depth=depth)
        if engine_move:
            print(f"  {engine_move}")
        else:
            print("  No move available (game over)")
    
    # Test with time limit
    print(f"\nEngine move (time limit 1000ms):")
    engine_move = board.get_engine_move(time_limit_ms=1000)
    if engine_move:
        print(f"  {engine_move}")
    
    print("\n✓ Engine move selection working!")


def demo_elo_tracking():
    """Demonstrate the ELO tracking system."""
    print("\n" + "=" * 60)
    print("DEMO 2: ELO Tracking System")
    print("=" * 60)
    
    # Create a temporary ELO tracker
    tracker = EloTracker(save_path="demo_elo.json")
    
    print("\nSimulating some training iterations...")
    
    # Simulate games from different model iterations
    iterations = [
        ("model_iter_0", "engine_depth_2", 0.0, "Initial model loses"),
        ("model_iter_0", "engine_depth_2", 0.0, "Still losing"),
        ("model_iter_1", "engine_depth_2", 0.5, "Getting better - draw"),
        ("model_iter_2", "engine_depth_2", 0.5, "Another draw"),
        ("model_iter_2", "engine_depth_3", 0.0, "Harder opponent"),
        ("model_iter_3", "engine_depth_2", 1.0, "First win!"),
        ("model_iter_3", "engine_depth_3", 0.5, "Draw against harder opponent"),
        ("model_iter_4", "engine_depth_3", 1.0, "Beating the harder opponent"),
    ]
    
    for model, opponent, score, description in iterations:
        new_rating, _ = tracker.update_ratings(model, opponent, score)
        print(f"  {description}: {model} ELO = {new_rating:.1f}")
    
    print("\nLeaderboard (Top 10):")
    for rank, (player, rating) in enumerate(tracker.get_leaderboard(10), 1):
        print(f"  {rank}. {player}: {rating:.1f}")
    
    best_model = tracker.get_best_model(prefix="model_iter_")
    print(f"\nBest model: {best_model}")
    print(f"Best model ELO: {tracker.get_rating(best_model):.1f}")
    
    print("\nDetailed stats for best model:")
    tracker.print_stats(best_model)
    
    # Save and reload
    tracker.save()
    print(f"\n✓ ELO history saved to {tracker.save_path}")
    
    # Clean up
    import os
    os.remove("demo_elo.json")


def demo_model_evaluation():
    """Demonstrate model evaluation against engine."""
    print("\n" + "=" * 60)
    print("DEMO 3: Model Evaluation Against Engine")
    print("=" * 60)
    
    print("\nCreating a random (untrained) model...")
    model = create_model({"hidden_dim": 64, "num_layers": 2})
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\nEvaluating against engine (depth 2, 4 games)...")
    print("(This will take a minute...)")
    
    results = evaluate_model(
        model,
        engine_depth=2,
        num_games=4,
        device="cpu",
        verbose=True,
    )
    
    print("\n✓ Model evaluation working!")


def demo_full_elo_integration():
    """Demonstrate full ELO integration with model evaluation."""
    print("\n" + "=" * 60)
    print("DEMO 4: Full ELO Integration")
    print("=" * 60)
    
    # Create model and tracker
    model = create_model({"hidden_dim": 64, "num_layers": 2})
    tracker = EloTracker(save_path="demo_elo_full.json")
    
    print("\nEvaluating model against multiple engine depths...")
    print("(Playing 4 games per depth...)")
    
    results = evaluate_and_update_elo(
        model=model,
        model_name="demo_model",
        elo_tracker=tracker,
        engine_depths=[2, 3],
        games_per_depth=4,
        device="cpu",
        verbose=True,
    )
    
    print("\nFinal ELO ratings:")
    for rank, (player, rating) in enumerate(tracker.get_leaderboard(), 1):
        print(f"  {rank}. {player}: {rating:.1f}")
    
    print("\n✓ Full ELO integration working!")
    
    # Clean up
    import os
    os.remove("demo_elo_full.json")


if __name__ == "__main__":
    print("\n🎮 NOKAMUTE ML TRAINING - NEW FEATURES DEMO 🎮\n")
    
    # Run all demos
    demo_engine_move()
    demo_elo_tracking()
    demo_model_evaluation()
    demo_full_elo_integration()
    
    print("\n" + "=" * 60)
    print("✨ ALL DEMOS COMPLETED SUCCESSFULLY! ✨")
    print("=" * 60)
    print("\nNew features are ready to use:")
    print("  1. ✓ ELO tracking system (elo_tracker.py)")
    print("  2. ✓ Engine move selection (board.get_engine_move())")
    print("  3. ✓ Model vs Engine evaluation (evaluate_vs_engine.py)")
    print("\nIntegrated into training with:")
    print("  - train.py --eval-interval N --eval-depths 2 3 4")
    print()
