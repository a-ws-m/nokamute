"""
Demo script showing the difference between traditional and branching MCMC.
"""

import time
from self_play import SelfPlayGame
from model import create_model
import torch

def demo_comparison():
    """
    Compare traditional vs. branching game generation.
    """
    print("=" * 70)
    print("BRANCHING MCMC DEMO")
    print("=" * 70)
    
    # Create a simple model for demonstration
    print("\nCreating a simple model...")
    model_config = {"hidden_dim": 64, "num_layers": 2}
    model = create_model(model_config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Device: {device}")
    
    num_games = 50
    
    # Test 1: Traditional sequential generation
    print("\n" + "=" * 70)
    print("TEST 1: Traditional Sequential Generation")
    print("=" * 70)
    
    player_traditional = SelfPlayGame(
        model=model,
        temperature=1.0,
        device=device,
        enable_branching=False
    )
    
    start_time = time.time()
    games_traditional = player_traditional.generate_games(num_games)
    time_traditional = time.time() - start_time
    
    print(f"\nGenerated {len(games_traditional)} games in {time_traditional:.2f}s")
    print(f"Average: {time_traditional/num_games:.3f}s per game")
    
    # Calculate statistics
    total_positions = sum(len(game_data) for game_data, _ in games_traditional)
    avg_game_length = total_positions / num_games
    print(f"Total positions evaluated: {total_positions}")
    print(f"Average game length: {avg_game_length:.1f} moves")
    
    # Test 2: Branching MCMC generation
    print("\n" + "=" * 70)
    print("TEST 2: Branching MCMC Generation")
    print("=" * 70)
    
    player_branching = SelfPlayGame(
        model=model,
        temperature=1.0,
        device=device,
        enable_branching=True
    )
    
    start_time = time.time()
    games_branching = player_branching.generate_games_with_branching(
        num_games=num_games,
        branch_ratio=0.5
    )
    time_branching = time.time() - start_time
    
    print(f"\nGenerated {len(games_branching)} games in {time_branching:.2f}s")
    print(f"Average: {time_branching/num_games:.3f}s per game")
    
    # Calculate statistics
    total_positions_br = sum(len(game_data) for game_data, _ in games_branching)
    avg_game_length_br = total_positions_br / num_games
    print(f"Total positions evaluated: {total_positions_br}")
    print(f"Average game length: {avg_game_length_br:.1f} moves")
    
    # Branch point statistics
    print(f"\nBranch point statistics:")
    print(f"  Unique positions in tree: {len(player_branching.game_tree)}")
    print(f"  Branch points collected: {len(player_branching.branch_points)}")
    if len(player_branching.game_tree) > 0:
        print(f"  Branch ratio: {len(player_branching.branch_points) / len(player_branching.game_tree):.2%}")
    
    # Show some example branch points
    if player_branching.branch_points:
        print(f"\nExample branch points (showing first 5):")
        for i, (board, node, depth) in enumerate(player_branching.branch_points[:5]):
            moves = list(node.move_probs.keys())
            top_probs = sorted(node.move_probs.values(), reverse=True)[:3]
            print(f"  {i+1}. Depth {depth}, {len(moves)} moves, "
                  f"top probs: {', '.join(f'{p:.2%}' for p in top_probs)}, "
                  f"visits: {node.visit_count}")
    
    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    
    speedup = time_traditional / time_branching
    print(f"Time speedup: {speedup:.2f}x")
    
    if speedup > 1.0:
        print(f"✓ Branching is {speedup:.2f}x faster!")
    else:
        print(f"✗ Branching is slower (might improve with more games)")
    
    # Estimate computational savings
    # In branching mode, early positions are reused
    positions_from_scratch = len(games_branching) * avg_game_length_br
    actual_positions = total_positions_br
    
    print(f"\nComputational efficiency:")
    print(f"  Positions if all from scratch: {positions_from_scratch:.0f}")
    print(f"  Actual positions generated: {actual_positions}")
    if positions_from_scratch > 0:
        efficiency = actual_positions / positions_from_scratch
        print(f"  Efficiency ratio: {efficiency:.2f} ({(1-efficiency)*100:.1f}% savings)")
    
    # Game outcome distribution
    print("\n" + "=" * 70)
    print("GAME OUTCOMES")
    print("=" * 70)
    
    def analyze_outcomes(games, label):
        white_wins = sum(1 for _, result in games if result > 0.5)
        black_wins = sum(1 for _, result in games if result < -0.5)
        draws = sum(1 for _, result in games if -0.5 <= result <= 0.5)
        
        print(f"\n{label}:")
        print(f"  White wins: {white_wins} ({white_wins/len(games)*100:.1f}%)")
        print(f"  Black wins: {black_wins} ({black_wins/len(games)*100:.1f}%)")
        print(f"  Draws: {draws} ({draws/len(games)*100:.1f}%)")
    
    analyze_outcomes(games_traditional, "Traditional")
    analyze_outcomes(games_branching, "Branching")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)


def demo_branch_point_selection():
    """
    Demonstrate how branch points are selected.
    """
    print("\n" + "=" * 70)
    print("BRANCH POINT SELECTION DEMO")
    print("=" * 70)
    
    player = SelfPlayGame(
        model=None,  # Random play for simplicity
        temperature=1.0,
        device="cpu",
        enable_branching=True
    )
    
    # Generate a few games to collect branch points
    print("\nGenerating games to collect branch points...")
    games = player.generate_games_with_branching(num_games=20, branch_ratio=0.0)
    
    print(f"\nCollected {len(player.branch_points)} branch points")
    
    # Show the distribution of branch points by depth
    from collections import Counter
    depth_counts = Counter(depth for _, _, depth in player.branch_points)
    
    print("\nBranch points by depth:")
    for depth in sorted(depth_counts.keys())[:10]:
        count = depth_counts[depth]
        bar = "█" * (count // 2)
        print(f"  Depth {depth:2d}: {bar} ({count})")
    
    # Show entropy distribution
    print("\nBranch points by entropy:")
    import numpy as np
    
    entropies = []
    for _, node, _ in player.branch_points:
        probs = list(node.move_probs.values())
        if len(probs) > 1:
            entropy = -sum(p * np.log(p + 1e-10) for p in probs)
            entropies.append(entropy)
    
    if entropies:
        print(f"  Min entropy: {min(entropies):.3f}")
        print(f"  Max entropy: {max(entropies):.3f}")
        print(f"  Mean entropy: {np.mean(entropies):.3f}")
        print(f"  Median entropy: {np.median(entropies):.3f}")


if __name__ == "__main__":
    print("Branching MCMC Demonstration\n")
    
    # Run comparison demo
    demo_comparison()
    
    # Run branch point selection demo
    demo_branch_point_selection()
    
    print("\nFor more details, see BRANCHING_MCMC.md")
