"""
ELO rating system for tracking ML model performance over time.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class EloTracker:
    """
    Tracks ELO ratings for different model versions and opponents.
    
    The ELO system uses the standard formula:
    - Expected score: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
    - New rating: R'_A = R_A + K * (S_A - E_A)
    
    Where:
    - R_A is player A's rating
    - R_B is player B's rating
    - S_A is actual score (1 for win, 0.5 for draw, 0 for loss)
    - K is the K-factor (determines how quickly ratings change)
    """

    def __init__(self, save_path: str = "checkpoints/elo_history.json", k_factor: int = 32):
        """
        Args:
            save_path: Path to save ELO history
            k_factor: ELO K-factor (higher = more volatile ratings)
        """
        self.save_path = save_path
        self.k_factor = k_factor
        self.ratings: Dict[str, float] = {}
        self.history: List[Dict] = []
        
        # Initialize default ratings
        self.ratings["engine_depth_1"] = 1200
        self.ratings["engine_depth_2"] = 1400
        self.ratings["engine_depth_3"] = 1600
        self.ratings["engine_depth_4"] = 1800
        self.ratings["engine_depth_5"] = 2000
        self.ratings["random"] = 800
        
        # Load existing history if available
        self.load()

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected score for player A against player B.
        
        Args:
            rating_a: Rating of player A
            rating_b: Rating of player B
            
        Returns:
            Expected score (0 to 1)
        """
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def update_ratings(
        self,
        player_a: str,
        player_b: str,
        score_a: float,
        game_metadata: Optional[Dict] = None,
    ) -> Tuple[float, float]:
        """
        Update ELO ratings after a game.
        
        Args:
            player_a: Name/identifier of player A
            player_b: Name/identifier of player B
            score_a: Score for player A (1.0 = win, 0.5 = draw, 0.0 = loss)
            game_metadata: Optional metadata about the game
            
        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        # Initialize ratings if not present
        if player_a not in self.ratings:
            self.ratings[player_a] = 1500  # Default starting rating
        if player_b not in self.ratings:
            self.ratings[player_b] = 1500
            
        rating_a = self.ratings[player_a]
        rating_b = self.ratings[player_b]
        
        # Calculate expected scores
        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = self.expected_score(rating_b, rating_a)
        
        # Calculate new ratings
        new_rating_a = rating_a + self.k_factor * (score_a - expected_a)
        new_rating_b = rating_b + self.k_factor * ((1 - score_a) - expected_b)
        
        # Update ratings
        self.ratings[player_a] = new_rating_a
        self.ratings[player_b] = new_rating_b
        
        # Record history
        record = {
            "timestamp": datetime.now().isoformat(),
            "player_a": player_a,
            "player_b": player_b,
            "score_a": score_a,
            "rating_a_old": rating_a,
            "rating_a_new": new_rating_a,
            "rating_b_old": rating_b,
            "rating_b_new": new_rating_b,
            "expected_a": expected_a,
        }
        
        if game_metadata:
            record["metadata"] = game_metadata
            
        self.history.append(record)
        
        return new_rating_a, new_rating_b

    def get_rating(self, player: str) -> float:
        """Get current rating for a player."""
        return self.ratings.get(player, 1500)

    def get_best_model(self, prefix: str = "model_iter_") -> Optional[str]:
        """
        Get the highest-rated model with a given prefix.
        
        Args:
            prefix: Prefix to filter model names
            
        Returns:
            Name of best model, or None if no models found
        """
        models = {k: v for k, v in self.ratings.items() if k.startswith(prefix)}
        if not models:
            return None
        return max(models.items(), key=lambda x: x[1])[0]

    def get_leaderboard(self, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Get top N players by rating.
        
        Args:
            top_n: Number of top players to return
            
        Returns:
            List of (player_name, rating) tuples
        """
        sorted_ratings = sorted(
            self.ratings.items(), key=lambda x: x[1], reverse=True
        )
        return sorted_ratings[:top_n]

    def save(self):
        """Save ratings and history to file."""
        Path(self.save_path).parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "k_factor": self.k_factor,
            "ratings": self.ratings,
            "history": self.history,
        }
        
        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load ratings and history from file."""
        if not os.path.exists(self.save_path):
            return
            
        with open(self.save_path, "r") as f:
            data = json.load(f)
            
        self.k_factor = data.get("k_factor", self.k_factor)
        self.ratings.update(data.get("ratings", {}))
        self.history = data.get("history", [])

    def print_stats(self, player: str):
        """Print statistics for a player."""
        if player not in self.ratings:
            print(f"No rating found for {player}")
            return
            
        rating = self.ratings[player]
        
        # Get games involving this player
        games = [h for h in self.history if h["player_a"] == player or h["player_b"] == player]
        
        if not games:
            print(f"{player}: Rating = {rating:.1f} (no games played)")
            return
            
        wins = sum(
            1 for h in games 
            if (h["player_a"] == player and h["score_a"] == 1.0) or
               (h["player_b"] == player and h["score_a"] == 0.0)
        )
        draws = sum(
            1 for h in games 
            if h["score_a"] == 0.5
        )
        losses = len(games) - wins - draws
        
        print(f"{player}:")
        print(f"  Rating: {rating:.1f}")
        print(f"  Record: {wins}W-{losses}L-{draws}D ({len(games)} games)")
        
        if len(games) > 1:
            first_rating = None
            for h in self.history:
                if h["player_a"] == player:
                    first_rating = h["rating_a_old"]
                    break
                elif h["player_b"] == player:
                    first_rating = h["rating_b_old"]
                    break
                    
            if first_rating:
                print(f"  Rating change: {rating - first_rating:+.1f}")


if __name__ == "__main__":
    # Test the ELO tracker
    print("Testing ELO tracker...")
    
    tracker = EloTracker(save_path="test_elo.json")
    
    # Simulate some games
    tracker.update_ratings("model_iter_0", "engine_depth_3", 0.0)  # Loss
    tracker.update_ratings("model_iter_0", "engine_depth_3", 0.5)  # Draw
    tracker.update_ratings("model_iter_1", "engine_depth_3", 0.5)  # Draw
    tracker.update_ratings("model_iter_1", "engine_depth_3", 1.0)  # Win
    tracker.update_ratings("model_iter_2", "engine_depth_3", 1.0)  # Win
    tracker.update_ratings("model_iter_2", "engine_depth_4", 0.5)  # Draw
    
    print("\nLeaderboard:")
    for rank, (player, rating) in enumerate(tracker.get_leaderboard(), 1):
        print(f"{rank}. {player}: {rating:.1f}")
    
    print(f"\nBest model: {tracker.get_best_model()}")
    
    print("\nStats for model_iter_2:")
    tracker.print_stats("model_iter_2")
    
    # Clean up test file
    os.remove("test_elo.json")
    print("\nELO tracker test passed!")
