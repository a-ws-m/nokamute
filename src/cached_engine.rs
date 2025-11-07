/// Cached Engine for efficient self-play
///
/// This module provides a stateful engine that maintains a transposition table
/// across multiple move generations, avoiding redundant minimax searches during
/// self-play games. This is particularly useful when generating training data,
/// where we make many sequential moves and the search tree from one position
/// overlaps significantly with the next position.
///
/// Key optimization: When we search ahead from position A and choose move M,
/// we've already evaluated position B (after applying M). When we then call
/// the engine again from position B, we can reuse all the evaluations from
/// the previous search instead of starting from scratch.

use crate::{BasicEvaluator, Board, Rules, Turn};
use minimax::{Game, Strategy};
use std::time::Duration;

/// A persistent engine that caches evaluations across moves
pub struct CachedEngine {
    /// The minimax strategy with a persistent transposition table
    strategy: minimax::IterativeSearch<BasicEvaluator>,
    /// Aggression level for the evaluator
    aggression: u8,
    /// Default search depth
    depth: u8,
    /// Optional time limit in milliseconds
    time_limit_ms: Option<u64>,
}

impl CachedEngine {
    /// Create a new cached engine with the given configuration
    ///
    /// # Arguments
    /// * `aggression` - Aggression level 1-5 for BasicEvaluator (default: 3)
    /// * `depth` - Default search depth for minimax (default: 3)
    /// * `time_limit_ms` - Optional time limit in milliseconds (overrides depth if set)
    /// * `table_size_mb` - Size of transposition table in megabytes (default: 100)
    pub fn new(
        aggression: Option<u8>,
        depth: Option<u8>,
        time_limit_ms: Option<u64>,
        table_size_mb: Option<usize>,
    ) -> Self {
        let aggression = aggression.unwrap_or(3);
        let depth = depth.unwrap_or(3);
        let table_size_mb = table_size_mb.unwrap_or(100);

        let eval = BasicEvaluator::new(aggression);
        let opts = minimax::IterativeOptions::new()
            .with_table_byte_size(table_size_mb << 20);

        let strategy = minimax::IterativeSearch::new(eval, opts);

        Self { strategy, aggression, depth, time_limit_ms }
    }

    /// Get the best move for the current position
    ///
    /// This method uses the cached transposition table from previous searches,
    /// making it much faster for sequential positions in self-play.
    ///
    /// # Arguments
    /// * `board` - The current board position
    /// * `depth` - Optional override for search depth
    /// * `time_limit_ms` - Optional override for time limit
    ///
    /// # Returns
    /// The best move, or None if the game is over
    pub fn get_move(
        &mut self,
        board: &Board,
        depth: Option<u8>,
        time_limit_ms: Option<u64>,
    ) -> Option<Turn> {
        // Check if game is over
        if Rules::get_winner(board).is_some() {
            return None;
        }

        // Configure search parameters
        let search_depth = depth.unwrap_or(self.depth);
        let search_time = time_limit_ms.or(self.time_limit_ms);

        // Set timeout or depth
        if let Some(time_ms) = search_time {
            self.strategy.set_timeout(Duration::from_millis(time_ms));
        } else {
            self.strategy.set_max_depth(search_depth);
        }

        // Perform search (using cached evaluations from previous searches)
        self.strategy.choose_move(board)
    }

    /// Get the evaluation for the current position
    ///
    /// # Arguments
    /// * `board` - The current board position
    /// * `depth` - Search depth (0 for static evaluation)
    ///
    /// # Returns
    /// Evaluation score on absolute scale (positive = White advantage, negative = Black advantage)
    pub fn get_evaluation(&mut self, board: &Board, depth: Option<u8>) -> i16 {
        use minimax::Evaluator;
        use crate::Color;

        let eval = BasicEvaluator::new(self.aggression);
        let depth = depth.unwrap_or(0);

        let score = if depth == 0 {
            // Static evaluation
            eval.evaluate(board)
        } else {
            // Minimax evaluation using the cached strategy
            self.strategy.set_max_depth(depth);
            
            let mut board_copy = board.clone();
            let best_move = self.strategy.choose_move(&board_copy);

            if best_move.is_none() {
                return eval.evaluate(board) as i16;
            }

            // Get the principal variation
            let pv = self.strategy.principal_variation();

            if pv.is_empty() {
                eval.evaluate(board)
            } else {
                // Apply all moves in the principal variation
                for &mv in &pv {
                    board_copy.apply(mv);
                }

                // Evaluate the resulting position
                let mut pv_score = eval.evaluate(&board_copy);

                // Adjust for whose turn it is after the PV
                if pv.len() % 2 == 1 {
                    pv_score = -pv_score;
                }

                pv_score
            }
        };

        // Convert to absolute scale
        let absolute_score = if board.to_move() == Color::Black { -score } else { score };

        absolute_score as i16
    }

    /// Choose a suboptimal move from the top-N best moves
    ///
    /// This evaluates all legal moves, ranks them, and randomly selects
    /// from the top-N moves. Useful for generating diverse training data.
    ///
    /// # Arguments
    /// * `board` - The current board position
    /// * `top_n` - Number of top moves to consider
    /// * `depth` - Optional search depth (uses engine's default if not provided)
    ///
    /// # Returns
    /// A randomly selected move from the top-N moves, or None if no legal moves
    pub fn get_suboptimal_move(
        &mut self,
        board: &Board,
        top_n: usize,
        depth: Option<u8>,
    ) -> Option<Turn> {
        let mut legal_moves = Vec::new();
        Rules::generate_moves(board, &mut legal_moves);

        if legal_moves.is_empty() {
            return None;
        }

        if legal_moves.len() <= 1 {
            return legal_moves.into_iter().next();
        }

        let search_depth = depth.unwrap_or(self.depth);

        // Evaluate each move
        let mut move_scores = Vec::with_capacity(legal_moves.len());
        for &mv in &legal_moves {
            let mut board_copy = board.clone();
            board_copy.apply(mv);
            
            // Get evaluation from the perspective of the player who just moved
            let score = -self.get_evaluation(&board_copy, Some(search_depth));
            move_scores.push((mv, score));
        }

        // Sort by score (descending)
        move_scores.sort_by(|a, b| b.1.cmp(&a.1));

        // Select from top-N
        let top_moves: Vec<_> = move_scores.iter()
            .take(top_n.min(move_scores.len()))
            .map(|(mv, _)| *mv)
            .collect();

        // Random selection from top moves using position hash for deterministic randomness
        if top_moves.len() == 1 {
            Some(top_moves[0])
        } else {
            let hash = Rules::zobrist_hash(board);
            let idx = (hash as usize) % top_moves.len();
            Some(top_moves[idx])
        }
    }

    /// Clear the transposition table
    ///
    /// This frees memory and resets the cache. Useful if you want to start
    /// fresh or if memory usage is a concern.
    pub fn clear_cache(&mut self) {
        // Recreate the strategy with a fresh table
        let eval = BasicEvaluator::new(self.aggression);
        let table_size = 100 << 20; // 100 MB default
        let opts = minimax::IterativeOptions::new()
            .with_table_byte_size(table_size);
        
        self.strategy = minimax::IterativeSearch::new(eval, opts);
    }

    /// Get statistics about cache efficiency
    ///
    /// Returns information about transposition table hits/misses
    /// (if available from the minimax library)
    pub fn get_cache_stats(&self) -> String {
        // This depends on what the minimax library exposes
        // For now, just return a placeholder
        "Cache stats not available".to_string()
    }
}

impl Default for CachedEngine {
    fn default() -> Self {
        Self::new(None, None, None, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cached_engine_basic() {
        let mut engine = CachedEngine::new(Some(3), Some(2), None, Some(50));
        let board = Board::default();

        // Should get a move for the initial position
        let move1 = engine.get_move(&board, None, None);
        assert!(move1.is_some());
    }

    #[test]
    fn test_cached_engine_sequential_moves() {
        let mut engine = CachedEngine::new(Some(3), Some(2), None, Some(50));
        let mut board = Board::default();

        // Make several moves and verify the engine still works
        for _ in 0..10 {
            if let Some(mv) = engine.get_move(&board, None, None) {
                board.apply(mv);
            } else {
                break;
            }
        }

        // Should have made some moves
        assert!(board.turn_num > 0);
    }

    #[test]
    fn test_suboptimal_move() {
        let mut engine = CachedEngine::new(Some(3), Some(2), None, Some(50));
        let board = Board::default();

        // Should get a suboptimal move
        let move1 = engine.get_suboptimal_move(&board, 3, None);
        assert!(move1.is_some());
    }

    #[test]
    fn test_evaluation() {
        let mut engine = CachedEngine::new(Some(3), None, None, Some(50));
        let board = Board::default();

        // Initial position should have zero or near-zero evaluation
        let eval = engine.get_evaluation(&board, Some(0));
        // Evaluation should be reasonable (not wildly large)
        assert!(eval.abs() < 1000);
    }
}
