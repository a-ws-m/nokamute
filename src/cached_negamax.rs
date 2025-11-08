//! A cached implementation of Negamax for self-play games.
//!
//! This maintains a transposition table across moves to avoid redundant searches
//! in sequential move generation. Ideal for generating training data from self-play.

use crate::{Board, Rules, Turn};
use minimax::*;
use rand::seq::SliceRandom;
use std::cmp::max;
use std::collections::HashMap;

const WORST_EVAL: Evaluation = -30_000;
const WIN_EVAL: Evaluation = 10_000;

/// Cache entry storing the evaluation and best move at a given depth
#[derive(Clone, Copy)]
pub struct CacheEntry {
    pub depth: u8,
    pub eval: Evaluation,
    pub best_move: Option<Turn>,
    pub node_type: NodeType,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum NodeType {
    Exact,    // Exact evaluation
    LowerBound, // Alpha cutoff (beta >= value)
    UpperBound, // Beta cutoff (alpha <= value)
}

pub struct CachedNegamax<E: Evaluator<G = Rules>> {
    pub max_depth: u8,
    prev_value: Evaluation,
    pub eval: E,
    // Transposition table: zobrist_hash -> CacheEntry
    pub table: HashMap<u64, CacheEntry>,
    // Statistics for monitoring cache effectiveness
    pub hits: usize,
    pub misses: usize,
}

impl<E: Evaluator<G = Rules>> CachedNegamax<E> {
    pub fn new(eval: E, depth: u8) -> Self {
        CachedNegamax {
            max_depth: depth,
            prev_value: 0,
            eval,
            table: HashMap::with_capacity(1_000_000), // Pre-allocate for performance
            hits: 0,
            misses: 0,
        }
    }

    /// Clear the cache (useful when starting a new game)
    pub fn clear_cache(&mut self) {
        self.table.clear();
        self.hits = 0;
        self.misses = 0;
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize, f64) {
        let total = self.hits + self.misses;
        let hit_rate = if total > 0 {
            self.hits as f64 / total as f64
        } else {
            0.0
        };
        (self.hits, self.misses, hit_rate)
    }

    #[doc(hidden)]
    pub fn root_value(&self) -> Evaluation {
        self.prev_value
    }

    fn negamax(
        &mut self, s: &mut Board, depth: u8, mut alpha: Evaluation, beta: Evaluation,
    ) -> Evaluation {
        let original_alpha = alpha;
        
        // Check for terminal position
        if let Some(winner) = Rules::get_winner(s) {
            return match winner {
                Winner::PlayerJustMoved => WIN_EVAL,
                Winner::PlayerToMove => -WIN_EVAL,
                Winner::Draw => 0,
            };
        }
        
        if depth == 0 {
            return self.eval.evaluate(s);
        }

        // Check transposition table
        let hash = Rules::zobrist_hash(s);
        if let Some(entry) = self.table.get(&hash) {
            if entry.depth >= depth {
                self.hits += 1;
                match entry.node_type {
                    NodeType::Exact => return entry.eval,
                    NodeType::LowerBound => {
                        alpha = max(alpha, entry.eval);
                    }
                    NodeType::UpperBound => {
                        if entry.eval <= alpha {
                            return entry.eval;
                        }
                    }
                }
                if alpha >= beta {
                    return entry.eval;
                }
            }
        } else {
            self.misses += 1;
        }

        // Generate and search moves
        let mut moves = Vec::new();
        Rules::generate_moves(s, &mut moves);
        
        let mut best = WORST_EVAL;
        let mut best_move = None;
        
        for m in moves.iter() {
            s.apply(*m);
            let value = -self.negamax(s, depth - 1, -beta, -alpha);
            s.undo(*m);
            
            if value > best {
                best = value;
                best_move = Some(*m);
            }
            
            alpha = max(alpha, value);
            if alpha >= beta {
                break;
            }
        }
        
        // Store in transposition table
        let node_type = if best <= original_alpha {
            NodeType::UpperBound
        } else if best >= beta {
            NodeType::LowerBound
        } else {
            NodeType::Exact
        };
        
        self.table.insert(hash, CacheEntry {
            depth,
            eval: best,
            best_move,
            node_type,
        });
        
        best
    }
}

impl<E: Evaluator<G = Rules>> Strategy<Rules> for CachedNegamax<E> {
    fn choose_move(&mut self, s: &Board) -> Option<Turn> {
        if self.max_depth == 0 {
            return None;
        }
        if Rules::get_winner(s).is_some() {
            return None;
        }
        
        // Check if we have a cached best move for this position
        let hash = Rules::zobrist_hash(s);
        if let Some(entry) = self.table.get(&hash) {
            if entry.depth >= self.max_depth && entry.node_type == NodeType::Exact {
                self.hits += 1;
                if let Some(best_move) = entry.best_move {
                    self.prev_value = entry.eval;
                    return Some(best_move);
                }
            }
        }
        
        let mut best = WORST_EVAL;
        let mut moves = Vec::new();
        Rules::generate_moves(s, &mut moves);
        
        // Randomly permute order to introduce non-determinism
        moves.shuffle(&mut rand::rng());

        let mut best_move = *moves.first()?;
        let mut s_clone = s.clone();
        
        for &m in moves.iter() {
            s_clone.apply(m);
            let value = -self.negamax(&mut s_clone, self.max_depth - 1, WORST_EVAL, -best);
            s_clone.undo(m);
            
            if value > best {
                best = value;
                best_move = m;
            }
        }
        
        self.prev_value = best;
        
        // Cache the result
        self.table.insert(hash, CacheEntry {
            depth: self.max_depth,
            eval: best,
            best_move: Some(best_move),
            node_type: NodeType::Exact,
        });
        
        Some(best_move)
    }

    fn set_max_depth(&mut self, depth: u8) {
        self.max_depth = depth;
    }
}
