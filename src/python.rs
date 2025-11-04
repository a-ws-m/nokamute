use crate::{Board as RustBoard, Bug as RustBug, Color as RustColor, Hex, Rules, Turn as RustTurn};
use minimax::Game;
use pyo3::prelude::*;
use pyo3::types::PyList;

/// Represents a bug type in the Hive game
#[pyclass]
#[derive(Clone)]
pub struct Bug {
    inner: RustBug,
}

#[pymethods]
impl Bug {
    #[new]
    fn new(name: &str) -> PyResult<Self> {
        let inner = match name.to_lowercase().as_str() {
            "queen" => RustBug::Queen,
            "grasshopper" => RustBug::Grasshopper,
            "spider" => RustBug::Spider,
            "ant" => RustBug::Ant,
            "beetle" => RustBug::Beetle,
            "mosquito" => RustBug::Mosquito,
            "ladybug" => RustBug::Ladybug,
            "pillbug" => RustBug::Pillbug,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid bug name")),
        };
        Ok(Bug { inner })
    }

    fn __repr__(&self) -> String {
        format!("Bug('{}')", self.inner.name())
    }

    fn __str__(&self) -> String {
        self.inner.name().to_string()
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }
}

/// Represents the color of a player (Black or White)
#[pyclass]
#[derive(Clone, Copy)]
pub struct Color {
    inner: RustColor,
}

#[pymethods]
impl Color {
    #[new]
    fn new(color: &str) -> PyResult<Self> {
        let inner = match color.to_lowercase().as_str() {
            "black" => RustColor::Black,
            "white" => RustColor::White,
            _ => return Err(pyo3::exceptions::PyValueError::new_err("Invalid color")),
        };
        Ok(Color { inner })
    }

    fn __repr__(&self) -> String {
        format!("Color('{}')", if self.inner == RustColor::Black { "Black" } else { "White" })
    }

    fn __str__(&self) -> String {
        if self.inner == RustColor::Black { "Black" } else { "White" }.to_string()
    }

    #[getter]
    fn name(&self) -> String {
        if self.inner == RustColor::Black { "Black" } else { "White" }.to_string()
    }
}

/// Represents a turn/move in the game
#[pyclass]
#[derive(Clone)]
pub struct Turn {
    inner: RustTurn,
}

#[pymethods]
impl Turn {
    #[staticmethod]
    fn place(hex: u16, bug: Bug) -> Self {
        Turn { inner: RustTurn::Place(hex as Hex, bug.inner) }
    }

    #[staticmethod]
    fn move_bug(from_hex: u16, to_hex: u16) -> Self {
        Turn { inner: RustTurn::Move(from_hex as Hex, to_hex as Hex) }
    }

    #[staticmethod]
    fn pass() -> Self {
        Turn { inner: RustTurn::Pass }
    }

    fn __repr__(&self) -> String {
        match self.inner {
            RustTurn::Place(hex, bug) => format!("Turn.place({}, Bug('{}'))", hex, bug.name()),
            RustTurn::Move(from, to) => format!("Turn.move_bug({}, {})", from, to),
            RustTurn::Pass => "Turn.pass()".to_string(),
        }
    }

    fn is_place(&self) -> bool {
        matches!(self.inner, RustTurn::Place(_, _))
    }

    fn is_move(&self) -> bool {
        matches!(self.inner, RustTurn::Move(_, _))
    }

    fn is_pass(&self) -> bool {
        matches!(self.inner, RustTurn::Pass)
    }

    fn get_place_info(&self) -> PyResult<(u16, String)> {
        match self.inner {
            RustTurn::Place(hex, bug) => Ok((hex as u16, bug.name().to_string())),
            _ => Err(pyo3::exceptions::PyValueError::new_err("Not a Place turn")),
        }
    }

    fn get_move_info(&self) -> PyResult<(u16, u16)> {
        match self.inner {
            RustTurn::Move(from, to) => Ok((from as u16, to as u16)),
            _ => Err(pyo3::exceptions::PyValueError::new_err("Not a Move turn")),
        }
    }
}

/// Represents the Hive game board state
#[pyclass]
pub struct Board {
    inner: RustBoard,
}

#[pymethods]
impl Board {
    #[new]
    fn new() -> Self {
        Board { inner: RustBoard::default() }
    }

    /// Generate all legal moves for the current position
    fn legal_moves(&self) -> Vec<Turn> {
        let mut moves = Vec::new();
        Rules::generate_moves(&self.inner, &mut moves);
        moves.into_iter().map(|m| Turn { inner: m }).collect()
    }

    /// Apply a move to the board
    fn apply(&mut self, turn: &Turn) {
        self.inner.apply(turn.inner);
    }

    /// Undo a move
    fn undo(&mut self, turn: &Turn) {
        self.inner.undo(turn.inner);
    }

    /// Get the current player to move
    fn to_move(&self) -> Color {
        Color { inner: self.inner.to_move() }
    }

    /// Get the current turn number
    fn turn_num(&self) -> u16 {
        self.inner.turn_num
    }

    /// Clone the board
    fn clone(&self) -> Self {
        Board { inner: self.inner.clone() }
    }

    /// Check if the game is over and get the winner
    fn get_winner(&self) -> Option<String> {
        Rules::get_winner(&self.inner).map(|w| match w {
            minimax::Winner::Draw => "Draw".to_string(),
            minimax::Winner::PlayerToMove => {
                if self.inner.to_move() == RustColor::Black {
                    "Black".to_string()
                } else {
                    "White".to_string()
                }
            }
            minimax::Winner::PlayerJustMoved => {
                if self.inner.to_move() == RustColor::Black {
                    "White".to_string()
                } else {
                    "Black".to_string()
                }
            }
        })
    }

    /// Convert board state to a graph representation
    /// Returns (node_features, edge_index, edge_features)
    fn to_graph(&self, py: Python) -> PyResult<PyObject> {
        use crate::hex_grid::adjacent;
        use std::collections::{HashMap, HashSet};

        // Map (hex, height) to node indices
        let mut position_to_node: HashMap<(Hex, u8), usize> = HashMap::new();
        let mut node_features = Vec::new();

        // First pass: collect all occupied positions (hex, height)
        let mut occupied_positions = Vec::new();
        for color_idx in 0..2 {
            for &hex in self.inner.occupied_hexes[color_idx].iter() {
                let height = self.inner.height(hex);
                // Add nodes for each height level (stacked pieces)
                for h in 1..=height {
                    occupied_positions.push((hex, h));
                }
            }
        }

        // Collect empty spaces adjacent to pieces
        let mut empty_spaces = HashSet::new();
        for &(hex, _) in &occupied_positions {
            for &adj_hex in adjacent(hex).iter() {
                // Only add empty space at height 1 if no piece exists there
                if self.inner.height(adj_hex) == 0 {
                    empty_spaces.insert(adj_hex);
                }
            }
        }

        // Create nodes for occupied positions
        for &(hex, h) in &occupied_positions {
            let node_idx = position_to_node.len();
            position_to_node.insert((hex, h), node_idx);

            let node = self.inner.node(hex);
            let color = node.color();
            let bug = node.bug();

            // One-hot encode bug type (8 bug types + 1 for empty = 9 total)
            // Queen=0, Grasshopper=1, Spider=2, Ant=3, Beetle=4, Mosquito=5, Ladybug=6, Pillbug=7, Empty=8
            let mut bug_onehot = vec![0.0f32; 9];
            bug_onehot[bug as usize] = 1.0;

            // Node features: [color (0/1), bug_onehot (9 values), height]
            let mut features = vec![color as usize as f32];
            features.extend(bug_onehot);
            features.push(h as f32);

            node_features.push(features);
        }

        // Create nodes for empty spaces
        for &hex in &empty_spaces {
            let node_idx = position_to_node.len();
            position_to_node.insert((hex, 1), node_idx);

            // Empty space: color=0, bug_type=Empty (index 8), height=1
            let mut bug_onehot = vec![0.0f32; 9];
            bug_onehot[8] = 1.0; // Empty space

            let mut features = vec![0.0]; // color doesn't matter for empty
            features.extend(bug_onehot);
            features.push(1.0); // height=1 for empty spaces

            node_features.push(features);
        }

        // Create edges
        let mut edges_from = Vec::new();
        let mut edges_to = Vec::new();

        // Iterate through all positions
        for &(hex, h) in position_to_node.keys() {
            let i = position_to_node[&(hex, h)];

            // 1. Vertical edges (same hex, different heights)
            if let Some(&j) = position_to_node.get(&(hex, h + 1)) {
                edges_from.push(i);
                edges_to.push(j);
                // Add reverse edge for undirected graph
                edges_from.push(j);
                edges_to.push(i);
            }

            // 2. Horizontal edges (adjacent hexes at same height)
            for &adj_hex in adjacent(hex).iter() {
                if let Some(&j) = position_to_node.get(&(adj_hex, h)) {
                    edges_from.push(i);
                    edges_to.push(j);
                }
            }

            // 3. Edges to adjacent hexes at different heights
            for &adj_hex in adjacent(hex).iter() {
                // Check all height levels at the adjacent hex
                for dh in 1..=self.inner.height(adj_hex).max(1) {
                    if dh != h {
                        if let Some(&j) = position_to_node.get(&(adj_hex, dh)) {
                            edges_from.push(i);
                            edges_to.push(j);
                        }
                    }
                }
            }
        }

        // Convert to Python objects
        let node_features_py = PyList::new(py, node_features.iter().map(|f| PyList::new(py, f)));
        let edge_index_py =
            PyList::new(py, vec![PyList::new(py, &edges_from), PyList::new(py, &edges_to)]);

        // Return as tuple
        Ok((node_features_py, edge_index_py).into_py(py))
    }

    /// Get board state as a compact representation for features
    /// Returns a list of (hex, color, bug, height) tuples
    fn get_pieces(&self) -> Vec<(u16, String, String, u8)> {
        let mut pieces = Vec::new();

        for color_idx in 0..2 {
            for &hex in self.inner.occupied_hexes[color_idx].iter() {
                let node = self.inner.node(hex);
                let height = self.inner.height(hex);

                let color_str = if node.color() == RustColor::Black { "Black" } else { "White" };

                pieces.push((
                    hex as u16,
                    color_str.to_string(),
                    node.bug().name().to_string(),
                    height,
                ));
            }
        }

        pieces
    }

    /// Get the hash of the board position
    fn zobrist_hash(&self) -> u64 {
        Rules::zobrist_hash(&self.inner)
    }

    /// Get the best move according to the minimax engine
    /// 
    /// Args:
    ///     depth: Search depth for minimax (default: 3)
    ///     time_limit_ms: Time limit in milliseconds (optional, overrides depth)
    ///     aggression: Aggression level 1-5 for the evaluator (default: 3)
    /// 
    /// Returns:
    ///     Turn: Best move according to the engine, or None if game is over
    fn get_engine_move(&self, depth: Option<u8>, time_limit_ms: Option<u64>, aggression: Option<u8>) -> PyResult<Option<Turn>> {
        use minimax::{Game, Strategy};
        use crate::BasicEvaluator;
        use std::time::Duration;

        // Check if game is over
        if Rules::get_winner(&self.inner).is_some() {
            return Ok(None);
        }

        let eval = BasicEvaluator::new(aggression.unwrap_or(3));
        let opts = minimax::IterativeOptions::new()
            .with_table_byte_size(100 << 20)
            .with_countermoves()
            .with_countermove_history();
        
        let mut strategy = minimax::IterativeSearch::new(eval, opts);
        
        if let Some(time_ms) = time_limit_ms {
            strategy.set_timeout(Duration::from_millis(time_ms));
        } else {
            strategy.set_max_depth(depth.unwrap_or(3));
        }

        let best_move = strategy.choose_move(&self.inner);
        Ok(best_move.map(|m| Turn { inner: m }))
    }

    fn __repr__(&self) -> String {
        format!(
            "Board(turn={}, to_move={})",
            self.inner.turn_num,
            if self.inner.to_move() == RustColor::Black { "Black" } else { "White" }
        )
    }
}

/// Python module initialization
#[pymodule]
fn nokamute(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Board>()?;
    m.add_class::<Turn>()?;
    m.add_class::<Bug>()?;
    m.add_class::<Color>()?;
    Ok(())
}
