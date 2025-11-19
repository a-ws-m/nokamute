use crate::{Board as RustBoard, Bug as RustBug, Color as RustColor, Hex, Rules, Turn as RustTurn};
use minimax::Game;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use std::collections::HashMap;

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

    #[staticmethod]
    fn from_game_string(s: &str) -> PyResult<Self> {
        match RustBoard::from_game_string(s) {
            Ok(b) => Ok(Board { inner: b }),
            Err(e) => Err(pyo3::exceptions::PyValueError::new_err(format!("{:?}", e))),
        }
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
    fn to_graph(&self, py: Python) -> PyResult<PyObject> {
        // Build a dictionary describing the heterogeneous graph.
        // Node buckets: in_play pieces, out_of_play pieces (one per bug type if any remain), destination hexes
        let graph = PyDict::new(py);

        // In-play pieces
        let mut in_play_nodes: Vec<PyObject> = Vec::new();
        let mut hex_to_inplay: HashMap<Hex, Vec<usize>> = HashMap::new();
        let mut idx: usize = 0;
        for color_idx in 0..2 {
            for &hex in self.inner.occupied_hexes[color_idx].iter() {
                let node = self.inner.node(hex);
                let entry = PyDict::new(py);
                entry.set_item("hex", hex)?;
                let color_str = if node.color() == RustColor::Black { "Black" } else { "White" };
                entry.set_item("color", color_str)?;
                entry.set_item("bug", node.bug().name())?;
                entry.set_item("height", self.inner.height(hex))?;
                entry.set_item("bug_idx", node.bug() as u8)?;
                entry.set_item("id", idx)?;
                // Topmost nodes: if the height() is > 1 there are pieces below it
                entry.set_item("is_underneath", false)?;
                entry.set_item("is_above", self.inner.height(hex) > 1)?;
                hex_to_inplay.insert(hex, vec![idx]);
                in_play_nodes.push(entry.into());
                idx += 1;
            }
        }

        // Add underworld nodes as in-play if any (so stacked pieces below the top appear as nodes)
        for under in self.inner.get_underworld() {
            let node = under.node();
            let entry = PyDict::new(py);
            entry.set_item("hex", under.hex())?;
            entry.set_item("color", if node.color() == RustColor::Black { "Black" } else { "White" })?;
            entry.set_item("bug", node.bug().name())?;
            entry.set_item("height", node.clipped_height())?;
            entry.set_item("bug_idx", node.bug() as u8)?;
            entry.set_item("id", idx)?;
            // Underworld pieces are underneath other pieces; is_above means there's any piece below
            entry.set_item("is_underneath", true)?;
            entry.set_item("is_above", node.clipped_height() > 1)?;
            hex_to_inplay.entry(under.hex()).or_default().push(idx);
            in_play_nodes.push(entry.into());
            idx += 1;
        }

        graph.set_item("in_play_nodes", PyList::new(py, in_play_nodes))?;

        // Out-of-play nodes (one per bug type for each color if remaining)
        let mut out_nodes: Vec<PyObject> = Vec::new();
        let mut out_idx = 0usize;
        for color_idx in 0..2 {
            let rem = self.inner.remaining[color_idx];
            for bug in RustBug::iter_all() {
                let num_left = rem[bug as usize];
                if num_left > 0 {
                    let entry = PyDict::new(py);
                    entry.set_item("bug", bug.name())?;
                    let color_str = if color_idx == 1 { "Black" } else { "White" };
                    entry.set_item("color", color_str)?;
                    entry.set_item("num_left", num_left)?;
                    entry.set_item("bug_idx", bug as u8)?;
                    entry.set_item("id", out_idx)?;
                    out_nodes.push(entry.into());
                    out_idx += 1;
                }
            }
        }

        graph.set_item("out_of_play_nodes", PyList::new(py, out_nodes))?;

        // Destination nodes: empty hex neighbors of in-play pieces, and a top-of-piece dest node
        let mut dest_nodes: Vec<PyObject> = Vec::new();
        let mut dest_map: HashMap<Hex, usize> = HashMap::new();
        let mut dest_idx: usize = 0;

        for &hex in self.inner.occupied_hexes[0].iter().chain(self.inner.occupied_hexes[1].iter()) {
            for neighbor in crate::hex_grid::adjacent(hex) {
                if !self.inner.occupied(neighbor) {
                    if !dest_map.contains_key(&neighbor) {
                        let entry = PyDict::new(py);
                        entry.set_item("hex", neighbor)?;
                        entry.set_item("is_top", false)?;
                        entry.set_item("id", dest_idx)?;
                        dest_nodes.push(entry.into());
                        dest_map.insert(neighbor, dest_idx);
                        dest_idx += 1;
                    }
                }
            }

            // Dest node on top of piece (if there is space above it)
            let height = self.inner.height(hex);
            // If the piece is not at a ridiculous height, allow a top-of-stack destination
            // Heuristic: allow destination when height < 4 (depth limit in gameplay rarely exceeds 3).
            if self.inner.node(hex).occupied() && height < 4 {
                if !dest_map.contains_key(&hex) {
                    let entry = PyDict::new(py);
                    entry.set_item("hex", hex)?;
                    entry.set_item("is_top", true)?;
                    entry.set_item("id", dest_idx)?;
                    dest_nodes.push(entry.into());
                    dest_map.insert(hex, dest_idx);
                    dest_idx += 1;
                } else if let Some(&eidx) = dest_map.get(&hex) {
                    // If there is already a destination for this hex (empty), mark it as top as well
                    // so the Python side can inspect.
                    let existing = dest_nodes.get(eidx).unwrap().as_ref(py).downcast::<PyDict>()?;
                    existing.set_item("is_top", true)?;
                }
            }
        }

        graph.set_item("destination_nodes", PyList::new(py, dest_nodes))?;

        // If there are no destination nodes (empty board), provide the start hex
        // so first-turn placements can be represented as edges to a single abstract destination.
        if dest_map.is_empty() {
            let entry = PyDict::new(py);
            entry.set_item("hex", crate::hex_grid::START_HEX)?;
            entry.set_item("is_top", false)?;
            entry.set_item("id", dest_idx)?;
            dest_map.insert(crate::hex_grid::START_HEX, dest_idx);
            // Append fallback to the list used earlier.
            let mut dest_nodes: Vec<PyObject> = Vec::new();
            dest_nodes.push(entry.into());
            graph.set_item("destination_nodes", PyList::new(py, dest_nodes))?;
        }

        // Adjacency edges: for each in-play piece, connect to neighboring in-play (if any),
        // or destination nodes for empty neighbors. Also add edge to top-of-piece destination if present.
        let mut adjacency: Vec<PyObject> = Vec::new();
        for (hex, ids) in hex_to_inplay.iter() {
            for &src_idx in ids.iter() {
                for neighbor in crate::hex_grid::adjacent(*hex) {
                    if let Some(dst_ids) = hex_to_inplay.get(&neighbor) {
                        if !dst_ids.is_empty() {
                            let tup = PyTuple::new(py, &[
                                "in_play".into_py(py),
                                (src_idx as u32).into_py(py),
                                "in_play".into_py(py),
                                (dst_ids[0] as u32).into_py(py),
                            ]);
                            adjacency.push(tup.into());
                        }
                    } else if let Some(&dst_idx) = dest_map.get(&neighbor) {
                        let tup = PyTuple::new(py, &[
                            "in_play".into_py(py),
                            (src_idx as u32).into_py(py),
                            "destination".into_py(py),
                            (dst_idx as u32).into_py(py),
                        ]);
                        adjacency.push(tup.into());
                    }
                }

                // Top space adjacency
                if let Some(&dst_idx) = dest_map.get(hex) {
                    let tup = PyTuple::new(py, &[
                        "in_play".into_py(py),
                        (src_idx as u32).into_py(py),
                        "destination".into_py(py),
                        (dst_idx as u32).into_py(py),
                    ]);
                    adjacency.push(tup.into());
                }
            }
        }
        // Add vertical adjacency between stacked pieces in the same hex.
        for ids in hex_to_inplay.values() {
            if ids.len() > 1 {
                for i in 0..ids.len() {
                    for j in 0..ids.len() {
                        if i == j { continue; }
                        let tup = PyTuple::new(py, &[
                            "in_play".into_py(py),
                            (ids[i] as u32).into_py(py),
                            "in_play".into_py(py),
                            (ids[j] as u32).into_py(py),
                        ]);
                        adjacency.push(tup.into());
                    }
                }
            }
        }

        graph.set_item("adjacency_edges", PyList::new(py, adjacency))?;

        // Current-player move edges using Rules::generate_moves
        let mut moves = Vec::new();
        Rules::generate_moves(&self.inner, &mut moves);
        let mut current_edges: Vec<PyObject> = Vec::new();
        for mv in moves.iter() {
            match mv {
                RustTurn::Place(hex, bug) => {
                    // Out-of-play bug -> destination
                    // find out_of_play node for this bug for this color
                    let color_idx = self.inner.to_move() as usize; // whose turn is placing
                    // find out_of_play id
                    // We used order color_idx 0..1 and Bug.iter_all order; recompute index
                    // We need mapping: out_id = number of earlier out nodes for earlier colors
                    // Build a combined mapping to ids
                }
                RustTurn::Move(from, to) => {
                    // From (in-play piece) -> destination node
                }
                _ => {}
            }
        }

        // For simplicity fill current move edges in Python side using the Rust turns list.
        // Provide a list of turns for the current player.
        let py_moves = PyList::new(py, moves.into_iter().map(|m| format!("{:?}", m)).collect::<Vec<String>>());
        graph.set_item("moves_current", py_moves)?;

        // Next-player moves: clone, apply pass, then generate moves
        let mut clone = self.inner.clone();
        clone.apply(RustTurn::Pass);
        let mut next_moves = Vec::new();
        Rules::generate_moves(&clone, &mut next_moves);
        let py_next_moves = PyList::new(py, next_moves.into_iter().map(|m| format!("{:?}", m)).collect::<Vec<String>>());
        graph.set_item("moves_next", py_next_moves)?;

        Ok(graph.into())
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
