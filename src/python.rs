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

    /// Convert board state to a heterogeneous graph representation
    ///
    /// Returns a dictionary with:
    ///   - 'node_features': Dict with keys 'in_play', 'out_of_play', 'destination'
    ///   - 'edge_index': Dict with keys 'neighbour', 'move'
    ///   - 'edge_attr': Dict with keys 'neighbour', 'move' (edge features)
    ///   - 'move_to_action': List mapping move edge indices (for current player) to action space indices
    ///
    /// Node types:
    ///   - in_play: Pieces currently on the board (color, bug_onehot)
    ///   - out_of_play: Pieces not yet placed (color, bug_onehot)
    ///   - destination: Empty spaces where pieces can be placed or moved to
    ///
    /// Edge types:
    ///   - neighbour: Adjacency/stacking relationships (feature: [0.0])
    ///   - move: Legal moves for both players (feature: [1.0] if current player, [0.0] if opponent)
    fn to_graph(&self, py: Python) -> PyResult<PyObject> {
        use crate::hex_grid::adjacent;
        use std::collections::{HashMap, HashSet};
        use pyo3::types::PyDict;

        // Node indices for each type
        let mut in_play_nodes = Vec::new();
        let mut out_of_play_nodes = Vec::new();
        let mut destination_nodes = Vec::new();
        
        // Map pieces/destinations to node indices (within their type)
        let mut hex_to_in_play: HashMap<Hex, usize> = HashMap::new();
        let mut hex_to_destination: HashMap<Hex, usize> = HashMap::new();
        let mut hex_on_top_to_destination: HashMap<Hex, usize> = HashMap::new();
        let mut bug_color_to_out_of_play: HashMap<(RustBug, RustColor), usize> = HashMap::new();
        
        let current_color = self.inner.to_move();

        // 1. Create in-play nodes (pieces currently on the board)
        // Features: [color (0/1), bug_onehot (9 values)]
        for color_idx in 0..2 {
            let color = if color_idx == 0 { RustColor::White } else { RustColor::Black };
            for &hex in self.inner.occupied_hexes[color_idx].iter() {
                let node_idx = in_play_nodes.len();
                hex_to_in_play.insert(hex, node_idx);

                let node = self.inner.node(hex);
                let bug = node.bug();

                // One-hot encode bug type (8 bug types)
                let mut bug_onehot = vec![0.0f32; 9];
                bug_onehot[bug as usize] = 1.0;

                // Node features: [color (0/1), bug_onehot (9 values)]
                let mut features = vec![color as usize as f32];
                features.extend(bug_onehot);

                in_play_nodes.push(features);
            }
        }

        // 2. Create out-of-play nodes (pieces not yet placed)
        // Features: [color (0/1), bug_onehot (9 values)]
        for color_idx in 0..2 {
            let color = if color_idx == 0 { RustColor::White } else { RustColor::Black };
            let remaining = &self.inner.remaining[color_idx];
            
            for (bug_idx, &count) in remaining.iter().enumerate() {
                if count > 0 {
                    let bug = unsafe { std::mem::transmute::<u8, RustBug>(bug_idx as u8) };
                    let node_idx = out_of_play_nodes.len();
                    bug_color_to_out_of_play.insert((bug, color), node_idx);

                    // One-hot encode bug type
                    let mut bug_onehot = vec![0.0f32; 9];
                    bug_onehot[bug as usize] = 1.0;

                    // Node features: [color (0/1), bug_onehot (9 values)]
                    let mut features = vec![color as usize as f32];
                    features.extend(bug_onehot);

                    out_of_play_nodes.push(features);
                }
            }
        }

        // 3. Create destination nodes
        // These are empty spaces where pieces can be placed or moved to
        // Also includes on-top-of-piece destinations
        // Features: [on_top (0/1), padding to match 10 features total]
        
        // We need to generate destination nodes that account for BOTH players' possible moves
        // Generate legal moves for current player to find all destination hexes
        let mut current_moves = Vec::new();
        Rules::generate_moves(&self.inner, &mut current_moves);
        
        // Generate legal moves for opponent to find their destination hexes
        // Special case: on turn 0, both players have the same placement options (START_HEX only)
        // Using Pass would advance to turn 1 and give incorrect placement options
        let mut opponent_moves = Vec::new();
        if self.inner.turn_num == 0 {
            // On turn 0, opponent has same placement options as current player (just START_HEX)
            // Generate moves for the opponent color
            let opponent_color = if self.inner.to_move() == RustColor::White {
                RustColor::Black
            } else {
                RustColor::White
            };
            
            // Opponent can place any non-Queen piece on START_HEX
            for (bug_idx, &count) in self.inner.remaining[opponent_color as usize].iter().enumerate() {
                if count > 0 {
                    let bug = unsafe { std::mem::transmute::<u8, RustBug>(bug_idx as u8) };
                    if bug != RustBug::Queen {
                        opponent_moves.push(RustTurn::Place(crate::hex_grid::START_HEX, bug));
                    }
                }
            }
        } else {
            // After turn 0, use Pass to simulate opponent's turn
            let mut board_copy = self.inner.clone();
            board_copy.apply(RustTurn::Pass);  // Switch to opponent
            Rules::generate_moves(&board_copy, &mut opponent_moves);
            board_copy.undo(RustTurn::Pass);  // Restore
        }
        
        // Collect all destination hexes from legal moves
        let mut destination_hexes = HashSet::new();
        let mut destination_on_top_hexes = HashSet::new();
        
        for &turn in current_moves.iter().chain(opponent_moves.iter()) {
            match turn {
                RustTurn::Place(hex, _) => {
                    // Check if this is placement on top of an existing piece
                    if self.inner.height(hex) > 0 {
                        destination_on_top_hexes.insert(hex);
                    } else {
                        destination_hexes.insert(hex);
                    }
                }
                RustTurn::Move(_, to_hex) => {
                    // Check if this is a move on top of an existing piece
                    if self.inner.height(to_hex) > 0 {
                        destination_on_top_hexes.insert(to_hex);
                    } else {
                        destination_hexes.insert(to_hex);
                    }
                }
                RustTurn::Pass => {}
            }
        }
        
        // Create destination nodes for on-top hexes
        for hex in destination_on_top_hexes.iter() {
            let node_idx = destination_nodes.len();
            hex_on_top_to_destination.insert(*hex, node_idx);
            
            // Features: [1.0 for on_top, then 9 zeros for padding]
            let mut features = vec![1.0f32];
            features.extend(vec![0.0f32; 9]);
            destination_nodes.push(features);
        }
        
        // Create destination nodes for empty space hexes
        for hex in destination_hexes.iter() {
            let node_idx = destination_nodes.len();
            hex_to_destination.insert(*hex, node_idx);

            // Features: [0.0 for on_bottom, then 9 zeros for padding]
            let mut features = vec![0.0f32];
            features.extend(vec![0.0f32; 9]);
            destination_nodes.push(features);
        }

        // 4. Create neighbour edges (adjacency/stacking relationships)
        // Track edges by their type tuple (src_type, edge_type, dst_type)
        let mut neighbour_edges: HashMap<(&str, &str, &str), (Vec<usize>, Vec<usize>)> = HashMap::new();

        // Initialize edge type collections
        let edge_types = vec![
            ("in_play", "neighbour", "in_play"),
            ("in_play", "neighbour", "destination"),
            ("destination", "neighbour", "in_play"),
            ("destination", "neighbour", "destination"),
        ];
        
        for &edge_type in &edge_types {
            neighbour_edges.insert(edge_type, (Vec::new(), Vec::new()));
        }

        // Edges between in-play nodes (pieces)
        for color_idx in 0..2 {
            for &hex in self.inner.occupied_hexes[color_idx].iter() {
                let i = hex_to_in_play[&hex];
                
                // Edges to adjacent pieces
                for &adj_hex in adjacent(hex).iter() {
                    if let Some(&j) = hex_to_in_play.get(&adj_hex) {
                        let (from_vec, to_vec) = neighbour_edges.get_mut(&("in_play", "neighbour", "in_play")).unwrap();
                        from_vec.push(i);
                        to_vec.push(j);
                    }
                }

                // Edge to destination on top (if it exists)
                if let Some(&dest_idx) = hex_on_top_to_destination.get(&hex) {
                    let (from_vec, to_vec) = neighbour_edges.get_mut(&("in_play", "neighbour", "destination")).unwrap();
                    from_vec.push(i);
                    to_vec.push(dest_idx);
                    // Bidirectional
                    let (from_vec, to_vec) = neighbour_edges.get_mut(&("destination", "neighbour", "in_play")).unwrap();
                    from_vec.push(dest_idx);
                    to_vec.push(i);
                }
            }
        }

        // Edges between destination nodes (adjacent empty spaces)
        for &hex1 in destination_hexes.iter() {
            if let Some(&i) = hex_to_destination.get(&hex1) {
                for &adj_hex in adjacent(hex1).iter() {
                    if let Some(&j) = hex_to_destination.get(&adj_hex) {
                        let (from_vec, to_vec) = neighbour_edges.get_mut(&("destination", "neighbour", "destination")).unwrap();
                        from_vec.push(i);
                        to_vec.push(j);
                    }
                }
            }
        }

        // Edges from destination nodes to adjacent in-play nodes
        // Use all destination hexes (both regular and on-top)
        for &hex in destination_hexes.iter() {
            if let Some(&dest_idx) = hex_to_destination.get(&hex) {
                for &adj_hex in adjacent(hex).iter() {
                    if let Some(&piece_idx) = hex_to_in_play.get(&adj_hex) {
                        let (from_vec, to_vec) = neighbour_edges.get_mut(&("destination", "neighbour", "in_play")).unwrap();
                        from_vec.push(dest_idx);
                        to_vec.push(piece_idx);
                        // Bidirectional
                        let (from_vec, to_vec) = neighbour_edges.get_mut(&("in_play", "neighbour", "destination")).unwrap();
                        from_vec.push(piece_idx);
                        to_vec.push(dest_idx);
                    }
                }
            }
        }

        // 5. Create move edges (legal moves for both players)
        // We already generated these moves above when creating destination nodes
        // Track edges by their type tuple and whether they're for current player
        let mut move_edges_in_play: (Vec<usize>, Vec<usize>, Vec<f32>) = (Vec::new(), Vec::new(), Vec::new());
        let mut move_edges_out_of_play: (Vec<usize>, Vec<usize>, Vec<f32>) = (Vec::new(), Vec::new(), Vec::new());
        let mut move_to_action = Vec::new();

        // Process current player's moves
        for &turn in current_moves.iter() {
            match turn {
                RustTurn::Place(hex, bug) => {
                    // Edge from out-of-play node to destination node
                    if let Some(&out_idx) = bug_color_to_out_of_play.get(&(bug, current_color)) {
                        let dest_idx = hex_to_destination.get(&hex)
                            .or_else(|| hex_on_top_to_destination.get(&hex));
                        if let Some(&d_idx) = dest_idx {
                            move_edges_out_of_play.0.push(out_idx);
                            move_edges_out_of_play.1.push(d_idx);
                            move_edges_out_of_play.2.push(1.0);  // Current player
                            // Map this move edge to action space
                            let move_str = self.inner.to_move_string(turn);
                            move_to_action.push(move_str);
                        }
                    }
                }
                RustTurn::Move(from_hex, to_hex) => {
                    // Edge from in-play node to destination node
                    if let Some(&from_idx) = hex_to_in_play.get(&from_hex) {
                        let dest_idx = hex_to_destination.get(&to_hex)
                            .or_else(|| hex_on_top_to_destination.get(&to_hex));
                        if let Some(&d_idx) = dest_idx {
                            move_edges_in_play.0.push(from_idx);
                            move_edges_in_play.1.push(d_idx);
                            move_edges_in_play.2.push(1.0);  // Current player
                            // Map this move edge to action space
                            let move_str = self.inner.to_move_string(turn);
                            move_to_action.push(move_str);
                        }
                    }
                }
                RustTurn::Pass => {
                    // Pass move - no edge, but add to action mapping
                    move_to_action.push("pass".to_string());
                }
            }
        }

        // Process opponent's moves (marked as is_current=false)
        let opponent_color = if current_color == RustColor::White {
            RustColor::Black
        } else {
            RustColor::White
        };
        
        for &turn in opponent_moves.iter() {
            match turn {
                RustTurn::Place(hex, bug) => {
                    if let Some(&out_idx) = bug_color_to_out_of_play.get(&(bug, opponent_color)) {
                        let dest_idx = hex_to_destination.get(&hex)
                            .or_else(|| hex_on_top_to_destination.get(&hex));
                        if let Some(&d_idx) = dest_idx {
                            move_edges_out_of_play.0.push(out_idx);
                            move_edges_out_of_play.1.push(d_idx);
                            move_edges_out_of_play.2.push(0.0);  // Opponent
                        }
                    }
                }
                RustTurn::Move(from_hex, to_hex) => {
                    if let Some(&from_idx) = hex_to_in_play.get(&from_hex) {
                        let dest_idx = hex_to_destination.get(&to_hex)
                            .or_else(|| hex_on_top_to_destination.get(&to_hex));
                        if let Some(&d_idx) = dest_idx {
                            move_edges_in_play.0.push(from_idx);
                            move_edges_in_play.1.push(d_idx);
                            move_edges_in_play.2.push(0.0);  // Opponent
                        }
                    }
                }
                RustTurn::Pass => {}
            }
        }

        // Convert to Python dictionary in PyTorch Geometric HeteroData format
        // Return separate variables for each node type and edge type
        let result = PyDict::new(py);

        // Node features - separate variable for each node type
        result.set_item(
            "x_in_play",
            PyList::new(py, in_play_nodes.iter().map(|f| PyList::new(py, f)))
        )?;
        result.set_item(
            "x_out_of_play",
            PyList::new(py, out_of_play_nodes.iter().map(|f| PyList::new(py, f)))
        )?;
        result.set_item(
            "x_destination",
            PyList::new(py, destination_nodes.iter().map(|f| PyList::new(py, f)))
        )?;

        // Edge indices - separate variable for each edge type
        // Format: edge_index_{src}_{edge}_{dst}
        for &(src_type, edge_type, dst_type) in &edge_types {
            let key = format!("edge_index_{}_{}_{}", src_type, edge_type, dst_type);
            if let Some((from_vec, to_vec)) = neighbour_edges.get(&(src_type, edge_type, dst_type)) {
                result.set_item(
                    key,
                    PyList::new(py, vec![
                        PyList::new(py, from_vec),
                        PyList::new(py, to_vec)
                    ])
                )?;
            }
        }
        
        // Move edge indices
        result.set_item(
            "edge_index_in_play_move_destination",
            PyList::new(py, vec![
                PyList::new(py, &move_edges_in_play.0),
                PyList::new(py, &move_edges_in_play.1)
            ])
        )?;
        result.set_item(
            "edge_index_out_of_play_move_destination",
            PyList::new(py, vec![
                PyList::new(py, &move_edges_out_of_play.0),
                PyList::new(py, &move_edges_out_of_play.1)
            ])
        )?;

        // Edge attributes - separate variable for each edge type
        // Format: edge_attr_{src}_{edge}_{dst}
        // Neighbour edges all have zero features
        for &(src_type, edge_type, dst_type) in &edge_types {
            let key = format!("edge_attr_{}_{}_{}", src_type, edge_type, dst_type);
            if let Some((from_vec, _)) = neighbour_edges.get(&(src_type, edge_type, dst_type)) {
                let attrs: Vec<Vec<f32>> = vec![vec![0.0f32]; from_vec.len()];
                result.set_item(
                    key,
                    PyList::new(py, attrs.iter().map(|f| PyList::new(py, f)))
                )?;
            }
        }
        
        // Move edge attributes have binary features (current player = 1.0, opponent = 0.0)
        result.set_item(
            "edge_attr_in_play_move_destination",
            PyList::new(py, move_edges_in_play.2.iter().map(|&f| PyList::new(py, &[f])))
        )?;
        result.set_item(
            "edge_attr_out_of_play_move_destination",
            PyList::new(py, move_edges_out_of_play.2.iter().map(|&f| PyList::new(py, &[f])))
        )?;

        // Move to action mapping (list of move strings for current player's legal moves)
        result.set_item("move_to_action", PyList::new(py, &move_to_action))?;

        Ok(result.into())
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

    /// Parse a game string in UHP format and create a board
    ///
    /// Format: "GameType;GameState;Turn;MoveString1;MoveString2;..."
    /// Example: "Base+MLP;InProgress;White[1];wQ;bQ wQ-;wA1 bQ-;..."
    ///
    /// Args:
    ///     game_string: UHP format game string
    ///
    /// Returns:
    ///     Board: New board with moves applied, or error if invalid
    #[staticmethod]
    fn from_game_string(game_string: &str) -> PyResult<Self> {
        RustBoard::from_game_string(game_string).map(|inner| Board { inner }).map_err(|e| match e {
            crate::notation::UhpError::InvalidGameString(s) => {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid game string: {}", s))
            }
            crate::notation::UhpError::InvalidMove(s) => {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid move: {}", s))
            }
            crate::notation::UhpError::InvalidGameType(s) => {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid game type: {}", s))
            }
            e => pyo3::exceptions::PyValueError::new_err(format!("Parse error: {:?}", e)),
        })
    }

    /// Parse a single move string in UHP format
    ///
    /// Examples:
    ///   - "wQ" (place white queen on first move)
    ///   - "bQ wQ-" (place black queen east of white queen)
    ///   - "wA1 bQ/" (move white ant to northeast of black queen)
    ///   - "pass" (pass turn)
    ///
    /// Args:
    ///     move_string: UHP format move string
    ///
    /// Returns:
    ///     Turn: Parsed turn, or error if invalid
    fn parse_move(&self, move_string: &str) -> PyResult<Turn> {
        self.inner.from_move_string(move_string).map(|inner| Turn { inner }).map_err(|e| match e {
            crate::notation::UhpError::InvalidMove(s) => {
                pyo3::exceptions::PyValueError::new_err(format!("Invalid move: {}", s))
            }
            e => pyo3::exceptions::PyValueError::new_err(format!("Parse error: {:?}", e)),
        })
    }

    /// Convert a turn to UHP move string format
    ///
    /// Args:
    ///     turn: The turn to convert
    ///
    /// Returns:
    ///     str: Move string in UHP format
    fn to_move_string(&self, turn: &Turn) -> String {
        self.inner.to_move_string(turn.inner)
    }

    /// Get the game log (move history) as a semicolon-separated string
    ///
    /// Returns:
    ///     str: Game log in UHP format (e.g., "wQ;bQ wQ-;wA1 bQ/")
    fn get_game_log(&self) -> String {
        self.inner.game_log()
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
    fn get_engine_move(
        &self, depth: Option<u8>, time_limit_ms: Option<u64>, aggression: Option<u8>,
    ) -> PyResult<Option<Turn>> {
        use crate::BasicEvaluator;
        use minimax::{Game, Strategy};
        use std::time::Duration;

        // Check if game is over
        if Rules::get_winner(&self.inner).is_some() {
            return Ok(None);
        }

        let eval = BasicEvaluator::new(aggression.unwrap_or(3));
        let opts = minimax::IterativeOptions::new().verbose().with_table_byte_size(100 << 20);

        let mut strategy = minimax::IterativeSearch::new(eval, opts);

        if let Some(time_ms) = time_limit_ms {
            strategy.set_timeout(Duration::from_millis(time_ms));
        } else {
            strategy.set_max_depth(depth.unwrap_or(3));
        }

        let best_move = strategy.choose_move(&self.inner);
        Ok(best_move.map(|m| Turn { inner: m }))
    }

    /// Get the best move and its evaluation in one search
    ///
    /// This is more efficient than calling get_engine_move() and get_evaluation()
    /// separately, as it performs only one minimax search.
    ///
    /// Args:
    ///     depth: Search depth for minimax (default: 3)
    ///     time_limit_ms: Time limit in milliseconds (optional, overrides depth)
    ///     aggression: Aggression level 1-5 for the evaluator (default: 3)
    ///
    /// Returns:
    ///     Tuple[Optional[Turn], i16]: (best_move, evaluation_score)
    ///         - best_move: Best move according to the engine, or None if game is over
    ///         - evaluation_score: Evaluation on absolute scale (positive = White advantage)
    fn get_engine_move_with_eval(
        &self, depth: Option<u8>, time_limit_ms: Option<u64>, aggression: Option<u8>,
    ) -> PyResult<(Option<Turn>, i16)> {
        use crate::BasicEvaluator;
        use minimax::{Evaluator, Game, Strategy};
        use std::time::Duration;

        // Check if game is over
        if Rules::get_winner(&self.inner).is_some() {
            let eval = BasicEvaluator::new(aggression.unwrap_or(3));
            let score = eval.evaluate(&self.inner);
            let absolute_score = if self.inner.to_move() == RustColor::Black { -score } else { score };
            return Ok((None, absolute_score));
        }

        let eval = BasicEvaluator::new(aggression.unwrap_or(3));
        let opts = minimax::IterativeOptions::new().verbose().with_table_byte_size(100 << 20);

        let mut strategy = minimax::IterativeSearch::new(eval, opts);

        if let Some(time_ms) = time_limit_ms {
            strategy.set_timeout(Duration::from_millis(time_ms));
        } else {
            strategy.set_max_depth(depth.unwrap_or(3));
        }

        let best_move = strategy.choose_move(&self.inner);
        
        // Get the evaluation from the principal variation
        let pv = strategy.principal_variation();
        let eval2 = BasicEvaluator::new(aggression.unwrap_or(3));
        let score = if pv.is_empty() {
            // Fallback to static eval
            eval2.evaluate(&self.inner)
        } else {
            // Apply all moves in the principal variation
            let mut board_copy = self.inner.clone();
            for &mv in &pv {
                board_copy.apply(mv);
            }

            // Evaluate the resulting position
            let mut pv_score = eval2.evaluate(&board_copy);

            // If the PV has an odd number of moves, the evaluation is from the opponent's perspective
            if pv.len() % 2 == 1 {
                pv_score = -pv_score;
            }

            pv_score
        };

        // Convert to absolute scale
        let absolute_score = if self.inner.to_move() == RustColor::Black { -score } else { score };

        Ok((best_move.map(|m| Turn { inner: m }), absolute_score))
    }

    /// Get the principal variation (best line of play) from the current position
    ///
    /// Args:
    ///     depth: Search depth for minimax (default: 3)
    ///     aggression: Aggression level 1-5 for the evaluator (default: 3)
    ///
    /// Returns:
    ///     List[Turn]: Sequence of best moves for both players
    fn get_principal_variation(
        &self, depth: Option<u8>, aggression: Option<u8>,
    ) -> PyResult<Vec<Turn>> {
        use crate::BasicEvaluator;
        use minimax::{Strategy};

        // Check if game is over
        if Rules::get_winner(&self.inner).is_some() {
            return Ok(Vec::new());
        }

        let eval = BasicEvaluator::new(aggression.unwrap_or(3));
        let opts = minimax::IterativeOptions::new().verbose().with_table_byte_size(100 << 20);

        let mut strategy = minimax::IterativeSearch::new(eval, opts);
        strategy.set_max_depth(depth.unwrap_or(3));

        // Run the search to populate the principal variation
        let _best_move = strategy.choose_move(&self.inner);

        // Get the principal variation
        let pv = strategy.principal_variation();

        // Convert to Python Turn objects
        Ok(pv.into_iter().map(|m| Turn { inner: m }).collect())
    }

    /// Get the analytical evaluation of the current position
    ///
    /// Args:
    ///     aggression: Aggression level 1-5 for the evaluator (default: 3)
    ///     depth: Search depth for minimax evaluation (default: 0 for static eval)
    ///            If depth > 0, returns the minimax evaluation after N moves
    ///
    /// Returns:
    ///     i16: Evaluation score on absolute scale.
    ///          Positive values favor White, negative values favor Black.
    fn get_evaluation(&self, aggression: Option<u8>, depth: Option<u8>) -> PyResult<i16> {
        use crate::BasicEvaluator;
        use minimax::{Evaluator, Strategy};

        let eval = BasicEvaluator::new(aggression.unwrap_or(3));
        let depth = depth.unwrap_or(0);

        let score = if depth == 0 {
            // Static evaluation
            eval.evaluate(&self.inner)
        } else {
            // Minimax evaluation at given depth
            // Use principal_variation to get the sequence of best moves
            use minimax::Negamax;
            let mut strategy = Negamax::new(eval, depth);
            let mut board_copy = self.inner.clone();

            // Run the search first (this populates the principal variation)
            let best_move = strategy.choose_move(&board_copy);

            if best_move.is_none() {
                // No moves available (game over or no legal moves)
                return Ok(eval.evaluate(&self.inner) as i16);
            }

            // Get the principal variation (best line of play for both sides)
            let pv = strategy.principal_variation();

            if pv.is_empty() {
                // Fallback to static eval if PV is empty
                eval.evaluate(&self.inner)
            } else {
                // Apply all moves in the principal variation
                for &mv in &pv {
                    board_copy.apply(mv);
                }

                // Evaluate the resulting position
                // We need to account for whose turn it is after the PV
                let mut pv_score = eval.evaluate(&board_copy);

                // If the PV has an odd number of moves, the evaluation is from the opponent's perspective
                // and we need to negate it
                if pv.len() % 2 == 1 {
                    pv_score = -pv_score;
                }

                pv_score
            }
        };

        // Convert from player-relative to absolute scale
        // BasicEvaluator returns score from perspective of player to move
        // We need to flip the sign if Black is to move
        let absolute_score = if self.inner.to_move() == RustColor::Black { -score } else { score };

        Ok(absolute_score)
    }

    fn __repr__(&self) -> String {
        format!(
            "Board(turn={}, to_move={})",
            self.inner.turn_num,
            if self.inner.to_move() == RustColor::Black { "Black" } else { "White" }
        )
    }
}

/// Cached Negamax strategy for efficient self-play game generation
///
/// This strategy maintains a transposition table across moves, making it
/// much more efficient for sequential move generation in self-play scenarios.
/// Use this instead of repeated calls to `get_engine_move()` when generating
/// training data from self-play games.
#[pyclass]
pub struct CachedNegamaxStrategy {
    #[cfg(not(target_arch = "wasm32"))]
    strategy: crate::CachedNegamax<crate::BasicEvaluator>,
}

#[pymethods]
impl CachedNegamaxStrategy {
    /// Create a new cached negamax strategy
    ///
    /// Args:
    ///     depth: Search depth for minimax (default: 3)
    ///     aggression: Aggression level 1-5 for the evaluator (default: 3)
    #[new]
    fn new(depth: Option<u8>, aggression: Option<u8>) -> PyResult<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let eval = crate::BasicEvaluator::new(aggression.unwrap_or(3));
            let strategy = crate::CachedNegamax::new(eval, depth.unwrap_or(3));
            Ok(CachedNegamaxStrategy { strategy })
        }
        #[cfg(target_arch = "wasm32")]
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "CachedNegamaxStrategy is not available in wasm32"
        ))
    }

    /// Get the best move for a given board position
    ///
    /// This uses the cached transposition table for efficiency.
    ///
    /// Args:
    ///     board: The board to evaluate
    ///
    /// Returns:
    ///     Optional[Turn]: Best move, or None if game is over
    fn choose_move(&mut self, board: &Board) -> PyResult<Option<Turn>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use minimax::Strategy;
            let best_move = self.strategy.choose_move(&board.inner);
            Ok(best_move.map(|m| Turn { inner: m }))
        }
        #[cfg(target_arch = "wasm32")]
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "CachedNegamaxStrategy is not available in wasm32"
        ))
    }

    /// Get the best move and its evaluation in one call
    ///
    /// Args:
    ///     board: The board to evaluate
    ///
    /// Returns:
    ///     Tuple[Optional[Turn], i16]: (best_move, evaluation_score)
    ///         - best_move: Best move, or None if game is over
    ///         - evaluation_score: Evaluation on absolute scale (positive = White advantage)
    fn choose_move_with_eval(&mut self, board: &Board) -> PyResult<(Option<Turn>, i16)> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use minimax::{Evaluator, Game, Strategy};

            // Check if game is over
            if Rules::get_winner(&board.inner).is_some() {
                let eval = crate::BasicEvaluator::new(self.strategy.eval.aggression());
                let score = eval.evaluate(&board.inner);
                let absolute_score = if board.inner.to_move() == RustColor::Black { -score } else { score };
                return Ok((None, absolute_score));
            }

            let best_move = self.strategy.choose_move(&board.inner);
            let score = self.strategy.root_value();
            
            // Convert to absolute scale
            let absolute_score = if board.inner.to_move() == RustColor::Black { -score } else { score };
            
            Ok((best_move.map(|m| Turn { inner: m }), absolute_score))
        }
        #[cfg(target_arch = "wasm32")]
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "CachedNegamaxStrategy is not available in wasm32"
        ))
    }

    /// Clear the transposition table cache
    ///
    /// Call this when starting a new game to free memory and reset statistics.
    fn clear_cache(&mut self) -> PyResult<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.strategy.clear_cache();
            Ok(())
        }
        #[cfg(target_arch = "wasm32")]
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "CachedNegamaxStrategy is not available in wasm32"
        ))
    }

    /// Get cache statistics
    ///
    /// Returns:
    ///     Tuple[int, int, float]: (hits, misses, hit_rate)
    fn cache_stats(&self) -> PyResult<(usize, usize, f64)> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Ok(self.strategy.cache_stats())
        }
        #[cfg(target_arch = "wasm32")]
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "CachedNegamaxStrategy is not available in wasm32"
        ))
    }

    /// Get the size of the cache (number of entries)
    ///
    /// Returns:
    ///     int: Number of cached positions
    fn cache_size(&self) -> PyResult<usize> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Ok(self.strategy.table.len())
        }
        #[cfg(target_arch = "wasm32")]
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "CachedNegamaxStrategy is not available in wasm32"
        ))
    }

    /// Set the search depth
    ///
    /// Args:
    ///     depth: New search depth
    fn set_depth(&mut self, depth: u8) -> PyResult<()> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            use minimax::Strategy;
            self.strategy.set_max_depth(depth);
            Ok(())
        }
        #[cfg(target_arch = "wasm32")]
        Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "CachedNegamaxStrategy is not available in wasm32"
        ))
    }

    fn __repr__(&self) -> String {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (hits, misses, hit_rate) = self.strategy.cache_stats();
            format!(
                "CachedNegamaxStrategy(depth={}, cache_size={}, hits={}, misses={}, hit_rate={:.2}%)",
                self.strategy.max_depth,
                self.strategy.table.len(),
                hits,
                misses,
                hit_rate * 100.0
            )
        }
        #[cfg(target_arch = "wasm32")]
        String::from("CachedNegamaxStrategy(not available in wasm32)")
    }
}

/// Python module initialization
#[pymodule]
fn nokamute(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Board>()?;
    m.add_class::<Turn>()?;
    m.add_class::<Bug>()?;
    m.add_class::<Color>()?;
    m.add_class::<CachedNegamaxStrategy>()?;
    Ok(())
}
