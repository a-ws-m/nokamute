"""
PettingZoo AEC environment for Hive board game.

This module implements the Hive board game as a PettingZoo AEC (Agent Environment Cycle) environment.
The action space represents all possible UHP move strings, and the observation space uses
Gymnasium's Graph space to represent the board state.
"""

import functools
from typing import Dict, List, Optional

import gymnasium
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo.utils import wrappers

import nokamute


def env(**kwargs):
    """
    Factory function that returns the environment wrapped with recommended wrappers.
    
    Args:
        **kwargs: Environment configuration arguments
        
    Returns:
        Wrapped Hive AEC environment
    """
    env = raw_env(**kwargs)
    # Add standard wrappers
    env = wrappers.TerminateIllegalWrapper(env, illegal_reward=-1)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(**kwargs):
    """
    Factory function that returns the raw Hive AEC environment without wrappers.
    
    Args:
        **kwargs: Environment configuration arguments
        
    Returns:
        Raw Hive AEC environment
    """
    return HiveAECEnv(**kwargs)


class HiveAECEnv(AECEnv):
    """
    PettingZoo AEC environment for the Hive board game.
    
    The Hive board game is a two-player abstract strategy game where players place
    and move insect-themed pieces on a hexagonal grid. The goal is to surround the
    opponent's Queen Bee.
    
    ## Action Space
    
    The action space is Discrete(N) where N is the total number of possible UHP move
    strings in Hive. Each action corresponds to a unique move string like:
    - "wQ" - place white queen on first move
    - "bQ wQ-" - place black queen east of white queen
    - "wA1 bQ/" - move white ant to northeast of black queen
    - "pass" - pass turn
    
    ## Observation Space
    
    The observation space is a Graph space representing the board state:
    - Nodes: Pieces on the board with features [color, bug_type_onehot, height, current_player]
    - Edges: Adjacency relationships between pieces on the hex grid
    
    ## Rewards
    
    - +1 for winning (surrounding opponent's queen)
    - -1 for losing
    - 0 for draw or ongoing game
    
    ## Starting State
    
    Empty board with both players having their full starting set of pieces
    
    ## Episode End
    
    The episode ends when:
    - One player's queen is surrounded (win/loss)
    - Both queens are surrounded simultaneously (draw)
    - Maximum number of moves is reached (draw)
    
    ## Arguments
    
    - `game_type`: str, default "Base+MLP" - The game variant (Base, Base+M, Base+L, Base+P, Base+MLP)
    - `max_moves`: int, default 400 - Maximum moves before declaring draw
    - `render_mode`: str, optional - Rendering mode ("human", "rgb_array", or None)
    """
    
    metadata = {
        "render_modes": ["human", "ansi"],
        "name": "hive_v0",
        "is_parallelizable": False,
        "render_fps": 2,
    }
    
    def __init__(
        self,
        game_type: str = "Base+MLP",
        max_moves: int = 400,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the Hive AEC environment.
        
        Args:
            game_type: Game variant ("Base", "Base+M", "Base+L", "Base+P", "Base+MLP")
            max_moves: Maximum number of moves before declaring draw
            render_mode: Rendering mode ("human", "ansi", or None)
        """
        super().__init__()
        
        self.game_type = game_type
        self.max_moves = max_moves
        self.render_mode = render_mode
        
        # Define agents
        self.possible_agents = ["player_0", "player_1"]  # White, Black
        self.agent_name_mapping = {"player_0": 0, "player_1": 1}
        
        # Initialize board
        self.board = nokamute.Board.from_game_string(game_type)
        
        # Build complete action space (all possible UHP move strings)
        self._action_to_string, self._string_to_action = self._build_action_space()
        
        # Define observation and action spaces
        # Graph space for board representation
        # Node features: [color(1), bug_onehot(9), height(1), current_player(1)] = 12 features
        node_space = spaces.Box(low=0.0, high=10.0, shape=(12,), dtype=np.float32)
        # Edge features: just adjacency, no additional features
        edge_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        
        # Use Dict space to include both graph and action mask
        self._observation_spaces = {
            agent: spaces.Dict({
                "observation": spaces.Graph(node_space=node_space, edge_space=edge_space),
                "action_mask": spaces.Box(low=0, high=1, shape=(len(self._action_to_string),), dtype=np.int8),
            })
            for agent in self.possible_agents
        }
        
        self._action_spaces = {
            agent: spaces.Discrete(len(self._action_to_string))
            for agent in self.possible_agents
        }
        
        # Initialize state
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
        # Episode state
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.num_moves = 0
        
    def _build_action_space(self) -> tuple[Dict[int, str], Dict[str, int]]:
        """
        Build the complete action space mapping between discrete indices and UHP move strings.
        
        The action space includes all possible placements and movements for all piece types.
        We enumerate ALL possible move strings that could ever be legal in the game.
        
        Returns:
            Tuple of (action_to_string, string_to_action) dictionaries
        """
        action_to_string = {}
        string_to_action = {}
        action_idx = 0
        
        # Bug types in the game
        bug_types = ["Q", "A", "G", "B", "S", "M", "L", "P"]
        colors = ["w", "b"]
        
        # Generate all possible piece identifiers
        piece_identifiers = []
        for color in colors:
            for bug in bug_types:
                if bug in ["A", "G", "B", "S"]:
                    # These bugs have multiple instances (numbered 1-3)
                    for num in [1, 2, 3]:
                        piece_identifiers.append(f"{color}{bug}{num}")
                else:
                    # Single instance bugs (Q, M, L, P)
                    piece_identifiers.append(f"{color}{bug}")
        
        # 1. First move placements (just piece name, no position)
        # This covers moves like "wQ", "wG1", "bA1" etc.
        for piece in piece_identifiers:
            move_str = piece
            if move_str not in string_to_action:
                action_to_string[action_idx] = move_str
                string_to_action[move_str] = action_idx
                action_idx += 1
        
        # 2. Placements and movements with direction: "piece direction+target"
        # This covers moves like "wQ \\bA1", "wG1 bQ/", "wA1 wQ-" etc.
        for piece in piece_identifiers:
            for target in piece_identifiers:
                if piece == target:
                    continue
                for dir_char in ["\\", "/", "-"]:
                    # Direction before target (e.g., "wQ \\bQ" = place wQ northwest of bQ)
                    move_str = f"{piece} {dir_char}{target}"
                    if move_str not in string_to_action:
                        action_to_string[action_idx] = move_str
                        string_to_action[move_str] = action_idx
                        action_idx += 1
                    
                    # Direction after target (e.g., "wQ bQ\\" = place wQ southeast of bQ)
                    move_str = f"{piece} {target}{dir_char}"
                    if move_str not in string_to_action:
                        action_to_string[action_idx] = move_str
                        string_to_action[move_str] = action_idx
                        action_idx += 1
        
        # 3. Add "pass" action
        action_to_string[action_idx] = "pass"
        string_to_action["pass"] = action_idx
        action_idx += 1
        
        return action_to_string, string_to_action
    
    def observation_space(self, agent):
        """Return the observation space for the specified agent."""
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        """Return the action space for the specified agent."""
        return self._action_spaces[agent]
    
    def observe(self, agent):
        """
        Return the observation for the specified agent.
        
        Converts the board state to a Graph observation using the Rust binding's
        to_graph() method.
        
        Args:
            agent: Agent name
            
        Returns:
            Dict observation with keys:
            - "observation": Graph with "nodes", "edges", "edge_links"
            - "action_mask": Binary mask of legal actions
        """
        # Get graph representation from board
        node_features, edge_index = self.board.to_graph()
        
        # Convert to numpy arrays with correct shapes
        if len(node_features) == 0:
            # Empty board - no nodes
            nodes = np.zeros((0, 12), dtype=np.float32)
        else:
            nodes = np.array(node_features, dtype=np.float32)
        
        if len(edge_index) == 0 or len(edge_index[0]) == 0:
            # No edges
            edge_links = np.zeros((0, 2), dtype=np.int32)  # Edge connectivity (pairs of node indices)
            edges = np.zeros((0, 1), dtype=np.float32)  # Edge features
        else:
            # Convert edge_index [sources, targets] to edge_links [[source, target], ...]
            edge_array = np.array(edge_index, dtype=np.int32)
            edge_links = edge_array.T  # Transpose to get (num_edges, 2)
            # Create edge features (all ones for now, just indicating connectivity)
            edges = np.ones((edge_links.shape[0], 1), dtype=np.float32)
        
        # Create GraphInstance for the observation
        from gymnasium.spaces.graph import GraphInstance
        graph_obs = GraphInstance(nodes=nodes, edges=edges, edge_links=edge_links)
        
        # Return dict with graph observation and action mask
        return {
            "observation": graph_obs,
            "action_mask": self.action_mask(agent),
        }
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options
            
        Returns:
            observation: Initial observation for the first agent
            info: Additional info dictionary
        """
        # Seed if provided
        if seed is not None:
            np.random.seed(seed)
        
        # Reset board
        self.board = nokamute.Board.from_game_string(self.game_type)
        
        # Reset agents
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
        # Reset episode state
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.num_moves = 0
        
        # Return initial observation
        observation = self.observe(self.agent_selection)
        info = self.infos[self.agent_selection]
        
        return observation, info
    
    def step(self, action):
        """
        Execute an action in the environment.
        
        Args:
            action: Discrete action index
        """
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            # Agent is done, call _was_dead_step
            return self._was_dead_step(action)
        
        # Convert action index to move string
        move_string = self._action_to_string[action]
        
        # Get legal moves for validation
        legal_moves = self.board.legal_moves()
        legal_move_strings = [self.board.to_move_string(move) for move in legal_moves]
        
        # Check if action is legal
        if move_string not in legal_move_strings:
            # Illegal move - this should be caught by TerminateIllegalWrapper
            # but we handle it here for safety
            self.rewards[self.agent_selection] = -1
            self.terminations[self.agent_selection] = True
            self.infos[self.agent_selection]["illegal_move"] = True
            self._accumulate_rewards()
            self.agent_selection = self._agent_selector.next()
            return
        
        # Parse and apply the move
        turn = self.board.parse_move(move_string)
        self.board.apply(turn)
        self.num_moves += 1
        
        # Check for game termination
        winner = self.board.get_winner()
        
        if winner is not None:
            # Game is over
            if winner == "Draw":
                # Draw
                self.rewards = {agent: 0 for agent in self.agents}
            elif winner == "WhiteWins":
                self.rewards["player_0"] = 1
                self.rewards["player_1"] = -1
            elif winner == "BlackWins":
                self.rewards["player_0"] = -1
                self.rewards["player_1"] = 1
            
            # Mark all agents as terminated
            self.terminations = {agent: True for agent in self.agents}
        elif self.num_moves >= self.max_moves:
            # Max moves reached - draw
            self.rewards = {agent: 0 for agent in self.agents}
            self.truncations = {agent: True for agent in self.agents}
        else:
            # Game continues - no rewards yet
            self.rewards = {agent: 0 for agent in self.agents}
        
        # Accumulate rewards
        self._accumulate_rewards()
        
        # Switch to next agent
        self.agent_selection = self._agent_selector.next()
    
    def _accumulate_rewards(self):
        """Accumulate rewards for all agents."""
        for agent in self.agents:
            self._cumulative_rewards[agent] += self.rewards[agent]
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            String representation of the board in render_mode="ansi"
            None in render_mode="human" (prints to console)
        """
        if self.render_mode is None:
            return
        
        # Get game state as string
        game_state = self.board.get_game_log()
        
        # Get current player
        current_player = "White" if self.board.to_move().name == "White" else "Black"
        
        # Create display
        display = f"\n{'=' * 60}\n"
        display += f"Hive Game - Move {self.num_moves}\n"
        display += f"Current Player: {current_player}\n"
        display += f"Game State: {game_state}\n"
        display += f"{'=' * 60}\n"
        
        # Add piece positions
        pieces = self.board.get_pieces()
        if pieces:
            display += "\nPieces on board:\n"
            for hex_pos, color, bug, height in pieces:
                display += f"  {color}{bug} at hex {hex_pos} (height {height})\n"
        else:
            display += "\nNo pieces on board yet.\n"
        
        display += f"\n{'=' * 60}\n"
        
        if self.render_mode == "ansi":
            return display
        elif self.render_mode == "human":
            print(display)
    
    def close(self):
        """Close the environment and release resources."""
        pass
    
    def action_mask(self, agent):
        """
        Return a binary mask indicating which actions are legal.
        
        Args:
            agent: Agent name
            
        Returns:
            np.ndarray: Binary mask of shape (num_actions,) where 1 = legal, 0 = illegal
        """
        # Get legal moves
        legal_moves = self.board.legal_moves()
        legal_move_strings = set(self.board.to_move_string(move) for move in legal_moves)
        
        # Create mask
        mask = np.zeros(len(self._action_to_string), dtype=np.int8)
        for action_idx, move_string in self._action_to_string.items():
            if move_string in legal_move_strings:
                mask[action_idx] = 1
        
        return mask
