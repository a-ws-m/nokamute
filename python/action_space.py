"""
Action space enumeration for Hive.

This module provides a fixed action space for Hive by enumerating all possible
UHP move strings that could ever be legal in the game. This enables:
1. Fixed-size policy networks (output a fixed number of action logits)
2. Action masking to select only legal moves
3. Efficient batch processing without dynamic action spaces

The action space includes:
- First move placements: "wQ", "wG1", etc.
- Placements with position: "wQ \\bA1", "wG1 bQ/", etc.
- Movements with position: "wA1 wQ-", etc.
- Pass move: "pass"
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple


def build_action_space() -> Tuple[Dict[int, str], Dict[str, int]]:
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


# Global action space (built once on module import for efficiency)
_ACTION_TO_STRING = None
_STRING_TO_ACTION = None
_ACTION_SPACE_SIZE = None


def get_action_space() -> Tuple[Dict[int, str], Dict[str, int], int]:
    """
    Get the global action space mappings.

    Returns:
        Tuple of (action_to_string, string_to_action, action_space_size)
    """
    global _ACTION_TO_STRING, _STRING_TO_ACTION, _ACTION_SPACE_SIZE

    if _ACTION_TO_STRING is None:
        _ACTION_TO_STRING, _STRING_TO_ACTION = build_action_space()
        _ACTION_SPACE_SIZE = len(_ACTION_TO_STRING)

    # Type assertion - we know these are not None after initialization
    assert _ACTION_TO_STRING is not None
    assert _STRING_TO_ACTION is not None
    assert _ACTION_SPACE_SIZE is not None

    return _ACTION_TO_STRING, _STRING_TO_ACTION, _ACTION_SPACE_SIZE


def action_to_string(action_idx: int) -> str:
    """Convert action index to move string."""
    action_to_str, _, _ = get_action_space()
    return action_to_str[action_idx]


def string_to_action(move_str: str) -> int:
    """Convert move string to action index. Returns -1 if not found."""
    _, str_to_action, _ = get_action_space()
    return str_to_action.get(move_str, -1)


def get_action_space_size() -> int:
    """Get the size of the action space."""
    _, _, size = get_action_space()
    return size


def save_action_space(path: str):
    """Save action space to file for quick loading."""
    action_to_str, str_to_action = build_action_space()
    with open(path, "wb") as f:
        pickle.dump((action_to_str, str_to_action), f)


def load_action_space(path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    """Load action space from file."""
    with open(path, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    # Test action space building
    print("Building Hive action space...")

    action_to_str, str_to_action, size = get_action_space()

    print(f"\nAction space size: {size}")
    print(f"\nFirst 10 actions:")
    for i in range(min(10, size)):
        print(f"  {i}: {action_to_str[i]}")

    print(f"\nLast 10 actions:")
    for i in range(max(0, size - 10), size):
        print(f"  {i}: {action_to_str[i]}")

    # Test round-trip conversion
    print(f"\nTesting round-trip conversion:")
    test_moves = ["wQ", "wA1 \\bQ", "bG2 wQ-", "pass"]
    for move_str in test_moves:
        action_idx = string_to_action(move_str)
        if action_idx != -1:
            recovered = action_to_string(action_idx)
            print(f"  '{move_str}' -> {action_idx} -> '{recovered}' ✓")
        else:
            print(f"  '{move_str}' -> NOT FOUND ✗")

    print(f"\nAction space module ready!")
