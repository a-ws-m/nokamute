try:
    from python.agents import GNNAgent, NokamuteEngine, RandomAgent
    from python.game import Game
except Exception:
    import os
    import sys

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    from python.agents import RandomAgent, NokamuteEngine, GNNAgent
    from python.game import Game

from nokamute import Board


def test_random_vs_nokamute():
    rand = RandomAgent(seed=42)
    engine = NokamuteEngine(depth=1)
    game = Game()
    res = game.play(rand, engine, max_steps=50)
    assert len(res.trajectory) > 0
    # Each trajectory step should have expected attributes
    for step in res.trajectory:
        assert hasattr(step, "board")
        assert hasattr(step, "action")
        assert hasattr(step, "player")


def test_gnn_agent_vs_random():
    gnn = GNNAgent(None)
    rand = RandomAgent(seed=1)
    game = Game()
    res = game.play(gnn, rand, max_steps=50)
    assert len(res.trajectory) > 0
    # verify trajectory steps captured
    assert res.trajectory[0].player in ("White", "Black")


def test_trajectory_storage():
    rand = RandomAgent(seed=3)
    engine = NokamuteEngine(depth=1)
    game = Game()
    res = game.play(rand, engine, max_steps=10)
    assert hasattr(res, "trajectory")
    assert isinstance(res.trajectory, list)
    # Each item should be a tuple-like dataclass with action_str
    assert res.trajectory[0].action_str is not None
