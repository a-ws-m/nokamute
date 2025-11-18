import torch
from hetero_graph_utils import ordered_unique_action_indices


def test_ordered_unique_action_indices_basic():
    arr = torch.tensor([4, 10, 1, -1, 4, 10])
    assert ordered_unique_action_indices(arr) == [4, 10, 1]


def test_ordered_unique_action_indices_empty():
    arr = torch.tensor([])
    assert ordered_unique_action_indices(arr) == []


def test_ordered_unique_action_indices_all_negative():
    arr = torch.tensor([-1, -1, -1], dtype=torch.long)
    assert ordered_unique_action_indices(arr) == []
