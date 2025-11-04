# Fix: TorchRL "next" Key Convention

## Issue
When calling `env.step(td)`, the reward and next observation were placed at the root level of the returned TensorDict, but TorchRL expects them under a `"next"` key.

## Error
```
KeyError: 'key "reward" not found in TensorDict with keys ['action', 'action_mask', 'done', 'edge_index', 'next', ...]'
```

## TorchRL Convention

In TorchRL, the `_step()` method should return data in a specific nested structure:

```python
# Root level: current state + action taken
{
    "action": ...,           # The action taken
    "node_features": ...,    # Current state obs
    # ... other current state data
    
    # Nested under "next": results of the action
    "next": {
        "node_features": ...,  # Next state obs
        "edge_index": ...,     # Next state graph
        "reward": ...,         # Reward received
        "done": ...,           # Episode termination flag
        "terminated": ...,     # Natural termination
        "truncated": ...,      # Timeout/truncation
    }
}
```

## Solution

Updated `_step()` method in `hive_env.py`:

```python
# Before (incorrect)
out = TensorDict(
    {
        "node_features": obs_dict["node_features"],
        "reward": torch.tensor([reward], ...),
        "done": torch.tensor([done], ...),
        # ...
    },
    batch_size=self.batch_size,
    device=self.device,
)

# After (correct)
out = TensorDict(
    {
        "next": TensorDict(
            {
                "node_features": obs_dict["node_features"],
                "reward": torch.tensor([reward], ...),
                "done": torch.tensor([done], ...),
                # ...
            },
            batch_size=self.batch_size,
            device=self.device,
        ),
    },
    batch_size=self.batch_size,
    device=self.device,
)
```

## Accessing Data

After the fix, access reward and next state like this:

```python
td = env.reset()
td["action"] = torch.tensor(0)
td_next = env.step(td)

# Access next state observations
next_obs = td_next["next"]["node_features"]

# Access reward
reward = td_next["next"]["reward"]

# Access done flag
done = td_next["next"]["done"]
```

## Why This Convention?

This nested structure:
1. **Separates state transitions**: Current state vs. next state are clearly distinguished
2. **Supports rollouts**: Makes it easy to stack transitions in trajectories
3. **Compatible with collectors**: TorchRL's data collectors expect this format
4. **Standard RL format**: Follows the (s, a, r, s') tuple convention

## Files Updated
- `python/hive_env.py` - Fixed `_step()` method and test code
- `python/test_ppo.py` - Updated to access `td_next["next"]["reward"]`
- `python/test_env_only.py` - Updated for correct access pattern
