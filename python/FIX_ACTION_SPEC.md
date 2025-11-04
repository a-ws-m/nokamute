# Fix: TorchRL Action Spec Import

## Issue
The initial implementation used `Discrete` from `torchrl.data.tensor_specs`, which doesn't exist in TorchRL.

## Error
```
cannot import name 'Discrete' from 'torchrl.data.tensor_specs'
```

## Solution
Changed the action spec to use `Categorical` from `torchrl.data`:

```python
# Before (incorrect)
from torchrl.data.tensor_specs import Discrete
self.action_spec = Discrete(n=self.max_actions, shape=(), dtype=torch.long)

# After (correct)
from torchrl.data import Categorical
self.action_spec = Categorical(n=self.max_actions, shape=(), dtype=torch.long)
```

## TorchRL Discrete Action Specs

In TorchRL, there are different spec types for discrete actions:

1. **`Categorical`** - For discrete actions represented as integer indices (0, 1, 2, ...)
   - Use when: action is a single integer index
   - Shape: `()` for scalar, `(batch_size,)` for batched
   - Example: `Categorical(n=10)` for 10 discrete actions

2. **`OneHot`** - For discrete actions represented as one-hot vectors
   - Use when: action is a one-hot encoded vector
   - Shape: `(n,)` where n is the number of categories
   - Example: `OneHot(n=10, shape=(10,))` for 10 discrete actions

For our Hive environment, we use `Categorical` because:
- Actions are integer indices into the legal moves list
- More memory efficient than one-hot encoding
- Direct compatibility with `MaskedCategorical` distribution

## Files Modified
- `python/hive_env.py` - Fixed import and action spec

## Testing
Run the test to verify:
```bash
python test_env_only.py
```

This should now pass without import errors.
