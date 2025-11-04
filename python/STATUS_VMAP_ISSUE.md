# Status: TorchRL vmap Incompatibility with GNNs

## Test Results

**Tests 1-7: ✓ PASSED**
1. ✓ Import all modules
2. ✓ Environment creation and reset  
3. ✓ Environment step
4. ✓ Model creation
5. ✓ Model forward pass (with eval mode)
6. ✓ Self-play evaluation
7. ✓ TorchRL components creation

**Test 8: ✗ FAILED - vmap Incompatibility**

## The Problem

TorchRL's `GAE` (Generalized Advantage Estimation) and efficient data collectors use PyTorch's `vmap` function to vectorize operations across the batch dimension. However, **Graph Neural Networks (GNNs) are fundamentally incompatible with vmap** because:

1. **Variable Structure**: Each graph has a different number of nodes and edges
2. **Assertion Failures**: GNN layers expect 2D tensors `[num_nodes, features]`, but vmap adds an extra batch dimension
3. **No Batching Standard**: Unlike images (fixed HxW) or text (padding), graphs can't be easily batched with vmap

## Error Chain

```
advantage_module(tensordict_data)  
→ GAE uses vmap(value_net)
→ vmap adds batch dimension to all tensors
→ GATv2Conv receives 3D tensor instead of 2D
→ AssertionError: assert x.dim() == 2
```

## Solutions

###  Option 1: Custom GAE without vmap (RECOMMENDED)

Implement a custom GAE that doesn't use vmap:

```python
class GraphGAE(nn.Module):
    """GAE for graph-based environments without vmap."""
    
    def __init__(self, value_network, gamma=0.99, lmbda=0.95):
        super().__init__()
        self.value_network = value_network
        self.gamma = gamma
        self.lmbda = lmbda
    
    def forward(self, tensordict):
        """Compute advantages without vmap."""
        rewards = tensordict["next", "reward"]
        dones = tensordict["next", "done"]
        
        # Compute values one at a time (no vmap)
        values = []
        for i in range(len(tensordict)):
            td_i = tensordict[i].unsqueeze(0)
            val = self.value_network(td_i)["state_value"]
            values.append(val)
        values = torch.cat(values)
        
        # Compute advantages using standard GAE formula
        # ... (implement GAE calculation)
        
        return tensordict
```

### Option 2: Disable vmap in GAE

```python
advantage_module = GAE(
    gamma=0.99,
    lmbda=0.95,
    value_network=critic,
    average_gae=True,
    device=device,
    # Try to disable vmap (may not be supported)
    vectorized=False,  # If this option exists
)
```

### Option 3: Use PyTorch Geometric Batching

Rewrite the encoder to use PyG's batching:

```python
from torch_geometric.data import Batch

# Manually batch graphs
graph_list = [...]
batched = Batch.from_data_list(graph_list)
```

But this requires major restructuring of the environment and data flow.

### Option 4: Use Simpler Network (MLP)

Replace GNN with MLP that flattens the graph:

```python
# Flatten node features
x_flat = node_features.view(batch_size, -1)  # [B, max_nodes * node_dim]
```

This loses the graph structure benefits but is vmap-compatible.

## Recommended Next Steps

1. **Implement Custom GAE** (Option 1): Most control, preserves GNN architecture
2. **Test with Custom GAE**: Verify it works with collectors
3. **Update training loop**: Use custom GAE instead of TorchRL's GAE
4. **Consider PPO Loss**: May also need custom implementation without vmap

## Current Implementation Status

- ✓ Environment: Fully functional with correct TorchRL conventions
- ✓ Models: GNN-based actor-critic networks working
- ✓ Self-play: Evaluation system working
- ✗ Training Loop: Blocked by vmap incompatibility

## Alternative: Skip Test 8

Since tests 1-7 pass, the implementation is mostly complete. Test 8 (mini training loop) is more of an integration test. We could:

1. Document the vmap limitation
2. Provide a custom training loop that doesn't use GAE with vmap
3. Use the models with manual advantage calculation

The core PPO algorithm doesn't require vmap - it's just an optimization used by TorchRL's default implementations.
