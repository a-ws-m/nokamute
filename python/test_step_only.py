"""Test just the environment step to debug the reward key issue."""
import torch
from hive_env import HiveEnv

device = "cpu"
env = HiveEnv(device=device)

print("Testing reset...")
td = env.reset()
print(f"Reset keys: {list(td.keys())}")

print("\nTesting step...")
action = torch.tensor(0, dtype=torch.long, device=device)
td["action"] = action
td_next = env.step(td)

print(f"\nStep return keys: {list(td_next.keys())}")
print(f"Step return type: {type(td_next)}")

if 'next' in td_next.keys():
    print(f"\n'next' keys: {list(td_next['next'].keys())}")

# Try to find reward
for key in td_next.keys():
    print(f"\nKey '{key}': {type(td_next[key])}")
    if hasattr(td_next[key], 'keys'):
        print(f"  Sub-keys: {list(td_next[key].keys())}")
