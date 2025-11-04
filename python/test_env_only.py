"""Quick test for just the environment."""
import torch
from hive_env import HiveEnv

print("Testing HiveEnv creation...")
device = "cpu"
env = HiveEnv(device=device)
print(f"✓ Environment created")
print(f"  Action spec: {env.action_spec}")

print("\nTesting reset...")
td = env.reset()
print(f"✓ Reset successful")
print(f"  Keys: {list(td.keys())}")
print(f"  Action mask sum: {td['action_mask'].sum().item()}")

print("\nTesting step...")
action = torch.tensor(0, dtype=torch.long, device=device)
td["action"] = action
td_next = env.step(td)
print(f"✓ Step successful")
print(f"  Reward: {td_next['reward'].item()}")
print(f"  Done: {td_next['done'].item()}")

print("\n✓ Environment test passed!")
