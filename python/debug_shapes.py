"""Debug script to check tensor shapes in collected data."""
import torch
from hive_env import HiveEnv
from ppo_models import make_ppo_models
from torchrl.envs import TransformedEnv, StepCounter
from torchrl.collectors import SyncDataCollector

device = "cpu"
env = HiveEnv(device=device)
actor, critic = make_ppo_models(device=device)

env_transformed = TransformedEnv(env, StepCounter())

collector = SyncDataCollector(
    env_transformed,
    actor,
    frames_per_batch=10,
    total_frames=20,
    split_trajs=False,
    device=device,
)

print("Collecting data...")
for i, tensordict_data in enumerate(collector):
    if i >= 1:
        break
    
    print(f"\nCollected batch:")
    print(f"Batch size: {tensordict_data.batch_size}")
    print(f"Keys: {list(tensordict_data.keys())}")
    print(f"\nnode_features shape: {tensordict_data['node_features'].shape}")
    print(f"edge_index shape: {tensordict_data['edge_index'].shape}")
    print(f"num_nodes shape: {tensordict_data['num_nodes'].shape}")
    print(f"reward shape: {tensordict_data['next']['reward'].shape}")
    
    # Try to access individual samples
    print(f"\nFirst sample node_features shape: {tensordict_data['node_features'][0].shape}")
    print(f"First sample num_nodes: {tensordict_data['num_nodes'][0]}")

collector.shutdown()
