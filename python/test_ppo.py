"""
Test script to verify PPO implementation components.

This script tests the environment, models, and basic training loop
to ensure everything is working correctly.
"""

import sys
import torch

print("Testing PPO Implementation...")
print("=" * 60)

# Test 1: Import all modules
print("\n1. Testing imports...")
try:
    from hive_env import HiveEnv
    from ppo_models import make_ppo_models
    from self_play_evaluator import SelfPlayEvaluator
    import nokamute
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Environment creation and reset
print("\n2. Testing environment...")
try:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = HiveEnv(device=device)
    td = env.reset()
    print(f"✓ Environment reset successful")
    print(f"  - Observation keys: {td.keys()}")
    print(f"  - Node features shape: {td['node_features'].shape}")
    print(f"  - Action mask sum: {td['action_mask'].sum().item()}")
except Exception as e:
    print(f"✗ Environment test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Environment step
print("\n3. Testing environment step...")
try:
    action = torch.tensor(0, dtype=torch.long, device=device)
    td["action"] = action
    td_next = env.step(td)
    print(f"✓ Environment step successful")
    print(f"  - Reward: {td_next['next']['reward'].item()}")
    print(f"  - Done: {td_next['next']['done'].item()}")
except Exception as e:
    print(f"✗ Environment step failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Model creation
print("\n4. Testing model creation...")
try:
    actor, critic = make_ppo_models(device=device)
    print(f"✓ Models created successfully")
    print(f"  - Actor params: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"  - Critic params: {sum(p.numel() for p in critic.parameters()):,}")
except Exception as e:
    print(f"✗ Model creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Model forward pass
print("\n5. Testing model forward pass...")
try:
    td_reset = env.reset()
    # Set models to eval mode to avoid BatchNorm issues with batch_size=1
    actor.eval()
    critic.eval()
    with torch.no_grad():
        td_with_action = actor(td_reset)
        td_with_value = critic(td_reset)
    # Set back to train mode
    actor.train()
    critic.train()
    print(f"✓ Model forward pass successful")
    print(f"  - Action shape: {td_with_action['action'].shape}")
    print(f"  - Value shape: {td_with_value['state_value'].shape}")
except Exception as e:
    print(f"✗ Model forward pass failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Self-play evaluation
print("\n6. Testing self-play evaluation...")
try:
    evaluator = SelfPlayEvaluator(device=device)
    results = evaluator.evaluate(
        agent1=actor,
        agent2=actor,
        num_games=3,
        verbose=False,
    )
    print(f"✓ Self-play evaluation successful")
    print(f"  - Games played: 3")
    print(f"  - Avg game length: {results['avg_game_length']:.1f}")
except Exception as e:
    print(f"✗ Self-play evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Check TorchRL components
print("\n7. Testing TorchRL components...")
try:
    from torchrl.envs import TransformedEnv, StepCounter
    from torchrl.objectives import ClipPPOLoss
    from graph_gae import GraphGAE
    
    env_transformed = TransformedEnv(env, StepCounter())
    
    # Use custom GraphGAE instead of TorchRL's vmap-based GAE
    advantage_module = GraphGAE(
        value_network=critic,
        gamma=0.99,
        lmbda=0.95,
        device=device,
    )
    
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coef=1e-4,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )
    
    print(f"✓ TorchRL components created successfully")
    print(f"  - Using custom GraphGAE (no vmap)")
except Exception as e:
    print(f"✗ TorchRL components failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Mini training loop
print("\n8. Testing mini training loop...")
try:
    from torchrl.collectors import SyncDataCollector
    from torchrl.data.replay_buffers import ReplayBuffer
    from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
    from torchrl.data.replay_buffers.storages import LazyTensorStorage
    
    # Create small collector
    collector = SyncDataCollector(
        env_transformed,
        actor,
        frames_per_batch=100,
        total_frames=200,
        split_trajs=False,
        device=device,
    )
    
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=100),
        sampler=SamplerWithoutReplacement(),
    )
    
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=1e-3)
    
    # Collect one batch
    for i, tensordict_data in enumerate(collector):
        if i >= 1:  # Just do one iteration
            break
        
        # Compute advantages
        with torch.no_grad():
            advantage_module(tensordict_data)
        
        # Add to replay buffer
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        
        # Sample and train
        subdata = replay_buffer.sample(32)
        loss_vals = loss_module(subdata.to(device))
        
        loss_value = (
            loss_vals["loss_objective"]
            + loss_vals["loss_critic"]
            + loss_vals["loss_entropy"]
        )
        
        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()
        
        print(f"✓ Mini training loop successful")
        print(f"  - Loss: {loss_value.item():.4f}")
        print(f"  - Batch size: {tensordict_data.batch_size}")
    
    collector.shutdown()
    
except Exception as e:
    print(f"✗ Mini training loop failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nThe PPO implementation is ready to use.")
print("\nTo start training, run:")
print("  python train_ppo.py --total-frames 100000")
