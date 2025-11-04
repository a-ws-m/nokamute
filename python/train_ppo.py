"""
PPO training script for Hive using TorchRL.

This script implements PPO (Proximal Policy Optimization) training with:
- Self-play data collection
- Graph neural network policy and value functions
- Model acceptance gate (55% win rate threshold)
- TensorBoard logging
"""

import argparse
import os
import time
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.utils.tensorboard import SummaryWriter

from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs import TransformedEnv, StepCounter
from torchrl.envs.utils import set_exploration_type, ExplorationType
from torchrl.objectives import ClipPPOLoss
from graph_gae import GraphGAE, normalize_advantages

from hive_env import HiveEnv
from ppo_models import make_ppo_models
from self_play_evaluator import SelfPlayEvaluator


def train_ppo(args):
    """Main PPO training loop."""
    
    # Setup
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.checkpoint_dir, "logs"))
    
    # Create environment with step counter
    base_env = HiveEnv(
        max_actions=args.max_actions,
        max_nodes=args.max_nodes,
        max_edges=args.max_edges,
        max_moves=args.max_moves,
        device=device,
    )
    
    env = TransformedEnv(base_env, StepCounter())
    
    print(f"Environment created")
    print(f"  Observation spec: {env.observation_spec}")
    print(f"  Action spec: {env.action_spec}")
    
    # Create actor and critic
    actor, critic = make_ppo_models(
        node_features=11,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        max_actions=args.max_actions,
        device=device,
    )
    
    print(f"Actor parameters: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic parameters: {sum(p.numel() for p in critic.parameters()):,}")
    
    # Create custom GAE module for advantage estimation (no vmap for GNN compatibility)
    advantage_module = GraphGAE(
        value_network=critic,
        gamma=args.gamma,
        lmbda=args.lmbda,
        device=device,
    )
    
    # Create PPO loss
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=args.clip_epsilon,
        entropy_bonus=bool(args.entropy_coef),
        entropy_coef=args.entropy_coef,
        critic_coef=1.0,
        loss_critic_type="smooth_l1",
    )
    
    # Create optimizer
    optim = torch.optim.Adam(loss_module.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=args.total_frames // args.frames_per_batch,
        eta_min=0.0,
    )
    
    # Create data collector
    collector = SyncDataCollector(
        env,
        actor,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.total_frames,
        split_trajs=False,
        device=device,
    )
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=args.frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )
    
    # Self-play evaluator
    evaluator = SelfPlayEvaluator(device=device)
    
    # Best model tracking
    best_actor_state = None
    current_actor_state = {k: v.cpu().clone() for k, v in actor.state_dict().items()}
    
    # Training logs
    logs = defaultdict(list)
    total_updates = 0
    iteration = 0
    
    print(f"\nStarting training...")
    print(f"Total frames: {args.total_frames}")
    print(f"Frames per batch: {args.frames_per_batch}")
    print(f"Iterations: {args.total_frames // args.frames_per_batch}")
    
    # Training loop
    for i, tensordict_data in enumerate(collector):
        iteration = i
        
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}")
        print(f"{'='*60}")
        
        # Compute advantages
        with torch.no_grad():
            advantage_module(tensordict_data)
        
        # Flatten data for training
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        
        # Training epochs
        epoch_losses = []
        for epoch in range(args.num_epochs):
            # Sample mini-batches and train
            batch_losses = []
            for _ in range(args.frames_per_batch // args.sub_batch_size):
                subdata = replay_buffer.sample(args.sub_batch_size)
                loss_vals = loss_module(subdata.to(device))
                
                loss_value = (
                    loss_vals["loss_objective"]
                    + loss_vals["loss_critic"]
                    + loss_vals["loss_entropy"]
                )
                
                # Backward pass
                optim.zero_grad()
                loss_value.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(),
                    args.max_grad_norm,
                )
                
                optim.step()
                
                batch_losses.append(loss_value.item())
                total_updates += 1
            
            epoch_loss = sum(batch_losses) / len(batch_losses)
            epoch_losses.append(epoch_loss)
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/{args.num_epochs}: Loss = {epoch_loss:.4f}")
        
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        
        # Log metrics
        logs["loss"].append(avg_loss)
        logs["lr"].append(optim.param_groups[0]["lr"])
        
        # Get episode statistics
        if "reward" in tensordict_data.keys():
            episode_reward = tensordict_data["reward"].sum().item()
            logs["episode_reward"].append(episode_reward)
        
        # TensorBoard logging
        writer.add_scalar("Loss/train", avg_loss, iteration)
        writer.add_scalar("Learning_rate", logs["lr"][-1], iteration)
        
        if "episode_reward" in logs and len(logs["episode_reward"]) > 0:
            writer.add_scalar("Reward/episode", logs["episode_reward"][-1], iteration)
        
        # Update learning rate
        scheduler.step()
        
        # Periodic evaluation and model acceptance
        if (iteration + 1) % args.eval_interval == 0:
            print(f"\n{'='*60}")
            print(f"SELF-PLAY EVALUATION")
            print(f"{'='*60}")
            
            # Load previous best model for comparison
            if best_actor_state is None:
                # First evaluation - accept current model as baseline
                print("First evaluation - accepting current model as baseline")
                best_actor_state = {k: v.cpu().clone() for k, v in actor.state_dict().items()}
                accepts = True
            else:
                # Create copy of old actor for comparison
                old_actor, _ = make_ppo_models(
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers,
                    num_heads=args.num_heads,
                    dropout=args.dropout,
                    max_actions=args.max_actions,
                    device=device,
                )
                old_actor.load_state_dict(best_actor_state)
                
                # Check if new model beats old model
                accepts = evaluator.check_improvement_threshold(
                    new_agent=actor,
                    old_agent=old_actor,
                    num_games=args.eval_games,
                    threshold=args.acceptance_threshold,
                    verbose=True,
                )
                
                if accepts:
                    print(f"✓ New model ACCEPTED (beats old model at {args.acceptance_threshold:.0%} threshold)")
                    best_actor_state = {k: v.cpu().clone() for k, v in actor.state_dict().items()}
                else:
                    print(f"✗ New model REJECTED (does not beat old model at {args.acceptance_threshold:.0%} threshold)")
                    print(f"  Reverting to previous best model...")
                    actor.load_state_dict(best_actor_state)
            
            # Log acceptance
            writer.add_scalar("Evaluation/model_accepted", 1 if accepts else 0, iteration)
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                f"checkpoint_iter_{iteration}.pt",
            )
            
            torch.save(
                {
                    "iteration": iteration,
                    "actor_state_dict": actor.state_dict(),
                    "critic_state_dict": critic.state_dict(),
                    "optimizer_state_dict": optim.state_dict(),
                    "best_actor_state_dict": best_actor_state,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved: {checkpoint_path}")
            
            # Save best model separately
            best_model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
            torch.save(
                {
                    "actor_state_dict": best_actor_state,
                    "args": vars(args),
                },
                best_model_path,
            )
            print(f"Best model saved: {best_model_path}")
    
    # Final save
    final_path = os.path.join(args.checkpoint_dir, "final_model.pt")
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "best_actor_state_dict": best_actor_state,
            "args": vars(args),
        },
        final_path,
    )
    
    collector.shutdown()
    writer.close()
    
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Total iterations: {iteration + 1}")
    print(f"Total updates: {total_updates}")
    print(f"Final model saved: {final_path}")
    print(f"Best model saved: {best_model_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Hive agent with PPO")
    
    # Environment parameters
    parser.add_argument("--max-actions", type=int, default=500, help="Maximum actions")
    parser.add_argument("--max-nodes", type=int, default=200, help="Maximum graph nodes")
    parser.add_argument("--max-edges", type=int, default=1000, help="Maximum graph edges")
    parser.add_argument("--max-moves", type=int, default=200, help="Maximum moves per game")
    
    # Model parameters
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of GNN layers")
    parser.add_argument("--num-heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # PPO parameters
    parser.add_argument("--frames-per-batch", type=int, default=1000, help="Frames per batch")
    parser.add_argument("--total-frames", type=int, default=100000, help="Total training frames")
    parser.add_argument("--sub-batch-size", type=int, default=64, help="Sub-batch size")
    parser.add_argument("--num-epochs", type=int, default=10, help="Epochs per iteration")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lmbda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--entropy-coef", type=float, default=1e-4, help="Entropy coefficient")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")
    
    # Evaluation parameters
    parser.add_argument("--eval-interval", type=int, default=5, help="Evaluation interval")
    parser.add_argument("--eval-games", type=int, default=50, help="Games per evaluation")
    parser.add_argument("--acceptance-threshold", type=float, default=0.55, help="Win rate threshold")
    
    # System parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_ppo", help="Checkpoint directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    
    args = parser.parse_args()
    
    train_ppo(args)


if __name__ == "__main__":
    main()
