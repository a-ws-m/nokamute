"""
Train a DQN agent on Hive using RLlib with custom GNN model.

This script demonstrates how to train a DQN agent on the Hive board game
using RLlib's old API stack with a custom Graph Neural Network model.
"""

import argparse
import os

import ray
from pettingzoo_env.hive_env import env as hive_env
from ray import tune
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from rllib_dqn import HiveGNNModel


def train_dqn(
    num_iterations: int = 100,
    checkpoint_freq: int = 10,
    checkpoint_dir: str = "./checkpoints/dqn_hive",
    hidden_dim: int = 128,
    num_layers: int = 4,
    num_heads: int = 4,
    learning_rate: float = 0.0005,
    gamma: float = 0.99,
    train_batch_size: int = 32,
    replay_buffer_capacity: int = 100000,
    num_workers: int = 4,
    num_gpus: int = 0,
    exploration_fraction: float = 0.3,
    final_epsilon: float = 0.05,
):
    """
    Train a DQN agent on Hive.
    
    Args:
        num_iterations: Number of training iterations
        checkpoint_freq: How often to save checkpoints
        checkpoint_dir: Directory to save checkpoints
        hidden_dim: Hidden dimension for GNN
        num_layers: Number of GNN layers
        num_heads: Number of attention heads
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        train_batch_size: Batch size for training
        replay_buffer_capacity: Replay buffer capacity
        num_workers: Number of parallel workers for rollouts
        num_gpus: Number of GPUs to use
        exploration_fraction: Fraction of training for epsilon decay
        final_epsilon: Final epsilon value for exploration
    """
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    # Create environment creator
    def env_creator(config=None):
        return hive_env(
            game_type="Base+MLP",
            max_moves=400,
            render_mode=None,
        )
    
    # Register environment
    env_name = "hive_v0"
    register_env(env_name, lambda config: PettingZooEnv(env_creator()))
    
    # Create a test environment to get spaces
    test_env = PettingZooEnv(env_creator())
    observation_space = test_env.observation_space
    action_space = test_env.action_space
    
    print(f"Observation space: {observation_space}")
    print(f"Action space: {action_space}")
    
    # Register custom model
    ModelCatalog.register_custom_model("hive_gnn_model", HiveGNNModel)
    
    # Configure DQN with parameter sharing (both players use same policy)
    config = (
        DQNConfig()
        .environment(
            env=env_name,
        )
        .framework("torch")
        .multi_agent(
            # Single shared policy for both players
            policies={"shared_policy"},
            # All agents map to the same policy (parameter sharing)
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: "shared_policy"),
        )
        .training(
            gamma=gamma,
            lr=learning_rate,
            train_batch_size=train_batch_size,
            replay_buffer_config={
                "type": "MultiAgentReplayBuffer",
                "capacity": min(replay_buffer_capacity, 10000),  # Limit to 10k for memory constraints
            },
            # Double DQN
            double_q=True,
            # Dueling DQN
            dueling=True,
            # Target network update frequency
            target_network_update_freq=500,
            # Set hiddens=[] so num_outputs matches action space size (5985)
            # Our custom model will handle the architecture internally
            hiddens=[],
        )
        .rollouts(
            num_rollout_workers=num_workers,
            num_envs_per_worker=1,
            rollout_fragment_length=200,
        )
        .resources(
            num_gpus=num_gpus,
            num_cpus_per_worker=1,
        )
        .reporting(
            min_time_s_per_iteration=0,
            min_sample_timesteps_per_iteration=200,
        )
        .debugging(
            log_level="INFO",
        )
        .exploration(
            exploration_config={
                "type": "EpsilonGreedy",
                "initial_epsilon": 0.0,  # Start with greedy (model-based) actions for testing
                "final_epsilon": final_epsilon,
                "epsilon_timesteps": int(exploration_fraction * num_iterations * train_batch_size * max(num_workers, 1)),
            }
        )
        # Configure custom model
        .update_from_dict({
            "model": {
                "custom_model": "hive_gnn_model",
                "custom_model_config": {
                    "node_features": 12,
                    "hidden_dim": hidden_dim,
                    "num_layers": num_layers,
                    "num_heads": num_heads,
                    "dropout": 0.1,
                },
            }
        })
    )
    
    # Build the algorithm
    print("Building DQN algorithm...")
    algo = config.build()
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    print(f"Starting training for {num_iterations} iterations...")
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"Iteration {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")
        
        # Train
        result = algo.train()
        
        # Print results
        print(f"\nEpisode Reward Mean: {result.get('env_runners/episode_return_mean', 'N/A')}")
        print(f"Episode Length Mean: {result.get('env_runners/episode_len_mean', 'N/A')}")
        print(f"Episodes This Iter: {result.get('env_runners/num_episodes', 'N/A')}")
        print(f"Timesteps Total: {result.get('num_env_steps_sampled_lifetime', 'N/A')}")
        print(f"Learning Rate: {result.get('info/learner/default_policy/cur_lr', 'N/A')}")
        
        # Save checkpoint
        if (iteration + 1) % checkpoint_freq == 0:
            checkpoint_path = algo.save(checkpoint_dir)
            print(f"\nCheckpoint saved at: {checkpoint_path}")
    
    # Final checkpoint
    final_checkpoint = algo.save(checkpoint_dir)
    print(f"\nFinal checkpoint saved at: {final_checkpoint}")
    
    # Cleanup
    algo.stop()
    ray.shutdown()
    
    return final_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train DQN agent on Hive")
    
    # Training parameters
    parser.add_argument("--num-iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--checkpoint-freq", type=int, default=10,
                        help="Checkpoint frequency (iterations)")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/dqn_hive",
                        help="Checkpoint directory")
    
    # Model parameters
    parser.add_argument("--hidden-dim", type=int, default=128,
                        help="Hidden dimension for GNN")
    parser.add_argument("--num-layers", type=int, default=4,
                        help="Number of GNN layers")
    parser.add_argument("--num-heads", type=int, default=4,
                        help="Number of attention heads")
    
    # DQN parameters
    parser.add_argument("--learning-rate", type=float, default=0.0005,
                        help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--train-batch-size", type=int, default=32,
                        help="Training batch size")
    parser.add_argument("--replay-buffer-capacity", type=int, default=100000,
                        help="Replay buffer capacity")
    parser.add_argument("--exploration-fraction", type=float, default=0.3,
                        help="Fraction of training for epsilon decay")
    parser.add_argument("--final-epsilon", type=float, default=0.05,
                        help="Final epsilon value")
    
    # Resource parameters
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of parallel workers")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    # Train
    train_dqn(
        num_iterations=args.num_iterations,
        checkpoint_freq=args.checkpoint_freq,
        checkpoint_dir=args.checkpoint_dir,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        train_batch_size=args.train_batch_size,
        replay_buffer_capacity=args.replay_buffer_capacity,
        num_workers=args.num_workers,
        num_gpus=args.num_gpus,
        exploration_fraction=args.exploration_fraction,
        final_epsilon=args.final_epsilon,
    )


if __name__ == "__main__":
    main()
