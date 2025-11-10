"""
Custom epsilon-greedy exploration that respects action masks.
"""

import numpy as np
from ray.rllib.utils.exploration.epsilon_greedy import EpsilonGreedy
from ray.rllib.utils.framework import try_import_torch

torch, _ = try_import_torch()


class MaskedEpsilonGreedy(EpsilonGreedy):
    """
    Epsilon-greedy exploration that only selects from legal actions.
    
    During exploration (epsilon > 0), randomly samples from legal actions.
    During exploitation (epsilon == 0), selects the best Q-value among legal actions.
    """
    
    def get_exploration_action(
        self,
        *,
        action_distribution,
        timestep,
        explore=True,
    ):
        """
        Sample an action from the distribution, respecting action mask.
        
        Args:
            action_distribution: The action distribution to sample from
            timestep: Current global timestep
            explore: Whether to explore or exploit
            
        Returns:
            Tuple of (action, logp)
        """
        # Get epsilon for this timestep
        epsilon = self.epsilon_schedule(timestep)
        
        # Get the action mask from the model's input dict
        # The action mask should be in the last forward input dict
        action_mask = None
        if hasattr(self, "last_timestep") and hasattr(self.model, "last_input"):
            obs = self.model.last_input.get("obs", {})
            if isinstance(obs, dict) and "action_mask" in obs:
                action_mask = obs["action_mask"]
        
        # If we don't have an action mask, fall back to default behavior
        if action_mask is None:
            return super().get_exploration_action(
                action_distribution=action_distribution,
                timestep=timestep,
                explore=explore,
            )
        
        # Explore: random action from legal actions
        if explore and np.random.rand() < epsilon:
            # Get legal actions (where mask == 1)
            if torch.is_tensor(action_mask):
                action_mask = action_mask.cpu().numpy()
            
            legal_actions = np.where(action_mask.flatten() > 0)[0]
            
            if len(legal_actions) == 0:
                # No legal actions - this shouldn't happen, but fall back to random
                action = action_distribution.sample()
            else:
                # Uniformly sample from legal actions
                action = np.random.choice(legal_actions)
                if torch.is_tensor(action):
                    action = action.item()
            
            # Compute log probability (uniform over legal actions)
            logp = -np.log(len(legal_actions)) if len(legal_actions) > 0 else 0.0
            
            return action, logp
        
        # Exploit: use the action distribution (which should have masked Q-values)
        # The model's forward() should have already applied the action mask to Q-values
        action = action_distribution.deterministic_sample()
        logp = action_distribution.logp(action)
        
        return action, logp
