import torch
import numpy as np

# Simulate what happens in our model
action_mask = torch.tensor([1, 0, 1, 0, 1], dtype=torch.float32)
q_values = torch.tensor([0.5, 0.3, 0.4, 0.6, 0.2], dtype=torch.float32)

print("Original Q-values:", q_values)
print("Action mask:", action_mask)

# Our current approach
inf_mask = torch.clamp(torch.log(action_mask), -1e10, 0.0)
print("Log mask:", inf_mask)

masked_q = q_values + inf_mask
print("Masked Q-values:", masked_q)

# What argmax gives us
action = torch.argmax(masked_q)
print("Selected action:", action.item())
print("Is legal?:", action_mask[action].item() == 1)
