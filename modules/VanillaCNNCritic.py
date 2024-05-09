import torch
import torch.nn as nn
from modules.VanillaCNN_base import VanillaCNN


# The critic module for SAC is now super straightforward:
class VanillaCNNQFunction(nn.Module):
    """
    Critic module for SAC.
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = VanillaCNN(q_net=True)  # q_net is True for a critic module

    def forward(self, obs, act):
        """
        Estimates the action-value of the (obs, act) state-action pair.

        In RL theory, the action-value is the expected sum of (gamma-discounted) future rewards
        when observing obs, taking action act, and following the current policy ever after.

        Args:
            obs: current observation
            act: tried next action

        Returns:
            The action-value of act in situation obs, as estimated by our critic network
        """
        # Since q_net is True, we append our action act to our observation obs.
        # Note that obs is a tuple of batched tensors: respectively the history of 4 images, speed, etc.
        x = (*obs, act)
        q = self.net(x)
        return torch.squeeze(q, -1)
