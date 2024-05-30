import torch.nn as nn
from CNN import VanillaCNNQFunctionDDPG
from ActorModuleDDPG import MyActorModuleDDPG


class VanillaCNNActorCriticDDPG(nn.Module):
    """
    Actor-critic module adapted for the DDPG algorithm.
    """

    def __init__(self, observation_space, action_space):
        super().__init__()

        # Policy network (actor) for deterministic action selection:
        self.actor = MyActorModuleDDPG(observation_space, action_space)
        # Critic network for evaluating state-action pairs:
        self.critic = VanillaCNNQFunctionDDPG(observation_space, action_space)
        # self.critic = VanillaCNN2(actor=False)

    # def forward(self, obs, act=None):
    #     """
    #     Forward pass through both actor and critic. The critic forward pass
    #     requires both obs and act, whereas the actor only needs obs.
    #
    #     Args:
    #         obs (torch.Tensor): The observation/state tensor.
    #         act (torch.Tensor): The action tensor, required for the critic.
    #
    #     Returns:
    #         actor_output (torch.Tensor): The deterministic action output by the actor.
    #         critic_output (torch.Tensor): The Q-value output by the critic given obs and act.
    #     """
    #     actor_output = self.actor(obs)
    #     critic_output = None
    #     if act is not None:  # Only compute critic output if an action is provided
    #         critic_output = self.critic(obs, act)
    #     return actor_output, critic_output
