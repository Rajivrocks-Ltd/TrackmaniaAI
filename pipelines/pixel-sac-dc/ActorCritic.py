import torch.nn as nn
from VanillaCNNCritic import VanillaCNNQFunction
from baseActor import MyActorModule


# Finally, let us merge this together into an actor-critic torch.nn.module for training.
# Classically, we use one actor and two parallel critics to alleviate the overestimation bias.
class VanillaCNNActorCritic(nn.Module):
    """
    Actor-critic module for the SAC algorithm.
    """
    def __init__(self, observation_space, action_space):
        super().__init__()

        # Policy network (actor):
        self.actor = MyActorModule(observation_space, action_space)
        # Value networks (critics):
        self.q1 = VanillaCNNQFunction(observation_space, action_space)
        self.q2 = VanillaCNNQFunction(observation_space, action_space)
