from tmrl.actor import TorchActorModule
import torch
import torch.nn as nn
from CNN import VanillaCNN2
from OUNoise import OUNoise
import numpy as np


class MyActorModuleDDPG(TorchActorModule):
    """
    Deterministic policy actor for DDPG, utilizing the VanillaCNN2 architecture.
    """

    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        dim_act = action_space.shape[0]  # Dimensionality of actions
        act_limit = action_space.high[0]  # Maximum amplitude of actions

        # Our hybrid CNN+MLP policy configured as an actor
        self.net = VanillaCNN2(actor=True)  # Instantiate VanillaCNN2 as an actor
        self.mu_layer = nn.Linear(256, dim_act)  # Output layer for actions

        self.act_limit = act_limit

        self.noise = OUNoise(dim_act)

    def save(self, path):
        # with open(path, 'w') as json_file:
        #     json.dump(self.state_dict(), json_file, cls=TorchJSONEncoder)
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.device = device
        # with open(path, 'r') as json_file:
        #     state_dict = json.load(json_file, cls=TorchJSONDecoder)
        # self.load_state_dict(state_dict)
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to_device(device)
        return self

    def forward(self, obs, test=False):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)  # Compute deterministic actions
        pi_action = torch.tanh(mu)  # Using tanh to ensure the action is within [-1, 1]
        pi_action = self.act_limit * pi_action  # Scale actions to the actual limits of the action space
        pi_action = pi_action.squeeze()  # Remove unnecessary batch dimensions if any
        return pi_action

    def act(self, obs, test=False):
        """
        This function produces a deterministic action from the policy network
        given the observation.
        """
        with torch.no_grad():
            a = self.forward(obs=obs, test=test)
            a = a.cpu().numpy()
            noise = self.noise.sample()
            a += noise
            return np.clip(a, -self.act_limit, self.act_limit)
