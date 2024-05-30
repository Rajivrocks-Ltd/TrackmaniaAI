import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from tmrl.util import prod
from tmrl.actor import TorchActorModule


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON = 1e-7

class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.3):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = np.copy(self.mu)

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class DeterministicMLPActor(TorchActorModule):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__(observation_space, action_space)
        try:
            dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
            self.tuple_obs = True
        except TypeError:
            dim_obs = prod(observation_space.shape)
            self.tuple_obs = False
        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self.net = mlp([dim_obs] + list(hidden_sizes) + [dim_act], activation, activation)
        # self.mu_layer = nn.Linear(hidden_sizes[-1], dim_act)
        # self.log_std_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.act_limit = act_limit

        self.noise = OUNoise(dim_act)

    def forward(self, obs, test=False):
        x = torch.cat(obs, -1) if self.tuple_obs else torch.flatten(obs, start_dim=1)
        mu = self.net(x)
        pi_action = torch.tanh(mu)
        pi_action = self.act_limit * pi_action

        return pi_action

    def act(self, obs, test=False):
        with torch.no_grad():
            a = self.forward(obs, test)
            a = a.cpu().numpy()
            noise = self.noise.sample()
            a += noise
            a = np.clip(a, -self.act_limit, self.act_limit)
            res = a.squeeze()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res


class MLPQFunction(nn.Module):
    def __init__(self, obs_space, act_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        try:
            obs_dim = sum(prod(s for s in space.shape) for space in obs_space)
            self.tuple_obs = True
        except TypeError:
            obs_dim = prod(obs_space.shape)
            self.tuple_obs = False
        act_dim = act_space.shape[0]
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        x = torch.cat((*obs, act), -1) if self.tuple_obs else torch.cat((torch.flatten(obs, start_dim=1), act), -1)
        q = self.q(x)
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.  # FIXME: understand this


class MLPActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        # obs_dim = observation_space.shape[0]
        # act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.actor = DeterministicMLPActor(observation_space, action_space, hidden_sizes, activation)
        self.critic = MLPQFunction(observation_space, action_space, hidden_sizes, activation)

    def act(self, obs, test=False):
        with torch.no_grad():
            a = self.actor(obs, test)
            res = a.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res
