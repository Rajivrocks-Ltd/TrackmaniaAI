import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from tmrl.util import prod
from tmrl.actor import TorchActorModule
import tmrl.config.config_constants as cfg

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON = 1e-7

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from tmrl.util import prod
from tmrl.actor import TorchActorModule
import tmrl.config.config_constants as cfg

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON = 1e-7

def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class PPOStochasticActor(TorchActorModule):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__(observation_space, action_space)

        try:
            obs_dim = sum(prod(s for s in obs_space.shape) for obs_space in observation_space)
            self.tuple_obs = True
        except TypeError:
            obs_dim = prod(observation_space.shape)
            self.tuple_obs = False

        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        self.net = mlp([obs_dim] + list(hidden_sizes), activation)
        self.mean_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

        self.act_limit = act_limit

    def forward(self, obs, test=False):
        if self.tuple_obs:
            x = torch.cat([torch.flatten(o, start_dim=1) for o in obs], -1)
        else:
            x = torch.flatten(obs, start_dim=1)

        actor_net_out = self.net(x)
        mean = self.mean_layer(actor_net_out)
        log_std = self.log_std_layer(actor_net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def act(self, obs, test=False, deterministic=False):
        with torch.no_grad():
            mean, log_std = self.forward(obs, test)
            std = torch.exp(log_std)
            dist = Normal(mean, std)

            if deterministic:
                action = mean
            else:
                action = dist.rsample()

            action = torch.tanh(action) * self.act_limit

            return action

    def get_log_prob(self, obs, actions):
        mean, log_std = self.forward(obs, False)
        std = torch.exp(log_std)
        dist = Normal(mean, std)

        log_prob = dist.log_prob(actions).sum(axis=-1)

        return log_prob
    
        
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.device = device
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to_device(device)
        return self

class PPOValueCritic(nn.Module):
    def __init__(self, observation_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        try:
            obs_dim = sum(prod(s for s in obs_space.shape) for obs_space in observation_space)
            self.tuple_obs = True
        except TypeError:
            obs_dim = prod(observation_space.shape)
            self.tuple_obs = False

        self.value_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        if self.tuple_obs:
            x = torch.cat([torch.flatten(o, start_dim=1) for o in obs], -1)
        else:
            x = torch.flatten(obs, start_dim=1)

        return torch.squeeze(self.value_net(x), -1)

class PPOActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        self.actor = PPOStochasticActor(observation_space, action_space, hidden_sizes, activation)
        self.critic = PPOValueCritic(observation_space, hidden_sizes, activation)

    def act(self, obs, test=False):
        with torch.no_grad():
            a = self.actor(obs, test)
            res = a.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res
