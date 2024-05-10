from tmrl.actor import TorchActorModule
from tmrl.util import prod
from auxiliary.create_jsons import TorchJSONDecoder, TorchJSONEncoder
from modules.DuelingCNN import DuelingCNN, mlp
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20
EPSILON = 1e-7

class MyDuelingDQNActorModule(TorchActorModule):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__(observation_space, action_space)

        # Copied from custom_models.py file
        try:
            dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
            self.tuple_obs = True
        except TypeError:
            dim_obs = prod(observation_space.shape)
            self.tuple_obs = False

        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self.net = mlp([dim_obs] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.act_limit = act_limit

        # And initialize our attributes:
        #dim_act = action_space.shape[0]  # dimensionality of actions

        # Initialize the Dueling DQN network
        # Anca DQN with Pixel data
        #self.net = DuelingCNN(dim_act)
        #self.net = mlp([dim_obs] + list(hidden_sizes), activation, activation)

        # # Initialize parameters for action selection
        # self.eps_start = 0.9
        # self.eps_end = 1e-3
        # self.eps_decay = 200
        # self.step = 0
        # self.action_correspondance = {
        #     i + 2 * j + 4 * k + 8 * l: [i, j, k, l]
        #     for i in range(2)
        #     for j in range(2)
        #     for k in range(2)
        #     for l in range(2)
        # }

    def save(self, path):
        with open(path, 'w') as json_file:
            json.dump(self.state_dict(), json_file, cls=TorchJSONEncoder)

    def load(self, path, device):
        print("test path: ", path)
        self.device = device
        with open(path, 'r') as json_file:
            state_dict = json.load(json_file, cls=TorchJSONDecoder)
        self.load_state_dict(state_dict)
        self.to_device(device)
        return self

    def epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -1.0 * self.step / self.eps_decay)

    def act(self, obs,  test=False):
        # if np.random.rand() < self.epsilon():
        #     self.step += 1
        #     return self.action_correspondance[
        #         np.argmax(self.net(obs).detach().cpu().numpy())
        #     ]
        # self.step += 1
        # return self.action_correspondance[
        #     np.random.randint(0, len(self.action_correspondance))
        # ]
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            res = a.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res

    # def forward(self, obs):
    #     x = torch.cat(obs, -1) if self.tuple_obs else torch.flatten(obs, start_dim=1)
    #     return self.net(x)


    def forward(self, obs, test=False, with_logprob=True):
        x = torch.cat(obs, -1) if self.tuple_obs else torch.flatten(obs, start_dim=1)
        net_out = self.net(x)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if test:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        # pi_action = pi_action.squeeze()

        return pi_action, logp_pi


