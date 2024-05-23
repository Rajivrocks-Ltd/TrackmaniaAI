import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from tmrl.util import prod
from tmrl.actor import TorchActorModule

# Global constants
LOG_STD_MAX = 2     # for clamping the log std of the action distribution
LOG_STD_MIN = -20   # for clamping the log std of the action distribution
EPSILON     = 1e-7  # a small value to avoid division by zero

def mlp(sizes, activation, output_activation=nn.Identity):
    """
    Helper function to define a sequential NN with specified layer sizes and activation functions.

    :param sizes: list of integers, specifying the number of units in each layer
    :param activation: activation function to use
    :param output_activation: activation function to use on the output layer
    :return: a PyTorch model
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class PPOStochasticActor(TorchActorModule)
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()

        # Check if the observation space is a tuple or not, with a tuple meaning that
        # a batch of observations is passed to the model (called a trace).
        try:
            obs_dim         = sum(prod(s for s in obs_space.shape) for obs_space in observation_space)
            self.tuple_obs  = True
        except TypeError:
            obs_dim         = prod(observation_space.shape)
            self.tuple_obs  = False

        act_dim     = action_space.shape[0] # number of dimensions in the action space
        act_limit   = action_space.high[0]  # upper bound of the action space

        # Build the MLP for the actor network
        # This network needs to produce mean and log std for the action distribution.
        self.net    = mlp([obs_dim] + list(hidden_sizes) + [1], activation, nn.Identity)

        # Output layer for the mean of the action distribution
        self.mean_layer = nn.Linear(hidden_sizes[-1], act_dim)

        # Output layer for the log standard deviation of the action distribution
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)

        self.act_limit = act_limit

    def forward(self, observations, test=False):
        """
        Forward pass of the actor network, so it computes the mean and log std of the action distribution.

        :param observations: the observation(s) to pass through the
        :param test: whether to use the model in test mode
        :return: the mean and log std of the action distribution
        """

        # Check if the observation space is a tuple or not
        if self.tuple_obs:
            x = torch.cat(obs, -1)
        else:
            x = torch.flatten(obs, start_dim=1)

        # Pass the observation through the network
        actor_net_out = self.net(x)

        # To get the probability distribution of the action, we need to calculate the mean and log std
        mean    = self.mean_layer(actor_net_out)
        log_std = self.log_std_layer(actor_net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std

    def get_action(self, observations, deterministic=False):
        """
        Generates an action based on the current observation

        :param observations: the observation(s) to pass through the network
        :param deterministic: whether to sample from the mean of the action distribution
        :return: the action to take based on the observation
        """

        mean, log_std = self.forward(observations)
        std     = torch.exp(log_std)
        dist    = Normal(mean, std)

        if deterministic:
            action = mean
        else:
            # https://stackoverflow.com/questions/60533150/what-is-the-difference-between-sample-and-rsample
            action = dist.rsample() # random sampling with reparameterization from probability distribution

        # Bounding the actions within range -1 and 1, while self.act_limit matches the range defined in the environment.
        # Causes more stable learning.
        action = torch.tanh(action) * self.act_limit

        return action


    def get_log_prob(self, observations, actions, test=False):
        """
        Calculates the log probability of an action given an observation

        :param observations: the observation(s) to pass through the
        :param actions: the action(s) to calculate the log probability for given the observation
        :param test: whether to use the model in test mode
        :return: the log probability of the action given the observation
        """

        mean, log_std = self.forward(obs)

        std     = torch.exp(log_std)  # Convert log_std to std
        dist    = distributions.Normal(mean, std)

        log_prob = dist.log_prob(action).sum(axis=-1)  # Sum log probs for multi-dimensional actions

        return log_prob

class PPOValueCritic(nn.Module):
    """
    The PPOValueCritic class defines the value function network for the PPO algorithm.

    The purpose of the PPO Critic is to estimate the value of a state (i.e. the expected
    return from that state, so V(s)). This is used to calculate the advantage function, which
    is used to update the policy.

    :param observation_space: the observation space of the environment
    :param hidden_sizes: a tuple of integers, specifying the number of units in each hidden layer
    :param activation: the activation function to use
    """

    def __init__(self, observation_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        # Calls the constructor of the parent class (nn.Module), so as to properly
        # initialize the object as a PyTorch model.
        super().__init__()

        # Check if the observation space is a tuple or not, with a tuple meaning that
        # a batch of observations is passed to the model (called a trace).
        try:
            obs_dim         = sum(prod(s for s in obs_space.shape) for obs_space in observation_space)
            self.tuple_obs  = True
        except TypeError:
            obs_dim         = prod(observation_space.shape)
            self.tuple_obs  = False

        # Build the MLP for the value function network.
        # Outputs a single scalar value V(s) representing the expected return from a state s
        self.value_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, observations):
        """
        Forward pass of the value function network, so it computes value estimate V(s) for a given state s.

        :param observations: the observation(s) to pass through the network
        :return: the value estimate V(s) for the given state s
        """
        # Check if the observation space is a tuple or not
        if self.tuple_obs:
            x = torch.cat(obs, -1)
        else:
            x = torch.flatten(obs, start_dim=1)

        return torch.squeeze(self.value_net(x), -1)


class PPOActorCritic(nn.Module):
    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        # calls the constructor of the parent class (nn.Module), so as to properly
        # initialize the object as a PyTorch model.
        super().__init__()

        # The actor and critic networks
        self.actor  = PPOStochasticActor(observation_space, action_space, hidden_sizes, activation)
        self.critic = PPOValueCritic(observation_space, hidden_sizes, activation) #

    def act(self, obs, test=False):
        """
        Used to generate an action based on the current observation

        :param obs: an observation
        :param test: whether to use the model in test mode
        :return: the action to take based on the observation
        """

        with torch.no_grad():
            a, _ = self.actor(obs, test)
            res = a.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res