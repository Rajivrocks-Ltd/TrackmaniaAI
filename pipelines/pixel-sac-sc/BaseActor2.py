from tmrl.actor import TorchActorModule
import torch
import torch.nn as nn
from VanillaCNN2 import VanillaCNN2
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

# The following constants are from the Spinup implementation of SAC
# that we simply copy/paste and adapt in this tutorial.
LOG_STD_MAX = 2
LOG_STD_MIN = -20


class MyActorModule2(TorchActorModule):
    """
    Our policy wrapped in the TMRL ActorModule class.

    The only required method is ActorModule.act().
    We also implement a forward() method for our training algorithm.

    (Note: TorchActorModule is a subclass of ActorModule and torch.nn.Module)
    """

    def __init__(self, observation_space, action_space):
        """
        When implementing __init__, we need to take the observation_space and action_space arguments.

        Args:
            observation_space: observation space of the Gymnasium environment
            action_space: action space of the Gymnasium environment
        """
        # We must call the superclass __init__:
        super().__init__(observation_space, action_space)

        # And initialize our attributes:
        dim_act = action_space.shape[0]  # dimensionality of actions
        act_limit = action_space.high[0]  # maximum amplitude of actions
        # Our hybrid CNN+MLP policy:
        self.net = VanillaCNN2(q_net=False)
        # The policy output layer, which samples actions stochastically in a gaussian, with means...:
        self.mu_layer = nn.Linear(256, dim_act)
        # ... and log standard deviations:
        self.log_std_layer = nn.Linear(256, dim_act)
        # We will squash this within the action space thanks to a tanh final activation:
        self.act_limit = act_limit

    def save(self, path):

        # with open(path, 'w') as json_file:
        #     json.dump(self.state_dict(), json_file, cls=TorchJSONEncoder)
        torch.save(self.state_dict(), path)

    import json
    def load(self, path, device):

        self.device = device
        # with open(path, 'r') as json_file:
        #     state_dict = json.load(json_file, cls=TorchJSONDecoder)
        # self.load_state_dict(state_dict)
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to_device(device)
        return self

    def forward(self, obs, test=False, compute_logprob=True):

        net_out = self.net(obs)
        # Now, the means of our multivariate gaussian (i.e., Normal law) are:
        mu = self.mu_layer(net_out)
        # And the corresponding standard deviations are:
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        # We can now sample our action in the resulting multivariate gaussian (Normal) distribution:
        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu  # at test time, our action is deterministic (it is just the means)
        else:
            pi_action = pi_distribution.rsample()  # during training, it is sampled in the multivariate gaussian
        # We retrieve the log probabilities of our multivariate gaussian as they will be useful for SAC:
        if compute_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        # And we squash our action within the action space:
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        # Finally, we remove the batch dimension:
        pi_action = pi_action.squeeze()
        return pi_action, logp_pi

    def act(self, obs, test=False):
        """
        Args:
            obs (object): the input observation (when using TorchActorModule, this is a torch.Tensor)
            test (bool): True at test-time (e.g., during evaluation...), False otherwise

        Returns:
            act (numpy.array): the computed action, in the form of a numpy array of 3 values between -1.0 and 1.0
        """
        with torch.no_grad():
            a, _ = self.forward(obs=obs, test=test, compute_logprob=False)
            return a.cpu().numpy()
