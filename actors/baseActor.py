from tmrl.actor import TorchActorModule
import torch
import torch.nn as nn
from auxiliary.create_jsons import TorchJSONDecoder, TorchJSONEncoder
from modules.VanillaCNN_base import VanillaCNN
import json
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np

# The following constants are from the Spinup implementation of SAC
# that we simply copy/paste and adapt in this tutorial.
LOG_STD_MAX = 2
LOG_STD_MIN = -20

class MyActorModule(TorchActorModule):
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
        self.net = VanillaCNN(q_net=False)
        # The policy output layer, which samples actions stochastically in a gaussian, with means...:
        self.mu_layer = nn.Linear(256, dim_act)
        # ... and log standard deviations:
        self.log_std_layer = nn.Linear(256, dim_act)
        # We will squash this within the action space thanks to a tanh final activation:
        self.act_limit = act_limit

    def save(self, path):
        """
        JSON-serialize a detached copy of the ActorModule and save it in path.

        IMPORTANT: FOR THE COMPETITION, WE ONLY ACCEPT JSON AND PYTHON FILES.
        IN PARTICULAR, WE *DO NOT* ACCEPT PICKLE FILES (such as output by torch.save()...).

        All your submitted files must be human-readable, for everyone's safety.
        Indeed, untrusted pickle files are an open door for hackers.

        Args:
            path: pathlib.Path: path to where the object will be stored.
        """
        with open(path, 'w') as json_file:
            json.dump(self.state_dict(), json_file, cls=TorchJSONEncoder)
        # torch.save(self.state_dict(), path)

    def load(self, path, device):
        """
        Load the parameters of your trained ActorModule from a JSON file.

        Adapt this method to your submission so that we can load your trained ActorModule.

        Args:
            path: pathlib.Path: full path of the JSON file
            device: str: device on which the ActorModule should live (e.g., "cpu")

        Returns:
            The loaded ActorModule instance
        """
        self.device = device
        with open(path, 'r') as json_file:
            state_dict = json.load(json_file, cls=TorchJSONDecoder)
        self.load_state_dict(state_dict)
        self.to_device(device)
        # self.load_state_dict(torch.load(path, map_location=self.device))
        return self

    def forward(self, obs, test=False, compute_logprob=True):
        """
        Computes the output action of our policy from the input observation.

        The whole point of deep RL is to train our policy network (actor) such that it outputs relevant actions.
        Training per-se will also rely on a critic network, but this is not part of the trained policy.
        Thus, our ActorModule will only implement the actor.

        Args:
            obs: the observation from the Gymnasium environment (when using TorchActorModule this is a torch.Tensor)
            test (bool): this is True for test episodes (deployment) and False for training episodes;
                in SAC, this enables us to sample randomly during training and deterministically at test-time.
            compute_logprob (bool): SAC will set this to True to retrieve log probabilities.

        Returns:
            the action sampled from our policy from observation obs
            the log probability of this action (this will be used for SAC)
        """
        # obs is our input observation.
        # We feed it to our actor neural network, which will output an action.

        # Let us feed it to our MLP:
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
            # (the next line is a correction formula for TanH squashing, present in the Spinup implementation of SAC)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None
        # And we squash our action within the action space:
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        # Finally, we remove the batch dimension:
        pi_action = pi_action.squeeze()
        return pi_action, logp_pi

    # Now, the only method that all participants are required to implement is act()
    # act() is the interface for TMRL to use your ActorModule as the policy it tests in TrackMania.
    # For the evaluation, the "test" argument will be set to True.
    def act(self, obs, test=False):
        """
        Computes an action from an observation.

        This method is the one all participants must implement.
        It is the policy that TMRL will use in TrackMania to evaluate your submission.

        Args:
            obs (object): the input observation (when using TorchActorModule, this is a torch.Tensor)
            test (bool): True at test-time (e.g., during evaluation...), False otherwise

        Returns:
            act (numpy.array): the computed action, in the form of a numpy array of 3 values between -1.0 and 1.0
        """
        # Since we have already implemented our policy in the form of a neural network,
        # act() is now pretty straightforward.
        # We don't need to compute the log probabilities here (they will be for our SAC training algorithm).
        # Also note that, when using TorchActorModule, TMRL calls act() in a torch.no_grad() context.
        # Thus, you don't need to use "with torch.no_grad()" here.
        # But let us do it anyway to be extra sure, for the people using ActorModule instead of TorchActorModule.
        with torch.no_grad():
            a, _ = self.forward(obs=obs, test=test, compute_logprob=False)
            return a.cpu().numpy()

