from tmrl.training import TrainingAgent
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property
from copy import deepcopy
import itertools
from torch.optim import Adam
import torch
from actors.ActorCritic import VanillaCNNActorCritic
from modules.VanillaCNN_base import VanillaCNN

# A TrainingAgent must implement two methods:
# -> train(batch): optimizes the model from a batch of RL samples
# -> get_actor(): outputs a copy of the current ActorModule

class SACTrainingAgent(TrainingAgent):
    """
    Custom TrainingAgents implement two methods: train(batch) and get_actor().
    The train method performs a training step.
    The get_actor method retrieves your ActorModule to save it and send it to the RolloutWorkers.

    Your implementation must also pass three required arguments to the superclass:
    - observation_space (gymnasium.spaces.Space): observation space (here for your convenience)
    - action_space (gymnasium.spaces.Space): action space (here for your convenience)
    - device (str): device that should be used for training (e.g., `"cpu"` or `"cuda:0"`)
    """

    # no-grad copy of the model used to send the Actor weights in get_actor():
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 observation_space=None,  # Gymnasium observation space (required argument here for your convenience)
                 action_space=None,  # Gymnasium action space (required argument here for your convenience)
                 device=None,  # Device our TrainingAgent should use for training (required argument)
                 model_cls=VanillaCNNActorCritic,  # An actor-critic module, encapsulating our ActorModule
                 gamma=0.99,  # Discount factor
                 polyak=0.995,  # Exponential averaging factor for the target critic
                 alpha=0.2,  # Value of the entropy coefficient
                 lr_actor=1e-3,  # Learning rate for the actor
                 lr_critic=1e-3):  # Learning rate for the critic

        # required arguments passed to the superclass:
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)

        # custom stuff:
        model = model_cls(observation_space, action_space)
        self.model = model.to(self.device)
        self.model_target = no_grad(deepcopy(self.model))
        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.q_params = itertools.chain(self.model.q1.parameters(), self.model.q2.parameters())
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.q_optimizer = Adam(self.q_params, lr=self.lr_critic)
        self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        """
        Returns a copy of the current ActorModule.
        We return a copy without gradients, as this is for sending to the RolloutWorkers.
        Returns:
            actor: ActorModule: updated actor module to forward to the worker(s)
        """
        return self.model_nograd.actor

    def train(self, batch):
        """
        Executes a training iteration from batched training samples (batches of RL transitions).

        A training sample is of the form (o, a, r, o2, d, t) where:
        -> o is the initial observation of the transition
        -> a is the selected action during the transition
        -> r is the reward of the transition
        -> o2 is the final observation of the transition
        -> d is the "terminated" signal indicating whether o2 is a terminal state
        -> t is the "truncated" signal indicating whether the episode has been truncated by a time-limit

        Args:
            batch: (previous observation, action, reward, new observation, terminated signal, truncated signal)

        Returns:
            logs: Dictionary: a python dictionary of training metrics you wish to log on wandb
        """
        # First, we decompose our batch into its relevant components, ignoring the "truncated" signal:
        o, a, r, o2, d, _ = batch

        # We sample an action in the current policy and retrieve its corresponding log probability:
        pi, logp_pi = self.model.actor(obs=o, test=False, compute_logprob=True)

        # We also compute our action-value estimates for the current transition:
        q1 = self.model.q1(o, a)
        q2 = self.model.q2(o, a)

        # Now we compute our value target, for which we need to detach from gradients computation:
        with torch.no_grad():
            a2, logp_a2 = self.model.actor(o2)
            q1_pi_targ = self.model_target.q1(o2, a2)
            q2_pi_targ = self.model_target.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha_t * logp_a2)

        # This gives us our critic loss, as the difference between the target and the estimate:
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # We can now take an optimization step to train our critics in the opposite direction of this loss' gradient:
        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # For the policy optimization step, we detach our critics from the gradient computation graph:
        for p in self.q_params:
            p.requires_grad = False

        # We use the critics to estimate the value of the action we have sampled in the current policy:
        q1_pi = self.model.q1(o, pi)
        q2_pi = self.model.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Our policy loss is now the opposite of this value estimate, augmented with the entropy of the current policy:
        loss_pi = (self.alpha_t * logp_pi - q_pi).mean()

        # Now we can train our policy in the opposite direction of this loss' gradient:
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        # We attach the critics back into the gradient computation graph:
        for p in self.q_params:
            p.requires_grad = True

        # Finally, we update our target model with a slowly moving exponential average:
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # TMRL enables us to log training metrics to wandb:
        ret_dict = dict(
            loss_actor=loss_pi.detach().item(),
            loss_critic=loss_q.detach().item(),
        )
        return ret_dict
