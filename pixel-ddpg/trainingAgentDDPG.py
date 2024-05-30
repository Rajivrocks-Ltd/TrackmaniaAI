from tmrl.training import TrainingAgent
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property
from copy import deepcopy
from torch.optim import Adam
import torch
from ActorCriticDDPG import VanillaCNNActorCriticDDPG


class DDPGTrainingAgent(TrainingAgent):
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
                 model_cls=VanillaCNNActorCriticDDPG,  # An actor-critic module, encapsulating our ActorModule
                 gamma=0.99,  # Discount factor
                 polyak=0.995,  # Exponential averaging factor for the target critic
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
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        # self.q_params = itertools.chain(self.model.q.parameters())
        self.actor_optimizer = Adam(self.model.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = Adam(self.model.critic.parameters(), lr=self.lr_critic)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):
        """
        A training sample is of the form (o, a, r, o2, d, t) where:
        -> o is the initial observation of the transition
        -> a is the selected action during the transition
        -> r is the reward of the transition
        -> o2 is the final observation of the transition
        -> d is the "terminated" signal indicating whether o2 is a terminal state
        -> t is the "truncated" signal indicating whether the episode has been truncated by a time-limit
        """
        # First, we decompose our batch into its relevant components, ignoring the "truncated" signal:
        o, a, r, o2, d, _ = batch

        with torch.no_grad():
            a2 = self.model_target.actor(o2)
            q_target = self.model_target.critic(o2, a2)
            backup = r + self.gamma * (1 - d) * q_target

        q = self.model.critic(o, a)

        # This gives us our critic loss, as the difference between the target and the estimate:
        critic_loss = ((q - backup) ** 2).mean()  # equivalent to the MSE loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        pi = self.model.actor(o)

        # Adding a part to detach the critic from the gradient computation graph
        for param in self.model.parameters():
            param.requires_grad = False

        actor_loss = -self.model.critic(o, pi).mean()

        # Now we can train our policy in the opposite direction of this loss' gradient:
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Adding the part to attach back the critic into the gradient computation graph
        for param in self.model.parameters():
            param.requires_grad = True

        # Finally, we update our target model with a slowly moving exponential average:
        with torch.no_grad():
            for param, param_targ in zip(self.model.parameters(), self.model_target.parameters()):
                param_targ.data.mul_(self.polyak)
                param_targ.data.add_((1 - self.polyak) * param.data)

        # TMRL enables us to log training metrics to wandb:
        ret_dict = dict(
            loss_actor=actor_loss.detach().item(),
            loss_critic=critic_loss.detach().item(),
        )
        return ret_dict
