from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import torch
from torch.optim import Adam

import tmrl.custom.custom_models as core
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property
from tmrl.training import TrainingAgent


@dataclass(eq=0)
class ActorCriticAgent(TrainingAgent):
    observation_space: type
    action_space: type
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model_cls: type = core.MLPActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    lr_entropy: float = 1e-3
    learn_entropy_coef: bool = True
    target_entropy: float = None
    optimizer_actor: Adam = Adam  # could be one of [Adam, AdamW, SGD] - however, would need to change typing
    optimizer_critic: Adam = Adam  # could be one of [Adam, AdamW, SGD] - however, would need to change typing

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        observation_space, action_space = self.observation_space, self.action_space
        model = self.model_cls(observation_space, action_space)
        self.model = model.to(self.device)
        self.model_target = no_grad(deepcopy(self.model))

        # Set up ADAM optimizers for policy and q-function:
        pi_optimizer_cls = self.optimizer_actor
        pi_optimizer_kwargs = {"lr": self.lr_actor}
        q_optimizer_cls = self.optimizer_critic
        q_optimizer_kwargs = {"lr": self.lr_critic}

        self.pi_optimizer = pi_optimizer_cls(self.model.actor.parameters(), **pi_optimizer_kwargs)
        self.q_optimizer = q_optimizer_cls(self.model.q1.parameters(), **q_optimizer_kwargs)

        if self.target_entropy is None:
            self.target_entropy = -np.prod(action_space.shape)
        else:
            self.target_entropy = float(self.target_entropy)

        if self.learn_entropy_coef:
            self.log_alpha = torch.log(torch.ones(1, device=self.device) * self.alpha).requires_grad_(True)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr_entropy)
        else:
            self.alpha_t = torch.tensor(float(self.alpha)).to(self.device)

    def get_actor(self):
        return self.model_nograd.actor

    def train(self, batch):

        o, a, r, o2, d, _ = batch

        pi, logp_pi = self.model.actor(o)

        # loss_alpha:
        loss_alpha = None
        if self.learn_entropy_coef:
            alpha_t = torch.exp(self.log_alpha.detach())
            loss_alpha = -(self.log_alpha * (logp_pi + self.target_entropy).detach()).mean()
        else:
            alpha_t = self.alpha_t

        # Optimize entropy coefficient, also called entropy temperature or alpha in the paper
        if loss_alpha is not None:
            self.alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self.alpha_optimizer.step()

        # Run one gradient descent step to calculate loss_q:
        q1 = self.model.q1(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.model.actor(o2)

            # Target Q-values
            q_pi_targ = self.model_target.q1(o2, a2)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - alpha_t * logp_a2)

        # MSE loss against Bellman backup
        loss_q = ((q1 - backup) ** 2).mean()

        self.q_optimizer.zero_grad()
        loss_q.backward()
        self.q_optimizer.step()

        # Freeze Q-networks, so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        self.model.q1.requires_grad_(False)

        # Next run one gradient descent step for actor. and calculate loss_pi:
        q_pi = self.model.q1(o, pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha_t * logp_pi - q_pi).mean()

        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks, so you can optimize it at next DDPG step.
        self.model.q1.requires_grad_(True)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.model.parameters(), self.model_target.parameters()):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        with torch.no_grad():
            ret_dict = dict(
                loss_actor=loss_pi.detach().item(),
                loss_critic=loss_q.detach().item(),
            )

        return ret_dict
