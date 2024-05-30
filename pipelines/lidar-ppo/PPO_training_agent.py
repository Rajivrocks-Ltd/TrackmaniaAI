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
class PPOTrainingAgent(TrainingAgent):
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
    lambda_param: float = 0.95
    clip_ratio = 0.2
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

        # Compute the old log probabilities of the actions taken using the old policy
        with torch.no_grad():
            old_logp_a = self.model.actor.get_log_prob(o, a)

        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        with torch.no_grad():
            v = self.model.q1(o, a).squeeze()
            v2 = self.model.q1(o2, a).squeeze()
            delta = r + self.gamma * (1 - d) * v2 - v
            adv = delta.clone()
            for t in reversed(range(len(delta) - 1)):
                adv[t] = delta[t] + self.gamma * self.lambda_param * (1 - d[t]) * adv[t + 1]
            ret = adv + v

        # Get the current policy's action distribution and log probabilities
        logp_a = self.model.actor.get_log_prob(o, a)
        ratio = torch.exp(logp_a - old_logp_a)  # Compute the ratio (pi(a|s) / pi_old(a|s))

        # Compute the surrogate loss
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv
        loss_pi = -torch.min(surr1, surr2).mean()  # PPO's clipped objective

        # Optimize policy (actor)
        self.pi_optimizer.zero_grad()
        loss_pi.backward()
        self.pi_optimizer.step()

        # Value function loss (critic)
        v = self.model.q1(o, a).squeeze()  # Get value estimates
        loss_v = ((v - ret) ** 2).mean()  # MSE loss against returns

        # Optimize value function (critic)
        self.q_optimizer.zero_grad()
        loss_v.backward()
        self.q_optimizer.step()

        with torch.no_grad():
            ret_dict = dict(
                loss_actor=loss_pi.detach().item(),
                loss_critic=loss_v.detach().item(),
            )

        return ret_dict
