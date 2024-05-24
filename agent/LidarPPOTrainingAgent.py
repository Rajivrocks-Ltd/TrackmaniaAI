import itertools
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import torch
from torch.optim import Adam, AdamW, SGD
import logging

from model.PPOActorCriticModel import PPOActorCritic
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property
from tmrl.training import TrainingAgent
import tmrl.config.config_constants as cfg
import time

@dataclass(eq=False)
class LidarPPOTrainingAgent(TrainingAgent):
    observation_space: type
    action_space: type
    device: str = None
    model_cls: type = PPOActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2
    lr_actor: float = 1e-3
    lr_critic: float = 1e-3
    optimizer_actor: str = "adam"
    optimizer_critic: str = "adam"
    betas_actor: tuple = (0.9, 0.999)
    betas_critic: tuple = (0.9, 0.999)
    l2_actor: float = 0.0
    l2_critic: float = 0.0

    def __post_init__(self):
        self.device = self.device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model_cls(self.observation_space, self.action_space).to(self.device)

        self.optimizer_actor = self._get_optimizer(self.optimizer_actor, self.model.actor.parameters(), self.lr_actor, self.betas_actor, self.l2_actor)
        self.optimizer_critic = self._get_optimizer(self.optimizer_critic, self.model.critic.parameters(), self.lr_critic, self.betas_critic, self.l2_critic)

    def _get_optimizer(self, name, params, lr, betas, weight_decay):
        if name == "adam":
            return Adam(params, lr=lr, betas=betas, weight_decay=weight_decay)
        elif name == "adamw":
            return AdamW(params, lr=lr, betas=betas, weight_decay=weight_decay)
        elif name == "sgd":
            return SGD(params, lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {name}")
    
    def get_actor(self):
        return self.model.actor

    def train(self, batch):
        states, actions, rewards, next_states, dones = batch

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Update Critic
        with torch.no_grad():
            target_v = rewards + self.gamma * (1 - dones) * self.model.critic(next_states)
        v = self.model.critic(states)
        critic_loss = ((v - target_v) ** 2).mean()

        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()

        # Update Actor
        log_probs_old = self.model.actor.get_log_prob(states, actions)
        adv = target_v - v.detach()
        log_probs = self.model.actor.get_log_prob(states, actions)
        ratio = torch.exp(log_probs - log_probs_old)
        actor_loss = -(torch.min(ratio * adv, torch.clamp(ratio, 1.0 - cfg.CLIP_RANGE, 1.0 + cfg.CLIP_RANGE) * adv)).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        return actor_loss.item(), critic_loss.item()

    # def compute_returns_and_advantages(self, batch):
    #     o, a, r, o2, d, _ = batch

    #     with torch.no_grad():
    #         values = self.model.q1(o, a)
    #         next_values = self.model.q1(o2, a)

    #     advantages = torch.zeros_like(r, device=self.device)
    #     returns = torch.zeros_like(r, device=self.device)
    #     gae = 0
    #     for step in reversed(range(len(r))):
    #         delta = r[step] + self.gamma * next_values[step] * (1 - d[step]) - values[step]
    #         gae = delta + self.gamma * self.lam * (1 - d[step]) * gae
    #         advantages[step] = gae
    #         returns[step] = gae + values[step]

    #     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    #     return returns, advantages

    # def train(self, batch):
    #     start_time = time.time()
    #     o, a, r, o2, d, _ = batch

    #     # Convert batch elements to tensors and move to the specified device
    #     conversion_start = time.time()
    #     o = tuple(
    #         item.to(self.device) if isinstance(item, torch.Tensor) else torch.tensor(item, device=self.device) for item
    #         in o)
    #     a = a.to(self.device) if isinstance(a, torch.Tensor) else torch.tensor(a, device=self.device)
    #     r = r.to(self.device) if isinstance(r, torch.Tensor) else torch.tensor(r, device=self.device)
    #     o2 = tuple(
    #         item.to(self.device) if isinstance(item, torch.Tensor) else torch.tensor(item, device=self.device) for item
    #         in o2)
    #     d = d.to(self.device) if isinstance(d, torch.Tensor) else torch.tensor(d, device=self.device)
    #     conversion_time = time.time() - conversion_start

    #     # Compute returns and advantages
    #     compute_start = time.time()
    #     ret, adv = self.compute_returns_and_advantages((o, a, r, o2, d, _))
    #     compute_time = time.time() - compute_start

    #     # Compute old log probabilities
    #     with torch.no_grad():
    #         _, logp_old = self.model.actor(o, compute_logprob=True)

    #     # Policy loss
    #     policy_loss_start = time.time()
    #     for _ in range(self.train_pi_iters):
    #         pi, logp = self.model.actor(o, compute_logprob=True)
    #         ratio = torch.exp(logp - logp_old)
    #         clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
    #         loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

    #         self.pi_optimizer.zero_grad()
    #         loss_pi.backward()
    #         self.pi_optimizer.step()
    #     policy_loss_time = time.time() - policy_loss_start

    #     # Value function loss
    #     value_loss_start = time.time()
    #     for _ in range(self.train_v_iters):
    #         q1 = self.model.q1(o, a)
    #         q2 = self.model.q2(o, a)
    #         loss_v = ((q1 - ret) ** 2).mean() + ((q2 - ret) ** 2).mean()
    #         self.vf_optimizer.zero_grad()
    #         loss_v.backward()
    #         self.vf_optimizer.step()
    #     value_loss_time = time.time() - value_loss_start

    #     ret_dict = dict(
    #         loss_actor=loss_pi.item(),
    #         loss_critc=loss_v.item(),
    #         kl_divergence=(logp_old - logp).mean().item()
    #     )

    #     end_time = time.time() - start_time

    #     if self.debug:
    #         print(f"Total time: {end_time:.2f}s, "
    #               f"Conversion time: {conversion_time:.2f}s, "
    #               f"Compute time: {compute_time:.2f}s, "
    #               f"Policy Loss time: {policy_loss_time:.2f}s, "
    #               f"Value Loss time: {value_loss_time:.2f}s")

    #     return ret_dict
