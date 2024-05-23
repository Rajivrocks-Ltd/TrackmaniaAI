# =====================================================================
# PPO TRAINING ALGORITHM
# =====================================================================

import itertools
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import torch
from torch.optim import Adam, AdamW, SGD
import logging

from ..models.lidar_ppo_ac_model import PPOActorCritic
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property
from tmrl.training import TrainingAgent
import tmrl.config.config_constants as cfg

# =====================================================================

@dataclass(eq=False)
class LidarPPOTrainingAgent(TrainingAgent):
    """
    PPO training agent for the LiDAR environment.
    """

    observation_space: type
    action_space: type
    device: str = None  # device where the model will live (None for auto)
    model_cls: type = PPOActorCritic
    gamma: float = 0.99
    polyak: float = 0.995
    alpha: float = 0.2  # fixed (v1) or initial (v2) value of the entropy coefficient
    lr_actor: float = 1e-3  # learning rate
    lr_critic: float = 1e-3  # learning rate
    optimizer_actor: str = "adam"  # one of ["adam", "adamw", "sgd"]
    optimizer_critic: str = "adam"  # one of ["adam", "adamw", "sgd"]
    betas_actor: tuple = None  # for Adam and AdamW
    betas_critic: tuple = None  # for Adam and AdamW
    l2_actor: float = None  # weight decay
    l2_critic: float = None  # weight decay

    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __post_init__(self):
        pass

    # TODO: Implement training agent.