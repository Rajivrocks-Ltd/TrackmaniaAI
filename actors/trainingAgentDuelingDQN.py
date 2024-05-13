import torch
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import numpy as np
from actors.DuelingDQNActor import MyDuelingDQNActorModule
from tmrl.util import cached_property
from tmrl.training import TrainingAgent
from tmrl.custom.utils.nn import copy_shared, no_grad
from copy import deepcopy

# A TrainingAgent must implement two methods:
# -> train(batch): optimizes the model from a batch of RL samples
# -> get_actor(): outputs a copy of the current ActorModule

class DuelingDQNTrainingAgent(TrainingAgent):

    # no-grad copy of the model used to send the Actor weights in get_actor():
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 batch_size,
                 memory_size,
                 update_buffer_interval,
                 observation_space=None,
                 action_space=None,
                 device=None,
                 model_cls=MyDuelingDQNActorModule,
                 gamma=0.99,
                 lr=1e-4):
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)

        model = model_cls(observation_space, action_space)
        self.model = model.to(self.device)
        self.target_model = no_grad(deepcopy(self.model))
        self.target_model.eval()
        self.lr = lr
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_buffer_interval = update_buffer_interval
        self.steps_done = 0

    def get_actor(self):
        return self.model

    def update(self, batch):

        o, a, r, o2, d, _ = batch

        pi_action, logp_pi = self.model(o)

        policy_loss = -(logp_pi * r).mean()

        # Gradient descent steps to update the model parameters
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return policy_loss

    def train(self, batch):

        # First, we decompose our batch into its relevant components, ignoring the "truncated" signal:
        o, a, r, o2, d, _ = batch

        loss = self.update(batch)

        # Optionally update the target network
        if self.steps_done % self.update_buffer_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self.steps_done += 1

        return {'loss': loss.item()}