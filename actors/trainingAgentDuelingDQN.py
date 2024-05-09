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
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.update_buffer_interval = update_buffer_interval
        self.steps_done = 0

    def get_actor(self):
        return self.model_nograd.actor

    def update(self):

        batch = random.sample(self.memory, self.batch_size)
        o, a, r, o2, d, _ = batch

        'Create from numpy arrays, torch tensor and make them 2D to work with batches'
        # Convert to tensors
        states = torch.FloatTensor(np.float32(o)).to(self.device)
        actions = torch.LongTensor(a).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.float32(o2)).to(self.device)
        dones = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        # Compute current Q values: Q(s, a)
        current_q_values = self.model(states).gather(1, actions)

        # Compute expected Q values from target actions: max_a' Q'(s', a')
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), expected_q_values)

        # Gradient descent update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def update_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self, batch):

        # First, we decompose our batch into its relevant components, ignoring the "truncated" signal:
        o, a, r, o2, d, _ = batch
        "TODO: how are we implementing the action selection"

        self.update_memory(o, a, r, o2, d)
        loss = self.update()


        # Optionally update the target network
        if self.steps_done % self.update_buffer_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        self.steps_done += 1

        return {'loss': loss.item()}