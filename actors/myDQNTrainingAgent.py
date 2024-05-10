import torch
import torch.optim as optim
import torch.nn as nn
from tmrl.training import TrainingAgent
from modules.DQN import EpsilonGreedyDQN
import torch.nn.functional as F
import numpy as np

class DQN_trainer(TrainingAgent):
    def __init__(self, batch_size=32, N_epochs=100):
        self.N_epochs = N_epochs
        self.GAMMA = 0.999
        self.target_update = 2
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.buffer = Buffer(capacity=10000)
        self.agent = EpsilonGreedyDQN(
            input_size=self.env.observation_space.shape[0], device=self.device
        )
        self.optimizer = optim.Adam(self.agent.policy.parameters(), lr=0.001)
        self.loss = nn.SmoothL1Loss()


    def train(self, batch):
        # Iterations over episodes and time-steps is already implemented
        # in this function put what is after while done
        # Decompose the batch into its components
        states, actions, rewards, next_states, dones = batch

        # Convert batch components to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q-values for current and next states
        q_values = self.model(states)
        q_values_next = self.model_target(next_states)

        # Gather Q-values corresponding to the actions taken
        q_values_predicted = q_values.gather(1, actions.unsqueeze(1))

        # Compute target Q-values using the Bellman equation
        target_q_values = rewards + self.gamma * (1 - dones) * torch.max(q_values_next, dim=1)[0].unsqueeze(1)

        # Compute the Huber loss
        loss = F.smooth_l1_loss(q_values_predicted, target_q_values.detach())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network periodically (every `target_update_freq` steps)

        return loss.item()
