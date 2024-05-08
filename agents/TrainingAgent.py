from tmrl.training import TrainingAgent
from torch.optim import Adam
import torch.nn.functional as F

from agents.ActorModule import SimpleActorModule


class SimpleTrainingAgent(TrainingAgent):
    def __init__(self, observation_space, action_space, device='cpu'):
        super().__init__(observation_space, action_space, device)
        self.model = SimpleActorModule(observation_space, action_space).to(device)
        self.optimizer = Adam(self.model.parameters(), lr=0.001)

    def get_actor(self):
        return self.model

    def train(self, batch):
        obs, actions, rewards, next_obs, done, _ = batch
        actions_pred = self.model(obs)
        loss = F.mse_loss(actions_pred, actions)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}
