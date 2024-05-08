import torch

from tmrl.actor import TorchActorModule

from models.MLP import SimpleMLP


class SimpleActorModule(TorchActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)
        self.net = SimpleMLP(observation_space.shape[0], action_space.shape[0])

    def act(self, obs, test=False):
        with torch.no_grad():
            return self.net(obs).cpu().numpy()
