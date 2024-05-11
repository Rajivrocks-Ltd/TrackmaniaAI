from tmrl.actor import TorchActorModule
from auxiliary.create_jsons import TorchJSONDecoder, TorchJSONEncoder
from modules.DuelingCNN import DuelingCNN
import json
import numpy as np
import torch

class MyDuelingDQNActorModule(TorchActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        # And initialize our attributes:
        dim_act = action_space.shape[0]  # dimensionality of actions

        # Initialize the Dueling DQN network
        self.net = DuelingCNN(dim_act)

        # Initialize parameters for action selection
        self.eps_start = 0.9
        self.eps_end = 1e-3
        self.eps_decay = 200
        self.step = 0
        self.action_correspondance = {
            i + 2 * j + 4 * k + 8 * l: [i, j, k, l]
            for i in range(2)
            for j in range(2)
            for k in range(2)
            for l in range(2)
        }

    def save(self, path):
        # with open(path, 'w') as json_file:
        #     json.dump(self.state_dict(), json_file, cls=TorchJSONEncoder)
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.device = device
        # with open(path, 'r') as json_file:
        #     state_dict = json.load(json_file, cls=TorchJSONDecoder)
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to_device(device)
        return self

    def epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -1.0 * self.step / self.eps_decay)

    def act(self, obs,  test=False):
        if np.random.rand() < self.epsilon():
            self.step += 1
            return self.action_correspondance[
                np.argmax(self.net(obs).detach().cpu().numpy())
            ]
        self.step += 1
        return self.action_correspondance[
            np.random.randint(0, len(self.action_correspondance))
        ]

    def forward(self, obs):
        return self.net(obs)
