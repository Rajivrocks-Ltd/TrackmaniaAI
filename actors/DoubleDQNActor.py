from tmrl.actor import TorchActorModule
from auxiliary.create_jsons import TorchJSONDecoder, TorchJSONEncoder
from modules.DoubleCNN import DQN
import numpy as np


class MyDoubleDQNActorModule(TorchActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        # And initialize our attributes:
        dim_act = action_space.shape[0]  # dimensionality of actions

        # Initialize the Dueling DQN networks
        self.online_net = DQN(dim_act)
        self.target_net = DQN(dim_act)
        self.target_net.load_state_dict(self.online_net.state_dict())  # Initialize target network with online network's parameters

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
        with open(path, 'w') as json_file:
            json.dump(self.online_net.state_dict(), json_file, cls=TorchJSONEncoder)

    def load(self, path, device):
        self.device = device
        with open(path, 'r') as json_file:
            state_dict = json.load(json_file, cls=TorchJSONDecoder)
        self.online_net.load_state_dict(state_dict)
        self.target_net.load_state_dict(state_dict)
        self.to_device(device)
        return self

    def epsilon(self):
        return self.eps_end + (self.eps_start - self.eps_end) * np.exp(
            -1.0 * self.step / self.eps_decay)

    def act(self, obs,  test=False):
        if np.random.rand() < self.epsilon():
            self.step += 1
            return self.action_correspondance[
                np.argmax(self.online_net(obs).detach().cpu().numpy())
            ]
        self.step += 1
        return self.action_correspondance[
            np.random.randint(0, len(self.action_correspondance))
        ]

    def forward(self, obs):
        return self.online_net(obs)
