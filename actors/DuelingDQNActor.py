from tmrl.actor import TorchActorModule
import torch
from auxiliary.create_jsons import TorchJSONDecoder, TorchJSONEncoder
from modules.DuelingCNN import DuelingCNN
import json

class MyDuelingDQNActorModule(TorchActorModule):
    def __init__(self, observation_space, action_space):
        super().__init__(observation_space, action_space)

        # And initialize our attributes:
        dim_act = action_space.shape[0]  # dimensionality of actions

        # Initialize the Dueling DQN network
        self.net = DuelingCNN(dim_act)

    def save(self, path):
        with open(path, 'w') as json_file:
            json.dump(self.state_dict(), json_file, cls=TorchJSONEncoder)

    def load(self, path, device):
        self.device = device
        with open(path, 'r') as json_file:
            state_dict = json.load(json_file, cls=TorchJSONDecoder)
        self.load_state_dict(state_dict)
        self.to_device(device)
        return self

    def act(self, obs, test=False):

        with torch.no_grad():
            q_values = self.net(obs)
            action_index = q_values.max(1)[1].item()  # Get the action with the highest Q-value
            return action_index

    def forward(self, obs):
        return self.net(obs)



