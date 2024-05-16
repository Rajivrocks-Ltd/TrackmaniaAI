import torch
import torch.nn as nn
import torch.nn.functional as F
import tmrl.config.config_constants as cfg
from math import floor

img_width = cfg.IMG_WIDTH
img_height = cfg.IMG_HEIGHT
img_grayscale = cfg.GRAYSCALE
imgs_buf_len = cfg.IMG_HIST_LEN


def mlp(sizes, activation, output_activation=nn.Identity):

    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


# The next utility computes the dimensionality of CNN feature maps when flattened together:
def num_flat_features(x):
    size = x.size()[1:]  # dimension 0 is the batch dimension, so it is ignored
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


# The next utility computes the dimensionality of the output in a 2D CNN layer:
def conv2d_out_dims(conv_layer, h_in, w_in):
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) / conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) / conv_layer.stride[1] + 1)
    return h_out, w_out


class VanillaCNN2(nn.Module):
    def __init__(self, actor):
        """
        Simple CNN (Convolutional Neural Network) model for DDPG.
        Can be configured as either actor or critic network.

        Actor Configuration:
        actor=True: The network functions as an actor, which means it outputs actions directly.
        The dimensionality of the MLP output layer is set to output three values, corresponding
        to the actions in the environment (such as steering, acceleration, and braking).

        Critic Configuration:
        actor=False: The network functions as a critic, which means it outputs a single scalar
        value representing the Q-value of the state-action pair. This Q-value is used to evaluate
        how good it is to take a particular action in a given state.

        Args:
            actor (bool): indicates whether this neural net is an actor network
        """
        super(VanillaCNN2, self).__init__()

        self.actor = actor

        # Convolutional layers processing screenshots:
        self.h_out, self.w_out = img_height, img_width
        self.conv1 = nn.Conv2d(imgs_buf_len, 64, 8, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, self.h_out, self.w_out)
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)
        self.conv4 = nn.Conv2d(128, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)
        self.out_channels = self.conv4.out_channels

        # Dimensionality of the CNN output:
        self.flat_features = self.out_channels * self.h_out * self.w_out

        # Dimensionality of the MLP input:
        float_features = 9 if self.actor else 12
        self.mlp_input_features = self.flat_features + float_features

        # mlp_output_size = 3 if self.actor else 1  # Output 3 actions for actor, 1 Q-value for critic
        # self.mlp_layers = [256, 256, mlp_output_size]
        self.mlp_layers = [256, 256] if self.actor else [256, 256, 1]
        self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.ReLU)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor (i.e., the observation fed to our deep neural network)

        Returns:
            the output of our neural network in the form of a torch.Tensor
        """
        if self.actor:
            speed, gear, rpm, images, act1, act2 = x
        else:
            speed, gear, rpm, images, act1, act2, act = x

        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        flat_features = num_flat_features(x)
        assert flat_features == self.flat_features, f"x.shape:{x.shape},\
                                                    flat_features:{flat_features},\
                                                    self.out_channels:{self.out_channels},\
                                                    self.h_out:{self.h_out},\
                                                    self.w_out:{self.w_out}"
        x = x.view(-1, flat_features)
        if self.actor:
            x = torch.cat((speed, gear, rpm, x, act1, act2), -1)
        else:
            x = torch.cat((speed, gear, rpm, x, act1, act2, act), -1)

        x = self.mlp(x)

        return x


class VanillaCNNQFunctionDDPG(nn.Module):
    """
    Critic module for DDPG, leveraging the VanillaCNN2 architecture.
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        # Assuming that we can set VanillaCNN2 for critic operation by specifying actor=False
        self.net = VanillaCNN2(actor=False)

    def forward(self, obs, act):
        """
        Forward pass to estimate the Q-value of the state-action pair.

        Args:
            obs (torch.Tensor): The observation/state tensor.
            act (torch.Tensor): The action tensor.

        Returns:
            q_value (torch.Tensor): The estimated Q-value.
        """
        # Concatenate the output of the network with the action
        x = (*obs, act)
        q = self.net(x)
        return torch.squeeze(q, -1)
