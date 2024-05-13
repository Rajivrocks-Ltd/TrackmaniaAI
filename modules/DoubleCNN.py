
import tmrl.config.config_constants as cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import floor

# Dimensions of the actual images in observations:
img_width = cfg.IMG_WIDTH
img_height = cfg.IMG_HEIGHT

# Number of consecutive screenshots in each observation:
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

class DQN(nn.Module):
    def __init__(self, num_actions):
        """
        DQN model.
        Args:
            img_height (int): Height of the input images.
            img_width (int): Width of the input images.
            imgs_buf_len (int): Number of image channels, generally the stack of frames.
            num_actions (int): Number of possible actions.
        """
        super(DQN, self).__init__()

        # Convolutional layers
        self.h_out, self.w_out = img_height, img_width
        self.conv1 = nn.Conv2d(imgs_buf_len, 64, 8, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv1, img_height, img_width)
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv2, self.h_out, self.w_out)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv3, self.h_out, self.w_out)
        self.conv4 = nn.Conv2d(128, 128, 4, stride=2)
        self.h_out, self.w_out = conv2d_out_dims(self.conv4, self.h_out, self.w_out)

        # Calculate flat features after convolution
        self.flat_features = self.conv4.out_channels * self.h_out * self.w_out

        # MLP for action values
        self.action_values = mlp([self.flat_features, 256, num_actions], nn.ReLU)

    def forward(self, x):
        """
        Forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor (batch of images).
        Returns:
            Q values for each action.
        """
        images = x[3] # according to the tutorial

        # Forward pass through convolutional layers
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, num_flat_features(x))  # Flatten the features for the MLP

        # Compute action values
        q_values = self.action_values(x)

        return q_values
