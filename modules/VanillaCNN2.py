import torch
import torch.nn as nn
import tmrl.config.config_constants as cfg
# Plus a couple useful imports:
import torch.nn.functional as F
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



class VanillaCNNQFunction2(nn.Module):
    """
    Critic module for SAC.
    """
    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = VanillaCNN2(q_net=True)  # q_net is True for a critic module

    def forward(self, obs, act):
        """
        Estimates the action-value of the (obs, act) state-action pair.

        """
        # Since q_net is True, we append our action act to our observation obs.
        # Note that obs is a tuple of batched tensors: respectively the history of 4 images, speed, etc.
        x = (*obs, act)
        q = self.net(x)
        return torch.squeeze(q, -1)

class VanillaCNN2(nn.Module):
    def __init__(self, q_net):
        """
        Simple CNN (Convolutional Neural Network) model for SAC (Soft Actor-Critic).

        Args:
            q_net (bool): indicates whether this neural net is a critic network
        """
        super(VanillaCNN2, self).__init__()

        self.q_net = q_net

        # Convolutional layers processing screenshots:
        # The default config.json gives 4 grayscale images of 64 x 64 pixels
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
        # The MLP input will be formed of:
        # - the flattened CNN output
        # - the current speed, gear and RPM measurements (3 floats)
        # - the 2 previous actions (2 x 3 floats), important because of the real-time nature of our controller
        # - when the module is the critic, the selected action (3 floats)
        float_features = 12 if self.q_net else 9
        self.mlp_input_features = self.flat_features + float_features

        # MLP layers:
        # (when using the model as a policy, we will sample from a multivariate gaussian defined later in the tutorial;
        # thus, the output dimensionality is  1 for the critic, and we will define the output layer of policies later)
        self.mlp_layers = [256, 256, 1] if self.q_net else [256, 256]
        self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.ReLU)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): input tensor (i.e., the observation fed to our deep neural network)

        Returns:
            the output of our neural network in the form of a torch.Tensor
        """
        if self.q_net:
            # The critic takes the next action (act) as additional input
            # act1 and act2 are the actions in the action buffer (real-time RL):
            speed, gear, rpm, images, act1, act2, act = x
        else:
            # For the policy, the next action (act) is what we are computing, so we don't have it:
            speed, gear, rpm, images, act1, act2 = x

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
        # All good, let us flatten our output feature map:
        x = x.view(-1, flat_features)

        # Finally, we can feed the result along our float values to the MLP:
        if self.q_net:
            x = torch.cat((speed, gear, rpm, x, act1, act2, act), -1)
        else:
            x = torch.cat((speed, gear, rpm, x, act1, act2), -1)
        x = self.mlp(x)

        # And this gives us the output of our deep neural network
        return x


