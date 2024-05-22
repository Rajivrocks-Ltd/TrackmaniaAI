# The constants that are defined in config.json:
import tmrl.config.config_constants as cfg
# Useful classes:
import tmrl.config.config_objects as cfg_obj
# The utility that TMRL uses to partially instantiate classes:
from tmrl.util import partial
# The TMRL three main entities (i.e., the Trainer, the RolloutWorker and the central Server):
from tmrl.networking import Trainer, RolloutWorker, Server

# The training class that we will customize with our own training algorithm in this tutorial:
from tmrl.training_offline import TrainingOffline

# And a couple external libraries:
import numpy as np
import os
import time

# =====================================================================
# USEFUL PARAMETERS
# =====================================================================
epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]

# Number of rounds per 'epoch':
# (training metrics are displayed in the terminal at the end of each round)
rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]

# Number of training steps per round:
# (a training step is a call to the train() function that we will define later in this tutorial)
steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]

# Minimum number of environment steps collected before training starts:
# (this is useful when you want to fill your replay buffer with samples from a baseline policy)
start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]

# Maximum training steps / environment steps ratio:
# (if training becomes faster than this ratio, it will be paused, waiting for new samples from the environment)
max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]

# Number of training steps performed between broadcasts of policy updates:
update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]

# Number of training steps performed between retrievals of received samples to put them in the replay buffer:
update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]

# Training device (e.g., "cuda:0"):
device_trainer = 'cuda' if cfg.CUDA_TRAINING else 'cpu'

# Maximum size of the replay buffer:
memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]

# Batch size for training:
batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]

wandb_run_id = cfg.WANDB_RUN_ID  # change this by a name of your choice for your run
wandb_project = cfg.TMRL_CONFIG["WANDB_PROJECT"]  # name of the wandb project in which your run will appear
wandb_entity = cfg.TMRL_CONFIG["WANDB_ENTITY"]  # wandb account
wandb_key = cfg.TMRL_CONFIG["WANDB_KEY"]  # wandb API key

os.environ['WANDB_API_KEY'] = wandb_key  # this line sets your wandb API key as the active key

# Number of time-steps after which episodes collected by the worker are truncated:
max_samples_per_episode = cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]

# Networking parameters:
# (In TMRL, networking is managed by tlspyo. The following are tlspyo parameters.)
server_ip_for_trainer = cfg.SERVER_IP_FOR_TRAINER  # IP of the machine running the Server (trainer point of view)
server_ip_for_worker = cfg.SERVER_IP_FOR_WORKER  # IP of the machine running the Server (worker point of view)
server_port = cfg.PORT  # port used to communicate with this machine
password = cfg.PASSWORD  # password that secures your communication
security = cfg.SECURITY  # when training over the Internet, it is safer to change this to "TLS"
# (please read the security instructions on GitHub)


# =====================================================================
# ADVANCED PARAMETERS
# =====================================================================

# Base class of the replay memory used by the trainer:
memory_base_cls = cfg_obj.MEM

# Sample compression scheme applied by the worker for this replay memory:
sample_compressor = cfg_obj.SAMPLE_COMPRESSOR

# Sample preprocessor for data augmentation:
sample_preprocessor = None

# Path from where an offline dataset can be loaded to initialize the replay memory:
dataset_path = cfg.DATASET_PATH

# Preprocessor applied by the worker to the observations it collects:
# (Note: if your script defines the name "obs_preprocessor", we will use your preprocessor instead of the default)
obs_preprocessor = cfg_obj.OBS_PREPROCESSOR

# =====================================================================
# COMPETITION FIXED PARAMETERS
# =====================================================================

# rtgym environment class (full TrackMania Gymnasium environment):
env_cls = cfg_obj.ENV_CLS

# Device used for inference on workers (change if you like but keep in mind that the competition evaluation is on CPU)
device_worker = 'cpu'

# =====================================================================
# ENVIRONMENT PARAMETERS
# =====================================================================

# Dimensions of the TrackMania window:
window_width = cfg.WINDOW_WIDTH  # must be between 256 and 958
window_height = cfg.WINDOW_HEIGHT  # must be between 128 and 488

# Dimensions of the actual images in observations:
img_width = cfg.IMG_WIDTH
img_height = cfg.IMG_HEIGHT

# Whether you are using grayscale (default) or color images:
# (Note: The tutorial will stop working if you use colors)
img_grayscale = cfg.GRAYSCALE

# Number of consecutive screenshots in each observation:
imgs_buf_len = cfg.IMG_HIST_LEN

# Number of actions in the action buffer (this is part of observations):
# (Note: The tutorial will stop working if you change this)
act_buf_len = cfg.ACT_BUF_LEN

# =====================================================================
# MEMORY CLASS
# =====================================================================

memory_cls = partial(memory_base_cls,
                     memory_size=memory_size,
                     batch_size=batch_size,
                     sample_preprocessor=sample_preprocessor,
                     dataset_path=cfg.DATASET_PATH,
                     imgs_obs=imgs_buf_len,
                     act_buf_len=act_buf_len,
                     crc_debug=False)

# =====================================================================
# CUSTOM MODEL
# =====================================================================
LOG_STD_MAX = 2
LOG_STD_MIN = -20

# Let us import the ActorModule that we are supposed to implement.
# We will use PyTorch in this tutorial.
# TMRL readily provides a PyTorch-specific subclass of ActorModule:
from tmrl.actor import TorchActorModule

# Plus a couple useful imports:
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
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



# Let us now define a module that will be the main building block of both our actor and critic:
class VanillaCNN(nn.Module):
    def __init__(self, q_net):
        super(VanillaCNN, self).__init__()

        self.q_net = q_net

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

        float_features = 12 if self.q_net else 9
        self.mlp_input_features = self.flat_features + float_features

        self.mlp_layers = [self.flat_features, 256, 1] if self.q_net else [self.flat_features, 256]
        self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.ReLU)

    def forward(self, x):
        
        images = x[3] # according to the tutorial

        print(images)
        x = F.relu(self.conv1(images))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        # Now we will flatten our output feature map.
        # Let us double-check that our dimensions are what we expect them to be:
        flat_features = num_flat_features(x)
        assert flat_features == self.flat_features, f"x.shape:{x.shape},\
                                                    flat_features:{flat_features},\
                                                    self.out_channels:{self.out_channels},\
                                                    self.h_out:{self.h_out},\
                                                    self.w_out:{self.w_out}"
        # All good, let us flatten our output feature map:
        x = x.view(-1, flat_features)

        #x = torch.cat(x, -1)
        x = self.mlp(x)

        # And this gives us the output of our deep neural network :)
        return x

import json


class TorchJSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for torch tensors, used in the custom save() method of our ActorModule.
    """

    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return json.JSONEncoder.default(self, obj)


class TorchJSONDecoder(json.JSONDecoder):
    """
    Custom JSON decoder for torch tensors, used in the custom load() method of our ActorModule.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    def object_hook(self, dct):
        for key in dct.keys():
            if isinstance(dct[key], list):
                dct[key] = torch.Tensor(dct[key])
        return dct

from tmrl.util import prod

class MyActorModule(TorchActorModule):

    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256), activation=nn.ReLU):
        # We must call the superclass __init__:
        super().__init__(observation_space, action_space)

        try:
            dim_obs = sum(prod(s for s in space.shape) for space in observation_space)
            self.tuple_obs = True
        except TypeError:
            dim_obs = prod(observation_space.shape)
            self.tuple_obs = False

        dim_act = action_space.shape[0]
        act_limit = action_space.high[0]
        self.net = mlp([dim_obs] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], dim_act)
        self.act_limit = act_limit


    def save(self, path):
        # with open(path, 'w') as json_file:
        #     json.dump(self.state_dict(), json_file, cls=TorchJSONEncoder)
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.device = device
        # with open(path, 'r') as json_file:
        #     state_dict = json.load(json_file, cls=TorchJSONDecoder)
        # self.load_state_dict(state_dict)
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to_device(device)
        return self

    def forward(self, obs, test=False, with_logprob=True):
        x = torch.cat(obs, -1) if self.tuple_obs else torch.flatten(obs, start_dim=1)
        net_out = self.net(x)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if test:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        # pi_action = pi_action.squeeze()

        return pi_action, logp_pi

    def act(self, obs, test=False):
        with torch.no_grad():
            a, _ = self.forward(obs, test, False)
            res = a.squeeze().cpu().numpy()
            if not len(res.shape):
                res = np.expand_dims(res, 0)
            return res


# The critic module for SAC is now super straightforward:
class VanillaCNNQFunction(nn.Module):
    """
    Critic module for SAC.
    """

    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = VanillaCNN(q_net=True)  # q_net is True for a critic module

    def forward(self, obs, act):
        # Since q_net is True, we append our action act to our observation obs.
        # Note that obs is a tuple of batched tensors: respectively the history of 4 images, speed, etc.
        x = (*obs, act)
        q = self.net(x)
        return torch.squeeze(q, -1)


# Finally, let us merge this together into an actor-critic torch.nn.module for training.
# Classically, we use one actor and two parallel critics to alleviate the overestimation bias.
class VanillaCNNActorCritic(nn.Module):
    """
    Actor-critic module for the SAC algorithm.
    """

    def __init__(self, observation_space, action_space):
        super().__init__()

        # Policy network (actor):
        self.actor = MyActorModule(observation_space, action_space)
        # Value networks (critics):
        self.q1 = VanillaCNNQFunction(observation_space, action_space)
        self.q2 = VanillaCNNQFunction(observation_space, action_space)


# =====================================================================
# CUSTOM TRAINING ALGORITHM
# =====================================================================

from tmrl.training import TrainingAgent

# We will also use a couple utilities, and the Adam optimizer:

from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property
from copy import deepcopy
import itertools
from torch.optim import Adam

class PPOTrainingAgent(TrainingAgent):
    model_nograd = cached_property(lambda self: no_grad(copy_shared(self.model)))

    def __init__(self,
                 observation_space=None,
                 action_space=None,
                 device=None,
                 model_cls=VanillaCNNActorCritic,
                 gamma=0.99,
                 lam=0.95,
                 clip_ratio=0.2,
                 pi_lr=3e-4,
                 vf_lr=1e-3,
                 train_pi_iters=80,
                 train_v_iters=80,
                 target_kl=0.01,
                 debug=False):
        super().__init__(observation_space=observation_space,
                         action_space=action_space,
                         device=device)
        self.model = model_cls(observation_space, action_space).to(self.device)
        self.model_target = deepcopy(self.model)
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.pi_lr, betas=(0.997, 0.997))
        self.vf_optimizer = Adam(itertools.chain(self.model.q1.parameters(), self.model.q2.parameters()), lr=self.vf_lr, betas=(0.997, 0.997))
        self.debug = debug

    def get_actor(self):
        return self.model_nograd.actor

    def compute_returns_and_advantages(self, batch):
        o, a, r, o2, d, _ = batch

        with torch.no_grad():
            values = self.model.q1(o, a)
            next_values = self.model.q1(o2, a)

        advantages = torch.zeros_like(r, device=self.device)
        returns = torch.zeros_like(r, device=self.device)
        gae = 0
        for step in reversed(range(len(r))):
            delta = r[step] + self.gamma * next_values[step] * (1 - d[step]) - values[step]
            gae = delta + self.gamma * self.lam * (1 - d[step]) * gae
            advantages[step] = gae
            returns[step] = gae + values[step]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return returns, advantages

    def train(self, batch):
        start_time = time.time()
        o, a, r, o2, d, _ = batch

        # Convert batch elements to tensors and move to the specified device
        conversion_start = time.time()
        o = tuple(
            item.to(self.device) if isinstance(item, torch.Tensor) else torch.tensor(item, device=self.device) for item
            in o)
        a = a.to(self.device) if isinstance(a, torch.Tensor) else torch.tensor(a, device=self.device)
        r = r.to(self.device) if isinstance(r, torch.Tensor) else torch.tensor(r, device=self.device)
        o2 = tuple(
            item.to(self.device) if isinstance(item, torch.Tensor) else torch.tensor(item, device=self.device) for item
            in o2)
        d = d.to(self.device) if isinstance(d, torch.Tensor) else torch.tensor(d, device=self.device)
        conversion_time = time.time() - conversion_start

        # Compute returns and advantages
        compute_start = time.time()
        ret, adv = self.compute_returns_and_advantages((o, a, r, o2, d, _))
        compute_time = time.time() - compute_start

        # Compute old log probabilities
        with torch.no_grad():
            _, logp_old = self.model.actor(o, compute_logprob=True)

        # Policy loss
        policy_loss_start = time.time()
        for _ in range(self.train_pi_iters):
            pi, logp = self.model.actor(o, compute_logprob=True)
            ratio = torch.exp(logp - logp_old)
            clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
            loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

            self.pi_optimizer.zero_grad()
            loss_pi.backward()
            self.pi_optimizer.step()
        policy_loss_time = time.time() - policy_loss_start

        # Value function loss
        value_loss_start = time.time()
        for _ in range(self.train_v_iters):
            q1 = self.model.q1(o, a)
            q2 = self.model.q2(o, a)
            loss_v = ((q1 - ret) ** 2).mean() + ((q2 - ret) ** 2).mean()
            self.vf_optimizer.zero_grad()
            loss_v.backward()
            self.vf_optimizer.step()
        value_loss_time = time.time() - value_loss_start

        ret_dict = dict(
            loss_actor=loss_pi.item(),
            loss_critc=loss_v.item(),
            kl_divergence=(logp_old - logp).mean().item()
        )

        end_time = time.time() - start_time

        if self.debug:
            print(f"Total time: {end_time:.2f}s, "
                  f"Conversion time: {conversion_time:.2f}s, "
                  f"Compute time: {compute_time:.2f}s, "
                  f"Policy Loss time: {policy_loss_time:.2f}s, "
                  f"Value Loss time: {value_loss_time:.2f}s")

        return ret_dict


training_agent_cls = partial(PPOTrainingAgent,
                             model_cls=VanillaCNNActorCritic,
                             gamma=0.99,
                             lam=0.95,
                             clip_ratio=0.6, # Normal value is 0.2
                             pi_lr=3e-4,
                             vf_lr=1e-3,
                             train_pi_iters=10,
                             train_v_iters=10,
                             target_kl=0.01,
                             debug=False)

training_cls = partial(
    TrainingOffline,
    env_cls=env_cls,
    memory_cls=memory_cls,
    training_agent_cls=training_agent_cls,
    epochs=epochs,
    rounds=rounds,
    steps=steps,
    update_buffer_interval=update_buffer_interval,
    update_model_interval=update_model_interval,
    max_training_steps_per_env_step=max_training_steps_per_env_step,
    start_training=start_training,
    device=device_trainer)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true', help='launches the server')
    parser.add_argument('--trainer', action='store_true', help='launches the trainer')
    parser.add_argument('--worker', action='store_true', help='launches a rollout worker')
    parser.add_argument('--test', action='store_true', help='launches a rollout worker in standalone mode')
    args = parser.parse_args()

    if args.trainer:
        my_trainer = Trainer(training_cls=training_cls,
                             server_ip=server_ip_for_trainer,
                             server_port=server_port,
                             password=password,
                             security=security)
        my_trainer.run()
    elif args.worker or args.test:
        rw = RolloutWorker(env_cls=env_cls,
                           actor_module_cls=MyActorModule,
                           sample_compressor=sample_compressor,
                           device=device_worker,
                           server_ip=server_ip_for_worker,
                           server_port=server_port,
                           password=password,
                           security=security,
                           max_samples_per_episode=max_samples_per_episode,
                           obs_preprocessor=obs_preprocessor,
                           standalone=args.test)
        rw.run(test_episode_interval=10)
    elif args.server:
        serv = Server(port=server_port,
                      password=password,
                      security=security)
        while True:
            time.sleep(1.0)