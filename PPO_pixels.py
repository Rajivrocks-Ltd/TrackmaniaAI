import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam
from math import floor

import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.util import partial
from tmrl.networking import Trainer, RolloutWorker, Server
from tmrl.actor import TorchActorModule
from tmrl.training_offline import TrainingOffline
from tmrl.training import TrainingAgent
from tmrl.custom.utils.nn import copy_shared, no_grad
from tmrl.util import cached_property

import numpy as np
import os
import time

epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]
rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]
steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]
start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]
max_training_steps_per_env_step = cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"]
update_model_interval = cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"]
update_buffer_interval = cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"]
device_trainer = 'cuda' if cfg.CUDA_TRAINING else 'cpu'
memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]
batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]

wandb_run_id = cfg.WANDB_RUN_ID  # change this by a name of your choice for your run
wandb_project = cfg.TMRL_CONFIG["WANDB_PROJECT"]  # name of the wandb project in which your run will appear
wandb_entity = cfg.TMRL_CONFIG["WANDB_ENTITY"]  # wandb account
wandb_key = cfg.TMRL_CONFIG["WANDB_KEY"]  # wandb API key

os.environ['WANDB_API_KEY'] = wandb_key  # this line sets your wandb API key as the active key

max_samples_per_episode = cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]

server_ip_for_trainer = cfg.SERVER_IP_FOR_TRAINER  # IP of the machine running the Server (trainer point of view)
server_ip_for_worker = cfg.SERVER_IP_FOR_WORKER  # IP of the machine running the Server (worker point of view)
server_port = cfg.PORT  # port used to communicate with this machine
password = cfg.PASSWORD  # password that secures your communication
security = cfg.SECURITY  # when training over the Internet, it is safer to change this to "TLS"


memory_base_cls = cfg_obj.MEM
sample_compressor = cfg_obj.SAMPLE_COMPRESSOR
sample_preprocessor = None
dataset_path = cfg.DATASET_PATH
obs_preprocessor = cfg_obj.OBS_PREPROCESSOR
env_cls = cfg_obj.ENV_CLS
device_worker = 'cpu'

window_width = cfg.WINDOW_WIDTH  # must be between 256 and 958
window_height = cfg.WINDOW_HEIGHT  # must be between 128 and 488

img_width = cfg.IMG_WIDTH
img_height = cfg.IMG_HEIGHT

img_grayscale = cfg.GRAYSCALE
imgs_buf_len = cfg.IMG_HIST_LEN
act_buf_len = cfg.ACT_BUF_LEN

memory_cls = partial(memory_base_cls,
                     memory_size=memory_size,
                     batch_size=batch_size,
                     sample_preprocessor=sample_preprocessor,
                     dataset_path=cfg.DATASET_PATH,
                     imgs_obs=imgs_buf_len,
                     act_buf_len=act_buf_len,
                     crc_debug=False)

LOG_STD_MAX = 2
LOG_STD_MIN = -20


# Here is the MLP:
def mlp(sizes, activation, output_activation=nn.Identity):
    """
    A simple MLP (MultiLayer Perceptron).

    Args:
        sizes: list of integers representing the hidden size of each layer
        activation: activation function of hidden layers
        output_activation: activation function of the last layer

    Returns:
        Our MLP in the form of a Pytorch Sequential module
    """
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
    h_out = floor((h_in + 2 * conv_layer.padding[0] - conv_layer.dilation[0] * (conv_layer.kernel_size[0] - 1) - 1) /
                  conv_layer.stride[0] + 1)
    w_out = floor((w_in + 2 * conv_layer.padding[1] - conv_layer.dilation[1] * (conv_layer.kernel_size[1] - 1) - 1) /
                  conv_layer.stride[1] + 1)
    return h_out, w_out


# Let us now define a module that will be the main building block of both our actor and critic:
class VanillaCNN(nn.Module):
    def __init__(self, q_net):
        """
        Simple CNN (Convolutional Neural Network) model.

        Args:
            q_net (bool): indicates whether this neural net is a critic network
        """
        super(VanillaCNN, self).__init__()

        self.q_net = q_net

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

        self.flat_features = self.out_channels * self.h_out * self.w_out

        float_features = 9 if self.q_net else 9
        self.mlp_input_features = self.flat_features + float_features

        self.mlp_layers = [256, 256, 1] if self.q_net else [256, 256]
        self.mlp = mlp([self.mlp_input_features] + self.mlp_layers, nn.ReLU)

    def forward(self, x):
        """
        In Pytorch, the forward function is where our neural network computes its output from its input.

        Args:
            x (torch.Tensor): input tensor (i.e., the observation fed to our deep neural network)

        Returns:
            the output of our neural network in the form of a torch.Tensor
        """

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

        x = x.view(-1, flat_features)

        x = torch.cat((speed, gear, rpm, x, act1, act2), -1)
        x = self.mlp(x)

        return x


class MyActorModule(TorchActorModule):
    """
    Our policy wrapped in the TMRL ActorModule class.

    The only required method is ActorModule.act().
    We also implement a forward() method for our training algorithm.

    (Note: TorchActorModule is a subclass of ActorModule and torch.nn.Module)
    """

    def __init__(self, observation_space, action_space):
        """
        When implementing __init__, we need to take the observation_space and action_space arguments.

        Args:
            observation_space: observation space of the Gymnasium environment
            action_space: action space of the Gymnasium environment
        """
        # We must call the superclass __init__:
        super().__init__(observation_space, action_space)

        dim_act = action_space.shape[0]  # dimensionality of actions
        act_limit = action_space.high[0]  # maximum amplitude of actions
        self.net = VanillaCNN(q_net=False)
        self.mu_layer = nn.Linear(256, dim_act)
        self.log_std_layer = nn.Linear(256, dim_act)
        self.act_limit = act_limit

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device):
        self.device = device
        self.load_state_dict(torch.load(path, map_location=self.device))
        self.to_device(device)
        return self

    def forward(self, obs, test=False, compute_logprob=True):
        """
        Computes the output action of our policy from the input observation.

        The whole point of deep RL is to train our policy network (actor) such that it outputs relevant actions.
        Training per-se will also rely on a critic network, but this is not part of the trained policy.
        Thus, our ActorModule will only implement the actor.

        Args:
            obs: the observation from the Gymnasium environment (when using TorchActorModule this is a torch.Tensor)
            test (bool): this is True for test episodes (deployment) and False for training episodes;
            compute_logprob (bool): SAC will set this to True to retrieve log probabilities.

        Returns:
            the action sampled from our policy from observation obs
            the log probability of this action, for PPO this is used to compute the loss
        """

        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        pi_distribution = Normal(mu, std)
        if test:
            pi_action = mu  # at test time, our action is deterministic (it is just the means)
        else:
            pi_action = pi_distribution.rsample()  # during training, it is sampled in the multivariate gaussian

        if compute_logprob:
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            # Without this line the agent goes haywire after the first episode. This is a SAC thing, but it helps
            # stabilize PPO too in our case.
            logp_pi -= (2 * (np.log(2) - pi_action - F.softplus(-2 * pi_action))).sum(axis=1)
        else:
            logp_pi = None
        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action
        pi_action = pi_action.squeeze()
        return pi_action, logp_pi

    def act(self, obs, test=False):
        """
        Computes an action from an observation.
        It is the policy that TMRL will use in TrackMania to evaluate your submission.

        Args:
            obs (object): the input observation (when using TorchActorModule, this is a torch.Tensor)
            test (bool): True at test-time (e.g., during evaluation...), False otherwise

        Returns:
            act (numpy.array): the computed action, in the form of a numpy array of 3 values between -1.0 and 1.0
        """

        with torch.no_grad():
            a, _ = self.forward(obs=obs, test=test, compute_logprob=False)
            return a.cpu().numpy()


# The critic module for PPO
class VanillaCNNValueFunction(nn.Module):
    """
    Critic module for PPO.
    """

    def __init__(self, observation_space):  # removed action_space from here
        super().__init__()
        self.net = VanillaCNN(q_net=True)  # q_net is True for a critic module

    def forward(self, obs):  # removed act from here
        x = obs  # was x = (*obs, act) before
        q = self.net(x)  # maybe add *x here?
        return torch.squeeze(q, -1)


class VanillaCNNActorCritic(nn.Module):
    """
    Actor-critic module for the PPO algorithm.
    """

    def __init__(self, observation_space, action_space):
        super().__init__()

        # Policy network (actor):
        self.actor = MyActorModule(observation_space, action_space)
        # Value networks (critic):
        self.critic = VanillaCNNValueFunction(observation_space)  # removed action_space from here


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
        self.gamma = gamma
        self.lam = lam
        self.clip_ratio = clip_ratio
        self.pi_lr = pi_lr
        self.vf_lr = vf_lr
        self.train_pi_iters = train_pi_iters
        self.train_v_iters = train_v_iters
        self.target_kl = target_kl
        self.pi_optimizer = Adam(self.model.actor.parameters(), lr=self.pi_lr, betas=(0.997, 0.997))
        self.vf_optimizer = Adam(self.model.critic.parameters(), lr=self.vf_lr, betas=(0.997, 0.997))
        self.debug = debug

    def get_actor(self):
        return self.model_nograd.actor

    def compute_returns_and_advantages(self, batch):
        o, a, r, o2, d, _ = batch

        with torch.no_grad():
            values = self.model.critic(o)
            next_values = self.model.critic(o2)

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
        # logging.info("Training error " + "-" * 50)
        start_time = time.time()
        o, a, r, o2, d, _ = batch

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

        compute_start = time.time()
        ret, adv = self.compute_returns_and_advantages((o, a, r, o2, d, _))
        compute_time = time.time() - compute_start

        with torch.no_grad():
            _, logp_old = self.model.actor(o, compute_logprob=True)

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

        value_loss_start = time.time()
        for _ in range(self.train_v_iters):
            values = self.model.critic(o)
            loss_v = ((values - ret) ** 2).mean()
            self.vf_optimizer.zero_grad()
            loss_v.backward()
            self.vf_optimizer.step()
        value_loss_time = time.time() - value_loss_start

        ret_dict = dict(
            loss_actor=loss_pi.item(),
            loss_critic=loss_v.item(),
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
                             clip_ratio=0.2,  # Normal value is 0.2
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

        my_trainer.run_with_wandb(entity=wandb_entity,
                                  project=wandb_project,
                                  run_id=wandb_run_id)

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
