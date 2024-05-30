import rtgym
from tmrl.custom.custom_checkpoints import update_run_instance
from tmrl.custom.custom_preprocessors import obs_preprocessor_tm_lidar_act_in_obs

from tmrl.envs import GenericGymEnv
from tmrl.custom.custom_gym_interfaces import TM2020InterfaceLidar
from tmrl.custom.custom_memories import MemoryTMLidar, get_local_buffer_sample_lidar
from tmrl.training_offline import TorchTrainingOffline
from tmrl.util import partial
import tmrl.config.config_constants as cfg

from actor_critic_model import MLPActorCritic, SquashedGaussianMLPActor
from actor_critic_agent import ActorCriticAgent


ALG_NAME = "LIDAR-SAC"

# Algorithm hyperparameters
learn_entropy_coef = False
lr_actor = 0.00001
lr_critic = 0.00005
lr_entropy = 0.0003
gamma = 0.995
polyak = 0.995
target_entropy = -0.5
alpha = 0.01
optimizer_actor = "adam"
optimizer_critic = "adam"
betas_actor = [0.997, 0.997]
betas_critic = [0.997, 0.997]
l2_actor = 0.0
l2_critic = 0.0

# Other algorithm's parameters
DUMP_RUN_INSTANCE_FN = None
LOAD_RUN_INSTANCE_FN = None
UPDATER_FN = update_run_instance if ALG_NAME in ["LIDAR-SAC"] else None


# Model and policy
TRAIN_MODEL = MLPActorCritic
POLICY = SquashedGaussianMLPActor


# LIDAR Interface and config dict
INTERFACE = partial(TM2020InterfaceLidar, img_hist_len=cfg.IMG_HIST_LEN, gamepad=cfg.PRAGMA_GAMEPAD)

CONFIG_DICT = rtgym.DEFAULT_CONFIG_DICT.copy()
CONFIG_DICT["interface"] = INTERFACE
CONFIG_DICT_MODIFIERS = cfg.ENV_CONFIG["RTGYM_CONFIG"]
for k, v in CONFIG_DICT_MODIFIERS.items():
    CONFIG_DICT[k] = v


# Sample and observation processors
OBS_PREPROCESSOR = obs_preprocessor_tm_lidar_act_in_obs
SAMPLE_COMPRESSOR = get_local_buffer_sample_lidar
SAMPLE_PREPROCESSOR = None


# MEMORY AND ENV:
MEMORY = partial(MemoryTMLidar,
                 memory_size=cfg.TMRL_CONFIG["MEMORY_SIZE"],
                 batch_size=cfg.TMRL_CONFIG["BATCH_SIZE"],
                 sample_preprocessor=SAMPLE_PREPROCESSOR,
                 dataset_path=cfg.DATASET_PATH,
                 imgs_obs=cfg.IMG_HIST_LEN,
                 act_buf_len=cfg.ACT_BUF_LEN,
                 crc_debug=cfg.CRC_DEBUG)

ENV_CLS = partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": CONFIG_DICT})

# ALGORITHM:
AGENT = partial(
        ActorCriticAgent,
        device='cuda' if cfg.CUDA_TRAINING else 'cpu',
        model_cls=TRAIN_MODEL,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
        lr_entropy=lr_entropy,
        gamma=gamma,
        polyak=polyak,
        learn_entropy_coef=learn_entropy_coef,
        target_entropy=target_entropy,
        alpha=alpha,
        optimizer_actor=optimizer_actor,
        optimizer_critic=optimizer_critic,
        # Take a look if the changes below didn't break the algorithm
        betas_actor=betas_actor,
        betas_critic=betas_critic,
        l2_actor=l2_actor,
        l2_critic=l2_critic
    )

# TRAINER:
TRAINER = partial(
        TorchTrainingOffline,
        env_cls=ENV_CLS,
        memory_cls=MEMORY,
        epochs=cfg.TMRL_CONFIG["MAX_EPOCHS"],
        rounds=cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"],
        steps=cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"],
        update_model_interval=cfg.TMRL_CONFIG["UPDATE_MODEL_INTERVAL"],
        update_buffer_interval=cfg.TMRL_CONFIG["UPDATE_BUFFER_INTERVAL"],
        max_training_steps_per_env_step=cfg.TMRL_CONFIG["MAX_TRAINING_STEPS_PER_ENVIRONMENT_STEP"],
        profiling=cfg.PROFILE_TRAINER,
        training_agent_cls=AGENT,
        agent_scheduler=None,
        start_training=cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"])
