from tmrl.util import partial
from actors.trainingAgent import SACTrainingAgent
from actors.ActorCritic import VanillaCNNActorCritic
from actors.baseActor import MyActorModule
import tmrl.config.config_constants as cfg
import tmrl.config.config_objects as cfg_obj
from tmrl.networking import Trainer, RolloutWorker, Server
from tmrl.training_offline import TrainingOffline
from argparse import ArgumentParser
import os

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

# Wandb credentials
wandb_run_id = cfg.WANDB_RUN_ID
wandb_project = cfg.TMRL_CONFIG["WANDB_PROJECT"]
wandb_entity = cfg.TMRL_CONFIG["WANDB_ENTITY"]
wandb_key = cfg.TMRL_CONFIG["WANDB_KEY"]

os.environ['WANDB_API_KEY'] = wandb_key

max_samples_per_episode = cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]

# Networking parameters:
server_ip_for_trainer = cfg.SERVER_IP_FOR_TRAINER
server_ip_for_worker = cfg.SERVER_IP_FOR_WORKER
server_port = cfg.PORT
password = cfg.PASSWORD
security = cfg.SECURITY

# Advanced parameters
memory_base_cls = cfg_obj.MEM
sample_compressor = cfg_obj.SAMPLE_COMPRESSOR
sample_preprocessor = None
dataset_path = cfg.DATASET_PATH
obs_preprocessor = cfg_obj.OBS_PREPROCESSOR

env_cls = cfg_obj.ENV_CLS
device_worker = 'cpu'

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

# Partially instantiate the training agent
training_agent_cls = partial(SACTrainingAgent,
                             model_cls=VanillaCNNActorCritic,
                             gamma=0.99,
                             polyak=0.995,
                             alpha=0.02,
                             lr_actor=0.000005,
                             lr_critic=0.00003)

# TMRL Trainer

training_cls = partial(TrainingOffline,
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
        rw.run()
    elif args.server:
        import time
        serv = Server(port=server_port,
                      password=password,
                      security=security)
        while True:
            time.sleep(1.0)