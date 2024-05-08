from argparse import ArgumentParser

from tmrl.networking import Server, Trainer, RolloutWorker
import tmrl.config.config_objects as cfg_obj
import tmrl.config.config_constants as cfg
from tmrl.util import partial
from tmrl.training_offline import TrainingOffline

from agents.ActorModule import SimpleActorModule
from agents.TrainingAgent import SimpleTrainingAgent


class TrackManiaPipeline:
    def __init__(self):
        # Load configuration
        self.epochs = cfg.TMRL_CONFIG["MAX_EPOCHS"]
        self.rounds = cfg.TMRL_CONFIG["ROUNDS_PER_EPOCH"]
        self.steps = cfg.TMRL_CONFIG["TRAINING_STEPS_PER_ROUND"]
        self.start_training = cfg.TMRL_CONFIG["ENVIRONMENT_STEPS_BEFORE_TRAINING"]
        self.device_trainer = 'cuda' if cfg.CUDA_TRAINING else 'cpu'
        self.batch_size = cfg.TMRL_CONFIG["BATCH_SIZE"]
        self.memory_size = cfg.TMRL_CONFIG["MEMORY_SIZE"]
        self.server_ip_for_trainer = cfg.SERVER_IP_FOR_TRAINER
        self.server_ip_for_worker = cfg.SERVER_IP_FOR_WORKER
        self.server_port = cfg.PORT
        self.password = cfg.PASSWORD
        self.security = cfg.SECURITY
        self.obs_preprocessor = cfg_obj.OBS_PREPROCESSOR
        self.max_samples_per_episode = cfg.TMRL_CONFIG["RW_MAX_SAMPLES_PER_EPISODE"]

        # Set the environment class
        self.env_cls = cfg_obj.ENV_CLS
        self.observation_space, self.action_space = self.get_environment_spaces()

        self.training_agent_cls = partial(SimpleTrainingAgent,
                                          observation_space=self.observation_space,
                                          action_space=self.action_space,
                                          device=self.device_trainer)

        self.memory_cls = partial(cfg_obj.MEM,
                                  memory_size=self.memory_size,
                                  batch_size=self.batch_size)

        self.training_cls = partial(
            TrainingOffline,
            env_cls=self.env_cls,
            memory_cls=self.memory_cls,
            training_agent_cls=self.training_agent_cls,
            epochs=self.epochs,
            rounds=self.rounds,
            steps=self.steps,
            device=self.device_trainer)

    def get_environment_spaces(self):
        """Retrieve observation and action spaces from the environment class."""
        env = self.env_cls()
        print(env.observation_space, env.action_space)
        return env.observation_space, env.action_space

    def run(self, mode):
        if mode == 'server':
            import time
            server = Server(port=self.server_port, password=self.password, security=self.security)
            while True:
                time.sleep(1.0)
        elif mode == 'trainer':
            trainer = Trainer(training_cls=self.training_cls, server_ip=self.server_ip_for_trainer,
                              server_port=self.server_port, password=self.password, security=self.security)
            trainer.run()
        elif mode == 'worker':
            worker = RolloutWorker(env_cls=self.env_cls, actor_module_cls=SimpleActorModule,
                                   sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
                                   device='cpu', server_ip=self.server_ip_for_worker,
                                   server_port=self.server_port, password=self.password,
                                   security=self.security, max_samples_per_episode=self.max_samples_per_episode,
                                   obs_preprocessor=self.obs_preprocessor, standalone=False)
            worker.run()
        elif mode == 'test':
            worker = RolloutWorker(env_cls=cfg_obj.ENV_CLS, actor_module_cls=SimpleActorModule,
                                   sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
                                   device='cpu', server_ip=self.server_ip_for_worker,
                                   server_port=self.server_port, password=self.password,
                                   security=self.security, max_samples_per_episode=self.max_samples_per_episode,
                                   obs_preprocessor=self.obs_preprocessor, standalone=True)
            worker.run()
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of ['server', 'trainer', 'worker', 'test'].")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true', help='launches the server')
    parser.add_argument('--trainer', action='store_true', help='launches the trainer')
    parser.add_argument('--worker', action='store_true', help='launches a rollout worker')
    parser.add_argument('--test', action='store_true', help='launches a rollout worker in standalone mode')
    args = parser.parse_args()

    tm_pipeline = TrackManiaPipeline()

    if args.server:
        tm_pipeline.run('server')
    elif args.trainer:
        tm_pipeline.run('trainer')
    elif args.worker:
        tm_pipeline.run('worker')
    elif args.test:
        tm_pipeline.run('test')
