import logging
import time
from argparse import ArgumentParser, ArgumentTypeError

from tmrl.networking import Server, Trainer, RolloutWorker
from tmrl.envs import GenericGymEnv
from tmrl.util import partial

import tmrl.config.config_constants as cfg
import pipeline_cfg as cfg_obj


def main(args):
    # Run server
    if args.server:
        serv = Server()
        while True:
            time.sleep(1.0)

    # Run worker or test
    elif args.worker or args.test:
        config = cfg_obj.CONFIG_DICT
        rw = RolloutWorker(env_cls=partial(GenericGymEnv, id=cfg.RTGYM_VERSION, gym_kwargs={"config": config}),
                           actor_module_cls=cfg_obj.POLICY,
                           sample_compressor=cfg_obj.SAMPLE_COMPRESSOR,
                           device='cuda' if cfg.CUDA_INFERENCE else 'cpu',
                           server_ip=cfg.SERVER_IP_FOR_WORKER,
                           max_samples_per_episode=cfg.RW_MAX_SAMPLES_PER_EPISODE,
                           model_path=cfg.MODEL_PATH_WORKER,
                           obs_preprocessor=cfg_obj.OBS_PREPROCESSOR,
                           crc_debug=cfg.CRC_DEBUG,
                           standalone=args.test)
        if args.worker:
            rw.run()
        else:
            rw.run_episodes(10000)

    # Run trainer
    elif args.trainer:
        trainer = Trainer(training_cls=cfg_obj.TRAINER,
                          server_ip=cfg.SERVER_IP_FOR_TRAINER,
                          model_path=cfg.MODEL_PATH_TRAINER,
                          checkpoint_path=cfg.CHECKPOINT_PATH,
                          dump_run_instance_fn=cfg_obj.DUMP_RUN_INSTANCE_FN,
                          load_run_instance_fn=cfg_obj.LOAD_RUN_INSTANCE_FN,
                          updater_fn=cfg_obj.UPDATER_FN)

        logging.info(f"--- NOW RUNNING {cfg_obj.ALG_NAME} on TrackMania ---")
        trainer.run_with_wandb(entity=cfg.WANDB_ENTITY,
                               project=cfg.WANDB_PROJECT,
                               run_id=cfg.WANDB_RUN_ID)
    else:
        raise ArgumentTypeError('Enter a valid argument')


if __name__ == "__main__":
    # Remember to run them in sequence: server -> trainer -> worker
    parser = ArgumentParser()
    parser.add_argument('--server', action='store_true', help='launches the server')
    parser.add_argument('--trainer', action='store_true', help='launches the trainer')
    parser.add_argument('--worker', action='store_true', help='launches a rollout worker')
    parser.add_argument('--test', action='store_true', help='launches a rollout worker in standalone mode')
    arguments = parser.parse_args()

    main(arguments)
