#!/usr/bin/env python
import sys
import os
import pickle
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path

import torch

from mapbt.config import get_config

from mapbt.envs.overcooked.Overcooked_Env import Overcooked
from mapbt.envs.overcooked.bc_tools import extract_human_data
from mapbt.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv, ChooseSubprocVecEnv, ChooseDummyVecEnv


def make_train_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                env = Overcooked(all_args, run_dir, featurize_type=("bc", "bc"))
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args, run_dir):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Overcooked":
                env = Overcooked(all_args, run_dir, featurize_type=("bc", "bc"))
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1:
        return ChooseDummyVecEnv([get_env_fn(0)])
    else:
        return ChooseSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])

def parse_args(args, parser):
    parser.add_argument("--use_image_state", default=False, action='store_true', help="transform the full state into the image-like observation (obs for ppo)")
    parser.add_argument("--raw_data_file_path", type=str, default=None)

    parser.add_argument("--human_data_refresh", default=False, action='store_true', help="re-compute formatted data from saved human trajectories")
    parser.add_argument("--human_data_split", type=str, default='2019-train', choices=["2019-train", "2019-test", "2020-train", "2024-train", "2024-test"])
    parser.add_argument("--human_layout_name", type=str, default='cramped_room', help="layout name for human trajectories")
    parser.add_argument('--bc_validation_split', type=float, default=0.1, help="how much data is used for validation")
    parser.add_argument('--bc_num_epochs', type=int, default=1, help="number of epochs for training")
    parser.add_argument('--bc_batch_size', type=int, default=64, help="number of epochs for training")
    parser.add_argument("--old_dynamics", default=False, action='store_true', help="old_dynamics in mdp")

    parser.add_argument("--layout_name", type=str, default='cramped_room', help="Name of Submap, 40+ in choice. See /src/data/layouts/.")
    parser.add_argument('--num_agents', type=int, default=1, help="number of players")
    parser.add_argument("--initial_reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_factor", type=float, default=1.0, help="Shaping factor of potential dense reward.")
    parser.add_argument("--reward_shaping_horizon", type=int, default=2.5e6, help="Shaping factor of potential dense reward.")
    parser.add_argument("--use_phi", default=False, action='store_true', help="While existing other agent like planning or human model, use an index to fix the main RL-policy agent.")
    parser.add_argument("--use_hsp", default=False, action='store_true') 
    parser.add_argument("--random_index", default=False, action='store_true') 
    parser.add_argument("--w0", type=str, default="1,1,1,1", help="Weight vector of dense reward 0 in overcooked env.")
    parser.add_argument("--w1", type=str, default="1,1,1,1", help="Weight vector of dense reward 1 in overcooked env.") 
    parser.add_argument("--predict_other_shaped_info", default=False, action='store_true', help="Predict other agent's shaped info within a short horizon, default False")
    parser.add_argument("--predict_shaped_info_horizon", default=50, type=int, help="Horizon for shaped info target, default 50")
    parser.add_argument("--predict_shaped_info_event_count", default=10, type=int, help="Event count for shaped info target, default 10")
    parser.add_argument("--use_task_v_out", default=False, action='store_true')
    parser.add_argument("--random_start_prob", default=0., type=float, help="Probability to use a random start state, default 0.")
    parser.add_argument("--use_detailed_rew_shaping", default=False, action="store_true")
    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                   0] + "/results") / all_args.env_name / all_args.layout_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.wandb_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.layout_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True,
                         tags=all_args.wandb_tags)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    human_data = extract_human_data(all_args)
    with open(all_args.raw_data_file_path, "wb") as f:
        pickle.dump(human_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == "__main__":
    main(sys.argv[1:])
