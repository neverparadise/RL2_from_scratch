import os
import time
from typing import Any, Dict, List
from distutils.util import strtobool
import argparse
import yaml

import gym
from gym.envs.registration import register
from gym.spaces import Box, Discrete
import pygame
from envs import *

import numpy as np
import random
import torch
import torch.optim as optim
from ppo import PPO
from networks import RNNActor, RNNCritic
from buffers.buffer import RolloutBuffer
from meta_learner import MetaLearner
from torch.utils.tensorboard import SummaryWriter
from utils.tb_logger import TBLogger
from utils.sampler import Sampler
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser()

    # ? Experiments information
    parser.add_argument('--exp_name', type=str, default="RL2PPO_Evaluation",
                        help="the name of this experiment")
    parser.add_argument("--meta_learning", type=bool, default=True)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)),
                        default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--render", type=bool, default=True)
    parser.add_argument("--render_mode", type=str, default="human")
    parser.add_argument("--weight_path", type=str, default="./weights",
                        help="weight path for saving model")
    parser.add_argument("--load", type=bool, default=True)
    parser.add_argument("--load_ckpt_num", type=int, default=100)
    parser.add_argument("--results_log_dir", type=str, default="./logs",
                        help="directory of tensorboard")

    #  Environments information
    parser.add_argument("--env_name", type=str, default="HalfCheetahDirEnv")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="total timesteps of the experiments")
    parser.add_argument('--rollout_steps', default=512)
    parser.add_argument('--max_episode_steps', default=500)
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument("--num-tasks", type=int, default=4)  # meta batch size

    # Hyperparameter config
    parser.add_argument('--config_path', type=str,
                        default='./configs/dir_config.yaml')
    args = parser.parse_args()
    return args


def make_env_tasks(args):
    env = gym.make(args.env_name, num_tasks=args.num_tasks)
    tasks: List[int] = env.get_all_task_idx()
    return env, tasks


def make_env(args):
    env = gym.make(args.env_name)
    return env


def add_state_action_info(env, configs):
    print(env.observation_space)
    print(env.action_space)
    state_dim = env.observation_space.shape[0]
    num_discretes = None
    if isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]
        is_continuous = True
    elif isinstance(env.action_space, Discrete):
        action_dim = 1
        num_discretes = env.action_space.n
        is_continuous = False
    configs.update({"state_dim": state_dim,
                    "action_dim": action_dim,
                    "num_discretes": num_discretes,
                    "is_continuous": is_continuous})
    return configs


# argparser, configs, logger
if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path) as file:
        configs: Dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)
    args.batch_size = int(configs["meta_batch_size"] * args.rollout_steps)
    tb_logger = TBLogger(args, configs)

    # env, seed
    if args.meta_learning:
        env, tasks = make_env_tasks(args)
    else:
        env = make_env(args)

    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    configs = add_state_action_info(env, configs)

    # agent
    agent = PPO(args, configs)

    ckpt_path = f"{args.weight_path}" + '/' +f"{args.env_name}_RL2_{args.seed}" + f"/checkpoint_" + f"{str(args.load_ckpt_num)}.pt"
    print(ckpt_path)
    agent.load(ckpt_path)
    # buffer, sampler
    buffer = RolloutBuffer(args, configs)

    sampler = Sampler(env, agent, args, configs)
    for i in range(3):
        test_return = 0.0
        trajs = sampler.obtain_samples()
        test_return += np.array([np.sum(trajs[i]["rewards"]) for i in range(len(trajs))]).mean().item()
        print(f"test return: {test_return}")
