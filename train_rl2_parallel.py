import os
import time
from typing import Any, Dict, List
from distutils.util import strtobool
import argparse
import yaml
import datetime
# import ray

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
from networks import RL2Actor, RL2Critic
from buffers.buffer import RolloutBuffer
from meta_learner_parallel import MetaLearner
from torch.utils.tensorboard import SummaryWriter
from utils.tb_logger import TBLogger
from utils.sampler import BaseSampler, RL2Sampler

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# GPU_NUM = 1 # 원하는 GPU 번호 입력
# device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
# torch.cuda.set_device(device) # change allocation of current GPU

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

def parse_args():
    parser = argparse.ArgumentParser()

    # ? Experiments information
    parser.add_argument('--exp_name', type=str, default="RL2_Parallel",
                        help="the name of this experiment")
    parser.add_argument("--meta_learning", type=bool, default=True)
    parser.add_argument("--load", type=bool, default=False)
    parser.add_argument("--load_ckpt_num", type=int, default=0)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)),
                        default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument('--device', default='cuda')
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--render", type=bool, default=True)
    parser.add_argument("--render_mode", type=str, default=None)
    parser.add_argument("--weight_path", type=str, default="./weights",
                        help="weight path for saving model")
    parser.add_argument("--save_periods", type=int, default=100)
    parser.add_argument("--results_log_dir", type=str, default="./logs",
                        help="directory of tensorboard")

    #  Environments information
    # parser.add_argument("--env_name", type=str, default="PointEnv")
    # parser.add_argument("--env_name", type=str, default="SparsePointEnv")
    parser.add_argument("--env_name", type=str, default="HalfCheetahDirEnv")
    # parser.add_argument("--env_name", type=str, default="AntDirEnv")
    # parser.add_argument("--env_name", type=str, default="AntDir2DEnv")
    # parser.add_argument("--env_name", type=str, default="AntGoalEnv")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="total timesteps of the experiments")
    parser.add_argument('--rollout_steps', default=512)
    parser.add_argument('--num_episodes_per_trial', default=2)
    parser.add_argument('--max_episode_steps', default=1000)
    parser.add_argument('--max_samples', default=2000)
    
    parser.add_argument("--num-envs", type=int, default=1,
                        help="the number of parallel game environments")
    parser.add_argument("--parallel_processing", type=bool, default=True)
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")

    # Hyperparameter config
    parser.add_argument('--config_path', type=str,
                        default='./configs/cheetah_dir_config.yaml')
    args = parser.parse_args()
    return args


def make_env_tasks(args, configs):
    env = gym.make(args.env_name, num_tasks=configs["num_tasks"])
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

def train():
    pass
    

if __name__ == "__main__":
    # argparser, configs, logger
    args = parse_args()
    with open(args.config_path) as file:
        configs: Dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)
    args.max_samples = args.num_episodes_per_trial * args.max_episode_steps
    args.batch_size = int(configs["meta_batch_size"] * args.max_samples)
    args.now = datetime.datetime.now().strftime('_%m.%d_%H:%M:%S')

    tb_logger = TBLogger(args, configs)

    # env, seed
    if args.meta_learning:
        env, tasks = make_env_tasks(args, configs)
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

    # buffer, sampler
    buffer = RolloutBuffer(args, configs)

    # execution
    # sampler = RL2Sampler(env, agent, args, configs)
    # for i in range(10):
    #     trajs = sampler.obtain_samples()
    #     buffer.add_trajs(trajs)

    # train
    print("Meta training start...")
    meta_learner = MetaLearner(
                    env_creator=make_env_tasks,
                    agent=agent,
                    tb_logger=tb_logger, 
                    train_tasks=list(tasks[: configs["num_train_tasks"]]),
                    test_tasks=list(tasks[-configs["num_test_tasks"] :]),
                    args=args, configs=configs
                    )

    meta_learner.meta_train_parallel()

