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
from torch.distributions import Categorical, Normal
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.tensorboard import SummaryWriter
from utils.tb_logger import TBLogger
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

from ppo import PPO
from networks import RNNActor, RNNCritic
from buffers.buffer import RolloutBuffer
from meta_learner import MetaLearner
from utils.sampler import BaseSampler, RL2Sampler

def parse_args():
    parser = argparse.ArgumentParser()

    # ? Experiments information
    parser.add_argument('--exp_name', type=str, default="CleanRL^2",
                        help="the name of this experiment")
    parser.add_argument("--load", type=bool, default=False)
    parser.add_argument("--load_ckpt_num", type=int, default=0)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)),
                        default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument("--seed", type=int, default=302,
                        help="seed of the experiment")
    parser.add_argument("--render", type=bool, default=False)
    parser.add_argument("--render_mode", type=str, default=None)
    parser.add_argument("--weight_path", type=str, default="./weights",
                        help="weight path for saving model")
    parser.add_argument("--save_periods", type=int, default=50)
    # ? training 시 바꿔줄 것 (False)
    parser.add_argument("--meta_learning", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--shuffle_mb", type=lambda x: bool(strtobool(x)), default=True)
    parser.add_argument("--same_task", type=lambda x: bool(strtobool(x)), default=True)
    
    parser.add_argument("--debug", type=bool, default=True)
    parser.add_argument("--results_log_dir", type=str, default="./logs",
                        help="directory of tensorboard")

    #  Environments information
    parser.add_argument("--env_name", type=str, default="HalfCheetahDirEnv")
    # parser.add_argument("--env_name", type=str, default="HalfCheetah-v3")
    # parser.add_argument("--env_name", type=str, default="CartPole-v1")
    
    # parser.add_argument("--total-timesteps", type=int, default=1000000,
    #                     help="total timesteps of the experiments")
    # parser.add_argument('--rollout_steps', default=512)
    parser.add_argument('--num_episodes_per_trial', default=2)
    parser.add_argument('--max_episode_steps', default=500)
    # parser.add_argument("--num-envs", type=int, default=1,
                        # help="the number of parallel game environments")
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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def to_tensor(np_array, device="cpu"):
    if isinstance(np_array, np.ndarray):
        return torch.from_numpy(np_array).float().to(device)
    return np_array.float().to(device)


def _format(x, dim, device, minibatch_size=1, is_training=False):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.to(device=device)
    else:
        x = x.to(device=device)
    if len(x.shape) < 3:
        x = x.reshape(1, 1, -1) # [L, N, flatten]
    if is_training:
        x = x.reshape(1, -1, dim)
    return x


class Agent(nn.Module):
    def __init__(self, args, configs):
        super().__init__()
        self.device = torch.device(args.device)
        self.state_dim = configs["state_dim"]
        self.linear_dim = configs["linear_dim"]
        self.action_dim= configs["action_dim"]
        self.meta_learning = args.meta_learning
        if self.meta_learning:
            self.input_dim = self.state_dim + self.action_dim + 2 # 2: reward, done dimension
        else:
            self.input_dim = self.state_dim
        self.minibatch_size = configs["mini_batch_size"]
        self.is_continuous = configs["is_continuous"]
        self.hidden_dim = configs["hidden_dim"]
        self.num_rnn_layers = configs["num_rnn_layers"]
        self.action_dim = configs["action_dim"]
        self.num_discretes = configs["num_discretes"]
        self.is_deterministic = False
        
        self.embedding = nn.Sequential(
            layer_init(nn.Linear(self.input_dim, self.linear_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.linear_dim, self.linear_dim)),
            nn.Tanh(),
            nn.LeakyReLU(),
            layer_init(nn.Linear(self.linear_dim, self.linear_dim)),
            nn.Tanh(),
        )
        
        self.gru = nn.GRU(self.linear_dim, self.hidden_dim, \
                            num_layers=self.num_rnn_layers, bias=True)
        if self.is_continuous:
            self.mean = nn.Sequential(
            layer_init(nn.Linear(self.hidden_dim, self.linear_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.linear_dim, self.linear_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.linear_dim, self.action_dim), std=0.01),
        )
            #self.actor_logstd = -0.5 * np.ones(self.action_dim, dtype=np.float32)
            #self.actor_logstd = torch.nn.Parameter(torch.Tensor(self.log_std))
            self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))
            
            #self.std =  layer_init(nn.Linear(self.hidden_dim, self.action_dim))
        else:
            self.policy_logits =  nn.Sequential(
                                layer_init(nn.Linear(self.hidden_dim, self.num_discretes)),
                                )
        
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.hidden_dim, self.linear_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.linear_dim, self.linear_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(self.linear_dim, 1))
        )
        
    def forward(self, transition, hidden=None, is_training=False):
        if self.meta_learning:
            state, action, reward, done = transition
            state = _format(state, self.state_dim, self.device, self.minibatch_size, is_training)
            action = _format(action, self.action_dim, self.device, self.minibatch_size, is_training)
            reward = _format(reward, 1, self.device, self.minibatch_size, is_training)
            done = _format(done, 1, self.device, self.minibatch_size, is_training)
            concatenated = torch.cat([state, action, reward, done], dim=-1)
        else:
            state = transition 
            concatenated = _format(state, self.state_dim, self.device, self.minibatch_size, is_training)
        if is_training:
            hidden = hidden.permute(1, 0, 2).contiguous() # (L, N, H)
            # ! mini_batch_size, num_rnn_layers, hidden_dim -> num_rnn, mb_size, hidden_dim
        x = self.embedding(concatenated)
        hidden = to_tensor(hidden, device=self.device)
        x, new_hidden = self.gru(x, hidden)
        if self.is_continuous:
            mu = self.mean(x)
            action_logstd = self.actor_logstd.expand_as(mu)
            std = torch.exp(action_logstd)
            # std = F.softplus(self.std(x))
            dist = Normal(mu, std)
        else:
            logits = self.policy_logits(x)
            prob = F.softmax(logits, dim=-1)
            dist = Categorical(prob)
        return x, dist, new_hidden

    def get_value(self, x, hidden, is_training=False):
        x, dist, new_hidden = self.forward(x, hidden, is_training=is_training)
        return self.critic(x).detach().cpu().numpy()
    
    def get_action_and_value(self, transition ,hidden, action=None, is_training=False):
        x, dist, new_hidden = self.forward(transition, hidden, is_training=is_training)
        value = self.critic(x)
        entropy = dist.entropy()
        
        if is_training:
            return action, dist.log_prob(action), entropy, value, new_hidden
        else: # evaluation
            if self.is_deterministic:
                action = dist.mean
                log_prob = torch.zeros(1)
            else: # stochastic
                action = dist.sample()
                log_prob = dist.log_prob(action)
            # while len(action.shape) != 1:
            #     action = action.squeeze(0)
            #     log_prob = log_prob.squeeze(0)

            log_prob = log_prob.detach()
            value = value.detach()
            if self.is_continuous:
                action = action.reshape(self.action_dim,)
                log_prob = log_prob.reshape(self.action_dim,)
                action = action.detach()
                return action.cpu(), log_prob.cpu(), entropy.cpu(), value.cpu(), new_hidden.cpu()
                
            else:
                action = action.detach().item() # .item() ? 
                return action, log_prob.cpu(), entropy.cpu(), value.cpu(), new_hidden.cpu()

if __name__ == "__main__":
    args = parse_args()
    with open(args.config_path) as file:
        configs: Dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)

    # environment
    if args.meta_learning:
        env, tasks = make_env_tasks(args, configs)
        num_train_tasks = int(configs["num_train_tasks"])
        train_tasks = [0, 1] # 1 direction
        test_tasks = [0, 1] # -1 direction
    
    
    else:
        args.num_episodes_per_trial = 1
        configs["meta_batch_size"] = 1
        args.exp_name = "CleanRL^2_No_MetaRL"
        #args.env_name = "HalfCheetah-v3"
        #args.env_name = "MountainCar-v0"
        #args.env_name = "MountainCarContinuous-v0"
        #args.env_name = "CartPole-v1"

        env = gym.make(args.env_name)
        train_tasks = [0]
        test_tasks = [0]
        
    if args.shuffle_mb:
        args.exp_name += "_ShuffleMB"
    else:
        args.exp_name += "_NoShuffle"
    
    args.batch_size = int(configs["meta_batch_size"] * args.max_episode_steps * args.num_episodes_per_trial)
    args.now = datetime.datetime.now().strftime('_%m.%d_%H:%M:%S')
    
    configs = add_state_action_info(env, configs)

    # logger
    tb_logger = TBLogger(args, configs)
    
    # seed
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # agent
    agent = Agent(args, configs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=float(configs["lr"]), eps=1e-5)
    # ! batch_size = meta_batch_size * max_steps or meta_batch_size * rollout_steps
    b_observations = torch.zeros((args.batch_size, configs["state_dim"]), dtype=torch.float32)
    b_actions = torch.zeros((args.batch_size, configs["action_dim"]), dtype=torch.float32)
    b_rewards = torch.zeros((args.batch_size,1), dtype=torch.float32)
    b_dones = torch.zeros((args.batch_size,1), dtype=torch.float32)
    b_log_probs = torch.zeros((args.batch_size, configs["action_dim"]), dtype=torch.float32)
    b_values = torch.zeros((args.batch_size,1), dtype=torch.float32)
    b_returns = torch.zeros((args.batch_size,1), dtype=torch.float32)
    b_advantages = torch.zeros((args.batch_size,1), dtype=torch.float32)
    b_hiddens = torch.zeros((args.batch_size, configs["num_rnn_layers"], configs["hidden_dim"]), dtype=torch.float32)

    total_start_time = time.time()
    for n_epoch in range(1, configs["n_epochs"] + 1):
            # buffer
        start_time = time.time()
        if args.anneal_lr:
            frac = 1.0 - (n_epoch - 1.0) / configs["n_epochs"]
            lrnow = frac * float(configs["lr"])
            optimizer.param_groups[0]["lr"] = lrnow
        
        print(f"=============== Epoch {n_epoch} ===============")
        start_time = time.time()
        results = {}
        
        indices = np.random.randint(len(train_tasks), size=configs["meta_batch_size"])
        print(f"Task indices {indices}")
        train_return = np.array([])
        
        # ? meta batch sampling (task sampling)
        pt = 0
        final_pt = 0
        if args.meta_learning:
            for i, index in enumerate(indices): # 0, 1, 2, 3
                env.seed(args.seed + i)
                if args.same_task:
                    index = 0
                env.reset_task(index)
                agent.is_deterministic = False
                meta_batch_size = configs["meta_batch_size"]
                print(f"[{i + 1}/{meta_batch_size}] collecting samples, current task: {env.get_task()}")
                
                # ? episode rollout per trial
                hidden = torch.zeros((configs["num_rnn_layers"], 1, configs["hidden_dim"]))
                obs = env.reset()
                action = torch.zeros(configs["action_dim"])
                reward = torch.zeros(1)
                done = torch.zeros(1)
                for epi in range(args.num_episodes_per_trial): # 0, 1
                    cur_step = 0
                    while not (done or cur_step == args.max_episode_steps):
                        tran = (obs, action, reward, done)
                        with torch.no_grad():
                            action, log_prob, entropy, value, new_hidden \
                                = agent.get_action_and_value(tran, hidden)
                        if configs['is_continuous']:
                            action = action.numpy()
                        next_obs, reward, done, info = env.step(action)
                        reward = np.array(reward)
                        done = np.array(done, dtype=int)
                        hidden = new_hidden
                        obs = next_obs
                        next_done = done
                                                      
                        b_observations[pt] = torch.tensor(obs).cpu()
                        b_actions[pt] = torch.tensor(action).cpu()
                        b_rewards[pt] = torch.tensor(reward).cpu()
                        b_dones[pt] = torch.tensor(done).cpu()
                        b_log_probs[pt] = torch.tensor(log_prob).cpu()
                        b_values[pt] = torch.tensor(value).cpu()
                        b_hiddens[pt] = hidden.reshape(configs["num_rnn_layers"], configs["hidden_dim"]).cpu()
                        cur_step += 1
                        pt += 1
                        if done or cur_step == args.max_episode_steps :
                            break
                                    
                with torch.no_grad():
                    trans = (b_observations[pt-1], b_actions[pt-1], b_rewards[pt-1], b_dones[pt-1])
                    next_value = agent.get_value(trans, b_hiddens[pt-1].reshape(1, 1, -1))
                    lastgaelam = 0
                    for t in reversed(range(final_pt, pt-1)):
                        if (t-final_pt) == args.max_episode_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - b_dones[t + 1]
                            nextvalues = b_values[t + 1]
                        delta = b_rewards[t] + configs["gamma"] * nextvalues * nextnonterminal - b_values[t]
                        b_advantages[t] = lastgaelam = delta + configs["gamma"] * configs["gae_lambda"] * nextnonterminal * lastgaelam
                    b_returns[final_pt:pt] = (b_advantages[final_pt:pt] + b_values[final_pt:pt]).clone()
                final_pt = pt

                    
                    # ? calculate return, advantages      
                    # with torch.no_grad():  
                    #     prev_value = 0  
                    #     running_return = 0
                    #     running_advant = 0
                    #     gamma = configs['gamma']
                    #     gae_lambda = configs['gae_lambda']
                    #     for t in reversed(range(final_pt, pt)):
                    #         running_return = rewards[t] + gamma * (1 - dones[t]) * running_return
                    #         returns[t] = running_return
                    #         running_tderror = (
                    #             rewards[t] + gamma * (1 - dones[t]) * prev_value - values[t]
                    #         )
                    #         running_advant = (
                    #             running_tderror + gamma * gae_lambda * (1 - dones[t]) * running_advant
                    #         )
                    #         advantages[t] = running_advant
                    #         prev_value = values[t]
                            
                        
            results["meta_train_return"] = b_rewards.sum() / len(train_tasks)
            meta_train_return = results["meta_train_return"]
            print(f"meta_train_return: {meta_train_return}")
            tb_logger.add("train/meta_train_mean_return", meta_train_return, n_epoch)
        
        else: 
            env.seed(args.seed)
            meta_batch_size = configs["meta_batch_size"]
            print(f"1/{meta_batch_size}] collecting samples")
            hidden = torch.zeros((configs["num_rnn_layers"], 1, configs["hidden_dim"]))
            cur_step = 0
            obs = env.reset()
            done = False
            train_return: float = 0.0
            
            while not (done or cur_step == args.max_episode_steps):
                with torch.no_grad():
                    action, log_prob, entropy, value, new_hidden \
                        = agent.get_action_and_value(obs, hidden)
                if configs['is_continuous']:
                    action = action.numpy()
                next_obs, reward, done, info = env.step(action)
                train_return += reward
                reward = np.array(reward)
                done = np.array(done, dtype=int)
                b_observations[pt] = torch.tensor(obs).cpu()
                b_actions[pt] = torch.tensor(action).cpu()
                b_rewards[pt] = torch.tensor(reward).cpu()
                b_dones[pt] = torch.tensor(done).cpu()
                b_log_probs[pt] = torch.tensor(log_prob).cpu()
                b_values[pt] = torch.tensor(value).cpu()
                b_hiddens[pt] = hidden.reshape(configs["num_rnn_layers"], configs["hidden_dim"]).cpu()
                hidden = new_hidden
                obs = next_obs
                next_done = done
                cur_step += 1
                pt += 1
                if done or cur_step == args.max_episode_steps:
                    break
            
            with torch.no_grad():
                next_value = agent.get_value(next_obs, new_hidden)
                lastgaelam = 0
                for t in reversed(range(final_pt, pt)):
                    if (t-final_pt) == args.max_episode_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - b_dones[t + 1]
                        nextvalues = b_values[t + 1]
                    delta = b_rewards[t] + configs["gamma"] * nextvalues * nextnonterminal - b_values[t]
                    b_advantages[t] = lastgaelam = delta + configs["gamma"] * configs["gae_lambda"] * nextnonterminal * lastgaelam
                b_returns[final_pt:pt] = (b_advantages[final_pt:pt] + b_values[final_pt:pt]).clone()
            final_pt = pt

            results["train_return"] = train_return
            print(f"train_return: {train_return}")
            tb_logger.add("train/train_return", train_return, n_epoch)
        # training with ppo-objective
        num_mini_batch = int(args.batch_size / configs["mini_batch_size"]) 
        sum_total_loss: float = 0
        sum_policy_loss: float = 0
        sum_value_loss: float = 0
        
        if b_observations.shape[0] > configs['mini_batch_size']:
            b_indices = np.arange(0, final_pt)
            for k in range(configs["k_epochs"]):
                sum_total_loss_mini_batch = 0
                sum_policy_loss_mini_batch = 0
                sum_value_loss_mini_batch = 0
                if args.shuffle_mb:
                    np.random.shuffle(b_indices)
                for start in range(0, args.batch_size, configs["mini_batch_size"]):
                    end = start + configs["mini_batch_size"]
                    if end > final_pt:
                        end = final_pt
                    mb_indices = b_indices[start:end]

                    # * mini batches
                    mb_obs = b_observations[mb_indices].to(args.device)
                    mb_acts = b_actions[mb_indices].to(args.device)
                    mb_rews = b_rewards[mb_indices].to(args.device)
                    mb_dones = b_dones[mb_indices].to(args.device)
                    mb_old_log_probs = b_log_probs[mb_indices].to(args.device)
                    mb_values = b_values[mb_indices].to(args.device)
                    mb_returns = b_returns[mb_indices].to(args.device)
                    mb_advants = b_advantages[mb_indices].to(args.device)
                    mb_hiddens = b_hiddens[mb_indices].to(args.device)
                    
                    if configs["norm_adv"]:
                        mb_advants = (mb_advants - mb_advants.mean()) / (mb_advants.std() + 1e-8)
                    #print(mb_advants)
                    if args.meta_learning:
                        mb_trans = (mb_obs, mb_acts, mb_rews, mb_dones)
                        _, mb_new_log_probs, mb_entropy, mb_new_values, mb_new_hiddens \
                            = agent.get_action_and_value(
                            mb_trans,
                            mb_hiddens,
                            mb_acts,
                            is_training=True
                        )
                    else: 
                        _, mb_new_log_probs, mb_entropy, mb_new_values, mb_new_hiddens \
                            = agent.get_action_and_value(
                            mb_obs,
                            mb_hiddens,
                            mb_acts,
                            is_training=True
                        )
                        # transition ,hidden, action=None, is_training=False,
                    
                    if configs['is_continuous']:
                        logratio = mb_new_log_probs.squeeze(0).sum(1) - mb_old_log_probs.sum(1)
                    else:
                        logratio = mb_new_log_probs - mb_old_log_probs
                        
                    logratio = mb_new_log_probs - mb_old_log_probs
                    ratio = logratio.exp()
                    
                    # Policy loss
                    pg_loss1 = -mb_advants * ratio
                    pg_loss2 = -mb_advants * torch.clamp(ratio, 1 - configs["lr_clip_range"], 1 + configs["lr_clip_range"])
                    policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    value_loss = F.smooth_l1_loss(mb_new_values, mb_returns)
                    
                    # mb_new_values = mb_new_values.view(-1)
                    # if configs["clip_vloss"]:
                    #     v_loss_unclipped = (mb_new_values - mb_returns) ** 2
                    #     v_clipped = mb_values + torch.clamp(
                    #         mb_new_values - mb_values,
                    #         -configs["lr_clip_range"],
                    #         configs["lr_clip_range"],
                    #     )
                    #     v_loss_clipped = (v_clipped - mb_returns) ** 2
                    #     v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    #     value_loss = 0.5 * v_loss_max.mean()
                    # else:
                    #     value_loss = 0.5 * ((mb_new_values - mb_returns) ** 2).mean()
                    
                    entropy_loss = mb_entropy.mean()
                    total_loss = policy_loss - configs["ent_coef"] * entropy_loss + configs["vf_coef"] * value_loss 
                    optimizer.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), configs["max_grad_norm"])
                    optimizer.step()
                    
                    sum_total_loss_mini_batch += total_loss
                    sum_policy_loss_mini_batch += policy_loss
                    sum_value_loss_mini_batch += value_loss
                    
                sum_total_loss += sum_total_loss_mini_batch / num_mini_batch
                sum_policy_loss += sum_policy_loss_mini_batch / num_mini_batch
                sum_value_loss += sum_value_loss_mini_batch / num_mini_batch
                
        # logging
        tb_logger.add("train/total_loss", sum_total_loss, n_epoch)
        tb_logger.add("train/policy_loss", sum_policy_loss, n_epoch)
        tb_logger.add("train/value_loss", sum_value_loss, n_epoch)
        tb_logger.add("time/total_time", time.time() - total_start_time, n_epoch)
        tb_logger.add("time/time_per_iter", time.time() - start_time, n_epoch)
     
        # meta-testing
        if args.meta_learning:
            test_return: float = 0.0
            for j, index in enumerate(test_tasks):
                if args.same_task:
                    index = 0
                env.seed(args.seed+2*j)
                env.reset_task(index)
                print(f"[{j + 1}/{len(test_tasks)}] meta evaluating, current task: {env.get_task()}")
                # ? episode rollout per trial
                hidden = torch.zeros((configs["num_rnn_layers"], 1, configs["hidden_dim"]))
                for epi in range(args.num_episodes_per_trial): # 0, 1
                    obs = env.reset()
                    action = torch.zeros(configs["action_dim"])
                    reward = torch.zeros(1)
                    done = torch.zeros(1)
                    cur_step = 0
        
                    while not (done or cur_step == args.max_episode_steps):
                        tran = (obs, action, reward, done)
                        with torch.no_grad():
                            action, log_prob, entropy, value, new_hidden \
                                = agent.get_action_and_value(tran, hidden)
                        if configs['is_continuous']:
                            action = action.numpy()
                        next_obs, reward, done, info = env.step(action)
                        reward = np.array(reward)
                        test_return += reward
                        done = np.array(done, dtype=int)
                        hidden = new_hidden
                        obs = next_obs
                        cur_step += 1
                        if done or cur_step == args.max_episode_steps:
                            break
            
            meta_test_return = test_return / len(test_tasks)
            results["meta_test_return"] = meta_test_return
            print(f"meta_test_return: {meta_test_return}")
            tb_logger.add("test/meta_test_mean_return", meta_test_return, n_epoch)
        
        else: # no meta rl evaluation
            test_return: float = 0.0
            env.seed(args.seed+1)
            print(f"1/{len(test_tasks)}] meta evaluating")
            hidden = np.zeros((configs["num_rnn_layers"], 1, configs["hidden_dim"]))
            obs = env.reset()
            done = False
            cur_step = 0
            while not (done or cur_step == args.max_episode_steps):
                with torch.no_grad():
                    action, log_prob, entropy, value, new_hidden \
                        = agent.get_action_and_value(obs, hidden)
                if configs['is_continuous']:
                    action = action.numpy()
                next_obs, reward, done, info = env.step(action)
                test_return += reward
                done = np.array(done, dtype=int)
                hidden = new_hidden
                obs = next_obs
                cur_step += 1
                if done or cur_step == args.max_episode_steps:
                    break
            
            results["test_return"] = test_return
            print(f"test_return: {test_return}")
            tb_logger.add("test/test_return", test_return, n_epoch)


        # save weight
        if n_epoch % args.save_periods == 0:
            save_file_path = f"{args.env_name}_{args.exp_name}_{args.seed}_{args.now}"
            if not os.path.exists(save_file_path):
                save_file_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.curdir, 'weights',save_file_path))
                try:
                    os.mkdir(save_file_path)
                except:
                    dir_path_head, dir_path_tail = os.path.split(save_file_path)
                    if len(dir_path_tail) == 0:
                        dir_path_head, dir_path_tail = os.path.split(dir_path_head)
                    try:
                        os.mkdir(dir_path_head)
                        os.mkdir(save_file_path)
                    except:
                        pass
            ckpt_path = os.path.join(save_file_path,"checkpoint_" + str(n_epoch) + ".pt")
            print(save_file_path)
            torch.save(agent.state_dict(), ckpt_path)

    
    # meta-train
    