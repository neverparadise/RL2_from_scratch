import os
import time
from typing import Any, Dict, List
from distutils.util import strtobool
import argparse
import yaml
import datetime

import gym
from gym.envs.registration import register
from gym.spaces import Box, Discrete
import pygame
from envs import *

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from ppo import PPO
from networks import RNNActor, RNNCritic
from buffers.buffer import RolloutBuffer
from torch.utils.tensorboard import SummaryWriter
from utils.tb_logger import TBLogger
from utils.sampler import BaseSampler, RL2Sampler

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser()

    # ? Experiments information
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--meta_learning", type=bool, default=False)
    parser.add_argument("--load", type=bool, default=False)
    parser.add_argument("--load_ckpt_num", type=int, default=0)
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)),
                        default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--render", type=bool, default=False)
    parser.add_argument("--render_mode", type=str, default="human")
    parser.add_argument("--weight_path", type=str, default="./weights",
                        help="weight path for saving model")
    parser.add_argument("--save_periods", type=int, default=20)
    parser.add_argument("--results_log_dir", type=str, default="./logs",
                        help="directory of tensorboard")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    
    #  Environments information
    parser.add_argument("--is_atari", type=bool, default=False)
    # parser.add_argument("--env_name", type=str, default="CartPole-v0",
            # help="the id of the environment")
    #parser.add_argument("--env_name", type=str, default="LunarLanderContinuous-v2",
    #       help="the id of the environment")
    parser.add_argument("--env_name", type=str, default="HalfCheetah-v3")
    # parser.add_argument("--env_name", type=str, default="Ant")
    parser.add_argument("--total-timesteps", type=int, default=1000000,
                        help="total timesteps of the experiments")
    parser.add_argument('--rollout_steps', default=256)
    parser.add_argument('--max_episode_steps', default=1000)
    parser.add_argument("--num-envs", type=int, default=4,
                        help="the number of parallel game environments")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")

    # Hyperparameter configs
    parser.add_argument('--config_path', type=str,
                        default='./configs/base_config.yaml')
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.rollout_steps)
    args.now = datetime.datetime.now().strftime('_%m.%d_%H:%M:%S')
    return args


def make_env(args, configs, seed, idx, run_name):
    env_id = args.env_name
    capture_video = args.capture_video
    is_atari = args.is_atari
    if is_atari:
        def thunk():
            env = gym.make(env_id)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            return env
    else:
        def thunk():
            env = gym.make(env_id)
            env.seed(seed)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            # env = gym.wrappers.ClipAction(env)
            # env = gym.wrappers.NormalizeObservation(env)
            # env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            # env = gym.wrappers.NormalizeReward(configs["gamma"])
            # env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
            return env
    return thunk


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


def _format(x, device, num_envs=4, minibatch_size=1, is_training=False):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.to(device=device)
    else:
        x = x.to(device=device)
    if len(x.shape) < 3:
        x = x.reshape(1, num_envs, -1) # [L, N, flatten]
    if is_training:
        x = x.reshape(1, minibatch_size, -1)
    return x

# hidden = (seq_len, batch_size, hidden_size)
# hidden = (1, num_envs*mb_size, hidden)


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class RNNAgent(nn.Module):
    def __init__(self, args, configs) -> None:
        super().__init__()
        self.num_envs = args.num_envs
        self.device = torch.device(args.device)
        self.input_dim = configs["state_dim"]
        self.linear_dim = configs["linear_dim"]
        self.minibatch_size = configs["minibatch_size"]
        self.is_continuous = configs["is_continuous"]
        self.is_deterministic = configs["is_dterministic"]
        self.hidden_dim = configs["hidden_dim"]
        self.num_rnn_layers = configs["num_rnn_layers"]
        self.action_dim = configs["action_dim"]
        self.num_discretes = configs["num_discretes"]

        self.embedding = nn.Sequential(
            layer_init(nn.Linear(self.input_dim, self.linear_dim)),
            nn.LeakyReLU(),
            layer_init(nn.Linear(self.linear_dim, self.linear_dim)),
            nn.LeakyReLU(),
        )
        self.gru = nn.GRU(self.linear_dim, self.hidden_dim, \
                            num_layers=self.num_rnn_layers, bias=True)
        if self.is_continuous:
            self.mean = layer_init(nn.Linear(self.hidden_dim, self.action_dim))
            self.actor_logstd = nn.Parameter(torch.zeros(1, self.action_dim))
            #self.actor_logstd =  layer_init(nn.Linear(self.hidden_dim, self.action_dim))
        else:
            self.policy_logits = layer_init(nn.Linear(self.hidden_dim, self.num_discretes))
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.hidden_dim, self.linear_dim)),
            layer_init(nn.Linear(self.linear_dim, 1))
        )

    def forward(self, state, hidden=None, is_training=False):
        # state = (num_envs, state_dim)
        state = _format(state, self.device, self.num_envs, self.minibatch_size, is_training)
        if is_training:
            hidden = hidden.permute(1, 0, 2).contiguous()
        x = self.embedding(state)
        hidden = to_tensor(hidden, device=self.device)
        x, new_hidden = self.gru(x, hidden)
        if self.is_continuous:
            mu = torch.tanh(self.mean(x))
            std = torch.exp(self.actor_logstd)
            # std = F.softplus(self.std(x))
            dist = Normal(mu, std)
        else:
            logits = self.policy_logits(x)
            prob = F.softmax(logits, dim=-1)
            dist = Categorical(prob)
        return x, dist, new_hidden

    def get_value(self, x, hidden, is_training=False):
        x, dist, new_hidden = self.forward(x, hidden, is_training=is_training)
        return self.critic(x)
    
    def get_action_and_value(self, x ,hidden, is_training=False, action=None):
        x, dist, new_hidden = self.forward(x, hidden, is_training=is_training)
        value = self.critic(x)
        if is_training:
            return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), value, new_hidden
        else: # evaluation
            if self.is_deterministic:
                action = dist.mean
                log_prob = torch.zeros(1)
            else: # stochastic
                action = dist.sample()
                log_prob = dist.log_prob(action)
                log_prob = log_prob.sum(-1)
            
            action = action.reshape(self.num_envs, -1)
            log_prob = log_prob.reshape(self.num_envs,)
            if self.is_continuous:
                action = action.detach()
            else:
                # ! 벡터환경이라 item 없애야할듯
                action = action.detach() # .item() ? 
            
            log_prob = log_prob.detach()
            entropy = dist.entropy().sum(-1)
            value = value.detach()
            return action, log_prob, entropy, value, new_hidden
                                

if __name__ == "__main__":
    # argparser, configs, logger
    args = parse_args()
    with open(args.config_path) as file:
        configs: Dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)
    tb_logger = TBLogger(args, configs)
    now = datetime.datetime.now().strftime('_%m.%d_%H:%M:%S')
    run_name = f"{args.env_name}__{args.exp_name}__{args.seed}__{now}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # env, seed
    env = gym.make(args.env_name)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device(
        args.device if torch.cuda.is_available() and args.cuda else "cpu")
    configs = add_state_action_info(env, configs)

    # agent
    # agent = PPO(args, configs)

    # buffer, sampler
    # buffer = RolloutBuffer(args, configs)

    # # execution
    # sampler = BaseSampler(env, agent, args, configs)
    # for i in range(10):
    #     trajs = sampler.obtain_samples()
    #     buffer.add_trajs(trajs)

    # train
    envs = gym.vector.SyncVectorEnv(
        [make_env(args, configs, args.seed + i, i, run_name) for i in range(args.num_envs)]
    )
    agent = RNNAgent(args, configs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=float(configs["lr"]), eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.rollout_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.rollout_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.rollout_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.rollout_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.rollout_steps, args.num_envs)).to(device)
    values = torch.zeros((args.rollout_steps, args.num_envs)).to(device)
    hiddens = torch.zeros((args.rollout_steps, configs["num_rnn_layers"], args.num_envs, configs["hidden_dim"])).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        hidden = hiddens[0].clone().to(device) # 2, 4, 18
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * float(configs["lr"])
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.rollout_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            hiddens[step] = hidden

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, entropy, value, hidden = agent.get_action_and_value(next_obs, hidden)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    tb_logger.add("charts/episodic_return", item["episode"]["r"], global_step)
                    tb_logger.add("charts/episodic_length", item["episode"]["l"], global_step)
                    break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs, hidden).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.rollout_steps)):
                if t == args.rollout_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + configs["gamma"] * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + configs["gamma"] * configs["gae_lambda"] * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_hiddens = hiddens.reshape((-1, configs["num_rnn_layers"], configs["hidden_dim"])).contiguous()

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(configs["k_epochs"]):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, configs["minibatch_size"]):
                end = start + configs["minibatch_size"]
                mb_inds = b_inds[start:end]
                mb_hiddens = b_hiddens[mb_inds]
                _, newlogprob, entropy, newvalue, newhiddens = agent.get_action_and_value(b_obs[mb_inds], hidden=mb_hiddens,\
                                                                    is_training=True, action=b_actions[mb_inds])
                mb_old_probs = b_logprobs[mb_inds]
                logratio = newlogprob - mb_old_probs
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > configs["lr_clip_range"]).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if configs["norm_adv"]:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - configs["lr_clip_range"], 1 + configs["lr_clip_range"])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if configs["clip_vloss"]:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -configs["lr_clip_range"],
                        configs["lr_clip_range"],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - float(configs["ent_coef"]) * entropy_loss + v_loss * float(configs["vf_coef"])

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), configs["max_grad_norm"])
                optimizer.step()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        tb_logger.add("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        tb_logger.add("losses/value_loss", v_loss.item(), global_step)
        tb_logger.add("losses/policy_loss", pg_loss.item(), global_step)
        tb_logger.add("losses/entropy", entropy_loss.item(), global_step)
        tb_logger.add("losses/old_approx_kl", old_approx_kl.item(), global_step)
        tb_logger.add("losses/approx_kl", approx_kl.item(), global_step)
        tb_logger.add("losses/clipfrac", np.mean(clipfracs), global_step)
        tb_logger.add("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        tb_logger.add("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()

