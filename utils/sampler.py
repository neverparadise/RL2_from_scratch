import random
import os
import time
import numpy as np
import torch
from typing import Dict, List, NamedTuple
from collections import namedtuple

import numpy as np

from ppo import PPO


class BaseSampler:
    def __init__(
        self,
        env,
        agent: PPO,
        args,
        configs
    ) -> None:

        self.env = env
        self.seed = args.seed
        self.max_step = args.max_episode_steps
        self.max_samples = args.batch_size
        self.render = args.render
        self.render_mode = args.render_mode
        self.agent = agent
        self.state_dim = configs["state_dim"]
        self.action_dim = configs["action_dim"]
        self.num_rnn_layers = configs["num_rnn_layers"]
        self.hidden_dim = configs["hidden_dim"]
        self.cur_samples = 0
        self.pi_hidden = None
        self.v_hidden = None

    def obtain_samples(self) -> List[Dict[str, np.ndarray]]:
        self.pi_hidden = np.zeros((self.num_rnn_layers, 1, self.hidden_dim))
        self.v_hidden = np.zeros((self.num_rnn_layers, 1, self.hidden_dim))

        trajs = []
        # ! 이 부분 ray로 병렬처리 가능할듯?
        while not self.cur_samples == self.max_samples:
            traj = self.rollout(self.max_samples)
            trajs.append(traj)

        self.cur_samples = 0
        return trajs

    def rollout(self, max_samples: int) -> Dict[str, np.ndarray]:
        observations = np.zeros((max_samples, self.state_dim), dtype=np.float32)
        actions = np.zeros((max_samples, self.action_dim), dtype=np.float32)
        rewards = np.zeros((max_samples,), dtype=np.float32)
        dones = np.zeros((max_samples,), dtype=np.int)
        pi_hiddens = np.zeros((max_samples, self.num_rnn_layers, self.hidden_dim))
        v_hiddens = np.zeros((max_samples, self.num_rnn_layers, self.hidden_dim))
        values = np.zeros((max_samples,), dtype=np.float32)
        log_probs = np.zeros((max_samples,), dtype=np.float32)
        infos = np.zeros((max_samples, ))

        cur_step = 0
        self.env.seed(seed=self.seed)
        obs = self.env.reset()
        done = False
        info = None
        while not (done or cur_step == self.max_step or self.cur_samples == max_samples):
            with torch.no_grad():
                action, log_prob, entropy, next_pi_hidden = self.agent.get_action(obs, self.pi_hidden)
            value, next_v_hidden = self.agent.get_value(obs, self.v_hidden)
            next_obs, reward, done, info = self.env.step(action)
            reward = np.array(reward).reshape(-1)
            done = np.array(done, dtype=int).reshape(-1)
            
            observations[cur_step] = obs.reshape(-1)
            actions[cur_step] = action
            rewards[cur_step] = reward
            dones[cur_step] = done
            values[cur_step] = value
            log_probs[cur_step] = log_prob
            pi_hiddens[cur_step] = self.pi_hidden.reshape(self.num_rnn_layers, -1)
            v_hiddens[cur_step] = self.v_hidden.reshape(self.num_rnn_layers, -1)
            # observations = np.concatenate((observations, obs.reshape(-1)))
            # actions = np.concatenate((actions, action.reshape(-1)))
            # rewards = np.concatenate((rewards, reward.reshape(-1)))
            # dones = np.concatenate((dones, done.reshape(-1)))
            # values = np.concatenate((values, value.reshape(-1)))
            # log_probs = np.concatenate((log_probs, log_prob.reshape(-1)))
            # hiddens = np.concatenate((hiddens, next_pi_hidden.reshape(self.num_rnn_layers , 1, -1)))
            # pi_hiddens = np.concatenate((pi_hiddens, self.pi_hidden.reshape(self.num_rnn_layers , 1, -1)))
            # v_hiddens = np.concatenate((v_hiddens, self.v_hidden.reshape(self.num_rnn_layers , 1, -1)))
            self.pi_hidden = next_pi_hidden
            self.v_hidden = next_v_hidden
            
            obs = next_obs.reshape(-1)
            cur_step += 1
            self.cur_samples += 1
            if self.render and (self.render_mode is not None):
               print(1)
               print(f"env: {self.env_name}, is_render: {self.render}")
               self.env.render(mode=self.render_mode)
            elif not self.render and (self.render_mode is None):
               pass
            elif self.render and (self.render_mode is None):
               print(2)
               self.env.render()


        return dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            dones=np.array(dones),
            pi_hiddens=np.array(pi_hiddens),
            v_hiddens=np.array(v_hiddens),
            values=np.array(values),
            log_probs=np.array(log_probs),
            infos=np.array(infos),
        )

class RL2Sampler(BaseSampler):
    def __init__(
        self,
        env,
        agent: PPO,
        args,
        configs
    ) -> None:
        super().__init__(env, agent, args, configs)
        self.num_tasks = configs["num_tasks"]

    def obtain_samples(self) -> List[Dict[str, np.ndarray]]:
        self.pi_hidden = np.zeros((self.num_rnn_layers, 1, self.hidden_dim))
        self.v_hidden = np.zeros((self.num_rnn_layers, 1, self.hidden_dim))

        trajs = []
        # ! 이 부분 ray로 병렬처리 가능할듯?
        while not self.cur_samples == self.max_samples:
            traj = self.rollout(self.max_samples)
            trajs.append(traj)

        self.cur_samples = 0
        return trajs

    def rollout(self, max_samples: int) -> Dict[str, np.ndarray]:
        observations = np.zeros((max_samples, self.state_dim), dtype=np.float32)
        actions = np.zeros((max_samples, self.action_dim), dtype=np.float32)
        rewards = np.zeros((max_samples,), dtype=np.float32)
        dones = np.zeros((max_samples,), dtype=np.int)
        pi_hiddens = np.zeros((max_samples, self.num_rnn_layers, self.hidden_dim))
        v_hiddens = np.zeros((max_samples, self.num_rnn_layers, self.hidden_dim))
        values = np.zeros((max_samples,), dtype=np.float32)
        log_probs = np.zeros((max_samples,), dtype=np.float32)
        infos = np.zeros((max_samples, ))

        cur_step = 0
        self.env.seed(seed=self.seed)

        # initialize transition
        obs = self.env.reset()
        action = np.zeros(self.action_dim)
        reward = np.zeros(1)
        done = np.zeros(1)

        while not (done or cur_step == self.max_step or self.cur_samples == max_samples):
            if self.render and (self.render_mode is not None):
               self.env.render(mode=self.render_mode)
            elif self.render and (self.render_mode is None):
               self.env.render()
            elif not self.render and (self.render_mode is None):
               pass
            
            # make transition tuple
            tran = (obs, action, reward, done)
            with torch.no_grad():
                action, log_prob, entropy, next_pi_hidden = self.agent.get_action(tran, self.pi_hidden)
            value, next_v_hidden = self.agent.get_value(tran, self.v_hidden)
            next_obs, reward, done, info = self.env.step(action)
            reward = np.array(reward).reshape(-1)
            done = np.array(done, dtype=int).reshape(-1)
            
            observations[cur_step] = obs.reshape(-1)
            actions[cur_step] = action
            rewards[cur_step] = reward
            dones[cur_step] = done
            values[cur_step] = value
            log_probs[cur_step] = log_prob
            pi_hiddens[cur_step] = self.pi_hidden.reshape(self.num_rnn_layers, -1)
            v_hiddens[cur_step] = self.v_hidden.reshape(self.num_rnn_layers, -1)
            self.pi_hidden = next_pi_hidden
            self.v_hidden = next_v_hidden
            
            obs = next_obs.reshape(-1)
            cur_step += 1
            self.cur_samples += 1

        return dict(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            dones=np.array(dones),
            pi_hiddens=np.array(pi_hiddens),
            v_hiddens=np.array(v_hiddens),
            values=np.array(values),
            log_probs=np.array(log_probs),
            infos=np.array(infos),
        )
