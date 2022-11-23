import ray
import numpy as np
import torch
import gym
from gym.envs.registration import register
from typing import Dict, List, NamedTuple
import psutil
import yaml
from typing import Any, Dict, List
import os
import math

pardir = os.pardir
cwd = os.getcwd()
print(cwd)
with open(f"{cwd}/configs/cheetah_dir_config.yaml") as file:
    config: Dict[str, Any] = yaml.load(file, Loader=yaml.FullLoader)

num_cpus = psutil.cpu_count()
num_gpus = torch.cuda.device_count()
meta_batch_size = config["meta_batch_size"]
num_test_tasks = config["num_test_tasks"]
worker_per_cpus = int(num_cpus / (meta_batch_size + num_test_tasks)) -1
worker_per_gpus = float(num_gpus / (meta_batch_size+ num_test_tasks))
if worker_per_gpus > 1:
    worker_per_gpus = int(worker_per_gpus)
print(psutil.cpu_count())
print(num_gpus)
print(worker_per_cpus)
print(worker_per_gpus)
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
fileHandler = logging.FileHandler('./log.txt')
streamHandler = logging.StreamHandler()
logger.addHandler(fileHandler)
logger.addHandler(streamHandler)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@ray.remote(num_cpus=worker_per_cpus, num_gpus=worker_per_gpus)
class RaySampler:
    # trial (MDP)을 parallel processing
    def __init__(self, env, agent, args, config, worker_idx):
        #self.env = gym.make("HalfCheetahDirEnv", num_tasks=2)
        #self.tasks: List[int] = self.env.get_all_task_idx()

        self.env = env
        self.seed = args.seed
        self.max_steps = args.max_episode_steps # episode max length
        self.num_episodes = args.num_episodes_per_trial  # num episodes in each trial
        self.max_samples = args.max_samples # num max transitions in buffer
        self.render = args.render
        self.render_mode = args.render_mode
        self.agent = agent
        self.state_dim = config["state_dim"]
        self.action_dim = config["action_dim"]
        self.num_rnn_layers = config["num_rnn_layers"]
        self.hidden_dim = config["hidden_dim"]
        self.num_tasks = config["num_tasks"]
        self.cur_samples = 0
        self.pi_hidden = None
        self.v_hidden = None
        self.worker_idx = worker_idx
    
    def print_worker_infos(self):
        logger.info(worker_per_cpus)
        logger.info(worker_per_gpus)
        logger.info(torch.cuda.is_available)
        logger.info("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        logger.info(f"current device: {torch.cuda.current_device()}")
        logger.info(f"count: {torch.cuda.device_count()}")
        
    def obtain_samples(self, task_idx) -> List[Dict[str, np.ndarray]]:
        print(f"start to collect trajectories")
        self.cur_samples = 0
        self.env.reset_task(task_idx)
        self.agent.policy.is_deterministic = False
        print(f"Worker: {self.worker_idx + 1} collecting samples, current task: {self.env.get_task()}")
        self.pi_hidden = np.zeros((self.num_rnn_layers, 1, self.hidden_dim))
        self.v_hidden = np.zeros((self.num_rnn_layers, 1, self.hidden_dim))
        trajs = []
        while not self.cur_samples == self.max_samples:
            traj = self.rollout()
            trajs.append(traj)
        print(f"Worker: {self.worker_idx + 1} finish collecting samples , current_samples: {self.cur_samples}")
        return trajs
    
    def rollout(self) -> Dict[str, np.ndarray]:
        observations = np.zeros((self.max_steps, self.state_dim), dtype=np.float32)
        actions = np.zeros((self.max_steps, self.action_dim), dtype=np.float32)
        rewards = np.zeros((self.max_steps,), dtype=np.float32)
        dones = np.zeros((self.max_steps,), dtype=np.int)
        pi_hiddens = np.zeros((self.max_steps, self.num_rnn_layers, self.hidden_dim))
        v_hiddens = np.zeros((self.max_steps, self.num_rnn_layers, self.hidden_dim))
        values = np.zeros((self.max_steps,), dtype=np.float32)
        log_probs = np.zeros((self.max_steps,), dtype=np.float32)
        infos = np.zeros((self.max_steps, ))
        
        cur_step = 0
        self.env.seed(seed=self.seed)
        obs = self.env.reset()
        action = np.zeros(self.action_dim)
        reward = np.zeros(1)
        done = np.zeros(1)
        info = None
        
        while not (done or cur_step == self.max_steps):
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

