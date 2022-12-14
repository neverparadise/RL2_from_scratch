import gym
import torch
import numpy as np
from dataclasses import asdict, dataclass
import torch as th
from torch.nn.utils.rnn import pad_sequence
from typing import Any, Generator, Optional, Tuple, Callable, Dict, List, NamedTuple, Union
from collections import namedtuple


def shuffle(x: torch.Tensor, device):
    indices = torch.randperm(x.size(0)).to(device)
    x_ = torch.index_select(x, dim=0, index=indices)
    return x_

def to_tensor(np_array, device):
    if isinstance(np_array, np.ndarray):
        return torch.from_numpy(np_array).float().to(device)
    return np_array.float().to(device)


def to_numpy(data):
    if isinstance(data, tuple):
        return tuple(to_numpy(x) for x in data)
    if isinstance(data, torch.autograd.Variable):
        return data.to('cpu').detach().numpy()
    return data

# RecurrentRolloutBufferSamples(*tuple([to_tensor(sample, device=self.device) for sample in samples]))

@dataclass
class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    values: th.Tensor
    returns: th.Tensor
    log_probs: th.Tensor
    advantages: th.Tensor
    
@dataclass
class RecurrentRolloutBufferSamples:
    observations: th.Tensor
    actions: th.Tensor
    rewards: th.Tensor
    dones:  th.Tensor
    oldvalues: th.Tensor
    returns: th.Tensor
    old_log_probs: th.Tensor
    advantages: th.Tensor
    pi_hiddens: th.Tensor
    v_hiddens: th.Tensor
    
    def get_transition(self):
        return self.observations, self.actions, self.rewards, self.dones, self.oldvalues, self.returns, \
            self.old_log_probs, self.advantages,  self.pi_hiddens, self.v_hiddens

            
class RolloutBuffer:
    def __init__(
        self,
        args,
        configs,
        ):
        
        self.device = args.device
        if args.meta_learning:
            self.buffer_size = args.batch_size
        else:
            self.buffer_size = args.rollout_steps
        self.state_dim = configs["state_dim"]
        self.action_dim = configs["action_dim"]
        self.is_continuous = configs["is_continuous"]
        self.gamma = configs["gamma"]
        self.gae_lambda = configs["gae_lambda"]
        self.is_recurrent = configs["is_recurrent"]
        if self.is_recurrent:
            self.hidden_dim = configs["hidden_dim"]
            self.num_rnn_layers = configs["num_rnn_layers"]
        
        self.reset()
    
    def reset(self):
        """
            Resets the buffer.
        """
        self.observations = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.values = np.zeros((self.buffer_size,1), dtype=np.float32)
        self.rewards = np.zeros((self.buffer_size,1), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size,1), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size,1), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size,1), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size,1), dtype=np.float32)
        self.real_size = 0
        if self.is_recurrent:
            self.pi_hiddens = np.zeros((self.buffer_size, self.num_rnn_layers, self.hidden_dim), dtype=np.float32)
            self.v_hiddens = np.zeros((self.buffer_size, self.num_rnn_layers, self.hidden_dim), dtype=np.float32)
            
        # List for tracking the trajectory indices because we don't know
        # how many we can collect
        self.trajectory_index = np.array([0])
        self.pt = 0
    
    @property
    def size(self):
        return self.real_size
    
    def add_sample(self, sample: Tuple[Any]) -> bool:
        """
            Adds a sample to the buffer and increments pointer.
            :param sample: Environment interaction for a single timestep;
                stores state value and log prob of action.
            :return: Whether buffer full or not
        """

        if self.pt != self.buffer_size:
            sample = tuple([to_numpy(item) for item in sample])
            if not self.is_recurrent:
                observation, action, value, reward, log_prob, done = sample
            else:
                observation, action, reward, done, pi_hidden, v_hidden, value, log_prob, = sample
            
            # 1. Observation
            self.observations[self.pt] = observation
            
            # 2. Action
            if self.is_continuous: # Continuous
                self.actions[self.pt] = action
            else: # Discrete
                self.actions[self.pt][action] = 1 # if discrete action, action is index
            
            # Value, Reward, Log prob, Done
            self.values[self.pt] = value
            self.rewards[self.pt] = reward
            self.log_probs[self.pt] = log_prob
            self.dones[self.pt] = done

            if self.is_recurrent:
                # hidden_shape = (num_rnn_layers, batch_size, hidden_size)
                # we need to squeeze batch dim
                self.pi_hiddens[self.pt] = pi_hidden
                self.v_hiddens[self.pt] = v_hidden
            self.pt += 1
        self.real_size = min(self.buffer_size, self.real_size + 1)
        return self.pt == self.buffer_size
    
    def add_trajs(self, trajs: List[Dict[str, np.ndarray]]) -> None:
        # ????????? ????????? ??????
        for traj in trajs:
            for (obs, action, reward, done, pi_hidden, v_hidden, value, log_prob) in zip(
                traj["observations"],
                traj["actions"],
                traj["rewards"],
                traj["dones"],
                traj["pi_hiddens"],
                traj["v_hiddens"],
                traj["values"],
                traj["log_probs"],
            ):
                sample = (obs, action, reward, done, pi_hidden, v_hidden, value, log_prob)
                self.add_sample(sample)
            #self.end_trajectory(traj["values"][-1], traj["dones"][-1])
                

    def end_trajectory(self, last_value: np.ndarray, last_done: int):
        """
            Ends trajectory calculating GAE https://arxiv.org/abs/1506.02438 and 
            the lambda-return (TD(lambda) estimate).
            The TD(lambda) estimator has also two special cases:
            - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
            - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))
            :param last_value: Last state value for observation resulting from final steps
            :param last_done: Whether final state was done or not.
        """
        # First create trajectory_advantage array to hold
        # trajectory length entries
        self.trajectory_index = np.concatenate((self.trajectory_index, np.array([self.pt]))) # add last pointer
        traj_range = np.arange(self.trajectory_index[-2], self.trajectory_index[-1])
        last_advantage = 0
        for step in reversed(traj_range):
            if step == traj_range[-1]:
                next_value = last_value
                next_non_done = 1.0 - last_done
            else: # step == traj_range[-2]
                next_value = self.values[step + 1]
                next_non_done = self.dones[step + 1]
            delta = self.rewards[step] + self.gamma * next_value * next_non_done - self.values[step]
            last_advantage = delta + self.gamma * self.gae_lambda * next_non_done * last_advantage
            self.advantages[step] = last_advantage
        self.advantages[traj_range] = (self.advantages[traj_range] - self.advantages[traj_range].mean()) / self.advantages[traj_range].std()
        # https://github.com/DLR-RM/stable-baselines3/blob/6f822b9ed7d6e8f57e5a58059923a5b24e8db283/stable_baselines3/common/buffers.py#L347-L348
        self.returns[traj_range] = self.advantages[traj_range] + self.values[traj_range]

    def compute_gae(self) -> None:
        prev_value = 0
        running_return = 0
        running_advant = 0

        for t in reversed(range(len(self.rewards))):
            running_return = self.rewards[t] + self.gamma * (1 - self.dones[t]) * running_return
            self.returns[t] = running_return
            running_tderror = (
                self.rewards[t] + self.gamma * (1 - self.dones[t]) * prev_value - self.values[t]
            )
            running_advant = (
                running_tderror + self.gamma * self.gae_lambda * (1 - self.dones[t]) * running_advant
            )
            self.advantages[t] = running_advant
            prev_value = self.values[t]
        
        # self.advantages = (self.advantages - self.advantages.mean()) / self.advantages.std()
        
    def sample_batch(self) -> RecurrentRolloutBufferSamples:
        assert self.pt == self.buffer_size
        self.pt = 0
        self.compute_gae()
        samples = dict(
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            dones=self.dones,
            returns=self.returns,
            oldvalues=self.values,
            old_log_probs=self.log_probs,
            advantages=self.advantages,
        )
        samples.update({"pi_hiddens":self.pi_hiddens,
                        "v_hiddens": self.v_hiddens
                        })
        #samples = {k: shuffle(to_tensor(v, device="cpu"), device="cpu") for k, v in samples.items()}
        samples = {k: to_tensor(v, device="cpu") for k, v in samples.items()}
        
        return RecurrentRolloutBufferSamples(*tuple(samples.values()))
    
    def sample_mini_batch(self, batch_size):
        indices = np.random.choice(self.pt, batch_size, replace=False)
        return self.get_samples_recurrent(indices)

