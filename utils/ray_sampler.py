"""This is an implementation of an on policy batch sampler.

Uses a data parallel design.
Included is a sampler that deploys sampler workers.
The sampler workers must implement some type of set agent parameters
function, and a rollout function.

"""
from collections import defaultdict
import itertools
import click
import cloudpickle
import psutil
import ray
from ppo import PPO

@ray.remote()
class RaySampler:
    def __init__(self,         
                 envs,
                agent: PPO,
                args,
                config):
        self.max_episode_steps = args.max_episode_steps
        self.n_workers = args.n_workers
        def rollout(self):
            pass
    