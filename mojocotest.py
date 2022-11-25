import gym
from envs import *
import numpy as np
import torch
env = gym.make("HalfCheetahDirEnv", num_tasks=2)
indices = [0, 1, 0, 1]
for i, index in enumerate(indices): # 0, 1, 2, 3
    env.seed(0)
    done = False
    for epi in range(2): # 0, 1
        cur_step = 0
        obs = env.reset()
        while not (done or cur_step == 500):
            action = env.action_space.sample()
            next_obs, reward, done, info = env.step(action)
            obs = next_obs
            next_done = done
            cur_step += 1
            if done or cur_step == 500:
                print(f"{info['episode']['r']}")
                break