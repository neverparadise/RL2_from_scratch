import gym

from envs.half_cheetah import HalfCheetahDirEnv, HalfCheetahVelEnv
from envs.ant import AntEnv, AntDirEnv, AntDir2DEnv, AntGoalEnv
from envs.point_env import SparsePointEnv, PointEnv

__all__ = [
    'HalfCheetahDirEnv',
    'HalfCheetahVelEnv',
    "SparsePointEnv",
    "PointEnv",
    "AntDirEnv",
    "AntDir2DEnv",
    "AntGoalEnv"
]
gym.envs.register(
     id='HalfCheetahDirEnv',
     entry_point='envs.half_cheetah:HalfCheetahDirEnv',
     max_episode_steps=200,
)
gym.envs.register(
     id='HalfCheetahVelEnv',
     entry_point='envs.half_cheetah:HalfCheetahVelEnv',
     max_episode_steps=200,
)
gym.envs.register(
     id='AntEnv',
     entry_point='envs.ant:AntEnv',
     max_episode_steps=200,
)
gym.envs.register(
     id='AntDirEnv',
     entry_point='envs.ant:AntDirEnv',
     max_episode_steps=200,
)
gym.envs.register(
     id='AntDir2DEnv',
     entry_point='envs.ant:AntDir2DEnv',
     max_episode_steps=200,
)
gym.envs.register(
     id='AntGoalEnv',
     entry_point='envs.ant:AntGoalEnv',
     max_episode_steps=200,
)
gym.envs.register(
     id='PointEnv',
     entry_point='envs.point_env:PointEnv',
     max_episode_steps=200,
)
gym.envs.register(
     id='SparsePointEnv',
     entry_point='envs.point_env:SparsePointEnv',
     max_episode_steps=200,
)
