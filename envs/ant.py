import matplotlib.pyplot as plt
import numpy as np
import torch
import random
from envs.mujoco_env import MujocoEnv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AntEnv(MujocoEnv):
    def __init__(self, use_low_gear_ratio=False):
        self.init_serialization(locals())
        if use_low_gear_ratio:
            xml_path = 'low_gear_ratio_ant.xml'
        else:
            xml_path = 'ant.xml'
        super().__init__(
            xml_path,
            frame_skip=5,
            automatically_set_obs_and_action_space=True,
        )

    def step(self, a):
        torso_xyz_before = self.get_body_com("torso")
        self.do_simulation(a, self.frame_skip)
        torso_xyz_after = self.get_body_com("torso")
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = torso_velocity[0] / self.dt
        ctrl_cost = 0.  # .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.  # 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )

    def _get_obs(self):
        # this is gym ant obs, should use rllab?
        # if position is needed, override this in subclasses
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def reset_task(self, task):
        if task is None:
            task = self.sample_tasks(1)[0]
        self.set_task(task)


class AntDirEnv(AntEnv):
    """
    Forward/backward ant direction environment
    """
    def __init__(self, num_tasks=2, max_episode_steps=200):
        directions = [-1.0, 1.0, -1.0, 1.0]
        self.tasks = [{"direction": direction} for direction in directions]
        assert num_tasks == len(self.tasks)
        self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 1
        super(AntDirEnv, self).__init__()

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self.goal_direction), np.sin(self.goal_direction))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2] / self.dt), direct)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        info = dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
            task=self.get_task())
        return ob, reward, done, info

    def get_all_task_idx(self) -> List[int]:
        return list(range(len(self.tasks)))

    def sample_tasks(self, n_tasks):
        # for fwd/bwd env, goal direc is backwards if - 1.0, forwards if + 1.0
        return [random.choice([-1.0, 1.0]) for _ in range(n_tasks, )]

    def set_task(self, task):
        if isinstance(task, np.ndarray):
            task = task[0]
        self.goal_direction = task

    def get_task(self):
        return np.array([self.goal_direction])


class AntDir2DEnv(AntDirEnv):
    def __init__(num_tasks=2, max_episode_steps=200):
        super().__init__()
    def sample_tasks(self, n_tasks):
        # for fwd/bwd env, goal direc is backwards if - 1.0, forwards if + 1.0
        directions = np.array([random.gauss(mu=0, sigma=1) for _ in range(n_tasks * 2)]).reshape((n_tasks, 2))
        directions /= np.linalg.norm(directions, axis=1)[..., np.newaxis]
        return directions


class AntDirOracleEnv(AntDirEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            [self.goal_direction],
        ])


class AntDir2DOracleEnv(AntDir2DEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            [self.goal_direction],
        ])



class AntGoalEnv(AntEnv):
    def __init__(self, max_episode_steps=200):
        self.set_task(self.sample_tasks(1)[0])
        self._max_episode_steps = max_episode_steps
        self.task_dim = 2
        super(AntGoalEnv, self).__init__()

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self.goal_pos))  # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            task=self.get_task()
        )

    def sample_tasks(self, num_tasks):
        a = np.array([random.random() for _ in range(num_tasks)]) * 2 * np.pi
        r = 3 * np.array([random.random() for _ in range(num_tasks)]) ** 0.5
        return np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)

    def set_task(self, task):
        self.goal_pos = task

    def get_task(self):
        return np.array(self.goal_pos)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])


class AntGoalOracleEnv(AntGoalEnv):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            self.goal_pos,
        ])


