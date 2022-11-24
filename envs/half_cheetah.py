from typing import List, Union, Any, Dict, List, Tuple
import random
import numpy as np
from gym import utils
from gym.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_
from gym.envs.mujoco import mujoco_env


class HalfCheetahEnv(HalfCheetahEnv_):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, "half_cheetah.xml", 5)
        utils.EzPickle.__init__(self)

    def _get_obs(self) -> np.ndarray:
        return (
            np.concatenate(
                [
                    self.data.qpos.flat[1:],
                    self.data.qvel.flat,
                    self.get_body_com("torso").flat,
                ],
            )
            .astype(np.float32)
            .flatten()
        )

    def viewer_setup(self) -> None:
        camera_id = self.model.camera_name2id("track")
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        self.viewer._hide_overlay = True

    def render(self, mode: str = "human") -> Union[List[float], None]:
        if mode == "rgb_array":
            self._get_viewer(mode).render()
            width, height = 500, 500
            data = self._get_viewer().read_pixels(width, height, depth=False)
            return data
        elif mode == "human":
            self._get_viewer(mode).render()


class HalfCheetahDirEnv(HalfCheetahEnv):
    def __init__(self, num_tasks=2) -> None:
        if num_tasks==1:
            directions = [1]
        elif num_tasks==2: 
            directions = [-1, 1]
        else:
            raise "num tasks > 2"
            
        self.tasks = [{"direction": direction} for direction in directions]
        #assert num_tasks == len(self.tasks)
        self._task =  self.tasks[0]
        self._goal_dir = self._task["direction"]
        super().__init__()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, Dict[str, Any]]:
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]

        progress = (xposafter - xposbefore) / self.dt
        run_cost = self._goal_dir * progress
        control_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = run_cost - control_cost
        done = False
        info = dict(run_cost=run_cost, control_cost=-control_cost, task=self._task)
        return observation, reward, done, info

    def get_all_task_idx(self) -> List[int]:
        return list(range(len(self.tasks)))

    def reset_task(self, idx: int) -> None:
        self._task = self.tasks[idx]
        self.set_task(self._task["direction"])
        self.reset()

    def set_task(self, task):
        self._goal_dir = task

    def get_task(self):
        return np.array(self._goal_dir)
    
class HalfCheetahVelEnv(HalfCheetahEnv):
    def __init__(self, num_tasks: int) -> None:
        self.tasks = self.sample_tasks(num_tasks)
        self._task = self.tasks[0]
        self._goal_vel = self._task["velocity"]
        super().__init__()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool, Dict[str, Any]]:
        xposbefore = self.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.data.qpos[0]

        progress = (xposafter - xposbefore) / self.dt
        run_cost = progress - self._goal_vel
        scaled_run_cost = -1.0 * abs(run_cost)
        control_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = scaled_run_cost - control_cost
        done = False
        info = dict(run_cost=run_cost, control_cost=-control_cost, task=self._task)
        return observation, reward, done, info

    def sample_tasks(self, num_tasks: int):
        np.random.seed(0)
        velocities = np.random.uniform(0.0, 2.0, size=(num_tasks,))
        tasks = [{"velocity": velocity} for velocity in velocities]
        return tasks

    def get_all_task_idx(self) -> List[int]:
        return list(range(len(self.tasks)))

    def reset_task(self, idx: int) -> None:
        self._task = self.tasks[idx]
        self._goal_vel = self._task["velocity"]
        self.reset()
