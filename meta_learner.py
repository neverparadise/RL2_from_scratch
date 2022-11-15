import datetime
import os
import time
import warnings
from collections import deque
from typing import Any, Dict, List
import ray
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from envs import *
from buffers.buffer import RolloutBuffer
from ppo import PPO
from utils.sampler import BaseSampler, RL2Sampler
from utils.ray_sampler import RaySampler
from utils.tb_logger import TBLogger
import datetime


import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class MetaLearner:
    def __init__(
        self,
        env,
        agent: PPO,
        tb_logger: TBLogger, 
        train_tasks: List[int],
        test_tasks: List[int],
        args, configs
    ) -> None:
        
        self.env = env
        self.env_name = args.env_name
        self.agent = agent
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        self.tb_logger = tb_logger
        self.save_periods = args.save_periods
        self.weight_path = args.weight_path
        self.save_file_path = f"{args.env_name}_{args.exp_name}_{args.seed}_{args.now}"
        self.max_step: int = args.max_episode_steps
        self.num_samples: int = args.rollout_steps
        self.num_iterations: int = configs["n_epochs"]
        self.meta_batch_size: int = configs["meta_batch_size"]
        self.batch_size: int = self.meta_batch_size * self.num_samples

        self.sampler = RL2Sampler(
            env=env,
            agent=agent,
            args=args,
            configs=configs)
        self.buffer = RolloutBuffer(
            args=args, configs=configs
        )

        if args.load:
            ckpt_path = os.path.join(
                self.weight_path,
                '/',
                args.env_name,
                '_',
                args.exp_name,
                '_',
                args.seed,
                '/',
                "checkpoint_" + str(args.load_ckpt_num) + ".pt",
            )
            ckpt = torch.load(ckpt_path)

            self.agent.policy.load_state_dict(ckpt["policy"])
            self.agent.vf.load_state_dict(ckpt["vf"])
            self.buffer = ckpt["buffer"]

        # 조기 학습 중단 조건 설정
        self.dq: deque = deque(maxlen=configs["num_stop_conditions"])
        self.num_stop_conditions: int = configs["num_stop_conditions"]
        self.stop_goal: int = configs["stop_goal"]
        self.is_early_stopping = False

    def meta_train_parallel(self) -> None:
        total_start_time = time.time()
        for iteration in range(self.num_iterations):
            start_time = time.time()

            # TODO ==========================
            # ? applying parallel processing
            print(f"=============== Iteration {iteration} ===============")
            pr = [sampler.print_worker_infos.remote() for sampler in self.samplers]
            ray.get(pr)
            traj_refs = [sampler.obtain_samples.remote(np.random.randint(len(self.train_tasks), size=1)) for sampler in self.samplers]
            trajs = ray.get(traj_refs)
            self.buffer.add_trajs(trajs)
            # TODO ==========================
            
            batch = self.buffer.sample_batch()

            print(f"Start the meta-gradient update of iteration {iteration}")
            log_values = self.agent.train_model(self.batch_size, batch)
            if iteration % self.save_periods == 0:
                if not os.path.exists(self.save_file_path):
                    self.save_file_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.curdir, 'weights',self.save_file_path))
                    try:
                        os.mkdir(self.save_file_path)
                    except:
                        dir_path_head, dir_path_tail = os.path.split(self.save_file_path)
                        if len(dir_path_tail) == 0:
                            dir_path_head, dir_path_tail = os.path.split(dir_path_head)
                        try:
                            os.mkdir(dir_path_head)
                            os.mkdir(self.save_file_path)
                        except:
                            pass
                ckpt_path = os.path.join(self.save_file_path,"checkpoint_" + str(iteration) + ".pt")
                print(self.save_file_path)
                torch.save(
                    {
                        "policy": self.agent.policy.state_dict(),
                        "vf": self.agent.vf.state_dict(),
                        "buffer": self.buffer,
                    },
                    ckpt_path,
                )
            self.meta_test(iteration, total_start_time, start_time, log_values)

    def meta_train(self) -> None:
        total_start_time = time.time()
        for iteration in range(self.num_iterations):
            start_time = time.time()

            print(f"=============== Iteration {iteration} ===============")
            indices = np.random.randint(len(self.train_tasks), size=self.meta_batch_size)
            
            print(f"Task indices {indices}")
            for i, index in enumerate(indices):
                self.env.reset_task(index)
                self.agent.policy.is_deterministic = False
                print(f"[{i + 1}/{self.meta_batch_size}] collecting samples, current task: {self.env.get_task()}")
                trajs: List[Dict[str, np.ndarray]] = self.sampler.obtain_samples()
                self.buffer.add_trajs(trajs)

            batch = self.buffer.sample_batch()

            print(f"Start the meta-gradient update of iteration {iteration}")
            log_values = self.agent.train_model(self.batch_size, batch)
            if iteration % self.save_periods == 0:
                if not os.path.exists(self.save_file_path):
                    self.save_file_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.curdir, 'weights',self.save_file_path))
                    try:
                        os.mkdir(self.save_file_path)
                    except:
                        dir_path_head, dir_path_tail = os.path.split(self.save_file_path)
                        if len(dir_path_tail) == 0:
                            dir_path_head, dir_path_tail = os.path.split(dir_path_head)
                        try:
                            os.mkdir(dir_path_head)
                            os.mkdir(self.save_file_path)
                        except:
                            pass
                ckpt_path = os.path.join(self.save_file_path,"checkpoint_" + str(iteration) + ".pt")
                print(self.save_file_path)
                torch.save(
                    {
                        "policy": self.agent.policy.state_dict(),
                        "vf": self.agent.vf.state_dict(),
                        "buffer": self.buffer,
                    },
                    ckpt_path,
                )
            self.meta_test(iteration, total_start_time, start_time, log_values)
    
    def log_on_tensorboard(self, test_results: Dict[str, Any], iteration: int) -> None:
        self.tb_logger.add("test/return", test_results["return"], iteration)
        self.tb_logger.add("train/total_loss", test_results["total_loss"], iteration)
        self.tb_logger.add("train/policy_loss", test_results["policy_loss"], iteration)
        self.tb_logger.add("train/value_loss", test_results["value_loss"], iteration)
        self.tb_logger.add("time/total_time", test_results["total_time"], iteration)
        self.tb_logger.add("time/time_per_iter", test_results["time_per_iter"], iteration)

    def meta_test(
        self,
        iteration: int,
        total_start_time: float,
        start_time: float,
        log_values: Dict[str, float],
    ) -> None:
        test_results = {}
        test_return: float = 0
        for index in self.test_tasks:
            self.env.reset_task(index)
            self.agent.policy.is_deterministic = True
            trajs: List[Dict[str, np.ndarray]] = self.sampler.obtain_samples()
            #test_return += np.array([np.sum(trajs[i]["rewards"])] for i in range(len(trajs))).mean().item()
            test_return += np.array([np.sum(trajs[i]["rewards"]) for i in range(len(trajs))]).mean().item()


        test_results["return"] = test_return / len(self.test_tasks)
        test_results["total_loss"] = log_values["total_loss"]
        test_results["policy_loss"] = log_values["policy_loss"]
        test_results["value_loss"] = log_values["value_loss"]
        test_results["total_time"] = time.time() - total_start_time
        test_results["time_per_iter"] = time.time() - start_time

        self.log_on_tensorboard(test_results, iteration)
