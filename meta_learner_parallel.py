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
        env_creator,
        agent: PPO,
        tb_logger: TBLogger, 
        train_tasks: List[int],
        test_tasks: List[int],
        args, configs
    ) -> None:
        
        self.env_creator = env_creator
        self.env, _ = self.env_creator(args, configs)
        self.env_name = args.env_name
        self.agent = agent
        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        self.tb_logger = tb_logger
        self.save_periods = args.save_periods
        self.weight_path = args.weight_path
        self.save_file_path = f"{args.env_name}_{args.exp_name}_{args.seed}_{args.now}"
        self.max_steps: int = args.max_episode_steps
        self.num_samples: int = args.rollout_steps
        self.debug = args.debug
        self.num_test_tasks = configs["num_test_tasks"]
        self.num_iterations: int = configs["n_epochs"]
        self.meta_batch_size: int = configs["meta_batch_size"]
        self.batch_size: int = args.batch_size
        self.eval_periods = args.eval_periods
        
        if args.parallel_processing:
            self.train_samplers = []
            for i in range(self.meta_batch_size):
                env, tasks = self.env_creator(args, configs)
                self.train_samplers.append(RaySampler.remote(env, self.agent, args, configs, i))
            print(self.train_samplers)
                
            self.test_samplers = []
            for j in range(self.num_test_tasks):
                env, tasks = self.env_creator(args, configs)
                self.test_samplers.append(RaySampler.remote(env, self.agent, args, configs, j))
            print(self.test_samplers)
        else:
            self.sampler = RL2Sampler(
                env=env,
                agent=agent,
                args=args,
                configs=configs
            )

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

        # ?????? ?????? ?????? ?????? ??????
        self.dq: deque = deque(maxlen=configs["num_stop_conditions"])
        self.num_stop_conditions: int = configs["num_stop_conditions"]
        self.stop_goal: int = configs["stop_goal"]
        self.is_early_stopping = False

    def meta_train_parallel(self) -> None:
        total_start_time = time.time()
        for iteration in range(self.num_iterations):
            start_time = time.time()
            results = {}

            # ? applying parallel processing
            print(f"=============== Iteration {iteration} ===============")
            traj_refs = []
            for sampler in self.train_samplers:
                #task_idx = np.random.randint(len(self.train_tasks), size=1).item()
                #task_idx = np.array([1 for i in len(self.train_tasks)])
                if self.debug:
                    task_idx = 1
                ref = sampler.obtain_samples.remote(task_idx)
                traj_refs.append(ref)
            workers_trajs = ray.get(traj_refs) # [[traj1, traj2], [traj1, traj2, ...]]
            
            train_return = np.array([])
            for trajs in workers_trajs:
                self.buffer.add_trajs(trajs)
                for traj in trajs:
                    return_ =  np.array([np.sum(traj["rewards"])])
                    train_return = np.concatenate([return_, train_return])
            mean_train_return = train_return.mean().item()
            results["mean_train_return"] = mean_train_return
            
            print(f"buffer_size: {self.buffer.size}")
            
            batch = self.buffer.sample_batch()

            print(f"Start the meta-gradient update of iteration {iteration}")
            log_values = self.agent.train_model(self.batch_size, batch)
            
            results["total_loss"] = log_values["total_loss"]
            results["policy_loss"] = log_values["policy_loss"]
            results["value_loss"] = log_values["value_loss"]
            results["total_time"] = time.time() - total_start_time
            results["time_per_iter"] = time.time() - start_time
            
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


            # * Meta-test procedure
            test_return = np.array([])
            print(f"Start the meta-test evaluation {iteration}")
            test_traj_refs = []
            for sampler in self.test_samplers:
                #task_idx = np.random.randint(len(self.test_tasks), size=1).item()
                if self.debug:
                    task_idx = 1
                #self.agent.policy.is_deterministic = True
                ref = sampler.obtain_samples.remote(task_idx)
                print(ref)
                test_traj_refs.append(ref)
            test_workers_trajs = ray.get(test_traj_refs) 
            print(f"Start the return calculation {iteration}")
            for trajs in test_workers_trajs:
                for traj in trajs:
                    return_ =  np.array([np.sum(traj["rewards"])])
                    test_return = np.concatenate([return_, test_return])
            mean_test_return = test_return.mean().item()
            results["mean_test_return"] = mean_test_return
            self.log_on_tensorboard(results, iteration)
            
    def log_on_tensorboard(self, results: Dict[str, Any], iteration: int) -> None:
        self.tb_logger.add("test/return", results["mean_test_return"], iteration)
        self.tb_logger.add("train/return", results["mean_train_return"], iteration)
        self.tb_logger.add("train/total_loss", results["total_loss"], iteration)
        self.tb_logger.add("train/policy_loss", results["policy_loss"], iteration)
        self.tb_logger.add("train/value_loss", results["value_loss"], iteration)
        self.tb_logger.add("time/total_time", results["total_time"], iteration)
        self.tb_logger.add("time/time_per_iter", results["time_per_iter"], iteration)

    def meta_test_parallel(
        self,
        iteration: int,
        total_start_time: float,
        start_time: float,
        log_values: Dict[str, float],
    ) -> None:
        results = {}
        test_return = np.array([])
        
        print(f"Start the meta-test evaluation {iteration}")
        
        #! TODO: ?????? ??????????????? ??????
        traj_refs = []

        for sampler in self.test_samplers:
            task_idx = np.random.randint(len(self.test_tasks), size=1).item()
            #self.agent.policy.is_deterministic = True
            ref = sampler.obtain_samples.remote(task_idx)
            traj_refs.append(ref)
            
        workers_trajs = ray.get(traj_refs) 
        
        print(f"Start the return calculation {iteration}")
        
        for trajs in workers_trajs:
            for traj in trajs:
           
                return_ =  np.array([np.sum(traj["rewards"])])
                test_return = np.concatenate([return_, test_return])
        test_return = test_return.mean().item()


        results["return"] = test_return / len(self.test_tasks)
        results["total_loss"] = log_values["total_loss"]
        results["policy_loss"] = log_values["policy_loss"]
        results["value_loss"] = log_values["value_loss"]
        results["total_time"] = time.time() - total_start_time
        results["time_per_iter"] = time.time() - start_time

        self.log_on_tensorboard(results, iteration)
