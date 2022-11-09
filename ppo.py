from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass, asdict, astuple
from networks import *


def to_tensor(np_array, device):
    if isinstance(np_array, np.ndarray):
        return torch.from_numpy(np_array).float().to(device)
    return np_array.float().to(device)

class PPO:
    def __init__(
        self,
        args,
        configs,
    ) -> None:
        self.device = torch.device(args.device)
        self.meta_learning = args.meta_learning
        self.num_epochs = configs["k_epochs"]
        self.mini_batch_size = configs["mini_batch_size"]
        self.clip_param = configs["lr_clip_range"]
        self.ent_coef = configs["ent_coef"]
        self.vf_coef = configs["vf_coef"]
        self.is_continuous = configs["is_continuous"]
        self.is_deterministic = configs["is_deterministic"]

        # Networks
        if self.meta_learning:
            self.policy = RL2Actor(args, configs).to(self.device)
            self.vf = RL2Critic(args, configs).to(self.device)
        else:
            self.policy = RNNActor(args, configs).to(self.device)
            self.vf = RNNCritic(args, configs).to(self.device)

        # if torch.cuda.device_count() > 1:
        #     print("Let's use", torch.cuda.device_count(), "GPUs!")
        #     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        #     self.policy = nn.DataParallel(self.policy)
        #     self.vf = nn.DataParallel(self.vf)

        self.optimizer = optim.Adam(
                list(self.policy.parameters()) + list(self.vf.parameters()), lr=float(configs["lr"]))

        self.net_dict = {
            "policy": self.policy,
            "vf": self.vf,
        }

    def get_action(self, transition, hidden, action=None, is_training=False):
        dist, hidden = self.policy(transition, hidden, is_training)
        if is_training:
            return action, dist.log_prob(action), dist.entropy().sum(1), hidden
        else: # evaluation
            if self.is_deterministic:
                action = dist.mean
                log_prob = torch.zeros(1)
            else: # stochastic
                action = dist.sample()
                log_prob = dist.log_prob(action)
                log_prob = log_prob.sum(-1)
            while len(action.shape) != 1:
                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)
            if self.is_continuous:
                action = action.detach().to('cpu').numpy()
            else:
                action = action.detach().to('cpu').numpy().item()
            log_prob = log_prob.detach().to('cpu').numpy()
            entropy = dist.entropy().sum(1).detach().to('cpu').numpy()
            return action, log_prob, entropy, hidden.detach().cpu().numpy()
                        
    def get_value(self, transition, hidden=None, is_training=False):
        value, new_hidden = self.vf(transition, hidden, is_training)
        return value.detach().cpu().numpy(), new_hidden.detach().cpu().numpy()

    # TODO: code refactoring
    def train_model(self, batch_size: int, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        
        # ! 이 부분 개선해야함. yield 써서 메모리 사용량 줄여야 한다.
        num_mini_batch = int(batch_size / self.mini_batch_size) # 8192 / 64
        batch=asdict(batch)
        observations = batch["observations"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        dones = batch["dones"]
        returns = batch["returns"]
        old_log_probs = batch["old_log_probs"]
        advants = batch["advantages"]
        pi_hiddens = batch["pi_hiddens"]
        v_hiddens = batch["v_hiddens"]

        obs_batches = torch.chunk(observations, num_mini_batch) # 512 / 128 =4
        action_batches = torch.chunk(actions, num_mini_batch)
        reward_batches = torch.chunk(rewards, num_mini_batch)
        done_batches = torch.chunk(dones, num_mini_batch)
        return_batches = torch.chunk(returns, num_mini_batch)
        log_prob_batches = torch.chunk(old_log_probs, num_mini_batch)
        advant_batches = torch.chunk(advants, num_mini_batch)
        pi_hidden_batches = torch.chunk(pi_hiddens, num_mini_batch)
        v_hidden_batches = torch.chunk(v_hiddens, num_mini_batch)

        sum_total_loss: float = 0
        sum_policy_loss: float = 0
        sum_value_loss: float = 0

        for _ in range(self.num_epochs):
            
            sum_total_loss_mini_batch = 0
            sum_policy_loss_mini_batch = 0
            sum_value_loss_mini_batch = 0
            inner_step = 0
            for (
                obs_batch,
                action_batch,
                rew_batch, 
                done_batch,
                pi_hidden_batch,
                v_hidden_batch,
                return_batch,
                advant_batch,
                log_prob_batch,
            ) in zip(
                obs_batches,
                action_batches,
                reward_batches,
                done_batches,
                pi_hidden_batches,
                v_hidden_batches,
                return_batches,
                advant_batches,
                log_prob_batches,
            ):
                inner_step += 1
                print(inner_step)
                # Value Loss
                trans_batch = (obs_batch.to(self.device), action_batch.to(self.device),\
                    rew_batch.to(self.device), done_batch.to(self.device))
                value_batch, _ = self.vf(trans_batch, v_hidden_batch.to(self.device), is_training=True)
                value_loss = F.mse_loss(value_batch.view(-1, 1), return_batch.to(self.device))

                # Policy Loss
                _, new_log_prob_batch, entropy, _ = self.get_action(
                    trans_batch,
                    pi_hidden_batch.to(self.device),
                    action_batch.to(self.device),
                    is_training=True
                )

                ratio = torch.exp(new_log_prob_batch.view(-1, 1) - log_prob_batch.to(self.device))

                policy_loss = ratio * advant_batch.to(self.device)
                clipped_loss = (
                    torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advant_batch.to(self.device)
                )

                entropy_loss = entropy.mean()
                
                policy_loss = -torch.min(policy_loss, clipped_loss).mean()

                # Backward()
                total_loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                sum_total_loss_mini_batch += total_loss
                sum_policy_loss_mini_batch += policy_loss
                sum_value_loss_mini_batch += value_loss

            sum_total_loss += sum_total_loss_mini_batch / num_mini_batch
            sum_policy_loss += sum_policy_loss_mini_batch / num_mini_batch
            sum_value_loss += sum_value_loss_mini_batch / num_mini_batch

        mean_total_loss = sum_total_loss / self.num_epochs
        mean_policy_loss = sum_policy_loss / self.num_epochs
        mean_value_loss = sum_value_loss / self.num_epochs
        return dict(
            total_loss=mean_total_loss.item(),
            policy_loss=mean_policy_loss.item(),
            value_loss=mean_value_loss.item(),
        )

    def load(self, ckpt_path):
        ckpt_state_dict = torch.load(ckpt_path)
        self.policy.load_state_dict(ckpt_state_dict["policy"])
        self.vf.load_state_dict(ckpt_state_dict["vf"])
        print("model load completed")

    def save():
        pass
