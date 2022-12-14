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
        self.ent_coef = float(configs["ent_coef"])
        self.vf_coef = float(configs["vf_coef"])
        self.is_continuous = configs["is_continuous"]
        self.is_deterministic = configs["is_deterministic"]
        self.norm_adv = configs["normalize_advantages"]
        self.max_grad_norm = configs["max_grad_norm"]

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
        dist, new_hidden = self.policy(transition, hidden, is_training)
        if is_training:
            return action, dist.log_prob(action).sum(-1), dist.entropy().sum(-1), new_hidden
        else: # evaluation
            if self.is_deterministic:
                action = dist.mean
                log_prob = torch.zeros(1)
            else: # stochastic
                action = dist.sample()
                log_prob = dist.log_prob(action)
            while len(action.shape) != 1:
                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)
            if self.is_continuous:
                action = action.detach().to('cpu').numpy()
            else:
                action = action.detach().to('cpu').numpy().item()
            log_prob = log_prob.sum(-1).detach().to('cpu').numpy()
            entropy = dist.entropy().sum(-1).detach().to('cpu').numpy()
            return action, log_prob, entropy, new_hidden.detach().cpu().numpy()
                        
    def get_value(self, transition, hidden=None, is_training=False):
        value, new_hidden = self.vf(transition, hidden, is_training)
        return value.detach().cpu().numpy(), new_hidden.detach().cpu().numpy()

    # TODO: code refactoring
    def train_model(self, batch_size: int, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        
        # ! ??? ?????? ???????????????. yield ?????? ????????? ????????? ????????? ??????.
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

        sum_total_loss: float = 0
        sum_policy_loss: float = 0
        sum_value_loss: float = 0
        
        b_indices = np.arange(batch_size)
        for epoch in range(self.num_epochs):
            sum_total_loss_mini_batch = 0
            sum_policy_loss_mini_batch = 0
            sum_value_loss_mini_batch = 0
            #np.random.shuffle(b_indices)
            for start in range(0, batch_size, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_indices = b_indices[start:end]
                
                # * mini batches
                mb_obs = observations[mb_indices].to(self.device)
                mb_acts = actions[mb_indices].to(self.device)
                mb_rews = rewards[mb_indices].to(self.device)
                mb_dones = dones[mb_indices].to(self.device)
                mb_returns = returns[mb_indices].to(self.device)
                mb_old_log_probs = old_log_probs[mb_indices].to(self.device)
                mb_advants = advants[mb_indices].to(self.device)
                mb_pi_hiddens = pi_hiddens[mb_indices].to(self.device)
                mb_v_hiddens = v_hiddens[mb_indices].to(self.device)
                
                mb_trans = (mb_obs, mb_acts, mb_rews, mb_dones)
                
                if self.norm_adv:
                    mb_advants = (mb_advants - mb_advants.mean()) / (mb_advants.std() + 1e-8)
                _, mb_new_log_probs, mb_entropy, _ = self.get_action(
                    mb_trans,
                    mb_pi_hiddens,
                    mb_acts,
                    is_training=True
                )
                
                logratio = mb_new_log_probs - mb_old_log_probs
                ratio = logratio.exp()
                
                # Policy loss
                pg_loss1 = -mb_advants * ratio
                pg_loss2 = -mb_advants * torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param)
                policy_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                mb_values, _ = self.vf(mb_trans, mb_v_hiddens, is_training=True)
                value_loss = F.mse_loss(mb_values, mb_returns)
                #value_loss = 0.5 * ((mb_values - mb_returns) ** 2).mean()
                
                entropy_loss = mb_entropy.mean()
                total_loss = policy_loss - self.ent_coef * entropy_loss + self.vf_coef * value_loss 
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                nn.utils.clip_grad_norm_(self.vf.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                sum_total_loss_mini_batch += total_loss
                sum_policy_loss_mini_batch += policy_loss
                sum_value_loss_mini_batch += value_loss
                
            sum_total_loss += sum_total_loss_mini_batch / num_mini_batch
            sum_policy_loss += sum_policy_loss_mini_batch / num_mini_batch
            sum_value_loss += sum_value_loss_mini_batch / num_mini_batch
                
            sum_total_loss_mini_batch = 0
            sum_policy_loss_mini_batch = 0
            sum_value_loss_mini_batch = 0
        
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
