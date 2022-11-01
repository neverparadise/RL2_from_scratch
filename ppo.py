from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass, asdict, astuple
from networks import *



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
        self.is_continuous = configs["is_continuous"]
        self.is_deterministic = configs["is_dterministic"]

        # Networks
        if self.meta_learning:
            self.policy = RL2Actor(args, configs).to(self.device)
            self.vf = RL2Critic(args, configs).to(self.device)
        else:
            self.policy = RNNActor(args, configs).to(self.device)
            self.vf = RNNCritic(args, configs).to(self.device)

        self.optimizer = optim.Adam(
                list(self.policy.parameters()) + list(self.vf.parameters()), lr=float(configs["lr"]))

        self.net_dict = {
            "policy": self.policy,
            "vf": self.vf,
        }

    def get_action(self, transition, hidden, is_training=False):
        dist, hidden = self.policy(transition, hidden, is_training)
        if self.is_deterministic:
            action = dist.mean
            log_prob = torch.zeros(1)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action)
            log_prob = log_prob.sum(-1)
        if is_training:
            return action, dist.log_prob(action), hidden
        else:
            while len(action.shape) != 1:
                action = action.squeeze(0)
                log_prob = log_prob.squeeze(0)
            if self.is_continuous:
                return action.detach().to('cpu').numpy(), log_prob.detach().to('cpu').numpy(), hidden.detach().cpu().numpy()
            else:
                return action.detach().to('cpu').numpy().item(), log_prob.detach().to('cpu').numpy(), hidden.detach().cpu().numpy()

    def get_value(self, transition, hidden=None, is_training=False):
        value, new_hidden = self.vf(transition, hidden, is_training)
        return value.detach().cpu().numpy(), new_hidden.detach().cpu().numpy()

    # TODO: code refactoring
    def train_model(self, batch_size: int, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
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
                # Value Loss
                trans_batch = (obs_batch, action_batch, rew_batch, done_batch)
                value_batch, _ = self.vf(trans_batch, v_hidden_batch, is_training=True)
                value_loss = F.mse_loss(value_batch.view(-1, 1), return_batch)

                # Policy Loss
                _, new_log_prob_batch, _ = self.get_action(
                    trans_batch,
                    pi_hidden_batch,
                    is_training=True
                )

                ratio = torch.exp(new_log_prob_batch.view(-1, 1) - log_prob_batch)

                policy_loss = ratio * advant_batch
                clipped_loss = (
                    torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advant_batch
                )

                policy_loss = -torch.min(policy_loss, clipped_loss).mean()

                # Backward()
                total_loss = policy_loss + 0.5 * value_loss
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
