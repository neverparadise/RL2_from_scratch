import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def to_tensor(np_array, device="cpu"):
    if isinstance(np_array, np.ndarray):
        return torch.from_numpy(np_array).float().to(device)
    return np_array.float().to(device)


def _format(x, device, minibatch_size=1, is_training=False):
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32)
        x = x.to(device=device)
    else:
        x = x.to(device=device)
    if len(x.shape) < 3:
        x = x.reshape(1, 1, -1) # [L, N, flatten]
    if is_training:
        x = x.reshape(1, minibatch_size, -1)
    return x


class RNNActor(nn.Module):
    def __init__(self, args, configs) -> None:
        super().__init__()
        self.device = torch.device(args.device)
        self.input_dim = configs["state_dim"]
        self.minibatch_size = configs["mini_batch_size"]
        self.is_continuous = configs["is_continuous"]
        self.hidden_dim = configs["hidden_dim"]
        self.num_rnn_layers = configs["num_rnn_layers"]
        self.action_dim = configs["action_dim"]
        self.num_discretes = configs["num_discretes"]

        self.embedding = nn.Sequential(
            layer_init(nn.Linear(self.input_dim, configs["linear_dim"])),
            nn.LeakyReLU(),
            layer_init(nn.Linear(configs["linear_dim"], configs["trans_dim"])),
            nn.LeakyReLU(),
        )
        self.gru = nn.GRU(configs["trans_dim"], configs["hidden_dim"], \
                            num_layers=self.num_rnn_layers, bias=True)
        if self.is_continuous:
            self.mean = layer_init(nn.Linear(self.hidden_dim, self.action_dim))
            self.log_std = -0.5 * np.ones(self.action_dim, dtype=np.float32)
            self.log_std = torch.nn.Parameter(torch.Tensor(self.log_std))
            #self.std =  layer_init(nn.Linear(self.hidden_dim, self.action_dim))
        else:
            self.policy_logits = layer_init(nn.Linear(configs["hidden_dim"], self.num_discretes))

    def forward(self, state, hidden=None, is_training=False):
        state = _format(state, self.device, self.minibatch_size, is_training)
        if is_training:
            hidden = hidden.permute(1, 0, 2).contiguous()
        x = self.embedding(state)
        hidden = to_tensor(hidden, device=self.device)
        x, new_hidden = self.gru(x, hidden)
        if self.is_continuous:
            mu = torch.tanh(self.mean(x))
            std = torch.exp(self.log_std)
            # std = F.softplus(self.std(x))
            dist = Normal(mu, std)
        else:
            logits = self.policy_logits(x)
            prob = F.softmax(logits, dim=-1)
            dist = Categorical(prob)
        return dist, new_hidden


class RNNCritic(nn.Module):
    def __init__(self, args, configs) -> None:
        super().__init__()
        # information        
        self.device = torch.device(args.device)
        self.minibatch_size = configs["mini_batch_size"]
        self.input_dim = configs["state_dim"]
        self.is_continuous = configs["is_continuous"]
        self.hidden_dim = configs["hidden_dim"]
        self.num_rnn_layers = configs["num_rnn_layers"]
        self.action_dim = configs["action_dim"]
        
        # network
        self.embedding = nn.Sequential(
            layer_init(nn.Linear(self.input_dim, configs["linear_dim"])),
            nn.LeakyReLU(),
            layer_init(nn.Linear(configs["linear_dim"], configs["linear_dim"])),
            nn.LeakyReLU(),
            layer_init(nn.Linear(configs["linear_dim"], configs["trans_dim"])),
        )
        self.gru = nn.GRU(configs["trans_dim"], configs["hidden_dim"], \
                            num_layers=self.num_rnn_layers, bias=True)
        self.v = nn.Linear(configs["hidden_dim"], 1)   
             
    def forward(self, state, hidden=None, is_training=False):
        state = _format(state, self.device, self.minibatch_size, is_training)
        if is_training:
            hidden = hidden.permute(1, 0, 2).contiguous()
        x = self.embedding(state)
        hidden = to_tensor(hidden, device=self.device)
        if is_training:
            hidden = hidden.permute(1, 0, 2).contiguous()
        x, new_hidden = self.gru(x, hidden)
        return self.v(x), new_hidden
    