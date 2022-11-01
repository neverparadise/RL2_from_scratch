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


class TupleEmbedding(nn.Module):
    def __init__(self, args, configs) -> None:
        super().__init__()
        input_dim = configs["state_dim"] + configs["action_dim"] + 2 # 2: reward, done dimension
        self.device = args.device
        self.minibatch_size = configs["mini_batch_size"]
        self.embedding = nn.Sequential(
            layer_init(nn.Linear(input_dim, configs["linear_dim"])),
            nn.LeakyReLU(),
            layer_init(nn.Linear(configs["linear_dim"], configs["linear_dim"])),
            nn.LeakyReLU(),
            layer_init(nn.Linear(configs["linear_dim"], configs["trans_dim"])),
        )
        
    def forward(self, transition, is_training=False):
        state, action, reward, done = transition
        state = self._format(state)
        action = self._format(action)
        reward = self._format(reward)
        done = self._format(done)
        concatenated = torch.cat([state, action, reward, done], dim=-1)
        if is_training:
            concatenated = concatenated.reshape(1, self.minibatch_size, -1)
        return self.embedding(concatenated)

    def _format(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
            x = x.to(device=self.device)
        else:
            x = x.to(device=self.device)
        if len(x.shape) < 3:
            x = x.reshape(1, 1, -1) # [L, N, flatten]
        return x

class RNNActor(nn.Module):
    def __init__(self, args, configs) -> None:
        super().__init__()
        self.device = torch.device(args.device)
        self.is_continuous = configs["is_continuous"]
        self.hidden_dim = configs["hidden_dim"]
        self.num_rnn_layers = configs["num_rnn_layers"]
        self.action_dim = configs["action_dim"]
        self.num_discretes = configs["num_discretes"]
        self.embedding = TupleEmbedding(args, configs)
        self.gru = nn.GRU(configs["trans_dim"], configs["hidden_dim"], \
                            num_layers=self.num_rnn_layers, bias=True)
        if self.is_continuous:
            self.mean = layer_init(nn.Linear(self.hidden_dim, self.action_dim))
            self.log_std = -0.5 * np.ones(self.action_dim, dtype=np.float32)
            self.log_std = torch.nn.Parameter(torch.Tensor(self.log_std))
            #self.std =  layer_init(nn.Linear(self.hidden_dim, self.action_dim))
        else:
            self.policy_logits = layer_init(nn.Linear(configs["hidden_dim"], self.num_discretes))

        if configs["is_shared_network"]:
            self.fc = nn.Linear(configs["hidden_dim"], configs["hidden_dim"])
            self.v = nn.Linear(configs["hidden_dim"], 1)   
             
    def forward(self, transition, hidden=None, is_training=False):
        x = self.embedding(transition, is_training)
        hidden = to_tensor(hidden, device=self.device)
        if is_training:
            hidden = hidden.permute(1, 0, 2).contiguous()
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

    def value(self, transition, hidden=None, is_training=False):
        x = self.embedding(transition, is_training)
        x, new_hidden = self.gru(x, hidden)
        x = F.leaky_relu(self.fc(x))
        return self.v(x), new_hidden

class RNNCritic(nn.Module):
    def __init__(self, args, configs) -> None:
        super().__init__()
        # information        
        self.device = torch.device(args.device)
        self.is_continuous = configs["is_continuous"]
        self.hidden_dim = configs["hidden_dim"]
        self.num_rnn_layers = configs["num_rnn_layers"]
        self.action_dim = configs["action_dim"]
        
        # network
        self.embedding = TupleEmbedding(args, configs)
        self.gru = nn.GRU(configs["trans_dim"], configs["hidden_dim"], \
                            num_layers=self.num_rnn_layers, bias=True)
        self.v = nn.Linear(configs["hidden_dim"], 1)   
             
    def forward(self, transition, hidden=None, is_training=False):
        x = self.embedding(transition, is_training)
        hidden = to_tensor(hidden, device=self.device)
        if is_training:
            hidden = hidden.permute(1, 0, 2).contiguous()
        x, new_hidden = self.gru(x, hidden)
        return self.v(x), new_hidden
    