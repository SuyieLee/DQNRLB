"""
Adapted from: https://github.com/udacity/deep-reinforcement-learning/blob/master/dqn/solution/model.py

The code was modified to add one more hidden layer as suggested by the paper: 
Budget Constrained Bidding by Model-free Reinforcement Learning in Display Advertising
(https://arxiv.org/pdf/1802.08365.pdf)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Network(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=100, 
                    fc2_units=100, fc3_units=100):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(self.weight_epsilon)
            bias = self.bias_mu + self.bias_sigma.mul(self.bias_epsilon)
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class NoisyDQN(nn.Module):
    def __init__(self, num_inputs, num_outputs, fc1_units=100,
                    fc2_units=100, fc3_units=100):
        super(NoisyDQN, self).__init__()

        self.linear = nn.Linear(num_inputs, fc1_units)
        self.noisy1 = NoisyLinear(fc1_units, fc2_units)
        self.noisy2 = NoisyLinear(fc2_units, fc3_units)
        self.noisy3 = NoisyLinear(fc3_units, num_outputs)

    def forward(self, x):
        x = F.relu(self.linear(x))
        x = F.relu(self.noisy1(x))
        x = F.relu(self.noisy2(x))
        x = self.noisy3(x)
        return x

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        q_value = self.forward(state)
        action = q_value.max(1)[1].data[0]
        return action

    def reset_noise(self):
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()
        self.noisy3.reset_noise()