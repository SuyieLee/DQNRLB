# https://github.com/udacity/deep-reinforcement-learning/blob/master/solution/dqn_agent.py

# Modified batch size to 32
# gamma is set to 1

import numpy as np
import random
from collections import namedtuple, deque

from model import Network

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32  # minibatch size
GAMMA = 1.0  # discount factor
TAU = 1e-3  # for soft update of target parameters
LR = 1e-3  # learning rate
MaxP = 1e9
# UPDATE_EVERY = 4  # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, UPDATE_EVERY, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.C = UPDATE_EVERY
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = Network(state_size, action_size, seed).to(device)
        self.qnetwork_target = Network(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        # self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)
        self.memory = NaivePrioritizedBuffer(BUFFER_SIZE, BATCH_SIZE)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done, beta):
        # Save experience in replay memory
        self.memory.push(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.C
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            # if len(self.memory) > BATCH_SIZE:
            # experiences = self.memory.sample()
            self.learn(beta, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def test_act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        return np.argmax(action_values.cpu().data.numpy())

    def learn(self, beta, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        state, action, reward, next_state, done, indices, weights = self.memory.sample(beta)
        # q_value = self.qnetwork_local(state).gather(1, action)
        #
        # next_q_values = self.qnetwork_local(next_state)
        # next_q_value = self.qnetwork_target(next_state).detach().gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1))
        # expected_q_value = reward + gamma * next_q_value * (1 - done)

        q_value = self.qnetwork_local(state).gather(1, action)

        next_q_values = self.qnetwork_local(next_state)
        next_q_value = self.qnetwork_target(next_state).detach().gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1))
        expected_q_value = reward + gamma * next_q_value * (1 - done)

        # Compute loss
        # loss = F.mse_loss(q_value, expected_q_value.data)

        loss = (q_value.squeeze(1) - expected_q_value.data.squeeze(1)).pow(2) * weights
        prios = loss + 1e-5
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.memory.update_priorities(indices, prios.data.cpu().numpy())
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def savemodel(self):
        torch.save(self.qnetwork_local.state_dict(), './model')
        return


class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, batch_size, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.batch_size = batch_size
        self.priorities = np.zeros((capacity,), dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        states = torch.from_numpy(np.vstack([e[0] for e in samples if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e[1] for e in samples if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e[2] for e in samples if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in samples if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in samples if e is not None]).astype(np.uint8)).float().to(device)
        weights = torch.FloatTensor(weights)
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)