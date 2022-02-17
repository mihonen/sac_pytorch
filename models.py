import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNet, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)


    def forward(self, state):
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        value = self.l3(x)

        return value




class TwinQNetwork(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(TwinQNetwork, self).__init__()

        # Q1
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)


        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)


        return q1, q2


class Policy(nn.Module):
    ''' GAUSSIAN POLICY '''
    def __init__(self, state_dim, action_dim, log_std_min=-20, log_std_max=2):
        super(Policy, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)

        self.mean = nn.Linear(256, action_dim)
        self.log_std  = nn.Linear(256, action_dim)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max


    def forward(self, state):
        x = self.l1(state)
        x = F.relu(x)
        x = self.l2(x)
        x = F.relu(x)
        mean = self.mean(x)
        log_std =  self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std


    def sample(self, state, epsilon=1e-6):
        '''Action sampled from  Squashed Gaussian Policy'''
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        dist = Normal(0, 1)
        e = dist.sample().to(device)
        action = torch.tanh(mean + e * std)
        log_prob = normal.log_prob(mean + e * std) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob














