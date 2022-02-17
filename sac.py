import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal

import numpy as np
import gym
import pybullet_envs
import os
import copy


from models import TwinQNetwork, ValueNetwork, Policy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SAC(object):
    def __init__(self, state_dim, action_dim, max_action, gamma=0.99, tau=0.005, alpha=0.2, lr=3e-4, target_entropy=None, automatic_entropy_tuning=True):


        self.gamma = gamma
        self.tau   = tau
        self.alpha = alpha


        # Q NETWORK AS CRITIC
        self.critic = TwinQNetwork(state_dim, action_dim).to(device)
        self.target_critic = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)


        # POLICY NETWORK
        self.policy = Policy(state_dim, action_dim).to(device)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)


        # ENTROPY
        self.target_entropy = target_entropy if target_entropy else -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

        self.automatic_entropy_tuning = automatic_entropy_tuning

    def select_action(self, x, action=None):

        state = torch.FloatTensor(x).to(device).unsqueeze(0)
        action, _ = self.policy.sample(state)    
        return action.detach().cpu().numpy()[0]


    def update_target(self):
        '''moving average update of target networks'''
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def update_critic(self, state, action, reward, next_state, not_done):
        '''UPDATES CRITIC NETWORK PARAMS'''

        with torch.no_grad():
            next_action, next_logprobs  = self.policy.sample(next_state)

            current_Q1, current_Q2 = self.target_critic(next_state, next_action)

            min_Q = torch.min(current_Q1, current_Q2)

            Q_target = reward + not_done * self.gamma * (min_Q - self.alpha * next_logprobs)


        # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        Q_1, Q_2 = self.critic(state, action)
        loss_1 = F.mse_loss(Q_1, Q_target)
        loss_2 = F.mse_loss(Q_2, Q_target)

        Q_loss = loss_1 + loss_2

        self.critic_optimizer.zero_grad()
        Q_loss.backward()
        self.critic_optimizer.step()


        return Q_loss


    def update_actor_and_alpha(self, state):
        ''' UPDATES ACTOR NETWORK '''

        action, logprobs = self.policy.sample(state)

        Q_1, Q_2 = self.critic(state, action)

        qval = torch.min(Q_1, Q_2)


        policy_loss = (self.alpha * logprobs - qval).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        if self.automatic_entropy_tuning:
            # Update alpha
            alpha_loss = -(self.log_alpha * (logprobs + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()

        else:
            alpha_loss = torch.tensor(0.).to(device)


        return policy_loss, alpha_loss

        


    def train(self, batch):

        # Sample replay buffer
        state, action, next_state, reward, not_done = batch

        critic_loss = self.update_critic(state, action, reward, next_state, not_done)

        actor_loss, entropy_loss = self.update_actor_and_alpha(state)


        self.update_target()


        return {"critic_loss": critic_loss.item(),
                "actor_loss": actor_loss.item(),
                "alpha": self.alpha,
                }




    def save(self, filename):
        torch.save({'critic': self.critic.state_dict(),
                    'policy': self.policy.state_dict()}, filename + "_sac.pth")


















