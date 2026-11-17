import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from network import PolicyNet,ValueNet,PolicyNetContinuous
from torch.utils.data import TensorDataset,DataLoader

class PPO():
    def __init__(self,
                 state_dim=4,
                 hidden_dim=128,
                 action_dim=2,
                 actor_lr=1e-3,
                 critic_lr=1e-2,
                 gamma=0.98,
                 lmbda=0.95,
                 epochs=10,
                 eps=0.2
                 ):
        self.actor = PolicyNet(state_dim,hidden_dim,action_dim)
        self.critic = ValueNet(state_dim,hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps

    def take_action(self,obs,deterministic=False):
        obs = torch.tensor(obs,dtype=torch.float32)
        with torch.no_grad():
            probs = self.actor(obs)
            if deterministic:
                return torch.argmax(probs).item()
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                return action.item()

    def compute_advantage(self,td_delta,dones):
        td_delta = td_delta.squeeze(-1).tolist()
        dones = dones.squeeze(-1).tolist()
        advantage = 0.0
        advantage_list = []
        for delta, done in zip(reversed(td_delta),reversed(dones)):
            advantage = delta + self.gamma * self.lmbda * (1.0 - done) * advantage
            advantage_list.append(advantage)
        advantage_list.reverse()
        advantage_list = torch.tensor(advantage_list,dtype=torch.float32).view(-1,1)
        return advantage_list

    def update(self,states,actions,rewards,next_states,dones):
        states = torch.tensor(np.array(states),dtype=torch.float32)
        actions = torch.tensor(np.array(actions),dtype=torch.int64).view(-1,1)
        rewards = torch.tensor(np.array(rewards),dtype=torch.float32).view(-1,1)
        next_states = torch.tensor(np.array(next_states),dtype=torch.float32)
        dones = torch.tensor(np.array(dones),dtype=torch.float32).view(-1,1)

        with torch.no_grad():
            old_values = self.critic(states)
            td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
            td_delta = td_target - old_values
            advantage = self.compute_advantage(td_delta, dones)
            returns = advantage + old_values
            if len(advantage) > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            old_dist = torch.distributions.Categorical(self.actor(states))
            old_log_probs = old_dist.log_prob(actions.squeeze(-1)).view(-1,1)

        for _ in range(self.epochs):
            new_dist = torch.distributions.Categorical(self.actor(states))
            new_log_probs = new_dist.log_prob(actions.squeeze(-1)).view(-1,1)
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio,1 - self.eps, 1 + self.eps) * advantage
            actor_loss = -torch.min(surr1,surr2).mean()
            critic_loss = F.mse_loss(self.critic(states),returns)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

        
class PPOContinuous():

    def __init__(
            self,
            state_dim=3,
            hidden_dim=128,
            action_dim=1,
            action_bound=2.0,
            actor_lr=3e-4,
            critic_lr=1e-3,
            gamma=0.9,
            lmbda=0.9,
            epochs=10,
            eps=0.2,
            batch_size=128
            ):
        self.actor = PolicyNetContinuous(state_dim,hidden_dim,action_dim,action_bound)
        self.critic = ValueNet(state_dim,hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.batch_size = batch_size

    def take_action(self,obs,deterministic=False):
        obs = torch.tensor(obs,dtype=torch.float32)
        with torch.no_grad():
            mu,std = self.actor(obs)
            if deterministic:
                action = mu
            else:
                dist = torch.distributions.Normal(mu,std)
                action = dist.sample()
            return action.detach().cpu().numpy()

    def compute_advantage(self,td_delta,dones):
            td_delta = td_delta.squeeze(-1).tolist()
            dones = dones.squeeze(-1).tolist()
            advantage = 0.0
            advantage_list = []
            for delta, done in zip(reversed(td_delta),reversed(dones)):
                advantage = delta + self.gamma * self.lmbda * (1.0 - done) * advantage
                advantage_list.append(advantage)
            advantage_list.reverse()
            advantage_list = torch.tensor(advantage_list,dtype=torch.float32).view(-1,1)
            return advantage_list

    def update(self,states,actions,rewards,next_states,dones,episode_ends):
        states = torch.tensor(np.array(states),dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards),dtype=torch.float32).view(-1,1)
        next_states = torch.tensor(np.array(next_states),dtype=torch.float32)
        dones = torch.tensor(np.array(dones),dtype=torch.float32).view(-1,1)
        episode_ends = torch.tensor(np.array(episode_ends),dtype=torch.float32).view(-1,1)

        with torch.no_grad():
            rewards = (rewards+8.0)/8.0
            old_values = self.critic(states)
            td_target = rewards+self.gamma*self.critic(next_states)*(1-dones)
            td_delta = td_target - old_values
            advantage = self.compute_advantage(td_delta,episode_ends)
            returns = advantage + old_values
            if len(advantage) > 1:
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            mu,std = self.actor(states)
            old_dist = torch.distributions.Normal(mu,std)
            old_log_probs = old_dist.log_prob(actions).sum(dim=-1,keepdim=True)

        dataset = TensorDataset(states, actions, old_log_probs, advantage, returns)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.epochs):
            for b_states, b_actions, b_old_log_probs, b_adv, b_returns in loader:
                mu,std = self.actor(b_states)
                new_dist = torch.distributions.Normal(mu, std)
                new_log_probs = new_dist.log_prob(b_actions).sum(dim=-1, keepdim=True)
                ratio = torch.exp(new_log_probs - b_old_log_probs)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * b_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(self.critic(b_states), b_returns)
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                actor_loss.backward()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.actor_optimizer.step()
                self.critic_optimizer.step()