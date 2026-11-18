import collections, random, copy, torch
import numpy as np  
from network import PolicyNet, QValueNet
import torch.optim as optim
import torch.nn.functional as F

class ReplayBuffer():

    def __init__(self, capacity = 10000):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self,state,action,reward,next_state,done):
        self.buffer.append((state,action,reward,next_state,done))

    def sample(self,batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done =zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)
    
    def size(self):
        return len(self.buffer)


class DDPG():

    def __init__(self, 
                state_dim=3,
                hidden_dim=128,
                action_dim=1,
                action_bound = 2.0,
                actor_lr = 3e-4,
                critic_lr = 3e-3,
                gamma = 0.98,
                tau = 0.005,
                sigma = 0.01
                ):
        self.actor = PolicyNet(state_dim,hidden_dim,action_dim,action_bound)
        self.critic = QValueNet(state_dim,hidden_dim,action_dim)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic = copy.deepcopy(self.critic)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma
        self.action_bound = action_bound
        self.action_dim = action_dim

    def take_action(self, obs, deterministic = False):
        obs = torch.tensor(obs,dtype=torch.float32)
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()
            if not deterministic:
                action += self.sigma * np.random.randn(self.action_dim)
                action = np.clip(action, -self.action_bound, self.action_bound)
            return action

    def soft_update(self,net,target_net):
        for param_target, param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self,states,actions,rewards,next_states,dones):
        states = torch.tensor(states,dtype=torch.float32)
        actions = torch.tensor(actions,dtype=torch.float32).view(-1,self.action_dim)
        rewards = torch.tensor(rewards,dtype=torch.float32).view(-1,1)
        next_states = torch.tensor(next_states,dtype=torch.float32)
        dones = torch.tensor(dones,dtype=torch.float32).view(-1,1)

        with torch.no_grad():
            next_q_values = self.target_critic(next_states,self.target_actor(next_states))
            q_targets = rewards + self.gamma * next_q_values * (1-dones)
        critic_loss = F.mse_loss(self.critic(states,actions), q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        actor_loss = -self.critic(states,self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.soft_update(self.actor,self.target_actor)
        self.soft_update(self.critic,self.target_critic)

class TD3():

    def __init__(self, 
                state_dim=3,
                hidden_dim=128,
                action_dim=1,
                action_bound = 2.0,
                actor_lr = 3e-4,
                critic_lr = 3e-3,
                gamma = 0.98,
                tau = 0.005,
                sigma = 0.01,
                policy_noise = 0.2,
                noise_clip =0.5,
                policy_freq = 2
                ):
        self.actor = PolicyNet(state_dim,hidden_dim,action_dim,action_bound)
        self.critic1 = QValueNet(state_dim,hidden_dim,action_dim)
        self.critic2 = QValueNet(state_dim,hidden_dim,action_dim)
        self.target_actor = copy.deepcopy(self.actor)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(),lr=critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(),lr=critic_lr)
        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma
        self.action_bound = action_bound
        self.action_dim = action_dim
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.total_it = 0

    def take_action(self, obs, deterministic = False):
        obs = torch.tensor(obs,dtype=torch.float32)
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()
            if not deterministic:
                action += self.sigma * np.random.randn(self.action_dim)
                action = np.clip(action, -self.action_bound, self.action_bound)
            return action

    def soft_update(self,net,target_net):
        for param_target, param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self,states,actions,rewards,next_states,dones):
        states = torch.tensor(states,dtype=torch.float32)
        actions = torch.tensor(actions,dtype=torch.float32).view(-1,self.action_dim)
        rewards = torch.tensor(rewards,dtype=torch.float32).view(-1,1)
        next_states = torch.tensor(next_states,dtype=torch.float32)
        dones = torch.tensor(dones,dtype=torch.float32).view(-1,1)
        self.total_it += 1

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.target_actor(next_states) + noise).clamp(-self.action_bound, self.action_bound)
            target_q1 = self.target_critic1(next_states, next_actions)
            target_q2 = self.target_critic2(next_states, next_actions)
            q_targets = rewards + self.gamma * torch.min(target_q1, target_q2) * (1 - dones)

        critic1_loss = F.mse_loss(self.critic1(states, actions), q_targets)
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        critic2_loss = F.mse_loss(self.critic2(states, actions), q_targets)
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic1, self.target_critic1)
            self.soft_update(self.critic2, self.target_critic2)