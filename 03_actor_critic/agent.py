from network import PolicyNet, ValueNet
import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim

class ActorCritic():
    def __init__(self,
                 state_dim=4,
                 hidden_dim=128,
                 action_dim=2,
                 actor_lr=1e-3,
                 critic_lr=1e-2,
                 gamma=0.98
                 ):
        self.actor = PolicyNet(state_dim,hidden_dim,action_dim)
        self.critic = ValueNet(state_dim,hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(),lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),lr=critic_lr)
        self.gamma = gamma

    def take_action(self,obs,deterministic=False):
        obs = torch.tensor(obs,dtype=torch.float32)
        probs = self.actor(obs)
        if deterministic:
            return torch.argmax(probs).item(),None
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            return action.item(),log_prob

    def update(self,states,log_probs,rewards,next_states,dones):
        states = torch.tensor(np.array(states),dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states),dtype=torch.float32)
        rewards = torch.tensor(rewards,dtype=torch.float32).view(-1,1)
        dones = torch.tensor(dones, dtype=torch.float32).view(-1, 1)
        log_probs = torch.stack(log_probs).view(-1,1)
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states) 
        actor_loss = -(log_probs * td_delta.detach()).mean()
        critic_loss = F.mse_loss(self.critic(states), td_target.detach())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

  