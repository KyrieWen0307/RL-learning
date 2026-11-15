import torch
import gymnasium as gym
from torch import nn
from torch import optim
import time
from torch.utils.tensorboard import SummaryWriter

learning_rate = 0.001

class PolicyNet(nn.Module):
    def __init__(self, state_dim=4,hidden_dim=128,action_dim=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim,hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,action_dim),
            nn.Softmax(dim=-1),
        )

    def forward(self,x):
        output = self.model(x)
        return output

def take_action(obs,policy_net):
    state = torch.tensor(obs,dtype=torch.float32)
    probs = policy_net(state)
    dist = torch.distributions.Categorical(probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return action.item(),log_prob

def compute_return(rewards,gamma=0.99):
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma*G
        returns.append(G)
    returns.reverse()
    return returns

writer = SummaryWriter('logs')
env = gym.make('CartPole-v1')
policy_net = PolicyNet()
optimizer = torch.optim.Adam(policy_net.parameters(),lr=learning_rate)
score_history = []
for epoch in range(800):
    obs,info = env.reset()
    rewards = []
    log_probs = []
    while True:
        action,log_prob = take_action(obs,policy_net)
        next_obs,reward,terminated,truncated,info = env.step(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        obs = next_obs
        if terminated or truncated:
            break
    returns = torch.tensor(compute_return(rewards),dtype=torch.float32)
    if len(returns)>1:
        returns = (returns - returns.mean())/(returns.std()+1e-8)
    log_probs_tensor = torch.stack(log_probs)
    loss = -(log_probs_tensor*returns).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    score_history.append(sum(rewards))
    writer.add_scalar('episode_score', sum(rewards), epoch)
    if epoch % 20 == 0:
        avg_score = sum(score_history[-20:]) / len(score_history[-20:])
        writer.add_scalar('avg_score_of_20rounds',avg_score,epoch)
        print(epoch,avg_score)


env.close()

eval_env = gym.make('CartPole-v1',render_mode='human')
policy_net.eval()
for ep in range(2):
    obs,info = eval_env.reset()
    with torch.no_grad():
        while True:
            state = torch.tensor(obs,dtype=torch.float32)
            probs = policy_net(state)
            action = torch.argmax(probs).item()
            next_obs,reward,terminated,truncated,info = eval_env.step(action)
            obs = next_obs
            if terminated:
                print('倒地/出界')
                print(f"倒地/出界! 位置 x={obs[0]:.2f}, 角度={obs[2]*180/3.14159:.2f}°")
                break
            if truncated:
                print('截断')
                break
            time.sleep(0.05)

eval_env.close()
writer.close()