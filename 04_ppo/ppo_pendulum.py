import time
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
from agent import PPOContinuous

writer = SummaryWriter('logs/ppo_pendulum-v2')
env = gym.make('Pendulum-v1',render_mode = None)
agent = PPOContinuous()
score_history = []

states, actions, rewards, next_states, dones, episode_ends = [], [], [], [], [], []
for epoch in range(2000):
    obs, info = env.reset()
    ep_score = 0.0
    while True:
        action = agent.take_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        states.append(obs)
        next_states.append(next_obs)
        rewards.append(reward)
        dones.append(float(terminated))
        actions.append(action)
        obs = next_obs
        ep_score += reward
        episode_ends.append(float(terminated or truncated))
        if terminated or truncated:
            break
    score_history.append(ep_score)
    writer.add_scalar('episode_score', ep_score, epoch)
    if (epoch + 1) % 10 == 0:
        agent.update(states, actions, rewards, next_states, dones, episode_ends)
        states, actions, rewards, next_states, dones, episode_ends = [], [], [], [], [], []
    if epoch % 50 == 0:
        avg_score = sum(score_history[-50:]) / len(score_history[-50:])
        writer.add_scalar('avg_score_of_50rounds', avg_score, epoch)
        print(epoch,avg_score)
    
env.close()

eval_env = gym.make('Pendulum-v1',render_mode = 'human')
agent.actor.eval()
for ep in range(2):
    obs, info = eval_env.reset()
    ep_score = 0.0
    with torch.no_grad():
        while True:
            action = agent.take_action(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = eval_env.step(action)
            ep_score += reward
            obs = next_obs
            if truncated:
                print(f"第 {ep+1} 局结束，总得分: {ep_score:.2f}")
                break
            time.sleep(0.03)

eval_env.close()
writer.close()