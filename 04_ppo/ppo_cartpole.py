import time
import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
from agent import PPO

writer = SummaryWriter('logs/ppo')
env = gym.make('CartPole-v1',render_mode = None)
agent = PPO()
score_history = []

for epoch in range(800):
    obs, info = env.reset()
    states, actions, rewards, next_states, dones = [], [], [], [], []

    while True:
        action = agent.take_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        next_states.append(next_obs)
        dones.append(terminated)
        obs = next_obs
        if terminated or truncated:
            break
    agent.update(states,actions,rewards,next_states,dones)
    score_history.append(sum(rewards))
    writer.add_scalar('episode_score',sum(rewards),epoch)
    if epoch % 20 == 0:
        avg_score = sum(score_history[-20:]) / len(score_history[-20:])
        writer.add_scalar('avg_score_of_20rounds',avg_score,epoch)
        print(epoch,avg_score)

env.close()

eval_env = gym.make('CartPole-v1',render_mode = 'human')
agent.actor.eval()
for ep in range(2):
    obs, info = eval_env.reset()
    with torch.no_grad():
        while True:
            action = agent.take_action(obs,deterministic=True)
            next_obs, reward, terminated, truncated, info = eval_env.step(action)
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