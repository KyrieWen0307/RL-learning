import gymnasium as gym
import time

def compute_return(rewards,gamma=0.99):
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma*G
        returns.append(G)
    returns.reverse()
    print("最后3步:", returns[-3:])
    return returns

env = gym.make('CartPole-v1',render_mode='human')
print('观察空间：',env.observation_space)
print('动作空间:',env.action_space)
obs,info=env.reset(seed=42)
rewards = []
while True:
# 杆子向右歪(theta>0)就向右推(1)，否则向左推(0)
    action = 1 if (0.05 * obs[0] + 0.1 * obs[1] + obs[2] + 0.2 * obs[3]) > 0 else 0
    next_obs,reward,terminated,truncated,info = env.step(action)
    rewards.append(reward)
    obs = next_obs
    if terminated:
        print('倒地/出界')
        print(f"倒地/出界! 位置 x={obs[0]:.2f}, 角度={obs[2]*180/3.14159:.2f}°")
        break
    if truncated:
        print('截断')
        break
    time.sleep(0.05)
env.close()

returns = compute_return(rewards)
print(len(rewards))
print(sum(rewards))
print(returns[0])
print(rewards[0],returns[0],rewards[1],returns[1],rewards[2],returns[2])




