from agent import ReplayBuffer, DDPG
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import time, torch



writer = SummaryWriter('logs/ddpg_pendulum-v1')
env = gym.make('Pendulum-v1',render_mode = None)
agent = DDPG()
replay_buffer = ReplayBuffer()
minimal_size = 1000
batch_size = 64
score_history = []

for epoch in range(200):
    obs, info = env.reset()
    ep_score = 0.0
    while True:
        action = agent.take_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        replay_buffer.add(obs,action,reward,next_obs,float(terminated))
        obs = next_obs
        ep_score +=reward
        if replay_buffer.size() > minimal_size:
            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
            agent.update(b_s,b_a,b_r,b_ns,b_d)
        if terminated or truncated:
            break

    score_history.append(ep_score)
    writer.add_scalar('episode_score',ep_score,epoch)
    if (epoch + 1) % 10 == 0:
        avg_10 = sum(score_history[-10:]) / len(score_history[-10:])
        print(f"Epoch {epoch + 1:3d} | 最近10局均分: {avg_10:.2f}")
    if epoch % 10 == 0:
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