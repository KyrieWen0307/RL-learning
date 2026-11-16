# 03 Actor-Critic 算法（CartPole-v1）

> **工程三层架构**：
> - 模型层：[`network.py`](./network.py)（`PolicyNet` + `ValueNet`）
> - 算法层：[`agent.py`](./agent.py)（`ActorCritic` 双网络 + 双 `Adam` 优化器）
> - 调度层：[`actor_critic_cartpole.py`](./actor_critic_cartpole.py)（环境交互、TensorBoard 记录与 `render_mode="human"` 验收）

---

## 🧠 核心实现要点

- **双网络分工**：
  - **Actor (`PolicyNet`)**：输入 4 维状态，输出动作概率分布并用 `Categorical` 采样。
  - **Critic (`ValueNet`)**：输入 4 维状态，输出单维实数状态价值 $V_\phi(s)$（最后一层不加激活函数）。
- **时序差分目标与误差（1-step TD）**：
  - `td_target = rewards + gamma * critic(next_states) * (1 - dones)`（仅对 `terminated` 清零未来价值）。
  - `td_delta = td_target - critic(states)` 作为单步优势估计 $A(s,a)$。
- **双 `.detach()` 隔离计算图**：
  - Critic 用 `F.mse_loss(critic(states), td_target.detach())` 回归学习打分（半梯度 TD，防止前向状态拉偏目标）。
  - Actor 用 `-(log_probs * td_delta.detach()).mean()` 更新策略（防止 Actor 梯度倒灌篡改 Critic）。

---

## 🔍 实验观察与第 4 章过渡动机

- 在 `actor_lr=2e-3, critic_lr=1e-2, gamma=0.98` 下，20 局滑动均分在第 580 轮达到 **470.55**，并在 `deterministic=True` 贪心验收中触发 **500 步 `truncated`** 通关。
- 同时观察到了基础版 1-step Actor-Critic 的典型局限：高分阶段由于缺少策略更新幅度限制且存在 1-step 自举偏差，训练后期出现了从 **470.55** 回落至 **118.1** 的策略塌方震荡——这正是第 4 章引入 **PPO（GAE + Clip 截断）** 要解决的核心问题。
