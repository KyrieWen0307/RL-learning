# 强化学习核心基础与具身控制（RL-learning）

> **背景定位**：面向机械工程转具身智能/机器人学习方向，衔接【小土堆 PyTorch 基础】与【斯坦福 CS123 四足机器狗强化学习（第 6 章）】的专项突击仓库。
>
> **学习路线（严格对齐 Datawhale 教程 1.3 推荐主线 + 《动手学强化学习》PyTorch 实战）**：
> ```text
> MDP 基础 (01) → 策略梯度 (02) → Actor-Critic (03) → PPO 必学主线 (04，直通四足机器狗)
>                                                    ↘ DDPG/TD3 支线 (05，衔接机械臂连续控制)
> ```
>
> **参考资料黄金组合**：
> - **理论主线大纲**：Datawhale《Dive into Embodied AI》强化学习基础篇
> - **PyTorch 实战蓝本**：上交大《动手学强化学习》（Hands-on RL，https://hrl.boyuai.com/ ）——极简纯 PyTorch 代码（每章 60~90 行，适配新版 `gymnasium` 接口）

---

## 🗺️ 章节目录与实战对照表

| 章节目录 | Datawhale 理论节点 | 对应《动手学强化学习》实战 | 核心攻克目标 & 亲手敲的实战项目 | 小土堆 PyTorch 概念映射 |
| :--- | :--- | :--- | :--- | :--- |
| [`01_mdp_and_gym/`](./01_mdp_and_gym/) | **MDP 基础** | **第 3 章**：马尔可夫决策过程 | • 状态 $s$、动作 $a$、奖励 $r$、折扣回报 $G_t$<br>• 状态价值 $V(s)$、动作价值 $Q(s,a)$ 与贝尔曼方程直觉<br>• **实战**：用 `gymnasium` 跑通 `CartPole-v1`（倒立摆）控制闭环，手算折扣回报 $G_t$ | **动态 `Dataset`**：<br>环境（Env）就是边交互边产数据的动态数据源 |
| [`02_policy_gradient/`](./02_policy_gradient/) | **策略梯度** | **第 9 章**：策略梯度算法 | • 直接用神经网络学策略 $\pi_\theta(a\mid s)$<br>• 策略梯度定理直觉与 REINFORCE 算法<br>• **实战**：手写 `PolicyNet`，让 `CartPole-v1` 倒立摆从坚持 20 步训练到 **500 步满分不倒** | **单 `nn.Module` + 自定义 `Loss`**：<br>`loss = -(log_prob * G_t).mean()`，让高分动作概率变大 |
| [`03_actor_critic/`](./03_actor_critic/) | **Actor-Critic** | **第 10 章**：Actor-Critic 算法 | • 引入评委网络 Critic $V_\phi(s)$ 实时估分（时序差分 TD）<br>• 时序差分误差（TD Error）与优势函数 $A(s,a)$<br>• **实战**：手写 `PolicyNet` + `ValueNet` 双网络协同训练闭环 | **双 `nn.Module` + 双 `Optimizer`**：<br>Critic 用 `F.mse_loss` 学打分；Actor 根据 Critic 算出的优势值更新 |
| [`04_ppo/`](./04_ppo/) | **PPO（必学主线）** | **第 12 章**：PPO 算法 | • 重要性采样比率 $r_t(\theta)$ 与广义优势估计（GAE）<br>• **PPO-Clip 截断损失**：把更新幅度锁在 $[1-\epsilon, 1+\epsilon]$ 安全区内<br>• **实战**：手写离散/连续双版本 PPO（搞定 `CartPole-v1` 与 `Pendulum-v1` 连续力矩控制），直通 CS123 四足狗 | **`torch.clamp` + 多 Epoch 切片复用**：<br>在一批轨迹数据上安全更新多次 |
| [`05_ddpg_branch/`](./05_ddpg_branch/) | **↘ DDPG 支线** | **第 13 章**：DDPG 算法 | • On-policy（PPO）vs Off-policy（DDPG）样本效率对比<br>• 确定性策略 $\mu_\theta(s)$ 与经验回放池 `ReplayBuffer`<br>• **实战**：手写 `ReplayBuffer` + `-critic(s, actor(s)).mean()` + 软更新，跑通 `Pendulum-v1`，衔接独立 Reacher 机械臂项目 | **手写 `ReplayBuffer` + 借道求导**：<br>冻结 Critic 权重，把 Actor 的输出直接喂给 Critic 求梯度 |

---

## 🛠️ 本地开发环境约定

- **Python 解释器**：`D:\Miniconda3\envs\pytorch\python.exe`（Miniconda `pytorch` 虚拟环境）
- **深度学习框架**：PyTorch `2.15.0+cpu`
- **仿真交互接口**：`gymnasium 1.4.0` + `pygame 2.6.1` + `mujoco 4.0.1`

---

## 📏 学习铁律与工作流

1. **绝不代写代码**：所有 `.py` 脚本均由自己亲手敲入，AI 助教仅负责用“小土堆 PyTorch 概念体系”拆解公式、调用 `generative_ui` 提供交互式可视化图解，并进行 Code Review。
2. **一章一对话**：每个模块开启独立对话，保持上下文清爽聚焦，每完成一章更新本 `README.md` 的打卡进度。

### ✅ 进度打卡
- [x] `00` 工作区初始化、路线对齐与 `gymnasium` / `mujoco` 环境安装
- [x] `01_mdp_and_gym`：MDP 核心概念、贝尔曼方程与 Gymnasium 接口实操
- [x] `02_policy_gradient`：策略梯度（REINFORCE）原理与手写实现
- [x] `03_actor_critic`：Actor-Critic 双网架构与优势函数手写实现
- [ ] `04_ppo`：GAE 广义优势估计与 PPO-Clip 完整手写实现（衔接 CS123 四足狗）
- [ ] `05_ddpg_branch`：Off-policy 支线、ReplayBuffer 与 DDPG 核心实现
