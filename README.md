# 🚀 LunarLander-v2 Solved with Transformer-XL & PPO

![Status](https://img.shields.io/badge/Status-Solved-success)
![Reward](https://img.shields.io/badge/Mean_Reward-+280-brightgreen)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange)
![Working video]([https://img.shields.io/badge/Framework-PyTorch-orange](https://github.com/thesis09/Lunar-Lander/blob/main/Videos/agent-episode-9.mp4))

A custom **Recurrent Reinforcement Learning** agent designed to solve control tasks with partial observability or complex physics. This project implements a **Transformer-XL** memory mechanism within a **Proximal Policy Optimization (PPO)** framework, achieving SOTA-level stability on the `LunarLander-v2` benchmark.

---

## 🧠 The Architecture: "Universal Recurrent Agent"

Unlike standard RL agents that rely on simple MLPs (Multi-Layer Perceptrons) or LSTMs, this agent uses a **Transformer-XL** based recurrent policy. This allows the agent to maintain a longer, more stable history of past states ("Memory"), enabling it to infer hidden physics variables like fuel consumption, momentum, and terrain instability.

### Core Components
1.  **Universal Backbone (The "Eye"):** * A dynamic perception module that automatically detects the input type.
    * **Vector Mode:** Uses a lightweight Identity/Linear encoder for efficiency (used in LunarLander).
    * **Vision Mode:** Switches to **DinoV2 / ViT** for pixel-based environments (scalable design).
2.  **Memory (The "Brain"):**
    * **Transformer-XL:** Processes sequences of state embeddings to capture temporal dependencies.
    * **Sliding Window:** Manages a continuous context window, preventing the "catastrophic forgetting" common in LSTMs.
3.  **Decision Making (The "Controller"):**
    * **PPO (Proximal Policy Optimization):** Uses Generalized Advantage Estimation (GAE) and clipped surrogate objectives for stable training.

---

## 📊 Performance & Results

The agent was trained on `LunarLander-v2` (Continuous/Discrete) and demonstrated a distinct **"Phase Transition"** learning curve.

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Final Mean Reward** | **+280.00** | Standard "Solved" threshold is +200. |
| **Convergence Time** | ~35,000 Steps | Extremely sample-efficient compared to standard PPO baselines (~100k+). |
| **Stability** | High | Agent demonstrates smooth, fuel-efficient landings with zero crashes in final eval. |

### Learning Curve
* **Phase 1 (0 - 20k steps):** *Context Gathering.* The agent explores the physics and fills its memory buffer. Rewards are noisy (-100 to -200).
* **Phase 2 (20k - 35k steps):** *The "Aha!" Moment.* A vertical spike in performance as the Transformer links memory to reward.
* **Phase 3 (35k+ steps):** *Mastery.* Stable convergence at +200 to +280.

---

Access Checkpoints_lunarlander folder where there are checkpoints of this model are saved from this drive link: [Checkpoints_Lunar-Lander](https://drive.google.com/drive/folders/1aRoePZi5LBPzo8jDiDHu962G6MjVyxqA?usp=sharing)


## 🛠️ Installation & Usage

### 1. Prerequisites
```bash
pip install torch gymnasium[box2d] numpy matplotlib
