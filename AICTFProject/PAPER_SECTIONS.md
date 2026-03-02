# Paper-ready paragraphs (OP4 vs OP3, network & hyperparameters)

---

## 1. OP4 vs OP3 and why evaluation against OP4 is challenging

We evaluate generalization using a **held-out scripted opponent (OP4)** that is **never used during training**. Training uses opponents OP1, OP2, and OP3 only; OP4 is reserved for test. OP4 is designed to be both **behaviorally distinct** and **harder** than OP3. Unlike OP3, which uses a medium attacker and medium defender (balanced play, defender_style = 1) with moderate role-switching (0.35) and moderate speed and deception, OP4 uses a medium attacker and an **easy defender** (defender_style = 0), so the red team commits more to attack and behaves in a more rusher-like, aggressive way. OP4 also has **higher role-switch probability** (0.50 vs 0.35), **higher speed multipliers** (e.g. 0.92–1.28 vs 0.75–1.25 for 2v2), and **higher deception bounds** (e.g. 0.22–0.48 vs 0.10–0.35 for 2v2). Because the policy never sees OP4 during training, facing OP4 only at test time is challenging: the agent must generalize to a faster, more deceptive, and more attack-oriented opponent without any direct experience, which tests robustness and transfer rather than memorization of a single adversary.

---

## 2. Network architecture and training hyperparameters (PPO)

**Network.** We use PPO with a custom **tokenized** feature extractor and shared policy/value MLPs. The observation is a dict: a spatial **grid** of shape (M, 7, 20, 20) and a **vector** of shape (M, 12) per agent (M = 2 for 2v2, 4 for 4v4). The feature extractor (**TokenizedCombinedExtractor**) processes each of the M agents with the same **NatureCNN**: three convolutional layers (32 filters 8×8 stride 4, 64 filters 4×4 stride 2, 64 filters 3×3 stride 1, each followed by ReLU), then a linear layer to 256 dimensions per agent. The 12-d vector per agent is concatenated without extra layers. The outputs are concatenated across agents, giving a feature dimension of **536 for 2v2** (2×256 + 2×12) and **1072 for 4v4** (4×256 + 4×12). The policy and value heads each use an MLP with two hidden layers of **256 units** (net_arch: pi = [256, 256], vf = [256, 256]). The action space is **MultiDiscrete**: each agent chooses one **macro** (5 options) and one **target** (8 options); invalid actions are masked before the policy softmax.

**Hyperparameters (specific values).** When using the curriculum (including League and Paper modes), we use the stable MARL PPO variant: **learning_rate = 1.5×10⁻⁴**, **ent_coef = 0.005**, **clip_range = 0.12**, **n_epochs = 4**, **batch_size = 1024**. For 4v4 we multiply learning rate by 0.75 (**1.125×10⁻⁴**). Other PPO parameters: **gamma = 0.995**, **gae_lambda = 0.95**, **vf_coef = 0.5**, **max_grad_norm = 0.5**, **n_steps = 2048** (or 1024 when n_envs or n_steps are reduced for 4v4/8v8 memory), **n_envs = 4** (or 2 for 4v4/8v8), **seed = 42**. Total training: **total_timesteps = 4,000,000** for 2v2 (with curriculum); 4v4 uses **6,000,000** and 8v8 **8,000,000**. Feature extractor: **cnn_output_dim = 256**, **normalized_image = True**. The grid has **7 channels** and **20×20** cells; the vector has **12** dimensions per agent.

---

## 3. Actions available to the agents

Each agent has a **discrete action space** composed of two choices per decision step. The first is a **macro action** (5 options): (0) **go to target** — move toward the waypoint selected by the second part of the action; (1) **GRAB_MINE** — collect a mine charge from a pickup; (2) **go to enemy flag** — navigate to the opponent’s flag (e.g. to capture it); (3) **PLACE_MINE** — place a mine at the agent’s current position if carrying a charge; (4) **go home** — return to the team’s own flag (e.g. after grabbing the enemy flag). The second is a **target** (8 options): a choice among eight fixed waypoints on the map (e.g. mid-field, left/right, near flags) that define where “go to target” directs the agent. The combined action is thus **MultiDiscrete**: one macro (5) and one target (8) per agent. Invalid actions (e.g. for disabled agents) are masked so the policy only samples from valid choices.

---

## 4. PPO setup, training length, and baselines (SEA-GUARD)

We use the PPO implementation from **Stable-Baselines3** (SB3) in Python. Training is run for **4 million** timesteps (2v2 curriculum); the **best** or final model is saved for evaluation. We compare **SEA-GUARD** against two baselines: (1) a PPO-based approach for the same problem presented in [4], and (2) a **self-play** baseline. The self-play baseline trains the agent against **past snapshots of itself** (checkpoints saved periodically during training) rather than against fixed scripted opponents, so the opponent distribution evolves with the policy and encourages robustness to a changing adversary.

---

*Source: opponent_params.py (OP3/OP4), rl/train_ppo.py (PPOConfig, policy_kwargs, PPO build, TrainMode.SELF_PLAY), game_field_gpu.py (obs space, n_macros, n_targets, _build_blue_targets_from_action), NETWORK_ARCHITECTURE.md.*
