# Network Architecture (PPO / CTF)

This document describes the policy network used for training and evaluation (2v2 and 4v4) in this project.

---

## Overview

- **Algorithm:** PPO (Proximal Policy Optimization).
- **Policy:** `MaskedMultiInputPolicy` (SB3 `MultiInputActorCriticPolicy` with action masking for invalid macro/target choices).
- **Observation:** Dict with `grid` (spatial, per-agent) and `vec` (vector, per-agent). Treated as a **tokenized** sequence of M agents; the same feature extractor is applied per token and outputs are concatenated.

---

## Observation Space

| Key    | Shape (2v2)   | Shape (4v4)   | Description |
|--------|---------------|---------------|-------------|
| `grid` | (M, 7, 20, 20)| (M, 7, 20, 20)| Spatial map per agent: M = 2 or 4, 7 channels, 20×20 cells. Channels encode self, teammates, enemies, flags, etc. |
| `vec`  | (M, 12)       | (M, 12)       | Normalized vector features per agent (e.g. positions, headings, speeds). |
| `mask` | (M×(5+8),)    | (M×(5+8),)    | Action mask: 1 = valid, 0 = invalid (e.g. dead agents). |

- **Grid:** `NUM_CNN_CHANNELS = 7`, `CNN_ROWS = CNN_COLS = 20` (from `game_field_gpu.py`).
- **Vec:** 12 floats per agent.

---

## Feature Extractor: TokenizedCombinedExtractor

- **Role:** Map dict observation to a single feature vector for the policy and value heads.
- **Per-token processing (shared across agents):**
  - **Grid (per token):** One **NatureCNN** over a single grid `(C, H, W) = (7, 20, 20)`:
    - Conv2d: 32 filters, kernel 8×8, stride 4 → ReLU
    - Conv2d: 64 filters, kernel 4×4, stride 2 → ReLU
    - Conv2d: 64 filters, kernel 3×3, stride 1 → ReLU
    - Flatten → Linear → **256** (configurable `cnn_output_dim`; default **256**).
  - **Vec (per token):** No extra layers; raw vec (12 dims) per agent.
- **Aggregation:** For M agents: concat `[CNN_1, …, CNN_M, vec_1, …, vec_M]` → feature dim = **M×256 + M×12** (no context in current setup).
  - 2v2: 2×256 + 2×12 = **536**.
  - 4v4: 4×256 + 4×12 = **1072**.
- **Optional:** `agent_mask` zeros out inactive agents (e.g. for zero-shot 2v2→4v4).
- **Config:** `cnn_output_dim=256`, `normalized_image=True` (in `rl/train_ppo.py`).

---

## MLP (Actor–Critic)

- **net_arch:** `dict(pi=[256, 256], vf=[256, 256])` (shared feature extractor; separate policy and value MLPs).
- **Policy (pi):** features → 256 → 256 → action logits.
- **Value (vf):** features → 256 → 256 → scalar value.

---

## Action Space

- **Type:** `MultiDiscrete`: each of M agents has **1 macro** (5 options) + **1 target** (8 options).
  - 2v2: `[5, 8, 5, 8]` (2 agents × (macro + target)).
  - 4v4: `[5, 8, 5, 8, 5, 8, 5, 8]`.
- **Masking:** Invalid actions (e.g. dead agents, invalid targets) are masked with −1e8 on logits before the softmax in the policy.

---

## Summary Table

| Component        | Specification |
|-----------------|----------------|
| **Feature extractor** | TokenizedCombinedExtractor |
| **CNN (per agent)**  | NatureCNN: 3 conv layers → 256-d |
| **CNN input**        | (7, 20, 20) per agent |
| **Vec (per agent)**  | 12-d, concatenated |
| **Feature dim (2v2)** | 536 |
| **Feature dim (4v4)**| 1072 |
| **Policy MLP**       | [256, 256] |
| **Value MLP**        | [256, 256] |
| **Action (per agent)** | 1 macro (5) + 1 target (8) |

---

## References in Code

- **Policy / feature extractor:** `rl/train_ppo.py` — `TokenizedCombinedExtractor`, `MaskedMultiInputPolicy`, `policy_kwargs` with `net_arch` and `features_extractor_kwargs`.
- **Observation layout:** `game_field_gpu.py` — `NUM_CNN_CHANNELS`, `CNN_ROWS`, `CNN_COLS`, `_build_grid_obs()`, `_build_vec_obs()`, observation space and action space for `GPUCTFVecEnv`.
- **NatureCNN:** `stable_baselines3.common.torch_layers.NatureCNN` (used inside `TokenizedCombinedExtractor`).
