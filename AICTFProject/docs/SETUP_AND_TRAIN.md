# Setup and train (Colab or local PC)

During the audit/refactor, the default training path in `rl/train_ppo.py` is
the vanilla PPO baseline. The latent team-strategy path remains available behind
`--latent-strategy`, but it is intentionally dormant while the baseline
interfaces are audited and hardened.

The latent path is designed around:

- discrete latent team strategy `z`
- strategy inference from global state only
- shared per-agent policy conditioned on local observation plus `z`
- sparse strategy resampling with persistence regularization

The paper-aligned defaults are `K=4`, `z` sampled once per episode, `lambda_H=0.01`,
and `lambda_P=0.02`. A plain `python rl/train_ppo.py` run now uses the vanilla
baseline; pass `--latent-strategy` only when intentionally running the legacy
latent implementation.

Follow these steps to clone the repo, install dependencies, and start training — no prior knowledge assumed.

---

## Colab vs local: what’s different

| Step | Google Colab | Local PC (Windows) |
|------|--------------|--------------------|
| Open project | New notebook; run cells | Open terminal in project folder (or PowerShell) |
| Clone location | `/content/MultiAgentUAV` | Any folder, e.g. `Desktop\MultiAgentUAV` |
| Commands | Prefix with `!` for shell, `%cd` for cd | Run `python` / `git` directly |
| GPU | Runtime → Change runtime type → GPU | Use a CUDA-capable GPU and drivers |

---

## Step 1: Clone the repo

**Colab** (run in a cell):

```python
!git clone https://github.com/TheK-B1000/MultiAgentUAV.git
%cd /content/MultiAgentUAV
```

**Local PC** (PowerShell or Command Prompt):

```bash
cd C:\Users\YourName\Desktop
git clone https://github.com/TheK-B1000/MultiAgentUAV.git
cd MultiAgentUAV
```

---

## Step 2: Use the `cuda` branch and update it

**Colab**:

```python
!git checkout cuda
!git pull origin cuda
```

**Local PC**:

```bash
git checkout cuda
git pull origin cuda
```

---

## Step 3: Install Python libraries

**Recommended (pinned, matches CI):** from `AICTFProject`, install **PyTorch** for your machine (CPU or CUDA) using [pytorch.org](https://pytorch.org), then:

```bash
cd AICTFProject
pip install -r requirements.txt
```

`requirements.txt` pins `gymnasium`, `stable-baselines3`, and support libraries; it does **not** pin `torch` (install that separately so you can pick CPU vs CUDA).

**Legacy one-liner** (older versions; may diverge from CI):

**Colab**:

```python
!pip install -q stable-baselines3==2.3.0 gymnasium==0.29.1 pygame==2.5.2
```

**Local PC** (from repo root or from `AICTFProject`):

```bash
pip install stable-baselines3==2.3.0 gymnasium==0.29.1 pygame==2.5.2
```

*(Optional: use a virtual environment, e.g. `python -m venv .venv` then `.venv\Scripts\activate` on Windows.)*

---

## Step 4: Check GPU (optional but recommended)

**Colab**:

```python
import torch
print(torch.__version__)
print("CUDA available:", torch.cuda.is_available())
```

**Local PC** (same code in a script or `python -c "..."`):

```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

- **Colab:** If CUDA is `False`, set Runtime → Change runtime type → **GPU**.
- **Local:** If CUDA is `False`, install a CUDA-enabled PyTorch from [pytorch.org](https://pytorch.org) or run on CPU (slower).

---

## Step 5: Go to the project folder and set import path (Colab only)

**Colab** — run this so `rl` and other modules are found:

```python
%cd /content/MultiAgentUAV/AICTFProject
%env PYTHONPATH=/content/MultiAgentUAV/AICTFProject
!ls rl
```

You should see files like `train_ppo.py`, `curriculum.py`, etc. under `rl`.

**Local PC** — just go to the project folder. No `PYTHONPATH` needed if you run from `AICTFProject`:

```bash
cd AICTFProject
dir rl
```

---

## Step 5b: Run unit tests (official test runner)

This repository’s tests live in `AICTFProject/tests/` and are written for the **Python standard library `unittest`** (not `pytest`).

**Official command** (from `AICTFProject`, same as training):

```bash
cd AICTFProject
python -m unittest discover -v tests
```

**Quieter output (suppresses TensorFlow/oneDNN info lines from TensorBoard):** run the same suite via the helper (sets `TF_*` env vars before any imports):

```bash
cd AICTFProject
python run_tests.py
```

- If you run `pytest` and see `No module named pytest`, that only means you have not installed `pytest`. The project does **not** require it unless you choose to add it.
- **Optional:** `pip install pytest` works too; `pytest` can discover and run most `unittest`-style tests, but the supported workflow above is `unittest` only.

**What runs in CI:** On every push/PR, GitHub Actions installs CPU PyTorch + `requirements.txt` and runs `python run_tests.py`, including two **short `train_ppo` smokes** (vanilla PPO and latent strategy) to catch integration breaks.

---

## Step 6: Choose what to train and run

All training commands are run **from `AICTFProject`** (Colab: after the `%cd` in Step 5; local: after `cd AICTFProject`).

### Paper-aligned latent strategy runs

Vanilla audit baseline:

```bash
python rl/train_ppo.py --mode FIXED_OPPONENT --fixed-opponent OP3 --max-blue-agents 2 --run-tag ppo_cnn_fixed_op3_2v2 --no-latent-strategy
```

Explicit latent run:

```bash
python rl/train_ppo.py --mode FIXED_OPPONENT --fixed-opponent OP3 --max-blue-agents 2 --run-tag marl_latent_2v2 --latent-strategy --latent-k 4 --latent-lam-p 0.02
```

Sparse-refresh variant (only if you are explicitly studying strategy switching):

```bash
python rl/train_ppo.py --mode FIXED_OPPONENT --fixed-opponent OP3 --max-blue-agents 2 --run-tag marl_latent_refresh_2v2 --latent-strategy --latent-k 4 --latent-lam-p 0.02 --latent-resample-n 20
```

Paper note:

- `--latent-k` is intentionally limited to `4` or `6`
- `--latent-resample-n 1` is intentionally disallowed because per-timestep resampling is not paper-aligned
- use `--latent-strategy` only when you intentionally want the legacy latent path
- curriculum aliases still exist for older experiments, but they are not the intended configuration for the latent-strategy paper

**Colab** — prefix the command with `!`:

```python
!python rl/train_ppo.py --mode CURRICULUM_LEAGUE --max-blue-agents 2 --run-tag ppo_league_2v2
```

**Local PC** — run without `!`:

```bash
python rl/train_ppo.py --mode CURRICULUM_LEAGUE --max-blue-agents 2 --run-tag ppo_league_2v2
```

---

## Example: 2v2, 4v4, 8v8 (League, Paper, Self-play)

Replace `!` with nothing on local PC.

### 2v2 (4M steps, saves in `checkpoints_sb3_2v2/`)

```bash
# League
!python rl/train_ppo.py --mode CURRICULUM_LEAGUE --max-blue-agents 2 --run-tag ppo_league_2v2

# Paper
!python rl/train_ppo.py --mode CURRICULUM_NO_LEAGUE --max-blue-agents 2 --run-tag ppo_paper_2v2

# Self-play
!python rl/train_ppo.py --mode SELF_PLAY --max-blue-agents 2 --run-tag ppo_selfplay_2v2
```

### 4v4 (6M steps, saves in `checkpoints_sb3_4v4/`)

```bash
# League
!python rl/train_ppo.py --mode CURRICULUM_LEAGUE --max-blue-agents 4 --run-tag ppo_league_4v4

# Paper
!python rl/train_ppo.py --mode CURRICULUM_NO_LEAGUE --max-blue-agents 4 --run-tag ppo_paper_4v4

# Self-play
!python rl/train_ppo.py --mode SELF_PLAY --max-blue-agents 4 --run-tag ppo_selfplay_4v4
```

### 8v8 (8M steps, saves in `checkpoints_sb3_8v8/`)

```bash
# League
!python rl/train_ppo.py --mode CURRICULUM_LEAGUE --max-blue-agents 8 --run-tag ppo_league_8v8

# Paper
!python rl/train_ppo.py --mode CURRICULUM_NO_LEAGUE --max-blue-agents 8 --run-tag ppo_paper_8v8

# Self-play
!python rl/train_ppo.py --mode SELF_PLAY --max-blue-agents 8 --run-tag ppo_selfplay_8v8
```

---

## Latent team strategy (ICRA): paper vs. this repository

Use this as the **consistency check** between the abstract design and the code you ship. Wording in the paper should follow the *implementation* column, not a simplified sketch.

| Topic | Paper-level idea | This codebase |
| --- | --- | --- |
| **Strategy `z`** | Discrete team intent, `z ∈ {1,…,K}` (or 0..K-1) | `K` is `--latent-k` ∈ `{4,6}`. `z_idx` / `z_onehot` in the dict obs. |
| **Inference `q_φ(z \| s_g)`** | MLP: global features → logits | `StrategyEncoder` in `rl/latent_marl.py`: 128–128–`K` (ReLU), input dim `GLOBAL_STATE_DIM` (14). See `rl/global_state.py` (`build_global_state_batch`). |
| **What `s_g` contains** | Team geometry, flag distances, capture flags, motion stats, etc. | **14** structured scalars: blue/red mean positions, blue/red position std, min distances to enemy flags, flag captured indicators, and average team speeds. |
| **Decentralized policy** | `π_θ(a_i \| o_i, z)` | Shared actor: **CNN** on each agent’s local `grid` (via `CNNEncoder` + vector `vec`) **concatenated** with `nn.Embedding(K, d_z)` (`d_z=16` default), then MLP → logits. The policy does **not** take raw `global_state` in the actor path. |
| **Centralized critic (CTDE)** | `Q(s_g, a, z)` or value conditioned on joint action | MLP on `[global_state, joint_action_onehot, z_onehot]` → scalar; trained with PPO’s **MSE to Monte Carlo returns** (`rl/latent_marl.py` `q_mlp`, `LatentStrategyPPO`). It is *critic-style* and action-conditioned, implemented inside **PPO** (not a separate off-policy `Q` learner). |
| **When `z` is sampled** | Once per episode and/or every `N` steps, not every timestep | `LatentStrategyVecEnvWrapper` (`rl/latent_vec_env.py`): `resample_every_n=0` → sample at episode start; `N≥2` → sparse refresh; `N=1` is **rejected** at init. |
| **Training objective** | `L_MARL + λ_p L_persist + λ_H H(z)` (sign conventions as in your draft) | `LatentStrategyPPO`: PPO loss + `latent_lam_p *` persistence term + **minus** `latent_lam_h *` mean entropy of `Categorical(logits from strategy_encoder(global_state))`. |
| **Persistence** | Penalize unnecessary switches | Implemented as a **differentiable** proxy `E[1 - p(z_{t-1} \| s_t)]` on resample steps (`expected_strategy_switch_penalty`), not a raw `1[z ≠ z']` through argmax. Describe that explicitly if reviewers ask. |
| **What this is *not*** | Not options / hierarchical RL / scripted switches | Still true; see module docstring in `rl/latent_marl.py` and training flags. |

**Abstract / contribution language:** You can use your one-sentence “strategy modulates coordination through task reward” framing as long as the method section points to: discrete `z`, global-state encoder, shared decentralized actor with `z` embedding, centralized action-conditioned value, PPO with entropy + persistence regularizers, and the **exact** 14-float global-state feature count and CNN-based local encoder.

---

## Quick reference

| Step | Colab | Local PC |
|------|--------|----------|
| 1 | `!git clone ...` then `%cd /content/MultiAgentUAV` | `git clone ...` then `cd MultiAgentUAV` |
| 2 | `!git checkout cuda` and `!git pull origin cuda` | `git checkout cuda` and `git pull origin cuda` |
| 3 | `!pip install -q stable-baselines3==2.3.0 gymnasium==0.29.1 pygame==2.5.2` | `pip install ...` (same packages) |
| 4 | `import torch` and print version/CUDA | Same in terminal with `python -c "..."` |
| 5 | `%cd /content/MultiAgentUAV/AICTFProject`, `%env PYTHONPATH=...`, `!ls rl` | `cd AICTFProject`, `dir rl` |
| 5b | `!python -m unittest discover -v tests` or `!python run_tests.py` | `python -m unittest discover -v tests` or `python run_tests.py` |
| 6 | `!python rl/train_ppo.py --mode ... --max-blue-agents N --run-tag ...` | `python rl/train_ppo.py ...` (same args) |

After training, checkpoints are in `checkpoints_sb3_2v2/`, `checkpoints_sb3_4v4/`, or `checkpoints_sb3_8v8/` (e.g. `final_ppo_league_2v2.zip`).
