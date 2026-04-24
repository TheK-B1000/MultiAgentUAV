# Setup and train (Colab or local PC)

For the latent team-strategy paper configuration, use the latent strategy path in
`rl/train_ppo.py`. That path is designed around:

- discrete latent team strategy `z`
- strategy inference from global state only
- shared per-agent policy conditioned on local observation plus `z`
- sparse strategy resampling with persistence regularization

The paper-aligned defaults are `K=4`, `z` sampled once per episode, `lambda_H=0.01`,
and `lambda_P=0.02`.

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

Use this install (no `[extra]` to avoid Atari/ale-py issues):

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

## Step 6: Choose what to train and run

All training commands are run **from `AICTFProject`** (Colab: after the `%cd` in Step 5; local: after `cd AICTFProject`).

### Paper-aligned latent strategy runs

Recommended latent run:

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

## Quick reference

| Step | Colab | Local PC |
|------|--------|----------|
| 1 | `!git clone ...` then `%cd /content/MultiAgentUAV` | `git clone ...` then `cd MultiAgentUAV` |
| 2 | `!git checkout cuda` and `!git pull origin cuda` | `git checkout cuda` and `git pull origin cuda` |
| 3 | `!pip install -q stable-baselines3==2.3.0 gymnasium==0.29.1 pygame==2.5.2` | `pip install ...` (same packages) |
| 4 | `import torch` and print version/CUDA | Same in terminal with `python -c "..."` |
| 5 | `%cd /content/MultiAgentUAV/AICTFProject`, `%env PYTHONPATH=...`, `!ls rl` | `cd AICTFProject`, `dir rl` |
| 6 | `!python rl/train_ppo.py --mode ... --max-blue-agents N --run-tag ...` | `python rl/train_ppo.py ...` (same args) |

After training, checkpoints are in `checkpoints_sb3_2v2/`, `checkpoints_sb3_4v4/`, or `checkpoints_sb3_8v8/` (e.g. `final_ppo_league_2v2.zip`).
