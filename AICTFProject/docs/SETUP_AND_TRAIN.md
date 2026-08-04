# Setup and train (Colab or local PC)

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

## Ablations (curriculum / league / reward shaping)

Leave-one-out matrix matching the paper revision plan:

| Name | Meaning | Mode | Reward |
|------|---------|------|--------|
| `ours` | Full method | `CURRICULUM_LEAGUE` | `full` |
| `no_league` | −league | `CURRICULUM_NO_LEAGUE` | `full` |
| `no_curriculum` | −curriculum (fixed OP3) | `FIXED_OPPONENT` | `full` |
| `no_shaping` | −dense/PBRS/team shaping | `CURRICULUM_LEAGUE` | `no_shaping` |

**Preview commands (recommended first):**

```bash
python rl/run_ablations.py --dry-run --agents 2
python rl/run_ablations.py --list
```

**Run the full matrix sequentially (one GPU job at a time — recommended):**

```bash
# Stop any other training first, then:
python rl/run_ablations.py --full --agents 2 --total-steps 1000000 --n-envs 4 --resume-oom --skip-finished
```

This queues all 4 arms × seeds 42/43/44 (12 jobs), runs them **one-by-one**, auto-loads `oom_save_*.zip` when present, skips arms that already have `final_*.zip`, and **stops on the first failure** so a later arm does not start after an OOM.

**Subset / multi-seed:**

```bash
python rl/run_ablations.py --only ours,no_shaping --seeds 42,43 --agents 2
```

Preview:

```bash
python rl/run_ablations.py --dry-run --full --n-envs 4 --resume-oom
```

**Single training run with reward ablation:**

```bash
python rl/train_ppo.py --mode CURRICULUM_LEAGUE --max-blue-agents 2 --reward-ablation no_shaping
python rl/train_ppo.py --mode NO_CURRICULUM --fixed-opponent OP3 --max-blue-agents 2
```

Checkpoints use tags like `ppo_ablate_no_league_2v2` (matrix) or `ppo_league_rew_no_shaping_2v2` (direct CLI).

### Shared fixed eval (after finals exist)

Training WRs are not comparable across arms (different opponent mixes). Run one shared OP3/OP4 eval:

```bash
python plot/eval_ablations.py --checkpoint-dir checkpoints_sb3/2v2 --list
python plot/eval_ablations.py --checkpoint-dir checkpoints_sb3/2v2 --episodes 100 \
  --out csv/eval_ablation_2v2.csv --per-seed-out csv/eval_ablation_2v2_per_seed.csv \
  --require-complete
python plot/plot_eval_metrics.py --metrics-csv csv/eval_ablation_2v2.csv --modes 2v2
```

### ROA-Star shared frozen eval (paper centerpiece)

Do **not** use training win rates. Evaluate all nine PFSP finals (2v2/3v3/4v4 × seeds 42/43/44)
against OP3 and OP4 with identical per-episode seeds. Primary metric: Match Score = (W + 0.5D)/(W+L+D).

```bash
python plot/eval_roastar_matrix.py --list --require-complete
python plot/eval_roastar_matrix.py --episodes 1000 --require-complete \
  --out csv/eval_roastar_shared.csv \
  --per-seed-out csv/eval_roastar_shared_per_seed.csv
python plot/plot_roastar_shared_eval.py \
  --metrics-csv csv/eval_roastar_shared.csv \
  --per-seed-csv csv/eval_roastar_shared_per_seed.csv \
  --out-dir figures
```

### Checkpoint tournament / cross-play / exploitability

Paper-revision helpers (VGC-Bench-style selection, method payoff matrix, approximate exploitability). Offline unit tests: `python -m unittest plot.test_paper_eval_tools`.

```bash
# 1) Select a winner among league snapshots for one run_tag
python plot/checkpoint_tournament.py --checkpoint-dir checkpoints_sb3/2v2 \
  --run-tag ppo_league_2v2 --agents 2 --list
python plot/checkpoint_tournament.py --checkpoint-dir checkpoints_sb3/2v2 \
  --run-tag ppo_league_2v2 --agents 2 --val-episodes 50 --cross-episodes 20 \
  --out csv/tournament_ppo_league_2v2.csv

# 2) Cross-play payoff matrix (row=blue, col=red) among named finals
python plot/eval_crossplay.py --checkpoint-dir checkpoints_sb3/2v2 --list
python plot/eval_crossplay.py --checkpoint-dir checkpoints_sb3/2v2 \
  --episodes 100 --seeds 42 --out csv/crossplay_2v2.csv --device cuda

# 3) Approximate exploitability via controlled red best-response oracle
#    (multi-init, peak validation match score — protocol in rl/eval_exploitability.py)
python plot/eval_exploitability.py --checkpoint-dir checkpoints_sb3/2v2 --list
python plot/eval_exploitability.py --checkpoint-dir checkpoints_sb3/2v2 \
  --exploiter-steps 300000 --exploiter-seeds 0,1,2 --n-envs 8 \
  --eval-episodes 200 \
  --out csv/exploitability_2v2.csv \
  --curve-out csv/exploitability_curves_2v2.csv
```

**Caveats (do not skip before citing numbers):**

1. **Retrain ROA-Star PFSP before claiming PFSP results.** Earlier `final_ppo_roastar_pfsp_*` runs recorded empty `win_rate_stats` in `*_league_state.json`, so PFSP weights never updated and sampling stayed uniform (effectively fictitious play, mislabeled). After the `record_result` fix in `train_ppo_roastar.py`, re-run PFSP (and prefer FP/DO too) before using those checkpoints in the paper table.
2. **Discard cross-play / tournament CSVs produced before the `scores_from_info` fix.** `run_two_policy_episodes` could silently return zero episodes → NaN match scores. Re-run `eval_crossplay.py` / `checkpoint_tournament.py` after that fix; do not recycle old matrices.

Reward presets:
- `full` — shaped reward (default)
- `no_shaping` / `sparse` — zero PBRS + team dense bonuses; keep terminal + offense events
- `terminal` — win/lose/draw only

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
