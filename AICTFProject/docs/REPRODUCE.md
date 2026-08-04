# Reproducing the results

Every command below is run from the project root with the project virtualenv
active. All randomness is seeded; where a result depends on a seed, the seed is
stated.

## 0. Environment

```bash
python -m venv .venv && . .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pytest tests/ plot/ -q          # no GPU required
```

Reference environment: Python 3.12.4, PyTorch 2.11.0+cu128,
stable-baselines3 2.7.1.

## 1. Protocol constants

Fixed across every run and every method — stated rather than varied:

| Constant | Value |
| --- | --- |
| Map | `map_a` |
| Ruleset | `OURS` |
| Max decision steps | 400 |
| Training seeds | 42, 43, 44 |
| Evaluation episodes | 200 per seed per configuration (600 games aggregated) |

Print the configuration space and its splits:

```bash
python -c "from rl.configuration_space import describe_split; print(describe_split(2))"
```

## 2. Training the baseline ladder

Only the opponent-selection rule differs between modes; architecture,
hyperparameters, and budget are identical.

```bash
# One job at a time. Do not launch a second sweep in parallel -- they contend
# for the same GPU and produce runs at different effective batch sizes.
python rl/run_roastar.py --modes fp,do,pfsp --seeds 42,43,44 \
    --agents 2 --total-steps 1000000 --n-envs 32 --n-steps 512

# ROA-Star (PFSP + goal-conditioned exploiters)
python rl/run_roastar.py --modes pfsp_exploiter --seeds 42,43,44 \
    --agents 2 --total-steps 1000000 --n-envs 32 --n-steps 512

# Repeat with --agents 3 and --agents 4 for the scalability study.
```

Self-play and curriculum-league baselines come from the ablation matrix:

```bash
python rl/run_ablations.py            # ours / no_league / no_curriculum / no_shaping
```

### 2a. Validate before reporting — do not skip

```bash
python rl/verify_league_runs.py --checkpoint-root checkpoints_sb3 --strict
```

A run whose sampling rule needs per-opponent results but recorded none ran as a
uniform draw regardless of its label. This check exists because that failure is
otherwise invisible: PFSP with no recorded results silently *is* fictitious play.
Any run that fails here must not appear in a table under its nominal method name.

## 3. Performance and generalization

```bash
python plot/eval_generalization.py \
    --settings 2v2,3v3,4v4 --seeds 42,43,44 --episodes 200 \
    --out csv/generalization.csv \
    --per-seed-out csv/generalization_per_seed.csv \
    --gap-out csv/generalization_gap.csv
```

Produces, per method and team size:

- **performance** — mean match score on `C_seen`
- **generalization** — mean match score on `C_heldout` (disjoint by construction)
- **generalization gap** — the difference, with a bootstrap CI over training seeds

Add `--list` to print the protocol and the checkpoints that would be used without
running anything.

## 4. Exploitability

```bash
python plot/eval_exploitability.py \
    --checkpoint-dir checkpoints_sb3/2v2 \
    --exploiter-steps 300000 --exploiter-seeds 0,1,2 \
    --eval-episodes 200 --n-envs 8 \
    --out csv/exploitability_2v2.csv \
    --curve-out csv/exploitability_curves_2v2.csv
```

The oracle is controlled so numbers are comparable across policies, not merely
defined for each one:

- identical exploiter architecture and PPO hyperparameters for every target
- identical training budget
- three oracle initializations per target (one unlucky run understates the result)
- validation seeds shared across every target and every initialization
- **peak validation** match score — never the final checkpoint's, never a
  training win rate
- the blue target frozen throughout

Report as an approximate **lower bound**.

## 5. Cross-play

```bash
python plot/eval_crossplay.py --checkpoint-dir checkpoints_sb3/2v2 --seeds 42,43,44
```

## 6. Statistical protocol

- **Common random numbers.** Every method faces the identical episode seed list
  for a given configuration; each configuration gets a disjoint seed block
  (`configuration_space.episode_seeds`), so no two configurations share an episode
  and no method draws an easier set.
- **Uncertainty across training seeds, not episodes.** Point estimates are the
  mean of per-seed means; intervals come from a paired bootstrap over shared
  episode indices. A binomial interval over pooled episodes would treat runs as
  independent draws and understate variance.
- **Checkpoint selection.** The final checkpoint is evaluated, not the best-of-
  training checkpoint. Per-seed training curves are released alongside.

## 7. Checkpoints

Checkpoints are not in git. Fetch the release assets, or retrain with §2:

```bash
# python tools/download_checkpoints.py --setting 2v2      # once assets are published
```

Expected layout:

```
checkpoints_sb3/
  2v2/  final_ppo_roastar_{fp,do,pfsp}_2v2_seed{42,43,44}.zip
        final_ppo_ablate_{ours,no_league,no_curriculum,no_shaping}_seed*_2v2.zip
  3v3/  ...
  4v4/  ...
```

## Known limitations

- Three training seeds, not five. Seed spread is reported so the reader can judge.
- Exploitability is a lower bound at the stated oracle budget.
- Zero-shot team-size transfer is not evaluated and not claimed; observation and
  action spaces are team-size dependent, so the 2v2/3v3/4v4 results are a
  scalability study.
- Blue executes centrally. Decentralized execution under communication limits is
  future work.
