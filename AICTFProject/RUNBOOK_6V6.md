# 6v6 Pipeline Runbook

Full pipeline: specialist training -> distillation state collection -> Rung-1 latent-policy
training -> sealed crossover evaluation -> deployment robustness. Scaling/robustness
validation of an already-proven mechanism -- not rediscovery of it.

## 0. Repo commit to use

**IMPORTANT -- action needed before this runbook is usable on a second machine.** As of
writing, all the code and specs this pipeline depends on are **uncommitted working-tree
changes** on branch `school-testing`, on top of commit:

```
c09cb661b5ed344967e36fa91992a180d40734ad
```

That commit alone does NOT contain this pipeline. Before moving to the other PC, the current
working tree must be committed (and the commit made available to that machine -- push to a
remote, or copy the repo directly). This runbook will be accurate once that commit exists;
substitute its hash below once you have it.

```bash
git rev-parse HEAD   # run this on the PC that has the working tree, AFTER committing,
                      # and use that hash on the second machine.
```

## 1. Environment / setup

On the second machine:

```bash
git clone <this repo>          # or copy the working tree directly
cd AICTFProject
git checkout <commit hash from step 0>
```

Create/verify the Python venv (must be named `.venv` in the repo root -- every command below
assumes `./.venv/Scripts/python.exe`; this project's own convention is to NEVER use a bare
`python`/PATH interpreter):

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt   # or however deps are pinned
```

Verify CUDA is visible (the pipeline driver does this automatically as its own Stage 0, but
check manually first if anything looks wrong):

```bash
./.venv/Scripts/python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 2. Required input files (already in the repo at the pinned commit)

All of these are frozen specs the pipeline reads from and fails closed if any is missing:

- `artifacts/strategic_demand/sppo/STRATEGIC_DEMAND_6v6_GUARD_DISTRIBUTED_V2_CERTIFICATION.json` -- 6v6 strategic demand, CERTIFIED
- `artifacts/strategic_demand/sppo/SCALE_6V6_PRODUCTION_SPECIALIST_SEEDS.json` -- stage 1 seeds
- `artifacts/strategic_demand/sppo/TEACHER_DISTILLATION_6V6_SPEC.json` -- stage 2 dataset size/schedule
- `artifacts/strategic_demand/sppo/TEACHER_DISTILLATION_6V6_SEED_RETIREMENT_AMENDMENT.json` -- stage 2's GOVERNING seed block (supersedes the spec above's original block; see section 3's caveat)
- `artifacts/strategic_demand/sppo/RUNG1_CONSTRUCTION_6V6_AMENDMENT.json` -- stage 3 architecture/seeds/paths
- `artifacts/strategic_demand/sppo/RUNG1_6V6_CROSSOVER_EVAL_SPEC.json` -- stage 4 gate/seeds
- `artifacts/strategic_demand/sppo/6V6_PIPELINE_SEED_ALLOCATION.json` -- stages 4-5 seed blocks, disjointness record
- `artifacts/strategic_demand/sppo/DEPLOYMENT_ROBUSTNESS_SPEC.json` -- stage 5 disturbance tiers
- `artifacts/strategic_demand/sppo/DEPLOYMENT_ONLY_GUARANTEE_CHECK.json` -- stage 5 precondition

## 3. Frozen scientific seeds (pre-registered; the pipeline code reads these itself -- listed
   here for reference, never hand-edit them)

| Stage | Purpose | Seeds |
|---|---|---|
| 1 | pi_A_6v6 | 7610001 |
| 1 | pi_B_6v6 | 7620001 |
| 2 | collection, Pole A | **13600003-13600098** (NOT 13600001-096 -- see caveat below) |
| 2 | collection, Pole B | **13600103-13600198** (NOT 13600101-196 -- see caveat below) |
| 3 | Rung-1 init/shuffle | 13621001, 13621002 |
| 4 | crossover eval | 13640001-13640128 |
| 5 | robustness | 13660001-13660128 |

**Caveat on stage 2's seeds:** `TEACHER_DISTILLATION_6V6_SPEC.json`'s *original* block was
13600001-096 / 13600101-196. Seeds 13600001, 13600002, 13600101, 13600102 were accidentally
spent during pipeline-verification (a plumbing smoke ran real episodes with junk 5k-step
checkpoints before `--smoke` mode existed; caught immediately, the resulting data was deleted,
no interpretation ever happened). Those four are permanently retired
(`TEACHER_DISTILLATION_6V6_SEED_RETIREMENT_AMENDMENT.json`). The table above is the corrected,
governing block -- `collect_distillation_states_scale.py`'s real (non-smoke) path reads this
amendment automatically and uses its block; you do not need to pass anything special.

## 4. Start the pipeline

One command, from the repo root:

```bash
bash experiments/run_6v6_pipeline.sh
```

This runs stages 1-5 in order. See "Resuming" below for partial runs.

**Where results land, and what to send back when it's done:** by default, this pipeline's
outputs are scattered across several directories (`artifacts/scale_6v6_specialists/`,
`artifacts/strategic_demand/sppo/...`, `artifacts/strategic_demand/sppo/robustness_eval_rows/`)
-- that matches how every team size in this project has always been organized, but it is NOT
one folder you can just zip. Once the pipeline finishes (or at any point mid-run, to check
progress), run:

```bash
bash experiments/export_6v6_results.sh
```

This COPIES (never moves -- nothing about resuming the pipeline is affected) everything
produced so far into one folder: `artifacts/6v6_results/`, organized as
`specialists/ distillation/ crossover_eval/ robustness/ specs/`. **That one folder is what to
zip and send.** `run_6v6_pipeline.sh` calls this automatically after EVERY stage (not just at
the end), so `artifacts/6v6_results/` is always up to date -- check it any time mid-run for
real progress, not just at completion.

## 5. What each stage does, where it writes, how long it takes

| Stage | Script | Writes | Approx. runtime |
|---|---|---|---|
| 0 | (built into the driver) | nothing -- verification only | seconds |
| 1 | `train_specialist_scale.py` x2 | `artifacts/scale_6v6_specialists/pi_{A,B}_specialist_6v6/ckpts/final_*.zip` | ~5h each, ~10h total (sequential) |
| 2 | `collect_distillation_states_scale.py` | `artifacts/strategic_demand/sppo/teacher_distillation_6v6/states/*.npz` + `TEACHER_DISTILLATION_6V6_DATASET.json` | CPU, likely 1-3h for 192 episodes |
| 3 | `run_ladder_rung1_distillation_scale.py` | `artifacts/strategic_demand/sppo/sharing_ladder_6v6/rung1/ckpts/final_rung1_6v6.pt` + `RUNG1_6V6_STUDENT_FROZEN.json` | supervised, 20 epochs -- minutes, not hours |
| 4 | `eval_rung1_crossover_scaled.py` | `RUNG1_6V6_CROSSOVER_EVAL_RESULT.json` (or `..._INTEGRITY_REQUIRED.json` if a tie/reversal is flagged) | 512 episodes -- several hours (GPU env-stepping is the bottleneck, not policy inference) |
| 5 | `eval_deployment_robustness.py` x4 (one per pole/z cell) | `artifacts/strategic_demand/sppo/robustness_eval_rows/rung1_6v6__6v6__pole{A,B}__z{0,1}__*.csv` (16 files: 4 cells x 4 conditions) | 2048 episodes total -- comparable order to stage 4, likely the longest stage |

Total, sequential on one GPU: **roughly 1.5-2.5 days**. If the other PC can run stage 1 while
this machine's 4v4 job finishes, or vice versa, that is the parallelism the PI asked for --
stages 2-5 all depend on stage 1's checkpoints existing, so true parallelism is at the
across-machine level (this machine does 4v4, the other does 6v6), not within the 6v6 pipeline
itself.

## 6. Reading stage 4's result

```bash
./.venv/Scripts/python.exe -c "
import json
d = json.load(open('artifacts/strategic_demand/sppo/RUNG1_6V6_CROSSOVER_EVAL_RESULT.json'))
print(d['PRIMARY_GATE'])
"
```

`PRIMARY_GATE.passes` is the PASS/FAIL verdict. `delta_A`/`delta_B` each carry `mean`,
`lcb95`, `ucb95`. If instead `RUNG1_6V6_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json` exists, stage 4
detected a tie or reversal and stopped before writing a verdict -- this needs human review
(read the raw rows in `rung1_6v6_crossover_eval_rows.csv`) before anything else proceeds; the
driver will not automatically re-attempt it.

## 7. Reading stage 5's results

Each of the 16 CSVs has one row per episode (`seed,blue,red,win,margin`). Per-condition
Delta_A/Delta_B (matching the 2v2 robustness study's own analysis) are NOT auto-computed by
the driver -- compute them the same way `ROBUSTNESS_2V2_RUNG1_RESULT.json` was: paired
percentile bootstrap (n_boot=20000, alpha=0.05, rng_seed=7) over
`delta_A(q) = win(z0,A,q) - win(z1,A,q)` and `delta_B(q) = win(z1,B,q) - win(z0,B,q)` for each
condition `q`, using the shared 128-seed block so every condition is paired seed-by-seed.

## 8. Resuming after an interruption

Just re-run the same command:

```bash
bash experiments/run_6v6_pipeline.sh
```

Every stage checks for its own completed output FIRST and skips with a log line if already
done -- it will never overwrite a completed scientific result (the underlying Python scripts
also refuse independently; the driver's skip logic is the friendly version of the same rule).
To resume from a specific stage without re-checking earlier ones:

```bash
bash experiments/run_6v6_pipeline.sh --from-stage 3
```

To run exactly one stage:

```bash
bash experiments/run_6v6_pipeline.sh --stage 4
```

If a stage was interrupted mid-run (e.g. machine restart during stage 1's 1M-step training),
its terminal checkpoint will not exist yet, so re-running the driver will re-launch that stage
from scratch (PPO training in this project does not checkpoint-resume mid-run; this is a
pre-existing property of `train_specialist_scale.py`, not something this driver changes).

## 9. GPU contention check

By default each stage refuses to start if `nvidia-smi` reports >20% GPU utilization (assumes
something else is already running). On a machine dedicated to this pipeline, skip the check:

```bash
SKIP_GPU_CHECK=1 bash experiments/run_6v6_pipeline.sh
```

## 10. What this pipeline deliberately does NOT do

Per PI direction: this does not rebuild the 2v2 sharing ladder (Rungs 0/2/3), does not
re-derive why Rung 1 is the right architecture, and stage 4 does not compute a ladder-relative
D_A/D_B against a Rung-0-at-6v6 reference (no such reference exists, and building one would be
exactly the rediscovery work this track skips). Stage 4 is a standalone absolute gate on the
one frozen Rung-1-at-6v6 checkpoint.

## 11. Smoke-testing (already done once on the original machine; repeat here only if you

   change any of the pipeline code)

Every stage supports a `--smoke` mode (collection and Rung-1 training) or accepts disposable
seeds directly (crossover eval, robustness) that never touch the frozen blocks in section 3
and never write to the real one-shot output filenames:

```bash
# stage 2 smoke (uses existing 5k-step non-scientific specialist smokes, not real teachers)
./.venv/Scripts/python.exe experiments/collect_distillation_states_scale.py --team-size 6 --device cpu --smoke \
  --pi-a-path artifacts/scale_6v6_specialists/smoke_pi_A_specialist_6v6/ckpts/final_smoke_pi_A_specialist_6v6.zip \
  --pi-b-path artifacts/scale_6v6_specialists/smoke_pi_B_specialist_6v6/ckpts/final_smoke_pi_B_specialist_6v6.zip

# stage 3 smoke
./.venv/Scripts/python.exe experiments/run_ladder_rung1_distillation_scale.py --team-size 6 --smoke --preflight --device cpu
./.venv/Scripts/python.exe experiments/run_ladder_rung1_distillation_scale.py --team-size 6 --smoke --device cpu

# stage 4/5: pass a disposable --seed-base (e.g. 999xxxxx) and a small --n-seeds directly --
# both scripts already accept arbitrary seeds/labels with no separate --smoke flag needed.
```
