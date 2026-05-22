# Latent PPO Methodology

This repo now keeps three latent-strategy lanes separate. Do not mix their claims, flags, or results.

## Lane 1: Plan-Faithful Baseline

Preset:

```text
latent_a1_plan_faithful
```

Purpose:

```text
Test the original Summer Implementation Plan as written.
```

Allowed behavior:

```text
q_phi(z | s) maps global state to a categorical strategy distribution.
z is sampled during training.
z is selected by argmax during deterministic deployment.
The decentralized actor receives local observation plus z embedding.
The centralized critic receives global state, joint action, and z.
Strategy persistence and entropy regularization are allowed.
```

Not allowed in this lane:

```text
oracle opponent IDs
router cross-entropy labels
forced-z training
per-z handcrafted shaping
auxiliary return heads
supervised strategy labels
hand-authored strategy definitions
```

This lane is the clean paper comparison. If it fails to route, that is a valid result:

```text
Reward-only latent routing failed to produce opponent-conditioned q_phi in this environment.
```

Do not silently patch this preset with staged-router rescue features.

## Lane 2: Staged Latent Strategy Routing

Preset:

```text
latent_slsr_router
```

Purpose:

```text
Make routing learnable after the reward-only baseline proved too weak.
```

This is a revised method, not the original faithful baseline.

Stages:

```text
Stage 1: force or rotate z so the actor practices distinct z-conditioned skills.
Stage 2: freeze actor/critic and train q_phi with supervised best-z labels.
Stage 2 oracle: give q_phi one-hot opponent ID as a wiring/control test.
Stage 2 feature: remove oracle and train q_phi from real state/behavior features.
Stage 3: jointly fine-tune while keeping router CE/KL anchors.
```

Key flags:

```text
--forced-z-mode fixed|rotate
--qphi-oracle one_hot
--qphi-oracle-dim 7
--freeze-actor-critic
--router-ce-labels path/to/best_z_labels.json
--router-ce-coef 1.0
--router-ce-mode hard|soft
```

C1 pass condition:

```text
With oracle one-hot input and hard labels, q_phi routes each known opponent to the requested z.
q_phi CE is nonzero at first update.
q_phi gradient norm is nonzero.
Actor/critic parameters stay frozen.
```

C2 pass condition:

```text
Without oracle input, q_phi still shows measurable MI(z; opponent) or routing accuracy from real features.
```

## Lane 3: Stage-1B Shaping Ablation

Preset:

```text
latent_stage1b_shaping
```

Purpose:

```text
Test whether lightweight per-z shaping can force more distinct latent skills.
```

This is explicitly not plan-faithful. It uses handcrafted shaping to bias z modes toward different behavior. Keep it out of the baseline and report it as an ablation or engineering rescue.

Guardrail:

```text
use_per_z_shaping requires forced_z_mode='rotate'
```

If a run enables per-z shaping without rotate mode, treat it as a configuration bug.

## Reporting Rules

Use these labels in tables, plots, and paper notes:

```text
A1 Reward-only latent PPO
SLSR-C1 Oracle supervised router
SLSR-C2 Feature supervised router
SLSR-C3 Joint fine-tune
Stage1B shaping ablation
Flat PPO baseline
Fixed-z ablations
```

Never call SLSR or Stage1B shaping "faithful" to the original plan.

## Artifact Hygiene

Training checkpoints, eval CSVs, smoke outputs, and temporary comparison files are run artifacts. They should not make the code diff unreadable.

Use stable source files for:

```text
rl/
tests/
tools/
docs/
```

Use ignored or archived paths for:

```text
AICTFProject/csv/eval_*.csv
AICTFProject/csv/archive_pre_slsr/
AICTFProject/.test_runs/
AICTFProject/checkpoints/
AICTFProject/logs/*.log
```

If an eval result is paper-relevant, summarize it in docs or a small manifest instead of committing every raw CSV.

## Current Interpretation

The latent issue is not that the actor/critic stack is fundamentally broken. The evidence so far is:

```text
z modes can behave differently under fixed-z eval.
q_phi stayed near-uniform in reward-only training.
deterministic argmax deployment turns a tiny q_phi bias into one repeated z.
opponent-conditioned routing is the missing piece.
```

That supports this research framing:

```text
Reward-only latent routing is a clean baseline but insufficient here.
Staged skill discovery plus supervised router alignment is the repaired method.
```
