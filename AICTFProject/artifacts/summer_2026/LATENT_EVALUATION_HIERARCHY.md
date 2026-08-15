# Latent evaluation hierarchy — design record

**Status: DESIGN RECORD, NOT FROZEN.** Thresholds are deliberately deferred
until we know which policies survive the specialist pilot and the discovered-pair
replication, because the acceptable fidelity loss depends on which teachers we
actually get. Freeze this before any distillation run, not before.

Companion to `LATENT_BATTERY_AUDIT.md`, which records what of the battery
already exists in the repo.

## The five levels

```
LEVEL 0   FAITHFUL        z0 ≈ Teacher A, z1 ≈ Teacher B, in useful performance
LEVEL 1   DISTINCT        z causally changes behavior
LEVEL 2   PERSISTENT      those changes are stable team-level fingerprints
LEVEL 3   COMPLEMENTARY   different conditions favor different z
LEVEL 4   ADAPTIVE        held-out routing beats the best fixed choice
```

These are five increasingly strong claims. The historical failure mode was
treating them as interchangeable — different embeddings, different actions,
different trajectories, different strategies, useful repertoire.

## Gate 0 is checked twice

```
teachers → distillation → GATE 0a → PPO fine-tuning → GATE 0b
```

PPO fine-tuning is on-policy and updates from freshly collected trajectories,
so it can erase exactly the separation distillation created. Passing 0a is not
evidence for 0b.

Gate 0 requires **performance retention**, not only behavioral imitation:

```
V(z_i) >= V(T_i) - epsilon    on the teacher's own evaluation conditions
```

`epsilon` is deferred. `experiments/forced_z_eval/equivalence.py` is the
intended home.

## Two sets that must not be conflated in G

```
SELECTABLE AT RUNTIME    { z0, z1 }
FIXED COMPARATORS        { z0, z1, Teacher A, Teacher B, generalists }
```

The router must **not** be allowed to select the original teachers. If it can,
we are evaluating a mixed policy population, not latent compression. But the
comparator must include them, or two equally degraded students can produce a
healthy `delta_latent` — routing between z0 and z1 beating either alone — while
both lose to simply using Teacher A for everything.

```
V_routed_latent  >  max{ V_z0, V_z1, V_TA, V_TB, V_generalists }
```

Hard gate, and the right one. Same principle as `V_fixed` spanning all five
policies in the specialist pilot rather than the specialist pair.

## G splits into two claims

```
G0   oracle repertoire potential   — privileged condition label used ONLY for evaluation
G1   deployable router value       — q(z | legal observation/history)
```

G0 proves the latents are worth choosing between. G1 proves the agent can
actually make that choice. G0 passing does not imply G1.

Both inherit the split-half discipline from `SPECIALIST_PILOT_FROZEN.json`, and
per `SPECIALIST_PILOT_AMENDMENT_1`, **both the adaptive mapping and the fixed
comparator are selected on the selection half** and scored on the held-out half.
`forced_z_eval/analysis/oracle.py` must not be reused as-is — see the audit.

## Sequencing

First-pass proof, sufficient for a strong result:

```
Gate 0  →  A (matched-state action difference)
        →  C (strategic fingerprints)
        →  D (forced-z whole-episode)
        →  F (payoff crossover)
        →  G0 (unbiased held-out selective value)
```

Mechanistic validation, added after:

```
B    trajectory decoder q(z|tau)
D2   mid-episode do(z) intervention
E    persistence horizon
```

## Language discipline until E lands

Until persistence is measured, write **"distinct latent-conditioned behaviors"**,
never **"distinct latent strategies."** C's seven features are
episode-aggregated, so a `z` that only changes the opening move and then
converges can still move them substantially. Persistence is what upgrades a
behavioral mode to a strategy.

## D2 must be a true counterfactual branch

When built, not a switch inside one running episode:

```
clone simulator state + RNG state + policy hidden state
        ├── branch A: force z0
        └── branch B: force z1
compare futures
```

## Replication limit to keep in the paper

The `9200000` replication can establish that **these two checkpoints
reproducibly cross on fresh episodes**. It cannot establish that **D1 training
systematically produces regime A and D7 training systematically produces regime
B** — that needs fresh training seeds, not fresh evaluation episodes.

One seed remains correct for screening and mechanism discovery. If the final
headline chain passes (repertoire → fidelity → complementarity → router), that
specific comparison earns additional independent training seeds before
submission. Spend the seeds there, not on every branch.
