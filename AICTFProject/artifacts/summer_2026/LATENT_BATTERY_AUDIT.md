# Latent diagnostic battery — reuse audit

Read-only audit of what the A–G latent-strategy battery already has in this
repo, done while S_OP7/S_OP8 train. Nothing here runs, and the latent phase
remains gated behind the specialist gates and the pair replication.

Status label: **OBSERVED** (files read at commit `a8953391`).

## Mapping

| Battery item | Status | Where |
|---|---|---|
| **A** action distinctness (matched-state JSD/KL) | REUSABLE | `rl/custom_ppo/diagnostics/counterfactual.py` — "forced-z profiling, JSD, KL sensitivity"; telemetry field `phase_a_actor_jsd_mean` |
| **B** trajectory decoder `q(z\|τ)` | **MISSING** | no trajectory-level classifier anywhere in the tree |
| **C** strategic fingerprint | REUSABLE | `rl/forced_z_behavior_vectors.py` — 7 named features with normalization bounds |
| **D** causal control — forced `z` | REUSABLE | `experiments/forced_z_eval/` — canonical matched-seed protocol, runner, equivalence checks |
| **D** causal control — mid-episode `do(z)` switch | **MISSING** | `diagnostics/switching.py` is *observational* (switch proximity, reward-after-switch), tied to router switching, not an intervention |
| **E** temporal persistence | **MISSING** | no horizon-of-effect measurement |
| **F** payoff complementarity | REUSABLE | `experiments/forced_z_eval/analysis/complementarity.py` — "complementarity ladder" |
| **G** selective value | **REUSABLE ONLY AFTER FIX** | `experiments/forced_z_eval/analysis/oracle.py` — see below |

## C already covers most of the wanted fingerprint

`FORCED_Z_BEHAVIOR_VECTOR_NAMES`:

```
attack_lane_preference     return_lane_preference
interception_pressure      escort_allocation
team_spread                objective_entry_timing
role_allocation
```

These map onto the requested metrics: carrier support ≈ `escort_allocation`,
tag attempts ≈ `interception_pressure`, lane choice ≈ the two lane preferences,
team spread is direct. Each has canonical `(lo, hi)` normalization bounds, so
pair distances are already comparable across features.

Not covered, would need adding: enemy-side/home-side occupancy fraction,
distance-to-flag traces, time-to-first-attack. `objective_entry_timing` is
adjacent to the last one but is not the same measurement.

## G has the bias this project already ruled out

`oracle.py:39`:

```python
oracle_vals.append(float(max(metric_value(ep_lists[z][i], metric) for z in latents)))
```

This takes the max **per episode** across latents, on the same episodes used to
score it. That is the naive oracle in its strongest-biased form — stronger than
a per-cell max, because selection happens at episode granularity. It cannot
return a non-positive gap even when all latents are identical.

`SPECIALIST_PILOT_FROZEN.json` already rejects this estimator for `delta_pool`
and requires split-half (select the argmax on half the episodes, score on the
other half). **Reusing `oracle.py` verbatim for battery item G would reintroduce
the exact bias the specialist gate was written to exclude.** Either add a
split-half path to it or compute G separately.

The matched-seed structure it is built on is still the right foundation — the
defect is the estimator, not the protocol.

## Gap not in the A–G list: distillation fidelity

A–G all measure the *distilled* network. None of them check that the distilled
policies still reproduce their teachers.

That admits a silent failure: `π(a|o,z0)` and `π(a|o,z1)` could be cleanly
distinct, persistent, and decodable — passing A–E — while **both** are worse
than Teacher A and Teacher B. The battery would report "distinct latent
behaviors" for two degraded policies.

Proposed prerequisite, before A–G is read at all:

```
FIDELITY:  WR(π(a|o,z0)) ≈ WR(Teacher A) on Teacher A's board
           WR(π(a|o,z1)) ≈ WR(Teacher B) on Teacher B's board
           within a preregistered tolerance
```

Existing `experiments/forced_z_eval/equivalence.py` (behavioral-equivalence
checking) is the natural place to build this, and the cross-play evaluator
already reports per-opponent win rates for a teacher baseline.

## Hierarchy caution

The proposed reading is `A–E pass → distinct`, `F → strategies`, `G → useful`.
Worth keeping explicit that A–E can pass for two policies that are merely
*different* rather than complementary; only F speaks to complementarity, and
only G to value. Two badly-distilled policies can be extremely distinct.

## Summary

```
REUSABLE NOW          A, C, D(forced-z), F
NEEDS AN ESTIMATOR FIX  G
MISSING                B, D(mid-episode switch), E, FIDELITY
```

Four of eight items exist and are reusable, one needs the split-half fix
already specified elsewhere in this campaign, and three plus the fidelity
prerequisite would be new work. The forced-z protocol, matched-seed
infrastructure, and behavior-vector definitions are the substantial existing
assets — the battery does not need rebuilding from scratch.
