# v6i2 dual-evidence gate protocol — frozen calibration record

**Status:** DRAFT — thresholds are **TBD** until calibration against v6i1
λ_cf baselines completes. Do **not** launch a confirmatory enforce run until this
document is marked **FROZEN** and every row below has a locked value.

**Owning preset:** `apply_plan_faithful_latent_v6i2_staged_team_intent_curriculum`
(`v6i2`, `gate_protocol_version = v6i2_dual_evidence`)

**Related docs:** `summer-fidelity-rules.md`, `latent-preset-registry.md`,
`experiment-and-evaluation-protocol.md`, `research-progress-tracker.md`

---

## Launch sequence (correct order)

1. **Short v6i2 smoke** — wiring, schedule clocks, checkpoint/resume, gate
   telemetry (not a scientific result).
2. **Threshold calibration** — distributions from existing v6i1 runs (below).
3. **Freeze this document** — record locked values + fingerprint.
4. **Full fresh enforce confirmatory run** — nominal 1.0M budget, potentially
   extending to 1.3M after late Phase A promotion; `--fresh-metrics-csv`.

A shortened run validates infrastructure only. The confirmatory result must use
the frozen table unchanged.

---

## Schedule clocks (runtime contract)

| Mechanism | Clock |
|-----------|-------|
| CF warmup/ramp | Fixed nominal `curriculum_nominal_timesteps` (1M) |
| Phase A gate boundaries | Fixed nominal fractions |
| Training loop termination | Dynamic `training_terminal_step` |
| Progress bar / checkpoints | Dynamic terminal |
| Phase B LR (critic, router) | Phase-B-local `phase_b_budget_steps` from `t_A` |
| Phase C LR (actor, critic, router) | Phase-C-local `phase_c_budget_steps` from end of B |
| Phase A actor/critic LR | Nominal curriculum clock (monotonic; no rebound on extension) |
| Entropy anneal (`latent_lam_h`) | Nominal curriculum clock |

Implementation: `resolve_v6i1_lr_progress_remaining`,
`resolve_v6i1_entropy_schedule_total_timesteps` in `v6i1_phase_runtime.py`.

---

## Frozen gate configuration

Record the resolved configuration and fingerprint at freeze time. The
confirmatory run must match exactly.

| Field | Locked value | Calibration basis |
|-------|-------------:|-------------------|
| `gate_protocol_version` | `v6i2_dual_evidence` | protocol |
| `gate_config_fingerprint` | TBD | hash of resolved gate config |
| `actor_jsd_margin` | TBD | strong-run pair EMA distribution |
| `actor_jsd_floor_fraction` | TBD | weakest stable pair in strong run |
| `actor_jsd_min_passing_pairs` | 5 | repertoire coverage (structural) |
| `actor_jsd_consecutive_updates` | 3 | temporal stability (structural) |
| `actor_jsd_ema_decay` | TBD | responsiveness vs noise |
| `macro_jsd_margin` | TBD | diagnostic only |
| `macro_jsd_floor_fraction` | TBD | diagnostic only |
| `macro_jsd_min_passing_pairs` | TBD | diagnostic only |
| `macro_jsd_ema_decay` | TBD | diagnostic only |
| `behavioral_realization_effect_threshold` | TBD | matched-seed null floor |
| `behavioral_realization_adverse_threshold` | TBD | matched-seed null floor |
| `behavioral_realization_min_opponents_pass` | TBD | coverage criterion |
| `behavioral_matched_seed_min_seeds_per_opponent` | TBD | uncertainty stability |
| `phase_a_earliest_end_fraction` | TBD | observed separation trajectory |
| `phase_a_max_end_fraction` | 0.70 | v6i2 preset (verify at freeze) |
| `phase_b_fixed_fraction` | 0.30 | fixed post-promotion budget |
| `phase_c_fixed_fraction` | 0.30 | fixed post-promotion budget |
| `curriculum_nominal_timesteps` | 1_000_000 | standard 4v4 budget |

**Structural actor-intervention rule (retain):**

- ≥ `actor_jsd_min_passing_pairs` of 6 pairs above `actor_jsd_margin`
- all 6 pairs above `actor_jsd_floor_fraction × margin`
- ≥ `actor_jsd_consecutive_updates` consecutive valid EMA updates

---

## Calibration data sources (v6i1 only — not confirmatory)

Use completed v6i1 staged runs as **calibration evidence only**:

| Run | Role |
|-----|------|
| λ_cf = 0.01 | Weak-pressure baseline / null-ish actor intervention |
| λ_cf = 1.0 | Strong-pressure calibration |

For λ_cf = 1.0, prefer steps with full CF coefficient (≥ 393k or 458k). The
131k row is encouraging but CF was only ~0.31 — not full-strength.

### Actor intervention (Gate A)

Derive thresholds from **distributions**, not a single convenient row:

1. Weak-run per-pair CF EMA distribution (across seeds if available).
2. Strong-run per-pair CF EMA distribution.
3. Initialization / no-CF distribution.
4. Cross-seed variation.

Lock `actor_jsd_margin` above the weak/null region while remaining reachable
consistently in the strong run.

### Behavioral realization (Gate B)

Establish matched-seed noise floors from repeated branches where no meaningful
intervention difference should exist. Lock semantic thresholds above that null
variability, with confidence and minimum-seed requirements.

---

## Calibration runs referenced at freeze

| Run ID / tag | Steps used | Purpose |
|--------------|------------|---------|
| TBD | TBD | weak λ_cf=0.01 actor EMA |
| TBD | TBD | strong λ_cf=1.0 actor EMA |
| TBD | TBD | matched-seed null floor |

---

## Confirmatory launch checklist

- [ ] This document status = **FROZEN**
- [ ] Every threshold row filled (no TBD)
- [ ] `gate_config_fingerprint` recorded here matches preset resolution
- [ ] Fresh EMA state (`--fresh-metrics-csv`); no v6i1 checkpoint import
- [ ] `phase_boundary_gate_mode = enforce`
- [ ] `confirmatory_gate_lineage_valid = True`
- [ ] `allow_gate_config_mismatch_on_resume = False`

---

## Changelog

| Date | Status | Notes |
|------|--------|-------|
| 2026-06-18 | DRAFT | Template + schedule-clock contract; calibration pending |
