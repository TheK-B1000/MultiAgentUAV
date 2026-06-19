# v6i2 dual-evidence gate protocol — frozen calibration record

**Status:** FROZEN — thresholds below are locked for confirmatory v6i2
and v6i3 lineage runs as of 2026-06-18. Runtime gate-bounding fields were
added on 2026-06-19 before any confirmatory promotion result was accepted.
Do not change any value below after looking at confirmatory results. Any
threshold change creates a new exploratory lineage and must use a new
protocol/fingerprint.

**Owning preset:** `apply_plan_faithful_latent_v6i2_staged_team_intent_curriculum`
(`v6i2`, `gate_protocol_version = v6i2_dual_evidence`)

**Related docs:** `summer-fidelity-rules.md`, `latent-preset-registry.md`,
`experiment-and-evaluation-protocol.md`, `research-progress-tracker.md`

---

## Launch sequence (correct order)

1. **Short v6i2 smoke** — wiring, schedule clocks, checkpoint/resume, gate
   telemetry (not a scientific result).
2. **Threshold calibration** — distributions from existing v6i1 runs (below).
3. **Freeze this document** — record locked values + fingerprint. **Done
   2026-06-18.**
4. **Full fresh enforce confirmatory run** — nominal 1.0M budget, potentially
   extending to 1.3M after late Phase A promotion; `--fresh-metrics-csv`.

A shortened run validates infrastructure only. The confirmatory result must use
the frozen table unchanged. Runs launched before this freeze are exploratory,
even if their metrics later satisfy the thresholds.

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
| `gate_config_fingerprint` | `224f1aea9ab36319` | hash of resolved gate config |
| `phase_a_earliest_end_fraction` | 0.40 | inherited v6i1 schedule floor |
| `phase_a_max_end_fraction` | 0.70 | v6i2 extends Phase A before forced promotion |
| `phase_b_fixed_fraction` | 0.30 | fixed post-promotion budget |
| `phase_c_fixed_fraction` | 0.30 | fixed post-promotion budget |
| `phase_c_start_fraction` | 0.70 | nominal start if Phase A/B do not extend |
| `curriculum_extend_terminal_on_late_promotion` | `True` | preserve fixed Phase B/C budgets after late Phase A |
| `phase_boundary_gate_mode` | `enforce` | confirmatory promotion must be gate-controlled |
| `phase_a_gate_max_seconds` | 900 | hard online gate wall-clock ceiling |
| `phase_a_gate_progress_interval_seconds` | 60 | progress JSON heartbeat cadence |
| `curriculum_gate_online_matched_seed_count` | 5 | bounded online promotion workload |
| `curriculum_gate_online_matched_seed_max_steps` | 64 | bounded online rollout horizon |
| `curriculum_gate_run_boundary_eval` | `True` | matched-seed behavior gate must run |
| `curriculum_gate_run_probe` | `True` | selector probe available as diagnostic |
| `curriculum_gate_selector_blocks_phase_a` | `False` | selector probe does not block Phase A promotion |
| `curriculum_probe_min_examples` | 10 | minimum probe support |
| `actor_jsd_margin` | 0.001 | actor CF pair EMA must clear nonzero intervention floor |
| `actor_jsd_floor_fraction` | 0.50 | weakest pair must stay above half-margin |
| `actor_jsd_min_passing_pairs` | 5 | at least 5 of 6 latent pairs must pass |
| `actor_jsd_consecutive_updates` | 3 | temporal stability |
| `actor_jsd_ema_decay` | 0.10 | responsive but smoothed online EMA |
| `actor_intervention_gate_rule` | `batch_margin_ema_floor_v1` | current-batch margin plus EMA floor stability |
| `macro_jsd_margin` | 0.0001 | supporting profile only |
| `macro_jsd_floor_fraction` | 0.50 | supporting profile only |
| `macro_jsd_min_passing_pairs` | 1 | supporting profile only |
| `macro_jsd_ema_decay` | 0.10 | supporting profile only |
| `behavioral_realization_effect_threshold` | 0.02 | matched-seed behavior effect floor |
| `behavioral_realization_adverse_threshold` | -0.01 | reject adverse matched-seed behavior |
| `behavioral_realization_min_opponents_pass` | 2 | at least two OP5/OP6/OP7 slices must pass |
| `behavioral_route_distance_scale` | 0.03 | frozen route-distance normalization scale |
| `behavioral_task_behavior_distance_scale` | 0.02 | frozen task-behavior normalization scale |
| `behavioral_performance_spread_scale` | 0.03 | frozen performance-spread normalization scale |
| `behavioral_route_distance_weight` | 0.25 | normalized aggregate route weight |
| `behavioral_task_behavior_distance_weight` | 0.50 | normalized aggregate task-behavior weight |
| `behavioral_performance_spread_weight` | 0.25 | normalized aggregate performance-spread weight |
| `behavioral_aggregate_effect_threshold` | 0.75 | normalized aggregate pass floor |
| `behavioral_min_task_behavior_distance` | 0.01 | component floor; route cannot compensate for zero task behavior |
| `behavioral_min_performance_spread` | 0.01 | component floor; route cannot compensate for zero performance spread |
| `behavioral_matched_seed_min_seeds_per_opponent` | 20 | full confirmatory evaluation; online promotion gate overrides to 5 |
| `curriculum_nominal_timesteps` | 1_000_000 | standard 4v4 budget |

**Structural actor-intervention rule (retain):**

- current CF batch has >= `actor_jsd_min_passing_pairs` of 6 pairs above
  `actor_jsd_margin`
- actor-CF EMA has >= `actor_jsd_min_passing_pairs` of 6 pairs above
  `actor_jsd_floor_fraction x actor_jsd_margin`
- >= `actor_jsd_consecutive_updates` consecutive valid updates satisfy both
  current-batch strength and EMA-floor stability

---

Online Phase A promotion is bounded: cheap online prerequisites (`coverage`,
`competence`, and `training_integrity`) are checked first. Failed prerequisites
skip matched-seed evaluation, write a report, and resume PPO with Phase A
unchanged. Actor-intervention, behavior-pair, and corridor diagnostics are gate
evidence, not prerequisites for collecting matched-seed evidence. If cheap
prerequisites pass, the online behavioral gate uses 3 opponents, 5 seeds, 4 z
branches, a 64-step horizon, and a 900-second wall-clock budget. Timeout records
`inconclusive_timeout`, does not promote, and resumes training. Full
20-seed matched-seed analysis, selector analysis, and bootstrap reporting
belong to offline confirmatory evaluation or final candidates.

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

Report `route_distance`, `task_behavior_distance`, `performance_spread`, and
`aggregate_effect` separately. The aggregate is computed from frozen scales and
weights, then component floors are enforced before an opponent can pass. A large
route effect alone is not behavioral realization.

---

## Calibration runs referenced at freeze

| Run ID / tag | Steps used | Purpose |
|--------------|------------|---------|
| `v6i1_staged_team_intent_curriculum_OP5_OP6_OP7_1m_4v4` | available CSV rows through pre-freeze run end | weak λ_cf=0.01 scale check |
| `v6i1_cf_coef1p0_OP5_OP6_OP7_1m_4v4` | available CSV rows through pre-freeze run end | strong λ_cf=1.0 scale check |
| `v6i2_staged_team_intent_curriculum_OP5_OP6_OP7_1m_4v4` | smoke/wiring only | fingerprint and schedule sanity, not confirmatory |

No v6i3 result metrics were used to choose the frozen v6i2 thresholds.

---

## Confirmatory launch checklist

- [x] This document status = **FROZEN**
- [x] Every threshold row filled
- [x] `gate_config_fingerprint` recorded here matches preset resolution
- [ ] Fresh EMA state (`--fresh-metrics-csv`); no v6i1 checkpoint import
- [x] `phase_boundary_gate_mode = enforce`
- [x] `confirmatory_gate_lineage_valid = True`
- [x] `allow_gate_config_mismatch_on_resume = False`

---

## Changelog

| Date | Status | Notes |
|------|--------|-------|
| 2026-06-18 | DRAFT | Template + schedule-clock contract; calibration pending |
| 2026-06-18 | FROZEN | Locked v6i2 gate thresholds and fingerprint `85506ab324d464c5`; pre-freeze runs remain exploratory |
| 2026-06-19 | FROZEN | Added bounded online gate runtime contract, selector non-blocking Phase A promotion, stale-aware skipped-gate semantics, and updated fingerprint `224f1aea9ab36319`. |
| 2026-06-19 | FROZEN | Switched actor-intervention streak to `batch_margin_ema_floor_v1` without changing numeric thresholds; updated fingerprint `224f1aea9ab36319`. |
| 2026-06-19 | FROZEN | Recorded normalized matched-seed component scales/floors and independent route/task/performance reporting; fingerprint remains `224f1aea9ab36319`. |
