# PPO updater refactor — engineering record

**Owner:** structural refactor of `rl/custom_ppo/ppo_updater.py` into
`rl/custom_ppo/update/`. Method definition and preset fidelity remain in
[`summer-method-spec.md`](summer-method-spec.md) and
[`latent-preset-registry.md`](latent-preset-registry.md).

> **Completed:** 2026-06-17. **172** pinning/smoke/v6i1 tests green at completion.

---

## Phase completion table (0–13)

| Phase | Goal | Deliverable | Status |
|-------|------|-------------|--------|
| **0** | Lock behavior before moves | Characterization tests (`test_ppo_updater_refactor_phase1.py`, review-fixes, smoke) | Done |
| **1** | Correctness fixes first | Valid CF gate evidence (no CSV-zero EMA), KL-stop guards, single phase source, optimizer-owned clip, `latent_k` pair count, separation RNG checkpoint | Done |
| **2** | Resolved update context | `update_context.py` — `PPOUpdateContext` + builder | Done |
| **3** | Typed loss results | `loss_result.py` — `LossComponent`, `MinibatchUpdateResult`, `PairwiseSeparationMeasurement`; wired in `minibatch_updater.py` | Done |
| **4** | Phase + parameter ownership | `phase_policy.py` (`set_model_requires_grad_for_phase`), `param_registry.py` (`ParameterRegistry`, `OptimizerRegistry`) | Done |
| **5** | Loss families extracted | `entropy_objectives.py`, `strategy_objectives.py`, `separation_objectives.py` | Done |
| **6** | Optimizer stepping | `optimizer_stepper.py` — `SharedOptimizerStepper`, `ThreeOptimizerStepper`, `build_optimizer_stepper()` | Done |
| **7** | Telemetry accumulator | `telemetry.py` — `UpdateStatsAccumulator`, `AggregationMode` (`MEAN` / `LAST` / `SUM`); rollout soft router cols use `LAST` | Done |
| **8** | Post-update pipeline | `post_update.py` — deferred latent PPO, diagnostics CSV, gate evidence via `ActorInterventionEvidenceUpdater` (not CSV scrape) | Done |
| **9** | Slim coordinator | `update/updater.py` (~236 lines); `ppo_updater.py` is a 21-line re-export facade | Done |
| **10** | Optimizer steppers (commit-sized) | Same as Phase 6 — landed with stepper module and three-optimizer path | Done |
| **11** | Telemetry schema (commit-sized) | Same as Phase 7 — explicit aggregation modes per metric | Done |
| **12** | Post-update extraction (commit-sized) | Same as Phase 8 — `PostUpdatePipeline.run()` | Done |
| **13** | Dead code + compatibility | Removed monolithic `update()` body; re-exports `_policy_z_separation_loss`, `_extract_rollout_resample_subset`, `set_model_requires_grad_for_phase`; trainer `ppo_updater_state` checkpoint | Done |

Phases 10–13 were the **recommended git commit sequence** for the same
modules as Phases 6–8; they are listed separately so the record matches
the original plan, not because additional code paths were left unfinished.

---

## Layout after refactor

```text
rl/custom_ppo/update/
├── updater.py                 # coordinator
├── update_context.py
├── phase_policy.py
├── param_registry.py
├── loss_result.py
├── minibatch_updater.py
├── entropy_objectives.py
├── strategy_objectives.py
├── separation_objectives.py
├── optimizer_stepper.py
├── telemetry.py
├── post_update.py
├── actor_intervention.py
├── helpers.py
└── pair_utils.py
```

---

## What the refactor did *not* prove

Cleaner plumbing does **not** establish that counterfactual separation
produces **strategically distinct** repertoires or that the router can
select among them. Those are empirical claims requiring Runs A–C below
and gate / matched-seed evaluation per
[`experiment-and-evaluation-protocol.md`](experiment-and-evaluation-protocol.md).

---

## Recommended experimental sequence (post-refactor)

See also [`research-progress-tracker.md`](research-progress-tracker.md)
for run rows once launched.

### Run A — Mechanical smoke

- Small budget, compressed phase boundaries (or debug timestep caps).
- **Pass if:** phase A/B/C transitions, parameter ownership (only intended
  modules change), valid CF measurement → gate evidence (no placeholder
  zeros), checkpoint resume matches uninterrupted next update.

### Run B — Phase A diagnostic

- Train Phase A long enough for gate diagnostics (not a headline 1M claim).
- **Inspect:** `cf_competence_z*`, `cf_to_ppo_grad_ratio`, `cf_hinge_effective`,
  `cf_valid_team_groups`, `forced_z_pair_jsd_*`, `actor_intervention_*`,
  matched-seed semantics (v6i2). **Do not** interpret router metrics.

### Run C — Full staged run

- Only after Run B shows real repertoire + behavioral realization support.
- Phase A → B → C at locked schedule; final eval: forced-z, natural router,
  fixed best-z, oracle router (matched seeds).

---

## Telemetry map (specialization diagnostics)

| Question | Columns / gate families |
|----------|-------------------------|
| CF gradient balance | `cf_to_ppo_grad_ratio`, `cf_actor_grad_norm`, `ppo_actor_grad_norm` |
| CF hinge active? | `cf_hinge_active`, `cf_hinge_effective`, `cf_batch_pairs_below_margin` |
| Competence weighting | `cf_competence_z0`…`z3`, `cf_competence_ready`, `cf_weight_sum` |
| Actor intervention (v6i2) | `cf_batch_pair_jsd_*`, `actor_intervention_measurement_valid`, `actor_intervention_gate_updated` |
| Macro preference (rollout) | `forced_z{z}_macro_*_prob`, `forced_z_pair_jsd_*` |
| Behavioral realization (boundary) | Gate family `behavioral_realization` / matched-seed semantics |
| Router learnability | Gate family `selector_learnability_probe` (Phase B+) |

**Gap:** granular team tactics (attack-lane vs return-lane preference,
escort allocation, team spread, objective timing) are not yet first-class
CSV columns; proxy via macro actions + spread/role bucket MI in
`latent_diagnostics.py` until dedicated behavioral probes are added.
