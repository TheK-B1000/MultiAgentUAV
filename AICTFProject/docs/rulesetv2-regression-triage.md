# RULESET_V2 regression triage

**Date:** 2026-07-30
**Question:** does any pre-existing test failure intersect the G0-v2 execution path?
**Baseline:** 33 failures before the rules change, 33 after (zero net new).

The G0-v2 path is: `no_latent_baseline` preset, no latent variable, 2v2, `map_a`,
OP6-OP12 mixture, checkpoint save/load, `train_ppo.py`.

---

## Blocking / non-blocking by file

| File | Count | Touches G0-v2 path? | Disposition |
|---|---|---|---|
| `test_latent_arc_credit.py` | 10 | No — latent arc-credit PPO only | **Defer** |
| `test_v6i7_parts_b_g.py` | 7 | No — balanced-z / arc modes, latent only | **Defer** |
| `test_option_advantage.py` | 4 | No — q_phi option advantages, latent only | **Defer** |
| `test_bt_opponents.py` | 7 | **Yes** — OP11/OP12 + defender tree | **Gate 1** |
| `test_tactical_opponents.py` | 2 | **Yes** — OP9 late-game pressure | **Gate 1** |
| `test_preset_resolution.py` | 1 | No — see below | **Defer** |
| `test_preset_system.py` | 1 | No — see below | **Defer** |
| `test_bt_team_sizes.py` | 1 | No — 4v4 role spread; G0-v2 is 2v2 | **Defer** |

## Preset failures do not touch `no_latent_baseline`

`test_resolved_configs_match_snapshot` fails because the stored snapshot predates a set of
**latent** presets:

```
latent_v6i23_population_birth, latent_v6i24_full_policy_population,
latent_v6i26_latent_response_oracle, plan_faithful_latent_*, v6i23*, v6i24*, v6i26*
```

`test_registry_covers_all_legacy_presets` fails on one legacy key,
`v6i13_opening_window_advantage_router` — a router preset.

Every name is latent/router. None is `no_latent_baseline`. Fix is `python
tools/snapshot_presets.py`, which is a housekeeping task unrelated to this experiment and
deliberately NOT run here (regenerating a snapshot mid-experiment would itself be an
unreviewed change).

## Opponent failures: quarantine pending Gate 1

These are the only failures on the G0-v2 path. All seven `test_bt_opponents` failures and
both `test_tactical_opponents` failures concern OP9 / OP11 / OP12 decision logic.

Example: `test_defender_chases_carrier` asserts `ROLE_DEFENDER` (1) appears when an enemy
carrier exists, but the tree returns `[3, 3]` = both agents `ROLE_INTERCEPTOR`. Intercepting
the enemy carrier *is* chasing it, so this is plausibly a stale assertion left after a BT
refactor rather than a functional defect — but that is a hypothesis, not a finding.

**Unit tests that encode role *names* are weaker evidence than measured behavior**, and all
of these predate RULESET_V2 (they were written for a game in which a lone defender could not
tag at all). Gate 1 therefore decides opponent admissibility empirically: an opponent enters
the training mixture only if it demonstrates legal, non-degenerate play under V2.

Per the standing rule: an opponent that fails Gate 1 is **quarantined from the mixture**, not
silently included, and not tuned merely for performing poorly. Only concrete rule violations,
deadlocks, or broken behavior are fixed.

## Net position

```
Failures blocking G0-v2 launch outright : 0
Failures resolved by Gate 1 evidence    : 9  (OP9 / OP11 / OP12)
Failures deferred as out-of-path        : 24 (latent, presets, 4v4)
```

No failure blocks the launch by itself; nine gate the composition of the training mixture.
