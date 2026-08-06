# C3 Item 10 — Code audit vs draft commitment-fork contract

**Status:** ITEM 10 IMPLEMENTATION PATCHED (2026-08-06); no C3 scan run  
**Scope:** surgical classification only — no redesign, no discovery scan.  
**Contract:** [`c3-decision-proximal-preregistration.md`](c3-decision-proximal-preregistration.md)  
(item 9 SETTLED + draft Stages 1–4 + execution authorization)

**Audited files:**

```text
rl/analysis/decision_proximal_features.py
rl/analysis/counterfactual_actionability.py
experiments/run_c3_decision_proximal_discovery.py
```

**Classes:** `MATCH` | `PATCH REQUIRED` | `REMOVE` | `MISSING`

---

## A–N checklist answers

| # | Question | Verdict | Notes |
|---|---|---|---|
| **A** | Natural G0 / `map_a` trajectories? | **MATCH** (with caveats) | Runner loads frozen G0-V5 ckpts `3200001/2/3`, rolls with `CANONICAL_MAP` + `V2_RULES`, seeds `9400000+`. No scenario injection. Caveat: seed block is evaluation-style replay of discovery seeds, not training trajectories — still natural unmodified G0 rollouts. |
| **B** | Pressure onset only as backward-trace **anchor**? | **PATCH REQUIRED** | Onset is treated as the **unit of analysis** itself (`collect_onsets` records onset rows; Stage 3 branches **at** onset). No “anchor then trace” semantics. |
| **C** | Search backward up to `T_trace`? | **MISSING** | No `T_trace`, no walk-back loop, no candidate earlier states. |
| **D** | Test ≥2 legal **TEAM** responses? | **PATCH REQUIRED** | `run_counterfactual_branches` overrides agent 0 and agent 1 macros **unilaterally** over `{0..4}`. No legality pruning. No joint carrier×mate enumeration. No check that ≥2 legal alternatives exist before testing. |
| **E** | Reject states already effectively doomed? | **MISSING** | No recoverability / “still contingent” filter. |
| **F** | Counterfactual uses expected task utility `U`? | **PATCH REQUIRED** | Branching uses single deterministic rollout. `outcome_shift` is `abs(survived_base - survived_branch)` (binary survival flip), not \(\mathbb{E}[U]\). Returns/score deltas are recorded but **not** used for the actionability gate. No stochastic \(N\)-seed expectation. |
| **G** | Divergence across `H_response` (not one-step only)? | **PARTIAL → PATCH REQUIRED** | Override is one-step-then-return-to-G0 (MATCH for “brief fork force”). Horizon default 30 rolls forward (MATCH as evaluation horizon). But there is **no** distinct `H_response` response-mode concept; survival-at-horizon is not the same as “divergence persists over `H_response`.” |
| **H** | Require divergence ≥ `delta`? | **PATCH REQUIRED** | `min_effect` default **0.05** on absolute survival flip. Draft target for useful improvement is **≥0.10** improvement over G0 (not absolute shift). Sign/definition wrong. |
| **I** | Choose the **EARLIEST** qualifying state? | **MISSING** | No earliest-state selection. Later onsets are first-class candidates. |
| **J** | Emit “no commitment fork” when none qualifies? | **PARTIAL → PATCH REQUIRED** | Writes `C3_NO_QUALIFIED_STRATEGIC_FORK.json` when Stage pipeline yields zero **feature** candidates. Does **not** emit per-episode / per-anchor “no fork” under item-9 R1–R4. Wrong semantic unit (features vs forks). |
| **K** | Task reward / rules / map untouched? | **MATCH** | Uses `V2_RULES`, `CANONICAL_MAP`, stock env + G0 policy. No reward edits in these three files. |
| **L** | Stage 3 only controllability screen (no latent claim)? | **MATCH** (docstring) / **PATCH REQUIRED** (science) | Runner docstring says discovery-only / no latent necessity. Scientifically Stage 3 still gates on obsolete absolute actionability-of-features, not item-9 commitment forks. Controllability-vs-strategy separation not implemented as contract language in outputs. |
| **M** | Fresh confirmation still separate? | **MATCH** | This runner is discovery-only (`9400000+`). No Stage 4 / `9810000+` path in code. Confirmation remains a later act (not implemented here — correct). |
| **N** | Nothing runs without authorization artifact? | **MATCH** | `_require_c3_execution_authorization()` is first line of `main()`; missing/mismatched auth → `SystemExit`. Verified by dry launch. |

**Bottom line:** plumbing for natural G0/`map_a` rollouts, snapshot/restore, one-step override, and auth hard-guard is usable. The **commitment-fork science (item 9) is almost entirely MISSING or wrong-signed**. Current code is a superseded pressure-onset feature scanner, not a commitment-fork detector.

---

## File-by-file classification

### `rl/analysis/decision_proximal_features.py`

| Behavior | Class | Detail |
|---|---|---|
| Instantaneous features from named core fields (`blue_x`, `carrying`, cooldowns, homes) | **MATCH** | Aligns with “named-state accessors” direction (item 8 still open for finalization). |
| Carrier-pressure detection via `PRESSURE_RADIUS_FRAC=0.18` | **PATCH REQUIRED** | Radius-only; draft wants geometry + tag readiness predicate (item 1 still open). Usable as provisional anchor detector after patch. |
| Onset = `just_picked_up OR became_pressured` | **REMOVE** (pickup) / **PATCH REQUIRED** (pressure) | Draft: **do not mix flag pickup** with pressure. Pickup must be removed from C3 onset family. Pressure-crossing alone remains anchor candidate. |
| `time_to_intercept = dist/DEFAULT_SPEED + 20` when not closing | **REMOVE** / **PATCH REQUIRED** | Explicitly rejected pseudo-ETA. Draft: if `closing_vel ≤ ε` → sentinel `H+1`. |
| `relative_closing_velocity` uses red velocity only toward carrier | **PATCH REQUIRED** | Draft wants \((v_d - v_c)\cdot\hat d\). Carrier velocity unused. |
| `commitment_imbalance = abs(attackers - defenders)` | **PATCH REQUIRED** | Draft: keep **signed** `attackers - defenders`; abs only secondary. |
| `mate_intervention_eta = escort_dist / DEFAULT_SPEED` | **PATCH REQUIRED** | Magic `0.15` speed; document or replace with env motion model. |
| Tracks `_prev_*` for velocity / pressure trend | **MATCH** | Needed for trends / onset edge detect. |
| Backward-trace / fork search helpers | **MISSING** | No API to walk earlier steps or score R1–R4. |

### `rl/analysis/counterfactual_actionability.py`

| Behavior | Class | Detail |
|---|---|---|
| Snapshot/restore via `q_probe_local_counterfactual` | **MATCH** | Keep; determinism self-test present. |
| One-step macro override then natural G0 continuation | **MATCH** | Matches “brief fork force / not 30-step forced specialist.” |
| Horizon roll-forward after override | **MATCH** (as eval horizon) | Default 30. Must be wired as contract `H` / distinguished from `H_response` once item 4 closes. |
| Unilateral agent-0 / agent-1 overrides over macros 0–4 | **PATCH REQUIRED** | Need legal team responses (carrier, mate, joint) after legality pruning; require ≥2 legal alts (item-9 R2). |
| `outcome_shift = abs(surv_base - surv_branch)` | **REMOVE** (as gate metric) | Absolute binary flip ≠ improvement \(A(s)\). |
| Gate uses max absolute shift ≥ `min_effect` | **PATCH REQUIRED** | Must be \(A(s)=\max_{a'} \mathbb{E}[U|s,a']-\mathbb{E}[U|s,G0] \ge \delta\) with \(\delta\) from items 3/6 (draft 0.10). |
| Expected utility / stochastic \(N\) | **MISSING** | Single deterministic continuation only. |
| Persist-over-`H_response` / reconvergence test | **MISSING** | No check that divergence persists vs trivially recovers. |
| Doomed-state rejection | **MISSING** | |
| Earliest-state selection among candidates | **MISSING** | Module is per-onset; no multi-state ranking. |

### `experiments/run_c3_decision_proximal_discovery.py`

| Behavior | Class | Detail |
|---|---|---|
| Auth hard-guard before any rollout | **MATCH** | Item N. |
| G0 ckpts + `CANONICAL_MAP` + `V2_RULES` + OP6–OP12 | **MATCH** | Item A/K. |
| Discovery seeds `9400000+` | **MATCH** | |
| Stage 4 / fresh `9810000+` absent | **MATCH** | Confirmation stays separate (M). |
| Progress bars / durable log | **MATCH** | Operational; not scientific. |
| `collect_onsets`: unit = pressure/pickup onset | **PATCH REQUIRED** | Must become: locate pressure **anchor** → backward trace → emit fork or no-fork. |
| Stage 1: rank **features** by fail/ctrl CI | **REMOVE** / **PATCH REQUIRED** | Superseded scientific unit. Contract is commitment **forks**, not aggregate feature scan. Retain features as descriptors **at the fork**, not as the selection object. |
| Stage 2: proximity / stub lag-band on features | **REMOVE** / **PATCH REQUIRED** | Draft Stage 2 is event-anchored temporal qualification of forks (min lead time, matched opportunities) — not C2-style feature proximity. Current Stage 2 is a stub (`passed = True` unless prox change tiny). |
| Stage 3: branch **at pressure onset**; gate features by actionability rate | **PATCH REQUIRED** | Must branch at candidate fork states from backward trace; gate = item-9 R4 / controllability screen; no feature promotion. |
| `_run_stage_3` call signature vs `run_counterfactual_branches` | **PATCH REQUIRED** (bug) | Runner passes kwargs (`snapshot_env_fn`, `policy`, `core`, …) that **do not match** the analysis module API. Also treats `compute_actionability` return as `dict` with `is_actionable` — actual return is `ActionabilityResult` dataclass **without** that field. Stage 3 path is currently **broken** even on the old contract. |
| Per-episode “no commitment fork” emission | **MISSING** | Only global no-qualified-**features** JSON. |
| Explicit Stage-3 “controllability only / no latent claim” in artifacts | **MISSING** | Should be stamped in outputs. |

---

## Contract-critical gaps (patch list, ordered)

Do **not** expand methodology. Patch to the settled draft:

1. **Remove pickup from onset.** Pressure-crossing only as **anchor**.
2. **Implement backward trace** (`T_trace`) selecting **earliest** state satisfying R1–R4; else emit no-fork for that anchor.
3. **Legal team response enumeration** (≥2) with legality pruning; joint options.
4. **Replace absolute `outcome_shift` gate** with improvement \(A(s)\) vs G0 using named `U` over horizon; threshold \(\delta\).
5. **Add doomed/recoverability reject** (R3).
6. **Persist-over-`H_response`** check (R4 persistence).
7. **Retarget Stages 1–3** from feature ranking → fork discovery / qualification / controllability screen.
8. **Fix Stage-3 API mismatch** (runner ↔ `counterfactual_actionability`).
9. **Feature math fixes** (TTI sentinel, relative velocity, signed commitment) — item 8 adjacent.
10. Keep auth guard; keep map/rules/reward untouched; keep Stage 4 out of this runner.

---

## What may be reused as-is

- Auth / hash verification  
- G0 load + `map_a` / V2 env construction  
- Snapshot/restore + determinism self-test  
- One-step override then natural continuation skeleton  
- Instantaneous feature extraction **structure** (after math patches)  
- Progress logging  

---

## Item 10 status

```text
ITEM 10 AUDIT     COMPLETE
ITEM 10 IMPLEMENTATION PATCHED
all structural contract gaps resolved
focused contract tests passing
NO C3 SCAN RUN

NEXT  close items 1–8 and freeze T_trace / H_response / delta / U
THEN  freeze contract → machine JSON → commit/hash → authorization artifact
      → tiny smoke → full scan
```

### Patch closure evidence

The patch uses the audit above as the change boundary:

```text
Stage-3 API          corrected typed branch-set and ActionabilityResult calls
pressure event       anchor only; flag pickup excluded
backward trace       bounded by frozen T_trace; chronological; earliest pass
team responses       Cartesian product of authoritative env macro masks
legal support        at least two legal team responses required
doomed state         rejected using the frozen U definition's doomed boundary
actionability        max_a E[U|s,a] - E[U|s,G0] >= delta; harmful shifts ignored
persistence          utility measured through frozen H_response
no-fork result       per-anchor NO_COMMITMENT_FORK; pressure never auto-promoted
claim boundary       CONTROLLABILITY_SCREEN_ONLY; O3 authorization always false
execution            hash authorization guard remains first; no Stage-4 path added
```

Owning focused tests:

```text
tests/test_counterfactual_actionability.py
tests/test_decision_proximal_features.py
tests/test_c3_commitment_fork_runner_contract.py
```

No discovery scan was run. No threshold, reward, ruleset, map, opponent,
feature family, or scientific claim was added by the patch.
