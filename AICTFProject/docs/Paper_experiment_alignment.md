# Paper experiment alignment (code ↔ manuscript)

This file is the **single source of truth** for what `AICTFProject` implements. Any external write-up should follow these definitions.

---

## 1. Centralized critic: **\(V_\phi(s,\mathbf{a},z)\)** (PPO value regression)

The centralized PPO target uses a **scalar value** :math:`V_\phi(s,\mathbf{a},z)` (see
`CentralizedCritic` in `rl/networks.py` and `SharedActorCentralizedCritic.values` in
`rl/custom_ppo.py`). It is **not** a tabular or DQN-style :math:`Q(s,\mathbf{a},z)` over
counterfactual joint actions.

```102:115:rl/networks.py
class CentralizedCritic(nn.Module):
    """
    Centralized **scalar value function** :math:`V_\\phi(s, \\mathbf{a}, z)` for clipped PPO / GAE.
    ...
```

The actor stack wires **`extra_dim = joint_action_onehot + latent_k`** when latent is on (`SharedActorCentralizedCritic` in `rl/custom_ppo.py`).

---

## 2. Persistence \(\lambda_p\) and entropy \(\lambda_H\): **event-masked** (exact semantics)

Rollout stores **`z_resampled`** and **`z_persist_mask`** from `_strategy_for_step`.

**When \(z\) is considered “resampled” (`z_resampled` / `resample_mask`):**

- Start of episode: env needs an initial \(z\) draw (`_needs_strategy_sample`).
- Mid-episode (Option B): `latent_resample_every_n > 0` and strategy age reaches that interval.
- Optional: `latent_resample_on_flag` per config.

**`persist_mask` (used only for \(\mathcal{L}_{\text{persist}}\)):**

```1687:1718:rl/custom_ppo.py
        resample_mask = self._needs_strategy_sample.clone()
        if self.latent_resample_every_n > 0:
            resample_mask |= self._strategy_age >= self.latent_resample_every_n

        prev_z = self._current_z.clone()
        z_idx = self._current_z.clone()
        persist_mask = resample_mask & (~self._needs_strategy_sample)
        ...
        aux = {
            ...
            "z_resampled": resample_mask,
            "z_persist_mask": persist_mask,
        }
```

So:

- **Episode-start** draws have `persist_mask == False` → they **do not** contribute to \(\lambda_p \mathbb{1}[z_t \neq z_{t-1}]\) (avoids penalizing mandatory first draw vs arbitrary `prev_z`).
- **Mid-episode refresh** rows have `persist_mask == True` when the timer (or combined mask) fires and the env is not in “needs first sample” state.

**Loss construction (`update`):**

```2506:2534:rl/custom_ppo.py
                if self.use_latent_strategy:
                    resample = batch["z_resampled"].bool()
                    persist_mask = batch["z_persist_mask"].bool()
                    ...
                    if bool(resample.any().item()):
                        h_mean = strategy_entropy[resample].mean()
                    else:
                        h_mean = torch.zeros((), dtype=torch.float32, device=self.device)
                    ...
                    switch = paper_strategy_switch_indicator(batch["z"], batch["prev_z"])
                    if bool(persist_mask.any().item()):
                        persist_loss = switch[persist_mask].mean()
                    else:
                        persist_loss = torch.zeros((), dtype=torch.float32, device=self.device)
                    if self.latent_resample_every_n == 0 and not self.latent_resample_on_flag:
                        assert persist_loss.item() == 0.0, (
                            "L_persist must be exactly 0 when no mid-episode resampling ..."
                        )
                    latent_loss = float(getattr(self.cfg, "latent_lam_p", 0.0)) * persist_loss + strategy_entropy_loss
```

**Implementation summary (for any write-up):**

- \(\mathcal{L}_{\text{persist}} = \mathbb{E}[\mathbb{1}[z_t \neq z_{t-1}] \mid \text{persist\_mask}]\): mean hard switch **only over `persist_mask` timesteps** (mid-episode resample decisions), not over every environment step.
- **Entropy term** uses **mean \(H(q_\phi)\)** over **`z_resampled`** timesteps (includes episode-start draws **and** sparse refreshes). Sign: **`latent_entropy_objective=maximize`** → add **\(-\lambda_H \bar{H}\)** to the minimized loss (see comments in the same block).

Hard switch definition: `paper_strategy_switch_indicator` in `rl/latent_marl.py`.

---

## 3. Strategy inference \(q_\phi(z \mid s_t)\): **plan global summary vector only**

**Specification:** \(q_\phi\) consumes the **fixed-dimension global summary** \(s_t \in \mathbb{R}^{D}\) built for CTDE (`GLOBAL_STATE_DIM = 19`). That vector **already encodes** team means/spreads, flag distances, capture bits, mean speeds, and score/clock pressure (see field list in `rl/global_state.py`). There is **no separate** “team statistics tensor” branch beyond this summary.

```29:56:rl/global_state.py
GLOBAL_STATE_DIM: int = 19
...
GLOBAL_STATE_FIELD_NAMES: tuple[str, ...] = (
    "blue_mean_x",
    "blue_mean_y",
    ...
    "sim_time_frac",
)
```

```35:47:rl/latent_marl.py
    def __init__(self, state_dim: int, latent_k: int, hidden: int = 128) -> None:
        ...
    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """Return strategy logits with shape ``(B, K)``."""
        return self.net(global_state.float())
```

---

## 4. Frozen eval matrix (abstract / tables)

Lock **one** eval protocol and reuse it for every row of the results table so numbers are comparable.

### 4.1 Training rows (methods ↔ `rl/train_ppo.py`)

| Row | Description | Suggested preset / CLI |
|-----|-------------|------------------------|
| **Ours (A)** | Discrete \(z\), **episode-start** resample only | `--preset latent_a1_plan_faithful` (or plan-faithful A1 variant you freeze in `run_config.json`) |
| **Ours (B)** | Sparse refresh + \(\lambda_p\) on refresh | e.g. `--preset hypothesis_latent_opprand_optionb_lamp_coef05_op35` |
| **Flat MARL** | Same env/reward, no \(z\) | `--preset hypothesis_flat_opprand_op35` or `--no-latent-strategy` with matched `--opponent-pool` |
| **Curriculum** | Scripted OP1→OP2→OP3 baseline | `--mode CURRICULUM` (trainer **forces latent off**; see `train_ppo.py` `train_ppo`) |

Keep **`--seed`**, **`--agents`**, **`--map-set` / pool**, and **timesteps** fixed per comparison unless the claim is explicitly about scaling.

### 4.2 Team sizes (abstract “up to six ASVs”)

The CLI supports **`--agents 2|4|6|8`**. For abstract faithfulness, report at least:

- **2v2** (main),
- **one larger team** (e.g. **4v4** or **6v6**) with the **same** method row names and eval protocol below.

### 4.3 Eval protocol (`plot/eval_checkpoint.py`)

Use identical arguments for every checkpoint:

| Parameter | Frozen suggestion |
|-----------|-------------------|
| Script | `python plot/eval_checkpoint.py` |
| `--agents` | Match training |
| `--opponents` | `OP3 OP5_RUSHER` (and `OP4` if the paper claims held-out style) |
| `--map-sets` | `train eval` (minimum: add **eval** if train WR saturates) |
| `--episodes` | ≥ 100 per (map_set, opponent); use 300–500 for tighter CIs in final tables |
| `--device` | `cuda` if available |
| `--seed` | Same int across all methods |
| `--deterministic` | Default greedy eval (script default) |

Record **`OP5_RUSHER_TUNING_TAG`** from `opponent_params.py` in any published table caption whenever OP5 appears.

### 4.4 Optional E3 / MI rows

If you claim coordination / \(z\)–behavior alignment, enable **`--e3-step-telemetry`** on latent training runs and cite columns from `E3_STEP_TELEMETRY_FIELDS` in `rl/custom_ppo.py` plus `plot/analyze_e3_latent_mi.py`.

---

## 5. Optional auxiliary (off by default)

**`latent_strategy_aux_return_head`:** Plan A2-style **auxiliary return regression** on the shared \(q_\phi\) trunk (per-\(z\) scalar predictions on **sampled** \(z\) only). This is **not** a full \(Q(s,\mathbf{a},z)\) critic and **not** off-policy \(Q\)-learning; the centralized critic remains on-policy \(V_\phi(s,\mathbf{a},z)\) with PPO targets. Default **`aux_return_head=False`** keeps the main path to task reward + entropy + persistence only (see `PPOConfig` in `rl/train_ppo.py`).

---

## 6. `q_phi` gradient channels (strict-Summer vs. v4i3 vs. v3c-episode-credit)

`docs/algorithm.md` locks the latent-strategy loss as

\[
L = L_{\text{PPO}} + \lambda_p\, L_{\text{persist}} - \lambda_H\, H\!\big(q_\phi(z\mid s)\big),
\]

with the clause *"PPO clipped ratio uses action log-probs only; `q_phi` is trained through strategy entropy and persistence, plus optional consecutive KL."* The implementation now supports three intentional supersets, all sharing the same actor / critic / opponent pool / map / budget so they are comparable as rows in the proof table.

### 6.1 Strict-Summer (preset `latent_v5_strict_summer`)

Literal reading of the equation above. The only gradient routes into `q_phi` are:

1. \(-\lambda_H \bar{H}\) over `z_resampled` steps (differentiable in `q_phi`).
2. \(\lambda_p\,\mathcal{L}_{\text{persist}}\) over `z_persist_mask` steps. NB: `strategy_persistence_loss` in `rl/latent_losses.py` uses the **differentiable** form \(1 - p_\phi(z_t = z_{t-1}\mid s_t)\) (a.k.a. `expected_strategy_switch_penalty`), not the hard indicator quoted in §2 above. The hard `paper_strategy_switch_indicator` is only used for diagnostic statistics; the gradient-bearing term is the soft form.
3. (Optional) `latent_kl_consecutive > 0` → KL\((q_\phi(s_t)\,\|\,q_\phi(s_{t-1}))\).

`latent_strategy_ppo_coef = 0`, `latent_arc_credit_enabled = False`, `latent_episode_strategy_ppo = False`, no aux heads, FiLM and z-onehot off — actor receives `z` strictly via `nn.Embedding(K, d_z)` per the algorithm spec.

### 6.2 v4i3 Summer-proof + arc-credit (preset `latent_v4i3_summer_proof`)

Same regularizers as strict-Summer, **plus** the v3i19 arc-credit channel: at every z-arc boundary `q_phi` receives a clipped PPO PG signal with `arc_return - V_\phi(\text{ctx}, z)` as advantage (`baseline = "context_value"`, `coef = 1.0`). No labels and no auxiliary prediction heads are introduced — the credit signal is the same task reward summed over the arc. This is the strongest "Summer-faithful in spirit" run because the only extra ingredient over §6.1 is *task reward at a different temporal aggregation*.

### 6.3 v3c-style episode-credit (preset `latent_v3c_router_lr` / `latent_episode_strategic`)

Replaces arc-credit with a per-episode PPO update on `q_phi`, optionally through a dedicated `latent_router_optimizer` (set when `latent_episode_strategy_lr is not None`). Useful when the per-arc credit is too sparse but the full-episode return is well-defined.

### 6.4 Why the main-loop gate matters (v5 fix)

Before this PR, `ppo_updater.update` zeroed the main-loop `q_phi` gradient routes (entropy, persistence, KL, strategy-PPO, aux-return) whenever `latent_strategy_ppo_coef == 0`. That single gate was originally introduced to prevent double-stepping when a dedicated router optimizer (`latent_router_optimizer`) was active alongside the shared optimizer — the v3c "Fix 5" double-step bug. Side-effect: in v3i19 / v4i1 / v4i3 (no dedicated optimizer, `latent_strategy_ppo_coef = 0`, arc-credit on), the configured `lam_p` and `lam_h` schedules were *silently zeroed* and only arc-credit reached `q_phi`. Telemetry would still print the computed `lamH`/`persist` values, but the backward pass never saw them.

The v5 gate triggers off the actual double-step hazard instead:

```text
has_dedicated_router_opt = (runtime.latent_router_optimizer is not None)
apply_strategy_ppo  = latent_strategy_ppo_coef > 0  and not has_dedicated_router_opt
apply_persistence   = (lam_p > 0 or sparse_tactical_refresh) and not has_dedicated_router_opt
apply_entropy       = lam_h > 0 and entropy_objective != "none" and not has_dedicated_router_opt
apply_kl            = lam_kl_consecutive > 0 and not has_dedicated_router_opt
```

Behavior change: v3i19 / v4i1 / v4i3 now *actually* apply their configured `lam_p = 0.03` and `lam_h` schedule via the main update, in addition to arc-credit. Earlier runs (checkpoints with this PR's hash older than the gate fix) trained `q_phi` via arc-credit only — their `lamH`/`persist` telemetry was a no-op. Pin the run-tag to the PR hash when comparing across the change.

### 6.5 Proof-table mapping

| Row | Preset alias | `q_phi` gradient channels |
|-----|-------------|---------------------------|
| Strict-Summer (literal) | `v5_strict_summer` | entropy + persistence (+ KL) -- no task-reward signal on `q_phi` |
| Conditional-entropy paper-faithful | `v5i4` / `v5i4_paper_faithful` | strict-Summer + per-step main-loop categorical PPO on `q_phi` (`latent_strategy_ppo_coef = 0.10`) + mean conditional entropy |
| Conditional entropy-floor ablation | `v5i5` / `v5i5_paper_faithful_entropy_floor` | v5i4 with `latent_lam_h_end = 0.001`; still mean conditional entropy |
| Summer-faithful split-lane row | `v5i7` / `v5i7_summer_faithful_split_lane` | v5i5 entropy-floor contract on `map_b_split_lane`; resolved diff vs v5i5 is exactly `map_layout` and `run_tag` |
| **Paper-faithful canonical** | **`v5i6` / `v5i6_paper_faithful`** | **v5i4 with batch-marginal entropy `H(E_s[q_phi(z|s)])` and the v5i5 lambda_H floor; no new labels, curriculum, actor path, or auxiliary channel** |
| v4i3 + arc-credit | `latent_v4i3_summer_proof` | strict-Summer + per-arc clipped PG (post-Summer extension) |
| v5i1 episode-credit | `latent_v5i1_reward_credit_router` | strict-Summer + per-episode clipped PG with a dedicated AdamW |
| v3c episode-credit | `latent_episode_strategic` | per-episode clipped PG (dedicated optimizer) |
| K=1 collapsed | `plan_faithful_latent_k1` | n/a (latent collapsed) |
| No-latent | `no_latent_v4i3_baseline` | n/a |

Same opponent pool (`OP5 OP6 OP7`), same seed set, same map, same total timesteps. `v5_strict_summer` tests whether the literal docs/algorithm.md loss alone can train `q_phi`; `v5i4` tests the same architecture plus the on-policy categorical PG term on `q_phi`; `v5i6` keeps that task-reward channel and clarifies `H(z)` as batch-marginal entropy over the router training batch. The row that beats no-latent under v5i6 supports the canonical Summer interpretation; v5i6 vs v5i4/v5i5 isolates the entropy reduction choice.

---

## 6.6 v5i3 forced-z anneal (coverage fix, not a new gradient channel)

v5i2 telemetry showed `q_phi` collapsing onto a single latent (`z2`) within the first 200k steps, with `z1` reaching <5% occupancy by 540k. The actor's per-z sensitivity grew steadily under FiLM, but only on the latents `q_phi` actually picked. Under-sampled latents stayed effectively untrained because the actor never saw enough of them to learn `pi(a|o, z_under)`, which made their episode returns arbitrary, which made `q_phi`'s gradient on them noise. This is a coverage problem, not a credit-assignment problem.

`v5i3_balanced_warmup` adds the smallest possible fix that keeps the loss objective unchanged: an annealed *exploration* fraction of episodes is forced onto a uniformly-sampled `z` before the rollout begins. The schedule is:

```text
global_step      0 - 200k  : forced fraction = 0.30 (constant)
                200k - 500k: linearly anneal 0.30 -> 0.00
                500k - 1M  : forced fraction = 0.00 (router only)
```

Configuration fields in `PPOConfig` (resolved by `rl.custom_ppo.schedules.resolve_latent_forced_z_frac`):

```text
latent_forced_z_episode_frac_start = 0.30
latent_forced_z_episode_frac_end   = 0.00
latent_forced_z_anneal_start       = 200_000
latent_forced_z_anneal_end         = 500_000
```

When any of these four fields is `None` the resolver falls back to the legacy constant `latent_forced_z_episode_frac`, so every pre-v5i3 preset (including v5i2 with its `0.0` constant) is bit-stable.

### Off-policy / Summer-plan invariants preserved

The forcing must not introduce off-policy bias into `q_phi`'s PPO update or convert v5i3 into a supervised-strategy preset. Both invariants are pinned by the existing routing in `rl/custom_ppo/latent_strategy_state.py`:

1. Forced episodes still enter the normal rollout buffer for actor + critic learning (the actor's PPO update is on-policy with respect to whichever `z` was used; that `z` is observed at the actor's input so the update is well-defined regardless of how `z` was chosen).
2. Forced episodes never set `episode_strategy_has_start`, so they are excluded from `rollout_strategy_episode_records`.
3. They early-return into `latent_preference_buffer` instead, which `apply_episode_strategy_ppo` does NOT consume for the router PPO gradient (it only reads from `rollout_strategy_episode_records`).
4. `latent_router_distill_enabled = False` and `latent_v3i3_event_preference_enabled = False` are pinned in the preset so the preference buffer is never read back into the router objective.

The result: `q_phi`'s PPO update sees only router-sampled, on-policy episodes; the forced episodes contribute only via the actor learning to handle previously under-sampled `z` values. This remains Summer-compatible because the forcing is unlabeled uniform exploration, not supervised role assignment.

### Resume safety

`resolve_latent_forced_z_frac(cfg, global_step=...)` is a pure function of `cfg` and the passed `global_step`. The trainer restores `self.global_step` from the checkpoint before the rollout loop resumes (`rl/custom_ppo/trainer.py` `load()`), so resuming mid-anneal picks up the schedule at the restored step rather than re-starting from `_start`. Pinned by `ForcedZScheduleResolverTests.test_resume_uses_passed_global_step_not_internal_state` and `ForcedZRuntimeRoutingTests.test_resume_at_mid_anneal_resolves_correctly` in `tests/test_forced_z_anneal.py`.

### Per-z router telemetry

`apply_episode_strategy_ppo` now emits per-`z` aggregates over the on-policy batch:

```text
router_sample_count_by_z_{i}      : # of router-sampled episodes with z=i this rollout
forced_sample_count_by_z_{i}      : # of forced episodes with z=i this rollout
episode_count_by_z_{i}            : router + forced count
mean_episode_advantage_by_z_{i}   : mean PPO advantage for episodes that chose z=i
std_episode_advantage_by_z_{i}    : std of those advantages (noise indicator)
mean_return_by_z_{i}              : mean raw episode return for those episodes
mean_logprob_ratio_by_z_{i}       : mean PPO ratio at the final inner epoch
clip_fraction_by_z_{i}            : fraction of those records that hit the clip
latent_forced_z_episode_frac_current : schedule output at the start of the update
```

These distinguish two failure modes the v5i2 post-mortem could not separate:

* `router_sample_count_by_z_1 == 0` + everything else 0: `z1` starved -- needs more exploration.
* `router_sample_count_by_z_1 > 0` + `std_episode_advantage_by_z_1` huge + `mean_episode_advantage_by_z_1 ~ 0`: `z1` sampled enough but receives noisy/weak credit -- the actor still cannot execute it well, or its true return is genuinely similar to other `z`'s in this opponent mix.

### Matched-schedule random-router evaluation

`plot/eval_checkpoint.py --latent-selection` now exposes four modes:

```text
--latent-selection router          : trained q_phi(z|s) (default)
--latent-selection random-matched  : uniform-random z, resampled at the same decision steps
                                     the router would have (inherits the checkpoint's
                                     latent_resample_every_n; override with --latent-resample-every)
--latent-selection random-episode  : uniform-random z, episode start only
                                     (forces strategy_interval=0 regardless of training config)
--latent-selection fixed           : clamp every episode to --fixed-latent-id
```

The decisive comparison is `router` vs `random-matched` with the same checkpoint and the same `--seed`, which isolates routing quality from actor quality (the actor weights are identical; only the `z` distribution at each decision step differs).

---

## 6.7 v5i4 paper-faithful end-to-end (task-reward PG on `q_phi` as part of `L_MARL`)

`v5_strict_summer` exposed an operational problem with the literal docs/algorithm.md loss: with only entropy and persistence on `q_phi`, the router has no signal that distinguishes which `z` improved performance. The paper's claim "the strategy inference network is trained end-to-end from task reward" requires a score-function gradient on the discrete latent. A discrete `q_phi` cannot inherit that gradient from the actor's PPO step merely because the actor consumes its sampled `z`; the gradient stops at the embedding-lookup. The standard fix is an on-policy categorical PPO term on `q_phi`, included **inside `L_MARL`**, not as an auxiliary task.

`v5i4_paper_faithful_end_to_end` is the conditional-entropy reference row that turns `v5_strict_summer` into an end-to-end task-reward row by enabling exactly that term. The canonical operational row is now `v5i6`, which keeps this task-reward channel and changes the entropy reduction to the batch marginal. The v5i4 loss is:

```text
L = L_actor_PPO + c_V * L_critic + c_Z * L_strategy_PPO + lam_p * L_persist - lam_H * E_s[H(q_phi(z | s))]
```

with

```text
L_strategy_PPO = - E_t [ min( rho_t(z) * A_t , clip(rho_t(z), 1 +/- eps) * A_t ) ]
rho_t(z)       = pi_phi(z | s_t) / pi_phi_old(z | s_t)
A_t            = GAE advantage from the centralized critic V(s, a, z)
```

evaluated only on the resample subset (the decision steps where a new `z` was sampled). The `c_Z` weight is `latent_strategy_ppo_coef = 0.10` and the term lives inside the same backward pass as the actor / critic / persistence / entropy terms (no dedicated router optimizer). The trainer's main-loop gate fires this PG term independently of `latent_episode_strategy_ppo` (which is the v5i1 episode-credit extension and stays OFF in v5i4); see `rl/custom_ppo/ppo_updater.py` and `MainLoopGatingTests` in `tests/test_marginal_baseline.py` for the gate semantics.

### Inheritance and forbidden-channel contract

| Channel | v5_strict_summer | **v5i4** | Why for v5i4 |
|---------|------------------|----------|--------------|
| Main-loop categorical strategy PPO (`latent_strategy_ppo_coef`) | 0.0 | **0.10** | The single change that makes the router actually learn from task reward. |
| Episode-credit (`latent_episode_strategy_ppo` + dedicated AdamW) | OFF | OFF | Mutually exclusive with the main-loop PG above (the dedicated optimizer would silence the main-loop gate). v5i1's extension, not the paper's mechanism. |
| Arc-credit (`latent_arc_credit_enabled`) | OFF | OFF | v3i19 / v4i3 extension. Not in the paper. |
| FiLM (`enable_actor_z_film`) | OFF | OFF | v5i2 extension. The paper's actor reads `z` via plain `nn.Embedding(K, d_z)` concat. |
| Forced-z curriculum (`latent_forced_z_episode_frac_*`) | OFF | OFF | v5i3 extension. The router learns purely from on-policy reward; no scheduled uniform sampling. |
| Auxiliary heads (return / phase / opponent prediction) | OFF | OFF | Forbidden by "no auxiliary prediction tasks". |
| Preferences / distillation (`latent_v3i3_event_preference_enabled`, `latent_router_distill_enabled`) | OFF | OFF | Post-Summer extensions with labels / teacher targets. |
| Persistence (`latent_lam_p`) | 0.03 | 0.03 | Paper regularizer. |
| Entropy maximization (`latent_lam_h`, `latent_entropy_objective = "maximize"`) | 0.003 | 0.003 | Paper regularizer; sign pinned so the entropy term is `- lam_H * H` (loss-minimization convention). |
| Resampling cadence (`latent_resample_every_n`) | 64 | 64 | Sparse switching per the paper. |
| Flag-triggered switching (`latent_resample_on_flag`) | False | False | Not in the paper's switching rule. |

The aliases `v5i4`, `v5i4_paper_faithful`, `v5i4_end_to_end`, `paper_faithful_end_to_end`, `latent_v5i4_paper_faithful`, `latent_v5i4_end_to_end`, and `plan_faithful_latent_v5i4_end_to_end` all resolve to the same `PPOConfig`. Pinned by `V5i4AliasSnapshotTests.test_all_aliases_resolve_to_identical_config` and the snapshot in `tests/preset_snapshots.json`.

### Launch-time audit banner

`rl/training/banner.py::_maybe_print_paper_faithful_audit` emits an invariant block when `cfg.run_tag` contains `"v5i4_paper_faithful"` (or `cfg.latent_paper_faithful_audit = True`). It lists every channel above with its resolved ON/OFF state and emits explicit `[PPO] v5i4 audit WARNING` lines if any of the documented mis-configurations are detected:

* `latent_strategy_ppo_coef <= 0` -- `q_phi` would receive no task-reward gradient.
* `latent_episode_strategy_lr is not None` -- the dedicated router optimizer would silence the main-loop PG term.
* FiLM / adapter / one-hot ON -- the actor-z pathway would no longer be the paper-literal concat one.

The actor input-dim block should still print `cnn(128) + per_agent_vec(20) + z_emb(16) = 164`. No phase, opponent identity, or global state enters the actor pathway.

### Required tests (pinned in `tests/test_v5i4_paper_faithful.py`)

1. **Preset inheritance:** v5i4 derives from `v5_strict_summer`; `latent_strategy_ppo_coef` is the only main-loop PG channel flipped on; FiLM / episode-credit / forced-z all stay OFF (so v5i4 does NOT silently inherit v5i1 / v5i2 / v5i3 behavior).
2. **Concat-only actor:** `enable_actor_z_film == False`, adapter / one-hot off, actor input dim `= cnn_feat(128) + 20 + z_embed_dim(16) = 164`.
3. **No-curriculum:** `resolve_latent_forced_z_frac(cfg, global_step=step)` returns `0.0` at every step.
4. **Router task-gradient is ON:** with nonzero advantages on the resample subset, `strategy_ppo_loss` produces a nonzero gradient through `log pi_phi(z|s)` that concentrates on the resample subset (the non-resample subset receives exactly zero gradient).
5. **Zero-advantage produces zero policy_loss:** with zero advantages, the categorical PPO term contributes exactly zero policy_loss.
6. **No forbidden channels:** episode-credit, arc-credit, aux heads, preferences, distillation, specialist router, behavior contrast, marginal balance, conditional entropy minimization are all OFF / zero-coef.
7. **Sparse 64-step resampling:** `latent_resample_every_n == 64`, `latent_resample_on_flag is False`, `use_latent_strategy is True`, `fixed_latent_strategy is False`, `latent_k == 4`.
8. **Alias snapshot:** all seven aliases resolve to byte-identical `PPOConfig` dicts.
9. **Banner:** the audit fires for v5i4 and stays silent for `v5_strict_summer`; the three mis-configuration paths each trigger their warning line.

### Launch command

```powershell
.\.venv\Scripts\python.exe rl/train_ppo.py `
    --preset v5i4_paper_faithful `
    --total-steps 1000000 `
    --agents 4 `
    --seed 0 `
    --device cuda `
    --n-envs 32 `
    --n-epochs 6 `
    --e3-step-telemetry `
    --checkpoint-dir checkpoints/4v4 `
    --fresh-metrics-csv `
    --periodic-checkpoint-steps 50000
```

The decisive comparison for the paper's "explicit strategy abstraction improves win rate" claim is:

* **v5_strict_summer** vs **v5i4** on the same opponent pool, seed set, and budget. The delta isolates the contribution of the on-policy categorical PG term on `q_phi`.
* **v5i4** vs **no_latent_v4i3_baseline** on the same opponent pool, seed set, and budget. The delta isolates the contribution of the *entire* latent mechanism, with the literal Summer architecture and a single (and minimal) task-reward gradient channel on `q_phi`.
* **v5i4 router** vs **v5i4 random-matched** at evaluation time (`plot/eval_checkpoint.py --latent-selection {router|random-matched}`). The delta isolates routing quality from actor quality at fixed actor weights.

---

## 6.8 v5i6 canonical marginal-entropy interpretation

`v5i6_paper_faithful_marginal_entropy` inherits v5i4 directly and changes
the entropy reduction, not the actor, critic, router PPO, persistence,
sampling cadence, reward, or supervision surface. The canonical loss is:

```text
L = L_actor_PPO + c_V * L_critic + c_Z * L_strategy_PPO
    + lam_p * L_persist + lam_H * KL(q_bar || Uniform)

q_bar(z) = E_s[q_phi(z | s)]
```

Minimizing `KL(q_bar || Uniform)` is equivalent to maximizing
`H(q_bar)` up to the constant `log(K)`. The term is implemented in
`rl/latent_losses.py::strategy_marginal_entropy_loss` and is selected
by `latent_entropy_mode = "marginal"` inside
`rl/custom_ppo/ppo_updater.py`.

Resolved-config contract:

```text
v5i4 -> v5i6: latent_entropy_mode, latent_lam_h_end, run_tag
v5i5 -> v5i6: latent_entropy_mode, run_tag
```

`latent_usage_balance_coef` stays `0.0`; v5i6 uses the shared
`lambda_H` schedule for marginal entropy rather than the legacy
episode-router usage-balance coefficient. The audit banner prints
`v5i6 paper-faithful audit` and reports `mode=marginal`.

---

## 7. Changelog

- **v6i17 / surface-pressure diagnostic:** Registered
  `apply_plan_faithful_latent_v6i17_surface_pressure_diagnostic` as a
  harder/asymmetric opponent-surface diagnostic over the v6i16 combined
  capacity + sharp-contract scaffold. Classification: `DIAGNOSTIC`
  non-Summer scaffold, not a paper-faithful row and not a Summer-compatible
  extension. The resolved diff vs `v6i16_capacity_sharp_contracts` is exactly
  `{experiment_id, opponent_pool, run_tag}`: the opponent pool expands from
  OP8/OP9/OP10 to OP8/OP9/OP10/OP11/OP12 while router training remains off,
  `balanced_episode` z assignment remains active, and the inherited sharp 3x
  contract/capacity settings stay unchanged. Extended the training opponent
  allowlist to preserve OP11/OP12 through validation. Added focused tests in
  `tests/test_v6i17_surface_pressure_diagnostic.py` and regenerated
  `tests/preset_snapshots.json`.
- **v6i16 / capacity + sharp-contract diagnostic:** Registered
  `apply_plan_faithful_latent_v6i16_sharp_contracts`,
  `apply_plan_faithful_latent_v6i16_capacity`, and
  `apply_plan_faithful_latent_v6i16_capacity_sharp_contracts` as a
  three-arm diagnostic over `v6i15_contract_pressure_3x`. Classification:
  `DIAGNOSTIC` non-Summer scaffold, not a paper-faithful row and not a
  Summer-compatible extension. The sharp-contract arm changes exactly
  `{experiment_id, latent_contract_specialist_variant, run_tag}`; the
  capacity arm changes exactly `{experiment_id,
  latent_actor_z_adapter_enabled, latent_actor_z_adapter_init_std,
  latent_actor_z_adapter_scale, latent_z_gate_init, run_tag}`; the
  combined arm changes the union of those fields. `v6i16` resolves to the
  combined capacity + sharp-contract arm. Added default-off
  `PPOConfig.latent_contract_specialist_variant`, added the `"sharp"`
  contract reward variant, made `z_adapter` trainable in the repertoire
  freeze allowlist when enabled, added focused tests in
  `tests/test_v6i16_capacity_feature_ablation.py`, and regenerated
  `tests/preset_snapshots.json`.
- **v6i15 / contract-pressure coefficient sweep:** Registered
  `apply_plan_faithful_latent_v6i15_contract_pressure_3x`,
  `apply_plan_faithful_latent_v6i15_contract_pressure_6x`, and
  `apply_plan_faithful_latent_v6i15_contract_pressure_10x` as direct
  pressure arms over `v6i14_contract_specialists`. Classification:
  `DIAGNOSTIC` non-Summer scaffold, not a paper-faithful row and not a
  Summer-compatible extension. The resolved diff vs v6i14 is exactly
  `{experiment_id, latent_contract_specialist_coef, run_tag}`; the
  coefficients are `0.75`, `1.50`, and `2.50`. `v6i15` and
  `v6i15_contract_pressure` resolve to the 3x arm. Added focused tests in
  `tests/test_v6i15_contract_pressure.py` and regenerated
  `tests/preset_snapshots.json` for the new aliases.
- **v6i14 / contract-specialist repertoire diagnostic:** Registered
  `apply_plan_faithful_latent_v6i14_contract_specialists` (aliases
  `v6i14`, `v6i14_contract_specialists`,
  `v6i14_contract_specialist_repertoire`,
  `latent_v6i14_contract_specialists`, and the long
  `plan_faithful_latent_...` alias) built directly on
  `v6i9_mapaware_repertoire_hardpool`. Classification: `DIAGNOSTIC`
  non-Summer scaffold, not a paper-faithful row and not a
  Summer-compatible extension. The resolved diff vs the repertoire parent
  is exactly `{experiment_id, latent_contract_specialist_coef,
  latent_contract_specialist_enabled, run_tag}`. The runtime now supports
  default-off contract-specialist reward fields and stores the active bonus
  as `reward_contract_specialist`. Added focused tests in
  `tests/test_v6i14_contract_specialists.py`; regenerated
  `tests/preset_snapshots.json` (adds five v6i14 aliases and three new
  default-off `PPOConfig` fields to every snapshot entry).
- **v6i13 / opening-window delayed-commit advantage-router extension:**
  Registered
  `apply_plan_faithful_latent_v6i13_opening_window_advantage_router`
  (aliases `v6i13`, `v6i13_opening_window_advantage_router`,
  `v6i13_opening_window`, `v6i13_advantage_router`,
  `latent_v6i13_opening_window_advantage_router`, and the long
  `plan_faithful_latent_...` alias) built directly on
  `v6i12_advantage_router_hardpool`. Classification:
  `SUMMER-COMPATIBLE EXTENSION`, not a paper-faithful row. The resolved
  diff vs v6i12 is exactly
  `{experiment_id, latent_episode_strategy_warmup_decision_steps,
  router_arc_post_commit_only, router_opening_context_mode,
  router_warmup_uniform_z, run_tag}`. The runtime now supports
  default-off delayed-commit controls: uniform warmup z, post-commit-only
  arc opening, and finalized arc records with
  `opening_context = [state_0, state_commit, state_commit - state_0]`.
  Added `experiments/run_v6i13_opening_window_advantage_router.py`,
  focused tests in `tests/test_v6i13_opening_window_advantage_router.py`,
  and regenerated `tests/preset_snapshots.json` (adds six v6i13 aliases
  and the three new default-off `PPOConfig` fields to every snapshot
  entry).
- **v6i10 / episode-router exploration extension:** Registered
  `apply_plan_faithful_latent_v6i10_episode_router_explore_hardpool`
  (aliases `v6i10`, `v6i10_episode_router_explore_hardpool`,
  `v6i10_episode_router_explore`,
  `latent_v6i10_episode_router_explore_hardpool`, and the long
  `plan_faithful_latent_...` alias) built directly on
  `v6i9_mapaware_router_feedforward_hardpool`. Classification:
  `SUMMER-COMPATIBLE EXTENSION`, not a paper-faithful row. The resolved
  diff vs the feedforward parent is exactly
  `{experiment_id, h_mode, latent_arc_credit_baseline,
  latent_arc_credit_enabled, latent_arc_credit_min_len,
  latent_entropy_anneal_end, latent_entropy_anneal_start,
  latent_entropy_mode, latent_entropy_objective, latent_lam_h,
  latent_lam_h_end, latent_lam_p, latent_resample_every_n,
  latent_strategy_ppo_coef, learning_rate, router_ent_coef,
  router_uniform_exploration_prob, run_tag, strategy_interval}`. The
  preset holds one z for the whole episode, freezes the validated v6i9
  repertoire, replaces critic-based router PPO with running-mean
  episode arc credit, adds training-only 20 percent uniform exploration,
  uses marginal coverage, and keeps labels/opponent IDs/oracle-z
  supervision/aux heads/forced-z curriculum off. Added
  `PPOConfig.router_uniform_exploration_prob`, wired behavior log-probs
  through the same mixture used to sample z, added
  `tests/test_v6i10_episode_router_explore.py`, and regenerated
  `tests/preset_snapshots.json`.
- **v6i9 feedforward running-mean arc-credit treatment (A/B):** Registered
  `apply_plan_faithful_latent_v6i9_arc_credit_running_mean_feedforward_hardpool`
  (aliases `v6i9_arc_credit_running_mean_feedforward_hardpool`,
  `v6i9_arc_credit_feedforward`, and the long `plan_faithful_latent_...`
  alias) as the direct A/B treatment for the feedforward router control
  `v6i9_mapaware_router_feedforward_hardpool`. Classification:
  `SUMMER-COMPATIBLE EXTENSION` (arc credit is the documented v3i19
  post-Summer channel; not a paper-faithful row). The resolved-config diff
  vs the feedforward control is exactly four keys —
  `latent_arc_credit_enabled` (False→True), `latent_arc_credit_baseline`
  (context_value→running_mean), `latent_strategy_ppo_coef` (0.1→0.0, which
  removes the biased critic-based router advantage), and `run_tag`. The
  feedforward router architecture, 35-dim context, strategy interval,
  learning rate, entropy coefficient, opponent/map pool, frozen actor +
  z-specific parameters, seed, and training budget are held identical
  (pinned by `tests/test_v6i9_arc_credit_feedforward.py`). Also added raw
  (pre-normalization) arc-advantage telemetry
  (`latent_arc_baseline_mean`, `latent_arc_raw_advantage_mean/std`,
  `latent_arc_positive_fraction`, `latent_arc_running_mean_count/value`),
  persisted the arc running-mean EMA in the latent checkpoint schema
  (mirroring the macro channel), added the one-update smoke gate helper
  `rl/custom_ppo/diagnostics/arc_credit_smoke.py` +
  `experiments/run_arc_credit_treatment_smoke.py`, and regenerated
  `tests/preset_snapshots.json`.
- **v6i6 / evidence-gated repertoire expansion contract:** Registered
  `apply_plan_faithful_latent_v6i6_strategy_expansion` (aliases `v6i6`,
  `v6i6_strategy_expansion`, and long plan/latent aliases) built on
  v6i5. The preset is `SUMMER-COMPATIBLE EXTENSION`, not a
  paper-faithful row. It is fail-closed until a validated anchor
  manifest selects `anchors`, `expansion_target`, and `dormant` latents;
  no latent index is hardcoded in the preset. Added E1 config fields for
  fixed-z episode attribution, target-only trainable scope, reference
  critic opportunity weights, no-op adapter initialization, draw scoring,
  and anchor bitwise invariants. Added
  `--v6i6-anchor-validation-manifest`, focused tests in
  `tests/test_v6i6_strategy_expansion.py`, and regenerated
  `tests/preset_snapshots.json`.
- **v6i3 frozen confirmatory local-communication lineage:** Updated
  `apply_plan_faithful_latent_v6i3_strategy_local_comm` so Phase A
  promotion separates communication access from proof of communication
  value. The five v6i3 aliases now resolve to `comm_num_symbols = 5`,
  `comm_silence_symbol = 0`, `comm_message_grid_channels = 4`,
  `comm_min_valid_boundaries = 1024`, `comm_min_deliveries = 4096`,
  `comm_min_symbols_used = 2`, `comm_entropy_floor = 0.0`,
  `comm_symbol_dominance_ceiling = 1.0`,
  `comm_listener_jsd_margin = 0.001`,
  `comm_listener_min_passing_pairs = 3`,
  `comm_listener_min_states = 64`, and
  `comm_listener_consecutive_updates = 1`. Listener causal response is
  evaluated and logged during Phase A but is no longer a required gate
  family; final V6I3 communication-value claims require matched-seed
  listener response plus silence/shuffle degradation over v6i2. The v6i3
  gate fingerprint is `9ef168d941f046fb`; the v6i2 parent fingerprint is
  `224f1aea9ab36319`. Updated listener/transport telemetry for the
  silence-plus-four-active-symbol vocabulary, extended the v6i3 CSV
  schema with silence counters and symbol occupancy 0..4, added focused
  v6i3 integration tests, updated `tests/preset_snapshots.json` for the
  five v6i3 aliases, and recorded the frozen lineage in
  `v6i2-gate-protocol-freeze.md`, `v6i3-local-communication-spec.md`,
  `latent-preset-registry.md`, and `research-progress-tracker.md`.
- **v5i9 / CSIA-guided specialization extension:** Added
  `apply_plan_faithful_latent_v5i9_csia_guided_specialization` (aliases
  `v5i9`, `v5i9_csia`, `v5i9_csia_guided_specialization`, and long
  plan/latent aliases) built directly on v5i8. The resolved diff vs
  v5i8 is exactly `{csia_enabled, csia_reward_coef, run_tag}`:
  `csia_enabled = True`, `csia_reward_coef = 0.02`, and
  `run_tag = "v5i9_csia_guided_specialization_OP5_OP6_OP7_1m_4v4"`.
  v5i9 is `SUMMER-COMPATIBLE EXTENSION`, not original
  Summer-/paper-faithful. Added `rl/csia.py` for forced-z payoff
  matrices, centered interaction `S(opponent,z)`, gates A/B/C, and
  router-vs-random-vs-oracle metrics. Wired a detached `reward_csia`
  component into rollout collection and update CSV telemetry
  (`csia_interaction_strength`, `centered_advantage_matrix`,
  `oracle_best_z_per_opponent`, `router_oracle_gap`, `routing_gain`,
  `gate_A_pass`, `gate_B_pass`, `gate_C_pass`, `csia_bonus_active`).
  Added CLI overrides (`--csia-*`) and focused tests in
  `tests/test_csia.py` and
  `tests/test_v5i9_csia_guided_specialization.py`; regenerated
  `tests/preset_snapshots.json`.
- **v5i8 / Summer-faithful split-lane v2 task-pressure row:** Added
  `apply_plan_faithful_latent_v5i8_split_lane_v2_task_pressure` (aliases
  `v5i8`, `v5i8_split_lane_v2`, `v5i8_split_lane_v2_task_pressure`,
  `v5i8_summer_faithful_split_lane_v2`, and long plan/latent aliases)
  built directly on v5i7. The resolved diff vs v5i7 is exactly
  `{map_layout, run_tag}`: `map_layout = "map_b_split_lane_v2"` and
  `run_tag = "v5i8_summer_faithful_split_lane_v2_task_pressure_OP5_OP6_OP7_1m_4v4"`.
  It keeps conditional entropy, the v5i5 `lambda_H` floor, concat-only
  actor conditioning, sparse 64-decision resampling, no forced-z
  curriculum, no marginal entropy, no auxiliary heads, and no new
  `q_phi` gradient channel. Added lower-friction split-lane v2 geometry,
  route-context telemetry (`attack` / `return` / `intercept` crossings),
  OP5/OP6/OP7 lane-pressure patterns inside scripted opponent movement,
  the audit trigger, and `tests/test_v5i8_split_lane_v2_task_pressure.py`;
  regenerated `tests/preset_snapshots.json`.
- **v5i7 / Summer-faithful entropy-floor split-lane row:** Added
  `apply_plan_faithful_latent_v5i7_entropy_floor_split_lane` (aliases
  `v5i7`, `v5i7_split_lane`, `v5i7_entropy_floor_split_lane`,
  `v5i7_summer_faithful_entropy_floor_split_lane`,
  `v5i7_summer_faithful_split_lane`, and long plan/latent aliases)
  built directly on v5i5. The resolved diff vs v5i5 is exactly
  `{map_layout, run_tag}`: `map_layout = "map_b_split_lane"` and
  `run_tag = "v5i7_summer_faithful_entropy_floor_split_lane_OP5_OP6_OP7_1m_4v4"`.
  It keeps conditional entropy, the v5i5 `lambda_H` floor, concat-only
  actor conditioning, sparse 64-decision resampling, no forced-z
  curriculum, no marginal entropy, no auxiliary heads, and no new
  `q_phi` gradient channel. Added the audit trigger and
  `tests/test_v5i7_entropy_floor_split_lane.py`; regenerated
  `tests/preset_snapshots.json`.
- **v5i6 / canonical marginal-entropy interpretation:** Added
  `apply_plan_faithful_latent_v5i6_paper_faithful_marginal_entropy`
  (aliases `v5i6`, `v5i6_paper_faithful`,
  `v5i6_paper_faithful_marginal_entropy`,
  `paper_faithful_marginal_entropy`, and long plan/latent aliases)
  built directly on v5i4. Added `PPOConfig.latent_entropy_mode` with
  default `"conditional"` and the main-loop marginal entropy loss path
  in `rl/custom_ppo/ppo_updater.py`, backed by
  `rl/latent_losses.py::strategy_marginal_entropy_loss`. v5i6 sets
  `latent_entropy_mode = "marginal"` and `latent_lam_h_end = 0.001`,
  so the diff vs v5i4 is exactly `{latent_entropy_mode,
  latent_lam_h_end, run_tag}` and the diff vs v5i5 is exactly
  `{latent_entropy_mode, run_tag}`. Updated the audit banner, metrics
  CSV schema (`strategy_marginal_entropy_{loss,nats,kl}`), registry,
  fidelity rules, protocol docs, and progress tracker. Added
  `tests/test_v5i6_paper_faithful_marginal_entropy.py`; regenerated
  `tests/preset_snapshots.json` (adds 10 v5i6 aliases and the new
  `latent_entropy_mode` field to all snapshot entries; existing
  presets keep default `conditional`).
- **Launch-log consistency: pool-mode initial opponent + v5i4 budget tag:** Two logging inconsistencies surfaced in the v5i4 launch log. (1) `rl/train_ppo.py::_resolve_initial_opponent_and_phase` used `cfg.fixed_opponent_tag` (default `"OP3"`) to seed the very first env reset in `OPPONENT_POOL` / `opponent_randomize` mode, leaking an out-of-pool opponent into the first telemetry slice for every preset in the v4i1 / v4i3 / v5* chain (whose pool is `("OP5", "OP6", "OP7")`). The resolver now falls back to the first `cfg.opponent_pool` entry when the legacy `fixed_opponent_tag` is out-of-pool; an explicit in-pool `fixed_opponent_tag` still wins. (2) `v5i4.run_tag` flipped `_2m_ -> _1m_` so the tag agrees with the actual `total_timesteps = 1_000_000` budget (the rest of the v5 chain keeps the misleading `_2m_` suffix to preserve existing artifact paths). Pinned by 4 new tests in `tests/test_v5i4_paper_faithful.py::V5i4RunTagAndInitialOpponentConsistencyTests`; regenerated `tests/preset_snapshots.json` (single delta: 7 v5i4 aliases' `run_tag` updated, no other scalar drift).
- **v5i4 / paper-faithful end-to-end (task-reward PG on `q_phi` as part of `L_MARL`):** Added `apply_plan_faithful_latent_v5i4_end_to_end` (aliases `v5i4`, `v5i4_paper_faithful`, `v5i4_end_to_end`, `paper_faithful_end_to_end`, `latent_v5i4_paper_faithful`, `latent_v5i4_end_to_end`, `plan_faithful_latent_v5i4_end_to_end`) built directly on `v5_strict_summer`. The single semantic change is `latent_strategy_ppo_coef = 0.10`: this enables the per-step main-loop categorical PPO term on `q_phi` (`rl/latent_losses.py::strategy_ppo_loss`), which is the on-policy task-reward gradient channel the paper's "trained end-to-end from task reward" wording requires. The term lives inside `L_MARL` (no dedicated optimizer; the main-loop gate in `rl/custom_ppo/ppo_updater.py` drives it via the shared optimizer). Every post-Summer extension stays OFF: FiLM (v5i2), forced-z anneal (v5i3), episode-credit + dedicated router AdamW (v5i1), arc-credit (v3i19/v4i3), aux return / phase heads, preferences, router distillation. Added `_maybe_print_paper_faithful_audit` to `rl/training/banner.py` that emits the v5i4 invariant block when `cfg.run_tag` contains `"v5i4_paper_faithful"` (or `cfg.latent_paper_faithful_audit = True`), with explicit `WARNING` lines for the three documented mis-configurations (`strategy_ppo_coef <= 0`, dedicated router AdamW set, non-concat actor-z pathway). Added `tests/test_v5i4_paper_faithful.py` (22 tests) pinning inheritance, concat-only actor input dim (`164`), zero forced-z at every step, router task-gradient is enabled, zero-advantage = zero policy_loss, no forbidden channels, sparse 64-step resampling, alias snapshot equality, banner output. Regenerated `tests/preset_snapshots.json` (only structural changes: 7 new v5i4 aliases + previously un-snapshotted v5i1 / v5i2 / v5i3 aliases + 7 new schedule/FiLM fields added to all existing presets; zero scalar drift in any existing field, verified by `json.load` set-diff against the prior HEAD snapshot).
- **v5i3 / forced-z anneal + per-z router telemetry + matched-schedule random-router eval:** Added `apply_plan_faithful_latent_v5i3_balanced_warmup` (aliases `v5i3`, `v5i3_balanced_warmup`, `balanced_warmup`, etc.) layering a `0.30 -> 0.00` forced-z anneal across `200k -> 500k` on top of v5i2. Added four `latent_forced_z_episode_frac_{start,end}` + `latent_forced_z_anneal_{start,end}` fields in `PPOConfig` and `resolve_latent_forced_z_frac` in `rl/custom_ppo/schedules.py`. Wired the resolved fraction into `latent_strategy_state.py` at the episode-start forcing decision; `trainer.global_step` restore makes resumes correct. Added 8 per-`z` telemetry columns to `_update_fieldnames` and `apply_episode_strategy_ppo`. Added `--latent-selection {router,random-matched,random-episode,fixed}` to `plot/eval_checkpoint.py` for the matched-schedule routing-quality ablation. Zero-config reproduction: any preset that does not set the four `_start/_end` fields gets `latent_forced_z_episode_frac` constant (v5i2 still resolves to 0.0 at every step).
- **v5 / strict-Summer preset + main-loop gate fix:** Added `apply_plan_faithful_latent_v5_strict_summer` (aliases `v5`, `v5_strict_summer`, etc.) implementing the literal `docs/algorithm.md` loss with no auxiliary `q_phi` PG channels. Refactored the main-loop gate in `rl/custom_ppo/ppo_updater.py` so entropy / persistence / KL / strategy-PPO / aux-return each fire on their own coefficient; the double-step safeguard now triggers off `runtime.latent_router_optimizer is not None` instead of `latent_strategy_ppo_coef == 0`. Behavior change: v3i19 / v4i1 / v4i3 now actually apply their configured `lam_p` and `lam_h` schedules to `q_phi` via the main update (previously silently zeroed).
- Add entries here when presets, `GLOBAL_STATE_DIM`, or OP5 tuning tags change so experiments stay reproducible.
- **v6i2 pairwise CF objective and normalized behavior gate:** Added
  `latent_cf_require_competence`, `latent_cf_weak_pair_boost`, and
  `latent_cf_worst_pair_coef` to v6i2 so actor-CF pressure starts only after
  competence, persistent weak pairs receive direct hinge weight, and the worst
  latent pair cannot be hidden by mean-pair loss. CSV telemetry now includes
  per-pair hinge and weight fields. The behavioral-realization gate now reports
  raw `route_distance`, `task_behavior_distance`, `performance_spread`, and
  normalized `aggregate_effect`; frozen component floors block route-only
  passes. Pinned by `tests/test_v6i1_cf_loss.py::V6I1CfLossTests` and
  `tests/test_v6i2_gate_protocol.py::SafeguardTests`; regenerated
  `tests/preset_snapshots.json`.
- **v6i2 `latent_cf_coef_max` fix (strong CF confirmatory):** `apply_plan_faithful_latent_v6i2_staged_team_intent_curriculum` inherited `latent_cf_coef_max = 0.01` from v6i1 without override, so default `--preset v6i2` launches capped CF at 0.01 through Phase A (not the calibrated 1.0 ceiling). Preset now sets `latent_cf_coef_max = 1.0`; diff vs v6i1 is `{experiment_id, gate_protocol_version, latent_cf_coef_max, phase_a_max_end_fraction, run_tag}` plus v6i2 gate-threshold fields that match `PPOConfig` defaults. v6i3 inherits the strong ceiling via v6i2. Pinned by `tests/test_v6i2_gate_protocol.py::V6i2PresetTests`; regenerated `tests/preset_snapshots.json` (v6i2 alias entries only: `latent_cf_coef_max` `0.01` → `1.0`). Active runs started before this fix trained under weak CF unless `--latent-cf-coef-max 1.0` was passed explicitly.
