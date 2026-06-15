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
| Strict-Summer | `v5_strict_summer` | entropy + persistence (+ KL) |
| v4i3 + arc-credit | `latent_v4i3_summer_proof` | strict-Summer + per-arc clipped PG |
| v3c episode-credit | `latent_episode_strategic` | per-episode clipped PG (dedicated optimizer) |
| K=1 collapsed | `plan_faithful_latent_k1` | n/a (latent collapsed) |
| No-latent | `no_latent_v4i3_baseline` | n/a |

Same opponent pool (`OP5 OP6 OP7`), same seed set, same map, same total timesteps. The row that beats no-latent *only via §6.1* is the one that supports the literal Summer claim; the row that requires §6.2 or §6.3 supports the weaker claim "the Summer plan plus a task-reward PG channel on `q_phi` is sufficient."

---

## 7. Changelog

- **v5 / strict-Summer preset + main-loop gate fix:** Added `apply_plan_faithful_latent_v5_strict_summer` (aliases `v5`, `v5_strict_summer`, etc.) implementing the literal `docs/algorithm.md` loss with no auxiliary `q_phi` PG channels. Refactored the main-loop gate in `rl/custom_ppo/ppo_updater.py` so entropy / persistence / KL / strategy-PPO / aux-return each fire on their own coefficient; the double-step safeguard now triggers off `runtime.latent_router_optimizer is not None` instead of `latent_strategy_ppo_coef == 0`. Behavior change: v3i19 / v4i1 / v4i3 now actually apply their configured `lam_p` and `lam_h` schedules to `q_phi` via the main update (previously silently zeroed).
- Add entries here when presets, `GLOBAL_STATE_DIM`, or OP5 tuning tags change so experiments stay reproducible.
