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

## 6. Changelog

- Add entries here when presets, `GLOBAL_STATE_DIM`, or OP5 tuning tags change so experiments stay reproducible.
