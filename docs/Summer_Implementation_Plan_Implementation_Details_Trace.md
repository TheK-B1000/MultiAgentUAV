# Summer Implementation Plan — Implementation details trace

**Canonical source (human-readable spec):** *Summer Implementation Plan.docx* — **IMPLEMENTATION DETAILS** and related subsections. Example path: `c:\Users\K-B\Desktop\Summer Implementation Plan.docx`. The Word document is the **normative** description; this repository **implements** that spec. This Markdown file is the **authoritative in-repo** record of what the code does and how it maps to the spec (green / red / blue).

**Color legend (HTML spans — use the companion `.html` in a browser or Word for reliable colors; many Markdown previews render the spans as well):**

- <span style="color:#1e8449">**Green** — matches the plan literally or the plan’s stated ranges/defaults</span>
- <span style="color:#a93226">**Red — tracked deviation** from the written sketch (intentional engineering choice; not silent drift)</span>
- <span style="color:#2471a3">**Blue — clarification** (paper notation vs code, or where the plan is illustrative)</span>

Code pointers are relative to the `AICTFProject/` root unless noted.

---

## 1. What is a “strategy”?

| Plan text | Status |
| --- | --- |
| Latent team coordination mode; not predefined labels, not hierarchical options, not action primitives. | <span style="color:#1e8449">Matches design intent. No hand-labeled “attack/defend” in code.</span> |
| “Spacing, aggression, risk tolerance, role allocation” are emergent; never labeled. | <span style="color:#1e8449">Qualitative; evaluation is via metrics/plots, not built-in class names.</span> |

---

## 2. Strategy representation (locked design)

| Plan text | Status |
| --- | --- |
| Discrete $z \in \{0,1,\ldots,K-1\}$. | <span style="color:#1e8449">Indices are `0..K-1` in `nn.Embedding` / `Categorical` (`rl/custom_ppo.py`, `rl/latent_marl.py`).</span> |
| $K=4$ recommended (ICRA); 6 acceptable. | <span style="color:#1e8449">Default `PPOConfig.latent_k = 4`. `train_ppo` **rejects** $K \notin \{4,6\}$ when latent is on.</span> |
| One-hot *or* embedding when feeding networks. | <span style="color:#1e8449">**Embedding** used: `nn.Embedding(K, d_z)` (plan prefers embedding over raw one-hot for the actor path).</span> |
| $d_z \in \{8, 16\}$. | <span style="color:#1e8449">Default `latent_z_embed_dim = 16`.</span> |

---

## 3. What information defines a strategy? (global state)

| Plan text | Status |
| --- | --- |
| Strategy is inferred from **global** information; **policy at execution** does not take raw global features—only $z$ (or its embedding). | <span style="color:#1e8449">`SharedActorCentralizedCritic` actor path uses per-agent `grid`/`vec` + `z_emb`; it does not concatenate `global_state` into actor inputs. `global_state` feeds `StrategyEncoder` and the centralized critic (CTDE). See `rl/custom_ppo.py` `policy_logits` / `values`.</span> |
| The doc’s **illustrative** `global_features` list (team geometry, relative distances, game phase, motion stats). | <span style="color:#1e8449">**Faithful order:** 14 floats in the same *semantic* order as the doc’s list — `mean_position_blue` (x,y) → `std_position_blue` (x,y) → `mean_position_red` (x,y) → `std_position_red` (x,y) → `min_blue_to_red_flag` → `min_red_to_blue_flag` → `blue_flag_captured` → `red_flag_captured` → `avg_blue_speed` → `avg_red_speed`. Implemented in `rl/global_state.py` as `GLOBAL_STATE_FIELD_NAMES` and `build_global_state_batch`.</span> |

---

## 4. Strategy inference network $q_\phi(z \mid s)$

| Plan text | Status |
| --- | --- |
| MLP: `Linear(state_dim, 128) → ReLU → 128 → ReLU → Linear(128, K)` logits. | <span style="color:#1e8449">`StrategyEncoder` in `rl/latent_marl.py` matches the **128–128–K** trunk; `state_dim` is `GLOBAL_STATE_DIM` (14).</span> |
| Initialization (doc code block has none) | <span style="color:#1e8449">`StrategyEncoder` uses default `nn.Linear` initialization to match the Word listing **verbatim**.</span> |

---

## 5. Sampling (training vs evaluation)

| Plan text | Status |
| --- | --- |
| Training: `Categorical(logits).sample()` — not argmax. | <span style="color:#1e8449">`SharedActorCentralizedCritic.sample_strategy(..., deterministic=False)` samples (see resample path in `CustomPPOTrainer._strategy_for_step`).</span> |
| Evaluation: `argmax` on logits. | <span style="color:#1e8449">`deterministic=True` → argmax in `sample_strategy`. Inference / entropy helpers in `CustomPPOInferencePolicy` use deterministic strategy where applicable.</span> |

---

## 6. When is strategy resampled?

| Plan text | Status |
| --- | --- |
| Option A: once per episode. | <span style="color:#1e8449">Default `latent_resample_every_n = 0` with `\_needs_strategy_sample` set True on done → new episode samples $z$.</span> |
| Option B: every $N$ steps, e.g. $N=20$; with persistence. | <span style="color:#1e8449">`latent_resample_every_n > 0` in `CustomPPOTrainer._strategy_for_step`.</span> |
| **Do not** resample every timestep. | <span style="color:#1e8449">`train_ppo` now **rejects** `latent_resample_every_n == 1` with a clear error (aligned with the plan’s warning).</span> |

---

## 7. Strategy persistence

| Plan text | Status |
| --- | --- |
| Penalize unnecessary switches: $\mathbb{1}[z \neq z_{\text{prev}}]$ on resamples. | <span style="color:#1e8449">`paper_strategy_switch_indicator` + mean on `z_persist_mask` in `rl/custom_ppo.py` `update()`. (The function `expected_strategy_switch_penalty` is kept in `rl/latent_marl.py` **only** for the unit test that checks its shape.)</span> |
| $\mathcal{L} = \mathcal{L}_{\text{MARL}} + \lambda_p \mathcal{L}_{\text{persist}} - \lambda_H \mathcal{H}(z)$. | <span style="color:#1e8449">**Form:** PPO loss + `latent_lam_p * persist_loss` + `(-latent_lam_h * mean(strategy entropy))` inside `latent_loss` in `rl/custom_ppo.py` `update()`. (Action-entropy and strategy-entropy are separate terms.) **Sign:** the trainer **minimizes** a scalar that includes **minus** weighted strategy entropy, which is **equivalent** to the plan’s intent of using $\mathcal{H}(z)$ as a **bonus** (higher entropy is preferred when `latent_lam_h > 0`).</span> |
| Initial episode draw should not pay persistence. | <span style="color:#1e8449">`z_persist_mask = resample_mask & ~needs_strategy_sample` in `_strategy_for_step` — the first $z$ of an episode has `needs_strategy_sample` true, so **persist** is false for that event.</span> |
| If there is **no** mid-episode resampling, persistence should be inactive. | <span style="color:#1e8449">With `latent_resample_every_n = 0` and `latent_resample_on_flag = false`, only the opening resample can occur; `persist_mask` stays false for that step, and later steps do not resample — so **L_persist is zero** the whole time (not “wrong”, just not applicable).</span> |
| Typical $\lambda_p$ / $\lambda_H$ (Word doc: §3.3 vs IMPLEMENTATION §6 give slightly different $\lambda_p$ bands). | <span style="color:#2471a3">**Clarification:** §3.3 states $\lambda_p \in [0.01, 0.1]$; IMPLEMENTATION §6 states $\lambda_p \in [0.01, 0.05]$ and $\lambda_H \in [0.001, 0.01]$. </span><span style="color:#1e8449">Defaults `latent_lam_p=0.02`, `latent_lam_h=0.005` lie in the intersection and are faithful default choices.</span> |

---

## 8. Policy: $\pi_\theta(a_i \mid o_i, z)$

| Plan text | Status |
| --- | --- |
| Sketch: `MLP(concat(flat_obs, z_emb))` with 256–256 ReLU, linear head. | <span style="color:#2471a3">Original flat-obs sketch kept as the baseline reference; see the next row for the active professor-approved CNN + MLP implementation.</span> |
| Professor-approved implementation change (2026-04-25): CNN + MLP. | <span style="color:#a93226">Tracked deviation from the original flat sketch: the active actor now runs each per-agent `grid` through `CNNEncoder`, concatenates that CNN output with scalar `vec` (including normalized flag-capture score counts) and optional `z_emb`, then feeds the 256-256 MLP/action head.</span> |
| Shared parameters across blue agents. | <span style="color:#1e8449">One shared network; tokenized per-agent forward.</span> |

---

## 9. Critic (CTDE): $Q$ / value with joint actions and $z$

| Plan text | Status |
| --- | --- |
| `critic_input = concat(global_state, joint_actions, z_onehot)`. | <span style="color:#1e8449">The document describes the centralized critic abstractly as $Q(s,\mathbf a, z)$. In this PPO implementation, that object is a **joint action- and $z$-conditioned value function** (scalar output per $(s,\mathbf a,z)$) used as the **PPO/GAE baseline**, not a full tabular $Q$ over all future counterfactual joint actions. `CentralizedCritic` implements `critic_input` as in the plan; see `rl/networks.py` / `SharedActorCentralizedCritic.values`.</span> |

---

## 10. Training loop (Python-level sketch in the plan)

| Plan text | Status |
| --- | --- |
| Nested `for episode` / `for t` with `env.step`. | <span style="color:#1e8449">**Same objective, vectorized implementation:** `GPUCTFVecEnv` + `CustomPPOTrainer` advance parallel envs; formally equivalent to independent nested loops (see `docs/rollout_semantics.md`).</span> |
| PPO with clipped objectives (plan abstract level). | <span style="color:#1e8449">`rl/ppo_core.py` implements clipped policy and value loss; GAE in-buffer.</span> |
| PPO ratio on **action** log-probs (sketch’s inner `for` over agents / actions). | <span style="color:#1e8449">`log_prob` in `update()` and stored rollout `log_probs` are **action-only**; $q_\phi$ is trained via strategy-entropy and persistence terms, not through the PPO ratio.</span> |

---

## 11. “Optional” items from the plan (Section 12)

| Plan text | Status |
| --- | --- |
| Resample only on flag/territory change; extra KL on consecutive $q_\phi$ distributions, etc. | <span style="color:#1e8449">**Optional flags (off by default):** `PPOConfig.latent_resample_on_flag` triggers a resample when the global-state flag/territory slice (indices 8:12, `GLOBAL_STATE_FLAG_TERRITORY_SLICE`) changes; `latent_kl_consecutive` adds forward KL between consecutive $q_\phi$ logits at valid timesteps. Set in `rl/train_ppo.py` / CLI (`--latent-resample-on-flag`, `--latent-kl-consecutive`). For **main paper results**, prefer a **clean** default: sample once per episode; treat periodic refresh, flag-triggers, and KL as **separate ablations** so each mechanism is easy to attribute.</span> |
| Avoid VAE, Gumbel-softmax, auxiliary heads. | <span style="color:#1e8449">Not used in the current latent path.</span> |

---

## 12. How to update this file

1. If you change behavior relative to a **red** item, **update the red paragraph** and add a one-line “since commit …” or date.
2. Keep *Summer Implementation Plan.docx* as the narrative paper/spec; this file is the **code↔spec diff** for engineering.

---

**Optional:** export this file to HTML/Word for colored tables; keep the `.md` in sync with *Summer Implementation Plan.docx* when the spec changes.

---

## 13. Operational notes (wording, validation, experiments — not “wrong code”)

| Topic | In-repo position |
| --- | --- |
| **Local vs global in the actor** | The actor does **not** take `global_state`. Per-agent `grid` / `vec` are **local** observations from `BatchedCTFCore._build_grid_obs` (see `docs/environment.md`); they are not a concatenated dump of the 14-d global feature vector. |
| **Scripted OP / phase / baselines** | May appear as **opponents** or **evaluation** settings. They are **not** labels or targets for $z$; $z$ is learned only from task-level reward and the plan’s inductive terms. |
| **E3-style ablation (prove latent helps)** | Not enforced in code, but the intended comparison is **identical** PPO+env+budget+seeds with **only** $z$ removed (`--no-latent-strategy` or `use_latent_strategy=False`); all else fixed. This is **config + interface** parity, not a guarantee of **bit-identical** training trajectories: architecture (parameter counts, optimizer state, init draws) still differs, so win-rate gaps are a fair *experimental* result, not a replay test. For paper wording, the no-latent baseline **omits** the strategy embedding and related heads (rather than **zeroing** a dead embedding) so the comparison is not confounded by unused parameters still eating optimizer state. |
| **Non-config confound: RNG** | If $q_\phi(z|s)$ and action `Categorical`s advanced the *same* default PyTorch stream, latent and no-latent rollouts would not even share the same **action** draw sequence (given identical seeds elsewhere). `CustomPPOTrainer` and `load_custom_ppo_policy` call `rl.custom_ppo.apply_deterministic_sampling_generators` so **strategy** and **action** use separate `torch.Generator` instances (fixed sub-seeds from `cfg.seed`). The simulator already uses its own `torch.Generator` (`game_field_gpu._rng` from `cfg.seed`); `opponent_params` also takes an explicit `generator` where it samples—those streams are not advanced by policy `Categorical` calls. |
| **Sub-seed protocol (intentional, stable)** | In `rl.custom_ppo`: `STRATEGY_GENERATOR_SEED_OFFSET` = `0x1_0000_00D` (decimal 268435469), `ACTION_GENERATOR_SEED_OFFSET` = `0x2_0000_02B` (decimal 536870955). Wire-up uses `(int(seed) + offset) & 0xFFFF_FFFF` for `torch.Generator.manual_seed` on the training/inference device. Do not change ad hoc without a paper/code note. |
| **E3 step telemetry (CSV)** | Set `PPOConfig.e3_step_telemetry_path` to append per–env-step rows when `use_latent_strategy` is on. Field order is `rl.custom_ppo.E3_STEP_TELEMETRY_FIELDS` (`z_t`, `q_phi_entropy`, `q_phi_argmax` = `argmax_z q_\phi`, `switched` = $z_t \ne z_{t-1}$ at the policy’s stored `prev_z`, `game_phase` = `rl.global_state.coarse_game_phase_from_global_state`). `update` is the PPO update index (0 on the first `collect_rollout`). |
| **Mechanical guards in repo** | `rl/config_presets.py` (``paper_default_latent_config`` / ``paper_default_no_latent_config`` / flag ablation), `tests/test_operational_gap_guards.py`, `tests/test_latent_core_import_graph.py` (no opponent/league roots in the latent PPO *module* import graph), `tests/test_e3_baseline_parity.py` (single-flag config diff + actor/critic width deltas), `tests/test_e3_rng_verification.py` (no-latent never calls `sample_strategy`, latent inference replay, E3 CSV header). L\_persist when $N=0$ is **exactly** $0$ in `update()`. |
