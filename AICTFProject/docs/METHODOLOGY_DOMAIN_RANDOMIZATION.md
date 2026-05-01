# Domain randomization: protocol and diagnostics

This document is the **pre-committed interpretation** for training runs that use episode-level domain randomization (DR). Commands and defaults also appear in `SETUP_AND_TRAIN.md`; this file is the **why and how to read results**.

## What gets recorded in artifacts

1. **Stdout at run start**  
   `train_ppo.py` prints a single line when `--domain-randomization` is on: max sensor noise, max dropout, and blue speed jitter. Capture full stdout in your job log so a checkpoint’s training context does not depend on memory or git history.

2. **Checkpoint payload**  
   Saved checkpoints include `cfg` (`asdict(PPOConfig)`), which contains `train_domain_randomization`, `dr_sensor_noise_sigma_max`, `dr_sensor_dropout_max`, and `dr_blue_speed_jitter`. Inspecting the zip with Python is enough to recover the training contract.

3. **CSV telemetry**  
   `*_metrics.csv` and `*_episodes.csv` are named from `run_tag`; keep `run_tag` descriptive (e.g. `dr_sanity_2v2`, `dr_no_latent_sanity_2v2`).

## 200k sanity run (latent + DR)

**Command (example):**

```bash
python rl/train_ppo.py --mode FIXED_OPPONENT --fixed-opponent OP3 --agents 2 \
  --total-steps 200000 --domain-randomization --run-tag dr_sanity_2v2
```

### Read the trajectory, not the final WR alone

DR makes the MDP harder; the **level** of win rate will usually sit below a non-DR run at the same wall-clock step count.

| Pattern | Interpretation |
|---------|----------------|
| WR **slope** positive over 200k (e.g. 50% → 53%) while non-DR at same stage might be 50% → 56% | Healthy: DR is biting but the policy still finds signal. |
| WR nearly flat (e.g. 50% → 51%) | Concerning: little learning under this DR strength or LR/clip mismatch. |
| WR **declining** with high variance (e.g. 50% → 49%) | Likely broken: instability, DR too harsh, or bug—compare to no-latent DR baseline. |

### Which WR column to use for “slope”

**`win_rate`** in `*_metrics.csv` is **cumulative** over all completed episodes since run start. It moves slowly early in training and can **lag** behind what you see in stdout episode logs or a **rolling** view.

For DR sanity and “healthy slope” thresholds, prefer:

- **`rolling_win_rate_50ep`** or **`rolling_win_rate_200ep`**, and/or  
- Episode-log / stdout W–L tallies,

when judging whether learning is happening over the first ~200k steps. Do **not** rely on cumulative `win_rate` alone for early-run slope diagnosis.

## Latent strategy (`q_phi`) under DR

Prior runs characterized `q_phi` under nearly deterministic dynamics. DR adds a new regime.

Reasonable outcomes:

- **(a)** Fewer **effective** strategies (some modes stop paying off under noise).
- **(b)** Similar occupancy / entropy to pre-DR (strategies were already robust).
- **(c)** Different occupancy as the policy exploits DR-stabilized features.

**(a)** warrants the closest watch: if **strategy entropy collapses sharply** and **one strategy dominates** rollout time, you may lose multi-strategy benefit. Diagnostic: **per-strategy win rate** (or return) from episode CSVs bucketed by `latent_z`—if one `z` carries almost all wins, the team-strategy story weakens.

## Critic and supervised ceiling under DR

Under DR, returns are **noisier** for the same 14-d `global_state`, so:

- **Rollout explained variance (EV)** will often **drop** vs non-DR (e.g. into a **~0.05–0.10** band) once the run is long enough or the critic is pinning diverse returns. On a **short early horizon**, EV may look **similar** to non-DR if the 14-d ceiling was already low—added outcome variance can sit under the same noise floor—or if DR mostly perturbs **per-step** observations without shifting **episode return** predictability much yet. Treat EV together with `value_loss` and later `critic_ceiling` reruns; a sustained drop is **expected** at longer horizons, not automatically a trainer regression.
- Re-running `tools/critic_ceiling.py` on rollouts from a **DR-trained** policy may show **held-out R² around ~0.10–0.18** (vs ~0.2–0.25 in a cleaner sim). That means the **information ceiling is binding harder**; large WR gains after DR will eventually push toward **richer critic inputs**, not more critic tuning on 14-d alone.

## No-latent DR baseline (system stress test)

Run vanilla PPO under the **same** DR settings to verify DR is not latent-specific breakage:

```bash
python rl/train_ppo.py --mode FIXED_OPPONENT --fixed-opponent OP3 --agents 2 \
  --total-steps 200000 --domain-randomization --no-latent-strategy \
  --run-tag dr_no_latent_sanity_2v2
```

Disambiguation (default DR knobs, same horizon):

| | non-latent + DR | latent + DR |
|--|-----------------|-------------|
| **Healthier slope** | latent × DR interaction → inspect strategy advantage / `q_phi` gradients, not just DR strength. | (baseline row) |
| **Also flat** | DR too strong globally → **halve knobs**, then re-run latent. | same conclusion |
| **Non-latent only slightly better** | Mostly knob strength, partial interaction → halve knobs first; check if gap persists. | compare after halved run |
| **Flat but different failure shape** | Inspect per-step rewards, episode length—outside the usual protocol before more tuning. | |

### Optional: per-strategy advantage telemetry (latent runs)

`_*_metrics.csv` can include **`strategy_resample_adv_mean_z*`** / **`strategy_resample_adv_std_z*`** / **`strategy_resample_adv_n_z*`**: raw GAE **advantages at z-resample steps**, grouped by chosen `z`. Large **std** with small **mean** suggests noisy `A_z` (smoothing / horizon); small **both** suggests strategies do not separate in value under the current signal.

## Implementation reminder (scope)

Current DR is **training-only**, **episode-resampled**, and **asymmetric**: blue learning agent gets observation noise, enemy dropout, and slowdown-only speed scale; scripted OP3 is not mirrored with the same distribution. Symmetric DR and eval-time sweeps are future work; robustness **curves** (WR vs noise level) are the next methodology step after a stable DR training recipe.
