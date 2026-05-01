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

Use cumulative or rolling win rate columns in `*_metrics.csv`; a single end-point number is insufficient.

## Latent strategy (`q_phi`) under DR

Prior runs characterized `q_phi` under nearly deterministic dynamics. DR adds a new regime.

Reasonable outcomes:

- **(a)** Fewer **effective** strategies (some modes stop paying off under noise).
- **(b)** Similar occupancy / entropy to pre-DR (strategies were already robust).
- **(c)** Different occupancy as the policy exploits DR-stabilized features.

**(a)** warrants the closest watch: if **strategy entropy collapses sharply** and **one strategy dominates** rollout time, you may lose multi-strategy benefit. Diagnostic: **per-strategy win rate** (or return) from episode CSVs bucketed by `latent_z`—if one `z` carries almost all wins, the team-strategy story weakens.

## Critic and supervised ceiling under DR

Under DR, returns are **noisier** for the same 14-d `global_state`, so:

- **Rollout explained variance (EV)** will often **drop** vs non-DR (e.g. into a **~0.05–0.10** band). That is **expected**, not a regression of the trainer.
- Re-running `tools/critic_ceiling.py` on rollouts from a **DR-trained** policy may show **held-out R² around ~0.10–0.18** (vs ~0.2–0.25 in a cleaner sim). That means the **information ceiling is binding harder**; large WR gains after DR will eventually push toward **richer critic inputs**, not more critic tuning on 14-d alone.

## No-latent DR baseline (system stress test)

Run vanilla PPO under the **same** DR settings to verify DR is not latent-specific breakage:

```bash
python rl/train_ppo.py --mode FIXED_OPPONENT --fixed-opponent OP3 --agents 2 \
  --total-steps 200000 --domain-randomization --no-latent-strategy \
  --run-tag dr_no_latent_sanity_2v2
```

- If **non-latent + DR** trains with a sane WR trajectory, DR plumbing is sound; latent-specific issues are isolated to the latent stack.
- If **both** latent and non-latent struggle, dial DR knobs down (see `SETUP_AND_TRAIN.md`) before debugging the encoder.

## Implementation reminder (scope)

Current DR is **training-only**, **episode-resampled**, and **asymmetric**: blue learning agent gets observation noise, enemy dropout, and slowdown-only speed scale; scripted OP3 is not mirrored with the same distribution. Symmetric DR and eval-time sweeps are future work; robustness **curves** (WR vs noise level) are the next methodology step after a stable DR training recipe.
