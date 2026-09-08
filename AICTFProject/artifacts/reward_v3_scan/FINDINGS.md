# Reward V3 economy scan — findings (read-only)

**Authored:** 2026-08-02  
**Isolation:** did not modify live V3 training (`artifacts/reward_v3/g0_v2_rewardv3_seed*`) or their configs.  
**Artifacts:** this directory (`static_enumeration.json`, `trajectory_rows.csv`, `trajectory_summary.json`).  
**Tests:** `tests/test_reward_alignment_v3.py` (10 passed).

---

## Headline

| Check | Result |
|-------|--------|
| Scripted discounted return: BOTH_ATTACK > TURTLE | **PASS** (margin +0.146) |
| BOTH_ATTACK win rate vs TURTLE | 1.00 vs 0.25 |
| OOB on these styles | 0.00 blue / episode |
| Discounting | γ=0.995 → win at t=240 worth **~0.30** face value |
| Hidden channel still live | **`enemy_mav_kill_reward = ±0.5` per tag in offense** |

Under V3 knobs (sparse tags=0, failed_commit=-0.004), a winning attack script still beats a camping TURTLE on discounted return. That is the Gate-2D reward equivalent and is good news for V3 **as far as these scripted styles go**.

It does **not** prove PPO will stay on that attractor — only that the raw economy no longer obviously prefers camping over winning for these controllers.

---

## 1. Discounting (quiet monster — confirmed real)

```text
gamma = 0.995 = pbrs_gamma (match: good)
gae_lambda = 0.99
horizon = 240

0.995^200 ≈ 0.367
0.995^240 ≈ 0.300
```

One immediate `enemy_mav_kill` (+0.5) ≈ **1.67×** a win delivered at step 240.

---

## 2. Reward-source table (composed path)

```text
raw = rterm + roff + rfail
    + dense_weight*(rpbrs + rteam)
    + sparse_weight*(sparse_points/100)
then tanh(raw/4), clip ±1
```

| Source | Value (V3) | Frequency | Farmable? | Notes |
|--------|-----------:|----------|-----------|-------|
| terminal win/loss/draw | +1 / -1 / -0.5 | 1/episode | No | Face value; discounted heavily if late |
| sparse capture | +100 → +1.0 composed | /capture | Hard | Objective |
| sparse tag no-flag | **0** | /tag | Closed | Was +100 |
| sparse tag carrier | **0** | /tag | Closed | Was +50 |
| **offense enemy_mav_kill** | **±0.5** | **/tag** | **YES — still open** | Not /100; not zeroed by V3 |
| offense flag pickup | +0.1 | /grab | Hard | |
| offense flag carry home | +0.5 | /capture | Hard | Double-pays with sparse capture |
| offense mine place | +0.2 | /place | Maybe | |
| sparse mine tag | +100 | /mine-tag | Maybe | Hardcoded constant |
| sparse OOB | -100 | /OOB | Avoidance | Rate ~0 on these scripts; cfg knob not wired into `_sparse_reward_points` yet |
| failed commit | **-0.004** | /failed macro | Avoidance | ~−0.74/ep at 184 events |
| team defense/escort/intercept | 0.03/0.02/0.02 | /step | Camp can collect defense | × dense_weight 0.25 |
| idle/spin penalties | 0.03/0.05 | /step | Camp pays idle | |
| PBRS | coef 0.5 | /step | Ideally no | γ matches PPO; phase-masked |

Full JSON: `static_enumeration.json`.

---

## 3. Scripted trajectory comparison (n=8 / style, OP6, map_a)

| Style | WR | B–R score | Rdisc | offense | team | failure | tags B/R |
|-------|---:|-----------|------:|--------:|-----:|--------:|----------|
| BLUE_BOTH_ATTACK_V2 | 1.00 | 3.0–2.0 | **+0.420** | +0.79 | +1.81 | −0.36 | 5.4 / 4.1 |
| BLUE_ONE_DEFENDER_V2 | 1.00 | 2.9–1.0 | **+0.842** | +2.58 | +0.53 | −0.38 | 5.2 / 2.9 |
| BLUE_TURTLE (camp proxy) | 0.25 | 1.0–0.9 | +0.274 | **+1.80** | +1.11 | −0.80 | **16.6 / 12.8** |

TURTLE’s **offense is higher than BOTH_ATTACK** because it farms tags through `enemy_mav_kill_reward`. Winning still wins on total discounted return — but the tag-kill channel is the next budget candidate if V3 PPO starts camping-with-tags.

OOB mass: **zero** on these styles (cannot yet budget −100 from this sample).

---

## 4. PBRS

- Form `coef * (γ Φ' − Φ)` with `pbrs_gamma = 0.995 = PPO gamma`.
- Pickup/capture transitions masked (attack/return inactive across event).
- No explicit terminal Φ=0; relies on reset. Worth a dedicated unit test later; not a smoking gun in this scan.

---

## 5. Failed-commit semantics

```text
ended = success | ticks_left≤0 | ~alive | tagged
failed = ended & ~success & was_alive
```

- Per-agent; both agents can pay in one step.
- **Being tagged converts an in-flight attack into a failed commit.**
- V3 cost at 184 events ≈ −0.74/ep (bounded).

---

## 6. What to watch while V3 finishes

From live metrics (~80–90k steps, read-only peek): shaping/outcome abs ratio still ~1.4–1.6 mid-run. That is not a failure yet, but keep eye on:

1. Does `reward_offense_mean` stay large while captures stay low? → kill-farm via tags.
2. Does win rate hold past the historical collapse zone (~225k)?
3. OOB still unmeasured under learned policies — keep instrumentation on.

---

## Recommended next budget fix (do NOT apply to live V3)

After V3 finishes, if camping-with-tags appears:

```text
env_enemy_mav_kill_reward → 0.0   # or budget like tags were
```

Treat as a separate single-axis ablation — do not reopen V3 mid-flight.
