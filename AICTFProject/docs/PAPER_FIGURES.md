# Paper Figures and Reproducibility

This document lists which plots to include in the paper and how to regenerate them so that **win rates match exactly** between individual winrate scripts and `plot_eval_metrics.py`.

## Matching Win Rates (OP4, 100 episodes)

All evaluation uses:
- **Opponent:** OP4 (held-out, never in training)
- **Episodes:** 100 per model
- **Seed:** 43 (base 42 + 1 when OP4)

### Quick commands

**OP4 (unseen, held-out):**
```bash
python plot_2v2_winrate.py --match-eval
python plot_3v3_winrate.py --match-eval
python plot_4v4_winrate.py --match-eval
```
Outputs: `2v2_winrate_OP4_100ep.png`, `3v3_winrate_OP4_100ep.png`, `4v4_winrate_OP4_100ep.png`

**OP3 (training-time opponent):**
```bash
python plot_2v2_winrate.py --match-eval-op3
python plot_3v3_winrate.py --match-eval-op3
python plot_4v4_winrate.py --match-eval-op3
```
Outputs: `2v2_winrate_OP3_100ep.png`, `3v3_winrate_OP3_100ep.png`, `4v4_winrate_OP3_100ep.png`

Or equivalently:
```bash
python plot_2v2_winrate.py --opponent OP4 --episodes 100 --seed 42
python plot_3v3_winrate.py --opponent OP4 --episodes 100 --seed 42
python plot_4v4_winrate.py --opponent OP4 --episodes 100 --seed 42
```

### Expected win rates (OP4, 100 ep)

| Mode | Ours | Jacob et al. | Self-play |
|------|------|--------------|-----------|
| 2v2  | 94%  | 38%          | 95%       |
| 3v3  | 95%  | 97%          | 93%       |
| 4v4  | 96%  | 94%          | 88%       |

---

## Recommended figures for the paper

### Win rate plots

| Figure              | File                          | Command                                               |
|---------------------|-------------------------------|-------------------------------------------------------|
| 2v2 win rate (OP4)  | `2v2_winrate_OP4_100ep.png`    | `python plot_2v2_winrate.py --match-eval`             |
| 3v3 win rate (OP4)  | `3v3_winrate_OP4_100ep.png`   | `python plot_3v3_winrate.py --match-eval`             |
| 4v4 win rate (OP4)  | `4v4_winrate_OP4_100ep.png`   | `python plot_4v4_winrate.py --match-eval`             |

### Combined eval metrics (success, coordination, stability, etc.)

Run once to generate all paper-ready figures and the CSV table:

```bash
python plot_eval_metrics.py --opponents OP4 --episodes 100 --table-out eval_table_OP4_100ep.csv --table-opponent OP4
```

| Figure          | File                            | Contents                          |
|-----------------|---------------------------------|-----------------------------------|
| Performance     | `eval_metrics_Performance_OP4.png` | Success rate, mean steps (2v2/3v3/4v4) |
| Coordination    | `eval_metrics_Coordination_OP4.png` | Coverage efficiency, collision-free |
| Robustness      | `eval_metrics_Robustness_OP3_OP4.png` | Success vs OP3 vs OP4              |
| Stability       | `eval_metrics_Stability_OP4.png` | Return variance (2v2/3v3/4v4)       |
| Robotics        | `eval_metrics_Robotics_OP4.png` | Collision-free rate                 |

### Regenerate from saved table (no model eval)

If you already have `eval_table_OP4_100ep.csv`:

```bash
python plot_eval_metrics.py --from-csv eval_table_OP4_100ep.csv --table-opponent OP4
```

---

## Verify correct plots

1. **Win rates:** Run the three winrate scripts with `--match-eval` and confirm the printed percentages match the table above.
2. **Eval metrics:** The table printed by `plot_eval_metrics.py` should show the same success rates in the "Paper-ready metrics" section.
3. **Consistency:** `2v2_winrate_OP4_100ep.png` bar values should equal the success_rate row in `eval_table_OP4_100ep.csv` for 2v2; similarly for 3v3 and 4v4.

---

## Model checkpoints (defaults)

| Mode | Ours (league) | Jacob et al. (paper) | Self-play |
|------|---------------|----------------------|-----------|
| 2v2  | `final_ppo_league_2v2_colab.zip` | `final_weekend_paper_2v2.zip` | `final_weekend_selfplay_2v2.zip` |
| 3v3  | `final_weekend_league_3v3.zip`   | `final_weekend_paper_3v3.zip` | `final_weekend_selfplay_3v3.zip` |
| 4v4  | `final_ppo_league_4v4_colab.zip` | `final_weekend_paper_4v4.zip` | `final_ppo_selfplay_4v4_colab.zip` |

All defaults are under `checkpoints_sb3/`.
