# AAMAS 2027 experiment writeup (local)

## Scientific reporting rule (locked)

**Primary results = sealed \(n=128\) per arm.** Those remain the authoritative
tables in `experiments_revised.tex`. Do not replace them with any \(n=50\) view.

## Figures

All paper figures live in `figures/` and carry no titles or annotations; the
explanation is in the LaTeX caption. Colors are fixed: blue = Pole-A strategy /
\(\Delta_A\), vermillion = Pole-B strategy / \(\Delta_B\). Error bars are 95% paired
percentile bootstrap intervals over seeds (20000 resamples, rng seed 7).

```powershell
# 1. agent-movement replays (sealed Ours episodes; refuses on any terminal mismatch)
.\.venv\Scripts\python.exe experiments/export_paper_trajectories.py --device cuda --scales 2 --occupancy-n 12
.\.venv\Scripts\python.exe experiments/export_paper_trajectories.py --device cuda --scales 4 --occupancy-n 12
.\.venv\Scripts\python.exe experiments/export_paper_trajectories.py --device cuda --scales 6 --occupancy-n 12
# 2. all n=128 figures (+ movement figures once replays are complete)
.\.venv\Scripts\python.exe paper/aamas2027/paper_figures.py
# 3. post-hoc top-50 tables and figures
.\.venv\Scripts\python.exe paper/aamas2027/build_true_top50.py
```

| Figure | Content | Source |
|--------|---------|--------|
| `fig_crossover` | Win rate of each strategy on each pole, 4 two-strategy systems x 3 team sizes | sealed n=128 rows |
| `fig_complementarity` | \((\Delta_A,\Delta_B)\) per system and team size; only upper-right \((\Delta_A,\Delta_B)>0\) shaded | sealed n=128 rows |
| `fig_baselines` | \(\Delta_A,\Delta_B\) per system, grouped by team size | sealed n=128 rows |
| `fig_worst_pole` | \(\min(\Delta_A,\Delta_B)\) per system vs team size | sealed n=128 rows |
| `fig_deployed_winrate` | Intended-strategy WR on Pole A / Pole B (panel titles); incl. Generalist | sealed n=128 rows |
| `fig_margin` | Ours \(\Delta\) on score margin | sealed n=128 rows |
| `fig_noise` | Ours \(\Delta_A,\Delta_B\) under nominal / localization / motion / delay | sealed n=128 noise rows |
| `fig_trajectories` | Rows=team size, cols=four crossover deployments (labeled); first 80 steps | replays |
| `fig_occupancy` | Blue-agent visit density, Pole-A vs Pole-B strategy, and difference | replays |
| `fig_posture` | Fraction of team in own half over the episode | replays |
| `fig_baselines_true_top50`, `fig_noise_true_top50` | Same layouts on the post-hoc top-50 subset | sealed rows, top-50 seeds |

Replay seeds: per team size, the lowest-ID sealed seed with the full crossover
pattern (trajectory figure) plus the first 12 seeds of the sealed block
(occupancy / posture), each run for both strategies on both poles. Selection is
recorded in `artifacts/qualitative_capture/paper_suite_trajectories/<N>v<N>/manifest.json`.

## Post-hoc TRUE top-50 (secondary)

Rank seeds by **deployed score**
\(\mathrm{outcome}(A{\to}A)+\mathrm{outcome}(B{\to}B)\); ties by ascending
seed ID; take top 50. Noise: select from **Nominal only**, carry IDs across
perturbations. Generalist ranked by
\(\mathrm{outcome}(G{\to}A)+\mathrm{outcome}(G{\to}B)\).

Outputs:
- `true_top50_tables.tex` — labeled post-hoc descriptive; ranking rule in captions
- `true_top50_stats.json` — metrics + selected seed IDs
- `true_top50_provenance/*_seeds.json` — per-arm seed lists and scores
- `paper_data_export/{baselines,noise}_true_top50.csv` and copy of tables

## Files

| File | Role |
|------|------|
| `experiments_revised.tex` | Primary Experimental Evaluation (\(n=128\)), with figure environments |
| `KB_methodology.tex` | Methodology writeup for KB / paper input (current code) |
| `methodology_suite.tex` | Same methods text (working copy) |
| `paper_figures.py` | All figures + shared figure style |
| `build_true_top50.py` | TRUE top-50 tables and figures |
| `../../experiments/export_paper_trajectories.py` | Agent-movement replays |
