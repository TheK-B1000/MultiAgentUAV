# AAMAS 2027 experiment writeup (local)

## Scientific reporting rule (locked)

**Primary results = sealed \(n=128\) per arm.** Those remain the authoritative
tables in `experiments_revised.tex`. Do not replace them with any \(n=50\) view.

## Two different \(n=50\) artifacts (do not confuse)

| Artifact | Selection rule | Use |
|----------|----------------|-----|
| **TRUE top-50** (`true_top50_*`, `fig_*_true_top50`) | Rank seeds by **deployed score** \(\mathrm{outcome}(A{\to}A)+\mathrm{outcome}(B{\to}B)\); ties by ascending seed ID; take top 50. Noise: select from **Nominal only**, carry IDs. | Requested post-hoc descriptive “best deployed” subset |
| **Seed-ID prefix** (`seedid50_*`, `fig_*_seedid50`) | First 50 by ascending seed ID (old mistaken “top50” filenames) | Obsolete sensitivity check; keep only for provenance |

## TRUE top-50 (current)

```powershell
.\.venv\Scripts\python.exe paper/aamas2027/build_true_top50.py
```

Outputs:
- `true_top50_tables.tex` — labeled post-hoc descriptive; ranking rule in captions
- `true_top50_stats.json` — metrics + selected seed IDs
- `true_top50_provenance/*_seeds.json` — per-arm seed lists and scores
- `figures/fig_{baselines,noise,specialization}_true_top50.{pdf,png}`

Generalist ranked by \(\mathrm{outcome}(G{\to}A)+\mathrm{outcome}(G{\to}B)\).

## Files

| File | Role |
|------|------|
| `experiments_revised.tex` | Primary Experimental Evaluation (\(n=128\)) |
| `methodology_suite.tex` | Methods |
| `build_true_top50.py` | TRUE top-50 regenerator |
| `build_top50_and_plots.py` | Legacy seed-ID prefix regenerator (renamed outputs) |
