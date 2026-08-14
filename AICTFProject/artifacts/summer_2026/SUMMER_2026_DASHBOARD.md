# Summer 2026 — live scientific dashboard

Updated: 2026-08-14 (one-canonical-seed protocol frozen)

```text
SEED PROTOCOL (frozen)
  1 seed to discover
  replicate only important findings
  new stages labeled EXPLORATORY_SINGLE_SEED

D1 ×3                     COMPLETE   (historical 3-seed row; unchanged)
D7 ×3 existing            COMPLETE   (historical 3-seed row; unchanged)
D1 vs D7                  FIRST RESULT COMPLETE
                          (~0.803 vs ~0.798; no specialization)

D3_POOL_PREFLIGHT         PASS
D3 3700001                COMPLETE
D3 3700002                COMPLETE
D3 3700003                STOPPED_BY_PROTOCOL (artifacts preserved; not in primary eval)

FP_SMOKE (formal)         PASS       (mechanism)
FP scientific phase       NOT AUTHORIZED

Demand analysis           after D3 3700003 + unified eval
New stages (if authorized)
  specialists / FP / selector / latent   1 canonical seed each
Latent work               BLOCKED BY DESIGN until gates pass
```

## Seed protocol

See `artifacts/summer_2026/SEED_PROTOCOL_FROZEN.json`.

This D3 campaign is **grandfathered** through seeds that had already started.
It is **not** the template for later stages. Later stages are exploratory
single-seed unless a result is explicitly authorized for replication.

## Critical path (still automated)

```text
3700003 STOPPED_BY_PROTOCOL  →  D1/D3(3700001,3700002)/D7 eval  →  frozen Path A/B classifier
```

Eval (do not classify yet):

```text
15 / 56 cells                    COLLECTED
D7-3200001                       COMPLETE   overall ~0.819
D7-3200002                       COMPLETE   overall ~0.781
D7-3200003                       IN PROGRESS
D1 ×3, D3 ×2                     PENDING

CHECKPOINT COMPATIBILITY         PASS
OP12 very easy / OP10 relatively hard / OP9 similar   PRELIMINARY (D7 only)
Seed-level fingerprint variance  CLEARLY PRESENT (OP7 .867 vs .600, same D7 condition)
Strategic complementarity        NOT YET ESTABLISHED
PATH_A / PATH_B                  UNCLASSIFIED
```

Same-condition Wald crossovers = SEED_NOISE, not PATH_B. PATH_B requires a
cross-condition (D1/D3/D7) two-way LCB95>0 reversal. Recorded before any D1/D3
eval row exists. Wald estimator unchanged. Let DIVERSITY_EVAL finish.

## Two paper paths (frozen before the full 8-policy board)

See `artifacts/summer_2026/PAPER_PATH_READOUT_FROZEN.json`.

```text
PATH_A  PPO absorbs strategic diversity into a robust generalist
        ← no credible ranking crossover

PATH_B  complementary best responses
        ← CROSSOVER_FOUND (Wald LCB95 > 0 in both directions)
```

Either path is a valid result. Classification waits for `d1_d3_d7_summary.json`.
The completed D7 seed 3200001 board is **one row**, not a path decision.

PATH_B hits are **discovery** (scan does not control family-wise error). Confirmation
is the later 1-seed specialist pilot on the specific crossover, not a midstream
gate change.

After eval, supervisor runs `analyze_summer_2026_paper_path.py` then stops
at `STOPPED_SCIENTIFIC_GATE` with the path recorded. Specialists are not
auto-launched.
