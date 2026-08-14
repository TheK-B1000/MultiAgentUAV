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

Eval **COMPLETE** (56/56). Frozen classifier applied post-hoc (stale supervisor had skipped it).

```text
56 / 56 cells                    COMPLETE
d1_d3_d7_summary.json            EXISTS
gate_results.json                EXISTS
PATH_B                           DISCOVERY (cross-condition; 0 seed-noise)
CROSSOVER_FOUND                  true (n=3)
hardest / easiest column         OP7 / OP12
D1 / D3 / D7 mean overall WR     0.803 / 0.814 / 0.798
next                             1-seed specialist pilot (human boundary)
```

PATH_B is **preregistered discovery**, not confirmation.

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
