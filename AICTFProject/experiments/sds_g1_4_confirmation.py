"""SDS_G1_4 Gate B confirmation on the amended, verified-unused seed block.

This is the CONFIRMATION run. Search numbers are not evidence for it and it
does not feed back into search. It spends the block once and reports PASS or
FAIL against thresholds frozen before the candidate existed.

Frozen package (artifacts/strategic_demand/CANDIDATE_A_SDS_G1_4_FROZEN.json):
    ruleset      RULESET_V3_M1 (own_flag_home_required_to_score=True)
    A pole       SDS_G1_4  (OP6 + overlay, the evolved GUARD-demanding opponent)
    B pole       OP7 canonical (BREACH anchor)
    GUARD        BLUE_ONE_DEFENDER_V2
    BREACH       BLUE_BOTH_ATTACK_V2
    map / team   map_a, 2v2, horizon 240
    seeds        5000001..5000032, 32 paired seeds

Gate B, all three required:
    A:  WR(GUARD, A) - WR(BREACH, A) >= 0.15  AND  paired LCB95 > 0
    B:  WR(BREACH, B) - WR(GUARD, B) >= 0.15  AND  paired LCB95 > 0
    O:  t_intent > t_commit  (intent is not readable before commitment is due)

Nothing scientific is redefined here. run_episode (the treatment semantics,
including the intent/commitment measurement) is imported unchanged from the
frozen searcher, and paired_ci is imported unchanged from the frozen M1 assay,
so the confirmation cannot silently drift from the protocol it is testing.
What this file adds is only: per-episode row retention, the paired bootstrap
applied to both poles, and the gate arithmetic.

Seed hygiene: block 5000001..5000032 was audited CLEAN over declared seeds, an
over-inclusive integer scan, declared block spans, and full git history, after
2500001 and 2600001 were both found to collide with existing training seeds.
See CONFIRMATION_SEED_BLOCK_AMENDMENT.json.

Run:  python experiments/sds_g1_4_confirmation.py --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Frozen scientific core -- imported, never reimplemented.
from experiments.strategic_demand_searcher import (  # noqa: E402
    BREACH, GUARD, MAP, MAX_STEPS, run_episode,
)
from experiments.m1_payoff_assay import paired_ci  # noqa: E402
from experiments.sds_genome import (  # noqa: E402
    SDSGenome, canonical_parent, degeneracy_penalty,
)

FREEZE = ROOT / "artifacts/strategic_demand/CANDIDATE_A_SDS_G1_4_FROZEN.json"
AMENDMENT = ROOT / "artifacts/strategic_demand/CONFIRMATION_SEED_BLOCK_AMENDMENT.json"
OUT_DIR = ROOT / "artifacts/strategic_demand/confirmation_sds_g1_4"

FLOOR = 0.15          # frozen Gate B floor, both directions
BOOTSTRAP_RNG = 7     # same constant the M1 assay used
DISQUALIFIED = {2500001, 2600001}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_frozen() -> tuple[SDSGenome, str, int, int]:
    """Read the package from the freeze record. Nothing is hardcoded here that
    the freeze already fixes, so an edited freeze changes the run rather than
    silently disagreeing with it."""
    f = json.loads(FREEZE.read_text(encoding="utf-8"))
    pkg = f["confirmation_package_frozen"]
    genome = SDSGenome.from_dict(f["candidate_genome"])
    b_pole = "OP7"
    if "OP7" not in pkg["B_pole"]:
        raise SystemExit(f"B pole is not OP7 in the freeze: {pkg['B_pole']!r}")
    seed_base = int(pkg["confirmation_seed_block"])
    n = int(str(pkg.get("confirmation_seed_range", "")).split("(")[-1].split()[0]
            or 32)
    return genome, b_pole, seed_base, n


def guard_rails(seed_base: int, n: int) -> None:
    lo, hi = seed_base, seed_base + n - 1
    for bad in DISQUALIFIED:
        if lo <= bad <= hi or bad <= lo <= bad + 31:
            raise SystemExit(
                f"REFUSING TO RUN: block {lo}..{hi} touches permanently "
                f"disqualified block {bad}. See CONFIRMATION_SEED_BLOCK_AMENDMENT.json")
    if not AMENDMENT.is_file():
        raise SystemExit("REFUSING TO RUN: seed-block amendment record is missing")
    existing = OUT_DIR / "summary.json"
    if existing.is_file():
        raise SystemExit(
            f"REFUSING TO RUN: {existing} already exists. The confirmation block is "
            "spent once. Move the old result aside deliberately if a rerun is "
            "genuinely authorized.")


def pole(genome: SDSGenome, *, label: str, seed_base: int, n: int, device: str,
         direction: str) -> dict:
    """One pole. direction='guard_minus_breach' for A, 'breach_minus_guard' for B."""
    rows, d = [], []
    for i in range(n):
        seed = seed_base + i
        g = run_episode(style=GUARD, genome=genome, seed=seed, device=device)
        b = run_episode(style=BREACH, genome=genome, seed=seed, device=device)
        for style, ep in ((GUARD, g), (BREACH, b)):
            rows.append({
                "pole": label, "genome_id": genome.genome_id,
                "episode_seed": seed, "blue_style": style,
                "blue_score": ep["blue_score"], "red_score": ep["red_score"],
                "win": ep["win"], "draw": ep["draw"], "steps": ep["steps"],
                "t_intent": ep["t_intent"], "t_commit": ep["t_commit"],
                "zero_zero": ep["zero_zero"], "total_score": ep["total_score"],
            })
        d.append((g["win"] - b["win"]) if direction == "guard_minus_breach"
                 else (b["win"] - g["win"]))
        print(f"  [{label}] seed {seed}  GUARD win={g['win']}  BREACH win={b['win']}",
              flush=True)

    d = np.asarray(d, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_RNG)
    mean, lo, hi = paired_ci(d, rng)
    gw = float(np.mean([r["win"] for r in rows if r["blue_style"] == GUARD]))
    bw = float(np.mean([r["win"] for r in rows if r["blue_style"] == BREACH]))
    zz = float(np.mean([r["zero_zero"] for r in rows]))
    tot = float(np.mean([r["total_score"] for r in rows]))
    return {
        "pole": label, "genome_id": genome.genome_id, "direction": direction,
        "n_paired": n, "seed_base": seed_base,
        "seed_range": f"{seed_base}..{seed_base + n - 1}",
        "guard_wr": gw, "breach_wr": bw,
        "delta": mean, "lcb95": lo, "ucb95": hi,
        "clears_floor": bool(mean >= FLOOR),
        "lcb_positive": bool(lo > 0.0),
        "passes": bool(mean >= FLOOR and lo > 0.0),
        "frac_0_0": zz, "mean_total_score": tot,
        "degeneracy_penalty": degeneracy_penalty(zz, tot),
        "_rows": rows,
    }


def observability(rows: list[dict]) -> dict:
    """t_intent > t_commit, measured on BREACH episodes: the question is whether
    intent becomes readable only after the allocation was already due."""
    br = [r for r in rows if r["blue_style"] == BREACH]
    miss = MAX_STEPS + 1
    gaps, both = [], 0
    for r in br:
        ti = r["t_intent"] if r["t_intent"] is not None else miss
        tc = r["t_commit"] if r["t_commit"] is not None else miss
        gaps.append(ti - tc)
        if r["t_intent"] is not None and r["t_commit"] is not None:
            both += 1
    gaps = np.asarray(gaps, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_RNG)
    mean, lo, hi = paired_ci(gaps, rng)
    return {
        "definition": "t_intent - t_commit on BREACH episodes; censored values -> MAX_STEPS+1",
        "n": len(br),
        "n_both_observed": both,
        "mean_intent_minus_commit": mean,
        "lcb95": lo, "ucb95": hi,
        "frac_positive": float(np.mean(gaps > 0)),
        "passes": bool(mean > 0.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate wiring and print the plan; run zero episodes")
    a = ap.parse_args()

    genome, b_key, seed_base, n = load_frozen()
    guard_rails(seed_base, n)
    b_genome = canonical_parent(b_key)

    print(f"SDS_G1_4 CONFIRMATION  {_now()}")
    print(f"  A pole      {genome.genome_id}  base={genome.base_opponent} "
          f"overlay={genome.overlay}")
    print(f"  B pole      {b_genome.genome_id}  base={b_genome.base_opponent}")
    print(f"  styles      GUARD={GUARD}  BREACH={BREACH}")
    print(f"  map/horizon {MAP} / {MAX_STEPS}, 2v2")
    print(f"  seeds       {seed_base}..{seed_base + n - 1}  ({n} paired)")
    print(f"  gate        delta >= {FLOOR} AND LCB95 > 0, both poles; "
          f"plus t_intent > t_commit")
    print(f"  episodes    {n * 2 * 2} total")
    if a.dry_run:
        print("\nDRY RUN -- wiring validated, no episode run, block untouched.")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n--- A pole (GUARD should pay) ---", flush=True)
    A = pole(genome, label="A_SDS_G1_4", seed_base=seed_base, n=n,
             device=a.device, direction="guard_minus_breach")
    print("\n--- B pole (BREACH should pay) ---", flush=True)
    B = pole(b_genome, label="B_OP7", seed_base=seed_base, n=n,
             device=a.device, direction="breach_minus_guard")

    rows = A.pop("_rows") + B.pop("_rows")
    obs = observability([r for r in rows if r["pole"] == "A_SDS_G1_4"])

    verdict = ("V3_STRATEGIC_DEMAND_VALIDATED"
               if (A["passes"] and B["passes"] and obs["passes"])
               else "V3_STRATEGIC_DEMAND_NOT_VALIDATED")

    summary = {
        "record": "SDS_G1_4 Gate B confirmation",
        "utc": _now(),
        "protocol": "artifacts/strategic_demand/CANDIDATE_A_SDS_G1_4_FROZEN.json",
        "seed_block_amendment":
            "artifacts/strategic_demand/CONFIRMATION_SEED_BLOCK_AMENDMENT.json",
        "seed_base": seed_base, "n_paired": n,
        "seed_range": f"{seed_base}..{seed_base + n - 1}",
        "total_episodes": len(rows),
        "floor": FLOOR, "bootstrap": {"n_boot": 20000, "alpha": 0.05,
                                      "rng_seed": BOOTSTRAP_RNG,
                                      "procedure": "paired percentile bootstrap, "
                                                   "imported from m1_payoff_assay"},
        "A_pole": A, "B_pole": B, "observability": obs,
        "gate_B": {
            "A_passes": A["passes"], "B_passes": B["passes"],
            "observability_passes": obs["passes"],
            "all_required": True,
        },
        "verdict": verdict,
        "thresholds_not_moved": "0.15 floor and LCB95>0 unchanged from the original freeze",
    }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2),
                                          encoding="utf-8")
    with open(OUT_DIR / "episode_rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print("\n" + "=" * 66)
    print(f"A  {A['genome_id']:<16} delta={A['delta']:+.4f} "
          f"LCB95={A['lcb95']:+.4f}  {'PASS' if A['passes'] else 'FAIL'}")
    print(f"B  {B['genome_id']:<16} delta={B['delta']:+.4f} "
          f"LCB95={B['lcb95']:+.4f}  {'PASS' if B['passes'] else 'FAIL'}")
    print(f"O  t_intent-t_commit  mean={obs['mean_intent_minus_commit']:+.3f} "
          f"frac>0={obs['frac_positive']:.3f}  "
          f"{'PASS' if obs['passes'] else 'FAIL'}")
    print("=" * 66)
    print(f"VERDICT: {verdict}")
    print(f"written: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
