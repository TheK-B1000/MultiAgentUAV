"""C3 Stage-4 analyzer: leg 1 (fresh natural) and leg 2 (fresh counterfactual).

Frozen cells: artifacts/c3_discovery/C3_STAGE4_CONFIRMATION_FROZEN.json (bebb626)

Transcription of already-frozen statistics. No analytical choice is made here;
the floor, threshold, estimator, bootstrap and replication rule were all fixed
before any 9810000+ data existed.

TWO INVARIANTS THAT ARE EASY TO GET WRONG
-----------------------------------------
ZERO-ANCHOR EPISODES. The anchors file contains only anchors, so an episode that
produced none contributes no rows. Bootstrapping over "episodes present in the
data" would silently condition on episodes that produced at least one anchor and
bias anchors-per-episode UPWARD. The complete episode universe is therefore
reconstructed from the frozen cells -- 7 opponents x 30 seeds = 210 episodes per
policy, 630 total -- and absent episodes are filled with zero.

POLICY UNIVERSE. The census runner accepts a --seeds subset. Analysing a subset
against a >=2/3-of-3 replication rule would be meaningless, so anything other
than exactly the three frozen policies aborts.

Run:  python experiments/analyze_c3_stage4.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

DISCOVERY_DIR = PROJECT_ROOT / "artifacts" / "c3_discovery"
STAGE4_DIR = PROJECT_ROOT / "artifacts" / "c3_stage4"
STAGE4_FROZEN = DISCOVERY_DIR / "C3_STAGE4_CONFIRMATION_FROZEN.json"
RESULT_PATH = STAGE4_DIR / "C3_STAGE4_RESULT.json"

QUALIFIED = "QUALIFIED_COMMITMENT_FORK"
NOT_QUALIFIED = "NO_COMMITMENT_FORK"
VERDICT_FIELD = "episode_status"
VALID_VERDICTS = frozenset({QUALIFIED, NOT_QUALIFIED})

RESAMPLES = 2000
BOOT_SEED = 12345
LCB_PCT = 2.5  # q_0.025, per C3_STAGE3_LCB_DEFINITION_AMENDMENT.json


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def episode_universe(policies, opponents, seed_base: int, episodes: int) -> dict[int, list[str]]:
    """Every episode that WILL have been run, per policy -- including empty ones.

    This is reconstructed from the frozen cells rather than from observed data,
    which is the whole point: episodes that produced zero anchors must be
    present with a count of zero.
    """
    out: dict[int, list[str]] = {}
    for p in policies:
        keys = []
        for o in opponents:
            for i in range(episodes):
                keys.append(f"{int(p)}|{o}|{seed_base + i}")
        out[int(p)] = keys
    return out


def anchors_per_episode_counts(anchors: list[dict], universe: dict[int, list[str]]) -> dict[int, dict[str, int]]:
    counts: dict[int, dict[str, int]] = {p: {k: 0 for k in keys} for p, keys in universe.items()}
    for a in anchors:
        p = int(a["train_seed"])
        key = f"{p}|{a['opponent']}|{int(a['eval_seed'])}"
        if p not in counts or key not in counts[p]:
            raise SystemExit(
                f"REFUSED: anchor outside the frozen episode universe: {key}. "
                "The census does not match the frozen Stage-4 cells."
            )
        counts[p][key] += 1
    return counts


def bootstrap_mean(values: list[float], *, rng) -> tuple[float, float, float]:
    """-> (point, lcb, ucb). Cluster unit is the episode: one value per episode."""
    arr = np.asarray(values, dtype=float)
    point = float(arr.mean())
    draws = arr[rng.integers(0, arr.size, (RESAMPLES, arr.size))].mean(axis=1)
    return point, float(np.percentile(draws, LCB_PCT)), float(np.percentile(draws, 100 - LCB_PCT))


def is_qualified(row: dict) -> bool:
    if VERDICT_FIELD not in row:
        raise SystemExit(f"REFUSED: Stage-4 row has no {VERDICT_FIELD!r} field")
    v = str(row[VERDICT_FIELD])
    if v not in VALID_VERDICTS:
        raise SystemExit(f"REFUSED: unrecognised {VERDICT_FIELD}={v!r}")
    return v == QUALIFIED


def analyze_leg1(anchors: list[dict], frozen: dict) -> dict:
    policies = [int(p) for p in frozen["policies"]]
    opponents = list(frozen["opponents"])
    base = int(frozen["seeds"]["base"])
    episodes = int(frozen["seeds"]["episodes_per_cell"])
    floor = float(frozen["leg_1_fresh_natural"]["floor"])

    universe = episode_universe(policies, opponents, base, episodes)
    counts = anchors_per_episode_counts(anchors, universe)

    rng = np.random.default_rng(BOOT_SEED)
    per_policy = {}
    for p in policies:
        vals = [float(counts[p][k]) for k in universe[p]]
        n_zero = sum(1 for v in vals if v == 0.0)
        point, lcb, ucb = bootstrap_mean(vals, rng=rng)
        per_policy[str(p)] = {
            "n_episodes": len(vals),
            "n_episodes_with_zero_anchors": n_zero,
            "n_anchors": int(sum(vals)),
            "anchors_per_episode": round(point, 4),
            "LCB95": round(lcb, 4), "UCB95": round(ucb, 4),
            "floor": floor,
            "LEG1_PASS": bool(lcb > floor),
        }
    return {
        "leg": "LEG1_FRESH_NATURAL",
        "floor": floor,
        "floor_derivation": frozen["leg_1_fresh_natural"]["floor_derivation"],
        "zero_anchor_episodes_included": True,
        "per_policy": per_policy,
    }


def analyze_leg2(results: list[dict], sample_manifest: dict | None) -> dict:
    if not results or sample_manifest is None:
        return {"leg": "LEG2_FRESH_COUNTERFACTUAL", "status": "PENDING",
                "note": "fresh counterfactual screen has not been run yet"}
    W = sample_manifest["W_h"]
    by_policy_stratum: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        by_policy_stratum[int(r["train_seed"])][f"{int(r['train_seed'])}|{r['opponent']}"].append(r)

    rng = np.random.default_rng(BOOT_SEED)
    per_policy = {}
    for p, strata in sorted(by_policy_stratum.items()):
        w_total = sum(W[h] for h in strata)
        point = sum((W[h] / w_total) * (sum(is_qualified(r) for r in rows) / len(rows))
                    for h, rows in strata.items())
        eps: dict[str, list[dict]] = defaultdict(list)
        for h, rows in strata.items():
            for r in rows:
                eps[f"{h}|{int(r['eval_seed'])}"].append(r)
        keys = sorted(eps)
        draws = np.empty(RESAMPLES)
        for i in range(RESAMPLES):
            idx = rng.integers(0, len(keys), len(keys))
            agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
            for j in idx:
                for r in eps[keys[int(j)]]:
                    h = f"{int(r['train_seed'])}|{r['opponent']}"
                    agg[h][0] += 1
                    agg[h][1] += int(is_qualified(r))
            tot = sum(W[h] for h in agg) or 1.0
            draws[i] = sum((W[h] / tot) * (q / n) for h, (n, q) in agg.items())
        lcb = float(np.percentile(draws, LCB_PCT))
        per_policy[str(p)] = {
            "n_anchors": sum(len(v) for v in strata.values()),
            "weighted_fork_rate": round(float(point), 4),
            "LCB95": round(lcb, 4),
            "UCB95": round(float(np.percentile(draws, 100 - LCB_PCT)), 4),
            "threshold": 0.20,
            "LEG2_PASS": bool(lcb > 0.20),
        }
    return {"leg": "LEG2_FRESH_COUNTERFACTUAL", "status": "COMPLETE",
            "per_policy": per_policy}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage4-dir", default=str(STAGE4_DIR))
    args = ap.parse_args()
    d = Path(args.stage4_dir)

    frozen = json.loads(STAGE4_FROZEN.read_text(encoding="utf-8"))
    if frozen.get("status") != "FROZEN":
        raise SystemExit("REFUSED: Stage-4 cells are not FROZEN")

    anchors_path = d / "C3_STAGE1_ANCHORS.jsonl"
    manifest_path = d / "C3_STAGE1_MANIFEST.json"
    if not anchors_path.exists() or not manifest_path.exists():
        raise SystemExit(f"REFUSED: fresh census incomplete under {d}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # POLICY UNIVERSE must be exactly the three frozen policies.
    got = sorted(int(s) for s in manifest.get("seeds", []))
    want = sorted(int(p) for p in frozen["policies"])
    if got != want:
        raise SystemExit(
            f"REFUSED: census covers policies {got}, frozen requires {want}. A "
            "subset cannot be judged against a >=2/3-of-3 replication rule."
        )
    if int(manifest.get("episodes_per_cell", -1)) != int(frozen["seeds"]["episodes_per_cell"]):
        raise SystemExit("REFUSED: episodes_per_cell does not match the frozen cells")
    if int(manifest.get("discovery_seed_base", -1)) != int(frozen["seeds"]["base"]):
        raise SystemExit("REFUSED: census seed base does not match the frozen Stage-4 base")

    anchors = [json.loads(l) for l in anchors_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    leg1 = analyze_leg1(anchors, frozen)

    sample_manifest = None
    sm = d / "C3_STAGE4_SAMPLE_MANIFEST.json"
    if sm.exists():
        sample_manifest = json.loads(sm.read_text(encoding="utf-8"))
    results_path = d / "C3_STAGE3_ANCHOR_RESULTS.jsonl"
    results = ([json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
               if results_path.exists() else [])
    leg2 = analyze_leg2(results, sample_manifest)

    confirms = {}
    for p in [str(x) for x in want]:
        l1 = leg1["per_policy"][p]["LEG1_PASS"]
        l2 = leg2.get("per_policy", {}).get(p, {}).get("LEG2_PASS")
        confirms[p] = {
            "LEG1_PASS": l1, "LEG2_PASS": l2,
            "POLICY_CONFIRM": (bool(l1) and bool(l2)) if l2 is not None else None,
        }
    decided = [v["POLICY_CONFIRM"] for v in confirms.values()]
    stage4 = (sum(1 for c in decided if c) >= 2) if all(c is not None for c in decided) else None

    doc = {
        "record": "C3 Stage-4 fresh confirmation",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "frozen_cells": "artifacts/c3_discovery/C3_STAGE4_CONFIRMATION_FROZEN.json",
        "frozen_cells_sha256": _sha256(STAGE4_FROZEN),
        "seed_block": frozen["seeds"]["range"],
        "leg_1": leg1,
        "leg_2": leg2,
        "per_policy_confirmation": confirms,
        "replication_rule": ">= 2 of 3 policies must pass BOTH legs",
        "STAGE4_PASS": stage4,
        "verdict": ("STAGE4_PASS" if stage4 else "STAGE4_FAIL") if stage4 is not None else "PENDING_LEG2",
        "authorizes": "O3 training only" if stage4 else "nothing",
        "does_not_establish": ["latent necessity", "distinct strategy",
                               "preference reversal", "repertoire gain"],
    }
    d.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print("=" * 74)
    print("C3 STAGE-4 CONFIRMATION")
    print("=" * 74)
    print(f"  LEG 1 floor: {leg1['floor']}   (zero-anchor episodes included)")
    for p, v in leg1["per_policy"].items():
        print(f"    {p}: {v['anchors_per_episode']:.3f}/ep  LCB95={v['LCB95']:.3f}  "
              f"zeros={v['n_episodes_with_zero_anchors']}/{v['n_episodes']}  -> {v['LEG1_PASS']}")
    if leg2.get("status") == "COMPLETE":
        print("  LEG 2 threshold: 0.20")
        for p, v in leg2["per_policy"].items():
            print(f"    {p}: rate={v['weighted_fork_rate']:.3f}  LCB95={v['LCB95']:.3f} -> {v['LEG2_PASS']}")
    else:
        print("  LEG 2: PENDING (fresh counterfactual screen not yet run)")
    print(f"\n  VERDICT: {doc['verdict']}")
    print(f"  wrote {RESULT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
