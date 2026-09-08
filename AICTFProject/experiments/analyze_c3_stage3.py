"""C3 Stage-3 analysis: weighted natural fork rate + clustered bootstrap.

Amendments: C3_STAGE3_SAMPLING_AMENDMENT.json (74b857c)
            C3_STAGE3_LCB_DEFINITION_AMENDMENT.json  (LCB95 := q_0.025)
Contract:   C3_DISCOVERY_PREREG_FROZEN.json

Read-only. Implemented BEFORE any Stage-3 outcome exists, so no analytical
choice can be made after seeing results. The procedure was already frozen; this
removes the remaining flexibility by making it executable rather than described.

THE ESTIMATOR
-------------
Equal allocation across strata would change the estimand: 10 anchors are drawn
per policy x opponent cell regardless of how common that cell is in natural
play. Weighting back to the complete census recovers the population quantity:

    W_h   = N_h / N          from the COMPLETE Stage-1 census
    p_h   = qualified_h / sampled_h      within the sampled anchors of h
    p_hat = sum_h W_h * p_h

THE INTERVAL
------------
Stratified, episode-clustered bootstrap. Several anchors can come from one
trajectory, so resampling anchors directly would understate the interval.
Episodes are resampled with replacement WITHIN each stratum; the anchors of the
drawn episodes form that stratum's replicate; p_h is recomputed and combined
with the FROZEN W_h.

    resamples 2000, seed 12345, LCB95 = q_0.025

PASS iff LCB95 > 0.20.

Run:  python experiments/analyze_c3_stage3.py
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

from rl.analysis.c3_discovery_artifacts import (  # noqa: E402
    STAGE3_RESULTS_NAME,
    read_jsonl,
)

OUT_DIR = PROJECT_ROOT / "artifacts" / "c3_discovery"
SAMPLE_MANIFEST = OUT_DIR / "C3_STAGE3_SAMPLE_MANIFEST.json"
LCB_AMENDMENT = OUT_DIR / "C3_STAGE3_LCB_DEFINITION_AMENDMENT.json"
RESULT_PATH = OUT_DIR / "C3_STAGE3_RESULT.json"

QUALIFIED = "QUALIFIED_COMMITMENT_FORK"
NOT_QUALIFIED = "NO_COMMITMENT_FORK"
VERDICT_FIELD = "episode_status"
VALID_VERDICTS = frozenset({QUALIFIED, NOT_QUALIFIED})


def is_qualified(row: dict) -> bool:
    """Read one anchor's verdict, refusing anything this reader cannot interpret.

    An earlier draft guessed at verdict/status/fork_verdict, found none of them
    in the persisted rows, and silently scored EVERY anchor as unqualified --
    which would have written C3_NOT_PASS after nine hours of compute, looking
    exactly like a legitimate negative result. A software failure must never be
    convertible into a scientifically plausible negative, so both a missing
    field and an unrecognised value abort.
    """
    if VERDICT_FIELD not in row:
        raise SystemExit(
            f"REFUSED: Stage-3 row has no {VERDICT_FIELD!r} field "
            f"(keys: {sorted(row)[:12]}). Refusing to score a verdict this "
            "reader cannot find; that would silently produce a false negative."
        )
    value = str(row[VERDICT_FIELD])
    if value not in VALID_VERDICTS:
        raise SystemExit(
            f"REFUSED: unrecognised {VERDICT_FIELD}={value!r}. Expected one of "
            f"{sorted(VALID_VERDICTS)}. An unknown verdict must abort rather "
            "than be counted as 'not qualified'."
        )
    return value == QUALIFIED


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _stratum_of(row: dict) -> str:
    return f"{int(row['train_seed'])}|{str(row['opponent'])}"


def _episode_of(row: dict) -> str:
    return f"{int(row['train_seed'])}|{row['opponent']}|{int(row['eval_seed'])}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(OUT_DIR / STAGE3_RESULTS_NAME))
    args = ap.parse_args()

    if not SAMPLE_MANIFEST.exists():
        raise SystemExit(f"REFUSED: sampled-anchor manifest missing at {SAMPLE_MANIFEST}")
    manifest = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))

    if not LCB_AMENDMENT.exists():
        raise SystemExit(
            f"REFUSED: LCB definition amendment missing at {LCB_AMENDMENT}. "
            "LCB95 is ambiguous without it."
        )
    lcb_doc = json.loads(LCB_AMENDMENT.read_text(encoding="utf-8"))
    lcb_q = str(lcb_doc["resolution"]["LCB95"])
    if lcb_q != "q_0.025":
        raise SystemExit(f"REFUSED: unexpected LCB definition {lcb_q!r}")
    lcb_pct = 2.5

    results = read_jsonl(Path(args.results))
    if not results:
        raise SystemExit(f"REFUSED: no Stage-3 results at {args.results}")

    W_h = {k: float(v) for k, v in manifest["W_h"].items()}
    n_h_planned = {k: int(v) for k, v in manifest["n_h_sampled"].items()}
    selected = set(manifest["selected_anchor_ids"])
    threshold = 0.20
    resamples = 2000
    seed = 12345

    # ---- integrity: results must be exactly the sampled anchors -----------
    seen: dict[str, dict] = {}
    for row in results:
        key = row.get("anchor_key") or (
            f"{row['train_seed']}|{row['opponent']}|{row['eval_seed']}|{row['pressure_step']}"
        )
        seen[key] = row
    extra = sorted(set(seen) - selected)
    missing = sorted(selected - set(seen))
    if extra:
        raise SystemExit(
            f"REFUSED: {len(extra)} Stage-3 results are for anchors NOT in the "
            f"sampled manifest, e.g. {extra[:3]}. The evaluated set must equal "
            "the drawn set."
        )
    if missing:
        raise SystemExit(
            f"REFUSED: {len(missing)} sampled anchors have no Stage-3 result, "
            f"e.g. {missing[:3]}. Analysis of a partial sample is not the "
            "preregistered estimator."
        )

    # ---- per-stratum point estimates -------------------------------------
    by_stratum: dict[str, list[dict]] = defaultdict(list)
    for key, row in seen.items():
        by_stratum[_stratum_of(row)].append(row)

    p_h: dict[str, float] = {}
    per_stratum: dict[str, dict] = {}
    for h, w in W_h.items():
        rows = by_stratum.get(h, [])
        n = len(rows)
        q = sum(1 for r in rows if is_qualified(r))
        p = (q / n) if n else 0.0
        p_h[h] = p
        per_stratum[h] = {
            "N_h": manifest["N_h"][h], "W_h": round(w, 8),
            "n_h_planned": n_h_planned.get(h, 0), "n_h_evaluated": n,
            "qualified": q, "p_h": round(p, 6),
        }
        if w > 0 and n == 0:
            raise SystemExit(
                f"REFUSED: stratum {h} carries weight {w:.6f} but has no evaluated "
                "anchors. That is a coverage failure, not a zero rate."
            )

    p_hat = float(sum(W_h[h] * p_h[h] for h in W_h))

    # ---- stratified, episode-clustered bootstrap -------------------------
    eps_by_stratum: dict[str, list[str]] = defaultdict(list)
    rows_by_episode: dict[str, list[dict]] = defaultdict(list)
    for row in seen.values():
        h = _stratum_of(row)
        e = _episode_of(row)
        rows_by_episode[e].append(row)
        if e not in eps_by_stratum[h]:
            eps_by_stratum[h].append(e)
    for h in eps_by_stratum:
        eps_by_stratum[h].sort()

    rng = np.random.default_rng(seed)
    draws = np.empty(resamples, dtype=float)
    for i in range(resamples):
        total = 0.0
        for h, w in W_h.items():
            if w <= 0.0:
                continue
            eps = eps_by_stratum.get(h, [])
            if not eps:
                continue
            idx = rng.integers(0, len(eps), len(eps))
            n = q = 0
            for j in idx:
                for r in rows_by_episode[eps[int(j)]]:
                    n += 1
                    q += int(is_qualified(r))
            total += w * ((q / n) if n else 0.0)
        draws[i] = total

    lcb = float(np.percentile(draws, lcb_pct))
    ucb = float(np.percentile(draws, 100 - lcb_pct))
    passed = bool(lcb > threshold)

    doc = {
        "record": "C3 Stage-3 result — controllability screen",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "science_scope": "CONTROLLABILITY_SCREEN_ONLY",
        "o3_authorized_by_this_result": False,
        "latent_necessity_claim": False,
        "verdict": "C3_PASS" if passed else "C3_NOT_PASS",
        "decision_rule": "C3_PASS iff q_0.025(bootstrap of weighted natural fork_rate) > 0.20",
        "threshold": threshold,
        "weighted_natural_fork_rate": round(p_hat, 6),
        "LCB95": round(lcb, 6),
        "UCB95": round(ucb, 6),
        "LCB95_definition": "q_0.025 (two-sided 95% lower endpoint)",
        "bootstrap": {
            "resamples": resamples, "seed": seed,
            "cluster_unit": "episode", "stratified_by": "policy x opponent",
        },
        "n_anchors_evaluated": len(seen),
        "n_anchors_sampled": len(selected),
        "n_qualified_total": sum(v["qualified"] for v in per_stratum.values()),
        "per_stratum": per_stratum,
        "provenance": {
            "sample_manifest_sha256": _sha256(SAMPLE_MANIFEST),
            "lcb_amendment_sha256": _sha256(LCB_AMENDMENT),
            "amendment_commit": manifest.get("amendment_commit"),
            "sampling_seed": manifest.get("sampling_seed"),
            "c3_contract_sha256": manifest.get("c3_contract_sha256"),
        },
        "interpretation": {
            "what_a_pass_means": (
                "Among natural carrier-pressure anchors, a qualifying commitment "
                "fork exists often enough that a different response could plausibly "
                "change outcomes. It does NOT mean a second strategy exists."
            ),
            "what_it_does_not_authorize": "O3 training, latent birth, or any router work",
            "next_step_if_pass": "fresh confirmation on an unspent seed block, then O3",
        },
    }
    RESULT_PATH.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    print("=" * 74)
    print("C3 STAGE-3 RESULT")
    print("=" * 74)
    print(f"  anchors evaluated : {len(seen)} / {len(selected)} sampled")
    print(f"  qualified forks   : {doc['n_qualified_total']}")
    print(f"  weighted rate     : {p_hat:.4f}")
    print(f"  95% interval      : [{lcb:.4f}, {ucb:.4f}]   (LCB = q_0.025)")
    print(f"  threshold         : {threshold}")
    print(f"\n  VERDICT           : {doc['verdict']}")
    if not passed:
        print("  C3 does not justify O3. Do not adjust the threshold or the")
        print("  LCB definition; both were frozen before any outcome existed.")
    print(f"\n  wrote {RESULT_PATH.relative_to(PROJECT_ROOT)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
