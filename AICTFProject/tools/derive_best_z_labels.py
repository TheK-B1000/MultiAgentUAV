"""Derive ``best_z_per_opponent`` labels for supervised q_phi (router) alignment.

Reads ``eval_*_fix_z{0..K-1}_<tag>_aggregate.csv`` files (produced by ``plot/eval_checkpoint.py``)
and writes a JSON bundle with three artifacts per opponent:

    * ``wr_by_z``  -- raw per-z success rates from the chosen map set
    * ``hard_z``   -- argmax over ``wr_by_z`` (single-class label)
    * ``soft``     -- ``softmax(wr_by_z / 100 / temperature)`` (probability target for CE)

The output schema also includes the opponent-id integer used by ``custom_ppo._opponent_id_int_from_info``
(OP1->0 .. OP7->6, with OP5_RUSHER==OP5), so the trainer can look up labels by ``opponent_id`` at minibatch
time without re-parsing tag strings.

Why soft labels matter
----------------------
When the per-z spread is small (e.g. OP3 at 99.2/99.0/99.8/98.6), an argmax label collapses to a single
class and CE just teaches q_phi to emit that class for everyone -- the same failure mode we already saw
with deterministic-deploy collapse to z=2. Soft labels keep mass on all z's proportional to their win
rate so q_phi can express ``OP3 slightly prefers z2, OP5 strongly prefers z0`` even when the difference is
a few percentage points. Tune the temperature to control how sharp/soft the target is (smaller temp =
sharper).

Usage
-----
    python tools/derive_best_z_labels.py \
        --aggregate-dir csv \
        --tag op5_bite_v3_4v4 \
        --map-set eval \
        --temperature 0.05 \
        --out checkpoints/4v4/best_z_labels.json

Optional ``--include-opponents OP3 OP5_RUSHER OP6 OP7`` to filter; default uses all opponents present
in the aggregates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence


SCHEMA_VERSION = 1

OPP_NAME_TO_ID: Dict[str, int] = {
    "OP1": 0,
    "OP2": 1,
    "OP3": 2,
    "OP4": 3,
    "OP5_RUSHER": 4,
    "OP5": 4,
    "OP6": 5,
    "OP7": 6,
}


def _canonical_opp_name(raw: str) -> str:
    tag = str(raw).strip().upper()
    if tag == "OP5":
        return "OP5_RUSHER"
    if tag == "OP6_TURTLE":
        return "OP6"
    if tag == "OP7_SWITCHER":
        return "OP7"
    return tag


def _opp_id(raw: str) -> int:
    return OPP_NAME_TO_ID.get(_canonical_opp_name(raw), -1)


def _read_aggregate_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _wr_for_setting(rows: Sequence[Dict[str, str]], map_set: str, opponent: str) -> Optional[float]:
    canon = _canonical_opp_name(opponent)
    for row in rows:
        if str(row.get("map_set", "")).strip().lower() != map_set.lower():
            continue
        if _canonical_opp_name(row.get("opponent", "")) != canon:
            continue
        try:
            return float(row["success_rate"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


def _softmax(x: Sequence[float], temperature: float) -> List[float]:
    t = max(1e-6, float(temperature))
    scaled = [v / t for v in x]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    z = sum(exps) or 1.0
    return [e / z for e in exps]


def _discover_aggregates(aggregate_dir: Path, tag: str, k: int) -> Dict[int, Path]:
    """Find ``eval_*_fix_zN_<tag>_aggregate.csv`` for ``N in [0, k)``."""
    out: Dict[int, Path] = {}
    for z in range(k):
        # Eval pipeline names follow eval_<label>_<tag>_aggregate.csv where label embeds fix_zN.
        candidates = sorted(aggregate_dir.glob(f"eval_*_fix_z{z}_*_{tag}_aggregate.csv"))
        if not candidates:
            candidates = sorted(aggregate_dir.glob(f"eval_*_fix_z{z}_{tag}_aggregate.csv"))
        if not candidates:
            continue
        out[z] = candidates[0]
    return out


def _episode_count(rows: Sequence[Dict[str, str]], map_set: str, opponent: str) -> Optional[int]:
    canon = _canonical_opp_name(opponent)
    for row in rows:
        if str(row.get("map_set", "")).strip().lower() != map_set.lower():
            continue
        if _canonical_opp_name(row.get("opponent", "")) != canon:
            continue
        try:
            return int(row["episodes"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


def derive_labels(
    aggregate_paths: Dict[int, Path],
    *,
    map_set: str,
    temperature: float,
    include_opponents: Optional[Sequence[str]] = None,
) -> Dict:
    if not aggregate_paths:
        raise SystemExit("No fixed-z aggregate CSVs found; check --aggregate-dir and --tag.")

    per_z_rows: Dict[int, List[Dict[str, str]]] = {
        z: _read_aggregate_rows(path) for z, path in aggregate_paths.items()
    }
    k = max(per_z_rows.keys()) + 1
    if include_opponents:
        opponents = [_canonical_opp_name(o) for o in include_opponents]
    else:
        opponents = sorted(
            {_canonical_opp_name(r.get("opponent", "")) for rows in per_z_rows.values() for r in rows}
        )
        opponents = [o for o in opponents if o]

    out_opponents: Dict[str, Dict] = {}
    for opp in opponents:
        wrs: List[Optional[float]] = []
        n_eps: Optional[int] = None
        for z in range(k):
            rows = per_z_rows.get(z, [])
            wrs.append(_wr_for_setting(rows, map_set, opp))
            if n_eps is None:
                n_eps = _episode_count(rows, map_set, opp)
        if any(w is None for w in wrs):
            print(f"[derive] WARNING: missing per-z WR for {opp} on map_set={map_set}; skipping.")
            continue
        clean_wrs: List[float] = [float(w) for w in wrs]  # type: ignore[arg-type]
        soft = _softmax([w / 100.0 for w in clean_wrs], temperature)
        hard_z = int(max(range(k), key=lambda z: clean_wrs[z]))
        out_opponents[opp] = {
            "opponent_id": _opp_id(opp),
            "wr_by_z": clean_wrs,
            "hard_z": hard_z,
            "soft": soft,
            "n_episodes_per_z": int(n_eps) if n_eps is not None else None,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "map_set": map_set,
        "temperature": float(temperature),
        "k": k,
        "opponent_id_map": {
            str(v): _canonical_opp_name(k_)
            for k_, v in OPP_NAME_TO_ID.items()
            if k_ == _canonical_opp_name(k_)
        },
        "sources": {str(z): str(p) for z, p in sorted(aggregate_paths.items())},
        "opponents": out_opponents,
    }


def _print_summary(bundle: Dict) -> None:
    k = int(bundle["k"])
    print(f"K={k} map_set={bundle['map_set']} temperature={bundle['temperature']:.4f}")
    print(f"sources: {len(bundle['sources'])} aggregate file(s)")
    header = "  " + "opp".ljust(12) + "id   " + "  ".join(f"  z{z}  " for z in range(k)) + "  hard  soft~"
    print(header)
    for opp, payload in bundle["opponents"].items():
        wr = "  ".join(f"{v:5.1f}" for v in payload["wr_by_z"])
        soft = ",".join(f"{p:.2f}" for p in payload["soft"])
        oid = payload.get("opponent_id", -1)
        print(f"  {opp.ljust(12)}{str(oid).rjust(3)}   {wr}   z{payload['hard_z']}  [{soft}]")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--aggregate-dir",
        type=str,
        required=True,
        help="Directory containing eval aggregate CSVs (typically AICTFProject/csv).",
    )
    parser.add_argument(
        "--tag",
        type=str,
        required=True,
        help="Tag suffix common to all fixed-z aggregate files (e.g. 'op5_bite_v3_4v4').",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of latent strategies (default 4).",
    )
    parser.add_argument(
        "--map-set",
        choices=("train", "eval"),
        default="eval",
        help="Which map set's success rates to use for labels (default eval).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.05,
        help=(
            "Softmax temperature applied to wr/100 when computing soft labels. "
            "Smaller = sharper; 0.01 -> nearly one-hot; 1.0 -> nearly uniform. Default 0.05."
        ),
    )
    parser.add_argument(
        "--include-opponents",
        nargs="*",
        default=None,
        help="Optional explicit list of opponents (canonical names, e.g. OP3 OP5_RUSHER OP6 OP7).",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to write the labels JSON.",
    )
    args = parser.parse_args(argv)

    aggregate_dir = Path(args.aggregate_dir)
    if not aggregate_dir.is_dir():
        raise SystemExit(f"--aggregate-dir does not exist: {aggregate_dir}")

    aggregate_paths = _discover_aggregates(aggregate_dir, tag=args.tag, k=int(args.k))
    bundle = derive_labels(
        aggregate_paths,
        map_set=args.map_set,
        temperature=float(args.temperature),
        include_opponents=args.include_opponents,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path} ({len(bundle['opponents'])} opponent label(s)).\n")
    _print_summary(bundle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
