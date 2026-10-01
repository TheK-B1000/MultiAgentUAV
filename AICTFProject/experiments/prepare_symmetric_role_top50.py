"""Prepare / verify the symmetric-role top-50 diagnostic (PI 2026-10-01).

Locks the Ours top-50 seed_ids, slices read-only baselines from existing 128-seed
CSVs, and optionally preseeds a PARTIAL with reusable A-role cells so a symmetric
eval only spends the new B-role episodes.

Does not launch training or evaluation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
TOP = SD / "symmetric_role_top50"
SPEC = SD / "SYMMETRIC_ROLE_TOP50_DIAGNOSTIC_SPEC.json"
LOCK = TOP / "SEED_LOCK.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_ids(scale: str) -> list[int]:
    path = TOP / f"{scale}_ours_top50_seed_ids.json"
    ids = [int(s) for s in json.loads(path.read_text(encoding="utf-8"))]
    if len(ids) != 50 or len(set(ids)) != 50 or ids != sorted(ids):
        raise SystemExit(f"REFUSING: {path.name} must be 50 unique ascending seed ids")
    return ids


def verify_lock() -> None:
    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: {SPEC.name} is not frozen")
    for scale, meta in lock["scales"].items():
        ids = _load_ids(scale)
        got = _sha(TOP / f"{scale}_ours_top50_seed_ids.json")
        if got != meta["sha256"]:
            raise SystemExit(f"REFUSING: {scale} seed file hash drifted ({got} != {meta['sha256']})")
        lo, hi = (int(x) for x in meta["block"].split(".."))
        if not all(lo <= s <= hi for s in ids):
            raise SystemExit(f"REFUSING: {scale} seeds outside {meta['block']}")
        for label, entry in spec["POST_HOC_MATCHED_ROLE_ABLATIONS"].items():
            if scale.upper() not in label:
                continue
            frozen = [int(s) for s in entry["seed_ids"]]
            if frozen != ids:
                raise SystemExit(f"REFUSING: {label} seed_ids != locked {scale} list")
        print(f"  OK {scale}: n=50 block={meta['block']} sha={got[:16]}...")
    print(f"  OK lock+spec agree ({SPEC.name})")


def slice_rows(source_csv: str, seed_ids: list[int], out_csv: Path) -> int:
    src = SD / source_csv
    if not src.is_file():
        raise SystemExit(f"REFUSING: missing source rows {src}")
    want = set(seed_ids)
    with src.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["seed"]) in want]
    seeds_found = {int(r["seed"]) for r in rows}
    if seeds_found != want:
        missing = sorted(want - seeds_found)
        raise SystemExit(f"REFUSING: {source_csv} missing {len(missing)} frozen seeds "
                         f"(e.g. {missing[:5]})")
    cells = {(r["policy"], r["pole"]) for r in rows}
    expect = {("pi_A", "A"), ("pi_A", "B"), ("pi_B", "A"), ("pi_B", "B")}
    if cells != expect:
        raise SystemExit(f"REFUSING: {source_csv} top-50 cells={cells}, want {expect}")
    for pol, pole in expect:
        n = sum(1 for r in rows if r["policy"] == pol and r["pole"] == pole)
        if n != 50:
            raise SystemExit(f"REFUSING: {pol}@{pole} has {n} rows, want 50")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def slice_readouts() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for name, entry in spec["READOUTS"].items():
        scale = {"2V2": "2v2", "4V4": "4v4", "6V6": "6v6"}[name.split("_")[1]]
        ids = _load_ids(scale)
        out = TOP / f"{name.lower()}_rows.csv"
        n = slice_rows(entry["source_rows"], ids, out)
        meta = {
            "record": name,
            "action": "READ_ONLY_SLICE",
            "source_rows": entry["source_rows"],
            "n_rows": n,
            "n_seeds": 50,
            "seed_ids_file": str((TOP / f"{scale}_ours_top50_seed_ids.json").as_posix()),
            "out_csv": str(out.as_posix()),
        }
        (TOP / f"{name.lower()}_SLICE.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"  sliced {name}: {n} rows -> {out.name}")


def preseed_a_cells(label: str, source_csv: str, fingerprint: dict) -> Path:
    """Write PARTIAL with fingerprint + pi_A cells only, so --resume spends B cells."""
    scale = {"TOP50_2V2_SYMMETRIC_OURS": "2v2", "TOP50_4V4_SYMMETRIC_OURS": "4v4"}[label]
    ids = _load_ids(scale)
    want = set(ids)
    src = SD / source_csv
    with src.open(newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh)
                if int(r["seed"]) in want and r["policy"] == "pi_A"]
    if len(rows) != 100:
        raise SystemExit(f"REFUSING: expected 100 pi_A rows for {label}, got {len(rows)}")
    partial = SD / f"{label.lower()}_specialist_crossover_eval_rows.PARTIAL.jsonl"
    if partial.is_file():
        raise SystemExit(f"REFUSING: {partial.name} already exists")
    with partial.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"fingerprint": fingerprint}) + "\n")
        for r in rows:
            # coerce types used by the evaluator
            out = dict(r)
            for k in ("seed", "blue", "red", "margin"):
                if k in out:
                    out[k] = int(float(out[k]))
            if "win" in out:
                out["win"] = int(float(out["win"]))
            fh.write(json.dumps(out) + "\n")
    print(f"  preseeded {partial.name} with {len(rows)} pi_A rows (B cells remain to run)")
    return partial


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-lock", action="store_true")
    ap.add_argument("--slice-readouts", action="store_true")
    ap.add_argument("--preseed-a-cells", default="",
                    help="label TOP50_2V2_SYMMETRIC_OURS or TOP50_4V4_SYMMETRIC_OURS; "
                         "requires --fingerprint-json from a dry matching run plan")
    ap.add_argument("--fingerprint-json", default="",
                    help="JSON object = exact evaluator fingerprint for the symmetric run")
    ap.add_argument("--source-csv", default="",
                    help="asymmetric Ours rows CSV to copy pi_A cells from")
    args = ap.parse_args()
    if not any((args.verify_lock, args.slice_readouts, args.preseed_a_cells)):
        ap.error("pass --verify-lock and/or --slice-readouts and/or --preseed-a-cells")
    if args.verify_lock:
        verify_lock()
    if args.slice_readouts:
        verify_lock()
        slice_readouts()
    if args.preseed_a_cells:
        if not args.fingerprint_json or not args.source_csv:
            raise SystemExit("REFUSING: --preseed-a-cells needs --fingerprint-json and --source-csv")
        fp = json.loads(Path(args.fingerprint_json).read_text(encoding="utf-8"))
        preseed_a_cells(args.preseed_a_cells, args.source_csv, fp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
