#!/usr/bin/env python3
"""Analyze G0 variant-tag sweep with synonym collapse to canonical niches.

The 21 declared tags are not 21 distinct behaviors. Historical synonyms
(e.g. OP6_TURTLE, OP7_SWITCHER) resolve to the seven LRO niches already
tested in the BASE sweep. This wrapper:

  1. loads episode_rows.csv from the variant sweep;
  2. adds opponent_canonical via canonicalize_opponent_key;
  3. writes a niche-collapsed CSV;
  4. runs analyze_g0_weakness on the collapsed file (one cell per niche).

If the collapsed analysis still finds no C1, do not hunt among synonym
labels — proceed to the same-map scenario bank (see
docs/same-map-tactical-regimes.md).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from gpu_env._core._bt_profiles import (  # noqa: E402
    LRO_AUDITED_OPPONENT_POOL,
    canonicalize_opponent_key,
)


def collapse(rows_path: Path, out_path: Path) -> dict:
    """Rewrite rows with opponent := canonical niche; keep first-seen label count."""
    by_canon: dict[str, set[str]] = defaultdict(set)
    n_in = n_out = 0
    with open(rows_path, newline="") as fin, open(out_path, "w", newline="") as fout:
        reader = csv.DictReader(fin)
        if not reader.fieldnames:
            raise SystemExit(f"[abort] empty or headerless: {rows_path}")
        writer = csv.DictWriter(fout, fieldnames=list(reader.fieldnames))
        writer.writeheader()
        for row in reader:
            n_in += 1
            raw = (row.get("opponent") or "").strip()
            canon = canonicalize_opponent_key(raw)
            by_canon[canon].add(raw)
            row["opponent"] = canon
            writer.writerow(row)
            n_out += 1
    return {
        "rows_in": n_in,
        "rows_out": n_out,
        "labels_per_niche": {k: sorted(v) for k, v in sorted(by_canon.items())},
        "n_canonical": len(by_canon),
        "expected_niches": list(LRO_AUDITED_OPPONENT_POOL),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rows", default="artifacts/g0_variant_weakness_sweep/episode_rows.csv")
    p.add_argument("--out-dir", default="artifacts/g0_variant_weakness_sweep")
    p.add_argument("--collapse-only", action="store_true",
                   help="Write collapsed CSV + synonym report; skip analyzer.")
    args = p.parse_args()

    rows_path = PROJECT_ROOT / args.rows
    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    collapsed = out_dir / "episode_rows_canonical.csv"
    report_path = out_dir / "synonym_collapse.json"

    if not rows_path.exists() or rows_path.stat().st_size == 0:
        print(f"[abort] no rows yet: {rows_path}", file=sys.stderr)
        return 1

    meta = collapse(rows_path, collapsed)
    import json
    report_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[collapse] {meta['rows_in']} rows -> {collapsed}")
    print(f"[collapse] {meta['n_canonical']} canonical niches "
          f"(expected {len(LRO_AUDITED_OPPONENT_POOL)})")
    for niche, labels in meta["labels_per_niche"].items():
        print(f"  {niche}: {labels}")

    if args.collapse_only:
        return 0

    # Reuse the locked BASE analyzer on the niche-collapsed rows.
    # analyze_g0_weakness prints to stdout; capture into analysis_canonical.txt.
    import io
    from contextlib import redirect_stdout
    from experiments.analyze_g0_weakness import main as analyze_main

    sys.argv = ["analyze_g0_weakness.py", "--rows", str(collapsed)]
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = int(analyze_main())
    text = buf.getvalue()
    (out_dir / "analysis_canonical.txt").write_text(text)
    print(text, end="")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
