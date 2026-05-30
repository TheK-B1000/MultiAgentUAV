#!/usr/bin/env python3
"""Print the marginal P(z) implied by ``strategy_occupancy_*`` columns in an episodes CSV.

The output is a single line of comma-separated probabilities suitable for piping into
``eval_op4_zero_shot.py --latent-mode shuffled --latent-marginal "..."``.

Usage:

    python experiments/_z_marginal_from_csv.py path/to/eval_<run_tag>_OP3_200ep.csv [--k 4]

If ``strategy_occupancy_{z}`` columns are missing (e.g. no-latent runs) or rows are empty,
prints a uniform marginal of length ``--k`` (default 4).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv_path", type=str, help="Path to a per-episode eval CSV with strategy_occupancy_* columns.")
    p.add_argument("--k", type=int, default=4, help="Latent K (used as fallback length; default 4).")
    p.add_argument(
        "--min-rows",
        type=int,
        default=1,
        help="Minimum non-empty rows before trusting the marginal (else uniform). Default 1.",
    )
    args = p.parse_args()

    K = max(1, int(args.k))
    if not os.path.isfile(args.csv_path):
        print(",".join(f"{1.0 / K:.6f}" for _ in range(K)))
        return 0

    sums = [0.0] * K
    rows = 0
    try:
        with open(args.csv_path, "r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if not any(k.startswith("strategy_occupancy_") for k in row.keys()):
                    break
                has_any = False
                for z in range(K):
                    raw = row.get(f"strategy_occupancy_{z}", "")
                    if raw == "" or raw is None:
                        continue
                    try:
                        sums[z] += float(raw)
                        has_any = True
                    except (TypeError, ValueError):
                        pass
                if has_any:
                    rows += 1
    except Exception as exc:
        print(f"[_z_marginal_from_csv] read failed: {exc}", file=sys.stderr)
        print(",".join(f"{1.0 / K:.6f}" for _ in range(K)))
        return 0

    if rows < args.min_rows or sum(sums) <= 0.0:
        print(",".join(f"{1.0 / K:.6f}" for _ in range(K)))
        return 0

    total = sum(sums)
    marginal = [s / total for s in sums]
    print(",".join(f"{p:.6f}" for p in marginal))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
