#!/usr/bin/env python3
"""Print / validate a five-line experiment contract before any GPU launch.

Usage
-----
::

    uv run python experiments/print_experiment_contract.py \\
      --hypothesis "Distinct cell pressures create comparative advantage" \\
      --single-change "training_cell_distribution only; contract OFF" \\
      --primary-metric "cross-fitted context oracle - best fixed (CI)" \\
      --failure "CI includes 0 AND payoff rows approximately parallel" \\
      --max-budget "stop after micro-probe if rows stay parallel"

Exit code 0 always prints the contract. Exit code 2 if required fields missing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REQUIRED = (
    "hypothesis",
    "single_change",
    "primary_metric",
    "failure_condition",
    "max_budget",
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Five-line experiment contract")
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--single-change", required=True, dest="single_change")
    p.add_argument("--primary-metric", required=True, dest="primary_metric")
    p.add_argument("--failure", required=True, dest="failure_condition")
    p.add_argument("--max-budget", required=True, dest="max_budget")
    p.add_argument(
        "--out",
        default=None,
        help="Optional JSON path to write the contract.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    contract = {k: str(getattr(args, k)).strip() for k in REQUIRED}
    missing = [k for k, v in contract.items() if not v]
    if missing:
        print(f"ERROR: empty fields: {missing}", file=sys.stderr)
        return 2

    print("=" * 64)
    print("EXPERIMENT CONTRACT (do not launch GPU without this)")
    print("=" * 64)
    print(f"Hypothesis:        {contract['hypothesis']}")
    print(f"Single change:     {contract['single_change']}")
    print(f"Primary metric:    {contract['primary_metric']}")
    print(f"Failure condition: {contract['failure_condition']}")
    print(f"Maximum budget:    {contract['max_budget']}")
    print("=" * 64)
    print("Promotion ladder: audit → micro → candidate → confirm → distill → paper")
    print("Primary gates: G_available, G_retention, G_realized, G_latent")
    print("Supporting only: JSD, MI, entropy, embedding distance")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(contract, indent=2), encoding="utf-8")
        print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
