#!/usr/bin/env python3
"""CLI for the v6i4 router-ablation evaluation protocol."""

from __future__ import annotations

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.evaluation.router_ablation import run_suite


def _parse_int_list(values: list[str] | None) -> list[int]:
    if not values:
        return []
    out: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.append(int(part))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run v6i4 router-ablation evaluation on a frozen v6i2 checkpoint. "
            "This command never trains or updates model parameters."
        )
    )
    parser.add_argument("--preset", default="v6i4", help="Protocol label for manifest metadata.")
    parser.add_argument("--checkpoint", required=True, help="Promoted v6i2 checkpoint .zip")
    parser.add_argument("--output-dir", required=True, help="Directory for v6i4 artifacts.")
    parser.add_argument("--opponents", nargs="+", default=["OP5", "OP6", "OP7"])
    parser.add_argument("--map-sets", nargs="+", default=["eval"], choices=["train", "eval"])
    parser.add_argument("--map-layout", default="map_a_open")
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--latent-k", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--calibration-seeds", nargs="+", default=["1000,1001,1002,1003"])
    parser.add_argument("--test-seeds", nargs="+", default=["2000,2001,2002,2003"])
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic action sampling instead of deterministic eval.")
    parser.add_argument(
        "--exploratory-allow-unpromoted",
        action="store_true",
        help="Allow exploratory evaluation of a checkpoint that metadata does not mark as promoted.",
    )
    args = parser.parse_args()

    paths = run_suite(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        preset=args.preset,
        opponents=[str(o).upper() for o in args.opponents],
        map_sets=[str(m).lower() for m in args.map_sets],
        calibration_seeds=_parse_int_list(args.calibration_seeds),
        test_seeds=_parse_int_list(args.test_seeds),
        agents=args.agents,
        latent_k=args.latent_k,
        map_layout=args.map_layout,
        device=args.device,
        deterministic_actions=not bool(args.stochastic),
        exploratory_allow_unpromoted=bool(args.exploratory_allow_unpromoted),
        n_bootstrap=int(args.n_bootstrap),
    )
    print("[v6i4] wrote artifacts:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
