#!/usr/bin/env python3
"""Run matched-seed forced-z episodes once; emit all analysis reports.

Pipeline::

    run_forced_z_eval.py  →  episode_results.csv + run_manifest.json
                         →  stage_c_report.json
                         →  complementarity_report.json
                         →  oracle_report.json
                         →  behavior_report.json

Re-run analysis only (no simulation)::

    run_forced_z_eval.py --from-run experiments/forced_z_runs/<stamp> --analyze-only
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.forced_z_eval.analysis.behavior import build_behavior_report  # noqa: E402
from experiments.forced_z_eval.analysis.complementarity import (  # noqa: E402
    build_complementarity_report,
    print_complementarity_report,
)
from experiments.forced_z_eval.analysis.oracle import build_oracle_report  # noqa: E402
from experiments.forced_z_eval.analysis.stage_c import build_stage_c_report, print_stage_c_report  # noqa: E402
from experiments.forced_z_eval.io import load_episode_results, write_run_artifacts  # noqa: E402
from experiments.forced_z_eval.protocol import (  # noqa: E402
    BEHAVIOR_JSON,
    COMPLEMENTARITY_JSON,
    DEFAULT_BASE_SEED,
    DEFAULT_EPISODES_PER_CELL,
    DEFAULT_LATENTS,
    DEFAULT_MAPS,
    DEFAULT_OPPONENTS,
    ORACLE_JSON,
    STAGE_C_JSON,
    ForcedZProtocol,
    audit_protocol_note,
)
from experiments.forced_z_eval.runner import run_forced_z_episodes  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Canonical forced-z eval: simulate once, analyze many")
    p.add_argument("--checkpoint", default=None, help="Checkpoint .zip (required unless --from-run)")
    p.add_argument("--from-run", default=None, help="Existing run directory with episode_results.csv")
    p.add_argument("--analyze-only", action="store_true", help="Skip simulation; analyze --from-run only")
    p.add_argument("--out-dir", default=None, help="Output directory for a new run")
    p.add_argument("--episodes", type=int, default=DEFAULT_EPISODES_PER_CELL)
    p.add_argument("--device", default="cuda")
    p.add_argument("--opponents", nargs="+", default=list(DEFAULT_OPPONENTS))
    p.add_argument("--maps", nargs="+", default=list(DEFAULT_MAPS))
    p.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    p.add_argument("--oracle-metric", choices=("return", "win_margin", "success"), default="return")
    p.add_argument("--stochastic", action="store_true")
    p.add_argument("--no-behavior-telemetry", action="store_true")
    p.add_argument("--progress-every", type=int, default=25)
    return p.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analyze_run(protocol: ForcedZProtocol, cells, run_dir: Path, *, oracle_metric: str) -> None:
    opponents = list(protocol.opponents)
    maps = list(protocol.maps)
    latents = tuple(protocol.latents)
    stage_c = build_stage_c_report(cells, opponents=opponents, maps=maps, latents=latents)
    oracle = build_oracle_report(cells, opponents=opponents, maps=maps, latents=latents, metric=oracle_metric)
    behavior = build_behavior_report(cells, opponents=opponents, maps=maps, latents=latents)
    complementarity = build_complementarity_report(
        cells, opponents=opponents, maps=maps, latents=latents, metric=oracle_metric
    )
    _write_json(run_dir / STAGE_C_JSON, stage_c)
    _write_json(run_dir / ORACLE_JSON, oracle)
    _write_json(run_dir / BEHAVIOR_JSON, behavior)
    _write_json(run_dir / COMPLEMENTARITY_JSON, complementarity)

    print("\n--- Stage C Gate ---")
    print_stage_c_report(stage_c)
    print_complementarity_report(complementarity)
    print(f"\nArtifacts in: {run_dir}")
    print(f"  episode_results.csv")
    print(f"  {STAGE_C_JSON}")
    print(f"  {COMPLEMENTARITY_JSON}")
    print(f"  {ORACLE_JSON}")
    print(f"  {BEHAVIOR_JSON}")


def main() -> None:
    args = _parse_args()
    try:
        import plot.eval_rollout  # noqa: F401
    except ImportError as exc:
        print(f"ERROR: eval infrastructure unavailable: {exc}")
        sys.exit(1)

    if args.from_run:
        run_dir = Path(args.from_run)
        protocol, cells = load_episode_results(run_dir)
        analyze_run(protocol, cells, run_dir, oracle_metric=args.oracle_metric)
        return

    if args.analyze_only:
        print("ERROR: --analyze-only requires --from-run")
        sys.exit(1)
    if not args.checkpoint:
        print("ERROR: --checkpoint is required for simulation")
        sys.exit(1)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir or (SCRIPT_DIR / "forced_z_runs" / stamp))
    protocol = ForcedZProtocol(
        checkpoint=str(args.checkpoint),
        opponents=tuple(args.opponents),
        maps=tuple(args.maps),
        latents=DEFAULT_LATENTS,
        episodes_per_cell=int(args.episodes),
        base_seed=int(args.base_seed),
        deterministic_actions=not bool(args.stochastic),
        device=str(args.device),
        collect_behavior_mean=not bool(args.no_behavior_telemetry),
        progress_every=int(args.progress_every),
    )
    print(audit_protocol_note())
    cells = run_forced_z_episodes(protocol)
    write_run_artifacts(run_dir, protocol=protocol, cells=cells)
    analyze_run(protocol, cells, run_dir, oracle_metric=args.oracle_metric)


if __name__ == "__main__":
    main()
