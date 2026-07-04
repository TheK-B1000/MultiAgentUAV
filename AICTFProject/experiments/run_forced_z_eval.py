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
import traceback
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
from experiments.forced_z_eval.io import (  # noqa: E402
    append_episode_rows,
    atomic_write_json,
    load_episode_results,
    write_manifest,
)
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
    EPISODE_RESULTS_CSV,
    RUN_MANIFEST_JSON,
    ForcedZProtocol,
    audit_protocol_note,
)
from experiments.forced_z_eval.env_overrides import resolve_forced_z_env_overrides  # noqa: E402
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
    p.add_argument("--max-decision-steps", type=int, default=None, help="Override episode horizon (default 400 or from run config)")
    p.add_argument("--run-config", default=None, help="Training run_config.json with max_decision_steps and env_surface_* fields")
    p.add_argument(
        "--inherit-training-config",
        action="store_true",
        help="Load env horizon/surface overrides from sibling *_run_config.json next to checkpoint",
    )
    return p.parse_args()


def _write_json(path: Path, payload: dict) -> None:
    atomic_write_json(path, payload)


def _partial_summary(protocol: ForcedZProtocol, completed_conditions: list[dict], episode_count: int) -> dict:
    expected = len(protocol.opponents) * len(protocol.maps) * len(protocol.latents)
    return {
        "status": "running",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": protocol.checkpoint,
        "completed_condition_count": len(completed_conditions),
        "expected_condition_count": expected,
        "episode_count": int(episode_count),
        "completed_conditions": completed_conditions,
    }


def _write_failure_report(run_dir: Path, *, protocol: ForcedZProtocol, reason: str, exc_text: str | None = None) -> None:
    atomic_write_json(
        run_dir / "failure_report.json",
        {
            "status": "failed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": protocol.checkpoint,
            "reason": reason,
            "traceback": exc_text,
        },
    )


def _validate_completed_artifacts(run_dir: Path) -> None:
    expected = [
        RUN_MANIFEST_JSON,
        EPISODE_RESULTS_CSV,
        STAGE_C_JSON,
        COMPLEMENTARITY_JSON,
        ORACLE_JSON,
        BEHAVIOR_JSON,
    ]
    missing = [name for name in expected if not (run_dir / name).exists()]
    if missing:
        raise RuntimeError(f"Eval exited but missing expected artifacts: {missing}")
    manifest = json.loads((run_dir / RUN_MANIFEST_JSON).read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError(f"Eval did not complete cleanly: {manifest}")


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
        manifest = json.loads((run_dir / RUN_MANIFEST_JSON).read_text(encoding="utf-8"))
        write_manifest(
            run_dir,
            protocol=protocol,
            status="completed",
            episode_count=sum(len(v) for v in cells.values()),
            completed_conditions=manifest.get("completed_conditions", []),
            extra_manifest={"analysis_only": True},
        )
        _validate_completed_artifacts(run_dir)
        return

    if args.analyze_only:
        print("ERROR: --analyze-only requires --from-run")
        sys.exit(1)
    if not args.checkpoint:
        print("ERROR: --checkpoint is required for simulation")
        sys.exit(1)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_dir or (SCRIPT_DIR / "forced_z_runs" / stamp))
    max_steps, env_reward_kwargs, run_config_source = resolve_forced_z_env_overrides(
        checkpoint=str(args.checkpoint),
        run_config_path=args.run_config,
        inherit_training_config=bool(args.inherit_training_config),
        max_decision_steps=args.max_decision_steps,
    )
    protocol = ForcedZProtocol(
        checkpoint=str(args.checkpoint),
        opponents=tuple(args.opponents),
        maps=tuple(args.maps),
        latents=DEFAULT_LATENTS,
        episodes_per_cell=int(args.episodes),
        base_seed=int(args.base_seed),
        deterministic_actions=not bool(args.stochastic),
        max_decision_steps=int(max_steps),
        env_reward_kwargs=dict(env_reward_kwargs),
        training_run_config=run_config_source,
        device=str(args.device),
        collect_behavior_mean=not bool(args.no_behavior_telemetry),
        progress_every=int(args.progress_every),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    completed_conditions: list[dict] = []
    episode_count = 0
    write_manifest(
        run_dir,
        protocol=protocol,
        status="running",
        started_at_utc=started_at,
        completed_conditions=completed_conditions,
        episode_count=episode_count,
    )
    atomic_write_json(run_dir / "partial_summary.json", _partial_summary(protocol, completed_conditions, episode_count))

    def _on_cell_complete(key, eps) -> None:
        nonlocal episode_count
        opponent, z, map_name = key
        append_episode_rows(run_dir, protocol=protocol, cells={key: eps})
        episode_count += len(eps)
        completed_conditions.append(
            {
                "opponent": opponent,
                "latent_z": int(z),
                "map": map_name,
                "episodes": len(eps),
            }
        )
        write_manifest(
            run_dir,
            protocol=protocol,
            status="running",
            started_at_utc=started_at,
            completed_conditions=completed_conditions,
            episode_count=episode_count,
        )
        atomic_write_json(run_dir / "partial_summary.json", _partial_summary(protocol, completed_conditions, episode_count))

    try:
        print(audit_protocol_note(protocol))
        cells = run_forced_z_episodes(protocol, on_cell_complete=_on_cell_complete)
        analyze_run(protocol, cells, run_dir, oracle_metric=args.oracle_metric)
        write_manifest(
            run_dir,
            protocol=protocol,
            status="completed",
            started_at_utc=started_at,
            completed_conditions=completed_conditions,
            episode_count=episode_count,
        )
        _validate_completed_artifacts(run_dir)
    except KeyboardInterrupt:
        write_manifest(
            run_dir,
            protocol=protocol,
            status="interrupted",
            started_at_utc=started_at,
            completed_conditions=completed_conditions,
            episode_count=episode_count,
            error="KeyboardInterrupt",
        )
        _write_failure_report(run_dir, protocol=protocol, reason="KeyboardInterrupt")
        raise
    except Exception as exc:
        exc_text = traceback.format_exc()
        write_manifest(
            run_dir,
            protocol=protocol,
            status="failed",
            started_at_utc=started_at,
            completed_conditions=completed_conditions,
            episode_count=episode_count,
            error=str(exc),
        )
        _write_failure_report(run_dir, protocol=protocol, reason=str(exc), exc_text=exc_text)
        raise


if __name__ == "__main__":
    main()
