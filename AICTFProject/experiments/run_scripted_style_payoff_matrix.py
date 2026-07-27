#!/usr/bin/env python3
"""Collect scripted-blue x scripted-red payoff matrices.

This is a pool-admissibility diagnostic, not PPO training. It answers whether
the current red opponent pool creates real strategic tradeoffs for hand-coded
blue styles before spending more latent/PPO compute on specialist birth.

Matched-seed contract:
  episode seed = f(red, map, episode_index), independent of blue style.
That makes every blue style face the same red/map episode starts for a given
red/map/episode cell.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.payoff_matrix_analysis import (  # noqa: E402
    analyze_pool,
    cells_from_rows,
    format_report,
)
from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402
from gpu_env._core._scripted_blue_styles import BLUE_STYLE_NAMES  # noqa: E402


DEFAULT_REDS = (
    "OP6_IMMEDIATE_DUAL_RUSH",
    "OP7_DEEP_FORTRESS",
    "OP8_PROTECTED_CARRIER_ESCORT",
    "OP9_SPLIT_LANE_FEINT",
    "OP10_AGGRESSIVE_INTERCEPTOR",
    "OP11_ADAPTIVE_EXPLOITER",
    "OP12_LATE_CONVERTER",
)
DEFAULT_MAPS = ("map_b_split_lane", "map_b_split_lane_v2")
EPISODE_RESULTS_CSV = "episode_results.csv"
POOL_REPORT_JSON = "pool_report.json"
POOL_REPORT_TXT = "pool_report.txt"
RUN_MANIFEST_JSON = "run_manifest.json"
PARTIAL_SUMMARY_JSON = "partial_summary.json"

ROW_FIELDS = [
    "blue_style",
    "red_style",
    "map",
    "episode_index",
    "episode_seed",
    "success",
    "blue_score",
    "red_score",
    "win_margin",
    "steps",
    "return",
    "outcome",
    "time_to_first_score",
    "collision_free",
    "zone_coverage",
    "split_detector_first_trigger_step",
    "split_detector_active_steps",
    "split_detector_max_lateral_sep",
    "split_detector_max_teammate_dist",
    "escort_detector_first_trigger_step",
    "escort_detector_active_steps",
    "conversion_phase_first_step",
    "carrier_intercept_attempts",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--episodes", type=int, default=16)
    p.add_argument("--base-seed", type=int, default=260726)
    p.add_argument("--device", default="cuda")
    p.add_argument("--reds", nargs="+", default=list(DEFAULT_REDS))
    p.add_argument("--maps", nargs="+", default=list(DEFAULT_MAPS))
    p.add_argument("--blue-styles", nargs="+", default=list(BLUE_STYLE_NAMES))
    p.add_argument("--max-decision-steps", type=int, default=240)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--analysis-seed", type=int, default=0)
    p.add_argument(
        "--min-br-diversity",
        type=int,
        default=2,
        help=(
            "Minimum distinct blue best-responses across red columns. "
            "For OP6-OP10 K=4 repertoire acceptance use 4 (every blue must "
            "be uniquely best somewhere)."
        ),
    )
    p.add_argument("--progress-every", type=int, default=25)
    return p.parse_args()


def _episode_seed(base_seed: int, red_index: int, map_index: int, episode_index: int) -> int:
    """Seed keyed only by red/map/episode, never by blue style."""
    return int(base_seed) + int(red_index) * 100_000 + int(map_index) * 10_000 + int(episode_index)


def _make_env(*, map_name: str, seed: int, max_decision_steps: int, device: str) -> GPUCTFVecEnv:
    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=2,
        max_red_agents=2,
        map_layout=str(map_name),
        max_decision_steps=int(max_decision_steps),
        aquaticus_profile=True,
        rules_profile="OURS",
        device=str(device),
        seed=int(seed),
    )
    return GPUCTFVecEnv(cfg)


def _zero_action(env: GPUCTFVecEnv) -> Any:
    sample = env.action_space.sample()
    return np.zeros_like(sample)


def _episode_result_row(
    *,
    blue_style: str,
    red_style: str,
    map_name: str,
    episode_index: int,
    episode_seed: int,
    episode_result: dict[str, Any],
    reward_return: float,
    extra_telemetry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    blue_score = int(episode_result.get("blue_score", 0))
    red_score = int(episode_result.get("red_score", 0))
    win_margin = blue_score - red_score
    row = {
        "blue_style": blue_style,
        "red_style": red_style,
        "map": map_name,
        "episode_index": int(episode_index),
        "episode_seed": int(episode_seed),
        "success": 1 if win_margin > 0 else 0,
        "blue_score": blue_score,
        "red_score": red_score,
        "win_margin": win_margin,
        "steps": int(episode_result.get("decision_steps", 0)),
        "return": float(reward_return),
        "outcome": "win" if win_margin > 0 else ("loss" if win_margin < 0 else "draw"),
        "time_to_first_score": episode_result.get("time_to_first_score", ""),
        "collision_free": int(episode_result.get("collision_free_episode", 1)),
        "zone_coverage": float(episode_result.get("zone_coverage", 0.0)),
    }
    if extra_telemetry:
        row.update(extra_telemetry)
    return row


def _scalar_core_int(core: Any, attr: str, default: int = -1) -> int:
    val = getattr(core, attr, None)
    if val is None:
        return int(default)
    try:
        return int(val[0].item())
    except Exception:
        return int(default)


def _scalar_core_float(core: Any, attr: str, default: float = 0.0) -> float:
    val = getattr(core, attr, None)
    if val is None:
        return float(default)
    try:
        return float(val[0].item())
    except Exception:
        return float(default)


def _core_detector_telemetry(core: Any) -> dict[str, int | float]:
    first_trigger = _scalar_core_int(core, "bt_adapt_split_first_trigger_step", -1)
    return {
        "split_detector_first_trigger_step": first_trigger,
        "split_detector_active_steps": _scalar_core_int(core, "bt_adapt_split_active_steps", 0),
        "split_detector_max_lateral_sep": _scalar_core_float(core, "bt_adapt_split_max_lateral_sep", 0.0),
        "split_detector_max_teammate_dist": _scalar_core_float(core, "bt_adapt_split_max_teammate_dist", 0.0),
        "escort_detector_first_trigger_step": _scalar_core_int(
            core, "bt_adapt_opening_escort_first_trigger_step", -1
        ),
        "escort_detector_active_steps": _scalar_core_int(core, "bt_adapt_opening_escort_active_steps", 0),
        "conversion_phase_first_step": first_trigger,
        "carrier_intercept_attempts": _scalar_core_int(core, "bt_tel_intercept_attempts", 0),
    }


def _merge_detector_telemetry(
    current: dict[str, int | float],
    sample: dict[str, int | float],
) -> dict[str, int | float]:
    out = dict(current)
    for key in ("split_detector_first_trigger_step", "escort_detector_first_trigger_step"):
        cur = int(out.get(key, -1))
        val = int(sample.get(key, -1))
        if cur < 0 and val >= 0:
            out[key] = val
        elif cur >= 0 and val >= 0:
            out[key] = min(cur, val)
    for key in (
        "split_detector_active_steps",
        "escort_detector_active_steps",
        "split_detector_max_lateral_sep",
        "split_detector_max_teammate_dist",
        "carrier_intercept_attempts",
    ):
        out[key] = max(float(out.get(key, 0)), float(sample.get(key, 0)))
    first_split = int(out.get("split_detector_first_trigger_step", -1))
    out["conversion_phase_first_step"] = first_split
    return out


def _run_one_episode(
    *,
    blue_style: str,
    red_style: str,
    map_name: str,
    episode_index: int,
    episode_seed: int,
    max_decision_steps: int,
    device: str,
) -> dict[str, Any]:
    env = _make_env(
        map_name=map_name,
        seed=episode_seed,
        max_decision_steps=max_decision_steps,
        device=device,
    )
    try:
        core = env.core
        env.env_method("set_phase", red_style)
        env.env_method("set_next_opponent", "SCRIPTED", red_style)
        core.blue_scripted = True
        core.set_blue_style(blue_style)
        env.reset()
        env.env_method("set_phase", red_style)
        env.env_method("set_next_opponent", "SCRIPTED", red_style)
        core.blue_scripted = True
        core.set_blue_style(blue_style)

        ep_return = 0.0
        last_info: dict[str, Any] = {}
        detector_telemetry: dict[str, int | float] = {
            "split_detector_first_trigger_step": -1,
            "split_detector_active_steps": 0,
            "split_detector_max_lateral_sep": 0.0,
            "split_detector_max_teammate_dist": 0.0,
            "escort_detector_first_trigger_step": -1,
            "escort_detector_active_steps": 0,
            "conversion_phase_first_step": -1,
            "carrier_intercept_attempts": 0,
        }
        for _ in range(int(max_decision_steps) + 5):
            action = _zero_action(env)
            env.step_async(action)
            _, reward, done, infos = env.step_wait()
            ep_return += float(reward[0])
            last_info = infos[0] if infos else {}
            detector_telemetry = _merge_detector_telemetry(detector_telemetry, _core_detector_telemetry(core))
            if bool(done.any()):
                ep_res = last_info.get("episode_result", last_info)
                return _episode_result_row(
                    blue_style=blue_style,
                    red_style=red_style,
                    map_name=map_name,
                    episode_index=episode_index,
                    episode_seed=episode_seed,
                    episode_result=dict(ep_res),
                    reward_return=ep_return,
                    extra_telemetry=detector_telemetry,
                )
        raise RuntimeError(
            f"episode did not terminate: blue={blue_style} red={red_style} map={map_name} seed={episode_seed}"
        )
    finally:
        env.close()


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("no rows to write")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _load_existing_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_key(row: dict[str, Any]) -> tuple[str, str, str, int]:
    return (
        str(row["blue_style"]),
        str(row["red_style"]),
        str(row["map"]),
        int(row["episode_index"]),
    )


def _expected_keys(args: argparse.Namespace) -> set[tuple[str, str, str, int]]:
    return {
        (str(blue), str(red), str(map_name), int(ep_i))
        for red in args.reds
        for map_name in args.maps
        for ep_i in range(int(args.episodes))
        for blue in args.blue_styles
    }


def _existing_complete_rows(
    rows: list[dict[str, Any]],
    expected: set[tuple[str, str, str, int]],
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, str, int], dict[str, Any]], list[dict[str, Any]]]:
    seen: dict[tuple[str, str, str, int], dict[str, Any]] = {}
    duplicates: list[dict[str, Any]] = []
    for row in rows:
        try:
            key = _row_key(row)
        except (KeyError, TypeError, ValueError):
            duplicates.append(row)
            continue
        if key not in expected:
            duplicates.append(row)
            continue
        if key in seen:
            duplicates.append(row)
            continue
        seen[key] = row
    ordered = sorted(
        seen.values(),
        key=lambda r: (
            str(r["red_style"]),
            str(r["map"]),
            int(r["episode_index"]),
            str(r["blue_style"]),
        ),
    )
    return ordered, seen, duplicates


def _init_episode_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        f.flush()
        os.fsync(f.fileno())


def _append_episode_row(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writerow({k: row.get(k, "") for k in ROW_FIELDS})
        f.flush()
        os.fsync(f.fileno())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Atomic-ish JSON write with Windows-friendly retries.

    Antivirus / Indexer / concurrent readers often hold ``path`` briefly on
    Windows, so ``Path.replace`` can raise ``PermissionError``. Progress
    writes must not abort a multi-hour matrix for that.
    """
    import time

    text = json.dumps(payload, indent=2) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    last_err: Exception | None = None
    for attempt in range(8):
        try:
            tmp.write_text(text, encoding="utf-8")
            try:
                tmp.replace(path)
            except PermissionError:
                # Fall back to in-place overwrite when replace is locked.
                path.write_text(text, encoding="utf-8")
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            return
        except (PermissionError, OSError) as exc:
            last_err = exc
            time.sleep(0.05 * (2**attempt))
    # Last resort: non-atomic overwrite so the episode loop can continue.
    try:
        path.write_text(text, encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 — surface both failures
        raise RuntimeError(f"failed to write {path}: {last_err}; fallback: {exc}") from exc


def _write_progress(
    path: Path,
    *,
    status: str,
    completed: int,
    expected: int,
    skipped_existing: int,
    duplicate_or_invalid_rows: int,
    last_row: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_episode_rows": int(completed),
        "expected_episode_rows": int(expected),
        "missing_episode_rows": int(max(0, expected - completed)),
        "skipped_existing_rows": int(skipped_existing),
        "duplicate_or_invalid_existing_rows": int(duplicate_or_invalid_rows),
    }
    if last_row is not None:
        payload["last_row"] = last_row
    if error is not None:
        payload["error"] = error
    _write_json(path, payload)


def _red_key(row: dict[str, Any]) -> str:
    return f"{row['red_style']}|{row['map']}"


def _analysis_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Treat red preset + map as the red column. The admissibility question is
    # whether any blue style is selectively best over the full red/map pool.
    out = []
    for row in rows:
        copied = dict(row)
        copied["red_style"] = _red_key(row)
        out.append(copied)
    return out


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(args.blue_styles) * len(args.reds) * len(args.maps) * int(args.episodes)
    started_at = datetime.now(timezone.utc).isoformat()
    episode_csv = out_dir / EPISODE_RESULTS_CSV
    expected_keys = _expected_keys(args)
    loaded_rows = _load_existing_rows(episode_csv)
    rows, completed_by_key, duplicate_rows = _existing_complete_rows(loaded_rows, expected_keys)
    if rows:
        _write_rows(episode_csv, rows)
    else:
        _init_episode_csv(episode_csv)
    count = len(rows)
    manifest_base = {
        "protocol": "scripted_blue_red_pool_admissibility",
        "status": "running",
        "started_at_utc": started_at,
        "blue_styles": list(args.blue_styles),
        "reds": list(args.reds),
        "maps": list(args.maps),
        "episodes_per_cell": int(args.episodes),
        "base_seed": int(args.base_seed),
        "matched_seed_contract": "episode_seed = f(red,map,episode_index), independent of blue_style",
        "max_decision_steps": int(args.max_decision_steps),
        "device": str(args.device),
        "expected_episode_rows": int(total),
        "artifacts": [EPISODE_RESULTS_CSV, POOL_REPORT_JSON, POOL_REPORT_TXT, PARTIAL_SUMMARY_JSON],
        "resume": {
            "loaded_existing_rows": int(len(loaded_rows)),
            "accepted_existing_rows": int(len(rows)),
            "duplicate_or_invalid_existing_rows": int(len(duplicate_rows)),
        },
    }
    _write_json(out_dir / RUN_MANIFEST_JSON, manifest_base)
    initial_status = "COMPLETED" if count == total else ("INTERRUPTED_RESUMABLE" if count > 0 else "RUNNING")
    _write_progress(
        out_dir / PARTIAL_SUMMARY_JSON,
        status=initial_status,
        completed=count,
        expected=total,
        skipped_existing=count,
        duplicate_or_invalid_rows=len(duplicate_rows),
        last_row=rows[-1] if rows else None,
    )

    try:
        for red_i, red_style in enumerate(args.reds):
            for map_i, map_name in enumerate(args.maps):
                for ep_i in range(int(args.episodes)):
                    seed = _episode_seed(int(args.base_seed), red_i, map_i, ep_i)
                    for blue_style in args.blue_styles:
                        key = (str(blue_style), str(red_style), str(map_name), int(ep_i))
                        if key in completed_by_key:
                            continue
                        row = _run_one_episode(
                            blue_style=str(blue_style),
                            red_style=str(red_style),
                            map_name=str(map_name),
                            episode_index=ep_i,
                            episode_seed=seed,
                            max_decision_steps=int(args.max_decision_steps),
                            device=str(args.device),
                        )
                        rows.append(row)
                        completed_by_key[key] = row
                        _append_episode_row(episode_csv, row)
                        count += 1
                        _write_progress(
                            out_dir / PARTIAL_SUMMARY_JSON,
                            status="RUNNING" if count < total else "COMPLETED",
                            completed=count,
                            expected=total,
                            skipped_existing=len(loaded_rows),
                            duplicate_or_invalid_rows=len(duplicate_rows),
                            last_row=row,
                        )
                        if int(args.progress_every) > 0 and count % int(args.progress_every) == 0:
                            print(f"[scripted-style matrix] {count}/{total} episodes", flush=True)
    except Exception as exc:
        _write_progress(
            out_dir / PARTIAL_SUMMARY_JSON,
            status="FAILED",
            completed=len(completed_by_key),
            expected=total,
            skipped_existing=len(loaded_rows),
            duplicate_or_invalid_rows=len(duplicate_rows),
            last_row=rows[-1] if rows else None,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise

    cells = cells_from_rows(_analysis_rows(rows))
    report = analyze_pool(
        cells,
        n_boot=int(args.n_boot),
        seed=int(args.analysis_seed),
        min_br_diversity=int(args.min_br_diversity),
    )
    (out_dir / POOL_REPORT_TXT).write_text(format_report(report) + "\n", encoding="utf-8")
    (out_dir / POOL_REPORT_JSON).write_text(json.dumps(asdict(report), indent=2) + "\n", encoding="utf-8")
    manifest = {
        **manifest_base,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "completed_episode_rows": int(count),
    }
    _write_json(out_dir / RUN_MANIFEST_JSON, manifest)

    print(format_report(report))
    print(f"\nArtifacts in: {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
