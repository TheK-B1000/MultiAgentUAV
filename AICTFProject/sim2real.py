"""
Sim-to-real pipeline (paper-aligned): log real-world speed and pickup times, then suggest sim adjustments.

Usage:
  - From real hardware or logs: call log_speed_pickup() or append rows to the CSV.
  - From sim: attach Sim2RealStepLogger to BatchedCTFCore.sim2real_logger to log per-step speed (and optional events).
  - Run suggest_sim_params_from_log() to get a config snippet to tune game_field_gpu / game_manager.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Default path for real-world measurements (speed in m/s or cells/s, pickup_time in seconds).
DEFAULT_LOG_PATH = "sim2real_log.csv"
CSV_HEADER = ["episode_id", "agent_id", "speed_observed", "pickup_time_observed", "source"]

# Step-level log (from sim or real) for speed and optional event type.
DEFAULT_STEP_LOG_PATH = "sim2real_steps.csv"
STEP_CSV_HEADER = ["episode_id", "step", "agent_id", "game_time_sec", "speed_cps", "event_type", "source"]


class Sim2RealStepLogger:
    """Logs per-step speed (and optional event_type) for sim-to-real. Attach to BatchedCTFCore.sim2real_logger."""

    def __init__(self, log_path: str | Path = DEFAULT_STEP_LOG_PATH, source: str = "sim"):
        self.log_path = Path(log_path)
        self.source = source
        self._written_header = False

    def log_step(
        self,
        episode_id: int,
        step: int,
        agent_id: int,
        game_time_sec: float,
        speed_cps: float,
        event_type: str = "",
    ) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a", newline="") as f:
            w = csv.writer(f)
            if not self._written_header:
                w.writerow(STEP_CSV_HEADER)
                self._written_header = True
            w.writerow([episode_id, step, agent_id, game_time_sec, speed_cps, event_type or "", self.source])


def log_step(
    episode_id: int,
    step: int,
    agent_id: int,
    game_time_sec: float,
    speed_cps: float,
    event_type: str = "",
    log_path: str | Path = DEFAULT_STEP_LOG_PATH,
    source: str = "sim",
) -> None:
    """Append one step row to the step log (for sim or real)."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(STEP_CSV_HEADER)
        w.writerow([episode_id, step, agent_id, game_time_sec, speed_cps, event_type or "", source])


def log_speed_pickup(
    episode_id: str | int,
    agent_id: int,
    speed_observed: float,
    pickup_time_observed: float,
    log_path: str | Path = DEFAULT_LOG_PATH,
    source: str = "real",
) -> None:
    """Append one row: measured speed and pickup-to-capture time for sim-to-real tuning."""
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(CSV_HEADER)
        w.writerow([episode_id, agent_id, speed_observed, pickup_time_observed, source])


def read_log(log_path: str | Path = DEFAULT_LOG_PATH) -> List[Dict[str, Any]]:
    """Read CSV of speed/pickup measurements into list of dicts."""
    path = Path(log_path)
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                row["speed_observed"] = float(row.get("speed_observed", 0))
                row["pickup_time_observed"] = float(row.get("pickup_time_observed", 0))
            except (ValueError, TypeError):
                continue
            rows.append(row)
    return rows


def read_step_log(step_log_path: str | Path = DEFAULT_STEP_LOG_PATH) -> List[Dict[str, Any]]:
    """Read step-level CSV into list of dicts (episode_id, step, agent_id, game_time_sec, speed_cps, event_type, source)."""
    path = Path(step_log_path)
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                row["game_time_sec"] = float(row.get("game_time_sec", 0))
                row["speed_cps"] = float(row.get("speed_cps", 0))
            except (ValueError, TypeError):
                continue
            rows.append(row)
    return rows


def suggest_sim_params_from_log(
    log_path: str | Path = DEFAULT_LOG_PATH,
    step_log_path: Optional[str | Path] = None,
    speed_scale_cells_per_m: float = 1.0,
) -> Dict[str, Any]:
    """
    Compute suggested sim parameters from real-world log (paper: measure real speed and pickup times, adjust sim).

    If step_log_path is provided (e.g. sim2real_steps.csv from Sim2RealStepLogger), speeds are aggregated from
    step-level log; otherwise from the summary log (speed_observed / pickup_time_observed).

    Returns a dict suitable for GPUFieldConfig or game_manager overrides, e.g.:
      - max_speed_cps: from median observed speed (converted to cells/step if you provide dt and scale).
      - mine_pickup_radius_cells / timing: optional if you log pickup duration and want to match it.
    """
    import statistics

    out: Dict[str, Any] = {}
    speeds: List[float] = []
    pickups: List[float] = []

    if step_log_path:
        step_rows = read_step_log(step_log_path)
        speeds = [r["speed_cps"] for r in step_rows if r.get("speed_cps") is not None]
    rows = read_log(log_path)
    if not step_log_path:
        speeds = [r["speed_observed"] for r in rows if r.get("speed_observed") is not None]
    pickups = [r["pickup_time_observed"] for r in rows if r.get("pickup_time_observed") is not None and r.get("pickup_time_observed", 0) > 0]

    if speeds:
        median_speed = statistics.median(speeds)
        out["max_speed_cps"] = median_speed * speed_scale_cells_per_m
        out["_log_median_speed_observed"] = median_speed
    if pickups:
        median_pickup = statistics.median(pickups)
        out["_log_median_pickup_time_observed_s"] = median_pickup
        out["_suggested_pickup_duration_s"] = median_pickup
    return out


def write_config_snippet(suggested: Dict[str, Any], out_path: str | Path = "sim2real_suggested_config.txt") -> None:
    """Write a human-readable config snippet and Python overrides for GPUFieldConfig."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Sim-to-real suggested overrides (from sim2real_log.csv)",
        "# Paste into GPUFieldConfig(..., **overrides) or game_manager.",
        "",
    ]
    for k, v in suggested.items():
        if k.startswith("_"):
            lines.append(f"# {k}: {v}")
        else:
            lines.append(f"{k} = {v!r}")
    lines.append("")
    lines.append("# Example: gpu_cfg = GPUFieldConfig(" + ", ".join(f"{k}={v!r}" for k, v in suggested.items() if not k.startswith("_")) + ")")
    path.write_text("\n".join(lines), encoding="utf-8")
