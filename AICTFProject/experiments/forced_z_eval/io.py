"""Load and save canonical forced-z episode tables."""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from experiments.forced_z_eval.protocol import EPISODE_RESULTS_CSV, RUN_MANIFEST_JSON, ForcedZProtocol

CellEpisodes = dict[tuple[str, int, str], list[dict[str, Any]]]

_META_COLS = ("checkpoint", "opponent", "latent_z", "map", "episode_index", "cell_seed", "episode_seed")


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write JSON through a same-directory temp file, then atomically replace."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)


def write_manifest(
    run_dir: str | Path,
    *,
    protocol: ForcedZProtocol,
    status: str,
    started_at_utc: str | None = None,
    completed_conditions: list[dict[str, Any]] | None = None,
    episode_count: int = 0,
    error: str | None = None,
    extra_manifest: dict[str, Any] | None = None,
) -> Path:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    manifest = protocol.to_manifest()
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest.update(
        {
            "status": status,
            "started_at_utc": started_at_utc,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "completed_conditions": completed_conditions or [],
            "completed_condition_count": len(completed_conditions or []),
            "episode_count": int(episode_count),
        }
    )
    if error:
        manifest["error"] = str(error)
    manifest_path = path / RUN_MANIFEST_JSON
    atomic_write_json(manifest_path, manifest)
    return manifest_path


def append_episode_rows(
    run_dir: str | Path,
    *,
    protocol: ForcedZProtocol,
    cells: CellEpisodes,
) -> Path:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    rows = episodes_to_rows(cells, protocol=protocol)
    csv_path = path / EPISODE_RESULTS_CSV
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = list(_META_COLS)
    existing_header: list[str] | None = None
    if csv_path.exists() and csv_path.stat().st_size > 0:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
    if existing_header is not None:
        fieldnames = list(existing_header)
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    raise ValueError(f"Cannot append row with new column {key!r} to {csv_path}")
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if existing_header is None:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    return csv_path


def _coerce_row(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in {"opponent", "map", "episode_start_phase", "fixed_latent_id"}:
            out[key] = value
            continue
        if key in {"latent_z", "episode_index", "cell_seed", "episode_seed", "success", "blue_score", "red_score", "steps", "collision_free", "fixed_latent_id"}:
            try:
                out[key] = int(float(value)) if value != "" else 0
            except ValueError:
                out[key] = value
            continue
        try:
            out[key] = float(value) if value != "" else float("nan")
        except ValueError:
            out[key] = value
    if "latent_z" in out:
        out["latent_z"] = int(out["latent_z"])
    return out


def episodes_to_rows(cells: CellEpisodes, *, protocol: ForcedZProtocol) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for opp_idx, opponent in enumerate(protocol.opponents):
        for map_idx, map_name in enumerate(protocol.maps):
            cell_seed = protocol.cell_seed(opp_idx, map_idx)
            for z in protocol.latents:
                for ep_idx, ep in enumerate(cells.get((opponent, z, map_name), [])):
                    rows.append(
                        {
                            "checkpoint": protocol.checkpoint,
                            "opponent": opponent,
                            "latent_z": int(z),
                            "map": map_name,
                            "episode_index": int(ep_idx),
                            "cell_seed": int(cell_seed),
                            "episode_seed": int(protocol.episode_seed(cell_seed, ep_idx)),
                            **ep,
                        }
                    )
    return rows


def rows_to_cells(rows: list[dict[str, Any]]) -> CellEpisodes:
    cells: CellEpisodes = {}
    for row in rows:
        opponent = str(row["opponent"])
        z = int(row["latent_z"])
        map_name = str(row["map"])
        ep = {k: v for k, v in row.items() if k not in _META_COLS and k != "checkpoint"}
        cells.setdefault((opponent, z, map_name), []).append(ep)
    for key in cells:
        cells[key].sort(key=lambda ep: int(ep.get("episode_index", 0)) if "episode_index" in ep else 0)
    return cells


def write_run_artifacts(
    run_dir: str | Path,
    *,
    protocol: ForcedZProtocol,
    cells: CellEpisodes,
    extra_manifest: dict[str, Any] | None = None,
) -> Path:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    rows = episodes_to_rows(cells, protocol=protocol)
    csv_path = path / EPISODE_RESULTS_CSV
    if csv_path.exists():
        csv_path.unlink()
    append_episode_rows(path, protocol=protocol, cells=cells)
    manifest = protocol.to_manifest()
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest.setdefault("status", "completed")
    manifest["episode_count"] = len(rows)
    manifest_path = path / RUN_MANIFEST_JSON
    atomic_write_json(manifest_path, manifest)
    return csv_path


def load_episode_results(run_dir: str | Path) -> tuple[ForcedZProtocol, CellEpisodes]:
    path = Path(run_dir)
    manifest_path = path / RUN_MANIFEST_JSON
    csv_path = path / EPISODE_RESULTS_CSV
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing episode table: {csv_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    protocol = ForcedZProtocol(
        checkpoint=str(manifest["checkpoint"]),
        opponents=tuple(manifest.get("opponents", [])),
        maps=tuple(manifest.get("maps", [])),
        latents=tuple(int(z) for z in manifest.get("latents", [])),
        episodes_per_cell=int(manifest.get("episodes_per_cell", 0)),
        base_seed=int(manifest.get("base_seed", 42)),
        deterministic_actions=bool(manifest.get("deterministic_actions", True)),
        max_decision_steps=int(manifest.get("max_decision_steps", 400)),
        device=str(manifest.get("device", "cuda")),
        collect_behavior_mean=bool(manifest.get("collect_behavior_mean", True)),
        progress_every=int(manifest.get("progress_every", 0)),
    )
    with csv_path.open(newline="", encoding="utf-8") as f:
        rows = [_coerce_row(row) for row in csv.DictReader(f)]
    return protocol, rows_to_cells(rows)


def load_episode_results_csv(csv_path: str | Path, *, protocol: ForcedZProtocol) -> CellEpisodes:
    with Path(csv_path).open(newline="", encoding="utf-8") as f:
        rows = [_coerce_row(row) for row in csv.DictReader(f)]
    return rows_to_cells(rows)


__all__ = [
    "CellEpisodes",
    "append_episode_rows",
    "atomic_write_json",
    "episodes_to_rows",
    "load_episode_results",
    "load_episode_results_csv",
    "rows_to_cells",
    "write_manifest",
    "write_run_artifacts",
]
