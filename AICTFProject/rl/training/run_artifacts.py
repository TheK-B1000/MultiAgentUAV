"""Run-artifact filesystem hygiene for the PPO trainer.

Pure file/run housekeeping helpers extracted from :mod:`rl.train_ppo`:

* Git metadata probing (``_find_git_root`` / ``_git_metadata``).
* ``run_config.json`` sidecar writer (``write_run_config_json``).
* Metrics CSV emptiness check + rotation (``_metrics_csv_nonempty`` /
  ``_rotate_csv_aside``).
* Per-run-tag lockfile (``_RunLock``, ``_acquire_run_lock``, ...).

No PPO math, no torch, no env construction lives here. Anything that touches
a ``PPOConfig`` only reads attribute names; the module never mutates training
hyperparameters.
"""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from rl.config.ppo_config import PPOConfig
from rl.custom_ppo.telemetry.schemas import (
    PERFORMANCE_METRICS_SCHEMA_VERSION,
    TRAINING_EVENTS_SCHEMA_VERSION,
    TRAINING_METRICS_SCHEMA_VERSION,
    coerce_telemetry_mode,
)

# Module dir is ``AICTFProject/rl/training``; walking upward 8 levels still
# reaches the repo root that contains ``.git`` from anywhere in this tree.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_git_root() -> str:
    """Walk upward from this file to find a directory containing ``.git``; else ``cwd``."""
    p = os.path.abspath(_SCRIPT_DIR)
    for _ in range(8):
        if os.path.isdir(os.path.join(p, ".git")):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return os.getcwd()


def _git_metadata() -> dict[str, Optional[str]]:
    """Best-effort ``git rev-parse`` / ``git describe`` from the repo root."""
    root = _find_git_root()
    meta: dict[str, Optional[str]] = {
        "git_sha": None,
        "git_describe": None,
        "git_root": root,
        "git_error": None,
    }
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if sha.returncode == 0 and sha.stdout.strip():
            meta["git_sha"] = sha.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        meta["git_error"] = str(exc)
    try:
        desc = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if desc.returncode == 0 and desc.stdout.strip():
            meta["git_describe"] = desc.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        meta["git_error"] = meta["git_error"] or str(exc)
    return meta


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    return str(obj)


def _artifact_base_dir(cfg: PPOConfig) -> str:
    base_dir = cfg.checkpoint_dir
    if getattr(cfg, "metrics_csv_path", None):
        d = os.path.dirname(str(cfg.metrics_csv_path))
        if d:
            base_dir = d
    os.makedirs(base_dir, exist_ok=True)
    return base_dir


def _run_config_json_path(cfg: PPOConfig) -> str:
    return os.path.join(_artifact_base_dir(cfg), f"{cfg.run_tag}_run_config.json")


def _require_run_identity(run_identity, *, writer: str):
    from rl.ruleset_identity import RunIdentity, RunIdentityError

    if run_identity is None or not isinstance(run_identity, RunIdentity):
        raise RunIdentityError(
            f"{writer} requires the run's resolved RunIdentity; "
            "refusing to write an unstamped artifact."
        )
    return run_identity


def write_run_config_json(cfg: PPOConfig, argv: Optional[list[str]] = None,
                          *, run_identity=None) -> str:
    """Write reproducibility sidecar JSON next to metrics CSV (or under ``checkpoint_dir``).

    ``run_identity`` is the run's single resolved :class:`RunIdentity`, taken from
    the live environment. It is MANDATORY for a formal run: without it the
    sidecar cannot be validated as part of the artifact bundle, and a writer
    that rebuilt identity from ``cfg`` defaults could disagree with the
    environment that actually ran.
    """
    identity = _require_run_identity(run_identity, writer="write_run_config_json")
    path = _run_config_json_path(cfg)
    argv_list = list(sys.argv) if argv is None else list(argv)
    git_meta = _git_metadata()
    payload: dict[str, Any] = {
        "utc_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "argv": argv_list,
        "working_directory": os.getcwd(),
        "python_executable": sys.executable,
        **git_meta,
        "run_tag": str(cfg.run_tag),
        "checkpoint_dir": str(cfg.checkpoint_dir),
        "total_timesteps": int(cfg.total_timesteps),
        "metrics_csv_path": cfg.metrics_csv_path,
        "episode_csv_path": cfg.episode_csv_path,
        "strategy_experience_csv_path": cfg.strategy_experience_csv_path,
        "load_path": cfg.load_path,
        "cli_preset": getattr(cfg, "cli_preset", None),
        "telemetry": {
            "mode": str(
                coerce_telemetry_mode(
                    getattr(
                        cfg,
                        "training_telemetry_mode",
                        getattr(
                            cfg,
                            "telemetry_mode",
                            "full"
                            if getattr(cfg, "training_events_jsonl_path", getattr(cfg, "telemetry_events_jsonl_path", ""))
                            else "off",
                        ),
                    )
                )
            ),
            "training_metrics_schema_version": TRAINING_METRICS_SCHEMA_VERSION,
            "training_events_schema_version": TRAINING_EVENTS_SCHEMA_VERSION,
            "performance_metrics_schema_version": PERFORMANCE_METRICS_SCHEMA_VERSION,
            "events_jsonl_path": str(
                getattr(cfg, "training_events_jsonl_path", getattr(cfg, "telemetry_events_jsonl_path", ""))
                or ""
            ),
            "artifact_dir": str(
                getattr(cfg, "telemetry_artifact_dir", "")
                or getattr(cfg, "checkpoint_dir", "")
                or ""
            ),
        },
        "resolved_ppo_config": _json_safe(asdict(cfg)),
    }
    # Single shared passport. NOT rebuilt here -- stamped from the object the
    # orchestrator resolved from the live environment, so this sidecar cannot
    # disagree with the game that actually ran.
    from rl.ruleset_identity import stamp_json_artifact

    stamp_json_artifact(payload, identity)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


def write_training_manifest_json(
    cfg: PPOConfig,
    *,
    run_identity=None,
    extra: Optional[dict[str, Any]] = None,
) -> str:
    """Write ``training_manifest.json`` stamped with the run's frozen identity."""
    from rl.ruleset_identity import stamp_json_artifact

    identity = _require_run_identity(run_identity, writer="write_training_manifest_json")
    path = os.path.join(_artifact_base_dir(cfg), "training_manifest.json")
    payload: dict[str, Any] = {
        "utc_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_tag": str(cfg.run_tag),
        "seed": int(getattr(cfg, "seed", 0) or 0),
        "total_timesteps": int(cfg.total_timesteps),
        "checkpoint_dir": str(cfg.checkpoint_dir),
        "episode_csv_path": cfg.episode_csv_path,
        "metrics_csv_path": cfg.metrics_csv_path,
        "cli_preset": getattr(cfg, "cli_preset", None),
        **_git_metadata(),
    }
    if extra:
        payload.update(extra)
    stamp_json_artifact(payload, identity)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


def write_evaluation_manifest_json(
    path: str,
    *,
    run_identity=None,
    evaluation_run_id: Optional[str] = None,
    source_training_run_id: Optional[str] = None,
    source_checkpoint_fingerprint: Optional[str] = None,
    source_checkpoint_ruleset_fingerprint: Optional[str] = None,
    extra: Optional[dict[str, Any]] = None,
) -> str:
    """Write ``evaluation_manifest.json`` stamped from a resolved identity."""
    from rl.ruleset_identity import stamp_json_artifact

    identity = _require_run_identity(run_identity, writer="write_evaluation_manifest_json")
    payload: dict[str, Any] = {
        "utc_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "evaluation_run_id": evaluation_run_id or identity.run_id,
        "source_training_run_id": source_training_run_id or identity.run_id,
        "source_checkpoint_fingerprint": source_checkpoint_fingerprint or "",
        "source_checkpoint_ruleset_fingerprint": (
            source_checkpoint_ruleset_fingerprint or identity.ruleset_fingerprint
        ),
    }
    if extra:
        payload.update(extra)
    stamp_json_artifact(payload, identity)
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


def write_result_summary_json(
    path: str,
    summary: dict[str, Any],
    *,
    run_identity=None,
    source_rows: Optional[list[dict[str, Any]]] = None,
) -> str:
    """Stamp and write a result summary; validate against source rows when given."""
    from rl.ruleset_identity import (
        RunIdentityError,
        stamp_json_artifact,
        validate_bundle,
    )

    identity = _require_run_identity(run_identity, writer="write_result_summary_json")
    payload = dict(summary)
    stamp_json_artifact(payload, identity)
    if source_rows is not None:
        if not source_rows:
            raise RunIdentityError(
                "write_result_summary_json refuses an empty source-row bundle "
                "for a formal result summary."
            )
        validate_bundle(
            {"result_summary.json": payload},
            {"episode_rows.csv": list(source_rows)},
            require_formal=bool(identity.formal_result_eligible),
        )
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


def write_startup_formal_artifacts(
    cfg: PPOConfig,
    *,
    run_identity=None,
    argv: Optional[list[str]] = None,
) -> dict[str, str]:
    """Write run_config + training_manifest from one frozen identity.

    Both must succeed before the first rollout step.
    """
    rc = write_run_config_json(cfg, argv=argv, run_identity=run_identity)
    tm = write_training_manifest_json(cfg, run_identity=run_identity)
    return {"run_config": rc, "training_manifest": tm}


def _metrics_csv_nonempty(path: Optional[str]) -> bool:
    return bool(path and os.path.isfile(path) and os.path.getsize(path) > 0)


def _rotate_csv_aside(path: Optional[str], *, label: str) -> None:
    if not _metrics_csv_nonempty(path):
        return
    assert path is not None
    bak = f"{path}.bak.{int(time.time())}"
    os.replace(path, bak)
    print(f"[PPO] Rotated existing {label} CSV aside: {bak!r} (--fresh-metrics-csv).")


@dataclass
class _RunLock:
    path: str
    token: str
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("token") != self.token:
                return
            os.unlink(self.path)
            print(f"[PPO] Run lock released: {self.path}")
        except FileNotFoundError:
            return
        except Exception as exc:
            print(f"[PPO] WARNING: failed to release run lock {self.path!r}: {exc}")


def _pid_is_running(pid: int) -> bool:
    pid = int(pid)
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            process_query_limited_information = 0x1000
            still_active = 259
            handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            return bool(ok) and int(exit_code.value) == still_active
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_run_lock(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _acquire_run_lock(cfg: PPOConfig) -> _RunLock:
    """Prevent duplicate trainers from sharing checkpoint/CSV artifacts for one run tag."""
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    lock_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}.run.lock")
    token = f"{os.getpid()}-{time.time_ns()}"
    payload = {
        "pid": os.getpid(),
        "token": token,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_tag": str(cfg.run_tag),
        "argv": sys.argv,
        "metrics_csv_path": str(getattr(cfg, "metrics_csv_path", "") or ""),
        "episode_csv_path": str(getattr(cfg, "episode_csv_path", "") or ""),
        "strategy_experience_csv_path": str(getattr(cfg, "strategy_experience_csv_path", "") or ""),
    }
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            existing = _read_run_lock(lock_path)
            pid = int(existing.get("pid", 0) or 0)
            if pid > 0 and _pid_is_running(pid):
                raise RuntimeError(
                    f"Active PPO run lock exists for run_tag={cfg.run_tag!r}: {lock_path!r} "
                    f"(pid={pid}). Stop that trainer or use a different --run-tag before starting another run."
                ) from exc
            stale_path = f"{lock_path}.stale.{int(time.time())}"
            os.replace(lock_path, stale_path)
            print(f"[PPO] Rotated stale run lock aside: {stale_path!r}")
            continue

        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        setattr(cfg, "run_id", token)
        setattr(cfg, "run_pid", os.getpid())
        lock = _RunLock(lock_path, token)
        atexit.register(lock.release)
        print(f"[PPO] Run lock acquired: {lock_path}")
        return lock


__all__ = [
    "_RunLock",
    "_acquire_run_lock",
    "_find_git_root",
    "_git_metadata",
    "_json_safe",
    "_metrics_csv_nonempty",
    "_pid_is_running",
    "_read_run_lock",
    "_rotate_csv_aside",
    "_run_config_json_path",
    "write_evaluation_manifest_json",
    "write_result_summary_json",
    "write_run_config_json",
    "write_startup_formal_artifacts",
    "write_training_manifest_json",
]
