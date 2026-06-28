from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def _run(cmd: list[str], cwd: Path) -> dict[str, Any]:
    try:
        p = subprocess.run(cmd, cwd=str(cwd), text=True, capture_output=True, timeout=30)
        return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except Exception as exc:
        return {"cmd": cmd, "error": repr(exc)}


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint

    py = sys.executable
    info: dict[str, Any] = {
        "label": args.label,
        "root": str(root),
        "python_executable": py,
        "platform": platform.platform(),
        "git_head": _run(["git", "-c", "safe.directory=K:/MultiAgentUAV", "rev-parse", "HEAD"], root),
        "git_status_porcelain": _run(["git", "-c", "safe.directory=K:/MultiAgentUAV", "status", "--porcelain"], root),
        "python_version_cmd": _run([py, "--version"], root),
        "uv_version": _run(["uv", "--version"], root),
        "uv_python_version": _run(["uv", "run", "python", "--version"], root),
        "uv_python_executable": _run(["uv", "run", "python", "-c", "import sys; print(sys.executable)"], root),
        "uv_torch": _run(["uv", "run", "python", "-c", "import torch; print(torch.__version__, torch.version.cuda)"], root),
        "uv_numpy": _run(["uv", "run", "python", "-c", "import numpy; print(numpy.__version__)"], root),
        "system_torch": _run([py, "-c", "import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no_cuda'); print(torch.backends.cuda.matmul.allow_tf32); print(torch.backends.cudnn.allow_tf32); print(torch.are_deterministic_algorithms_enabled())"], root),
        "system_numpy": _run([py, "-c", "import numpy; print(numpy.__version__)"], root),
        "uv_lock_sha256": _sha256(root / "uv.lock"),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _sha256(checkpoint),
        "environment_variables": {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith(("CUDA", "CUBLAS", "CUDNN", "PYTORCH", "OMP", "MKL", "UV", "PYTHON"))},
        "benchmark_config": {
            "checkpoint": str(checkpoint),
            "map": "map_b_split_lane",
            "opponent": "OP9",
            "env_counts": [16, 64, 256],
            "telemetry_modes": ["off", "basic", "full"],
            "warmup_rollouts": 2,
            "measured_rollouts": 10,
            "rollout_length": 64,
            "device": "cuda",
            "seed": 42,
            "agents": 2,
            "ppo_epochs": 1,
            "batch_size_rule": "max(1024, env_count * n_steps)",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
