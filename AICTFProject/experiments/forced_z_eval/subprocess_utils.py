"""Subprocess helpers for eval wrappers."""
from __future__ import annotations

import os
import signal
import subprocess
import time
from collections.abc import Sequence


def run_with_process_tree_timeout(
    cmd: Sequence[str],
    *,
    cwd: str,
    timeout_seconds: int | None,
    terminate_grace_seconds: int = 30,
) -> subprocess.CompletedProcess[str]:
    """Run a child process and tear down the whole process tree on timeout."""
    popen_kwargs = {
        "cwd": cwd,
        "text": True,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_kwargs["preexec_fn"] = os.setsid

    proc = subprocess.Popen(list(cmd), **popen_kwargs)
    try:
        returncode = proc.wait(timeout=timeout_seconds)
        return subprocess.CompletedProcess(list(cmd), returncode)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_tree(proc, grace_seconds=terminate_grace_seconds)
        raise subprocess.TimeoutExpired(list(cmd), timeout_seconds) from exc


def _terminate_process_tree(proc: subprocess.Popen[str], *, grace_seconds: int) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], check=False)
        deadline = time.time() + float(grace_seconds)
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.25)
        if proc.poll() is None:
            proc.kill()
        proc.wait()
        return

    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


__all__ = ["run_with_process_tree_timeout"]
