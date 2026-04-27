from __future__ import annotations

import os
from typing import Optional


def _try_paths(*candidates: str) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def _resolve_snapshot_path(path: str) -> Optional[str]:
    if not path:
        return None
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [path]
    if not path.endswith(".zip"):
        candidates.append(path + ".zip")
    if not os.path.isabs(path):
        local = os.path.join(project_dir, path)
        cwd_local = os.path.join(os.getcwd(), path)
        candidates.extend([local, cwd_local])
        if not local.endswith(".zip"):
            candidates.append(local + ".zip")
        if not cwd_local.endswith(".zip"):
            candidates.append(cwd_local + ".zip")
    return _try_paths(*candidates)


__all__ = ["_resolve_snapshot_path", "_try_paths"]
