from __future__ import annotations

from pathlib import Path
from typing import Any
import zipfile

import torch

from .errors import CheckpointArchiveError, CheckpointNotFoundError


def _torch_load_checkpoint(path: str | Path, *, map_location: str | torch.device) -> Any:
    p = Path(path)
    if not p.exists():
        raise CheckpointNotFoundError("Checkpoint file not found", checkpoint_path=str(p))
    try:
        return torch.load(str(p), map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(str(p), map_location=map_location)
    except Exception as weights_exc:
        # Legacy trainer checkpoints store Python dictionaries and may require the
        # historical unpickling path. Keep the fallback local to checkpoint archives.
        try:
            return torch.load(str(p), map_location=map_location, weights_only=False)
        except Exception as exc:
            raise CheckpointArchiveError("Unable to read checkpoint archive", checkpoint_path=str(p), observed=exc) from weights_exc


def inspect_archive_members(path: str | Path) -> tuple[str, ...]:
    p = Path(path)
    if not zipfile.is_zipfile(p):
        return ()
    try:
        with zipfile.ZipFile(p, "r") as zf:
            names = tuple(zf.namelist())
    except zipfile.BadZipFile as exc:
        raise CheckpointArchiveError("Corrupted checkpoint archive", checkpoint_path=str(p)) from exc
    critical = {"data.pkl", "model_state_dict"}
    basenames = [Path(n).name for n in names]
    duplicates = sorted({n for n in basenames if basenames.count(n) > 1 and n in critical})
    if duplicates:
        raise CheckpointArchiveError("Duplicate critical checkpoint archive members", checkpoint_path=str(p), observed=duplicates)
    return names


def read_checkpoint_payload(path: str | Path, *, map_location: str | torch.device) -> dict[str, Any]:
    inspect_archive_members(path)
    payload = _torch_load_checkpoint(path, map_location=map_location)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise CheckpointArchiveError("Not a custom PPO checkpoint", checkpoint_path=str(path))
    return payload
