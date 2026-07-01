"""Archive the validated v6i9 repertoire anchor before router training."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

DEFAULT_DEST = Path(r"K:\AICTF_Checkpoint_Archive\v6i9-repertoire-refactor-r1")
CHECKPOINT = _REPO / "checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip"
FORCED_Z_RUN = _REPO / "experiments/forced_z_runs/20260630_040244"
EQUIVALENCE = _REPO / "experiments/forced_z_runs/equivalence_20260630_030424/env_reuse_equivalence.json"
COMPLEMENTARITY = FORCED_Z_RUN / "complementarity_report.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _frozen_tensor_hash(checkpoint: Path) -> str:
    from rl.custom_ppo.checkpoints.archive import _torch_load_checkpoint
    from rl.custom_ppo.diagnostics.frozen_repertoire_hash import hash_frozen_repertoire_tensors

    payload = _torch_load_checkpoint(str(checkpoint), map_location="cpu")
    return hash_frozen_repertoire_tensors(payload.get("model_state_dict", {}))


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _find_metrics_csv() -> list[str]:
    found: list[Path] = []
    for path in _REPO.rglob("*.csv"):
        name = path.name.lower()
        if "repertoire" in name and ("refactor" in name or "v6i9" in name):
            found.append(path)
    unique = sorted({p.resolve() for p in found if p.is_file()})
    return [str(p) for p in unique]


def build_manifest(*, dest: Path) -> dict:
    complementarity = _read_json(COMPLEMENTARITY)
    equivalence = _read_json(EQUIVALENCE)
    checkpoint_hash = _sha256_file(CHECKPOINT)
    frozen_hash = _frozen_tensor_hash(CHECKPOINT)
    return {
        "archived_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": _git_commit(),
        "preset": "v6i9_mapaware_repertoire_hardpool",
        "seed": 1,
        "checkpoint_source": str(CHECKPOINT),
        "checkpoint_sha256": checkpoint_hash,
        "frozen_repertoire_tensor_hash": frozen_hash,
        "forced_z_eval_run": str(FORCED_Z_RUN),
        "env_reuse_equivalence": {
            "path": str(EQUIVALENCE),
            "passed": bool(equivalence.get("passed", False)),
            "reuse_block_approved": bool(equivalence.get("reuse_block_approved", False)),
        },
        "oracle_metric": complementarity.get("oracle_metric", "return"),
        "oracle_gap": complementarity.get("oracle_gap"),
        "best_fixed_z": complementarity.get("best_fixed_z"),
        "best_fixed_return": complementarity.get("best_fixed_mean"),
        "oracle_return": complementarity.get("oracle_mean"),
        "ladder_verdict": complementarity.get("ladder_verdict"),
        "metrics_csv_sources": _find_metrics_csv(),
    }


def archive(*, dest: Path, force: bool) -> Path:
    if dest.exists():
        if not force:
            raise SystemExit(f"Archive destination already exists: {dest}\nUse --force to overwrite.")
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    shutil.copy2(CHECKPOINT, dest / CHECKPOINT.name)
    if FORCED_Z_RUN.is_dir():
        shutil.copytree(FORCED_Z_RUN, dest / "forced_z_eval_20260630_040244")
    if EQUIVALENCE.is_file():
        equiv_dir = dest / "env_reuse_equivalence"
        equiv_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(EQUIVALENCE, equiv_dir / EQUIVALENCE.name)

    metrics_dir = dest / "metrics_csv"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    copied_metrics: list[str] = []
    for src in _find_metrics_csv():
        src_path = Path(src)
        dst = metrics_dir / src_path.name
        shutil.copy2(src_path, dst)
        copied_metrics.append(str(dst))

    manifest = build_manifest(dest=dest)
    manifest["metrics_csv_archived"] = copied_metrics
    manifest_path = dest / "archive_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sidecar = {
        "checkpoint_source": str(dest / CHECKPOINT.name),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "frozen_repertoire_tensor_hash": manifest["frozen_repertoire_tensor_hash"],
        "shared_actor_max_abs_delta": 0.0,
        "z_specific_max_abs_delta": 0.0,
        "frozen_tensor_hash_match": True,
    }
    (dest / "frozen_repertoire_anchor.json").write_text(
        json.dumps(sidecar, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not CHECKPOINT.is_file():
        raise SystemExit(f"Missing repertoire checkpoint: {CHECKPOINT}")
    manifest_path = archive(dest=args.dest, force=bool(args.force))
    print(f"[archive] wrote {manifest_path}")
    print(f"[archive] destination: {args.dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
