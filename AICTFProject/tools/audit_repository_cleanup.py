"""Phase 11 repository cleanup inventory tool.

Walks the project tree, classifies every file, and emits structured CSV/JSON
outputs under artifacts/cleanup_audit/.

Usage:
    uv run python tools/audit_repository_cleanup.py
    uv run python tools/audit_repository_cleanup.py --project-root . --output-dir artifacts/cleanup_audit
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SKIP_DIRS = {
    ".venv", ".git", "__pycache__", ".mypy_cache", ".ruff_cache",
    ".pytest_cache", "node_modules", ".idea",
}

CHECKPOINT_EXTS = {".zip", ".pth", ".pt", ".ckpt"}
LOG_EXTS = {".log", ".err", ".out"}
CSV_EXTS = {".csv", ".tsv"}
JSON_EXTS = {".json", ".jsonl", ".json5"}
PYTHON_EXTS = {".py", ".pyi"}
DOC_EXTS = {".md", ".rst", ".txt"}
ARTIFACT_PATTERNS = re.compile(
    r"(closeout|phase\d+|cleanup_audit|refactor_audit|proof_ladder|v6i[0-9]+_map)"
)

# Files whose deletion is low-risk GENERATED
GENERATED_PATTERNS = [
    "__pycache__", ".pyc", ".pyo", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".coverage", "htmlcov", ".tmp", ".partial",
    ".bak.", ".lock",
]

# Phase 10 intermediate directories — candidates for DELETE
PHASE10_INTERMEDIATES = {
    "phase10_equivalence_aggregation_gates",
    "phase10_equivalence_artifact_writer",
    "phase10_equivalence_episode_matched_seed",
    "phase10_equivalence_obstacle_probes",
    "phase10_equivalence_config_loading_preflight",
    "phase10_baseline",
}

# Phase 6.1 intermediate directories — candidates for DELETE
PHASE6_INTERMEDIATES = {
    "phase6_1_perf_investigation",
}

# Canonical closeout directories — always KEEP_GOLDEN
GOLDEN_DIRS = {
    "phase5_closeout", "phase6_1_closeout", "phase7_closeout",
    "phase8_closeout", "phase9_closeout", "phase10_closeout",
    "phase11_closeout", "refactor_audit", "cleanup_audit",
}

# Proof ladder directories — obsolete benchmark attempts
PROOF_LADDER_PATTERN = re.compile(r"proof_ladder_\d{8}_\d{6}")

# v6i9 smoke/probe/attempt directories
V6I9_ATTEMPT_PATTERN = re.compile(
    r"v6i9_map_awareness_(smoke2?|exact_telemetry_smoke|full_exact_telemetry|p15_smoke|probe_fix)$"
)

# The ONE canonical v6i9 directory
V6I9_CANONICAL = "v6i9_map_awareness"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                block = f.read(chunk)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()
    except OSError:
        return "ERROR"


def is_git_tracked(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    result = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(rel)],
        cwd=root, capture_output=True,
    )
    return result.returncode == 0


def classify_file(rel: Path, abs_path: Path, root: Path) -> tuple[str, str]:
    """Return (classification, reason)."""
    parts = rel.parts
    name = rel.name
    ext = rel.suffix.lower()
    rel_str = str(rel).replace("\\", "/")

    # Generated files
    for pat in GENERATED_PATTERNS:
        if pat in rel_str:
            return "GENERATED", f"matches generated pattern '{pat}'"

    # Checkpoint files
    if ext in CHECKPOINT_EXTS:
        return "UNKNOWN", "checkpoint — requires Stage 13 lineage review"

    # Artifacts directory routing
    if parts and parts[0] == "artifacts":
        subdir = parts[1] if len(parts) > 1 else ""

        if subdir in GOLDEN_DIRS:
            return "KEEP_GOLDEN", f"canonical closeout directory {subdir}"

        if subdir in PHASE10_INTERMEDIATES:
            return "DELETE", f"Phase 10 intermediate artifact directory — superseded by phase10_closeout"

        if subdir in PHASE6_INTERMEDIATES:
            return "DELETE", "Phase 6.1 intermediate perf investigation — superseded by phase6_1_closeout"

        if PROOF_LADDER_PATTERN.match(subdir):
            return "DELETE", "obsolete proof-ladder benchmark attempt — superseded"

        if V6I9_ATTEMPT_PATTERN.match(subdir):
            return "DELETE", "obsolete v6i9 map-awareness attempt — superseded by v6i9_map_awareness canonical"

        if subdir == V6I9_CANONICAL:
            return "KEEP_GOLDEN", "canonical v6i9 map-awareness evaluation artifacts"

        return "UNKNOWN", f"artifact in unclassified subdirectory '{subdir}'"

    # Logs
    if parts and parts[0] == "logs":
        return "DELETE", "runtime training log — unreferenced by any report"

    # Python source
    if ext in PYTHON_EXTS:
        if parts and parts[0] in {"rl", "gpu_env", "tests"}:
            return "KEEP", "active source or test module"
        if parts and parts[0] == "tools":
            return "UNKNOWN", "tool script — requires Stage 10 classification"
        if parts and parts[0] == "experiments":
            return "UNKNOWN", "experiment script — requires Stage 10 classification"
        if parts and parts[0] == "configs":
            return "KEEP", "config file"
        return "UNKNOWN", "Python file outside standard package directories"

    # Documentation
    if ext in DOC_EXTS:
        if parts and parts[0] == "docs":
            return "KEEP", "documentation"
        return "UNKNOWN", "doc file outside docs/"

    # CSV files
    if ext in CSV_EXTS:
        if parts and parts[0] == "csv":
            return "UNKNOWN", "CSV file — requires Stage 12 classification"
        if parts and parts[0] == "checkpoints":
            return "KEEP", "checkpoint training metric CSV"
        return "UNKNOWN", "CSV file — requires Stage 12 classification"

    # JSON files
    if ext in JSON_EXTS:
        if parts and parts[0] == "configs":
            return "KEEP", "training config JSON"
        if parts and parts[0] == "checkpoints":
            return "KEEP", "checkpoint run config JSON"
        return "UNKNOWN", "JSON file — requires further classification"

    # Scripts
    if ext in {".ps1", ".sh", ".bat"}:
        return "UNKNOWN", "shell script — requires Stage 10 classification"

    return "UNKNOWN", f"unclassified extension '{ext}'"


# ---------------------------------------------------------------------------
# Main inventory
# ---------------------------------------------------------------------------

def build_inventory(root: Path) -> list[dict]:
    records = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skip dirs in-place
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]

        for fname in filenames:
            abs_path = Path(dirpath) / fname
            try:
                rel = abs_path.relative_to(root)
            except ValueError:
                continue

            stat = abs_path.stat()
            classification, reason = classify_file(rel, abs_path, root)

            records.append({
                "path": str(rel).replace("\\", "/"),
                "extension": abs_path.suffix.lower(),
                "size_bytes": stat.st_size,
                "classification": classification,
                "reason": reason,
            })

    return records


def find_duplicates(records: list[dict], root: Path) -> list[dict]:
    """Find exact duplicate file contents by SHA-256 (skip large checkpoints for speed)."""
    hash_map: dict[str, list[str]] = defaultdict(list)
    dup_records = []

    for rec in records:
        size = rec["size_bytes"]
        # Skip files > 5 MB for speed (checkpoints)
        if size > 5 * 1024 * 1024:
            continue
        if rec["classification"] == "GENERATED":
            continue
        path = root / rec["path"]
        digest = sha256_file(path)
        hash_map[digest].append(rec["path"])

    for digest, paths in hash_map.items():
        if len(paths) > 1:
            for p in paths:
                dup_records.append({"sha256": digest, "path": p, "duplicate_count": len(paths)})

    return dup_records


def checkpoint_inventory(root: Path) -> list[dict]:
    ckpt_dir = root / "checkpoints"
    if not ckpt_dir.exists():
        return []

    records = []
    for f in sorted(ckpt_dir.rglob("*")):
        if not f.is_file():
            continue
        ext = f.suffix.lower()
        rel = f.relative_to(root)
        stat = f.stat()
        name = f.name

        # Classify by name patterns
        if name.startswith("final_"):
            ckpt_class = "KEEP"
            reason = "final checkpoint for a training run"
        elif name.startswith("interrupt_"):
            ckpt_class = "KEEP"
            reason = "interrupt checkpoint — may be most recent state of a run"
        elif re.match(r"ckpt_.*_\d+\.zip$", name):
            step_match = re.search(r"_(\d+)\.zip$", name)
            step = int(step_match.group(1)) if step_match else 0
            if step in {50_000, 100_000}:
                ckpt_class = "ARCHIVE"
                reason = f"early periodic checkpoint at step {step} — non-final, low-step"
            else:
                ckpt_class = "KEEP"
                reason = f"periodic checkpoint at step {step}"
        elif ext in {".csv", ".json"}:
            ckpt_class = "KEEP"
            reason = "checkpoint training metric/config file"
        elif ext == ".lock":
            ckpt_class = "DELETE"
            reason = "stale .lock file from completed training run"
        elif ".bak." in name:
            ckpt_class = "DELETE"
            reason = "backup file from interrupted CSV append"
        elif f.is_dir():
            ckpt_class = "UNKNOWN"
            reason = "checkpoint subdirectory"
        else:
            ckpt_class = "UNKNOWN"
            reason = "unclassified checkpoint file"

        records.append({
            "path": str(rel).replace("\\", "/"),
            "name": name,
            "extension": ext,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "classification": ckpt_class,
            "reason": reason,
        })

    return records


def source_inventory(root: Path) -> list[dict]:
    records = []
    for pkg in ["rl", "gpu_env", "tests", "tools", "experiments"]:
        pkg_dir = root / pkg
        if not pkg_dir.exists():
            continue
        for f in sorted(pkg_dir.rglob("*.py")):
            if "__pycache__" in f.parts:
                continue
            rel = f.relative_to(root)
            stat = f.stat()
            records.append({
                "path": str(rel).replace("\\", "/"),
                "package": pkg,
                "size_bytes": stat.st_size,
                "lines": sum(1 for _ in open(f, encoding="utf-8", errors="replace")),
            })
    return records


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("(empty)\n", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 11 repository cleanup inventory")
    parser.add_argument("--project-root", default=".", type=Path)
    parser.add_argument("--output-dir", default="artifacts/cleanup_audit", type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    out = (root / args.output_dir) if not args.output_dir.is_absolute() else args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    print(f"Project root: {root}")
    print(f"Output dir:   {out}")
    print()

    # Full file inventory
    print("Building full file inventory...")
    records = build_inventory(root)
    write_csv(out / "file_inventory.csv", records)
    print(f"  {len(records)} files inventoried")

    # Classification summary
    by_class = Counter(r["classification"] for r in records)
    print("  Classification summary:")
    for cls, count in sorted(by_class.items()):
        size = sum(r["size_bytes"] for r in records if r["classification"] == cls)
        print(f"    {cls:<30} {count:>6} files  {size / 1_048_576:>8.1f} MB")

    # Source inventory
    print("\nBuilding source inventory...")
    src_records = source_inventory(root)
    write_csv(out / "source_inventory.csv", src_records)
    total_lines = sum(r["lines"] for r in src_records)
    print(f"  {len(src_records)} Python files, {total_lines:,} total lines")

    # Checkpoint inventory
    print("\nBuilding checkpoint inventory...")
    ckpt_records = checkpoint_inventory(root)
    write_csv(out / "checkpoint_inventory.csv", ckpt_records)
    ckpt_by_class = Counter(r["classification"] for r in ckpt_records)
    ckpt_size_by_class = defaultdict(float)
    for r in ckpt_records:
        ckpt_size_by_class[r["classification"]] += r["size_bytes"] / 1_048_576
    print(f"  {len(ckpt_records)} checkpoint files")
    for cls in sorted(ckpt_by_class):
        print(f"    {cls:<20} {ckpt_by_class[cls]:>4} files  {ckpt_size_by_class[cls]:>8.1f} MB")

    # Duplicate detection (small files only for speed)
    print("\nFinding duplicate files (< 5 MB)...")
    dup_records = find_duplicates(records, root)
    write_csv(out / "duplicate_hashes.csv", dup_records)
    print(f"  {len(dup_records)} duplicate file entries across {len(set(r['sha256'] for r in dup_records))} unique hashes")

    # Cleanup candidates
    delete_candidates = [r for r in records if r["classification"] == "DELETE"]
    unknown_candidates = [r for r in records if r["classification"] == "UNKNOWN"]

    write_json(out / "cleanup_candidates.json", {
        "delete_candidates": delete_candidates,
        "unknown_candidates": unknown_candidates,
        "summary": dict(by_class),
    })

    # Large files
    large = sorted(
        [r for r in records if r["size_bytes"] > 10 * 1024 * 1024],
        key=lambda r: r["size_bytes"], reverse=True
    )
    write_csv(out / "large_files.csv", [
        {**r, "size_mb": round(r["size_bytes"] / 1_048_576, 1)} for r in large
    ])
    print(f"\n  {len(large)} files > 10 MB")

    # Artifact inventory (just artifacts/)
    art_records = [r for r in records if r["path"].startswith("artifacts/")]
    write_csv(out / "artifact_inventory.csv", art_records)

    # Log inventory
    log_records = [r for r in records if r["extension"] in LOG_EXTS or r["path"].startswith("logs/")]
    write_csv(out / "log_inventory.csv", log_records)

    # CSV inventory
    csv_records = [r for r in records if r["extension"] in CSV_EXTS]
    write_csv(out / "csv_inventory.csv", csv_records)

    print(f"\nInventory written to {out}")
    print(f"\nDELETE candidates: {len(delete_candidates)}")
    print(f"UNKNOWN candidates: {len(unknown_candidates)}")
    print("\nDone.")


if __name__ == "__main__":
    main()
