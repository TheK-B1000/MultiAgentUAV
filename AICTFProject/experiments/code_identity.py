r"""Scientific-code identity of a commit -- what an experiment executes, not what the repository holds.

    python experiments/code_identity.py <sha_a> <sha_b> [--write PATH]

A git SHA is provenance, but it is too coarse as an experimental identity: committing a CSV moves HEAD
without changing one instruction the experiment executes. Two commits are implementation-equivalent when
(1) the collector file's blob is identical and (2) nothing under SCIENTIFIC_PATHS differs. Generated
artifacts (artifacts/, checkpoints/, logs, ...) are outside SCIENTIFIC_PATHS; the frozen specs, pins and
pole certifications that live there are verified separately by the audits (by sha256), not here.

SCIENTIFIC_PATHS = every in-project directory the suite collector imports at runtime (experiments, rl,
gpu_env), the project-root *.py modules it imports (game_manager, macro_actions, _classes, _ops), and
configs/. Paths are relative to the project root (AICTFProject/).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCIENTIFIC_DIRS = ("experiments", "rl", "gpu_env", "configs")
COLLECTOR_REL = "experiments/collect_suite_distillation_states.py"
STATUS_PATHSPECS = (*SCIENTIFIC_DIRS, ":(glob)*.py")


def _git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {root}: {r.stderr.strip()}")
    return r.stdout


def is_scientific(rel: str) -> bool:
    """rel is relative to the project root; root-level *.py and anything under SCIENTIFIC_DIRS count."""
    head, _, rest = rel.partition("/")
    return head in SCIENTIFIC_DIRS if rest else rel.endswith(".py")


def _prefix(root: Path) -> str:
    return _git(root, "rev-parse", "--show-prefix").strip()


def _project_rel(path: str, prefix: str) -> str | None:
    return path[len(prefix):] if path.startswith(prefix) else None


def scientific_tree_sha256(rev: str, root: Path = ROOT) -> str:
    """sha256 over (path, blob id) of every scientific file at rev -- equal iff the scientific code is equal."""
    prefix = _prefix(root)
    lines = []
    for line in _git(root, "ls-tree", "-r", "--full-tree", rev).splitlines():
        meta, _, path = line.partition("\t")
        rel = _project_rel(path, prefix)
        if rel is not None and is_scientific(rel):
            lines.append(f"{rel}\t{meta.split()[2]}")
    if not lines:
        raise RuntimeError(f"no scientific files at {rev} under {root}")
    return hashlib.sha256("\n".join(sorted(lines)).encode()).hexdigest()


def blob_id(rev: str, rel: str, root: Path = ROOT) -> str:
    return _git(root, "rev-parse", f"{rev}:{_prefix(root)}{rel}").strip()


def scientific_dirty(root: Path = ROOT) -> bool:
    """Uncommitted changes anywhere in the scientific code (not only experiments/)."""
    return bool(_git(root, "status", "--porcelain", "--", *STATUS_PATHSPECS).strip())


def code_equivalence(sha_a: str, sha_b: str, root: Path = ROOT) -> dict:
    """Both conditions must hold: identical collector blob AND no scientific-path diff. Real SHAs are kept."""
    a, b = (_git(root, "rev-parse", "--verify", f"{s}^{{commit}}").strip() for s in (sha_a, sha_b))
    prefix = _prefix(root)
    changed = [p for p in _git(root, "diff", "--name-only", a, b).splitlines() if p]
    rels = [_project_rel(p, prefix) for p in changed]
    sci = sorted(r for r in rels if r is not None and is_scientific(r))
    other = sorted({(r.split("/")[0] if r is not None else "<outside project>") for r in rels} - {""})
    blob_a, blob_b = blob_id(a, COLLECTOR_REL, root), blob_id(b, COLLECTOR_REL, root)
    tree_a, tree_b = scientific_tree_sha256(a, root), scientific_tree_sha256(b, root)
    return {
        "sha_a": a, "sha_b": b,
        "collector": COLLECTOR_REL,
        "collector_blob_a": blob_a, "collector_blob_b": blob_b,
        "collector_identical": blob_a == blob_b,
        "scientific_paths": {"dirs": list(SCIENTIFIC_DIRS), "root_level": "*.py"},
        "scientific_tree_sha256_a": tree_a, "scientific_tree_sha256_b": tree_b,
        "scientific_changed_files": sci,
        "n_changed_files": len(changed),
        "changed_top_level": [d for d in other if not (d.endswith(".py") or d in SCIENTIFIC_DIRS)],
        "equivalent": blob_a == blob_b and tree_a == tree_b and not sci,
    }


def attestation_name(sha_a: str, sha_b: str) -> str:
    return f"COLLECTOR_COMMIT_EQUIVALENCE_{sha_a[:8]}_{sha_b[:8]}.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sha_a")
    ap.add_argument("sha_b")
    ap.add_argument("--write", type=Path, help="write a FROZEN attestation (refused unless equivalent)")
    a = ap.parse_args()
    eq = code_equivalence(a.sha_a, a.sha_b, ROOT)
    print(json.dumps(eq, indent=2))
    if a.write:
        if not eq["equivalent"]:
            raise SystemExit("REFUSING: the commits are not implementation-equivalent; nothing written")
        if a.write.exists():
            raise SystemExit(f"REFUSING: {a.write} exists (frozen attestations are never overwritten)")
        dirs = ", ".join(eq["changed_top_level"]) or "nothing"
        a.write.write_text(json.dumps({
            "record_id": "COLLECTOR_COMMIT_EQUIVALENCE",
            "status": "FROZEN",
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "statement": (f"{eq['sha_b'][:8]} differs from {eq['sha_a'][:8]} only in {dirs} "
                          f"({eq['n_changed_files']} files). No collector, environment, policy, or "
                          f"dataset-generation code changed. Collections run at the two commits are "
                          f"therefore implementation-equivalent despite differing repository HEADs. Both "
                          f"real SHAs are kept in their manifests; neither manifest is rewritten."),
            "equivalence": eq,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"-> {a.write}")
    return 0 if eq["equivalent"] else 1


if __name__ == "__main__":
    sys.exit(main())
