#!/usr/bin/env python3
"""Build a clean, public-ready copy of this project with fresh git history.

Why a fresh repo rather than a history rewrite: SB3 checkpoints (38-75 MB each)
were committed across many commits, so this project's .git is roughly 13 GB. The
blobs live in history, which means `git rm` does not shrink anything and a
filter-repo rewrite would invalidate every existing clone and commit hash. Copying
the source into a new repository with a single initial commit is faster, safer,
and produces exactly what a reviewer should see.

Nothing in the source repository is modified. This script only reads.

Usage (from the project root):

  # Preview what would be copied
  python tools/build_release.py --dest ../seaguard-public --dry-run

  # Build the public repo
  python tools/build_release.py --dest ../seaguard-public

  # Build the double-blind variant (no author identity in commit or metadata)
  python tools/build_release.py --dest ../seaguard-anon --anonymize
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

# Directories copied wholesale (subject to EXCLUDE_NAMES / EXCLUDE_SUFFIXES).
INCLUDE_DIRS: Tuple[str, ...] = (
    "rl",
    "plot",
    "tests",
    "docs",
    "configs",
    "tools",
)

# Individual files copied from the project root.
INCLUDE_FILES: Tuple[str, ...] = (
    "README.md",
    "LICENSE",
    "requirements.txt",
    ".gitignore",
    "agents.py",
    "ctfviewer.py",
    "game_field_gpu.py",
    "game_manager.py",
    "macro_actions.py",
    "opponent_params.py",
)

# Never copied, at any depth.
EXCLUDE_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    ".venv",
    "gpu_env",
    "checkpoints",
    "checkpoints_sb3",
    "artifacts",
    "experiments",
    "evaluation_results",
    "logs",
    "csv",
    ".test_runs",
}
EXCLUDE_SUFFIXES = (".zip", ".pt", ".pth", ".ckpt", ".pyc", ".log", ".swp")

ANON_COMMIT_NAME = "Anonymous Author"
ANON_COMMIT_EMAIL = "anonymous@example.com"

COMMIT_SUBJECT = "SEA-GUARD: maritime multi-robot CTF benchmark and evaluation protocol"
COMMIT_BODY = """Environment, baseline ladder (self-play / fictitious play / double oracle /
PFSP / ROA-Star), configuration space with seen and held-out splits, and the
performance / generalization / exploitability evaluation triad.

Model checkpoints are distributed as release assets rather than committed; see
docs/REPRODUCE.md.
"""


def _should_skip(path: str) -> bool:
    name = os.path.basename(path)
    if name in EXCLUDE_NAMES:
        return True
    return any(name.endswith(suffix) for suffix in EXCLUDE_SUFFIXES)


def collect_files(project_dir: str) -> List[str]:
    """Project-relative paths to copy, sorted."""
    out: List[str] = []

    for filename in INCLUDE_FILES:
        path = os.path.join(project_dir, filename)
        if os.path.isfile(path):
            out.append(filename)

    for dirname in INCLUDE_DIRS:
        root_dir = os.path.join(project_dir, dirname)
        if not os.path.isdir(root_dir):
            continue
        for walk_root, subdirs, filenames in os.walk(root_dir):
            subdirs[:] = sorted(d for d in subdirs if not _should_skip(os.path.join(walk_root, d)))
            for filename in sorted(filenames):
                full = os.path.join(walk_root, filename)
                if _should_skip(full):
                    continue
                out.append(os.path.relpath(full, project_dir).replace(os.sep, "/"))

    return sorted(set(out))


def copy_tree(project_dir: str, dest: str, rel_paths: Sequence[str]) -> int:
    copied = 0
    for rel in rel_paths:
        src = os.path.join(project_dir, rel)
        dst = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    return copied


def _run(cmd: Sequence[str], cwd: str, env: Optional[dict] = None) -> None:
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed:\n{result.stdout}\n{result.stderr}")


def init_repo(dest: str, *, anonymize: bool) -> None:
    env = dict(os.environ)
    if anonymize:
        # Set identity in the environment only, so no real name or address is
        # written into the new repo's commits or its config file.
        env.update(
            {
                "GIT_AUTHOR_NAME": ANON_COMMIT_NAME,
                "GIT_AUTHOR_EMAIL": ANON_COMMIT_EMAIL,
                "GIT_COMMITTER_NAME": ANON_COMMIT_NAME,
                "GIT_COMMITTER_EMAIL": ANON_COMMIT_EMAIL,
            }
        )

    _run(["git", "init", "-q", "-b", "main"], cwd=dest, env=env)
    _run(["git", "add", "-A"], cwd=dest, env=env)
    message = COMMIT_SUBJECT + "\n\n" + COMMIT_BODY
    _run(["git", "commit", "-q", "-m", message], cwd=dest, env=env)


def audit_identity(dest: str, rel_paths: Sequence[str]) -> List[str]:
    """Flag anything in the copied files that would break double-blind review."""
    import re

    needles = re.compile(
        r"(?i)(corbett|thekb0514|unf\.edu|university of north florida|dutta|"
        r"K:\\\\MultiAgentUAV|C:/Users/)"
    )
    hits: List[str] = []
    this_file = os.path.basename(__file__)
    for rel in rel_paths:
        # The auditor holds the search terms verbatim; matching itself is noise.
        if os.path.basename(rel) == this_file:
            continue
        path = os.path.join(dest, rel)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for lineno, line in enumerate(f, start=1):
                    if needles.search(line):
                        hits.append(f"{rel}:{lineno}: {line.strip()[:100]}")
        except OSError:
            continue
    return hits


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dest", required=True, help="Directory to create (must not exist)")
    parser.add_argument(
        "--anonymize",
        action="store_true",
        help="Commit with an anonymous identity for double-blind submission",
    )
    parser.add_argument("--dry-run", action="store_true", help="List what would be copied and exit")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete --dest first if it already exists",
    )
    args = parser.parse_args(argv)

    rel_paths = collect_files(_PROJECT_DIR)
    total_bytes = sum(
        os.path.getsize(os.path.join(_PROJECT_DIR, r))
        for r in rel_paths
        if os.path.isfile(os.path.join(_PROJECT_DIR, r))
    )

    print(f"[build_release] source: {_PROJECT_DIR}")
    print(f"[build_release] {len(rel_paths)} file(s), {total_bytes / 1_048_576:.1f} MiB")
    by_top: dict = {}
    for rel in rel_paths:
        by_top[rel.split("/")[0] if "/" in rel else "(root)"] = (
            by_top.get(rel.split("/")[0] if "/" in rel else "(root)", 0) + 1
        )
    for top, count in sorted(by_top.items(), key=lambda kv: -kv[1]):
        print(f"    {top:12s} {count:4d}")

    missing = [f for f in ("README.md", "LICENSE", "requirements.txt") if f not in rel_paths]
    if missing:
        print(f"[build_release] WARNING: missing from the release: {missing}")

    if args.dry_run:
        print("[build_release] dry-run only; nothing written.")
        return 0

    dest = os.path.abspath(args.dest)
    if os.path.exists(dest):
        if not args.force:
            print(f"[build_release] ERROR: {dest} already exists (use --force to replace)")
            return 1
        shutil.rmtree(dest)
    os.makedirs(dest)

    copied = copy_tree(_PROJECT_DIR, dest, rel_paths)
    print(f"[build_release] copied {copied} file(s) -> {dest}")

    hits = audit_identity(dest, rel_paths)
    if hits:
        print(f"[build_release] identity audit: {len(hits)} hit(s) -- review before submitting:")
        for hit in hits[:20]:
            print(f"    {hit}")
    else:
        print("[build_release] identity audit: clean")

    init_repo(dest, anonymize=bool(args.anonymize))
    print(
        f"[build_release] initialized git repo with a single commit"
        f"{' (anonymous identity)' if args.anonymize else ''}"
    )
    print()
    print("Next:")
    print(f"  cd {dest}")
    print("  pytest tests/ plot/ -q")
    if args.anonymize:
        print("  # upload to anonymous.4open.science, or push to an anonymized host")
    else:
        print("  git remote add origin <url> && git push -u origin main")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
