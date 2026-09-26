r"""Refuse a training launch unless its governing spec is committed, clean, frozen and consistent.

    python experiments/frozen_launch_guard.py --spec <spec.json> --policy A [--repo-root .]

Exit 0 and one OK line per check if every guard passes; exit 1 with REFUSE lines otherwise.
Called by detached launchers before each policy, so a launcher cannot start a run the spec does
not authorize. Guards, all required:

  spec_tracked          `git ls-files --error-unmatch` -- an UNTRACKED spec is invisible to
                        `git diff HEAD`, which is how seed 23200001 was spent from a spec that
                        had never been committed (2026-09-26)
  spec_no_unstaged      `git diff --quiet -- <spec>` (worktree vs index)
  spec_no_staged        `git diff --cached --quiet -- <spec>` (index vs HEAD)
  spec_frozen           status starts with FROZEN
  launch_seed           the LAUNCH command's --seed equals SEEDS[<policy>]
  launch_experiment_id  the LAUNCH command's --experiment-id equals SEEDS.registry_experiment_id
  registry_seed         seed_registry.check_training_seed: registered, inside the block, RESERVED
  seed_never_trained    no trainer run manifest already holds the seed
  warm_start_path       the LAUNCH command's --load-path equals SOURCE_CHECKPOINTS[<policy>].path
  warm_start_sha256     that file exists and its sha256 equals the pin

A missing key is a refusal, never a default.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import seed_registry as SR  # noqa: E402

MANIFEST_RECORD = "train_specialist_scale run manifest"


def _git(repo: Path, *args: str) -> int:
    return subprocess.run(["git", "-C", str(repo), *args], stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode


def _flag(tokens: list[str], flag: str) -> str | None:
    return tokens[tokens.index(flag) + 1] if flag in tokens and tokens.index(flag) + 1 < len(tokens) else None


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_launch(spec_path: Path, policy: str, repo_root: Path) -> list[tuple[str, bool, str]]:
    """Every guard, in order. ``repo_root`` is the git repo that must hold the spec, and the
    root that relative checkpoint paths resolve against."""
    out: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str) -> bool:
        out.append((name, bool(ok), detail))
        return bool(ok)

    repo_root = repo_root.resolve()
    spec_path = spec_path.resolve()
    try:
        rel = spec_path.relative_to(repo_root).as_posix()
    except ValueError:
        check("spec_inside_repo", False, f"{spec_path} is not under {repo_root}")
        return out
    if not spec_path.is_file():
        check("spec_exists", False, rel)
        return out

    tracked = check("spec_tracked", _git(repo_root, "ls-files", "--error-unmatch", "--", rel) == 0,
                    rel)
    if tracked:
        check("spec_no_unstaged", _git(repo_root, "diff", "--quiet", "--", rel) == 0, rel)
        check("spec_no_staged", _git(repo_root, "diff", "--cached", "--quiet", "--", rel) == 0, rel)

    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        status = spec["status"]
        seeds = spec["SEEDS"]
        exp_id = seeds["registry_experiment_id"]
        seed = int(seeds[policy])
        tokens = shlex.split(spec["LAUNCH"][policy], posix=True)
        src = spec["SOURCE_CHECKPOINTS"][policy]
        pin_path, pin_sha = src["path"], src["sha256"]
    except (KeyError, TypeError, ValueError) as e:
        check("spec_keys_present", False, f"{type(e).__name__}: {e}")
        return out

    check("spec_frozen", str(status).startswith("FROZEN"), str(status))
    cmd_seed = _flag(tokens, "--seed")
    check("launch_seed", cmd_seed is not None and cmd_seed == str(seed), f"--seed {cmd_seed} vs SEEDS.{policy}={seed}")
    cmd_id = _flag(tokens, "--experiment-id")
    check("launch_experiment_id", cmd_id == exp_id, f"--experiment-id {cmd_id} vs {exp_id}")
    ok, msg, _ = SR.check_training_seed(seed, exp_id)
    check("registry_seed", ok, msg)
    prior = SR.prior_training_uses(seed, MANIFEST_RECORD)
    check("seed_never_trained", not prior, str(prior) if prior else f"no run manifest holds {seed}")

    cmd_load = _flag(tokens, "--load-path")
    check("warm_start_path", cmd_load == pin_path, f"--load-path {cmd_load} vs pinned {pin_path}")
    ck = repo_root / pin_path
    if check("warm_start_exists", ck.is_file(), pin_path):
        have = _sha256(ck)
        check("warm_start_sha256", have == pin_sha, f"{have[:16]}... vs pinned {str(pin_sha)[:16]}...")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--policy", required=True, choices=("A", "B"))
    ap.add_argument("--repo-root", default=str(ROOT))
    a = ap.parse_args()
    results = check_launch(Path(a.spec), a.policy, Path(a.repo_root))
    for name, ok, detail in results:
        print(f"  {'OK    ' if ok else 'REFUSE'} {name:<22} {detail}")
    passed = bool(results) and all(ok for _, ok, _ in results)
    print(f"frozen launch guard pi_{a.policy}: {'PASS' if passed else 'REFUSE'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
