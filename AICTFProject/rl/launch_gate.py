"""Automated launch gate for the K=2 selective-supervision latent run.

Every check here exists because a specific previous experiment was lost to the
condition it tests. The point is to stop relying on "remember not to make that
mistake again" -- the run refuses to start instead.

Two halves:

STATIC   ``require_launch_authorized(...)`` runs before a single environment step
         and raises ``LaunchGateError`` unless every blocking check passes.

RUNTIME  ``PoleAssignmentAuditor``, ``UpdateCounters`` and ``use_time_lookup``
         catch the failures that are invisible until the run is already moving:
         a treatment that never instantiated, an assignment that silently became
         random, a loss that never reached the optimizer.

Deliberately torch-free so the whole gate is testable without a GPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

# ---------------------------------------------------------------- frozen facts

SUPPORT_FLOOR = 32
N_CELLS = 16
COLLECTION_BLOCK = (10_700_001, 10_700_160)
CALIB_BLOCK = (10_700_097, 10_700_128)
FINAL_BLOCKS = ((10_600_001, 10_600_192),)   # RASR FINAL: sealed, never touched
# All four must be frozen before launch. kappa is the commit/abstain confidence
# cutoff and is a SEPARATE object from tau, the Gate-1 EVAL accuracy criterion --
# see AMENDMENT_3. A run that starts with kappa unfrozen has no defined abstention
# behaviour at all, which silently collapses the three-class design to two.
REQUIRED_THRESHOLDS = ("tau", "rho", "o_max", "kappa")


class LaunchGateError(RuntimeError):
    """Raised when the intended experiment is not the experiment about to run."""


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    blocking: bool = True

    @property
    def status(self) -> str:
        return "PASS" if self.passed else ("FAIL" if self.blocking else "WARN")


# ------------------------------------------------------------- static checks

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_collection_complete(data_dir: Path) -> Check:
    marker = Path(data_dir) / "COLLECTION_COMPLETE.json"
    if not marker.is_file():
        return Check("collection_complete", False,
                     f"{marker} does not exist; collection has not reached its barrier")
    rec = json.loads(marker.read_text(encoding="utf-8"))
    if rec.get("verdict") != "COLLECTION_COMPLETE":
        return Check("collection_complete", False, f"verdict is {rec.get('verdict')!r}")
    return Check("collection_complete", True, f"complete at {rec.get('utc')}")


def check_seed_block(data_dir: Path) -> Check:
    """The collected seeds must be EXACTLY the frozen block -- no more, no fewer."""
    shards = sorted((Path(data_dir) / "seed_shards").glob("seed_*.npz"))
    seeds = {int(p.stem.split("seed_")[-1]) for p in shards}
    expected = set(range(COLLECTION_BLOCK[0], COLLECTION_BLOCK[1] + 1))
    if seeds == expected:
        return Check("seed_block", True, f"exactly {len(seeds)} frozen seeds")
    extra, missing = sorted(seeds - expected), sorted(expected - seeds)
    return Check("seed_block", False,
                 f"block mismatch: {len(missing)} missing, {len(extra)} unexpected"
                 + (f"; first unexpected {extra[:3]}" if extra else "")
                 + (f"; first missing {missing[:3]}" if missing else ""))


def check_support_floor(audit_path: Path) -> Check:
    """All 16 cells at or above the floor. Reads the audit; never recomputes it."""
    audit_path = Path(audit_path)
    if not audit_path.is_file():
        return Check("support_floor", False,
                     f"{audit_path.name} does not exist; the one-shot audit has not run")
    rec = json.loads(audit_path.read_text(encoding="utf-8"))
    cells = rec.get("cells", {})
    if len(cells) != N_CELLS:
        return Check("support_floor", False, f"audit covers {len(cells)} cells, expected {N_CELLS}")
    if rec.get("VERDICT") != "VALID":
        return Check("support_floor", False,
                     f"audit VERDICT is {rec.get('VERDICT')!r}; invalid cells "
                     f"{rec.get('invalid_cells')}")
    short = {k: v["n_distinct_seeds"] for k, v in cells.items()
             if v.get("n_distinct_seeds", 0) < SUPPORT_FLOOR}
    if short:
        return Check("support_floor", False, f"cells below floor despite VALID verdict: {short}")
    worst = min(v["n_distinct_seeds"] for v in cells.values())
    return Check("support_floor", True, f"16/16 cells pass; scarcest cell has {worst} seeds")


def check_final_untouched(*search_roots: Path) -> Check:
    """No FINAL seed may appear anywhere in the training-side artifacts."""
    offenders: list[str] = []
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in (".json", ".jsonl", ".npz"):
                continue
            name_seed = path.stem.replace("seed_", "")
            if name_seed.isdigit():
                seed = int(name_seed)
                if any(lo <= seed <= hi for lo, hi in FINAL_BLOCKS):
                    offenders.append(str(path))
    if offenders:
        return Check("final_untouched", False, f"FINAL seeds present: {offenders[:3]}")
    return Check("final_untouched", True, "no FINAL seed artifacts found")


def check_calib_split(data_dir: Path) -> Check:
    shards = (Path(data_dir) / "seed_shards")
    lo, hi = CALIB_BLOCK
    present = [s for s in range(lo, hi + 1) if (shards / f"seed_{s}.npz").is_file()]
    if len(present) != hi - lo + 1:
        return Check("calib_split", False,
                     f"CALIB has {len(present)}/{hi - lo + 1} seeds present")
    return Check("calib_split", True, f"CALIB {lo}..{hi} complete ({len(present)} seeds)")


def check_thresholds_frozen(thresholds_path: Path) -> Check:
    """tau, rho and o_max must exist, be numeric, and be marked frozen."""
    thresholds_path = Path(thresholds_path)
    if not thresholds_path.is_file():
        return Check("thresholds_frozen", False,
                     f"{thresholds_path.name} does not exist; calibration has not happened")
    rec = json.loads(thresholds_path.read_text(encoding="utf-8"))
    if not str(rec.get("status", "")).startswith("FROZEN"):
        return Check("thresholds_frozen", False, f"status is {rec.get('status')!r}")
    values = rec.get("thresholds", {})
    missing = [k for k in REQUIRED_THRESHOLDS if not isinstance(values.get(k), (int, float))]
    if missing:
        return Check("thresholds_frozen", False, f"missing or non-numeric: {missing}")
    if rec.get("calibrated_on") != "CALIB":
        return Check("thresholds_frozen", False,
                     f"calibrated_on is {rec.get('calibrated_on')!r}, must be 'CALIB'")
    shown = {k: values[k] for k in REQUIRED_THRESHOLDS}
    return Check("thresholds_frozen", True, f"frozen on CALIB: {shown}")


def check_content_hashes(expected: Mapping[str, str]) -> Check:
    """Frozen documents must hash to exactly what was recorded at freeze time."""
    drift = []
    for rel, want in expected.items():
        path = Path(rel)
        if not path.is_file():
            drift.append(f"{path.name}: missing")
            continue
        got = _sha256(path)
        if got != want:
            drift.append(f"{path.name}: {got[:12]} != {want[:12]}")
    if drift:
        return Check("content_hashes", False, "; ".join(drift))
    return Check("content_hashes", True, f"{len(expected)} frozen document(s) match")


def check_opponent_mode(cfg: Any) -> Check:
    """OPPONENT_POOL resamples the scripted opponent every episode.

    EXP2B/EXP2C never instantiated their stated persistent assigned-pole treatment
    because of this; measured occupancy came out ~50/50.
    """
    mode = str(getattr(cfg, "mode", ""))
    randomize = bool(getattr(cfg, "randomize_scripted_opponent", False))
    if mode != "FIXED_OPPONENT":
        return Check("opponent_mode", False, f"mode is {mode!r}, must be FIXED_OPPONENT")
    if randomize:
        return Check("opponent_mode", False,
                     "mode is FIXED_OPPONENT but randomize_scripted_opponent is True, "
                     "which reproduces OPPONENT_POOL behaviour")
    return Check("opponent_mode", True, "FIXED_OPPONENT, no per-episode resampling")


def check_fresh_training(cfg: Any) -> Check:
    """Fresh means step 0. No checkpoint, no resume, no accidental warm start."""
    load_path = getattr(cfg, "load_path", None)
    if load_path:
        return Check("fresh_training", False, f"load_path is set: {load_path!r}")
    return Check("fresh_training", True, "no load_path; run starts at step 0")


def check_rollout_overshoot(cfg: Any) -> Check:
    """Compute the overshoot prospectively rather than discovering it in the logs."""
    n_envs = int(getattr(cfg, "n_envs", 0))
    n_steps = int(getattr(cfg, "n_steps", 0))
    total = int(getattr(cfg, "total_timesteps", 0))
    if min(n_envs, n_steps, total) <= 0:
        return Check("rollout_overshoot", False,
                     f"cannot compute: n_envs={n_envs} n_steps={n_steps} total={total}")
    per_rollout = n_envs * n_steps
    rollouts = -(-total // per_rollout)          # ceil
    actual = rollouts * per_rollout
    over = actual - total
    detail = (f"{rollouts} rollouts x {per_rollout} = {actual} steps; "
              f"overshoot {over} ({100.0 * over / total:.2f}%)")
    return Check("rollout_overshoot", True, detail, blocking=False)


def check_process_uniqueness(marker: str = "train_ppo") -> Check:
    """Two trainers launched independently is the specific failure this prevents."""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
             "| Select-Object -ExpandProperty CommandLine"],
            capture_output=True, text=True, timeout=30).stdout
    except Exception as exc:                      # noqa: BLE001 - best effort probe
        return Check("process_uniqueness", True, f"probe unavailable ({exc}); not blocking",
                     blocking=False)
    hits = [line.strip() for line in out.splitlines() if marker in line]
    if len(hits) > 1:
        return Check("process_uniqueness", False,
                     f"{len(hits)} processes already match {marker!r}")
    if hits:
        return Check("process_uniqueness", False,
                     f"a process matching {marker!r} is already running")
    return Check("process_uniqueness", True, f"no other {marker!r} process")


def provenance_record(repo_root: Path, *artifacts: Path) -> dict:
    """SHA + git commit + block identity, attached to whatever is about to run."""
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(repo_root),
                                capture_output=True, text=True, timeout=30).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=str(repo_root),
                                    capture_output=True, text=True, timeout=30).stdout.strip())
    except Exception:                             # noqa: BLE001
        commit, dirty = "unavailable", True
    return {
        "git_commit": commit,
        "working_tree_dirty": dirty,
        "seed_block": list(COLLECTION_BLOCK),
        "calib_block": list(CALIB_BLOCK),
        "artifacts": {str(p): (_sha256(Path(p)) if Path(p).is_file() else "missing")
                      for p in artifacts},
    }


def require_launch_authorized(
    cfg: Any,
    data_dir: Path,
    audit_path: Path,
    thresholds_path: Path,
    expected_hashes: Mapping[str, str] | None = None,
    search_roots: Sequence[Path] = (),
    strict_process_check: bool = True,
) -> list[Check]:
    """Run every static check. Raise unless all blocking checks pass."""
    checks = [
        check_collection_complete(data_dir),
        check_seed_block(data_dir),
        check_support_floor(audit_path),
        check_calib_split(data_dir),
        check_final_untouched(*search_roots),
        check_thresholds_frozen(thresholds_path),
        check_opponent_mode(cfg),
        check_fresh_training(cfg),
        check_rollout_overshoot(cfg),
    ]
    if expected_hashes:
        checks.append(check_content_hashes(expected_hashes))
    if strict_process_check:
        checks.append(check_process_uniqueness())

    failed = [c for c in checks if c.blocking and not c.passed]
    if failed:
        lines = "\n".join(f"  [{c.status}] {c.name}: {c.detail}" for c in checks)
        raise LaunchGateError(
            f"LAUNCH REFUSED -- {len(failed)} blocking check(s) failed:\n{lines}")
    return checks


# ------------------------------------------------------------ runtime auditors

@dataclass
class PoleAssignmentAuditor:
    """Assert z -> pole persistence after EVERY reset, not just episode 1.

    EXP2B/EXP2C stated a persistent 16/0/0/16 assigned-pole treatment and measured
    ~50/50 occupancy. Episode 1 was correct; nothing checked episode 2.
    """
    expected: Mapping[int, str] = field(default_factory=lambda: {0: "A", 1: "B"})
    resets: int = 0
    violations: list[str] = field(default_factory=list)
    occupancy: dict[tuple[int, str], int] = field(default_factory=dict)

    def observe_reset(self, z: int, pole: str) -> None:
        self.resets += 1
        key = (int(z), str(pole))
        self.occupancy[key] = self.occupancy.get(key, 0) + 1
        want = self.expected.get(int(z))
        if want is not None and str(pole) != want:
            self.violations.append(f"reset #{self.resets}: z={z} -> pole {pole!r}, expected {want!r}")

    def require_clean(self, min_resets: int = 1) -> None:
        if self.resets < min_resets:
            raise LaunchGateError(
                f"pole auditor saw {self.resets} resets, expected at least {min_resets}; "
                "the treatment may never have been instantiated")
        if self.violations:
            raise LaunchGateError(
                f"z->pole persistence violated {len(self.violations)} time(s): "
                + "; ".join(self.violations[:5]))
        exercised = {z for z, _ in self.occupancy}
        missing = set(self.expected) - exercised
        if missing:
            raise LaunchGateError(f"latents never exercised: {sorted(missing)}")

    def telemetry(self) -> dict:
        total = max(1, self.resets)
        return {"resets": self.resets,
                "violations": len(self.violations),
                "occupancy": {f"z{z}->{p}": n for (z, p), n in sorted(self.occupancy.items())},
                "occupancy_frac": {f"z{z}->{p}": n / total
                                   for (z, p), n in sorted(self.occupancy.items())}}


@dataclass
class UpdateCounters:
    """Nonzero update counts before a run may be called valid.

    A loss that is constructed but never reaches the optimizer is invisible in the
    logs and fatal to the claim. So is a run that dies before training on a
    NameError -- both show up here as a zero.
    """
    counts: dict[str, int] = field(default_factory=dict)

    def bump(self, name: str, n: int = 1) -> None:
        self.counts[name] = self.counts.get(name, 0) + int(n)

    def require_nonzero(self, names: Iterable[str]) -> None:
        zero = [n for n in names if self.counts.get(n, 0) <= 0]
        if zero:
            raise LaunchGateError(
                f"these components never updated: {zero}; counts={dict(self.counts)}")

    def require_no_pressure(self, name: str) -> None:
        """Unresolved examples must receive NO A/B ranking pressure."""
        if self.counts.get(name, 0) != 0:
            raise LaunchGateError(
                f"{name!r} received {self.counts[name]} update(s); unresolved examples "
                "must receive no ranking pressure")


def use_time_lookup(registry: Mapping[str, Callable], key: str) -> Callable:
    """Resolve a treatment hook AT USE TIME.

    SAPPO's treatment never instantiated because the runner was cached at
    construction and the hook was bound to a stale object. Looking it up on every
    call is the fix; this wrapper exists so the intent is greppable.
    """
    if key not in registry:
        raise LaunchGateError(f"treatment hook {key!r} is not registered at use time; "
                              f"available: {sorted(registry)}")
    hook = registry[key]
    if not callable(hook):
        raise LaunchGateError(f"treatment hook {key!r} is not callable: {type(hook)!r}")
    return hook


def format_checks(checks: Sequence[Check]) -> str:
    width = max((len(c.name) for c in checks), default=0)
    return "\n".join(f"  [{c.status:4s}] {c.name:{width}s}  {c.detail}" for c in checks)
