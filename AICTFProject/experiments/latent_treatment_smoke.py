"""End-to-end treatment smoke for the K=2 selective-supervision latent run.

Deliberately tiny and structurally incapable of being mistaken for a scientific
run: a couple of rollouts, a single optimizer step, no evaluation, no verdict about
learning. It answers one question only --

    does the complete treatment path exist?
    data -> labels -> loss -> optimizer -> parameter update

It proves five things, and refuses to claim any of them on synthetic inputs:

  1. TREATMENT EXISTS      both z0 and z1 execute; runtime counters are nonzero
  2. ASSIGNMENT PERSISTS   many resets, every (z, live opponent) pair still correct
  3. SUPERVISION IS REAL   resolved examples create strategic pressure;
                           unresolved examples create EXACTLY zero
  4. OPTIMIZER IS REAL     intended trainable parameters change; frozen components
                           remain bit-identical
  5. LAUNCH STAYS BARRED   without earned artifacts -- COLLECTION_COMPLETE, a VALID
                           support audit, and frozen tau/rho/o_max -- the live smoke
                           hard-refuses

Fixtures may exercise the plumbing. Nothing synthetic may ever satisfy ``--live``:
there are no placeholder thresholds anywhere in this file, by design.

Run:  python experiments/latent_treatment_smoke.py --live
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl import launch_audit_hooks as hooks                     # noqa: E402
from rl.launch_gate import (                                   # noqa: E402
    Check,
    LaunchGateError,
    check_collection_complete,
    check_final_untouched,
    check_support_floor,
    check_thresholds_frozen,
    format_checks,
)

SD = ROOT / "artifacts" / "strategic_demand"
DATA = SD / "stratified_regime_data"
AUDIT = DATA / "SUPPORT_VALIDITY.json"
THRESHOLDS = SD / "sppo" / "ABSTENTION_THRESHOLDS.json"
OUT = SD / "sppo" / "LATENT_TREATMENT_SMOKE.json"

RESOLVED = "ranking_on_resolved"
UNRESOLVED = "ranking_on_unresolved"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _digest(arrays: Mapping[str, Any]) -> str:
    """Order-independent bit-exact digest over a set of parameter tensors."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        value = arrays[name]
        raw = getattr(value, "detach", lambda: value)()
        raw = getattr(raw, "cpu", lambda: raw)()
        raw = getattr(raw, "numpy", lambda: raw)()
        h.update(name.encode("utf-8"))
        h.update(memoryview(raw).cast("B") if hasattr(raw, "__buffer__") else bytes(raw.tobytes()))
    return h.hexdigest()


# ------------------------------------------------------------------- protocol

class TreatmentProbe(Protocol):
    """The minimum surface a treatment must expose to be smoke-testable.

    Kept deliberately small. A treatment that cannot answer these questions is not
    ready to consume a million steps.
    """

    def episode_boundaries(self) -> Iterable[tuple[int, int, int]]:
        """Yield ``(env_index, z, live_opponent_id)`` per episode close."""

    def named_parameters(self) -> Mapping[str, Any]: ...

    def frozen_parameter_names(self) -> Sequence[str]: ...

    def trainable_parameter_names(self) -> Sequence[str]: ...

    def apply_supervision(self, bump: Callable[[str, int], None]) -> None:
        """Run one supervision pass, bumping counters for the pressure applied."""

    def optimizer_step(self) -> int:
        """Perform the tiny update. Return the number of optimizer steps taken."""


# ------------------------------------------------------------------- witness

class ParameterWitness:
    """Snapshot parameters, then prove which ones did and did not move."""

    def __init__(self, source: Callable[[], Mapping[str, Any]]):
        self._source = source
        self._before: dict[str, bytes] = {}

    @staticmethod
    def _raw(value: Any) -> bytes:
        raw = getattr(value, "detach", lambda: value)()
        raw = getattr(raw, "cpu", lambda: raw)()
        raw = getattr(raw, "numpy", lambda: raw)()
        return raw.tobytes()

    def snapshot(self) -> None:
        self._before = {k: self._raw(v) for k, v in self._source().items()}

    def compare(self) -> tuple[list[str], list[str]]:
        if not self._before:
            raise LaunchGateError("ParameterWitness.compare() called before snapshot()")
        after = {k: self._raw(v) for k, v in self._source().items()}
        missing = sorted(set(self._before) - set(after))
        if missing:
            raise LaunchGateError(f"parameters vanished during the update: {missing}")
        changed = sorted(k for k, v in after.items() if self._before.get(k) != v)
        unchanged = sorted(k for k in after if k not in changed)
        return changed, unchanged

    def digest(self, names: Sequence[str]) -> str:
        current = self._source()
        return _digest({n: current[n] for n in names if n in current})


# ------------------------------------------------------------------- result

@dataclass
class SmokeResult:
    verdict: str = "INVALID_BEFORE_TRAINING"
    utc: str = field(default_factory=_now)
    mode: str = "fixture"
    resets_observed: int = 0
    z0_episodes: int = 0
    z1_episodes: int = 0
    pole_assignment_violations: int = 0
    resolved_pressure_count: int = 0
    unresolved_pressure_count: int = 0
    optimizer_steps: int = 0
    changed_parameter_groups: list[str] = field(default_factory=list)
    unchanged_parameter_groups: list[str] = field(default_factory=list)
    frozen_parameter_hash_match: bool = False
    threshold_artifact_sha: str = ""
    failures: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ------------------------------------------------------- live artifact gating

def require_live_artifacts(
    data_dir: Path = DATA,
    audit_path: Path = AUDIT,
    thresholds_path: Path = THRESHOLDS,
) -> list[Check]:
    """The live smoke may only run on artifacts that were genuinely earned."""
    checks = [
        check_collection_complete(data_dir),
        check_support_floor(audit_path),
        check_thresholds_frozen(thresholds_path),
        check_final_untouched(data_dir),
    ]
    failed = [c for c in checks if c.blocking and not c.passed]
    if failed:
        raise LaunchGateError(
            "LIVE SMOKE REFUSED -- the treatment path may not be exercised on "
            f"unearned artifacts:\n{format_checks(checks)}")
    return checks


# ---------------------------------------------------------------- the smoke

def run_smoke(
    probe: TreatmentProbe,
    *,
    z_to_pole: Mapping[int, str],
    opponent_to_pole: Mapping[int, str],
    mode: str = "fixture",
    thresholds_path: Path | None = None,
    min_resets: int = 4,
    required_counters: Sequence[str] = ("ppo", RESOLVED),
) -> SmokeResult:
    """Exercise the whole treatment path once and report a receipt."""
    result = SmokeResult(mode=mode)

    class _Host:                       # stands in for the trainer
        pass

    host = _Host()
    auditors = hooks.attach(host, z_to_pole, opponent_to_pole,
                            hard_fail=False, artifact_dir=None)

    # 1 + 2 -- both latents execute, and assignment survives every reset
    for env_index, z, opponent_id in probe.episode_boundaries():
        auditors.observe_episode_close(env_index, z, opponent_id)
    result.resets_observed = auditors.episodes_observed
    occupancy = auditors.pole.occupancy
    result.z0_episodes = sum(n for (z, _), n in occupancy.items() if z == 0)
    result.z1_episodes = sum(n for (z, _), n in occupancy.items() if z == 1)
    result.pole_assignment_violations = len(auditors.pole.violations)
    if result.pole_assignment_violations:
        result.failures.append(
            f"pole assignment violated {result.pole_assignment_violations} time(s): "
            + "; ".join(auditors.pole.violations[:3]))
    if result.resets_observed < min_resets:
        result.failures.append(
            f"only {result.resets_observed} resets observed, need at least {min_resets}")
    for z, label in ((0, "z0"), (1, "z1")):
        if sum(n for (zz, _), n in occupancy.items() if zz == z) == 0:
            result.failures.append(f"{label} never executed")

    # 4a -- witness parameters before the update
    witness = ParameterWitness(probe.named_parameters)
    witness.snapshot()
    frozen_names = list(probe.frozen_parameter_names())
    frozen_before = witness.digest(frozen_names)

    # 3 -- supervision reaches the optimizer, and only where it should
    probe.apply_supervision(auditors.bump)
    result.resolved_pressure_count = auditors.counters.counts.get(RESOLVED, 0)
    result.unresolved_pressure_count = auditors.counters.counts.get(UNRESOLVED, 0)
    if result.unresolved_pressure_count != 0:
        result.failures.append(
            f"unresolved examples received {result.unresolved_pressure_count} unit(s) of "
            "ranking pressure; the third class must generate exactly zero")
    if result.resolved_pressure_count <= 0:
        result.failures.append("resolved examples generated no strategic pressure")

    # 4b -- the optimizer actually moved what it was supposed to move
    result.optimizer_steps = int(probe.optimizer_step())
    if result.optimizer_steps <= 0:
        result.failures.append("no optimizer step was taken")
    changed, unchanged = witness.compare()
    result.changed_parameter_groups = changed
    result.unchanged_parameter_groups = unchanged
    result.frozen_parameter_hash_match = witness.digest(frozen_names) == frozen_before
    if not result.frozen_parameter_hash_match:
        moved = sorted(set(frozen_names) & set(changed))
        result.failures.append(f"frozen parameters changed: {moved or frozen_names}")
    expected_trainable = set(probe.trainable_parameter_names())
    stuck = sorted(expected_trainable - set(changed))
    if stuck:
        result.failures.append(f"trainable parameters did not move: {stuck}")

    # counters that must be nonzero regardless
    zero = [n for n in required_counters if auditors.counters.counts.get(n, 0) <= 0]
    if zero:
        result.failures.append(f"components that never updated: {zero}")

    if thresholds_path is not None and Path(thresholds_path).is_file():
        result.threshold_artifact_sha = hashlib.sha256(
            Path(thresholds_path).read_bytes()).hexdigest()
    elif mode == "live":
        result.failures.append("live smoke has no threshold artifact")
    else:
        result.notes.append("fixture mode: no threshold artifact, and none may be invented")

    result.verdict = "PASS" if not result.failures else "INVALID_BEFORE_TRAINING"
    return result


def write_receipt(result: SmokeResult, path: Path = OUT) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["record"] = "K=2 selective-supervision end-to-end treatment smoke"
    payload["meaning"] = (
        "Proves the treatment path exists: data -> labels -> loss -> optimizer -> "
        "parameter update. Says NOTHING about whether learning works.")
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="exercise the real treatment; refuses without earned artifacts")
    args = ap.parse_args()

    if not args.live:
        print("Fixture plumbing is exercised by tests/test_latent_treatment_smoke.py.\n"
              "This entry point runs only --live, against real artifacts.")
        return 0

    try:
        checks = require_live_artifacts()
    except LaunchGateError as exc:
        print(str(exc))
        return 1
    print("live artifacts verified:\n" + format_checks(checks))
    print("\nThe K=2 selective-supervision treatment is not implemented yet, so there "
          "is no probe to exercise.\nThis is the correct state: the harness is ready "
          "and refuses to fabricate one.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
