"""Runtime audit hooks: prove the intended treatment still exists mid-run.

The static launch gate checks that a run is *allowed* to start. These hooks check
that what started is still the experiment we think it is, every episode, until it
ends. EXP2B is the cautionary case: it would have passed a static gate cleanly and
then drifted to ~50/50 pole occupancy during resets, with nothing watching.

Failures here are HARD. On violation the run writes a classification artifact and
raises immediately, rather than completing and being discovered invalid afterwards.

Attachment is explicit and opt-in via a trainer attribute -- deliberately NOT a
PPOConfig field, because adding one invalidates all 541 preset-snapshot entries.
When nothing is attached every hook is a no-op, so an unaudited run pays only a
``getattr``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from rl.launch_gate import LaunchGateError, PoleAssignmentAuditor, UpdateCounters

ATTR = "launch_auditors"


@dataclass
class RuntimeAuditors:
    """Bundle attached to a trainer for the lifetime of a run."""
    z_to_pole: Mapping[int, str]
    opponent_to_pole: Mapping[int, str]
    pole: PoleAssignmentAuditor
    counters: UpdateCounters
    hard_fail: bool = True
    artifact_dir: Path | None = None
    classified: bool = False
    episodes_observed: int = 0
    unknown_opponents: dict[int, int] = field(default_factory=dict)
    # per-env evidence. env_index arrives at every episode close and was previously
    # discarded, which is why per-env reset coverage could not be proven from the
    # episode CSV -- that file carries no env identifier at all.
    per_env: dict[int, dict[str, Any]] = field(default_factory=dict)

    # -------------------------------------------------------------- internals
    def _classify(self, reason: str, detail: str) -> None:
        """Write a self-classification before dying, so the run explains itself."""
        self.classified = True
        if self.artifact_dir is None:
            return
        try:
            out = Path(self.artifact_dir)
            out.mkdir(parents=True, exist_ok=True)
            (out / "RUNTIME_AUDIT_FAILURE.json").write_text(json.dumps({
                "record": "runtime audit failure -- run terminated by the auditors",
                "classification": reason,
                "detail": detail,
                "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "episodes_observed": self.episodes_observed,
                "pole_telemetry": self.pole.telemetry(),
                "update_counts": dict(self.counters.counts),
                "verdict": "INVALID_TREATMENT -- not a scientific result",
            }, indent=2), encoding="utf-8")
        except OSError:
            pass                                   # never mask the real error

    def _fail(self, reason: str, detail: str) -> None:
        self._classify(reason, detail)
        if self.hard_fail:
            raise LaunchGateError(f"{reason}: {detail}")

    # ------------------------------------------------------------------ hooks
    def observe_episode_close(self, env_index: int, z: int, opponent_id: int) -> None:
        """One observation per episode boundary, per environment."""
        self.episodes_observed += 1
        env = self.per_env.setdefault(int(env_index), {
            "closes": 0, "z_seen": set(), "opponents_seen": set(),
            "poles_seen": set(), "mismatches": 0})
        env["closes"] += 1
        env["z_seen"].add(int(z))
        env["opponents_seen"].add(int(opponent_id))

        pole = self.opponent_to_pole.get(int(opponent_id))
        unknown = pole is None
        if unknown:
            self.unknown_opponents[int(opponent_id)] = (
                self.unknown_opponents.get(int(opponent_id), 0) + 1)
            pole = f"unknown({opponent_id})"
        env["poles_seen"].add(str(pole))
        before = len(self.pole.violations)
        self.pole.observe_reset(int(z), pole)
        drifted = len(self.pole.violations) > before
        if drifted or unknown:
            env["mismatches"] += 1
        # An unknown opponent always registers as drift too. Report the specific
        # cause rather than letting the generic one mask it.
        if unknown:
            self._fail("UNKNOWN_OPPONENT_ID",
                       f"env {env_index}: opponent {opponent_id} is not in the expected "
                       f"mapping {sorted(self.opponent_to_pole)}")
        elif drifted:
            self._fail("Z_POLE_ASSIGNMENT_DRIFT",
                       f"env {env_index}: {self.pole.violations[-1]}")

    def bump(self, name: str, n: int = 1) -> None:
        self.counters.bump(name, n)

    def require_no_pressure(self, name: str) -> None:
        try:
            self.counters.require_no_pressure(name)
        except LaunchGateError as exc:
            self._fail("RANKING_PRESSURE_ON_UNRESOLVED", str(exc))

    def require_runtime_clean(self, required: Iterable[str], min_resets: int = 1) -> None:
        """Call at the end of a run -- or a smoke -- before claiming validity."""
        # Most specific cause first: an unknown opponent also shows up as pole
        # drift, and reporting the generic symptom would bury the actual defect.
        if self.unknown_opponents:
            self._fail("UNKNOWN_OPPONENT_IDS",
                       f"opponent ids not in the expected mapping: {self.unknown_opponents}")
        try:
            self.pole.require_clean(min_resets=min_resets)
        except LaunchGateError as exc:
            self._fail("POLE_AUDIT_FAILED", str(exc))
        try:
            self.counters.require_nonzero(required)
        except LaunchGateError as exc:
            self._fail("TREATMENT_NEVER_UPDATED", str(exc))

    def per_env_evidence(self) -> dict:
        """Per-env observations. An env with N closes crossed N-1 resets."""
        out = {}
        for env_i, rec in sorted(self.per_env.items()):
            out[str(env_i)] = {
                "closes": rec["closes"],
                "resets": rec["closes"] - 1,
                "z": sorted(rec["z_seen"]),
                "opponents": sorted(rec["opponents_seen"]),
                "poles": sorted(rec["poles_seen"]),
                "mismatches": rec["mismatches"],
                "z_stable": len(rec["z_seen"]) == 1,
                "opponent_stable": len(rec["opponents_seen"]) == 1,
            }
        return out

    def coverage(self, *, expected_envs: int, min_resets: int) -> dict:
        """Fail-closed coverage summary. Absence is an error, never a default."""
        ev = self.per_env_evidence()
        failures: list[str] = []
        if not ev:
            failures.append("no per-env evidence was observed at all")
        if len(ev) != expected_envs:
            failures.append(f"{len(ev)}/{expected_envs} environments produced observations")
        low = {k: v["resets"] for k, v in ev.items() if v["resets"] < min_resets}
        if low:
            failures.append(
                f"{len(low)} env(s) below the required {min_resets} resets: "
                f"{dict(list(low.items())[:6])}")
        unstable_z = [k for k, v in ev.items() if not v["z_stable"]]
        unstable_op = [k for k, v in ev.items() if not v["opponent_stable"]]
        if unstable_z:
            failures.append(f"env(s) changed latent mid-run: {unstable_z[:6]}")
        if unstable_op:
            failures.append(f"env(s) changed opponent mid-run: {unstable_op[:6]}")
        total_mm = sum(v["mismatches"] for v in ev.values())
        if total_mm:
            failures.append(f"{total_mm} per-env assignment mismatches")
        return {
            "envs_observed": len(ev),
            "expected_envs": expected_envs,
            "min_resets_required": min_resets,
            "min_resets_observed": min((v["resets"] for v in ev.values()), default=None),
            "envs_with_reset": sum(1 for v in ev.values() if v["resets"] >= 1),
            "total_mismatches": total_mm,
            "per_env": ev,
            "failures": failures,
            "passed": not failures,
        }

    def telemetry(self) -> dict:
        return {"episodes_observed": self.episodes_observed,
                "pole": self.pole.telemetry(),
                "update_counts": dict(self.counters.counts),
                "unknown_opponents": dict(self.unknown_opponents),
                "per_env": self.per_env_evidence()}


# ------------------------------------------------------------------ module API

def attach(trainer: Any, z_to_pole: Mapping[int, str],
           opponent_to_pole: Mapping[int, int | str],
           *, hard_fail: bool = True, artifact_dir: Path | None = None) -> RuntimeAuditors:
    """Attach auditors to a trainer. Explicit, never implicit."""
    auditors = RuntimeAuditors(
        z_to_pole=dict(z_to_pole),
        opponent_to_pole={int(k): str(v) for k, v in opponent_to_pole.items()},
        pole=PoleAssignmentAuditor(expected=dict(z_to_pole)),
        counters=UpdateCounters(),
        hard_fail=hard_fail,
        artifact_dir=artifact_dir)
    setattr(trainer, ATTR, auditors)
    return auditors


def get(trainer: Any) -> RuntimeAuditors | None:
    return getattr(trainer, ATTR, None)


def observe_episode_close(trainer: Any, env_index: int, z: int, opponent_id: int) -> None:
    auditors = get(trainer)
    if auditors is not None:
        auditors.observe_episode_close(env_index, z, opponent_id)


def bump(trainer: Any, name: str, n: int = 1) -> None:
    auditors = get(trainer)
    if auditors is not None:
        auditors.bump(name, n)
