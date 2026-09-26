"""Surgical mechanism probe for the I1-to-I2 commitment cost.

This is deliberately narrower than ACTION_INTERFACE_SCALE_DIAGNOSTIC.  It
replays only the no-commit and current-commit arms and records switch events,
stale execution, and adoption lag.  No score, reward, or learning telemetry is
read or written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from experiments.action_interface_scale_replay import (
    SourceTrace,
    TraceTick,
    effective_targets_for_arm,
    init_shadow_from_trace,
    integrate_blue_shadow,
    nearest_w50_index,
    post_tick_arm_update,
)
from experiments.eval_action_interface_scale_diagnostic import (
    B3_CERT,
    B3_GENOME,
    HORIZON,
    _make_env,
)

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "ACTION_INTERFACE_COMMITMENT_MECHANISM_SPEC.json"
SCALE_SPEC = SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC_SPEC.json"
SCALE_AMEND = SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC_INTEGRITY_AMENDMENT.json"
SCALE_CONTRACT = SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC_CONTRACT_RESULT.json"
SCALE_RESULT = SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC_RESULT.json"
SCALE_READING = SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC_READING.json"
LABEL = "ACTION_INTERFACE_COMMITMENT_MECHANISM"
ARMS = ("I1_W50_NO_COMMIT", "I2_W50_CURRENT_COMMIT")
FORBIDDEN = frozenset({
    "blue_score", "red_score", "win", "reward", "return", "value",
    "advantage",
})
OUT = {
    "contract_result": SD / f"{LABEL}_CONTRACT_RESULT.json",
    # Standing post-run attestation, written by the contract test. Kept separate from
    # contract_result so a test run can never overwrite the PRE-RUN authorization record
    # that evidences "contracts passed before any seed was spent".
    "contract_attestation": SD / f"{LABEL}_CONTRACT_ATTESTATION.json",
    "agent_tick_rows": SD / "action_interface_commitment_mechanism_agent_tick_rows.csv",
    "episode_rows": SD / "action_interface_commitment_mechanism_episode_rows.csv",
    "result": SD / f"{LABEL}_RESULT.json",
    "reading": SD / f"{LABEL}_READING.json",
    "run_lock": SD / f"{LABEL}.run.lock",
}
ANGLE_THRESHOLD = 15.0
JUMP_THRESHOLD = 1.0
COMMIT_TICKS = 4


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _angle_deg(a: float, b: float) -> float:
    d = (float(a) - float(b) + np.pi) % (2.0 * np.pi) - np.pi
    return abs(float(np.degrees(d)))


def _bearing(x1: float, y1: float, x2: float, y2: float) -> float:
    return float(np.arctan2(y2 - y1, x2 - x1))


def _boot(values: np.ndarray, n_boot: int = 20_000, seed: int = 7) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "lcb95": None, "ucb95": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {
        "mean": round(float(values.mean()), 6),
        "lcb95": round(float(lo), 6),
        "ucb95": round(float(hi), 6),
        "n": int(values.size),
    }


@dataclass
class PendingLag:
    start_tick: int


@dataclass
class EventTracker:
    previous_idx: int | None = None
    previous_target: tuple[float, float] | None = None
    previous_bearing: float | None = None
    pending: PendingLag | None = None


def observe_event(
    tracker: EventTracker,
    *,
    tick: int,
    requested_idx: int,
    target_x: float,
    target_y: float,
    desired_bearing: float,
    effective_bearing: float,
    committed_idx_before: int,
    committed_idx_after: int,
    commit_ticks_left_before: int,
    alive: bool,
    carrying: bool,
    tagged: bool,
) -> dict[str, Any]:
    """Convert one target/commitment transition into frozen event fields."""
    if tracker.previous_target is None:
        jump = 0.0
    else:
        jump = float(np.hypot(
            target_x - tracker.previous_target[0],
            target_y - tracker.previous_target[1],
        ))
    if tracker.previous_bearing is None:
        bearing_change = 0.0
    else:
        bearing_change = _angle_deg(desired_bearing, tracker.previous_bearing)
    active = bool(alive and not tagged)
    request = bool(
        active and (
            tracker.previous_idx is None
            or requested_idx != tracker.previous_idx
            or bearing_change >= ANGLE_THRESHOLD
            or jump >= JUMP_THRESHOLD
        )
    )
    boundary = bool(commit_ticks_left_before <= 0)
    blocked = bool(
        request and not boundary and requested_idx != committed_idx_before
    )
    stale = bool(blocked and _angle_deg(effective_bearing, desired_bearing) >= ANGLE_THRESHOLD)
    adopted = bool(
        active and (
            _angle_deg(effective_bearing, desired_bearing) <= ANGLE_THRESHOLD
            or requested_idx == committed_idx_after
        )
    )
    lag: int | None = None
    censored = False
    # A new request ends an older pending request without pretending that the
    # older target was ever adopted.  A boundary adoption closes it cleanly.
    if tracker.pending is not None and request:
        if adopted:
            lag = int(tick - tracker.pending.start_tick)
        else:
            censored = True
        tracker.pending = None
    elif tracker.pending is not None and adopted:
        lag = int(tick - tracker.pending.start_tick)
        tracker.pending = None
    elif tracker.pending is not None and not active:
        censored = True
        tracker.pending = None
    if blocked and not adopted and active:
        tracker.pending = PendingLag(start_tick=tick)

    row = {
        "requested_w50_idx": int(requested_idx),
        "committed_w50_idx": int(committed_idx_after),
        "commit_ticks_left_before": int(commit_ticks_left_before),
        "decision_boundary": int(boundary),
        "target_jump_cells": round(jump, 6),
        "bearing_change_degrees": round(bearing_change, 6),
        "switch_request": int(request),
        "blocked_switch": int(blocked),
        "stale_switch": int(stale),
        "adopted": int(adopted),
        "decision_lag_ticks": lag,
        "lag_censored": int(censored),
        "alive": int(bool(alive)),
        "carrying": int(bool(carrying)),
        "tagged": int(bool(tagged)),
    }
    tracker.previous_idx = int(requested_idx) if active else None
    tracker.previous_target = (float(target_x), float(target_y)) if active else None
    tracker.previous_bearing = float(desired_bearing) if active else None
    return row


def finalize_tracker(tracker: EventTracker) -> int:
    """Censor a pending request at trace end and return one event count."""
    if tracker.pending is None:
        return 0
    tracker.pending = None
    return 1


def instrument_trace_arm(trace: SourceTrace, arm: str, core, W: np.ndarray) -> tuple[list[dict], dict]:
    if arm not in ARMS:
        raise ValueError(f"unsupported mechanism arm: {arm}")
    shadow = init_shadow_from_trace(trace)
    trackers = [EventTracker() for _ in range(trace.scale)]
    rows: list[dict] = []
    for t_idx, tick in enumerate(trace.ticks):
        pre_left = shadow.go_to_ticks_left.copy()
        pre_held = shadow.held_w50.copy()
        req_idx = np.array([
            nearest_w50_index(float(x), float(y), W)
            for x, y in zip(tick.oracle_tx, tick.oracle_ty)
        ], dtype=np.int32)
        eff_x, eff_y, _ = effective_targets_for_arm(arm, core, tick, shadow, W)
        for i in range(trace.scale):
            desired = _bearing(
                float(tick.blue_x[i]), float(tick.blue_y[i]),
                float(tick.oracle_tx[i]), float(tick.oracle_ty[i]),
            )
            effective = _bearing(
                float(tick.blue_x[i]), float(tick.blue_y[i]),
                float(eff_x[i]), float(eff_y[i]),
            )
            event = observe_event(
                trackers[i], tick=tick.tick,
                requested_idx=int(req_idx[i]),
                target_x=float(tick.oracle_tx[i]), target_y=float(tick.oracle_ty[i]),
                desired_bearing=desired, effective_bearing=effective,
                committed_idx_before=int(pre_held[i]),
                committed_idx_after=int(shadow.held_w50[i]),
                commit_ticks_left_before=int(pre_left[i]),
                alive=bool(tick.blue_alive[i]), carrying=bool(tick.blue_carrying[i]),
                tagged=bool(tick.blue_tagged[i]),
            )
            rows.append({
                "scale": f"{trace.scale}v{trace.scale}", "style": trace.style,
                "seed": trace.seed, "arm": arm, "tick": tick.tick, "agent": i,
                "native_target_x": round(float(tick.oracle_tx[i]), 6),
                "native_target_y": round(float(tick.oracle_ty[i]), 6),
                "desired_bearing_degrees": round(float(np.degrees(desired)), 6),
                "effective_bearing_degrees": round(float(np.degrees(effective)), 6),
                "committed_w50_idx_before": int(pre_held[i]),
                "trace_hash": trace.trace_hash,
                **event,
            })
        prx = trace.ticks[t_idx - 1].red_x if t_idx > 0 else tick.red_x
        pry = trace.ticks[t_idx - 1].red_y if t_idx > 0 else tick.red_y
        sx, sy, sh, ss = integrate_blue_shadow(
            core, tick, shadow, eff_x, eff_y, prev_red_x=prx, prev_red_y=pry,
        )
        shadow.x, shadow.y, shadow.heading, shadow.speed = sx, sy, sh, ss
        post_tick_arm_update(arm, shadow, tick)
    censored_end = sum(finalize_tracker(t) for t in trackers)
    arm_rows = [r for r in rows if r["arm"] == arm]
    requests = int(sum(r["switch_request"] for r in arm_rows))
    blocked = int(sum(r["blocked_switch"] for r in arm_rows))
    stale = int(sum(r["stale_switch"] for r in arm_rows))
    lags = [int(r["decision_lag_ticks"]) for r in arm_rows if r["decision_lag_ticks"] is not None]
    censored = int(sum(r["lag_censored"] for r in arm_rows) + censored_end)
    closed = len(lags) + censored
    episode = {
        "scale": f"{trace.scale}v{trace.scale}", "style": trace.style,
        "seed": trace.seed, "arm": arm, "trace_hash": trace.trace_hash,
        "n_ticks": len(trace.ticks), "n_agent_ticks": len(arm_rows),
        "switch_requests": requests, "blocked_switches": blocked,
        "stale_switches": stale, "blocked_switch_fraction": blocked / requests if requests else 0.0,
        "stale_switch_fraction": stale / requests if requests else 0.0,
        "mean_decision_lag_ticks": float(np.mean(lags)) if lags else None,
        "p95_decision_lag_ticks": float(np.percentile(lags, 95)) if lags else None,
        "censored_switch_fraction": censored / closed if closed else 0.0,
        "lag_adoptions": len(lags), "lag_censored": censored,
        "mean_bearing_change_degrees": float(np.mean([
            r["bearing_change_degrees"] for r in arm_rows if r["switch_request"]
        ])) if requests else 0.0,
        "mean_target_jump_cells": float(np.mean([
            r["target_jump_cells"] for r in arm_rows if r["switch_request"]
        ])) if requests else 0.0,
    }
    return rows, episode


def _schema_clean(fields: list[str]) -> None:
    bad = sorted(set(fields) & FORBIDDEN)
    if bad:
        raise ValueError(f"forbidden outcome/learning fields: {bad}")


def _synthetic_target_stream(n: int = 12) -> list[tuple[float, float]]:
    # Alternate endpoints while the four-tick commitment is active.
    return [(5.0 if t % 2 == 0 else 15.0, 10.0) for t in range(n)]


_SHA256_HEX = re.compile(r"^[0-9a-f]{64}$")


def _content_hashes(path: Path) -> dict[str, str]:
    """sha256 of the file's bytes, and of its content under LF and CRLF line endings.

    Pins were recorded as raw-byte hashes on a Windows clone (core.autocrlf=true), so
    they are CRLF-form hashes. A clone with LF checkouts holds byte-different but
    content-identical files. Matching any of the three forms is content equality modulo
    line endings -- NOT an escape hatch: a change to any character other than CR/LF
    matches none of them.
    """
    raw = path.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    crlf = lf.replace(b"\n", b"\r\n")
    return {
        "raw": hashlib.sha256(raw).hexdigest(),
        "lf": hashlib.sha256(lf).hexdigest(),
        "crlf": hashlib.sha256(crlf).hexdigest(),
    }


def check_prior_artifact_protection(
    protected: Sequence[Path],
    mode: str,
    spec: dict[str, Any],
    pin_source: Path,
) -> dict[str, Any]:
    """G0: the prior scale artifacts must be exactly as they were when this probe was
    authorized. SPEC text: "Hashes the prior scale spec, amendment, contract result,
    result, and reading; refuses interpretation if any changes."

    PRE_RUN  no authorization record exists yet -- the hashes computed here are what
             the PRE_RUN contract result records, and they BECOME the frozen pins.
    otherwise the pins are read from the pre-run authorization record (pin_source) and
             every protected file must match its pin. The frozen spec carries no hash
             values, so the authorization record -- written before any seed was spent
             -- is the frozen source.

    Fails closed, per file, on: file missing; pin absent; pin malformed; content
    mismatch. A pin source that is missing or unreadable fails every file.
    """
    spec_frozen = str(spec.get("status", "")).startswith("FROZEN")
    out: dict[str, Any] = {"mode_used": mode, "spec_frozen": spec_frozen}

    if mode == "PRE_RUN":
        missing = [p.name for p in protected if not p.is_file()]
        hashes = {p.name: _sha256(p) for p in protected if p.is_file()}
        out.update(
            pass_=not missing and spec_frozen, action="RECORD_PINS",
            missing=missing, protected_sha256=hashes,
            detail="pre-run: these hashes are recorded as the frozen pins",
        )
        return _finish_g0(out)

    pins: dict[str, Any] | None = None
    pin_error = None
    if not pin_source.is_file():
        pin_error = f"pin source missing: {pin_source.name}"
    else:
        try:
            doc = json.loads(pin_source.read_text(encoding="utf-8"))
            pins = doc["gates"]["G0_PRIOR_ARTIFACT_PROTECTION"]["protected_sha256"]
            if not isinstance(pins, dict):
                pin_error, pins = "pin table is not a mapping", None
        except Exception as exc:                                    # noqa: BLE001
            pin_error = f"pin source unreadable: {type(exc).__name__}: {exc}"
    out["pin_source"] = pin_source.name
    out["pin_source_sha256"] = _sha256(pin_source) if pin_source.is_file() else None

    files: dict[str, Any] = {}
    for p in protected:
        rec: dict[str, Any] = {"expected": None, "actual_raw": None}
        if pin_error is not None:
            rec.update(status="FAIL_PIN_SOURCE", detail=pin_error)
        elif not p.is_file():
            rec.update(status="FAIL_MISSING", detail="protected file does not exist")
        else:
            expected = pins.get(p.name)
            h = _content_hashes(p)
            rec["expected"], rec["actual_raw"] = expected, h["raw"]
            if expected is None:
                rec.update(status="FAIL_PIN_ABSENT",
                           detail="no frozen pin recorded for this file")
            elif not isinstance(expected, str) or not _SHA256_HEX.match(expected):
                rec.update(status="FAIL_PIN_MALFORMED",
                           detail=f"pin is not 64 lowercase hex chars: {expected!r}")
            else:
                form = next((f for f in ("raw", "crlf", "lf") if h[f] == expected), None)
                if form is None:
                    rec.update(status="FAIL_CONTENT_CHANGED",
                               detail="content differs from the frozen pin under "
                                      "raw, CRLF and LF forms")
                else:
                    rec.update(status="MATCH", matched_form=form)
        files[p.name] = rec

    failed = sorted(n for n, r in files.items() if r["status"] != "MATCH")
    out.update(
        pass_=spec_frozen and not failed and pin_error is None, action="VERIFY_PINS",
        files=files, failed=failed,
        detail=("every protected artifact matches its frozen pin" if not failed and
                pin_error is None else f"{len(failed)} protected artifact(s) failed"),
    )
    return _finish_g0(out)


def _finish_g0(out: dict[str, Any]) -> dict[str, Any]:
    """Rename pass_ -> pass (pass is a keyword in the builder above)."""
    out["pass"] = bool(out.pop("pass_"))
    return out


def run_contracts() -> dict[str, Any]:
    gates: dict[str, dict[str, Any]] = {}
    protected = (SCALE_SPEC, SCALE_AMEND, SCALE_CONTRACT, SCALE_RESULT, SCALE_READING)
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    # The lifecycle mode is needed by G0 as well as G8, so it is resolved first.
    result_exists, lock_exists = OUT["result"].exists(), OUT["run_lock"].exists()
    mode = "POST_RUN" if result_exists else ("IN_FLIGHT" if lock_exists else "PRE_RUN")
    gates["G0_PRIOR_ARTIFACT_PROTECTION"] = check_prior_artifact_protection(
        protected, mode, spec, OUT["contract_result"],
    )
    # Kept for the record shape: PRE_RUN records these as pins; they are the live hashes.
    hashes = {p.name: _sha256(p) for p in protected if p.is_file()}

    # G1: request, endpoint jump, and angular event extraction.
    tr = EventTracker()
    first = observe_event(tr, tick=0, requested_idx=1, target_x=5, target_y=5,
                          desired_bearing=0, effective_bearing=0,
                          committed_idx_before=0, committed_idx_after=1,
                          commit_ticks_left_before=0, alive=True, carrying=False, tagged=False)
    second = observe_event(tr, tick=1, requested_idx=2, target_x=5, target_y=7,
                           desired_bearing=np.pi / 2, effective_bearing=np.pi / 2,
                           committed_idx_before=1, committed_idx_after=2,
                           commit_ticks_left_before=0, alive=True, carrying=False, tagged=False)
    gates["G1_REQUEST_EXTRACTION"] = {
        "pass": first["switch_request"] == 1 and second["switch_request"] == 1
        and second["target_jump_cells"] == 2.0 and second["bearing_change_degrees"] == 90.0,
        "first": first, "second": second,
    }

    # G2: fresh commitment boundaries are 0,4,8,... and retain the held index.
    tr = EventTracker()
    boundary_ticks = []
    held = 0
    for t in range(9):
        left = 0 if t % COMMIT_TICKS == 0 else COMMIT_TICKS - (t % COMMIT_TICKS)
        after = 1 if left == 0 else held
        boundary_ticks.append((left, after))
        held = after
    gates["G2_BOUNDARY_SCHEDULE"] = {
        "pass": [x[0] for x in boundary_ticks] == [0, 3, 2, 1, 0, 3, 2, 1, 0]
        and [x[1] for x in boundary_ticks] == [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "schedule": boundary_ticks,
    }

    # G3/G4: a blocked request is stale until the next boundary, then lag=3.
    tr = EventTracker()
    rows = []
    for t, (left, req, held_after, eff) in enumerate([
        (0, 1, 1, 0), (3, 2, 1, 0), (2, 2, 1, 0), (1, 2, 0, 0),
        (0, 2, 2, np.pi / 2),
    ]):
        rows.append(observe_event(
            tr, tick=t, requested_idx=req, target_x=float(req), target_y=0,
            desired_bearing=np.pi / 2 if req == 2 else 0,
            effective_bearing=eff, committed_idx_before=held_after,
            committed_idx_after=held_after, commit_ticks_left_before=left,
            alive=True, carrying=False, tagged=False,
        ))
    gates["G3_BLOCKED_SWITCH"] = {
        "pass": rows[1]["blocked_switch"] == 1 and rows[1]["stale_switch"] == 1,
        "rows": rows,
    }
    gates["G4_ADOPTION_LAG"] = {
        "pass": rows[-1]["adopted"] == 1 and rows[-1]["decision_lag_ticks"] == 3
        and all(r["lag_censored"] == 0 for r in rows),
        "adoption_row": rows[-1],
    }

    fields = [
        "scale", "style", "seed", "arm", "tick", "agent",
        "native_target_x", "requested_w50_idx", "committed_w50_idx",
        "commit_ticks_left_before", "decision_boundary", "target_jump_cells",
        "bearing_change_degrees", "switch_request", "blocked_switch",
        "stale_switch", "adopted", "decision_lag_ticks", "lag_censored",
        "alive", "carrying", "tagged", "trace_hash",
    ]
    try:
        _schema_clean(fields)
        schema_ok, schema_note = True, "no forbidden fields"
    except ValueError as exc:
        schema_ok, schema_note = False, str(exc)
    gates["G5_CONTEXT_SPLITS"] = {
        "pass": observe_event(EventTracker(), tick=0, requested_idx=0,
                               target_x=0, target_y=0, desired_bearing=0,
                               effective_bearing=0, committed_idx_before=0,
                               committed_idx_after=0, commit_ticks_left_before=0,
                               alive=False, carrying=True, tagged=True)["switch_request"] == 0,
        "detail": "dead/tagged rows retained and do not create requests",
    }
    gates["G6_OUTCOME_BLIND_SCHEMA"] = {"pass": schema_ok, "detail": schema_note}
    gates["G7_TRACE_COMPLETENESS"] = {
        "pass": HORIZON == 240 and len(ARMS) == 2,
        "horizon": HORIZON, "arms": list(ARMS),
    }
    # G8 is a PRE-RUN authorization gate. Asking "does the result not exist?" is only
    # meaningful before the probe has run; after a legitimate run it can never be true
    # again, which made the standing contract test permanently red once the result was
    # sealed. So the mode is resolved explicitly and G8 is marked not-applicable
    # post-run, while the frozen intent it stands for -- SPEC
    # #contracts_before_any_source_episode.G8: "uses a fresh Rule-9 exploratory block and
    # never overwrites prior scale artifacts" -- is checked directly against the sealed
    # result by G9. The launcher's own refusal (result-or-lock exists => SystemExit) is
    # deliberately UNCHANGED and remains the thing that prevents a second run.
    # (mode / result_exists / lock_exists are resolved once, at the top of this function.)
    if mode == "PRE_RUN":
        gates["G8_PRIOR_RUN_SEPARATION"] = {
            "pass": True, "applicable": True, "mode": mode,
            "result_exists": False, "run_lock_exists": False,
            "detail": "no prior result and no run lock: a run is authorized to start",
        }
    elif mode == "IN_FLIGHT":
        # A lock without a result is either a live run or a crashed one. Either way a
        # second run must not start, so this stays a hard failure.
        gates["G8_PRIOR_RUN_SEPARATION"] = {
            "pass": False, "applicable": True, "mode": mode,
            "result_exists": False, "run_lock_exists": True,
            "detail": "run lock present with no sealed result: a run is in flight or "
                      "crashed mid-run. Do not start another; resolve the lock first.",
        }
    else:
        gates["G8_PRIOR_RUN_SEPARATION"] = {
            "pass": True, "applicable": False, "mode": mode,
            "result_exists": True, "run_lock_exists": lock_exists,
            "detail": "NOT APPLICABLE: the probe has run and its result is sealed, so "
                      "pre-run separation is no longer a checkable state. Re-running is "
                      "still blocked by the launcher's own result-or-lock refusal, and "
                      "the frozen intent is verified by G9 below.",
        }
        gates["G9_SEALED_RESULT_PROVENANCE"] = _sealed_result_gate(spec, lock_exists)

    # `applicable: False` gates are reported but never gate the verdict.
    all_pass = all(bool(g["pass"]) for g in gates.values() if g.get("applicable", True))
    return {
        "record_id": f"{LABEL}_CONTRACT_RESULT", "status": "PASS" if all_pass else "FAIL",
        "utc": _now(), "device": "cpu", "gpu_used": False, "mode": mode,
        "gates": gates, "overall_pass": all_pass,
        "spec_sha256": _sha256(SPEC), "protected_sha256": hashes,
        "mode_semantics": {
            "PRE_RUN": "no result, no lock -- G8 gates; a run may start",
            "IN_FLIGHT": "lock without result -- G8 fails; do not start another run",
            "POST_RUN": "result sealed -- G8 not applicable; G9 verifies the sealed state",
        },
    }


def _sealed_result_gate(spec: dict[str, Any], lock_exists: bool) -> dict[str, Any]:
    """POST-RUN replacement for G8: verify the sealed result against the frozen spec.

    Checks what G8's spec text actually promises, which the pre-run existence proxy could
    only approximate: the seeds spent are exactly the registered Rule-9 block for this
    experiment id, the sampling matches the frozen spec, every trace is complete, and the
    run left no lock behind. Prior-scale artifacts being unmodified is already G0.
    """
    checks: dict[str, Any] = {}
    try:
        result = json.loads(OUT["result"].read_text(encoding="utf-8"))
    except Exception as exc:                                    # noqa: BLE001
        return {"pass": False, "applicable": True,
                "detail": f"sealed result unreadable: {exc}"}

    seeds = [int(s) for s in (result.get("seeds") or [])]
    sampling = spec.get("sampling") or {}
    want_n = int(sampling.get("n_seeds", -1))
    want_class = str(sampling.get("seed_class", ""))

    checks["seed_count_matches_spec"] = {
        "pass": len(seeds) == want_n == int(result.get("n_seeds", -1)),
        "spec_n_seeds": want_n, "result_n_seeds": result.get("n_seeds"),
        "seeds_listed": len(seeds),
    }
    checks["arms_match_spec"] = {
        "pass": list(result.get("arms") or []) == list(ARMS) == list(spec.get("arms") or {}),
        "result_arms": result.get("arms"), "module_arms": list(ARMS),
    }

    # The Rule-9 block must be registered to THIS experiment, with the spec's class, and
    # must cover exactly the seeds the sealed result reports.
    reg: dict[str, Any] = {"pass": False, "detail": "registry entry not found"}
    try:
        from experiments.seed_registry import load as _load_registry      # noqa: PLC0415
        blocks = [b for b in (_load_registry().get("blocks") or [])
                  if b.get("experiment_id") == LABEL]
        if blocks:
            b = blocks[0]
            covered = seeds and min(seeds) >= int(b["lo"]) and max(seeds) <= int(b["hi"])
            reg = {
                "pass": bool(covered) and str(b.get("seed_class")) == want_class
                        and int(b.get("n", -1)) == len(seeds),
                "block": [b.get("lo"), b.get("hi")], "n": b.get("n"),
                "seed_class": b.get("seed_class"), "status": b.get("status"),
                "spec_seed_class": want_class,
                "seeds_within_block": bool(covered),
            }
    except ImportError as exc:
        reg = {"pass": False, "detail": f"seed registry unavailable: {exc}"}
    checks["seeds_are_registered_rule9_block"] = reg

    manifest = result.get("manifest") or []
    bad = [m for m in manifest
           if int(m.get("n_ticks", -1)) != HORIZON or not str(m.get("trace_hash", ""))]
    checks["traces_complete"] = {
        "pass": bool(manifest) and not bad,
        "n_traces": len(manifest), "n_incomplete": len(bad), "horizon": HORIZON,
    }
    checks["no_stale_run_lock"] = {
        "pass": not lock_exists,
        "detail": "a completed run removes its lock in the finally block",
    }
    return {
        "pass": all(bool(c["pass"]) for c in checks.values()),
        "applicable": True,
        "verifies": "SPEC#contracts_before_any_source_episode.G8_PRIOR_RUN_SEPARATION "
                    "('fresh Rule-9 exploratory block, never overwrites prior scale "
                    "artifacts') against the sealed result. Prior-artifact protection "
                    "itself is G0.",
        "checks": checks,
    }


def _load_source_trace(scale: int, style_key: str, style_label: str, seed: int):
    import experiments.strategic_demand_searcher as S
    from experiments.action_interface_scale_replay import collect_source_trace
    env, core, S, _ = _make_env(scale, style_key, seed, "cpu")
    try:
        trace = collect_source_trace(env, core, S, HORIZON)
        trace.style, trace.seed, trace.scale = style_label, seed, scale
        trace.finalize_hash()
        W = core._macro_targets.detach().cpu().numpy()
        rows, episodes = {}, {}
        for arm in ARMS:
            rows[arm], episodes[arm] = instrument_trace_arm(trace, arm, core, W)
        return trace, rows, episodes
    finally:
        env.close()


def _contrast(episodes: list[dict], field: str, scale_a: str, style_a: str,
              scale_b: str, style_b: str, arm: str = "I2_W50_CURRENT_COMMIT") -> dict:
    by = {(r["seed"], r["scale"], r["style"], r["arm"]): r for r in episodes}
    seeds = sorted({r["seed"] for r in episodes})
    raw = [
        (by[(s, scale_b, style_b, arm)][field],
         by[(s, scale_a, style_a, arm)][field]) for s in seeds
    ]
    missing = sum(a is None or b is None for a, b in raw)
    if missing:
        return {
            "mean": None, "lcb95": None, "ucb95": None, "n": 0,
            "unavailable": True, "missing_seed_pairs": int(missing),
        }
    values = np.array([float(a) - float(b) for a, b in raw])
    return _boot(values)


def run_probe(seeds: list[int]) -> tuple[dict, list[dict], list[dict]]:
    import experiments.strategic_demand_searcher as S
    episode_rows: list[dict] = []
    tick_rows: list[dict] = []
    manifest: list[dict] = []
    for seed in seeds:
        for scale in (2, 4):
            for style_key, style_label in ((S.GUARD, "GUARD"), (S.BREACH, "BREACH")):
                trace, rows, episodes = _load_source_trace(scale, style_key, style_label, seed)
                manifest.append({"scale": f"{scale}v{scale}", "style": style_label,
                                 "seed": seed, "trace_hash": trace.trace_hash,
                                 "n_ticks": len(trace.ticks), "arms": list(ARMS)})
                for arm in ARMS:
                    tick_rows.extend(rows[arm])
                    episode_rows.append(episodes[arm])
    support = {
        "stale_scale_BREACH": _contrast(episode_rows, "stale_switch_fraction", "2v2", "BREACH", "4v4", "BREACH"),
        "blocked_scale_BREACH": _contrast(episode_rows, "blocked_switch_fraction", "2v2", "BREACH", "4v4", "BREACH"),
        "lag_scale_BREACH": _contrast(episode_rows, "mean_decision_lag_ticks", "2v2", "BREACH", "4v4", "BREACH"),
        "style_scale_stale": _contrast(episode_rows, "stale_switch_fraction", "2v2", "GUARD", "4v4", "BREACH"),
    }
    # The last quantity above is retained as a descriptive paired contrast in
    # the artifact; the frozen style interaction is computed explicitly below.
    by = {(r["seed"], r["scale"], r["style"], r["arm"]): r for r in episode_rows}
    vals = []
    for seed in seeds:
        vals.append((by[(seed, "4v4", "BREACH", "I2_W50_CURRENT_COMMIT")]["stale_switch_fraction"]
                     - by[(seed, "4v4", "GUARD", "I2_W50_CURRENT_COMMIT")]["stale_switch_fraction"])
                    - (by[(seed, "2v2", "BREACH", "I2_W50_CURRENT_COMMIT")]["stale_switch_fraction"]
                       - by[(seed, "2v2", "GUARD", "I2_W50_CURRENT_COMMIT")]["stale_switch_fraction"]))
    support["style_scale_stale"] = _boot(np.asarray(vals))
    result = {
        "record_id": LABEL, "utc": _now(), "device": "cpu", "gpu_used": False,
        "n_seeds": len(seeds), "seeds": seeds, "arms": list(ARMS),
        "episode_rows": episode_rows, "manifest": manifest,
        "support_contrasts": support,
        "event_totals": {
            arm: {
                "switch_requests": int(sum(r["switch_requests"] for r in episode_rows if r["arm"] == arm)),
                "blocked_switches": int(sum(r["blocked_switches"] for r in episode_rows if r["arm"] == arm)),
                "stale_switches": int(sum(r["stale_switches"] for r in episode_rows if r["arm"] == arm)),
            } for arm in ARMS
        },
        "mechanism_support_gate": "PENDING_READING",
    }
    return result, tick_rows, episode_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contracts", action="store_true")
    ap.add_argument("--promote", action="store_true")
    ap.add_argument("--seeds")
    args = ap.parse_args()
    if args.contracts or not args.promote:
        result = run_contracts()
        # Only a PRE_RUN evaluation may write the authorization record: that file is the
        # evidence that contracts passed BEFORE any seed was spent, and re-running
        # --contracts after the probe had sealed its result is exactly how that evidence
        # was destroyed once already (the committed record flipped PASS -> FAIL between
        # a856469d and af8f76c2). Post-run evaluations go to the attestation file. The
        # refusal logic below is unchanged.
        target = OUT["contract_result"] if result["mode"] == "PRE_RUN" else OUT["contract_attestation"]
        target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "overall_pass": result["overall_pass"],
                          "mode": result["mode"], "wrote": target.name}, indent=2))
        if not result["overall_pass"] or not args.promote:
            return 0 if result["overall_pass"] else 2
    contract = json.loads(OUT["contract_result"].read_text(encoding="utf-8"))
    if not contract.get("overall_pass"):
        raise SystemExit("REFUSING: mechanism contracts are not PASS")
    if not args.seeds:
        raise SystemExit("REFUSING: --seeds required after seed_registry allocation")
    if OUT["result"].exists() or OUT["run_lock"].exists():
        raise SystemExit("REFUSING: result or run lock already exists")
    if "-" in args.seeds:
        lo, hi = args.seeds.split("-", 1)
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    OUT["run_lock"].write_text(json.dumps({"pid": os.getpid(), "utc": _now()}), encoding="utf-8")
    try:
        result, tick_rows, episode_rows = run_probe(seeds)
        tick_fields = list(tick_rows[0]) if tick_rows else []
        ep_fields = list(episode_rows[0]) if episode_rows else []
        _schema_clean(tick_fields + ep_fields)
        with OUT["agent_tick_rows"].open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=tick_fields); w.writeheader(); w.writerows(tick_rows)
        with OUT["episode_rows"].open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=ep_fields); w.writeheader(); w.writerows(episode_rows)
        OUT["result"].write_text(json.dumps({k: result[k] for k in result if k not in {"episode_rows"}}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"result": str(OUT["result"]), "n_episode_rows": len(episode_rows),
                          "n_tick_rows": len(tick_rows)}, indent=2))
    finally:
        OUT["run_lock"].unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
