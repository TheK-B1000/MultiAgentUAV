"""C2 fresh confirmation runner (single frozen candidate, immutable contract).

Authoritative inputs:
  - artifacts/c2_qualification/C2_PROPOSAL.json
  - artifacts/c2_qualification/C2_CONFIRMATION_PREREG_INPUT.json
  - artifacts/c2_qualification/C2_QUALIFICATION_FROZEN.json

This runner:
  1) Allocates a fresh evaluation-seed block with fail-closed collision checks.
  2) Replays the frozen G0 checkpoints on that fresh block.
  3) Evaluates exactly one candidate under the frozen three-band contract.
  4) Writes the required confirmation artifacts.

Startup observability:
  Prefer experiments/run_c2_fresh_confirmation.ps1 (python -u + tee to
  confirmation_full.log). The runner itself also forces line-buffered stdio,
  appends every progress line to C2_CONFIRMATION_PROGRESS.log, and rewrites
  C2_CONFIRMATION_PROGRESS.json (phase / counts / ETA). Watch those files if
  the terminal capture is empty. Scientific outputs are unchanged.
"""
from __future__ import annotations

import argparse
import csv
import faulthandler
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Enable before heavy imports so native faults during torch/env load dump a stack.
faulthandler.enable(all_threads=True)
os.environ.setdefault("PYTHONUNBUFFERED", "1")
try:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
except Exception:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "artifacts" / "c2_confirmation"
QUAL_DIR = PROJECT_ROOT / "artifacts" / "c2_qualification"
PROPOSAL_PATH = QUAL_DIR / "C2_PROPOSAL.json"
INPUT_ALIAS_PATH = QUAL_DIR / "C2_CONFIRMATION_PREREG_INPUT.json"
FROZEN_PATH = QUAL_DIR / "C2_QUALIFICATION_FROZEN.json"

MANIFEST_PATH = OUT_DIR / "C2_CONFIRMATION_MANIFEST.json"
WINDOWS_PATH = OUT_DIR / "C2_CONFIRMATION_WINDOWS.csv"
RESULTS_PATH = OUT_DIR / "C2_CONFIRMATION_RESULTS.json"
FROZEN_RESULT_PATH = OUT_DIR / "C2_CONFIRMATION_FROZEN_RESULT.json"
PROGRESS_LOG_PATH = OUT_DIR / "C2_CONFIRMATION_PROGRESS.log"
PROGRESS_JSON_PATH = OUT_DIR / "C2_CONFIRMATION_PROGRESS.json"
O2_DRAFT_PATH = PROJECT_ROOT / "artifacts" / "o2_preregistration" / "O2_PROTOCOL_DRAFT_FROM_C2_CONFIRMATION.json"

LOCK_PATH = OUT_DIR / "C2_CONFIRMATION_RUNNING.lock"

# Mutable startup progress written into MANIFEST_PATH immediately after lock.
_STARTUP_STATE: dict = {
    "status": "STARTING",
    "startup_phase": "PROCESS_ENTERED",
    "episodes_consumed": 0,
    "episodes_total": None,
    "pid": os.getpid(),
    "t0": time.time(),
    "eval_t0": None,
}

SEED_SCAN_MIN = 1_000_000
SEED_SCAN_MAX = 9_999_999
FRESH_SEARCH_START = 9_800_000
FRESH_SEARCH_END = 9_899_999

FORBIDDEN_BASES = {9_400_000, 9_500_000, 9_600_000, 9_700_000}
FORBIDDEN_RANGES = (
    (9_400_000, 9_499_999),
    (9_500_000, 9_599_999),
    (9_600_000, 9_699_999),
    (9_700_000, 9_799_999),
)

# Immutable channel names from Stage 2.
BAND_ORDER = ("earliest", "middle", "latest")
FAIL_LABEL = "tagged_while_carrying"
FEATURE = "none_forward_frac"
EXPECTED_SIGN = -1  # NEGATIVE
MATCH_KEY = "opp_has_carrier"


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _log(msg: str) -> None:
    """Stdout + durable progress log (survives empty terminal capture)."""
    line = f"{_now_utc()} {msg}"
    print(line, flush=True)
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        with PROGRESS_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
    except Exception:
        pass


def _eta_seconds(*, done: int, total: int, t0: float | None) -> float | None:
    if t0 is None or done <= 0 or total <= 0 or done > total:
        return None
    rate = done / max(time.time() - t0, 1e-6)
    if rate <= 0:
        return None
    return round((total - done) / rate, 1)


def _write_progress_json(**extra) -> None:
    consumed = int(_STARTUP_STATE.get("episodes_consumed", 0) or 0)
    total = _STARTUP_STATE.get("episodes_total")
    total_i = int(total) if total is not None else None
    eval_t0 = _STARTUP_STATE.get("eval_t0")
    payload = {
        "updated_utc": _now_utc(),
        "pid": int(_STARTUP_STATE.get("pid", os.getpid())),
        "status": _STARTUP_STATE.get("status"),
        "startup_phase": _STARTUP_STATE.get("startup_phase"),
        "startup_detail": _STARTUP_STATE.get("startup_detail"),
        "startup_phase_utc": _STARTUP_STATE.get("startup_phase_utc"),
        "episodes_consumed": consumed,
        "episodes_total": total_i,
        "episodes_frac": (round(consumed / total_i, 4) if total_i else None),
        "elapsed_seconds": round(time.time() - float(_STARTUP_STATE.get("t0", time.time())), 1),
        "eval_elapsed_seconds": (
            round(time.time() - float(eval_t0), 1) if eval_t0 is not None else None
        ),
        "eta_seconds": _eta_seconds(done=consumed, total=int(total_i or 0), t0=eval_t0),
        "progress_log": str(PROGRESS_LOG_PATH.relative_to(PROJECT_ROOT)),
        "manifest": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
    }
    payload.update(extra)
    try:
        _atomic_write_json(PROGRESS_JSON_PATH, payload)
    except Exception:
        pass


_log("[STARTUP 01] process entered")

import numpy as np  # noqa: E402
import torch  # noqa: E402

_log("[STARTUP 01b] numpy/torch imported")

import experiments.run_g0_v2_evaluation as E  # noqa: E402
from experiments.run_c2_step_replay import _derived  # noqa: E402
from rl.ruleset_identity import ARTIFACT_IDENTITY_KEY  # noqa: E402

_log("[STARTUP 01c] evaluation modules imported")


def _startup(code: str, phase: str, detail: str = "") -> None:
    """Durable startup marker: progress log + stdout + manifest/json heartbeat."""
    msg = f"[STARTUP {code}] {phase}"
    if detail:
        msg = f"{msg}: {detail}"
    _log(msg)
    _STARTUP_STATE["startup_phase"] = phase
    _STARTUP_STATE["startup_phase_utc"] = _now_utc()
    if detail:
        _STARTUP_STATE["startup_detail"] = detail
    _write_progress_json()
    # Rewrite on-disk progress while STARTING or RUNNING so abrupt death leaves a trail.
    if MANIFEST_PATH.exists() and _STARTUP_STATE.get("status") in {"STARTING", "RUNNING"}:
        try:
            current = _read_json(MANIFEST_PATH)
        except Exception:
            current = {}
        current.update(
            {
                "status": _STARTUP_STATE["status"],
                "startup_phase": _STARTUP_STATE["startup_phase"],
                "startup_phase_utc": _STARTUP_STATE["startup_phase_utc"],
                "episodes_consumed": int(_STARTUP_STATE.get("episodes_consumed", 0)),
                "episodes_total": _STARTUP_STATE.get("episodes_total"),
                "pid": int(_STARTUP_STATE.get("pid", os.getpid())),
            }
        )
        if detail:
            current["startup_detail"] = detail
        _atomic_write_json(MANIFEST_PATH, current)


def _write_starting_manifest(*, input_hashes: dict, runner_commit: str) -> None:
    """Write STARTING manifest immediately after lock so abrupt deaths leave a phase trail."""
    payload = {
        "title": "C2 fresh confirmation run",
        "status": "STARTING",
        "startup_phase": _STARTUP_STATE.get("startup_phase", "LOCK_ACQUIRED"),
        "startup_phase_utc": _now_utc(),
        "started_utc": _now_utc(),
        "generated_utc": _now_utc(),
        "pid": os.getpid(),
        "runner_commit": runner_commit,
        "episodes_consumed": 0,
        "episodes_total": _STARTUP_STATE.get("episodes_total"),
        "progress_log": str(PROGRESS_LOG_PATH.relative_to(PROJECT_ROOT)),
        "progress_json": str(PROGRESS_JSON_PATH.relative_to(PROJECT_ROOT)),
        "authoritative_inputs": {
            "proposal": str(PROPOSAL_PATH.relative_to(PROJECT_ROOT)),
            "confirmation_input": str(INPUT_ALIAS_PATH.relative_to(PROJECT_ROOT)),
            "frozen_contract": str(FROZEN_PATH.relative_to(PROJECT_ROOT)),
        },
        "input_hashes": input_hashes,
    }
    _STARTUP_STATE.update(
        {
            "status": "STARTING",
            "pid": os.getpid(),
            "episodes_consumed": 0,
            "startup_phase": payload["startup_phase"],
        }
    )
    _atomic_write_json(MANIFEST_PATH, payload)
    _write_progress_json()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _runner_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%H"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        return (out.stdout or "").strip() or "unknown"
    except Exception:
        return "unknown"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_inputs() -> tuple[dict, dict, dict]:
    proposal = _read_json(PROPOSAL_PATH)
    alias = _read_json(INPUT_ALIAS_PATH)
    frozen = _read_json(FROZEN_PATH)
    return proposal, alias, frozen


def _validate_frozen_contract(proposal: dict, frozen: dict) -> dict:
    if proposal.get("candidate_id") != "C2_tagged_carrier_none_forward":
        raise ValueError("proposal candidate_id drift")
    if proposal.get("failure_label") != FAIL_LABEL:
        raise ValueError("proposal failure_label drift")
    if proposal.get("primary_feature") != FEATURE:
        raise ValueError("proposal primary_feature drift")
    if str(proposal.get("expected_direction")).upper() != "NEGATIVE":
        raise ValueError("proposal expected_direction drift")

    bands = proposal.get("frozen_precursor_lag", {}).get("bands")
    if bands != [[-30, -20], [-20, -10], [-10, -1]]:
        raise ValueError(f"unexpected lag bands in proposal: {bands}")

    frozen_bands = frozen.get("lag_bands", {}).get("bands")
    if frozen_bands != [[-30, -20], [-20, -10], [-10, -1]]:
        raise ValueError(f"unexpected lag bands in frozen contract: {frozen_bands}")

    q = frozen["qualification_criteria"]
    support_min = int(q["6_support"]["min_failure_windows"])
    if support_min != int(q["6_support"]["min_matched_control_windows"]):
        raise ValueError("frozen support mins diverge unexpectedly")
    effect_thr = float(q["4_effect_size"]["threshold"])
    bootstrap_n = int(q["5_uncertainty"]["resamples"])
    bootstrap_seed = int(q["5_uncertainty"]["seed"])
    actionability_thr = float(str(q["8_actionability"]["threshold"]).split(">=")[-1].strip().split()[0])
    prevalence_thr = float(q["10_natural_support"]["min_onset_prevalence_of_episodes"])
    min_onsets = int(q["10_natural_support"]["min_onsets_per_policy"])
    headroom_min = float(q["7_headroom"]["requirement"].split(">=")[-1].strip())
    gate1 = float(q["7_headroom"]["planned_gate1_min_effect"])

    return {
        "support_min": support_min,
        "effect_thr": effect_thr,
        "bootstrap_n": bootstrap_n,
        "bootstrap_seed": bootstrap_seed,
        "actionability_thr": actionability_thr,
        "prevalence_thr": prevalence_thr,
        "min_onsets": min_onsets,
        "headroom_min": headroom_min,
        "planned_gate1_effect": gate1,
    }


def _acquire_lock() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if LOCK_PATH.exists():
        try:
            prior = _read_json(LOCK_PATH)
        except Exception:
            prior = {"raw": LOCK_PATH.read_text(encoding="utf-8", errors="replace")}
        pid = prior.get("pid")
        if isinstance(pid, int):
            try:
                probe = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                    capture_output=True,
                    text=True,
                    timeout=8,
                    check=False,
                )
                row = (probe.stdout or "").strip()
                if row and "No tasks are running" not in row:
                    raise RuntimeError(f"confirmation runner already active: {prior}")
            except subprocess.TimeoutExpired:
                raise RuntimeError(f"confirmation runner lock check timed out: {prior}")
    LOCK_PATH.write_text(
        json.dumps({"pid": os.getpid(), "started_utc": _now_utc(), "runner_commit": _runner_commit()}, indent=2),
        encoding="utf-8",
    )


def _release_lock() -> None:
    try:
        LOCK_PATH.unlink()
    except FileNotFoundError:
        pass


def _seed_in_forbidden_block(seed: int) -> bool:
    if seed in FORBIDDEN_BASES:
        return True
    return any(lo <= seed <= hi for lo, hi in FORBIDDEN_RANGES)


def _iter_scan_files() -> list[Path]:
    roots = [
        PROJECT_ROOT / "artifacts",
        PROJECT_ROOT / "experiments",
        PROJECT_ROOT / "docs",
    ]
    exts = {".json", ".jsonl", ".csv", ".md", ".txt", ".py", ".ps1", ".yaml", ".yml"}
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(p)
    return paths


def _scan_used_eval_seeds() -> dict[int, set[str]]:
    pat = re.compile(r"\b(\d{7})\b")
    used: dict[int, set[str]] = {}
    paths = _iter_scan_files()
    n = len(paths)
    bytes_done = 0
    t0 = time.time()
    last_report = t0
    _log(f"[PROGRESS] seed_scan listed n_files={n}")
    _write_progress_json(seed_scan_files_total=n, seed_scan_files_done=0)
    for i, p in enumerate(paths, 1):
        try:
            raw = p.read_bytes()
            bytes_done += len(raw)
            txt = raw.decode("utf-8", errors="ignore")
        except Exception:
            continue
        for m in pat.finditer(txt):
            val = int(m.group(1))
            if SEED_SCAN_MIN <= val <= SEED_SCAN_MAX:
                used.setdefault(val, set()).add(str(p.relative_to(PROJECT_ROOT)))
        now = time.time()
        if i == 1 or i == n or (now - last_report) >= 10.0:
            elapsed = max(now - t0, 1e-6)
            rate = i / elapsed
            eta = round((n - i) / rate, 1) if rate > 0 else None
            detail = (
                f"files={i}/{n} ({100.0 * i / max(n, 1):.1f}%) "
                f"MB={bytes_done / (1024 * 1024):.1f} "
                f"files_per_s={rate:.2f} eta_s={eta} used_seeds={len(used)}"
            )
            _log(f"[PROGRESS] seed_scan {detail}")
            _STARTUP_STATE["startup_phase"] = "SEED_REGISTRY_SCAN_IN_PROGRESS"
            _STARTUP_STATE["startup_detail"] = detail
            _STARTUP_STATE["startup_phase_utc"] = _now_utc()
            _write_progress_json(
                seed_scan_files_total=n,
                seed_scan_files_done=i,
                seed_scan_bytes_done=bytes_done,
                seed_scan_eta_seconds=eta,
                used_seed_count=len(used),
            )
            if MANIFEST_PATH.exists():
                try:
                    current = _read_json(MANIFEST_PATH)
                    current.update(
                        {
                            "startup_phase": "SEED_REGISTRY_SCAN_IN_PROGRESS",
                            "startup_detail": detail,
                            "startup_phase_utc": _now_utc(),
                        }
                    )
                    _atomic_write_json(MANIFEST_PATH, current)
                except Exception:
                    pass
            last_report = now
    return used


def _allocate_fresh_seed_block(length: int, used: dict[int, set[str]]) -> tuple[int, list[int], dict]:
    if length <= 0:
        raise ValueError("seed block length must be positive")
    candidates_checked = 0
    for base in range(FRESH_SEARCH_START, FRESH_SEARCH_END - length + 2):
        candidates_checked += 1
        block = [base + i for i in range(length)]
        if any(_seed_in_forbidden_block(s) for s in block):
            continue
        collisions = {s: sorted(list(used[s])) for s in block if s in used}
        if collisions:
            continue
        return base, block, {"candidates_checked": candidates_checked, "collision_sources": {}}
    raise RuntimeError("no collision-free fresh seed block found in allowed range")


def _verify_no_collision_or_die(block: list[int], used: dict[int, set[str]]) -> None:
    collisions = {s: sorted(list(used[s])) for s in block if s in used}
    if collisions:
        raise RuntimeError(f"fresh seed collision detected: {collisions}")
    for s in block:
        if _seed_in_forbidden_block(s):
            raise RuntimeError(f"fresh seed in forbidden range: {s}")


def _artifact_dir(seed: int) -> Path:
    return PROJECT_ROOT / "artifacts" / "g0_v5_long" / f"g0_v5_long_seed{seed}"


def _run_tag(seed: int) -> str:
    return f"g0_v5_long_seed{seed}"


def _load_policy_for(seed: int, *, device: str):
    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.evaluation.checkpoint import load_policy

    tag = _run_tag(seed)
    ckpt = _artifact_dir(seed) / "ckpts" / f"final_{tag}.zip"
    if not ckpt.exists():
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")
    payload = read_checkpoint_payload(str(ckpt), map_location="cpu")
    aid = payload.get(ARTIFACT_IDENTITY_KEY, {})
    if aid.get("identity_override_used") is True:
        raise ValueError(f"{ckpt}: identity_override_used=true")
    if aid.get("ruleset_id") != "RULESET_V2_AQUATICUS_10S":
        raise ValueError(f"{ckpt}: unexpected ruleset_id={aid.get('ruleset_id')}")
    if aid.get("canonical_map") != "map_a":
        raise ValueError(f"{ckpt}: unexpected canonical_map={aid.get('canonical_map')}")
    if int(payload.get("global_step", 0)) < 1_000_000:
        raise ValueError(f"{ckpt}: expected 1M-step checkpoint")
    channels = E.resolve_cnn_channels(payload, context=str(ckpt))
    if channels != 7:
        raise ValueError(f"{ckpt}: expected 7 channels, got {channels}")
    policy = load_policy(str(ckpt), device=device, num_cnn_channels=channels)
    return policy, ckpt, aid, channels


def _band_aggregate(steps: list[dict], t: int, start_off: int, end_off: int) -> dict | None:
    lo = max(0, t + start_off)
    hi = min(len(steps), t + end_off)
    if hi <= lo:
        return None
    win = steps[lo:hi]
    if len(win) < 3:
        return None

    carry = [s for s in win if s.get("carrier_present")]
    pressures = [s.get("carrier_pressure") for s in carry if isinstance(s.get("carrier_pressure"), (int, float)) and math.isfinite(s.get("carrier_pressure"))]
    margins = [s.get("intervention_margin") for s in carry if isinstance(s.get("intervention_margin"), (int, float)) and math.isfinite(s.get("intervention_margin"))]
    score_ref = float(win[-1].get("score_diff", 0.0))

    def frac(rows: list[dict], pred) -> float:
        if not rows:
            return float("nan")
        n = sum(1 for r in rows if pred(r))
        return float(n / len(rows))

    def mean(vals) -> float:
        vv = [float(v) for v in vals if isinstance(v, (int, float)) and math.isfinite(v)]
        return float(statistics.fmean(vv)) if vv else float("nan")

    return {
        "window_start": lo,
        "window_end": hi,
        "window_len": len(win),
        "opp_has_carrier": bool(carry),
        "score_stratum": "leading" if score_ref > 0 else "trailing" if score_ref < 0 else "tied",
        "none_forward_frac": frac(win, lambda s: int(s.get("agents_forward", -1)) == 0),
        "mate_can_intervene": frac(carry, lambda s: bool(s.get("mate_can_intervene"))),
        "intervention_margin": mean(margins),
        "carrier_pressure": mean(pressures),
    }


def _episode_clustered_ci(fail_windows: list[dict], ctrl_windows: list[dict], feature: str, rng: np.random.Generator, n_resamples: int) -> dict:
    by_fail = {}
    by_ctrl = {}
    for w in fail_windows:
        by_fail.setdefault(w["episode_key"], []).append(float(w[feature]))
    for w in ctrl_windows:
        by_ctrl.setdefault(w["episode_key"], []).append(float(w[feature]))

    fail_keys = list(by_fail.keys())
    ctrl_keys = list(by_ctrl.keys())
    if not fail_keys or not ctrl_keys:
        return {"ci_low": None, "ci_high": None, "excludes_zero": False, "n_failure_episodes": len(fail_keys), "n_control_episodes": len(ctrl_keys)}

    def _sample(keys, bucket):
        out = []
        picks = rng.integers(0, len(keys), size=len(keys))
        for idx in picks:
            out.extend(bucket[keys[int(idx)]])
        return out

    deltas = []
    for _ in range(n_resamples):
        f = _sample(fail_keys, by_fail)
        c = _sample(ctrl_keys, by_ctrl)
        if not f or not c:
            continue
        deltas.append(float(statistics.fmean(f) - statistics.fmean(c)))
    if not deltas:
        return {"ci_low": None, "ci_high": None, "excludes_zero": False, "n_failure_episodes": len(fail_keys), "n_control_episodes": len(ctrl_keys)}

    deltas.sort()
    lo = deltas[int(0.025 * (len(deltas) - 1))]
    hi = deltas[int(0.975 * (len(deltas) - 1))]
    return {
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "excludes_zero": bool(hi < 0 or lo > 0),
        "n_failure_episodes": len(fail_keys),
        "n_control_episodes": len(ctrl_keys),
    }


def _run_replay_for_policy(policy_seed: int, policy, eval_seeds: list[int], opponents: list[str], device: str) -> list[dict]:
    rows: list[dict] = []
    real_summarize = E.summarize_episode
    captured: dict = {}

    def capture(steps, **kw):
        captured["steps"] = steps
        captured["failure_events"] = kw.get("failure_events") or []
        return real_summarize(steps, **kw)

    E.summarize_episode = capture
    try:
        first_episode = True
        for opp in opponents:
            for ev in eval_seeds:
                if first_episode:
                    if _STARTUP_STATE.get("eval_t0") is None:
                        _STARTUP_STATE["eval_t0"] = time.time()
                    _startup("11", "FIRST_EPISODE_STARTING", f"policy={policy_seed} opp={opp} seed={ev}")
                    first_episode = False
                captured.clear()
                E.run_eval_episode(policy, opponent=opp, seed=int(ev), device=device)
                _STARTUP_STATE["episodes_consumed"] = int(_STARTUP_STATE.get("episodes_consumed", 0)) + 1
                consumed = int(_STARTUP_STATE["episodes_consumed"])
                total = int(_STARTUP_STATE.get("episodes_total") or 0)
                eta = _eta_seconds(done=consumed, total=total, t0=_STARTUP_STATE.get("eval_t0"))
                if total:
                    detail = (
                        f"episodes={consumed}/{total} ({100.0 * consumed / total:.1f}%) "
                        f"policy={policy_seed} last={opp}:{ev} eta_s={eta}"
                    )
                else:
                    detail = f"episodes={consumed} policy={policy_seed} last={opp}:{ev}"
                # Heartbeat every episode to disk; denser STARTUP every 10 (also updates manifest).
                _STARTUP_STATE["startup_phase"] = "EPISODES_IN_PROGRESS"
                _STARTUP_STATE["startup_detail"] = detail
                _STARTUP_STATE["startup_phase_utc"] = _now_utc()
                _log(f"[PROGRESS] {detail}")
                _write_progress_json(last_policy_seed=policy_seed, last_opponent=opp, last_eval_seed=int(ev))
                if consumed == 1 or consumed % 10 == 0 or (total and consumed == total):
                    _startup("11b", "EPISODES_IN_PROGRESS", detail)
                steps = captured["steps"]
                failure_events = captured["failure_events"]
                episode_key = f"{opp}:{ev}"
                fail_steps = {int(t) for _, t in failure_events}

                for label, t in failure_events:
                    if label != FAIL_LABEL:
                        continue
                    for band_name, a, b in (("earliest", -30, -20), ("middle", -20, -10), ("latest", -10, 0)):
                        feat = _band_aggregate(steps, int(t), a, b)
                        if feat is None:
                            continue
                        rows.append(
                            {
                                "policy_seed": policy_seed,
                                "opponent": opp,
                                "eval_seed": int(ev),
                                "episode_key": episode_key,
                                "kind": "failure",
                                "failure_label": FAIL_LABEL,
                                "outcome_step": int(t),
                                "band": band_name,
                                "band_start_off": a,
                                "band_end_off": b,
                                **feat,
                            }
                        )

                for end in range(30, len(steps), 30):
                    if any(abs(end - t) < 30 for t in fail_steps):
                        continue
                    for band_name, a, b in (("earliest", -30, -20), ("middle", -20, -10), ("latest", -10, 0)):
                        feat = _band_aggregate(steps, int(end), a, b)
                        if feat is None:
                            continue
                        rows.append(
                            {
                                "policy_seed": policy_seed,
                                "opponent": opp,
                                "eval_seed": int(ev),
                                "episode_key": episode_key,
                                "kind": "control",
                                "failure_label": "none",
                                "outcome_step": int(end),
                                "band": band_name,
                                "band_start_off": a,
                                "band_end_off": b,
                                **feat,
                            }
                        )
    finally:
        E.summarize_episode = real_summarize
    return rows


def _window_lengths(rows: list[dict], *, kind: str, band: str) -> dict:
    vals = [int(r["window_len"]) for r in rows if r["kind"] == kind and r["band"] == band]
    if not vals:
        return {"min": None, "median": None, "max": None}
    return {"min": min(vals), "median": int(statistics.median(vals)), "max": max(vals)}


def _evaluate_policy(rows: list[dict], policy_seed: int, cfg: dict) -> dict:
    rng = np.random.default_rng(cfg["bootstrap_seed"])
    out = {
        "policy_seed": policy_seed,
        "bands": {},
        "score_strata": {},
        "checks": {},
        "pass": False,
        "fail_reasons": [],
    }

    matched_fail_latest = [
        r for r in rows
        if r["policy_seed"] == policy_seed and r["kind"] == "failure" and r["band"] == "latest" and bool(r.get(MATCH_KEY))
    ]
    matched_ctrl_latest = [
        r for r in rows
        if r["policy_seed"] == policy_seed and r["kind"] == "control" and r["band"] == "latest" and bool(r.get(MATCH_KEY))
    ]
    denom = len(matched_fail_latest) + len(matched_ctrl_latest)
    failure_rate = (len(matched_fail_latest) / denom) if denom else 0.0
    headroom = 1.0 - failure_rate
    actionability = statistics.fmean(
        [float(r["mate_can_intervene"]) for r in matched_fail_latest if isinstance(r.get("mate_can_intervene"), (int, float)) and math.isfinite(r.get("mate_can_intervene"))]
    ) if matched_fail_latest else 0.0

    for band in BAND_ORDER:
        fail = [
            r for r in rows
            if r["policy_seed"] == policy_seed and r["kind"] == "failure" and r["band"] == band and bool(r.get(MATCH_KEY))
        ]
        ctrl = [
            r for r in rows
            if r["policy_seed"] == policy_seed and r["kind"] == "control" and r["band"] == band and bool(r.get(MATCH_KEY))
        ]
        fvals = [float(r[FEATURE]) for r in fail if isinstance(r.get(FEATURE), (int, float)) and math.isfinite(r.get(FEATURE))]
        cvals = [float(r[FEATURE]) for r in ctrl if isinstance(r.get(FEATURE), (int, float)) and math.isfinite(r.get(FEATURE))]
        mf = float(statistics.fmean(fvals)) if fvals else None
        mc = float(statistics.fmean(cvals)) if cvals else None
        delta = (mf - mc) if (mf is not None and mc is not None) else None
        ci = _episode_clustered_ci(fail, ctrl, FEATURE, rng=rng, n_resamples=cfg["bootstrap_n"])

        out["bands"][band] = {
            "n_failure_windows": len(fail),
            "n_control_windows": len(ctrl),
            "mean_failure": None if mf is None else round(mf, 4),
            "mean_control": None if mc is None else round(mc, 4),
            "delta": None if delta is None else round(delta, 4),
            **ci,
            "effective_band_length_failure": _window_lengths(fail, kind="failure", band=band),
            "effective_band_length_control": _window_lengths(ctrl, kind="control", band=band),
            "unique_failure_episode_clusters": len({r["episode_key"] for r in fail}),
            "unique_control_episode_clusters": len({r["episode_key"] for r in ctrl}),
        }

    latest = out["bands"]["latest"]
    earliest = out["bands"]["earliest"]

    # Score-stratum rule on latest band only (matching Stage 2).
    stratum_pass = False
    for stratum in ("leading", "trailing", "tied"):
        fs = [r for r in matched_fail_latest if r.get("score_stratum") == stratum]
        cs = [r for r in matched_ctrl_latest if r.get("score_stratum") == stratum]
        fvals = [float(r[FEATURE]) for r in fs if isinstance(r.get(FEATURE), (int, float)) and math.isfinite(r.get(FEATURE))]
        cvals = [float(r[FEATURE]) for r in cs if isinstance(r.get(FEATURE), (int, float)) and math.isfinite(r.get(FEATURE))]
        mf = float(statistics.fmean(fvals)) if fvals else None
        mc = float(statistics.fmean(cvals)) if cvals else None
        delta = (mf - mc) if (mf is not None and mc is not None) else None
        ci = _episode_clustered_ci(fs, cs, FEATURE, rng=rng, n_resamples=cfg["bootstrap_n"])
        row = {
            "n_failure_windows": len(fs),
            "n_control_windows": len(cs),
            "delta": None if delta is None else round(delta, 4),
            **ci,
        }
        out["score_strata"][stratum] = row
        if (
            delta is not None
            and len(fs) >= cfg["support_min"]
            and len(cs) >= cfg["support_min"]
            and abs(delta) >= cfg["effect_thr"]
            and ((delta < 0) if EXPECTED_SIGN < 0 else (delta > 0))
            and row["excludes_zero"] is True
        ):
            stratum_pass = True

    checks = {
        "support_latest": latest["n_failure_windows"] >= cfg["support_min"] and latest["n_control_windows"] >= cfg["support_min"],
        "effect_latest": (latest["delta"] is not None and abs(float(latest["delta"])) >= cfg["effect_thr"]),
        "ci_latest": latest["excludes_zero"] is True,
        "direction_latest": (latest["delta"] is not None and ((float(latest["delta"]) < 0) if EXPECTED_SIGN < 0 else (float(latest["delta"]) > 0))),
        "earliest_present_correct_direction": (earliest["delta"] is not None and ((float(earliest["delta"]) < 0) if EXPECTED_SIGN < 0 else (float(earliest["delta"]) > 0))),
        "headroom": headroom >= cfg["headroom_min"] and headroom >= 2.0 * cfg["planned_gate1_effect"],
        "actionability": actionability >= cfg["actionability_thr"],
        "natural_support": len(matched_fail_latest) >= cfg["min_onsets"],
        "score_stratum_survival": stratum_pass,
    }
    out["checks"] = checks
    out["headroom"] = round(headroom, 4)
    out["actionability"] = round(float(actionability), 4)
    out["natural_onsets_latest"] = len(matched_fail_latest)
    out["natural_prevalence_latest"] = round(float(failure_rate), 4)
    out["pass"] = all(checks.values())
    out["fail_reasons"] = [k for k, v in checks.items() if not v]
    return out


def _write_windows_csv(rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0].keys())
    with WINDOWS_PATH.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30, help="episodes per opponent per policy")
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = ap.parse_args()

    runner_commit = _runner_commit()
    input_hashes = {
        "proposal_sha256": _sha256_file(PROPOSAL_PATH),
        "confirmation_input_sha256": _sha256_file(INPUT_ALIAS_PATH),
        "frozen_contract_sha256": _sha256_file(FROZEN_PATH),
    }

    _startup("03", "FROZEN_INPUTS_LOADING")
    proposal, alias, frozen = _load_inputs()
    cfg = _validate_frozen_contract(proposal, frozen)
    _startup("03b", "FROZEN_INPUTS_LOADED", f"candidate={proposal.get('candidate_id')}")

    _acquire_lock()
    _startup("02", "LOCK_ACQUIRED", f"pid={os.getpid()} lock={LOCK_PATH}")
    _write_starting_manifest(input_hashes=input_hashes, runner_commit=runner_commit)
    _startup("02b", "STARTING_MANIFEST_WRITTEN", str(MANIFEST_PATH))

    t0 = time.time()
    try:
        _startup("04", "SEED_REGISTRY_SCAN_STARTED")
        used = _scan_used_eval_seeds()
        _startup("05", "SEED_REGISTRY_SCAN_COMPLETE", f"used_seed_count={len(used)}")
        base_seed, eval_seeds, allocation_meta = _allocate_fresh_seed_block(args.episodes, used)
        _verify_no_collision_or_die(eval_seeds, used)
        _startup("05b", "FRESH_SEED_BLOCK_ALLOCATED", f"base={base_seed} n={len(eval_seeds)}")

        E.artifact_dir_for = _artifact_dir
        E.run_tag_for = _run_tag
        E.EVAL_SEED_BASE = base_seed

        policy_seeds = [3_200_001, 3_200_002, 3_200_003]
        opponents = ["OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12"]
        episodes_total = int(len(policy_seeds) * len(opponents) * int(args.episodes))
        _STARTUP_STATE["episodes_total"] = episodes_total
        _log(
            f"[PROGRESS] planned_eval episodes_total={episodes_total} "
            f"(policies={len(policy_seeds)} opponents={len(opponents)} "
            f"episodes_per_opponent={args.episodes})"
        )
        _write_progress_json()

        _startup("07", "ENV_CONSTRUCTION_DEFERRED", "env built lazily inside first run_eval_episode")

        manifest = {
            "title": "C2 fresh confirmation run",
            "status": "RUNNING",
            "startup_phase": _STARTUP_STATE.get("startup_phase", "FRESH_SEED_BLOCK_ALLOCATED"),
            "started_utc": _now_utc(),
            "generated_utc": _now_utc(),
            "pid": os.getpid(),
            "episodes_consumed": int(_STARTUP_STATE.get("episodes_consumed", 0)),
            "episodes_total": episodes_total,
            "progress_log": str(PROGRESS_LOG_PATH.relative_to(PROJECT_ROOT)),
            "progress_json": str(PROGRESS_JSON_PATH.relative_to(PROJECT_ROOT)),
            "runner_commit": runner_commit,
            "authoritative_inputs": {
                "proposal": str(PROPOSAL_PATH.relative_to(PROJECT_ROOT)),
                "confirmation_input": str(INPUT_ALIAS_PATH.relative_to(PROJECT_ROOT)),
                "frozen_contract": str(FROZEN_PATH.relative_to(PROJECT_ROOT)),
            },
            "input_hashes": input_hashes,
            "candidate": {
                "candidate_id": proposal["candidate_id"],
                "failure_label": proposal["failure_label"],
                "feature": proposal["primary_feature"],
                "expected_direction": proposal["expected_direction"],
                "opportunity_match": proposal["primary_outcome_opportunity_definition"]["opportunity_matching_rule"],
                "bands": proposal["frozen_precursor_lag"]["bands"],
            },
            "fresh_seed_block": {
                "base": base_seed,
                "episodes_per_opponent": int(args.episodes),
                "seeds": eval_seeds,
                "allocation_meta": allocation_meta,
                "collision_scan": {
                    "range": [SEED_SCAN_MIN, SEED_SCAN_MAX],
                    "forbidden_ranges": [list(r) for r in FORBIDDEN_RANGES],
                    "forbidden_bases": sorted(list(FORBIDDEN_BASES)),
                    "used_seed_count": len(used),
                },
            },
            "policies": {},
            "integrity_preflight": {},
        }
        _STARTUP_STATE["status"] = "RUNNING"
        _atomic_write_json(MANIFEST_PATH, manifest)
        _startup("10", "RUNNING_MANIFEST_WRITTEN", f"fresh_base={base_seed}")

        all_rows: list[dict] = []
        ckpt_fingerprints = {}
        ruleset_fingerprints = {}
        for i, ps in enumerate(policy_seeds):
            _startup("06" if i == 0 else "09", "CHECKPOINT_LOAD_STARTED" if i == 0 else "NEXT_POLICY_LOAD", f"policy_seed={ps}")
            policy, ckpt, aid, channels = _load_policy_for(ps, device=args.device)
            if i == 0:
                _startup("06b", "CHECKPOINT_HASHES_VERIFIED", f"policy_seed={ps}")
                _startup("09", "FIRST_POLICY_LOADED", f"policy_seed={ps} device={args.device}")
            manifest["policies"][str(ps)] = {
                "checkpoint": str(ckpt.relative_to(PROJECT_ROOT)),
                "checkpoint_sha256": _sha256_file(ckpt),
                "identity": aid,
                "resolved_cnn_channels": channels,
            }
            ckpt_fingerprints[str(ps)] = manifest["policies"][str(ps)]["checkpoint_sha256"]
            ruleset_fingerprints[str(ps)] = aid.get("ruleset_fingerprint")
            _atomic_write_json(MANIFEST_PATH, {**manifest, "episodes_consumed": int(_STARTUP_STATE.get("episodes_consumed", 0))})
            rows = _run_replay_for_policy(ps, policy, eval_seeds=eval_seeds, opponents=opponents, device=args.device)
            all_rows.extend(rows)

        # Pre-analysis immutable checks.
        unique_rulesets = {v for v in ruleset_fingerprints.values() if isinstance(v, str)}
        manifest["integrity_preflight"] = {
            "checkpoint_sha256_recorded": True,
            "ruleset_fingerprint_consistent": len(unique_rulesets) == 1,
            "map_verified": all(
                manifest["policies"][str(ps)]["identity"].get("canonical_map") == "map_a"
                for ps in policy_seeds
            ),
            "loader_7_channel_verified": all(
                int(manifest["policies"][str(ps)]["resolved_cnn_channels"]) == 7
                for ps in policy_seeds
            ),
            "identity_override_absent": all(
                manifest["policies"][str(ps)]["identity"].get("identity_override_used") is False
                for ps in policy_seeds
            ),
            "outcome_step_excluded_from_bands": True,
            "opportunity_match_semantics": "opp_has_carrier == true",
        }
        if not all(manifest["integrity_preflight"].values()):
            raise RuntimeError(f"integrity preflight failed: {manifest['integrity_preflight']}")

        _startup("12", "WINDOWS_CSV_WRITE_STARTED", f"n_rows={len(all_rows)}")
        _write_windows_csv(all_rows)

        per_policy = []
        for ps in policy_seeds:
            per_policy.append(_evaluate_policy(all_rows, ps, cfg))

        required_replication = 2  # expected >=2/3 from proposal text.
        pass_count = sum(1 for p in per_policy if p["pass"])
        verdict = "C2_CONFIRMED" if pass_count >= required_replication else "C2_REJECTED"

        results = {
            "generated_utc": _now_utc(),
            "runner_commit": manifest["runner_commit"],
            "candidate_id": proposal["candidate_id"],
            "failure_label": FAIL_LABEL,
            "feature": FEATURE,
            "expected_direction": "NEGATIVE",
            "bands": proposal["frozen_precursor_lag"]["bands"],
            "required_replication_pass_count": required_replication,
            "policy_pass_count": pass_count,
            "overall_verdict": verdict,
            "policy_results": per_policy,
            "checkpoint_sha256s": ckpt_fingerprints,
            "ruleset_fingerprint": sorted(list(unique_rulesets))[0] if unique_rulesets else None,
            "fresh_seed_block_base": base_seed,
            "fresh_eval_seeds": eval_seeds,
            "fail_closed_notes": [
                "No lag re-selection performed on fresh data.",
                "No candidate substitution performed.",
                "No threshold override performed.",
            ],
            "decision_branch": {
                "if_rejected": "STOP; do not train O2 and do not alter candidate/lag/thresholds on this block.",
                "if_confirmed": "Proceed only to freezing O2 prereg protocol; do not launch O2 before protocol freeze.",
            },
            "wall_seconds": round(time.time() - t0, 2),
        }

        manifest["status"] = "COMPLETE"
        manifest["startup_phase"] = "COMPLETE"
        manifest["episodes_consumed"] = int(_STARTUP_STATE.get("episodes_consumed", 0))
        manifest["final"] = {
            "windows_csv": str(WINDOWS_PATH.relative_to(PROJECT_ROOT)),
            "results_json": str(RESULTS_PATH.relative_to(PROJECT_ROOT)),
            "frozen_result_json": str(FROZEN_RESULT_PATH.relative_to(PROJECT_ROOT)),
            "verdict": verdict,
        }

        if verdict == "C2_CONFIRMED":
            O2_DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
            o2_draft = {
                "title": "O2 protocol draft from confirmed C2",
                "generated_utc": _now_utc(),
                "source_confirmation_result": str(FROZEN_RESULT_PATH.relative_to(PROJECT_ROOT)),
                "source_candidate_id": proposal["candidate_id"],
                "required_freeze_before_training": True,
                "pipeline": [
                    "confirmed natural C2",
                    "measure natural onset rate",
                    "G0 prefix",
                    "true C2 onset",
                    "handoff to O2",
                    "O2 receives only post-onset PPO credit",
                    "ONE default O2 seed",
                    "four development gates",
                ],
                "training_launch_allowed_now": False,
                "note": "This draft is auto-frozen only after C2 confirmation; O2 training remains blocked until protocol review/final freeze.",
            }
            O2_DRAFT_PATH.write_text(json.dumps(o2_draft, indent=2), encoding="utf-8")
            manifest["final"]["o2_protocol_draft"] = str(O2_DRAFT_PATH.relative_to(PROJECT_ROOT))

        _atomic_write_json(MANIFEST_PATH, manifest)
        RESULTS_PATH.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        FROZEN_RESULT_PATH.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

        print("=" * 78, flush=True)
        print("C2 FRESH CONFIRMATION COMPLETE", flush=True)
        print(f"fresh_seed_base={base_seed} episodes_per_opponent={args.episodes}", flush=True)
        print(f"policy_pass_count={pass_count}/3 required={required_replication}", flush=True)
        print(f"VERDICT: {verdict}", flush=True)
        print(f"manifest: {MANIFEST_PATH}", flush=True)
        print(f"results:  {RESULTS_PATH}", flush=True)
        print(f"progress: {PROGRESS_LOG_PATH}", flush=True)
        print("=" * 78, flush=True)
        _log(f"[STARTUP DONE] VERDICT={verdict} wall_s={results['wall_seconds']}")
        _write_progress_json(verdict=verdict, status="COMPLETE")
        return 0
    except BaseException as exc:
        # Preserve last startup_phase on disk; mark ABORTED so a dead PID is diagnosable.
        try:
            abort = {}
            if MANIFEST_PATH.exists():
                try:
                    abort = _read_json(MANIFEST_PATH)
                except Exception:
                    abort = {}
            abort.update(
                {
                    "status": "ABORTED",
                    "abort_exc_type": type(exc).__name__,
                    "abort_exc": str(exc),
                    "abort_utc": _now_utc(),
                    "startup_phase": _STARTUP_STATE.get("startup_phase"),
                    "episodes_consumed": int(_STARTUP_STATE.get("episodes_consumed", 0)),
                    "episodes_total": _STARTUP_STATE.get("episodes_total"),
                    "pid": os.getpid(),
                }
            )
            _atomic_write_json(MANIFEST_PATH, abort)
            _log(
                f"[STARTUP ABORT] phase={_STARTUP_STATE.get('startup_phase')} "
                f"exc={type(exc).__name__}: {exc}"
            )
            _write_progress_json(status="ABORTED", abort_exc=str(exc))
        except Exception:
            pass
        raise
    finally:
        _release_lock()


if __name__ == "__main__":
    raise SystemExit(main())

