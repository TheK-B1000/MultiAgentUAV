"""LEARNED_COMPOSITION_OPENING_PROBE_V1_SPEC.json -- two sealed stages.

READ-ONLY. No training, no parameter change, no optimizer step. It measures what the
existing learned 6v6 specialists do in the OPENING of an episode, with an instrument that
is validated on scripted ground truth first.

  calibrate  Stage 1. Scripted compositions (known true defender count) are run on cuda for
             the first 20 ticks, on fresh seeds, and the instrument is scored on whether it
             recovers the truth. Sealed. The verdict is a pre-declared kill rule.
  probe      Stage 2. Runs ONLY if Stage 1 sealed as valid. Reads pi_A and pi_B with the same
             instrument. Sealed.

Why the opening only. Position near the own flag recovers a scripted composition only in the
first ~10 ticks. Afterwards tagged agents walking home to untag and the adapter's
all-attackers-GO_HOME-when-any-teammate-carries send every composition through the home
region, and all seven read alike. A resolved-target (intent) instrument failed its
pre-declared criterion too and is reported as failed, not rescued. So nothing here says
anything about sustained role composition.

CLAIM BOUNDARY, frozen: the instrument is validated on SCRIPTED roles. A learned policy that
moves in ways scripted agents never do is read through scripted-role glasses. And the 6v6
Pole B is canonical SDS_PARENT_OP7, not the 4v4 B3-3 construction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.probe_learned_composition as P  # noqa: E402
from experiments.run_routed_composition_outcome import ALPHA, N_BOOT, RNG_SEED, _bootstrap  # noqa: E402
from experiments.tqdm_loop import tqdm_iter  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / "LEARNED_COMPOSITION_OPENING_PROBE_V1_SPEC.json"
CAL_CONTRACT = SD / "LEARNED_COMPOSITION_PROBE_CALIBRATION_CONTRACT_RESULT.json"
CAL_RESULT = SD / "LEARNED_COMPOSITION_PROBE_CALIBRATION_RESULT.json"
CAL_ROWS = SD / "LEARNED_COMPOSITION_PROBE_CALIBRATION_ROWS.csv"
CAL_TRACES = SD / "LEARNED_COMPOSITION_PROBE_CALIBRATION_TRACES.npz"
CAL_PARTIAL = SD / "LEARNED_COMPOSITION_PROBE_CALIBRATION_PARTIAL.jsonl"
PROBE_CONTRACT = SD / "LEARNED_COMPOSITION_PROBE_CONTRACT_RESULT.json"
PROBE_RESULT = SD / "LEARNED_COMPOSITION_PROBE_RESULT.json"
PROBE_ROWS = SD / "LEARNED_COMPOSITION_PROBE_ROWS.csv"
PROBE_TRACES = SD / "LEARNED_COMPOSITION_PROBE_TRACES.npz"
PROBE_PARTIAL = SD / "LEARNED_COMPOSITION_PROBE_PARTIAL.jsonl"
CONFIRM_SPEC = SD / "LEARNED_COMPOSITION_PROBE_CONFIRMATORY_CALIBRATION_V1_SPEC.json"
CONFIRM_CONTRACT = SD / "LEARNED_COMPOSITION_PROBE_CONFIRMATORY_CALIBRATION_CONTRACT_RESULT.json"
CONFIRM_RESULT = SD / "LEARNED_COMPOSITION_PROBE_CONFIRMATORY_CALIBRATION_RESULT.json"
CONFIRM_ROWS = SD / "LEARNED_COMPOSITION_PROBE_CONFIRMATORY_CALIBRATION_ROWS.csv"
CONFIRM_TRACES = SD / "LEARNED_COMPOSITION_PROBE_CONFIRMATORY_CALIBRATION_TRACES.npz"
CONFIRM_PARTIAL = SD / "LEARNED_COMPOSITION_PROBE_CONFIRMATORY_CALIBRATION_PARTIAL.jsonl"
STAGE2_SPEC = SD / "LEARNED_COMPOSITION_PROBE_STAGE2_V1_SPEC.json"
EXP_CAL = "LEARNED_COMPOSITION_PROBE_CALIBRATION"
EXP_PROBE = "LEARNED_COMPOSITION_PROBE"
EXP_CONFIRM = "LEARNED_COMPOSITION_PROBE_CONFIRMATORY_CALIBRATION"

DEVICE = "cuda"
R_POS = 4.5                  # frozen from the scripted cpu prototype; validated out of sample below
WIN_PRIMARY = (0, 10)
WIN_ALT = (5, 15)            # declared stability check, not a second choice
T_TRUNC = 20
COMPS = ["6A_0D", "5A_1D", "4A_2D", "3A_3D", "2A_4D", "1A_5D", "0A_6D"]
D_OF = {c: int(c.split("_")[1][:-1]) for c in COMPS}
CAL_SEEDS = list(range(20_700_001, 20_700_001 + 16))
PROBE_SEEDS = list(range(20_700_101, 20_700_101 + 128))
CONFIRM_SEEDS = list(range(20_800_001, 20_800_001 + 16))
CONFIRM_CANDIDATES = ("share_pos_alt", "share_int_open")
CONFIRM_LABEL = {"share_pos_alt": "position_alt_window_5_15", "share_int_open": "intent_opening_window_0_10"}
STAGE2_INSTRUMENTS = {"d_hat_pos": ("share_pos_alt", "STABILITY_position_alt_window", "position_alt_window_5_15"),
                     "d_hat_int": ("share_int_open", "SECONDARY_intent_opening_pre_declared_not_required", "intent_opening_window_0_10")}
PARITY_SEEDS = (13_680_001, 13_680_002, 13_680_003, 13_680_004)
MIN_RECOVERY, TOL_D = 0.90, 1
BAND_ATTACK_HEAVY, BAND_INTERMEDIATE = 1.5, 2.5      # D_hat cutoffs, declared before any learned reading
SEED_CLASS = "exploratory"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cal_cells() -> list[tuple[str, str]]:
    """(composition, assignment). SUFFIX is a distinct placement only for 1..5 defenders."""
    cells = []
    for c in COMPS:
        cells.append((c, "PREFIX"))
        if 0 < D_OF[c] < 6:
            cells.append((c, "SUFFIX"))
    return cells


# ------------------------------------------------------------------ readings

def _wmean(x: np.ndarray, lo: int, hi: int) -> float:
    v = np.asarray(x[lo:hi], dtype=float)
    v = v[~np.isnan(v)]
    return float(v.mean()) if v.size else float("nan")


def opening_reading(trace: dict[str, Any]) -> dict[str, float]:
    """Every per-episode number the probe reports, from a recorded trace."""
    pos = P.instrument(trace, R_POS)
    inte = P.intent_instrument(trace)
    out = {"share_pos_open": _wmean(pos["share"], *WIN_PRIMARY),
           "share_pos_alt": _wmean(pos["share"], *WIN_ALT),
           "share_int_open": _wmean(inte["share"], *WIN_PRIMARY),
           "n_active_open": float(np.mean(pos["n_active"][WIN_PRIMARY[0]:WIN_PRIMARY[1]]))}
    lo, hi = WIN_PRIMARY
    for i in range(P.N_AGENTS):
        out[f"hold_id{i}"] = float(pos["defend_like"][lo:hi, i].mean())
    macros = np.asarray(trace["eff_macro"])[lo:hi]
    for m in range(8):
        out[f"macro{m}"] = float((macros == m).mean())
    return out


def reference_reading(trace: dict[str, Any]) -> dict[str, float]:
    """The same three shares from the independent loop implementations."""
    pos, inte = P.instrument_reference(trace, R_POS), P.intent_instrument_reference(trace)
    return {"share_pos_open": _wmean(pos["share"], *WIN_PRIMARY), "share_pos_alt": _wmean(pos["share"], *WIN_ALT),
            "share_int_open": _wmean(inte["share"], *WIN_PRIMARY)}


TRACE_KEYS = ("pos", "alive", "tagged", "carrying", "flag_home", "flag_pos", "intent", "eff_macro", "action")


def pack_traces(traces: list[dict[str, Any]], keys: list[str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {"keys": np.asarray(keys)}
    for k in TRACE_KEYS:
        out[k] = np.stack([np.asarray(t[k]) for t in traces])
    return out


def trace_from_packed(packed: dict[str, np.ndarray], i: int) -> dict[str, Any]:
    return {k: packed[k][i] for k in TRACE_KEYS}


# ------------------------------------------------------------- validity rule

def recover(value: float, means: dict[str, float]) -> int:
    """Nearest calibration composition, returned as its true defender count."""
    return D_OF[min(COMPS, key=lambda c: abs(value - means[c]))]


def evaluate_validity(rows: list[dict], key: str) -> dict[str, Any]:
    """The frozen kill rule, applied to one reading.

    Per pole: V1 the PREFIX class means rise strictly with the true defender count; V2
    leave-one-out recovery of the PREFIX episodes within +-1; V3 cross-assignment recovery,
    where the SUFFIX episodes are read against the PREFIX-built calibration. Each recovery
    rate must reach 0.90. V3 is the id-geometry control: agents 2 and 3 spawn inside the
    instrument radius, so the two assignments differ in spawn geometry only.
    """
    out: dict[str, Any] = {}
    for pole in P.POLES:
        pre = {c: [r[key] for r in rows if r["pole"] == pole and r["composition"] == c and r["assignment"] == "PREFIX"]
               for c in COMPS}
        suf = [(D_OF[r["composition"]], r[key]) for r in rows if r["pole"] == pole and r["assignment"] == "SUFFIX"]
        means = {c: float(np.nanmean(v)) for c, v in pre.items()}
        v1 = bool(np.all(np.diff([means[c] for c in COMPS]) > 0))
        ok2 = ex2 = n2 = 0
        for c in COMPS:
            for j, val in enumerate(pre[c]):
                loo = {cc: (float(np.nanmean([v for jj, v in enumerate(pre[cc]) if not (cc == c and jj == j)]))) for cc in COMPS}
                d_hat = recover(val, loo)
                n2 += 1
                ok2 += abs(d_hat - D_OF[c]) <= TOL_D
                ex2 += d_hat == D_OF[c]
        ok3 = ex3 = 0
        for d_true, val in suf:
            d_hat = recover(val, means)
            ok3 += abs(d_hat - d_true) <= TOL_D
            ex3 += d_hat == d_true
        rate2, rate3 = ok2 / n2, (ok3 / len(suf) if suf else float("nan"))
        out[pole] = {"class_means_prefix": {c: round(means[c], 6) for c in COMPS},
                     "V1_monotone": v1, "V2_loo_within1": round(rate2, 4), "V2_loo_exact": round(ex2 / n2, 4),
                     "V3_cross_assignment_within1": round(rate3, 4),
                     "V3_cross_assignment_exact": round(ex3 / len(suf), 4) if suf else None,
                     "n_prefix": n2, "n_suffix": len(suf),
                     "pole_valid": bool(v1 and rate2 >= MIN_RECOVERY and rate3 >= MIN_RECOVERY)}
    out["valid"] = bool(all(out[p]["pole_valid"] for p in P.POLES))
    return out


def d_hat_from(value: float, means: dict[str, float]) -> float:
    """Inverse of the monotone calibration curve by piecewise-linear interpolation, clipped
    to [0, 6]."""
    xs = np.asarray([means[c] for c in COMPS], dtype=float)
    ds = np.asarray([D_OF[c] for c in COMPS], dtype=float)
    return float(np.interp(value, xs, ds))


# ------------------------------------------------------------------- audit plans

def build_plan(rows_csv: Path, expected_rows: int, seeds: list[int], group_by: tuple[str, ...],
               experiment_id: str, spec_path: Path = SPEC_PATH):
    from experiments import run_state as rs
    return rs.AuditPlan(
        rows_csv=rows_csv, expected_rows=expected_rows, expected_seeds=seeds, group_by=group_by,
        seed_field="seed", int_fields=("seed", "steps"), binary_fields=(), derived={},
        spec_path=spec_path, seed_class=SEED_CLASS,
        # load-bearing: without the owner the audit reads this run's own Rule-9 reservation
        # as foreign reuse (ROUTED_COMPOSITION_OUTCOME_AUDIT_CORRECTION.json)
        experiment_id=experiment_id, n_boot=N_BOOT, alpha=ALPHA, rng_seed=RNG_SEED, claims=())


def _registered(experiment_id: str, lo: int, n: int) -> tuple[bool, str]:
    from experiments import seed_registry as sr
    ok, msg = sr.check_block(lo, lo + n - 1, SEED_CLASS, experiment_id=experiment_id)
    entry = next((b for b in sr.load()["blocks"] if b["experiment_id"] == experiment_id), None)
    good = ok and entry is not None and entry["status"] == "RESERVED" and entry["lo"] == lo and entry["hi"] == lo + n - 1
    return good, f"{msg}; registry status = {entry['status'] if entry else None}"


def _param_digest(policy) -> str:
    h = hashlib.sha256()
    for k, v in sorted(policy.model.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ------------------------------------------------------------------ contracts 1

def contracts_calibration() -> dict:
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str, **data: Any) -> None:
        checks.append({"name": name, "gating": True, "passed": bool(passed), "detail": detail, **data})

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    record("C0_SPEC_FROZEN", str(spec.get("status", "")).startswith("FROZEN"), f"spec status = {spec.get('status')!r}")
    g = P.pole_genomes()

    # C1: the frozen parameters equal the spec
    p = spec["INSTRUMENT_FROZEN"]
    ok = (float(p["R_pos"]) == R_POS and tuple(p["window_primary"]) == WIN_PRIMARY and tuple(p["window_alt"]) == WIN_ALT
          and int(p["truncate_at_tick"]) == T_TRUNC and float(p["R_intent"]) == P.R_INTENT
          and float(spec["VALIDITY_KILL_RULE"]["min_recovery"]) == MIN_RECOVERY and int(spec["VALIDITY_KILL_RULE"]["tolerance_defenders"]) == TOL_D)
    record("C1_EXECUTOR_CONSTANTS_EQUAL_THE_SPEC", ok, f"R_pos {R_POS} windows {WIN_PRIMARY}/{WIN_ALT} T {T_TRUNC} R_intent {P.R_INTENT} rule {MIN_RECOVERY}/+-{TOL_D}")

    # C2: truncation is exact ON CUDA: a run stopped at 20 ticks equals the first 20 of a full run
    same, detail = True, []
    for comp, asg, pole, seed in (("3A_3D", "PREFIX", "A", 99_900_681), ("5A_1D", "SUFFIX", "B", 99_900_682)):
        roles = P.roles_for(comp, asg)
        full = P.run_scripted_episode(comp, pole, seed, g, DEVICE, roles=roles)
        trunc = P.run_scripted_episode(comp, pole, seed, g, DEVICE, max_ticks=T_TRUNC, roles=roles)
        eq = all(np.array_equal(full[k][:T_TRUNC], trunc[k]) for k in ("pos", "alive", "tagged", "carrying", "intent", "eff_macro"))
        same &= eq
        detail.append(f"{comp}/{asg}/{pole}: {eq}")
    record("C2_TRUNCATION_EQUALS_THE_FIRST_TICKS_OF_A_FULL_RUN_ON_CUDA", same, "; ".join(detail))

    # C3: SUFFIX really places defenders on the last ids (checked on a live scripted episode)
    tr = P.run_scripted_episode("4A_2D", "A", 99_900_683, g, DEVICE, max_ticks=6, roles=P.roles_for("4A_2D", "SUFFIX"))
    record("C3_SUFFIX_ASSIGNMENT_IS_LIVE", tr["true_defend"].tolist() == [False, False, False, False, True, True],
           f"true_defend recorded on a live episode: {tr['true_defend'].tolist()}")

    # (Loading a checkpoint imports rl.custom_ppo.trainer, so stage 2's read-only guarantee is behavioural:
    #  a parameter digest and file hashes equal before and after. Stage 1 loads no checkpoint.)
    ok1, msg1 = _registered(EXP_CAL, CAL_SEEDS[0], len(CAL_SEEDS))
    ok2, msg2 = _registered(EXP_PROBE, PROBE_SEEDS[0], len(PROBE_SEEDS))
    record("C4_SEED_BLOCKS_OWNED_AND_UNSPENT", ok1 and ok2, f"calibration: {msg1} | probe: {msg2}")

    # C6: the REAL audit plan seals a block already registered to its owner, with a negative control
    import tempfile
    from experiments import run_state as rs
    tmp = Path(tempfile.mkdtemp()) / "rows.csv"
    cells = cal_cells()
    rows = [{"composition": c, "assignment": a, "pole": pole, "seed": s, "steps": T_TRUNC, "share_pos_open": 0.5}
            for pole in P.POLES for (c, a) in cells for s in CAL_SEEDS]
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    plan = build_plan(tmp, len(rows), CAL_SEEDS, ("composition", "assignment", "pole"), EXP_CAL)
    with_owner = rs.run_audit(plan)
    no_owner = rs.run_audit(rs.AuditPlan(**{**plan.__dict__, "experiment_id": None}))
    failed = sorted(c["name"] for c in no_owner["checks"] if c["result"] == "FAIL")
    record("C5_AUDIT_PLAN_SEALS_A_REGISTERED_BLOCK", with_owner["passed"] and failed == ["seed_class"],
           f"real AuditPlan with owner on {len(rows)} rows: passed={with_owner['passed']} ({with_owner['n_gating']} gating, "
           f"{with_owner['n_failed']} failed); negative control without the owner fails exactly {failed}")

    n_failed = sum(1 for c in checks if not c["passed"])
    report = {"record_id": "LEARNED_COMPOSITION_PROBE_CALIBRATION_CONTRACT_RESULT", "utc": _now(), "implements": SPEC_PATH.name,
              "n_checks": len(checks), "n_gating": len(checks), "n_failed": n_failed, "checks": checks,
              "DECISION": "CONTRACTS_PASS" if n_failed == 0 else "CONTRACT_FAILURE",
              "claim_boundary": "Contracts only. No calibration episode was scored and no seed was spent. C2/C3 use disposable smoke seeds."}
    CAL_CONTRACT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'=' * 66}\n  CALIBRATION CONTRACTS: {report['DECISION']}  ({n_failed}/{len(checks)} gating failed)\n{'=' * 66}")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    return report


# ---------------------------------------------------------------- stage 1 run

def _load_partial(path: Path) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                out[tuple(o["key"])] = o
    return out


def _trace_to_json(tr: dict[str, Any]) -> dict[str, Any]:
    return {k: np.asarray(tr[k]).tolist() for k in TRACE_KEYS}


def _trace_from_json(d: dict[str, Any]) -> dict[str, Any]:
    dt = {"pos": np.float32, "flag_home": np.float32, "flag_pos": np.float32, "intent": np.float32,
          "alive": bool, "tagged": bool, "carrying": bool, "eff_macro": np.int16, "action": np.int16}
    return {k: np.asarray(d[k], dtype=dt[k]) for k in TRACE_KEYS}


def run_calibration() -> int:
    from experiments import run_state as rs
    from experiments import seed_registry as sr
    import torch

    contracts = json.loads(CAL_CONTRACT.read_text(encoding="utf-8"))
    if contracts.get("DECISION") != "CONTRACTS_PASS":
        raise SystemExit(f"REFUSING: calibration contracts did not pass ({contracts.get('DECISION')!r})")
    if CAL_RESULT.exists():
        raise SystemExit(f"REFUSING: calibration result already exists: {CAL_RESULT}")
    ok, msg = _registered(EXP_CAL, CAL_SEEDS[0], len(CAL_SEEDS))
    if not ok:
        raise SystemExit(f"REFUSING: calibration block not reserved as frozen: {msg}")

    g = P.pole_genomes()
    keys = [(pole, c, a, s) for pole in P.POLES for (c, a) in cal_cells() for s in CAL_SEEDS]
    state = rs.RunState(SD, "LEARNED_COMPOSITION_PROBE_CALIBRATION").begin(spec=SPEC_PATH.name, n_episodes=len(keys))
    done = _load_partial(CAL_PARTIAL)
    if done:
        print(f"  resuming: {len(done)}/{len(keys)} episodes recorded", flush=True)
    pending = [k for k in keys if k not in done]
    with CAL_PARTIAL.open("a", encoding="utf-8") as fh, torch.no_grad():
        for key in tqdm_iter(pending, desc="probe calibration (scripted, cuda)", total=len(pending), unit="ep"):
            pole, comp, asg, seed = key
            tr = P.run_scripted_episode(comp, pole, seed, g, DEVICE, max_ticks=T_TRUNC, roles=P.roles_for(comp, asg))
            row = {"composition": comp, "assignment": asg, "pole": pole, "seed": seed, "steps": int(tr["steps"]),
                   "true_defenders": D_OF[comp], **opening_reading(tr)}
            rec = {"key": list(key), "row": row, "trace": _trace_to_json(tr)}
            done[key] = rec
            fh.write(json.dumps(rec) + "\n"); fh.flush()

    rows = [done[k]["row"] for k in keys]
    traces = [_trace_from_json(done[k]["trace"]) for k in keys]
    fields = list(rows[0])
    with CAL_ROWS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    np.savez_compressed(CAL_TRACES, **pack_traces(traces, ["|".join(map(str, k)) for k in keys]))

    # independent re-derivation of every reading from the saved traces
    worst = 0.0
    for row, tr in zip(rows, traces):
        ref = reference_reading(tr)
        for k, v in ref.items():
            if not (np.isnan(v) and np.isnan(row[k])):
                worst = max(worst, abs(v - row[k]))
    if worst > 1e-9:
        raise SystemExit(f"ABORT: independent instrument disagrees with the vectorised one by {worst:.3e}")
    print(f"  independent instrument re-derivation PASS (max |diff| {worst:.2e} over {len(rows)} episodes)", flush=True)

    primary = evaluate_validity(rows, "share_pos_open")
    secondary = evaluate_validity(rows, "share_int_open")
    alt = evaluate_validity(rows, "share_pos_alt")
    payload = {
        "record_id": "LEARNED_COMPOSITION_PROBE_CALIBRATION_RESULT", "implements": SPEC_PATH.name, "utc": _now(),
        "device": DEVICE, "seed_block": {"base": CAL_SEEDS[0], "n": len(CAL_SEEDS), "class": SEED_CLASS},
        "instrument": {"R_pos": R_POS, "window_primary": list(WIN_PRIMARY), "window_alt": list(WIN_ALT), "truncate_at_tick": T_TRUNC},
        "kill_rule": {"min_recovery": MIN_RECOVERY, "tolerance_defenders": TOL_D},
        "PRIMARY_position_opening": primary,
        "SECONDARY_intent_opening_pre_declared_not_required": secondary,
        "STABILITY_position_alt_window": alt,
        "VERDICT": "INSTRUMENT_VALID_OPENING" if primary["valid"] else "INSTRUMENT_NOT_VALID",
        "independent_reference_check": {"max_abs_diff": worst, "episodes": len(rows)},
        "traces_sha256": _sha(CAL_TRACES),
        "claim_boundary": "Validated on SCRIPTED roles only, in the opening window only. Says nothing about sustained composition.",
    }
    plan = build_plan(CAL_ROWS, len(rows), CAL_SEEDS, ("composition", "assignment", "pole"), EXP_CAL)
    rs.seal(out_path=CAL_RESULT, payload=payload, plan=plan, state=state, strict=False)
    sealed = json.loads(CAL_RESULT.read_text(encoding="utf-8"))
    sr.set_status(EXP_CAL, "SPENT", note=f"sealed {sealed.get('status')}; {payload['VERDICT']}")
    print(json.dumps({"VERDICT": payload["VERDICT"], "status": sealed.get("status"),
                      "primary": {p: {k: primary[p][k] for k in ("V1_monotone", "V2_loo_within1", "V3_cross_assignment_within1", "pole_valid")} for p in P.POLES},
                      "secondary_valid": secondary["valid"]}, indent=2))
    return 0 if payload["VERDICT"] == "INSTRUMENT_VALID_OPENING" else 2


# --------------------------------------------------------- confirmatory stage
# Amendment (Option B, PI direction 2026-09-20, see CONFIRM_SPEC): the original
# calibration sealed INSTRUMENT_NOT_VALID for the primary (share_pos_open), but two
# pre-declared, non-gating readings (share_pos_alt, share_int_open) passed the same
# rule on the same data. Promoting either now would be validity-by-selection. This
# stage re-runs the UNCHANGED kill rule against both survivors, independently, on a
# FRESH seed block that had no part in choosing them.

def contracts_confirmatory() -> dict:
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str, **data: Any) -> None:
        checks.append({"name": name, "gating": True, "passed": bool(passed), "detail": detail, **data})

    spec = json.loads(CONFIRM_SPEC.read_text(encoding="utf-8"))
    record("C0_AMENDMENT_SPEC_FROZEN", str(spec.get("status", "")).startswith("FROZEN"),
           f"spec status = {spec.get('status')!r}")

    # C1: the amendment's declared kill rule is byte-identical to the one already sealed
    cal = json.loads(CAL_RESULT.read_text(encoding="utf-8"))
    rule = spec["VALIDITY_KILL_RULE_UNCHANGED"]
    ok = (float(rule["min_recovery"]) == float(cal["kill_rule"]["min_recovery"]) == MIN_RECOVERY
          and int(rule["tolerance_defenders"]) == int(cal["kill_rule"]["tolerance_defenders"]) == TOL_D)
    record("C1_KILL_RULE_UNCHANGED_FROM_THE_SEALED_CALIBRATION", ok,
           f"min_recovery {rule['min_recovery']} tol {rule['tolerance_defenders']} vs sealed {cal['kill_rule']}")

    # C2: the two candidates are exactly the two readings the sealed calibration recorded as valid
    sec_valid = bool(cal["SECONDARY_intent_opening_pre_declared_not_required"]["valid"])
    alt_valid = bool(cal["STABILITY_position_alt_window"]["valid"])
    record("C2_CANDIDATES_MATCH_THE_SEALED_SURVIVORS", sec_valid and alt_valid,
           f"sealed calibration: intent_opening valid={sec_valid}, position_alt valid={alt_valid} "
           f"(both must be true -- these are the only two candidates this amendment is allowed to re-test)")

    # C3: the fresh block does not overlap either prior block (belt-and-braces; check_block below is authoritative)
    lo, n = spec["FRESH_SEED_BLOCK"]["base"], spec["FRESH_SEED_BLOCK"]["n"]
    hi = lo + n - 1
    disjoint = (hi < CAL_SEEDS[0] or lo > CAL_SEEDS[-1]) and (hi < PROBE_SEEDS[0] or lo > PROBE_SEEDS[-1])
    record("C3_FRESH_BLOCK_DISJOINT_FROM_PRIOR_BLOCKS", disjoint,
           f"{lo}..{hi} vs calibration {CAL_SEEDS[0]}..{CAL_SEEDS[-1]} and probe {PROBE_SEEDS[0]}..{PROBE_SEEDS[-1]}")

    # C4: the seed block is reserved to this experiment, unspent
    ok4, msg4 = _registered(EXP_CONFIRM, CONFIRM_SEEDS[0], len(CONFIRM_SEEDS))
    record("C4_SEED_BLOCK_OWNED_AND_UNSPENT", ok4, msg4)

    # C5: PPO unreachable (read-only guarantee; this stage loads no checkpoint)
    live_ppo = sorted(m for m in sys.modules if m.startswith(("rl.custom_ppo.trainer", "rl.custom_ppo.ppo_updater")))
    record("C5_NO_PPO_REACHABLE", not live_ppo, f"imported PPO/trainer modules: {live_ppo or 'none'}")

    n_failed = sum(1 for c in checks if not c["passed"])
    report = {"record_id": "LEARNED_COMPOSITION_PROBE_CONFIRMATORY_CALIBRATION_CONTRACT_RESULT", "utc": _now(),
              "implements": CONFIRM_SPEC.name, "n_checks": len(checks), "n_gating": len(checks), "n_failed": n_failed,
              "checks": checks, "DECISION": "CONTRACTS_PASS" if n_failed == 0 else "CONTRACT_FAILURE",
              "claim_boundary": "Contracts only. No confirmatory episode was scored and no fresh seed was spent."}
    CONFIRM_CONTRACT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'=' * 66}\n  CONFIRMATORY CONTRACTS: {report['DECISION']}  ({n_failed}/{len(checks)} gating failed)\n{'=' * 66}")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    return report


def run_confirmatory_calibration() -> int:
    from experiments import run_state as rs
    from experiments import seed_registry as sr
    import torch

    contracts = json.loads(CONFIRM_CONTRACT.read_text(encoding="utf-8"))
    if contracts.get("DECISION") != "CONTRACTS_PASS":
        raise SystemExit(f"REFUSING: confirmatory contracts did not pass ({contracts.get('DECISION')!r})")
    if CONFIRM_RESULT.exists():
        raise SystemExit(f"REFUSING: confirmatory result already exists: {CONFIRM_RESULT}")
    ok, msg = _registered(EXP_CONFIRM, CONFIRM_SEEDS[0], len(CONFIRM_SEEDS))
    if not ok:
        raise SystemExit(f"REFUSING: confirmatory block not reserved as frozen: {msg}")

    g = P.pole_genomes()
    keys = [(pole, c, a, s) for pole in P.POLES for (c, a) in cal_cells() for s in CONFIRM_SEEDS]
    state = rs.RunState(SD, "LEARNED_COMPOSITION_PROBE_CONFIRMATORY_CALIBRATION").begin(
        spec=CONFIRM_SPEC.name, n_episodes=len(keys))
    done = _load_partial(CONFIRM_PARTIAL)
    if done:
        print(f"  resuming: {len(done)}/{len(keys)} episodes recorded", flush=True)
    pending = [k for k in keys if k not in done]
    with CONFIRM_PARTIAL.open("a", encoding="utf-8") as fh, torch.no_grad():
        for key in tqdm_iter(pending, desc="confirmatory calibration (scripted, cuda)", total=len(pending), unit="ep"):
            pole, comp, asg, seed = key
            tr = P.run_scripted_episode(comp, pole, seed, g, DEVICE, max_ticks=T_TRUNC, roles=P.roles_for(comp, asg))
            row = {"composition": comp, "assignment": asg, "pole": pole, "seed": seed, "steps": int(tr["steps"]),
                   "true_defenders": D_OF[comp], **opening_reading(tr)}
            rec = {"key": list(key), "row": row, "trace": _trace_to_json(tr)}
            done[key] = rec
            fh.write(json.dumps(rec) + "\n"); fh.flush()

    rows = [done[k]["row"] for k in keys]
    traces = [_trace_from_json(done[k]["trace"]) for k in keys]
    fields = list(rows[0])
    with CONFIRM_ROWS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    np.savez_compressed(CONFIRM_TRACES, **pack_traces(traces, ["|".join(map(str, k)) for k in keys]))

    worst = 0.0
    for row, tr in zip(rows, traces):
        ref = reference_reading(tr)
        for k, v in ref.items():
            if not (np.isnan(v) and np.isnan(row[k])):
                worst = max(worst, abs(v - row[k]))
    if worst > 1e-9:
        raise SystemExit(f"ABORT: independent instrument disagrees with the vectorised one by {worst:.3e}")
    print(f"  independent instrument re-derivation PASS (max |diff| {worst:.2e} over {len(rows)} episodes)", flush=True)

    validity = {key: evaluate_validity(rows, key) for key in CONFIRM_CANDIDATES}
    confirmed = [CONFIRM_LABEL[key] for key in CONFIRM_CANDIDATES if validity[key]["valid"]]
    if len(confirmed) == 2:
        verdict = "BOTH_CONFIRMED"
    elif len(confirmed) == 1:
        verdict = f"{confirmed[0].upper()}_ONLY_CONFIRMED"
    else:
        verdict = "NONE_CONFIRMED"

    fyi_primary = evaluate_validity(rows, "share_pos_open")   # disclosed, NOT gating: the dead primary window
    payload = {
        "record_id": "LEARNED_COMPOSITION_PROBE_CONFIRMATORY_CALIBRATION_RESULT", "implements": CONFIRM_SPEC.name,
        "utc": _now(), "device": DEVICE, "seed_block": {"base": CONFIRM_SEEDS[0], "n": len(CONFIRM_SEEDS), "class": SEED_CLASS},
        "kill_rule": {"min_recovery": MIN_RECOVERY, "tolerance_defenders": TOL_D},
        "POSITION_ALT_WINDOW_5_15": validity["share_pos_alt"],
        "INTENT_OPENING_WINDOW_0_10": validity["share_int_open"],
        "FYI_PRIMARY_POSITION_OPEN_NOT_GATING": fyi_primary,
        "CONFIRMED_INSTRUMENTS": confirmed,
        "VERDICT": verdict,
        "independent_reference_check": {"max_abs_diff": worst, "episodes": len(rows)},
        "traces_sha256": _sha(CONFIRM_TRACES),
        "claim_boundary": "Scripted roles only, opening window only, cuda, seed block spent for the first time by "
                          "this spec. No learned policy was read. This verdict gates which instrument(s) a future, "
                          "separately-declared Stage 2 may use; it does not itself authorize Stage 2 to run.",
    }
    plan = build_plan(CONFIRM_ROWS, len(rows), CONFIRM_SEEDS, ("composition", "assignment", "pole"), EXP_CONFIRM,
                      spec_path=CONFIRM_SPEC)
    rs.seal(out_path=CONFIRM_RESULT, payload=payload, plan=plan, state=state, strict=False)
    sealed = json.loads(CONFIRM_RESULT.read_text(encoding="utf-8"))
    sr.set_status(EXP_CONFIRM, "SPENT", note=f"sealed {sealed.get('status')}; {verdict}")
    print(json.dumps({"VERDICT": verdict, "status": sealed.get("status"), "confirmed": confirmed}, indent=2))
    return 0 if confirmed else 2


# ------------------------------------------------------------------ stage 2
# Authorized (PI direction 2026-09-20, see STAGE2_SPEC): a dual-instrument diagnostic, both
# confirmed survivors used as co-equal, separately-reported measurements. No averaging, no
# instrument selection, no refitting the calibration mapping on confirmatory or Stage-2 data.

def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _checkpoint_pins() -> dict[str, str]:
    spec = _load_json(SPEC_PATH)
    return {n: d["sha256"] for n, d in spec["TARGET_POLICIES_PINNED"].items() if n in P.LEARNED}


def _calibration_means() -> dict[str, dict[str, dict[str, float]]]:
    """{d_hat_key: {pole: {composition: mean}}}, read ONLY from the original sealed calibration."""
    cal = _load_json(CAL_RESULT)
    return {d_hat_key: {pole: cal[cal_section][pole]["class_means_prefix"] for pole in P.POLES}
            for d_hat_key, (_row_key, cal_section, _label) in STAGE2_INSTRUMENTS.items()}


def _share0_rows() -> dict[tuple, dict]:
    with P.SHARE0_ROWS.open(encoding="utf-8") as fh:
        return {(r["policy"], r["pole"], int(r["seed"])): r for r in csv.DictReader(fh)}


def _bootstrap_stat(values: np.ndarray, stat_fn) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"value": None, "lcb95": None, "ucb95": None, "n": 0}
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, values.size, size=(N_BOOT, values.size))
    stats = stat_fn(values[idx], axis=1)
    lo, hi = np.percentile(stats, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"value": round(float(stat_fn(values)), 6), "lcb95": round(float(lo), 6),
            "ucb95": round(float(hi), 6), "n": int(values.size)}


def contracts_stage2() -> dict:
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str, **data: Any) -> None:
        checks.append({"name": name, "gating": True, "passed": bool(passed), "detail": detail, **data})

    spec = _load_json(STAGE2_SPEC)
    record("C0_STAGE2_SPEC_FROZEN", str(spec.get("status", "")).startswith("FROZEN"),
           f"spec status = {spec.get('status')!r}")

    confirm = _load_json(CONFIRM_RESULT)
    confirmed = list(confirm.get("CONFIRMED_INSTRUMENTS", []))
    declared = sorted(v[2] for v in STAGE2_INSTRUMENTS.values())
    record("C1_GATE_CONDITION_MET_AND_INSTRUMENTS_MATCH",
           confirm.get("status") == "SEALED" and bool(confirmed) and sorted(confirmed) == declared,
           f"confirmatory status={confirm.get('status')} confirmed={confirmed} vs spec declares {declared}")

    got_sha = _sha(CAL_RESULT)
    want_sha = spec["CALIBRATION_MAPPING_FROZEN"]["source_sha256"]
    record("C2_CALIBRATION_MAPPING_SOURCE_UNCHANGED_SINCE_SEALING", got_sha == want_sha,
           f"{got_sha} vs pinned {want_sha}")

    pins = _checkpoint_pins()
    got_pins = {n: _sha(p) for n, p in P.CHECKPOINTS.items()}
    record("C3_CHECKPOINTS_MATCH_THE_PINNED_HASHES", got_pins == pins, f"{got_pins} vs pinned {pins}")

    ok4, msg4 = _registered(EXP_PROBE, PROBE_SEEDS[0], len(PROBE_SEEDS))
    record("C4_SEED_BLOCK_OWNED_AND_UNSPENT", ok4, msg4)

    # C5/C6: parity on already-spent SHARE0 seeds -- a real read-only run, no probe-block seed spent
    import torch
    g = P.pole_genomes()
    pols = P.load_policies(DEVICE)
    before = {n: _param_digest(p) for n, p in pols.items()}
    share0 = _share0_rows()
    mismatches = []
    with torch.no_grad():
        for policy in P.LEARNED:
            for pole in P.POLES:
                for seed in PARITY_SEEDS:
                    full = P.run_learned_episode(pols[policy], pole, seed, g, DEVICE)
                    trunc = P.run_learned_episode(pols[policy], pole, seed, g, DEVICE, max_ticks=T_TRUNC)
                    rec = share0[(policy, pole, seed)]
                    got_tuple = (full["blue"], full["red"], full["win"], full["margin"])
                    want_tuple = (int(rec["blue"]), int(rec["red"]), int(rec["win"]), int(rec["margin"]))
                    if got_tuple != want_tuple:
                        mismatches.append(f"{policy}/{pole}/{seed}: outcome {got_tuple} != recorded {want_tuple}")
                        continue
                    n = min(T_TRUNC, len(trunc["pos"]))
                    eq = all(np.array_equal(full[k][:n], trunc[k][:n])
                             for k in ("pos", "alive", "tagged", "carrying", "intent", "eff_macro"))
                    if not eq:
                        mismatches.append(f"{policy}/{pole}/{seed}: truncated run diverges from the first {n} "
                                          f"ticks of the full run")
    after = {n: _param_digest(p) for n, p in pols.items()}
    record("C5_PARITY_16_FULL_EPISODES_MATCH_SHARE0_AND_TRUNCATION_IS_EXACT", not mismatches,
           "; ".join(mismatches) if mismatches else "16/16 episodes reproduced the recorded outcome exactly; "
                                                     "truncated run equals the first ticks of the full run")
    record("C6_POLICY_PARAMETERS_UNCHANGED_BY_THE_PARITY_RUN", before == after,
           f"parameter digests equal before/after: {before == after}")

    n_failed = sum(1 for c in checks if not c["passed"])
    report = {"record_id": "LEARNED_COMPOSITION_PROBE_STAGE2_CONTRACT_RESULT", "utc": _now(),
              "implements": STAGE2_SPEC.name, "n_checks": len(checks), "n_gating": len(checks),
              "n_failed": n_failed, "checks": checks,
              "DECISION": "CONTRACTS_PASS" if n_failed == 0 else "CONTRACT_FAILURE",
              "claim_boundary": "Contracts only, including a real parity run on 16 already-spent SHARE0 seeds. "
                                "No seed from the 20700101-20700228 probe block was spent."}
    PROBE_CONTRACT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'=' * 66}\n  STAGE 2 CONTRACTS: {report['DECISION']}  ({n_failed}/{len(checks)} gating failed)\n{'=' * 66}")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    return report


def run_stage2() -> int:
    from experiments import run_state as rs
    from experiments import seed_registry as sr
    import torch

    contracts = _load_json(PROBE_CONTRACT)
    if contracts.get("DECISION") != "CONTRACTS_PASS":
        raise SystemExit(f"REFUSING: stage 2 contracts did not pass ({contracts.get('DECISION')!r})")
    if PROBE_RESULT.exists():
        raise SystemExit(f"REFUSING: stage 2 result already exists: {PROBE_RESULT}")
    ok, msg = _registered(EXP_PROBE, PROBE_SEEDS[0], len(PROBE_SEEDS))
    if not ok:
        raise SystemExit(f"REFUSING: probe block not reserved as frozen: {msg}")

    g = P.pole_genomes()
    pols = P.load_policies(DEVICE)
    before = {n: _param_digest(p) for n, p in pols.items()}

    keys = [(policy, pole, s) for policy in P.LEARNED for pole in P.POLES for s in PROBE_SEEDS]
    state = rs.RunState(SD, "LEARNED_COMPOSITION_PROBE").begin(spec=STAGE2_SPEC.name, n_episodes=len(keys))
    done = _load_partial(PROBE_PARTIAL)
    if done:
        print(f"  resuming: {len(done)}/{len(keys)} episodes recorded", flush=True)
    pending = [k for k in keys if k not in done]
    with PROBE_PARTIAL.open("a", encoding="utf-8") as fh, torch.no_grad():
        for key in tqdm_iter(pending, desc="stage 2 learned probe (cuda)", total=len(pending), unit="ep"):
            policy, pole, seed = key
            tr = P.run_learned_episode(pols[policy], pole, seed, g, DEVICE, max_ticks=T_TRUNC)
            row = {"policy": policy, "pole": pole, "seed": seed, "steps": int(tr["steps"]),
                   "win": int(tr["win"]), "margin": int(tr["margin"]), **opening_reading(tr)}
            rec = {"key": list(key), "row": row, "trace": _trace_to_json(tr)}
            done[key] = rec
            fh.write(json.dumps(rec) + "\n"); fh.flush()

    after = {n: _param_digest(p) for n, p in pols.items()}
    if before != after:
        raise SystemExit("ABORT: policy parameters changed during the read-only probe")

    rows = [done[k]["row"] for k in keys]
    traces = [_trace_from_json(done[k]["trace"]) for k in keys]
    fields = list(rows[0])
    with PROBE_ROWS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)
    np.savez_compressed(PROBE_TRACES, **pack_traces(traces, ["|".join(map(str, k)) for k in keys]))

    worst = 0.0
    for row, tr in zip(rows, traces):
        ref = reference_reading(tr)
        for k, v in ref.items():
            if not (np.isnan(v) and np.isnan(row[k])):
                worst = max(worst, abs(v - row[k]))
    if worst > 1e-9:
        raise SystemExit(f"ABORT: independent instrument disagrees with the vectorised one by {worst:.3e}")
    print(f"  independent instrument re-derivation PASS (max |diff| {worst:.2e} over {len(rows)} episodes)", flush=True)

    # invert readings to D_hat using ONLY the frozen original calibration mapping -- never refit here
    means = _calibration_means()
    for row in rows:
        for d_hat_key, (row_key, _sec, _label) in STAGE2_INSTRUMENTS.items():
            row[d_hat_key] = d_hat_from(row[row_key], means[d_hat_key][row["pole"]])

    def cell(policy: str, pole: str, d_hat_key: str) -> np.ndarray:
        return np.asarray([r[d_hat_key] for r in rows if r["policy"] == policy and r["pole"] == pole])

    def stat_block(vals: np.ndarray) -> dict:
        return {"mean": _bootstrap(vals), "median": _bootstrap_stat(vals, np.median),
                "prop_attack_heavy_D_le_1_5": round(float(np.mean(vals <= BAND_ATTACK_HEAVY)), 4),
                "dist_from_5A1D_target_abs_D_minus_1": _bootstrap(np.abs(vals - 1.0)), "n": int(vals.size)}

    per_cell: dict[str, dict[str, dict[str, dict]]] = {}
    for d_hat_key in STAGE2_INSTRUMENTS:
        for policy in P.LEARNED:
            for pole in P.POLES:
                per_cell.setdefault(d_hat_key, {}).setdefault(policy, {})[pole] = stat_block(cell(policy, pole, d_hat_key))

    def paired(d_hat_key: str, pole: str) -> dict:
        a = {r["seed"]: r[d_hat_key] for r in rows if r["policy"] == "pi_A" and r["pole"] == pole}
        b = {r["seed"]: r[d_hat_key] for r in rows if r["policy"] == "pi_B" and r["pole"] == pole}
        return _bootstrap(np.asarray([b[s] - a[s] for s in PROBE_SEEDS]))

    contrast = {d_hat_key: {pole: paired(d_hat_key, pole) for pole in P.POLES} for d_hat_key in STAGE2_INSTRUMENTS}

    def band(mean_d_hat: float) -> str:
        if mean_d_hat <= BAND_ATTACK_HEAVY:
            return "attack_heavy"
        if mean_d_hat < BAND_INTERMEDIATE:
            return "intermediate"
        return "balanced_or_defence_heavy"

    concordance = {}
    for policy in P.LEARNED:
        for pole in P.POLES:
            b_pos = band(per_cell["d_hat_pos"][policy][pole]["mean"]["mean"])
            b_int = band(per_cell["d_hat_int"][policy][pole]["mean"]["mean"])
            concordance[f"{policy}_{pole}"] = {"position_band": b_pos, "intent_band": b_int, "agree": b_pos == b_int}

    payload = {
        "record_id": "LEARNED_COMPOSITION_PROBE_RESULT", "implements": STAGE2_SPEC.name, "utc": _now(),
        "device": DEVICE, "seed_block": {"base": PROBE_SEEDS[0], "n": len(PROBE_SEEDS), "class": SEED_CLASS},
        "calibration_mapping_source": CAL_RESULT.name, "calibration_mapping_source_sha256": _sha(CAL_RESULT),
        "POSITION_ALT_WINDOW_5_15": per_cell["d_hat_pos"],
        "INTENT_OPENING_WINDOW_0_10": per_cell["d_hat_int"],
        "PAIRED_CONTRAST_PI_B_MINUS_PI_A": {"POSITION_ALT_WINDOW_5_15": contrast["d_hat_pos"],
                                            "INTENT_OPENING_WINDOW_0_10": contrast["d_hat_int"]},
        "INSTRUMENT_CONCORDANCE_DESCRIPTIVE_NOT_GATING": concordance,
        "read_only_guarantee": {"param_digest_equal_before_after": before == after,
                                "checkpoint_sha256_matches_pinned": {n: _sha(p) for n, p in P.CHECKPOINTS.items()} == _checkpoint_pins()},
        "independent_reference_check": {"max_abs_diff": worst, "episodes": len(rows)},
        "traces_sha256": _sha(PROBE_TRACES),
        "claim_boundary": "Observational, scripted-role-calibrated, opening window only (ticks 0-15 across the two "
                          "instruments). A learned role-composition claim is licensed only where "
                          "INSTRUMENT_CONCORDANCE_DESCRIPTIVE_NOT_GATING reports agree=true for that (policy, pole) "
                          "cell; otherwise the reading is instrument-dependent / inconclusive for that cell.",
    }
    plan = build_plan(PROBE_ROWS, len(rows), PROBE_SEEDS, ("policy", "pole"), EXP_PROBE, spec_path=STAGE2_SPEC)
    rs.seal(out_path=PROBE_RESULT, payload=payload, plan=plan, state=state, strict=False)
    sealed = json.loads(PROBE_RESULT.read_text(encoding="utf-8"))
    sr.set_status(EXP_PROBE, "SPENT", note=f"sealed {sealed.get('status')}")
    print(json.dumps({"status": sealed.get("status"), "concordance": concordance}, indent=2))
    return 0 if sealed.get("status") == "SEALED" else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("contracts_cal", "calibrate", "contracts_probe", "probe",
                                        "contracts_confirm", "confirm"), required=True)
    args = ap.parse_args()
    if args.stage == "contracts_cal":
        return 0 if contracts_calibration()["DECISION"] == "CONTRACTS_PASS" else 2
    if args.stage == "calibrate":
        return run_calibration()
    if args.stage == "contracts_confirm":
        return 0 if contracts_confirmatory()["DECISION"] == "CONTRACTS_PASS" else 2
    if args.stage == "confirm":
        return run_confirmatory_calibration()
    if args.stage == "contracts_probe":
        return 0 if contracts_stage2()["DECISION"] == "CONTRACTS_PASS" else 2
    if args.stage == "probe":
        return run_stage2()
    raise SystemExit(f"unhandled stage {args.stage!r}")


if __name__ == "__main__":
    raise SystemExit(main())
