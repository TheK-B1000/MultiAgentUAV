r"""Action-interface decomposition: which component kills 4v4 GUARD@A?

Implements ACTION_INTERFACE_DECOMP_SPEC.json.

Arms
  ORIGINAL     continuous scripted teacher (every tick)
  SPATIAL_ONLY W50/semantic adapter, but force a new commit every tick
  COMMIT_ONLY  continuous targets, refreshed only on the student GO_TO schedule
  FULL         adapter + real commitment (same as PROJECTED_TEACHER_ORACLE)

Primary cell: 4v4 GUARD Pole A. Control: 2v2 GUARD Pole A.
Seeds: 17600001..17600024 (oracle block) for paired comparison.

    python -m experiments.eval_action_interface_decomp --promote
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "ACTION_INTERFACE_DECOMP_SPEC.json"
LABEL = "ACTION_INTERFACE_DECOMP"
ARMS = ("ORIGINAL", "SPATIAL_ONLY", "COMMIT_ONLY", "FULL")
FIELDS = ["scale", "strategy", "pole", "arm", "seed", "blue", "red", "win",
          "n_decisions", "n_waypoint_actions"]


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _boot(d: np.ndarray, n_boot=20000, alpha=0.05, seed=7) -> dict:
    if d.size == 0:
        return {"mean": None}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    b = d[idx].mean(axis=1)
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": round(float(d.mean()), 6), "lcb95": round(float(lo), 6),
            "ucb95": round(float(hi), 6), "n": int(d.size)}


def _make_env(scale, seed, device):
    import experiments.strategic_demand_searcher as S
    from experiments.opponent_spec import pole_A_genome
    from experiments.strategic_demand_searcher import apply_genome_to_core
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    S.AGENTS = scale
    g = pole_A_genome(scale)
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=scale, max_red_agents=scale,
        map_set="train", map_layout=S.MAP, max_decision_steps=S.MAX_STEPS,
        aquaticus_profile=True, rules_profile="OURS", device=device, seed=seed,
        obstacle_obs_channel=True, tag_telemetry_enabled=True,
        own_flag_home_required_to_score=True, **S.RULESET)
    env = GPUCTFVecEnv(cfg)
    core = env.core
    opp = g.base_opponent
    env.env_method("set_phase", opp)
    env.env_method("set_next_opponent", "SCRIPTED", opp)
    apply_genome_to_core(core, g)
    core.blue_scripted = True
    core.set_blue_style(S.GUARD)
    env.reset()
    apply_genome_to_core(core, g)
    core.drain_tag_events()
    return env, core, S, g


def run_episode(scale: int, arm: str, seed: int, device: str) -> dict:
    from experiments.teacher_action_adapter import adapt
    from macro_actions import MacroAction

    env, core, S, _g = _make_env(scale, seed, device)
    n_dec = n_wp = 0
    try:
        W = core._macro_targets.detach().cpu().numpy()
        go_to_horizon = int(core.cfg.macro_commit_go_to_ticks)
        action = np.zeros((scale, 2), dtype=np.int64)
        have = False
        term = None

        # COMMIT_ONLY: hold continuous teacher targets across a student-like schedule.
        held_tx = None  # torch [1, N]
        held_ty = None
        commit_left = np.zeros(scale, dtype=np.int32)
        orig_get = core._get_scripted_targets

        def held_get(side: str):
            if side != "blue" or held_tx is None:
                return orig_get(side)
            return held_tx, held_ty

        if arm == "COMMIT_ONLY":
            core._get_scripted_targets = held_get  # type: ignore[method-assign]
            core.blue_scripted = True

        for _t in range(S.MAX_STEPS):
            if arm == "ORIGINAL":
                core.blue_scripted = True
                env.step_async(env.action_space.sample() * 0)

            elif arm == "SPATIAL_ONLY":
                # Spatial quantization WITHOUT temporal hold: force every tick to
                # be a decision boundary, then adapt and issue.
                core.blue_commit_ticks_left[:] = 0
                core.blue_scripted = True
                btx, bty = orig_get("blue")
                core.blue_scripted = False
                for i in range(scale):
                    a = adapt(core, float(btx[0, i]), float(bty[0, i]), i, W)
                    n_dec += 1
                    if a.uses_waypoint:
                        n_wp += 1
                        action[i] = (a.macro, a.target_idx)
                    elif a.macro is not None:
                        action[i] = (a.macro, 0)
                    else:
                        action[i] = (0, 0)
                env.step_async(action.reshape(-1))

            elif arm == "COMMIT_ONLY":
                # Refresh continuous teacher target only when the virtual commit
                # clock hits zero; otherwise hold last continuous target.
                live_tx, live_ty = orig_get("blue")
                if held_tx is None:
                    held_tx = live_tx.clone()
                    held_ty = live_ty.clone()
                for i in range(scale):
                    if commit_left[i] <= 0:
                        held_tx[0, i] = live_tx[0, i]
                        held_ty[0, i] = live_ty[0, i]
                        commit_left[i] = go_to_horizon
                        n_dec += 1
                    else:
                        commit_left[i] -= 1
                core.blue_scripted = True
                env.step_async(env.action_space.sample() * 0)

            elif arm == "FULL":
                core.blue_scripted = False
                nc = (core.blue_commit_ticks_left[0] <= 0).detach().cpu().numpy()
                core.blue_scripted = True
                btx, bty = orig_get("blue")
                core.blue_scripted = False
                for i in range(scale):
                    if have and not bool(nc[i]):
                        continue
                    a = adapt(core, float(btx[0, i]), float(bty[0, i]), i, W)
                    n_dec += 1
                    if a.uses_waypoint:
                        n_wp += 1
                        action[i] = (a.macro, a.target_idx)
                    elif a.macro is not None:
                        action[i] = (a.macro, 0)
                    else:
                        action[i] = (0, 0)
                have = True
                env.step_async(action.reshape(-1))
            else:
                raise SystemExit(f"unknown arm {arm!r}")

            _o, _r, done, info = env.step_wait()
            if bool(np.asarray(done).any()):
                i0 = info[0] if isinstance(info, (list, tuple)) else info
                er = (i0 or {}).get("episode_result") or {}
                term = (int(er.get("blue_score", 0)), int(er.get("red_score", 0)))
                break
        if term is None:
            term = (int(core.blue_score[0]), int(core.red_score[0]))
        b, r = term
        return {
            "scale": f"{scale}v{scale}", "strategy": "GUARD", "pole": "A",
            "arm": arm, "seed": seed, "blue": b, "red": r, "win": int(b > r),
            "n_decisions": n_dec, "n_waypoint_actions": n_wp,
            "go_to_horizon": go_to_horizon,
        }
    finally:
        env.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scales", type=int, nargs="+", default=[4, 2])
    ap.add_argument("--n-seeds", type=int, default=24)
    ap.add_argument("--promote", action="store_true")
    a = ap.parse_args()

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: spec not frozen: {spec.get('status')!r}")
    lo, hi = spec["SEEDS"]["block"]
    seeds = list(range(lo, lo + a.n_seeds))
    if seeds[-1] > hi:
        raise SystemExit(f"REFUSING: seed range exceeds frozen block {lo}..{hi}")

    ROWS = SD / f"{LABEL.lower()}_rows.csv"
    LIVE = SD / f"{LABEL}_LIVE_STATUS.json"
    LOCK = SD / f"{LABEL}.run.lock"
    if LOCK.is_file():
        raise SystemExit(f"REFUSING: {LOCK.name} exists.")
    LOCK.write_text(json.dumps({"pid": os.getpid(), "utc": _now()}), encoding="utf-8")

    from experiments.teacher_action_adapter import golden_anchors
    import experiments.strategic_demand_searcher as S0
    env, core, _, _ = _make_env(a.scales[0], seeds[0], a.device)
    try:
        fails = golden_anchors(core, core._macro_targets.detach().cpu().numpy(), 0)
        go_to_h = int(core.cfg.macro_commit_go_to_ticks)
    finally:
        env.close()
    if fails:
        LOCK.unlink(missing_ok=True)
        raise SystemExit("RULE 11 ADAPTER ANCHORS FAILED:\n  " + "\n  ".join(fails))

    # ORIGINAL must match strategic_demand_searcher on a few seeds.
    from experiments.opponent_spec import pole_A_genome
    S0.AGENTS = a.scales[0]
    g = pole_A_genome(a.scales[0])
    for seed in seeds[:3]:
        ref = S0.run_episode(style=S0.GUARD, genome=g, seed=seed, device=a.device)
        mine = run_episode(a.scales[0], "ORIGINAL", seed, a.device)
        if not (ref["win"] == mine["win"] and ref["blue_score"] == mine["blue"]
                and ref["red_score"] == mine["red"]):
            LOCK.unlink(missing_ok=True)
            raise SystemExit(
                f"HARNESS ANCHOR FAILED seed={seed}: ref "
                f"b={ref['blue_score']} r={ref['red_score']} w={ref['win']} "
                f"!= ORIGINAL b={mine['blue']} r={mine['red']} w={mine['win']}")

    print(f"{LABEL}  {_now()}")
    print("  rule-11 adapter anchors: PASS")
    print("  harness ORIGINAL == run_episode: PASS")
    print(f"  cells: GUARD@A x scales {a.scales} x arms {list(ARMS)}")
    print(f"  seeds {seeds[0]}..{seeds[-1]} (n={len(seeds)})  GO_TO horizon={go_to_h}")
    print("  Privileged-vs-Standard BC: PAUSED pending this diagnosis\n", flush=True)

    done_keys = set()
    rows = []
    if ROWS.is_file():
        with ROWS.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
                done_keys.add((r["scale"], r["arm"], int(r["seed"])))
        print(f"  RESUME: {len(done_keys)} episodes already on disk", flush=True)
    else:
        with ROWS.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=FIELDS).writeheader()

    # Cross-check FULL against sealed oracle rows when present (same seeds).
    oracle_rows = SD / "projected_teacher_oracle_rows.csv"
    oracle_guard_a = {}
    if oracle_rows.is_file():
        with oracle_rows.open(newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["strategy"] == "GUARD" and r["pole"] == "A":
                    oracle_guard_a[(r["scale"], r["arm"], int(r["seed"]))] = int(r["win"])

    from experiments.tqdm_loop import set_postfix, tqdm_iter
    cells = [(n, arm) for n in a.scales for arm in ARMS]
    t0 = time.time()
    bar = tqdm_iter(cells, desc=LABEL, unit="cell")
    for n, arm in bar:
        set_postfix(bar, f"{n}v{n} GUARD@A {arm}")
        wins = []
        for seed in seeds:
            key = (f"{n}v{n}", arm, seed)
            if key in done_keys:
                wins.append(int(next(
                    r["win"] for r in rows
                    if r["scale"] == f"{n}v{n}" and r["arm"] == arm
                    and int(r["seed"]) == seed)))
                continue
            rec = run_episode(n, arm, seed, a.device)
            # Drop helper field not in CSV schema
            rec.pop("go_to_horizon", None)
            wins.append(rec["win"])
            with ROWS.open("a", newline="", encoding="utf-8") as fh:
                csv.DictWriter(fh, fieldnames=FIELDS).writerow(rec)
                fh.flush(); os.fsync(fh.fileno())
            rows.append({k: str(v) for k, v in rec.items()})
        print(f"  {n}v{n} GUARD@A {arm:<12s} V={np.mean(wins):.4f}", flush=True)
        LIVE.write_text(json.dumps({
            "label": LABEL, "pid": os.getpid(),
            "current": f"{n}v{n} GUARD@A {arm}",
            "elapsed_s": round(time.time() - t0, 1),
            "heartbeat_utc": _now()}, indent=2), encoding="utf-8")

    def V(n, arm):
        d = {int(r["seed"]): int(r["win"]) for r in rows
             if r["scale"] == f"{n}v{n}" and r["arm"] == arm}
        return np.array([d[s] for s in seeds if s in d], dtype=float)

    out = {}
    print("\n  PAIRED LOSS vs ORIGINAL  (positive => arm weaker than ORIGINAL)")
    for n in a.scales:
        cell = {}
        vo = V(n, "ORIGINAL")
        for arm in ARMS:
            v = V(n, arm)
            m = min(vo.size, v.size)
            cell[arm] = {
                "V": round(float(v.mean()), 4) if v.size else None,
                "L_ORIGINAL_minus_arm": _boot(vo[:m] - v[:m]) if m else None,
            }
            L = cell[arm]["L_ORIGINAL_minus_arm"]
            print(f"    {n}v{n} {arm:<12s} V={cell[arm]['V']:.4f}  "
                  f"L={L['mean']:+.4f} [{L['lcb95']:+.4f},{L['ucb95']:+.4f}]")
        # Oracle cross-check for ORIGINAL / FULL -- Rule-12 HARD CONTRACT
        xcheck = {}
        hard_fail_lines = []
        for arm_oracle, arm_local in (("ORIGINAL", "ORIGINAL"),
                                      ("PROJECTED", "FULL")):
            mismatches = []
            compared = 0
            for seed in seeds:
                k = (f"{n}v{n}", arm_oracle, seed)
                if k not in oracle_guard_a:
                    hard_fail_lines.append(f"MISSING oracle row {k}")
                    continue
                compared += 1
                local = int(next(
                    r["win"] for r in rows
                    if r["scale"] == f"{n}v{n}" and r["arm"] == arm_local
                    and int(r["seed"]) == seed))
                if local != oracle_guard_a[k]:
                    msg = (f"{n}v{n} seed={seed}: decomp.{arm_local} win={local} "
                           f"!= oracle.{arm_oracle} win={oracle_guard_a[k]}")
                    mismatches.append(msg)
                    hard_fail_lines.append(msg)
            xcheck[f"{arm_local}_vs_oracle_{arm_oracle}"] = {
                "compared": compared, "mismatches": len(mismatches),
                "mismatch_seeds": mismatches,
            }
        cell["oracle_row_crosscheck"] = xcheck
        cell["rule12_hard_contract"] = {
            "passed": len(hard_fail_lines) == 0,
            "n_failures": len(hard_fail_lines),
        }
        if hard_fail_lines:
            audit = SD / f"{LABEL}_INTEGRITY_REQUIRED.json"
            audit.write_text(json.dumps({
                "record": f"{LABEL} Rule-12 oracle contract FAILED",
                "status": "INTEGRITY_REQUIRED", "utc": _now(),
                "implements": "ACTION_INTERFACE_DECOMP_RULE12_ORACLE_CONTRACT_AMENDMENT.json",
                "failures": hard_fail_lines,
                "reading_refused": True,
                "note": ("Overlapping arms disagree with the sealed oracle. "
                         "Do NOT interpret L_S/L_C/L_F. Audit the harness."),
            }, indent=2), encoding="utf-8")
            LOCK.unlink(missing_ok=True)
            raise SystemExit(
                f"RULE 12 HARD CONTRACT FAILED ({len(hard_fail_lines)} issues). "
                f"Interpretation refused. -> {audit}")

        # Binding diagnosis (precommitted) -- secondary to L_S/L_C/L_F
        Ls = {arm: cell[arm]["L_ORIGINAL_minus_arm"]["mean"] for arm in ARMS
              if arm != "ORIGINAL"}
        spatial = Ls["SPATIAL_ONLY"]
        commit = Ls["COMMIT_ONLY"]
        full = Ls["FULL"]
        # Collapse = large fraction of FULL's loss
        def frac(x):
            return None if abs(full) < 1e-9 else float(x / full)
        reading = {
            "spatial_frac_of_full_loss": frac(spatial),
            "commit_frac_of_full_loss": frac(commit),
            "L_S": cell["SPATIAL_ONLY"]["L_ORIGINAL_minus_arm"],
            "L_C": cell["COMMIT_ONLY"]["L_ORIGINAL_minus_arm"],
            "L_F": cell["FULL"]["L_ORIGINAL_minus_arm"],
            "note": "binder label is secondary; report L_S/L_C/L_F first",
        }
        if full <= 0.05:
            reading["binder"] = "NO_MATERIAL_FULL_LOSS -- unexpected vs oracle; audit"
        elif spatial >= 0.7 * full and commit < 0.4 * full:
            reading["binder"] = "SPATIAL"
        elif commit >= 0.7 * full and spatial < 0.4 * full:
            reading["binder"] = "COMMIT"
        elif spatial >= 0.5 * full and commit >= 0.5 * full:
            reading["binder"] = "EITHER_FACTOR_SUFFICIENT"
        elif spatial < 0.4 * full and commit < 0.4 * full and full > 0.1:
            reading["binder"] = "INTERACTION"
        else:
            reading["binder"] = "MIXED_OR_INCONCLUSIVE"
        cell["reading"] = reading
        out[f"{n}v{n}"] = cell
        print(f"    {n}v{n} L_S={reading['L_S']['mean']:+.4f} "
              f"L_C={reading['L_C']['mean']:+.4f} "
              f"L_F={reading['L_F']['mean']:+.4f}  "
              f"binder(secondary)={reading['binder']}")

    cfg_sig = json.dumps({"scales": a.scales, "n_seeds": a.n_seeds,
                          "device": a.device, "seed_lo": seeds[0],
                          "go_to_horizon": go_to_h}, sort_keys=True)
    run_id = (f"{_now().replace(':', '').replace('-', '')}_"
              f"{hashlib.sha256(cfg_sig.encode()).hexdigest()[:8]}")
    rec = {
        "record": f"{LABEL} GUARD@A action-interface binder probe",
        "status": "FROZEN_RESULT", "utc": _now(), "run_id": run_id,
        "arm": "DIAGNOSTIC", "confirmatory": False,
        "study_class": "MECHANISTIC_FOLLOW_UP",
        "not": "INDEPENDENT_CONFIRMATION",
        "implements": [SPEC.name,
                       "ACTION_INTERFACE_DECOMP_RULE12_ORACLE_CONTRACT_AMENDMENT.json"],
        "config_signature": json.loads(cfg_sig),
        "elapsed_s": round(time.time() - t0, 1),
        "pauses": ["Privileged-vs-Standard BC", "PPO", "SNR"],
        "primary_report": ["L_S", "L_C", "L_F"],
        "results": out,
    }
    p = SD / f"{LABEL}_{run_id}_RESULT.json"
    if p.exists():
        raise SystemExit(f"REFUSING: {p.name} exists.")
    p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    if a.promote:
        (SD / f"{LABEL}_CANONICAL.json").write_text(json.dumps({
            "record": f"CANONICAL {LABEL}", "points_to": p.name,
            "run_id": run_id, "promoted_utc": _now(),
            "config_signature": json.loads(cfg_sig)}, indent=2), encoding="utf-8")
    LOCK.unlink(missing_ok=True)
    print(f"\n  -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
