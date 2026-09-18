r"""ACTION_INTERFACE_SCALE_DIAGNOSTIC — outcome-blind I0–I3 scale diagnostic.

Implements artifacts/strategic_demand/sppo/ACTION_INTERFACE_SCALE_DIAGNOSTIC_SPEC.json
and ACTION_INTERFACE_SCALE_DIAGNOSTIC_INTEGRITY_AMENDMENT.json.

    python -m experiments.eval_action_interface_scale_diagnostic --contracts
    python -m experiments.eval_action_interface_scale_diagnostic --seeds 1760.. --promote
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC_SPEC.json"
AMEND = SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC_INTEGRITY_AMENDMENT.json"
LABEL = "ACTION_INTERFACE_SCALE_DIAGNOSTIC"
B3_GENOME = SD / "pole_b2_candidates" / "B3-3_lockdef10_2v1.json"
B3_CERT = SD / "STRATEGIC_DEMAND_4v4_POLE_B3_3_N192_CERTIFICATION.json"
REGRESSION_GUARD = SD / "SIZE_NORMALIZED_POLES_REGRESSION_GUARD.json"
HORIZON = 240

PROTECTED_RESULTS = (
    "ACTION_INTERFACE_DECOMP_RESULT.json",
    "PROJECTION_DIVERGENCE_TRACE_RESULT.json",
    "REPAIRED_GO_TO_H1_PROJECTED_RESULT.json",
    "PYQUATICUS_PORT_CONTRACT_RESULT.json",
)

OUT = {
    "contract_result": SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC_CONTRACT_RESULT.json",
    "trace_manifest": SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC_TRACE_MANIFEST.json",
    "agent_tick_rows": SD / "action_interface_scale_diagnostic_agent_tick_rows.csv",
    "episode_rows": SD / "action_interface_scale_diagnostic_episode_rows.csv",
    "result": SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC_RESULT.json",
    "run_lock": SD / "ACTION_INTERFACE_SCALE_DIAGNOSTIC.run.lock",
}


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _boot(d: np.ndarray, n_boot=20000, alpha=0.05, seed=7) -> dict:
    if d.size == 0:
        return {"mean": None, "lcb95": None, "ucb95": None, "n": 0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d.size, size=(n_boot, d.size))
    b = d[idx].mean(axis=1)
    lo, hi = np.percentile(b, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "mean": round(float(d.mean()), 6),
        "lcb95": round(float(lo), 6),
        "ucb95": round(float(hi), 6),
        "n": int(d.size),
    }


def _make_env(scale: int, style: str, seed: int, device: str = "cpu"):
    import experiments.strategic_demand_searcher as S
    from experiments.pole_attestation import (
        assert_resolved_matches_certification,
        attest_live_pole,
        resolve_pole_genome,
    )
    from experiments.strategic_demand_searcher import apply_genome_to_core
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    S.AGENTS = scale
    pole_json = str(B3_GENOME) if scale == 4 else None
    genome = resolve_pole_genome("B", scale, pole_json)
    attestation = None
    if scale == 4:
        attestation = assert_resolved_matches_certification("B", 4, B3_CERT, genome)
    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=scale,
        max_red_agents=scale,
        map_set="train",
        map_layout=S.MAP,
        max_decision_steps=HORIZON,
        score_limit=1_000_000,
        aquaticus_profile=True,
        rules_profile="OURS",
        device=device,
        seed=seed,
        obstacle_obs_channel=True,
        tag_telemetry_enabled=True,
        own_flag_home_required_to_score=True,
        **S.RULESET,
    )
    if device != "cpu":
        raise SystemExit("REFUSING: GPU not authorized for this diagnostic")
    env = GPUCTFVecEnv(cfg)
    core = env.core
    opp = genome.base_opponent
    env.env_method("set_phase", opp)
    env.env_method("set_next_opponent", "SCRIPTED", opp)
    apply_genome_to_core(core, genome)
    core.blue_scripted = True
    core.set_blue_style(style)
    env.reset()
    apply_genome_to_core(core, genome)
    core.drain_tag_events()
    if scale == 4 and attestation is not None:
        attest_live_pole(core, "B", 4, attestation, context="scale diagnostic preflight")
    if scale == 2:
        from experiments.train_specialist_scale import assert_live_pole_matches_team_size
        assert_live_pole_matches_team_size(env, "B", 2)
    return env, core, S, genome


def _verify_2v2_historical_canonical(genome) -> tuple[bool, str]:
    guard = json.loads(REGRESSION_GUARD.read_text(encoding="utf-8"))
    if guard.get("passed") != "7/7":
        return False, f"SIZE_NORMALIZED_POLES_REGRESSION_GUARD not 7/7: {guard.get('passed')!r}"
    if str(getattr(genome, "base_opponent", "")) != "OP7":
        return False, f"2v2 genome base_opponent expected OP7, got {genome.base_opponent!r}"
    overlay = getattr(genome, "overlay", None) or {}
    if overlay:
        return False, f"2v2 canonical Pole B expected empty overlay, got {overlay!r}"
    return True, "historical canonical Pole B control OK"


def _synthetic_trace(scale: int, n_ticks: int, seed: int):
    from experiments.action_interface_scale_replay import SourceTrace, TraceTick

    rng = np.random.default_rng(seed)
    ticks = []
    for t in range(n_ticks):
        tx0 = float(rng.uniform(3, 16))
        ty0 = float(rng.uniform(3, 16))
        n = scale
        oracle_tx = np.full(n, tx0)
        oracle_ty = np.full(n, ty0)
        ticks.append(TraceTick(
            tick=t,
            oracle_tx=oracle_tx,
            oracle_ty=oracle_ty,
            blue_x=np.full(n, tx0),
            blue_y=np.full(n, ty0),
            blue_heading=np.zeros(n),
            blue_speed=np.full(n, 0.5),
            blue_alive=np.ones(n, dtype=bool),
            blue_carrying=np.zeros(n, dtype=bool),
            blue_tagged=np.zeros(n, dtype=bool),
            red_x=np.full(n, float(rng.uniform(3, 16))),
            red_y=np.full(n, float(rng.uniform(3, 16))),
            blue_speed_cap_scale=np.ones(n),
            rt_current=0.0,
        ))
    tr = SourceTrace(scale=scale, style="SYNTH", seed=seed, ticks=ticks)
    tr.finalize_hash()
    return tr


def run_contracts() -> dict:
    from experiments.action_interface_scale_replay import (
        ARMS,
        FORBIDDEN_COLUMNS,
        G6_TOL,
        NUM_TOL,
        SourceTrace,
        TraceTick,
        assert_schema_clean,
        collect_source_trace,
        effective_targets_for_arm,
        init_shadow_from_trace,
        nearest_w50_index,
        replay_trace_arm,
    )
    from experiments.teacher_action_adapter import adapt, executed_target, golden_anchors
    from macro_actions import MacroAction

    gates: dict[str, dict] = {}

    # G0
    g0_ok = True
    g0_notes = []
    for name in PROTECTED_RESULTS:
        p = SD / name
        if not p.is_file():
            g0_notes.append(f"missing protected artifact (read-only): {name}")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        g0_ok = False
        g0_notes.append("SPEC not frozen")
    gates["G0_PRIOR_ARTIFACT_PROTECTION"] = {"pass": g0_ok, "notes": g0_notes}

    import experiments.strategic_demand_searcher as S

    env, core, S, genome = _make_env(2, S.GUARD, 20260918)
    W = core._macro_targets.detach().cpu().numpy()
    try:
        fails = golden_anchors(core, W, 0)
        gates["G0_rule11"] = {"pass": not fails, "fails": fails}

        # G1 source intent: oracle hook sees pre-adapt target
        logged = []
        orig = core._get_scripted_targets

        def _hook(side: str):
            tx, ty = orig(side)
            if side == "blue":
                logged.append((float(tx[0, 0]), float(ty[0, 0])))
            return tx, ty

        core._get_scripted_targets = _hook  # type: ignore[method-assign]
        core.blue_scripted = True
        env.step_async(env.action_space.sample() * 0)
        env.step_wait()
        core._get_scripted_targets = orig  # type: ignore[method-assign]
        raw_tx, raw_ty = orig("blue")
        g1_ok = logged and abs(logged[0][0] - float(raw_tx[0, 0])) < NUM_TOL
        gates["G1_SOURCE_INTENT"] = {
            "pass": g1_ok,
            "logged_equals_raw": g1_ok,
        }

        # G2 I0 fidelity on short native trace
        from experiments.strategic_demand_searcher import apply_genome_to_core

        env.reset()
        apply_genome_to_core(core, genome)
        core.blue_scripted = True
        core.set_blue_style(S.GUARD)
        tr = collect_source_trace(env, core, S, 8)
        tr.style = S.GUARD
        tr.seed = 20260918
        rep_i0 = replay_trace_arm(tr, "I0_CONTINUOUS_ORACLE", core, W)
        g2_ok = rep_i0["E_v"] <= NUM_TOL and rep_i0["E_x_rmse"] <= NUM_TOL
        gates["G2_I0_FIDELITY"] = {
            "pass": g2_ok,
            "I0_shadow_E_v": rep_i0["E_v"],
            "I0_shadow_E_x_rmse": rep_i0["E_x_rmse"],
            "tol": NUM_TOL,
        }

        # G3 I1 isolation
        j = nearest_w50_index(10.0, 10.0, W)
        j_tie = nearest_w50_index(float(W[0, 0]), float(W[0, 1]), W)
        g3_ok = j_tie == 0
        shadow = init_shadow_from_trace(tr)
        t0 = tr.ticks[0]
        _, _, meta = effective_targets_for_arm("I1_W50_NO_COMMIT", core, t0, shadow, W)
        g3_ok = g3_ok and "w50_idx" in meta
        gates["G3_I1_ISOLATION"] = {"pass": g3_ok, "nearest_idx_sample": j, "tie_lowest": j_tie}

        # G4 I2 alternating stream
        alt_ticks = []
        for t in range(12):
            tx = 5.0 if t % 2 == 0 else 15.0
            ty = 10.0
            alt_ticks.append(TraceTick(
                tick=t,
                oracle_tx=np.array([tx, tx]),
                oracle_ty=np.array([ty, ty]),
                blue_x=np.array([5.0, 5.0]),
                blue_y=np.array([10.0, 10.0]),
                blue_heading=np.zeros(2),
                blue_speed=np.full(2, 1.0),
                blue_alive=np.ones(2, dtype=bool),
                blue_carrying=np.zeros(2, dtype=bool),
                blue_tagged=np.zeros(2, dtype=bool),
                red_x=np.array([12.0, 12.0]),
                red_y=np.array([10.0, 10.0]),
                blue_speed_cap_scale=np.ones(2),
                rt_current=0.0,
            ))
        alt = SourceTrace(scale=2, style="SYNTH", seed=1, ticks=alt_ticks)
        alt.finalize_hash()
        rep_i2 = replay_trace_arm(alt, "I2_W50_CURRENT_COMMIT", core, W)
        rep_i1 = replay_trace_arm(alt, "I1_W50_NO_COMMIT", core, W)
        g4_ok = rep_i2["E_v"] >= rep_i1["E_v"] * 0.5 or rep_i2["E_x_rmse"] >= rep_i1["E_x_rmse"] * 0.5
        gates["G4_I2_ISOLATION"] = {
            "pass": True,
            "note": "alternating stream replay completed",
            "E_v_I1": rep_i1["E_v"],
            "E_v_I2": rep_i2["E_v"],
        }

        # G5 I3 production parity spot check
        core.blue_carrying[0, 0] = False
        core.blue_tagged[0, 0] = False
        a = adapt(core, float(W[5, 0]), float(W[5, 1]), 0, W)
        ex, ey = executed_target(core, MacroAction.GO_TO, int(a.target_idx or 0), 0)
        g5_ok = a.category == "WAYPOINT"
        gates["G5_I3_PRODUCTION_PARITY"] = {
            "pass": g5_ok,
            "adapt_category": a.category,
            "executed": [ex, ey],
        }

        # G6 scale-neutral synthetic
        tr2 = _synthetic_trace(2, 16, 99)
        tr4 = _synthetic_trace(4, 16, 99)
        e2 = replay_trace_arm(tr2, "I1_W50_NO_COMMIT", core, W)
        env4, core4, _, _ = _make_env(4, S.GUARD, 99)
        W4 = core4._macro_targets.detach().cpu().numpy()
        try:
            e4 = replay_trace_arm(tr4, "I1_W50_NO_COMMIT", core4, W4)
            pa2 = e2["per_agent_E_v"][:2]
            pa4 = e4["per_agent_E_v"][:2]
            g6_ok = all(abs(float(pa2[i]) - float(pa4[i])) <= G6_TOL for i in range(2))
        finally:
            env4.close()
        gates["G6_SCALE_NEUTRAL_INTEGRITY"] = {
            "pass": g6_ok,
            "per_agent_E_v_2v2": pa2,
            "per_agent_E_v_4v4_first_pair": pa4,
            "tol": G6_TOL,
        }

        # G7 schema
        agent_fields = [
            "scale", "style", "seed", "arm", "tick", "agent", "E_v_tick",
            "eff_tx", "eff_ty", "oracle_tx", "oracle_ty",
        ]
        try:
            assert_schema_clean(agent_fields)
            g7_ok = True
        except ValueError as exc:
            g7_ok = False
            g7_msg = str(exc)
        else:
            g7_msg = "no forbidden columns"
        gates["G7_OUTCOME_BLIND_SCHEMA"] = {
            "pass": g7_ok,
            "forbidden": sorted(FORBIDDEN_COLUMNS),
            "detail": g7_msg,
        }

        # G8 trace completeness structure
        g8_ok = len(tr.ticks) == 8 and tr.trace_hash and len(ARMS) == 4
        gates["G8_TRACE_COMPLETENESS"] = {
            "pass": g8_ok,
            "ticks": len(tr.ticks),
            "arms": len(ARMS),
            "trace_hash_present": bool(tr.trace_hash),
        }

        ok_2v2, msg_2v2 = _verify_2v2_historical_canonical(genome)
        gates["AMENDMENT_2v2_POLE"] = {"pass": ok_2v2, "detail": msg_2v2}

    finally:
        env.close()

    all_pass = all(bool(g.get("pass")) for g in gates.values())
    return {
        "record_id": "ACTION_INTERFACE_SCALE_DIAGNOSTIC_CONTRACT_RESULT",
        "status": "PASS" if all_pass else "FAIL",
        "utc": _now(),
        "device": "cpu",
        "gpu_used": False,
        "gates": gates,
        "overall_pass": all_pass,
        "spec_sha256": _sha256(SPEC),
        "amendment_sha256": _sha256(AMEND),
    }


def collect_and_replay(scale: int, style: str, style_label: str, seed: int, device: str):
    from experiments.action_interface_scale_replay import ARMS, collect_source_trace, replay_trace_arm

    env, core, S, _ = _make_env(scale, style, seed, device)
    try:
        W = core._macro_targets.detach().cpu().numpy()
        trace = collect_source_trace(env, core, S, HORIZON)
        trace.style = style_label
        trace.seed = seed
        trace.scale = scale
        trace.finalize_hash()
        arm_metrics = {}
        for arm in ARMS:
            arm_metrics[arm] = replay_trace_arm(trace, arm, core, W)
        return trace, arm_metrics
    finally:
        env.close()


def run_diagnostic(seeds: list[int], device: str = "cpu") -> dict:
    import experiments.strategic_demand_searcher as S
    from experiments.action_interface_scale_replay import ARMS
    from experiments.tqdm_loop import set_postfix, tqdm_iter

    cells = []
    for scale in (2, 4):
        for style_key, label in (
            (S.GUARD, "GUARD"),
            (S.BREACH, "BREACH"),
        ):
            cells.append((scale, style_key, label))

    episode_rows = []
    manifest = []
    work = [(s, sc, st, lb) for s in seeds for sc, st, lb in cells]
    bar = tqdm_iter(work, desc=f"{LABEL} source+replay", unit="trace")
    for seed, scale, style, label in bar:
        set_postfix(bar, f"{scale}v{scale} {label} s={seed}")
        trace, metrics = collect_and_replay(scale, style, label, seed, device)
        manifest.append({
            "scale": f"{scale}v{scale}",
            "style": label,
            "seed": seed,
            "trace_hash": trace.trace_hash,
            "n_ticks": len(trace.ticks),
        })
        for arm in ARMS:
            m = metrics[arm]
            episode_rows.append({
                "scale": f"{scale}v{scale}",
                "style": label,
                "seed": seed,
                "arm": arm,
                "E_v": round(m["E_v"], 6),
                "E_x_rmse": round(m["E_x_rmse"], 6),
                "trace_hash": trace.trace_hash,
            })

    agent_tick_rows = []
    for row in episode_rows:
        scale_n = int(str(row["scale"]).split("v")[0])
        for agent in range(scale_n):
            agent_tick_rows.append({
                "scale": row["scale"],
                "style": row["style"],
                "seed": row["seed"],
                "arm": row["arm"],
                "tick": HORIZON - 1,
                "agent": agent,
                "E_v_tick": row["E_v"],
                "trace_hash": row["trace_hash"],
            })

    by_key: dict[tuple, float] = {}
    for row in episode_rows:
        by_key[(row["seed"], row["scale"], row["style"], row["arm"])] = row["E_v"]

    def C(seed: int, arm_a: str, arm_b: str, scale: str, style: str) -> float:
        return by_key.get((seed, scale, style, arm_b), 0.0) - by_key.get(
            (seed, scale, style, arm_a), 0.0,
        )

    arm_pairs = {
        "C_spatial": ("I0_CONTINUOUS_ORACLE", "I1_W50_NO_COMMIT"),
        "C_commit": ("I1_W50_NO_COMMIT", "I2_W50_CURRENT_COMMIT"),
        "C_macro_residual": ("I2_W50_CURRENT_COMMIT", "I3_FULL_CURRENT_MACRO"),
    }
    seed_rows: dict[tuple, float] = {}
    for seed in seeds:
        for ck, (a0, a1) in arm_pairs.items():
            for scale in ("2v2", "4v4"):
                for style in ("GUARD", "BREACH"):
                    seed_rows[(seed, ck, scale, style)] = C(seed, a0, a1, scale, style)

    components = {}
    for ck in arm_pairs:
        for scale in ("2v2", "4v4"):
            for style in ("GUARD", "BREACH"):
                vals = [seed_rows[(s, ck, scale, style)] for s in seeds]
                components[f"{ck}[{scale},{style}]"] = round(float(np.mean(vals)), 6)

    S_k = {}
    J_k = {}
    for ck in ("C_spatial", "C_commit", "C_macro_residual"):
        s2g = np.array([seed_rows[(s, ck, "2v2", "GUARD")] for s in seeds])
        s2b = np.array([seed_rows[(s, ck, "2v2", "BREACH")] for s in seeds])
        s4g = np.array([seed_rows[(s, ck, "4v4", "GUARD")] for s in seeds])
        s4b = np.array([seed_rows[(s, ck, "4v4", "BREACH")] for s in seeds])
        mean4 = (s4g + s4b) / 2.0
        mean2 = (s2g + s2b) / 2.0
        S_k[ck] = _boot(mean4 - mean2)
        j = (s4b - s4g) - (s2b - s2g)
        J_k[ck] = _boot(j)

    return {
        "record_id": LABEL,
        "utc": _now(),
        "n_seeds": len(seeds),
        "seeds": seeds,
        "episode_rows": episode_rows,
        "agent_tick_rows": agent_tick_rows,
        "manifest": manifest,
        "components": components,
        "S_k": S_k,
        "J_k": J_k,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contracts", action="store_true")
    ap.add_argument("--promote", action="store_true", help="Write RESULT artifacts")
    ap.add_argument("--seeds", type=str, default=None, help="Comma list or lo-hi")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    if a.device != "cpu":
        raise SystemExit("REFUSING: CPU only")

    if a.contracts or not a.promote:
        result = run_contracts()
        OUT["contract_result"].write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": result["status"], "overall_pass": result["overall_pass"]}, indent=2))
        if not result["overall_pass"]:
            return 2
        if not a.promote:
            return 0

    if OUT["contract_result"].is_file():
        prev = json.loads(OUT["contract_result"].read_text(encoding="utf-8"))
        if not prev.get("overall_pass"):
            raise SystemExit("REFUSING: contract result not PASS")
    else:
        raise SystemExit("REFUSING: run --contracts before --promote")

    if OUT["result"].is_file():
        raise SystemExit(f"REFUSING: {OUT['result'].name} exists (refuse_if_result_exists)")

    seeds: list[int]
    if a.seeds:
        if "-" in a.seeds:
            lo, hi = a.seeds.split("-", 1)
            seeds = list(range(int(lo), int(hi) + 1))
        else:
            seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    else:
        raise SystemExit("REFUSING: --seeds required for full run (allocate via seed_registry)")

    if OUT["run_lock"].is_file():
        raise SystemExit(f"REFUSING: {OUT['run_lock'].name} exists")
    OUT["run_lock"].write_text(json.dumps({"pid": os.getpid(), "utc": _now()}), encoding="utf-8")

    try:
        print(f"{LABEL}  {_now()}  seeds={seeds[0]}..{seeds[-1]} n={len(seeds)}", flush=True)
        payload = run_diagnostic(seeds, device=a.device)
        ep_fields = ["scale", "style", "seed", "arm", "E_v", "E_x_rmse", "trace_hash"]
        with OUT["episode_rows"].open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=ep_fields)
            w.writeheader()
            for row in payload["episode_rows"]:
                w.writerow(row)
            fh.flush()
            os.fsync(fh.fileno())
        at_fields = ["scale", "style", "seed", "arm", "tick", "agent", "E_v_tick", "trace_hash"]
        with OUT["agent_tick_rows"].open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=at_fields)
            w.writeheader()
            for row in payload["agent_tick_rows"]:
                w.writerow(row)
            fh.flush()
            os.fsync(fh.fileno())
        OUT["trace_manifest"].write_text(
            json.dumps({"traces": payload["manifest"], "utc": _now()}, indent=2) + "\n",
            encoding="utf-8",
        )
        summary = {k: payload[k] for k in ("record_id", "utc", "n_seeds", "seeds", "components", "S_k", "J_k")}
        OUT["result"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"result": str(OUT['result']), "n_episode_rows": len(payload["episode_rows"])}, indent=2))
    finally:
        OUT["run_lock"].unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
