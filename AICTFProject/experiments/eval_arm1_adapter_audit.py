r"""Exhaustive teacher->student action-adapter audit + conditional e_q.

Supersedes ARM1_EQ_WAYPOINT_AUDIT (marked INVALID_METRIC_MISSPECIFIED: it charged
the student a waypoint cost on targets reached exactly by semantic macros).

Every teacher decision lands in EXACTLY ONE category, determined by calling the
real engine target builder -- never by heuristic. Conditional e_q is reported
ONLY for decisions whose executed path actually uses the 50-waypoint vocabulary.

Gated by RULE 11 anchors: if experiments/teacher_action_adapter.golden_anchors
reports any failure, the audit refuses to run.

    python -m experiments.eval_arm1_adapter_audit --promote
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "ARM1_TEACHER_IDENTIFIABILITY_SPEC.json"
LABEL = "ARM1_ADAPTER_AUDIT"


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _summ(e: np.ndarray) -> dict:
    if e.size == 0:
        return {"n": 0}
    return {"n": int(e.size), "median": round(float(np.median(e)), 4),
            "p90": round(float(np.percentile(e, 90)), 4),
            "p95": round(float(np.percentile(e, 95)), 4),
            "max": round(float(e.max()), 4),
            "frac_gt_0.5": round(float((e > 0.5).mean()), 4),
            "frac_gt_1": round(float((e > 1.0).mean()), 4),
            "frac_gt_2": round(float((e > 2.0).mean()), 4)}


def run_cell(scale: int, strategy: str, seeds: list[int], device: str) -> dict:
    import experiments.strategic_demand_searcher as S
    from experiments.opponent_spec import pole_A_genome
    from experiments.strategic_demand_searcher import apply_genome_to_core
    from experiments.teacher_action_adapter import CATEGORIES, adapt, golden_anchors
    from experiments.tqdm_loop import set_postfix, tqdm_iter
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    S.AGENTS = scale
    g = pole_A_genome(scale)
    style = S.GUARD if strategy == "GUARD" else S.BREACH
    n_def = (scale + 1) // 2
    def_lo = scale - n_def

    cats = Counter()
    cats_def, cats_atk = Counter(), Counter()
    eq_wp, eq_wp_def, eq_wp_atk = [], [], []
    anchors_checked = False
    bar = tqdm_iter(seeds, desc=f"adapter {scale}v{scale} {strategy}", unit="seed")
    for seed in bar:
        set_postfix(bar, f"n={sum(cats.values())}")
        cfg = GPUFieldConfig(n_envs=1, max_blue_agents=scale, max_red_agents=scale,
            map_set="train", map_layout=S.MAP, max_decision_steps=S.MAX_STEPS,
            aquaticus_profile=True, rules_profile="OURS", device=device, seed=seed,
            obstacle_obs_channel=True, tag_telemetry_enabled=True,
            own_flag_home_required_to_score=True, **S.RULESET)
        env = GPUCTFVecEnv(cfg); core = env.core
        try:
            opp = g.base_opponent
            env.env_method("set_phase", opp)
            env.env_method("set_next_opponent", "SCRIPTED", opp)
            apply_genome_to_core(core, g)
            core.blue_scripted = True
            core.set_blue_style(style)
            env.reset(); apply_genome_to_core(core, g); core.drain_tag_events()
            W = core._macro_targets.detach().cpu().numpy()

            if not anchors_checked:
                fails = golden_anchors(core, W, agent=0)
                if fails:
                    raise SystemExit("RULE 11 ANCHORS FAILED -- audit refuses to run:\n  "
                                     + "\n  ".join(fails))
                anchors_checked = True

            for _t in range(S.MAX_STEPS):
                btx, bty = core._get_scripted_targets("blue")
                for i in range(scale):
                    a = adapt(core, float(btx[0, i]), float(bty[0, i]), i, W)
                    cats[a.category] += 1
                    (cats_def if i >= def_lo else cats_atk)[a.category] += 1
                    if a.uses_waypoint:
                        eq_wp.append(a.residual)
                        (eq_wp_def if i >= def_lo else eq_wp_atk).append(a.residual)
                env.step_async(env.action_space.sample() * 0)
                _o, _r, d, _i = env.step_wait()
                if bool(np.asarray(d).any()):
                    break
        finally:
            env.close()

    tot = sum(cats.values())
    unmapped = cats.get("UNMAPPED", 0)
    return {
        "n_decisions": tot,
        "categories": {c: cats.get(c, 0) for c in CATEGORIES},
        "category_fractions": {c: round(cats.get(c, 0) / max(1, tot), 4) for c in CATEGORIES},
        "UNMAPPED_count": unmapped,
        "exhaustive": sum(cats.values()) == tot and unmapped == 0,
        "conditional_e_q_WAYPOINT_ONLY": _summ(np.array(eq_wp)),
        "defenders": {"category_fractions": {c: round(cats_def.get(c, 0) / max(1, sum(cats_def.values())), 4)
                                             for c in CATEGORIES},
                      "conditional_e_q": _summ(np.array(eq_wp_def))},
        "attackers": {"category_fractions": {c: round(cats_atk.get(c, 0) / max(1, sum(cats_atk.values())), 4)
                                             for c in CATEGORIES},
                      "conditional_e_q": _summ(np.array(eq_wp_atk))},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scales", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--promote", action="store_true")
    a = ap.parse_args()

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: spec not frozen: {spec.get('status')!r}")
    lo, _ = spec["SEEDS"]["block"]
    seeds = list(range(lo, lo + a.n_seeds))

    print(f"{LABEL}  {_now()}")
    print(f"  every teacher decision -> exactly one category, by CALLING the real")
    print(f"  _build_targets_from_action (rule 11: executed action beats intended)")
    print(f"  conditional e_q reported ONLY for the WAYPOINT pathway\n", flush=True)

    t0 = time.time()
    results = {}
    any_unmapped = 0
    for n in a.scales:
        for strat in ("GUARD", "BREACH"):
            r = run_cell(n, strat, seeds, a.device)
            results[f"{n}v{n}_{strat}"] = r
            any_unmapped += r["UNMAPPED_count"]
            cf = r["category_fractions"]
            eq = r["conditional_e_q_WAYPOINT_ONLY"]
            eqs = (f"median={eq['median']:.3f} p90={eq['p90']:.3f} >1={eq['frac_gt_1']:.3f}"
                   if eq["n"] else "(no waypoint decisions)")
            print(f"  {n}v{n} {strat:<7s} n={r['n_decisions']:>5}  "
                  f"FLAG={cf['GET_FLAG']:.3f} HOME={cf['GO_HOME']:.3f} "
                  f"WP={cf['WAYPOINT']:.3f} carry={cf['FORCED_HOME_CARRYING']:.3f} "
                  f"tag={cf['FORCED_HOME_TAGGED']:.3f} UNMAPPED={r['UNMAPPED_count']}")
            print(f"      e_q|WAYPOINT: {eqs}")
            if strat == "GUARD":
                d = r["defenders"]; k = r["attackers"]
                de, ke = d["conditional_e_q"], k["conditional_e_q"]
                print(f"      defenders WP={d['category_fractions']['WAYPOINT']:.3f} "
                      f"e_q med={de.get('median', float('nan')):.3f}   "
                      f"attackers WP={k['category_fractions']['WAYPOINT']:.3f} "
                      f"e_q med={ke.get('median', float('nan')):.3f}")
            print(flush=True)

    cfg_sig = json.dumps({"scales": a.scales, "n_seeds": a.n_seeds,
                          "device": a.device, "seed_lo": seeds[0]}, sort_keys=True)
    run_id = f"{_now().replace(':', '').replace('-', '')}_{hashlib.sha256(cfg_sig.encode()).hexdigest()[:8]}"
    rec = {
        "record": f"{LABEL} exhaustive action-adapter audit + conditional e_q",
        "status": "FROZEN_RESULT" if any_unmapped == 0 else "AUDIT_FAILED_UNMAPPED",
        "utc": _now(), "run_id": run_id, "arm": "DIAGNOSTIC", "confirmatory": False,
        "supersedes": "ARM1_EQ_WAYPOINT_AUDIT (INVALID_METRIC_MISSPECIFIED)",
        "rule_11_anchors": "PASSED before any cell ran; audit refuses to run otherwise",
        "config_signature": json.loads(cfg_sig),
        "elapsed_s": round(time.time() - t0, 1),
        "units": "grid cells",
        "total_UNMAPPED": any_unmapped,
        "results": results,
        "HOW_TO_READ": {
            "category_fractions": "exhaustive per cell; sum to 1.0 by construction",
            "conditional_e_q": "defined ONLY on WAYPOINT decisions. Semantic-macro and "
                               "forced-home decisions execute exactly, so e_q is not "
                               "merely small for them -- it is undefined, and including "
                               "them is what invalidated the previous audit.",
            "UNMAPPED_expectation": "0 is EXPECTED, not a strong validation: any in-map "
                                    "target is reachable through the waypoint pathway "
                                    "with bounded residual. UNMAPPED exists to catch "
                                    "out-of-map or pathological commands.",
        },
    }
    p = SD / f"{LABEL}_{run_id}_RESULT.json"
    if p.exists():
        raise SystemExit(f"REFUSING: {p.name} exists; immutable artifacts are never overwritten.")
    p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    if a.promote:
        (SD / f"{LABEL}_CANONICAL.json").write_text(json.dumps({
            "record": f"CANONICAL {LABEL}", "points_to": p.name, "run_id": run_id,
            "promoted_utc": _now(), "config_signature": json.loads(cfg_sig)}, indent=2),
            encoding="utf-8")
        print(f"  -> {p}\n  -> {LABEL}_CANONICAL.json  (PROMOTED)")
    else:
        print(f"  -> {p}  (not promoted)")
    if any_unmapped:
        raise SystemExit(f"AUDIT FAILED: {any_unmapped} UNMAPPED decisions require explanation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
