r"""ARM 1 sidecar, run STANDALONE: waypoint quantization error e_q.

Implements ARM1_TEACHER_IDENTIFIABILITY_SPEC.json#SIDECAR_WAYPOINT_EXPRESSIVITY_GAP.

    e_q(t) = min over w in W50 of || t - w ||_2

How much of the teacher's command is lost SOLELY because the student can only
pick one of 50 fixed map waypoints? This separates two failures that need
different repairs:

    "the student cannot know which target the teacher wants"   (observation)
    "the student knows roughly what the teacher wants but cannot express it"  (action)

TEST_0 already established the first at N>=4. This measures the second.

Run standalone because its result can change the very next experiment: if e_q is
large for GUARD at 4v4/6v6, then BC must not be designed as though observation
is the only difference between arms.

    python -m experiments.eval_arm1_eq_waypoint_audit --promote
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "ARM1_TEACHER_IDENTIFIABILITY_SPEC.json"
LABEL = "ARM1_EQ_WAYPOINT_AUDIT"


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _eq(tx: np.ndarray, ty: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Vectorized min distance from each target to the waypoint vocabulary."""
    d2 = ((W[None, :, 0] - tx[:, None]) ** 2 + (W[None, :, 1] - ty[:, None]) ** 2)
    return np.sqrt(d2.min(axis=1))


def _waypoint_geometry(W: np.ndarray) -> dict:
    """Reference scale: how far apart ARE the waypoints? Without this, an e_q
    number has no meaning -- it must be read relative to waypoint spacing."""
    d2 = ((W[:, None, 0] - W[None, :, 0]) ** 2 + (W[:, None, 1] - W[None, :, 1]) ** 2)
    np.fill_diagonal(d2, np.inf)
    nn = np.sqrt(d2.min(axis=1))
    return {
        "n_waypoints": int(W.shape[0]),
        "nearest_neighbour_spacing": {
            "median": round(float(np.median(nn)), 4),
            "mean": round(float(nn.mean()), 4),
            "max": round(float(nn.max()), 4),
        },
        "extent": {"x": [round(float(W[:, 0].min()), 2), round(float(W[:, 0].max()), 2)],
                   "y": [round(float(W[:, 1].min()), 2), round(float(W[:, 1].max()), 2)]},
        "expected_e_q_for_a_uniformly_random_point": round(float(np.median(nn)) / 2.0, 4),
        "how_to_read": "a point dropped at random between waypoints sits roughly half "
                       "a nearest-neighbour spacing from the closest one. e_q well "
                       "BELOW that means the teacher happens to command points the "
                       "vocabulary covers; e_q AT or ABOVE it means the vocabulary is "
                       "no better than arbitrary for this teacher.",
    }


def _summarize(e: np.ndarray) -> dict:
    if e.size == 0:
        return {"n": 0}
    return {
        "n": int(e.size),
        "median": round(float(np.median(e)), 4),
        "p90": round(float(np.percentile(e, 90)), 4),
        "p95": round(float(np.percentile(e, 95)), 4),
        "max": round(float(e.max()), 4),
        "mean": round(float(e.mean()), 4),
        "frac_gt_0.5_cells": round(float((e > 0.5).mean()), 4),
        "frac_gt_1_cell": round(float((e > 1.0).mean()), 4),
        "frac_gt_2_cells": round(float((e > 2.0).mean()), 4),
    }


def collect(scale: int, strategy: str, seeds: list[int], device: str,
            max_ticks: int) -> dict:
    import experiments.strategic_demand_searcher as S
    from experiments.opponent_spec import pole_A_genome
    from experiments.strategic_demand_searcher import apply_genome_to_core
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    from experiments.tqdm_loop import set_postfix, tqdm_iter

    S.AGENTS = scale
    genome = pole_A_genome(scale)
    style = S.GUARD if strategy == "GUARD" else S.BREACH
    n_def = (scale + 1) // 2
    def_lo = scale - n_def

    eq_all, eq_def, eq_atk = [], [], []
    W_ref = None
    bar = tqdm_iter(seeds, desc=f"e_q {scale}v{scale} {strategy}", unit="seed")
    for seed in bar:
        set_postfix(bar, f"samples={sum(len(a) for a in eq_all)}")
        cfg = GPUFieldConfig(
            n_envs=1, max_blue_agents=scale, max_red_agents=scale,
            map_set="train", map_layout=S.MAP, max_decision_steps=S.MAX_STEPS,
            aquaticus_profile=True, rules_profile="OURS", device=device, seed=seed,
            obstacle_obs_channel=True, tag_telemetry_enabled=True,
            own_flag_home_required_to_score=True, **S.RULESET,
        )
        env = GPUCTFVecEnv(cfg)
        core = env.core
        try:
            opp = genome.base_opponent
            env.env_method("set_phase", opp)
            env.env_method("set_next_opponent", "SCRIPTED", opp)
            apply_genome_to_core(core, genome)
            core.blue_scripted = True
            core.set_blue_style(style)
            env.reset()
            apply_genome_to_core(core, genome)
            core.drain_tag_events()
            W = core._macro_targets.detach().cpu().numpy()
            if W_ref is None:
                W_ref = W

            for _tick in range(min(max_ticks, S.MAX_STEPS)):
                btx, bty = core._get_scripted_targets("blue")
                tx = btx[0].detach().cpu().numpy().astype(np.float64)
                ty = bty[0].detach().cpu().numpy().astype(np.float64)
                e = _eq(tx, ty, W)
                eq_all.append(e)
                eq_def.append(e[def_lo:])
                eq_atk.append(e[:def_lo])
                env.step_async(env.action_space.sample() * 0)
                _o, _r, done, _i = env.step_wait()
                if bool(np.asarray(done).any()):
                    break
        finally:
            env.close()

    out = {"overall": _summarize(np.concatenate(eq_all) if eq_all else np.array([]))}
    if strategy == "GUARD":
        out["defenders"] = _summarize(np.concatenate(eq_def) if eq_def else np.array([]))
        out["attackers"] = _summarize(np.concatenate(eq_atk) if eq_atk else np.array([]))
    out["n_defenders"] = n_def
    out["waypoint_geometry"] = _waypoint_geometry(W_ref) if W_ref is not None else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scales", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--max-ticks", type=int, default=240)
    ap.add_argument("--promote", action="store_true")
    a = ap.parse_args()

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: spec not frozen: {spec.get('status')!r}")
    lo, _hi = spec["SEEDS"]["block"]
    seeds = list(range(lo, lo + a.n_seeds))

    print(f"{LABEL}  {_now()}")
    print(f"  implements  {SPEC.name}#SIDECAR_WAYPOINT_EXPRESSIVITY_GAP")
    print(f"  question    e_q = min_w ||t_teacher - w||  over the 50 fixed waypoints")
    print(f"  scales      {a.scales} x GUARD/BREACH   seeds {seeds[0]}..{seeds[-1]}")
    print(f"  CPU-only, trains nothing\n", flush=True)

    t0 = time.time()
    results = {}
    for n in a.scales:
        for strat in ("GUARD", "BREACH"):
            r = collect(n, strat, seeds, a.device, a.max_ticks)
            results[f"{n}v{n}_{strat}"] = r
            o = r["overall"]
            line = (f"  {n}v{n} {strat:<7s} n={o['n']:>6}  median={o['median']:.3f}  "
                    f"p90={o['p90']:.3f}  p95={o['p95']:.3f}  max={o['max']:.3f}  "
                    f">1cell={o['frac_gt_1_cell']:.3f}")
            if "defenders" in r:
                line += (f"\n      defenders: median={r['defenders']['median']:.3f} "
                        f"p90={r['defenders']['p90']:.3f} "
                        f">1cell={r['defenders']['frac_gt_1_cell']:.3f}"
                        f"   attackers: median={r['attackers']['median']:.3f} "
                        f"p90={r['attackers']['p90']:.3f} "
                        f">1cell={r['attackers']['frac_gt_1_cell']:.3f}")
            print(line, flush=True)

    cfg_sig = json.dumps({"scales": a.scales, "n_seeds": a.n_seeds,
                          "max_ticks": a.max_ticks, "device": a.device,
                          "seed_lo": seeds[0]}, sort_keys=True)
    run_id = f"{_now().replace(':', '').replace('-', '')}_{hashlib.sha256(cfg_sig.encode()).hexdigest()[:8]}"
    rec = {
        "record": f"{LABEL} waypoint quantization-error audit",
        "status": "FROZEN_RESULT", "utc": _now(), "run_id": run_id,
        "arm": "DIAGNOSTIC", "confirmatory": False,
        "implements": f"{SPEC.name}#SIDECAR_WAYPOINT_EXPRESSIVITY_GAP",
        "config_signature": json.loads(cfg_sig),
        "elapsed_s": round(time.time() - t0, 1),
        "units": "grid cells (arena is 20x20 cells; CNN grid is 20x20, so 1 unit = 1 cell)",
        "results": results,
        "DECISION_RULE_precommitted_by_PI": {
            "small_e_q_everywhere": "waypoint quantization is probably not the main bottleneck -> proceed to macro-adapter audit and privileged-vs-standard BC",
            "large_e_q_for_GUARD_at_4v4_6v6": "the ACTION interface is likely part of the scaling defect -> do NOT design BC as though observation is the only difference",
            "large_e_q_even_at_2v2": "PAUSE -- the action projection may not faithfully represent even the WORKING strategy",
            "BREACH_small_GUARD_large": "particularly compelling: it matches the exact strategic component that is causing trouble",
        },
    }
    p = SD / f"{LABEL}_{run_id}_RESULT.json"
    if p.exists():
        raise SystemExit(f"REFUSING: {p.name} exists; immutable artifacts are never overwritten.")
    p.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    canon = SD / f"{LABEL}_CANONICAL.json"
    if a.promote:
        canon.write_text(json.dumps({
            "record": f"CANONICAL {LABEL} run, explicitly promoted",
            "points_to": p.name, "run_id": run_id, "promoted_utc": _now(),
            "config_signature": json.loads(cfg_sig),
        }, indent=2), encoding="utf-8")
        print(f"\n  -> {p}\n  -> {canon}  (PROMOTED)")
    else:
        print(f"\n  -> {p}\n  (not promoted; use --promote)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
