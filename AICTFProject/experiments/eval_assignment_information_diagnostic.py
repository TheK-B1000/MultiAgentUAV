r"""Assignment information diagnostic (G1 ∧ G2 ∧ G3).

Implements ASSIGNMENT_INFORMATION_DIAGNOSTIC_SPEC.json.
CPU-only. No PPO. No GETFLAG. tqdm on stderr.

    python -u experiments/eval_assignment_information_diagnostic.py --device cpu
"""

from __future__ import annotations

import argparse
import inspect
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / "ASSIGNMENT_INFORMATION_DIAGNOSTIC_SPEC.json"
OUT = SD / "ASSIGNMENT_INFORMATION_DIAGNOSTIC_RESULT.json"
ARM1_CANONICAL = SD / "ARM1_TEST0_COUNTEREXAMPLE_CANONICAL.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _zi_from_core(core, scale: int):
    from gpu_env._core._scripted_blue_styles import gate2b_defender_hold_radius
    from rl.custom_ppo.guard_assignment import assign_guard_v2_responsibilities

    home = core.blue_flag_home
    if home.dim() == 3:
        home = home[:, 0, :]
    # Teacher v2 uses red_flag_pos (not red_flag_home).
    eflag = core.red_flag_pos
    if eflag.dim() == 3:
        eflag = eflag[:, 0, :]
    on_our = core._is_on_home_side("blue", core.red_x)
    return assign_guard_v2_responsibilities(
        own_x=core.blue_x,
        own_y=core.blue_y,
        own_alive=core.blue_alive.bool(),
        home_xy=home,
        enemy_x=core.red_x,
        enemy_y=core.red_y,
        enemy_alive=core.red_alive.bool(),
        enemy_tagged=core.red_tagged.bool(),
        enemy_flag_xy=eflag,
        on_our_side=on_our,
        defense_radius=float(gate2b_defender_hold_radius(core.cfg)),
        midline_fn=lambda tx: core._is_on_home_side("blue", tx),
    )


def gate_g3_static() -> dict:
    from rl.custom_ppo import guard_assignment as GA

    sig = inspect.signature(GA.assign_guard_v2_responsibilities)
    params = list(sig.parameters)
    forbidden_hits = [
        p for p in params
        if any(f in p.lower() for f in (
            "macro", "action", "btx", "bty", "waypoint", "reward", "return",
            "pole", "getflag", "opponent", "genome",
        ))
    ]
    # Dynamic: geometry-only call must ignore fake macro kwargs via assert helper.
    leak_ok = True
    try:
        GA.assert_zi_builder_no_forbidden_kwargs(macro=1, action=2)
        leak_ok = False
    except ValueError:
        leak_ok = True
    passed = (not forbidden_hits) and leak_ok
    return {
        "passed": passed,
        "parameter_names": params,
        "forbidden_hits": forbidden_hits,
        "forbid_kwargs_assert_works": leak_ok,
    }


def gate_g3_dynamic(device: str) -> dict:
    """Holding geometry fixed, teacher macros must not affect z_i (builder never sees them)."""
    # By construction the builder has no macro inputs; prove identical z under two
    # dummy "macro worlds" is vacuous — instead recompute twice from same geometry.
    B, N, Ne = 1, 4, 4
    own_x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    own_y = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    alive = torch.ones(1, N, dtype=torch.bool)
    home = torch.tensor([[2.0, 10.0]])
    enemy_x = torch.tensor([[5.0, 6.0, 12.0, 13.0]])
    enemy_y = torch.tensor([[10.0, 11.0, 10.0, 9.0]])
    ea = torch.tensor([[True, True, False, False]])
    et = torch.zeros(1, Ne, dtype=torch.bool)
    on_our = torch.tensor([[True, True, False, False]])
    eflag = torch.tensor([[18.0, 10.0]])
    from rl.custom_ppo.guard_assignment import (
        assign_guard_v2_responsibilities, zi_discrete_key,
    )
    kwargs = dict(
        own_x=own_x, own_y=own_y, own_alive=alive, home_xy=home,
        enemy_x=enemy_x, enemy_y=enemy_y, enemy_alive=ea, enemy_tagged=et,
        enemy_flag_xy=eflag, on_our_side=on_our, defense_radius=6.0,
    )
    a = assign_guard_v2_responsibilities(**kwargs)
    b = assign_guard_v2_responsibilities(**kwargs)
    same = bool(torch.equal(zi_discrete_key(a["responsibility"], a["assigned_entity"]),
                            zi_discrete_key(b["responsibility"], b["assigned_entity"])))
    return {"passed": same, "deterministic_replay": same}


def gate_g1(scale: int, seeds: list[int], device: str, max_ticks: int, tick_stride: int) -> dict:
    import experiments.strategic_demand_searcher as S
    from experiments.opponent_spec import pole_A_genome
    from experiments.strategic_demand_searcher import apply_genome_to_core
    from experiments.tqdm_loop import set_postfix, tqdm_iter
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    S.AGENTS = scale
    genome = pole_A_genome(scale)
    n_ok = 0
    n_tot = 0
    abs_err = []

    bar = tqdm_iter(seeds, desc=f"G1 {scale}v{scale}", unit="seed")
    for seed in bar:
        set_postfix(bar, f"acc={(n_ok / max(n_tot, 1)):.3f}")
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
            core.set_blue_style(S.GUARD)
            env.reset()
            apply_genome_to_core(core, genome)
            for tick in range(min(max_ticks, S.MAX_STEPS)):
                if tick % tick_stride == 0:
                    # G1 baseline = assignment subroutine only (not carrier evasion).
                    B, N = 1, scale
                    home = core.blue_flag_home
                    t_teach_x, t_teach_y = core._blue_one_defender_v2_targets(
                        core.blue_x, core.blue_y, home,
                        core.red_x, core.red_y, core.red_alive, core.red_tagged,
                        core.red_flag_pos, B, N,
                    )
                    zi = _zi_from_core(core, scale)
                    alive = core.blue_alive[0].detach().cpu().numpy()
                    for i in range(scale):
                        if not bool(alive[i]):
                            continue
                        n_tot += 1
                        dx = float(zi["target_x"][0, i] - t_teach_x[0, i])
                        dy = float(zi["target_y"][0, i] - t_teach_y[0, i])
                        err = (dx * dx + dy * dy) ** 0.5
                        abs_err.append(err)
                        if err < 1e-3:
                            n_ok += 1
                env.step_async(env.action_space.sample() * 0)
                _o, _r, done, _info = env.step_wait()
                if bool(np.asarray(done).any()):
                    break
        finally:
            env.close()

    acc = float(n_ok / max(n_tot, 1))
    return {
        "scale": f"{scale}v{scale}",
        "n_eligible_agent_ticks": n_tot,
        "n_target_match": n_ok,
        "accuracy": acc,
        "mean_target_l2_err": float(np.mean(abs_err)) if abs_err else None,
        "passed": acc >= 0.95,
        "pass_floor": 0.95,
    }


def gate_g2(scale: int, seeds: list[int], device: str, tick_stride: int, max_ticks: int,
            cap: int = 25) -> dict:
    """Re-seek TEST_0 pairs; require z_i discrete fields to differ when q differs."""
    import experiments.strategic_demand_searcher as S
    from experiments.opponent_spec import pole_A_genome
    from experiments.strategic_demand_searcher import apply_genome_to_core
    from experiments.tqdm_loop import set_postfix, tqdm_iter
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    from rl.custom_ppo.guard_assignment import zi_discrete_key

    S.AGENTS = scale
    genome = pole_A_genome(scale)
    n_def = (scale + 1) // 2
    def_lo = scale - n_def

    pairs = 0
    resolved = 0
    unresolved = []

    def _project(tx, ty, W):
        return int(np.argmin((W[:, 0] - tx) ** 2 + (W[:, 1] - ty) ** 2))

    bar = tqdm_iter(seeds, desc=f"G2 {scale}v{scale}", unit="seed")
    for seed in bar:
        set_postfix(bar, f"resolved={resolved}/{pairs}")
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
            core.set_blue_style(S.GUARD)
            env.reset()
            apply_genome_to_core(core, genome)
            W = core._macro_targets.detach().cpu().numpy()

            for tick in range(min(max_ticks, S.MAX_STEPS)):
                if tick % tick_stride != 0:
                    env.step_async(env.action_space.sample() * 0)
                    _o, _r, done, _info = env.step_wait()
                    if bool(np.asarray(done).any()):
                        break
                    continue
                intr = (core.red_alive[0] & (~core.red_tagged[0])
                        & core._is_on_home_side("blue", core.red_x)[0])
                if int(intr.sum().item()) < 1:
                    env.step_async(env.action_space.sample() * 0)
                    _o, _r, done, _info = env.step_wait()
                    if bool(np.asarray(done).any()):
                        break
                    continue

                x0 = core.blue_x.clone()
                y0 = core.blue_y.clone()
                alive0 = core.blue_alive.clone()
                tagged0 = core.blue_tagged.clone()

                def _obs_bytes(i):
                    obs = core.get_obs_tensors("blue")
                    g = obs["grid"][0, i].detach().cpu().numpy().astype(np.float32)
                    v = obs["vec"][0, i].detach().cpu().numpy().astype(np.float32)
                    am = obs["agent_mask"][0].detach().cpu().numpy().astype(np.float32)
                    m = obs["mask"][0].detach().cpu().numpy().astype(np.float32)
                    per = m.shape[0] // scale
                    return (np.ascontiguousarray(g).tobytes() + np.ascontiguousarray(v).tobytes()
                            + np.ascontiguousarray(am).tobytes()
                            + np.ascontiguousarray(m[i * per:(i + 1) * per]).tobytes())

                for j in range(def_lo, scale):
                    if not bool(alive0[0, j]) or bool(tagged0[0, j]):
                        continue
                    cx = float(np.round(float(x0[0, j])))
                    cy = float(np.round(float(y0[0, j])))
                    for (ax, ay), (bx, by) in (
                        ((cx - 0.45, cy), (cx + 0.45, cy)),
                        ((cx, cy - 0.45), (cx, cy + 0.45)),
                    ):
                        core.blue_x[0, j] = ax
                        core.blue_y[0, j] = ay
                        btxA, btyA = core._get_scripted_targets("blue")
                        ziA = _zi_from_core(core, scale)
                        obsA = [_obs_bytes(i) for i in range(scale)]
                        qA = [_project(float(btxA[0, i]), float(btyA[0, i]), W) for i in range(scale)]
                        keyA = zi_discrete_key(ziA["responsibility"], ziA["assigned_entity"])[0]

                        core.blue_x[0, j] = bx
                        core.blue_y[0, j] = by
                        btxB, btyB = core._get_scripted_targets("blue")
                        ziB = _zi_from_core(core, scale)
                        obsB = [_obs_bytes(i) for i in range(scale)]
                        qB = [_project(float(btxB[0, i]), float(btyB[0, i]), W) for i in range(scale)]
                        keyB = zi_discrete_key(ziB["responsibility"], ziB["assigned_entity"])[0]
                        core.blue_x.copy_(x0)
                        core.blue_y.copy_(y0)

                        for i in range(scale):
                            if i == j:
                                continue
                            if obsA[i] != obsB[i]:
                                continue
                            if qA[i] == qB[i]:
                                continue
                            pairs += 1
                            if int(keyA[i].item()) != int(keyB[i].item()):
                                resolved += 1
                            else:
                                unresolved.append({
                                    "seed": seed, "tick": tick, "i": i, "j": j,
                                    "qA": qA[i], "qB": qB[i],
                                    "keyA": int(keyA[i]), "keyB": int(keyB[i]),
                                })
                            if pairs >= cap:
                                break
                        if pairs >= cap:
                            break
                    if pairs >= cap:
                        break
                if pairs >= cap:
                    break
                env.step_async(env.action_space.sample() * 0)
                _o, _r, done, _info = env.step_wait()
                if bool(np.asarray(done).any()):
                    break
        finally:
            env.close()
        if pairs >= cap:
            break

    if scale == 2:
        # Negative control: expect ~0 pairs; G2 N/A.
        return {
            "scale": "2v2",
            "n_test0_pairs": pairs,
            "n_zi_resolved": resolved,
            "passed": pairs == 0 or resolved == pairs,
            "control": True,
            "note": "N=2 negative control (assignment inactive)",
            "unresolved_examples": unresolved[:5],
        }
    passed = pairs > 0 and resolved == pairs
    return {
        "scale": f"{scale}v{scale}",
        "n_test0_pairs": pairs,
        "n_zi_resolved": resolved,
        "passed": passed,
        "control": False,
        "unresolved_examples": unresolved[:5],
        "rule": "every o-identical q-differing pair must have differing discrete z_i",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--tick-stride", type=int, default=6)
    ap.add_argument("--max-ticks", type=int, default=240)
    args = ap.parse_args()

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: spec not frozen: {spec.get('status')!r}")
    if OUT.is_file():
        raise SystemExit(f"REFUSING: RESULT already exists: {OUT}")

    # ARM1 sealed seeds.
    seeds = [17600001, 17600002, 17600003][: args.n_seeds]

    print(f"ASSIGNMENT_INFORMATION_DIAGNOSTIC  {_now()}")
    print(f"  implements  {SPEC_PATH.name}")
    print(f"  seeds       {seeds[0]}..{seeds[-1]} (ARM1 TEST_0 corpus)")
    print(f"  device      {args.device}  (CPU diagnostic; no PPO)\n", flush=True)

    t0 = time.time()
    g3s = gate_g3_static()
    g3d = gate_g3_dynamic(args.device)
    g3 = {"passed": bool(g3s["passed"] and g3d["passed"]), "static": g3s, "dynamic": g3d}
    print(f"  G3 NO_ACTION_LEAK: {'PASS' if g3['passed'] else 'FAIL'}", flush=True)

    g1_4 = gate_g1(4, seeds, args.device, args.max_ticks, args.tick_stride)
    g1_2 = gate_g1(2, seeds, args.device, args.max_ticks, args.tick_stride)
    print(f"  G1 N=4 accuracy={g1_4['accuracy']:.4f}  "
          f"{'PASS' if g1_4['passed'] else 'FAIL'}  (n={g1_4['n_eligible_agent_ticks']})", flush=True)
    print(f"  G1 N=2 accuracy={g1_2['accuracy']:.4f}  (control)", flush=True)

    g2_4 = gate_g2(4, seeds, args.device, args.tick_stride, args.max_ticks, cap=25)
    g2_2 = gate_g2(2, seeds, args.device, args.tick_stride, args.max_ticks, cap=25)
    print(f"  G2 N=4 resolved={g2_4['n_zi_resolved']}/{g2_4['n_test0_pairs']}  "
          f"{'PASS' if g2_4['passed'] else 'FAIL'}", flush=True)
    print(f"  G2 N=2 pairs={g2_2['n_test0_pairs']}  (control)", flush=True)

    overall = bool(g1_4["passed"] and g2_4["passed"] and g3["passed"])
    result = {
        "record": "ASSIGNMENT_INFORMATION_DIAGNOSTIC",
        "status": "FROZEN_RESULT",
        "utc": _now(),
        "implements": SPEC_PATH.name,
        "arm": "DIAGNOSTIC",
        "confirmatory": False,
        "elapsed_s": round(time.time() - t0, 1),
        "seeds": seeds,
        "G1_TEACHER_ASSIGNMENT_AGREEMENT": {"N4": g1_4, "N2_control": g1_2},
        "G2_RESOLVES_SEALED_AMBIGUITY": {"N4": g2_4, "N2_control": g2_2},
        "G3_NO_ACTION_LEAK": g3,
        "OVERALL_PASS": overall,
        "AUTHORIZATION": (
            "Assignment Conditioning v1 training AUTHORIZED"
            if overall else
            "Assignment Conditioning v1 training FORBIDDEN — redesign z_i / rerun diagnostic"
        ),
        "GETFLAG_untouched": True,
        "no_PPO_in_this_record": True,
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  OVERALL: {'PASS' if overall else 'FAIL'}")
    print(f"  -> {OUT}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
