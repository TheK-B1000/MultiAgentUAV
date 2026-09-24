r"""Observation-augmentation ladder on TEST_0's counterexamples.

TEST_0 established constructively that at N>=4 two physically valid states can
share a BYTE-IDENTICAL student observation while requiring different teacher
actions. This asks the follow-up that selects the repair:

    what is the SMALLEST observation augmentation that kills the contradiction?

RUNGS (each adds to the previous)
    BASE                grid[i] + vec[i] + agent_mask + mask[i]   (current student)
    +TEAMMATES          exact labeled teammate (x, y), fixed index order
    +TEAMMATES+ENEMIES  exact labeled enemy (x, y), fixed index order

WHY THE ENEMY RUNG EXISTS (not in the original proposal)
    Reading `_blue_one_defender_v2_targets`: the assignment uses EXACT enemy
    coordinates twice -- `d_home` to rank threats by proximity to our flag, and
    `dd` for the defender-to-threat distance matrix. Enemies reach the student
    only through the same CELL-QUANTIZED channel-2 scatter as teammates. So
    exact teammate geometry alone should NOT close the gap: a counterexample
    should survive by perturbing an ENEMY within its cell.

    Also note `taken` is allocated fresh inside each call -- it is NOT persisted
    hidden state across ticks. The assignment is therefore a deterministic
    function of (exact defender positions, exact threat positions, home), which
    is why the geometric rungs are worth testing before resorting to an explicit
    role code.

PERTURBATIONS
    Two families, each staying inside the perturbed entity's own 20x20 grid cell
    (round-to-nearest, so +-0.45 is in-cell):
        DEFENDER  move a defender -- invisible at BASE, visible at +TEAMMATES
        ENEMY     move a live intruder -- invisible until +TEAMMATES+ENEMIES

    A perturbation only counts if the AUGMENTED fingerprint is byte-identical.

    python -m experiments.eval_observation_ladder --scales 4 2 --n-seeds 3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
LABEL = "OBSERVATION_LADDER"
RUNGS = ("BASE", "TEAMMATES", "TEAMMATES_ENEMIES")


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fingerprint(core, i: int, n: int, rung: str) -> bytes:
    """COMPLETE per-agent policy input at this rung, as raw bytes."""
    obs = core.get_obs_tensors("blue")
    g = obs["grid"][0, i].detach().cpu().numpy().astype(np.float32)
    v = obs["vec"][0, i].detach().cpu().numpy().astype(np.float32)
    am = obs["agent_mask"][0].detach().cpu().numpy().astype(np.float32)
    m = obs["mask"][0].detach().cpu().numpy().astype(np.float32)
    per = m.shape[0] // n
    parts = [np.ascontiguousarray(g).tobytes(), np.ascontiguousarray(v).tobytes(),
             np.ascontiguousarray(am).tobytes(),
             np.ascontiguousarray(m[i * per:(i + 1) * per]).tobytes()]
    if rung in ("TEAMMATES", "TEAMMATES_ENEMIES"):
        tx = core.blue_x[0].detach().cpu().numpy().astype(np.float64)
        ty = core.blue_y[0].detach().cpu().numpy().astype(np.float64)
        parts.append(np.ascontiguousarray(np.stack([tx, ty])).tobytes())
    if rung == "TEAMMATES_ENEMIES":
        ex = core.red_x[0].detach().cpu().numpy().astype(np.float64)
        ey = core.red_y[0].detach().cpu().numpy().astype(np.float64)
        parts.append(np.ascontiguousarray(np.stack([ex, ey])).tobytes())
    return b"".join(parts)


def _project(tx, ty, W):
    return int(np.argmin((W[:, 0] - tx) ** 2 + (W[:, 1] - ty) ** 2))


def _capture(core, n, W, rung):
    btx, bty = core._get_scripted_targets("blue")
    tx = btx[0].detach().cpu().numpy(); ty = bty[0].detach().cpu().numpy()
    return ([fingerprint(core, i, n, rung) for i in range(n)],
            [_project(float(tx[i]), float(ty[i]), W) for i in range(n)],
            [(float(tx[i]), float(ty[i])) for i in range(n)])


def probe(scale, seeds, device, tick_stride, max_ticks, cap=25):
    import experiments.strategic_demand_searcher as S
    from experiments.opponent_spec import pole_A_genome
    from experiments.strategic_demand_searcher import apply_genome_to_core
    from experiments.tqdm_loop import set_postfix, tqdm_iter
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    S.AGENTS = scale
    g = pole_A_genome(scale)
    n_def = (scale + 1) // 2
    def_lo = scale - n_def

    found = {(r, p): 0 for r in RUNGS for p in ("DEFENDER", "ENEMY")}
    ident = {(r, p): 0 for r in RUNGS for p in ("DEFENDER", "ENEMY")}
    tries = {p: 0 for p in ("DEFENDER", "ENEMY")}
    examples = defaultdict(list)

    bar = tqdm_iter(seeds, desc=f"ladder {scale}v{scale}", unit="seed")
    for seed in bar:
        set_postfix(bar, f"found={sum(found.values())}")
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
            core.set_blue_style(S.GUARD)
            env.reset(); apply_genome_to_core(core, g); core.drain_tag_events()
            W = core._macro_targets.detach().cpu().numpy()

            for tick in range(min(max_ticks, S.MAX_STEPS)):
                if tick % tick_stride == 0:
                    intr = (core.red_alive[0] & (~core.red_tagged[0])
                            & core._is_on_home_side("blue", core.red_x)[0])
                    if int(intr.sum().item()) >= 1:
                        bx0 = core.blue_x.clone(); by0 = core.blue_y.clone()
                        rx0 = core.red_x.clone(); ry0 = core.red_y.clone()
                        alive = core.blue_alive.clone(); tag = core.blue_tagged.clone()

                        # ---- family 1: perturb a DEFENDER within its cell ----
                        for j in range(def_lo, scale):
                            if not bool(alive[0, j]) or bool(tag[0, j]):
                                continue
                            cx = float(np.round(float(bx0[0, j])))
                            cy = float(np.round(float(by0[0, j])))
                            for (ax, ay), (bx, by) in (((cx - .45, cy), (cx + .45, cy)),
                                                       ((cx, cy - .45), (cx, cy + .45))):
                                tries["DEFENDER"] += 1
                                for rung in RUNGS:
                                    core.blue_x[0, j] = ax; core.blue_y[0, j] = ay
                                    fA, qA, tA = _capture(core, scale, W, rung)
                                    core.blue_x[0, j] = bx; core.blue_y[0, j] = by
                                    fB, qB, tB = _capture(core, scale, W, rung)
                                    core.blue_x.copy_(bx0); core.blue_y.copy_(by0)
                                    for i in range(scale):
                                        if i == j or fA[i] != fB[i]:
                                            continue
                                        ident[(rung, "DEFENDER")] += 1
                                        if qA[i] != qB[i]:
                                            found[(rung, "DEFENDER")] += 1
                                            if len(examples[(rung, "DEFENDER")]) < 3:
                                                examples[(rung, "DEFENDER")].append(
                                                    {"seed": seed, "tick": tick, "perturbed_j": j,
                                                     "subject_i": i, "q": [qA[i], qB[i]],
                                                     "t_i": [tA[i], tB[i]]})

                        # ---- family 2: perturb a live ENEMY within its cell ----
                        idx = np.where(intr.detach().cpu().numpy())[0]
                        for k in idx[:3]:
                            cx = float(np.round(float(rx0[0, k])))
                            cy = float(np.round(float(ry0[0, k])))
                            for (ax, ay), (bx, by) in (((cx - .45, cy), (cx + .45, cy)),
                                                       ((cx, cy - .45), (cx, cy + .45))):
                                tries["ENEMY"] += 1
                                for rung in RUNGS:
                                    core.red_x[0, k] = ax; core.red_y[0, k] = ay
                                    fA, qA, tA = _capture(core, scale, W, rung)
                                    core.red_x[0, k] = bx; core.red_y[0, k] = by
                                    fB, qB, tB = _capture(core, scale, W, rung)
                                    core.red_x.copy_(rx0); core.red_y.copy_(ry0)
                                    for i in range(scale):
                                        if fA[i] != fB[i]:
                                            continue
                                        ident[(rung, "ENEMY")] += 1
                                        if qA[i] != qB[i]:
                                            found[(rung, "ENEMY")] += 1
                                            if len(examples[(rung, "ENEMY")]) < 3:
                                                examples[(rung, "ENEMY")].append(
                                                    {"seed": seed, "tick": tick, "perturbed_enemy": int(k),
                                                     "subject_i": i, "i_is_defender": bool(i >= def_lo),
                                                     "q": [qA[i], qB[i]], "t_i": [tA[i], tB[i]]})
                env.step_async(env.action_space.sample() * 0)
                _o, _r, d, _i = env.step_wait()
                if bool(np.asarray(d).any()):
                    break
        finally:
            env.close()
        if sum(found.values()) >= cap * len(RUNGS):
            break

    return {"n_defenders": n_def, "perturbation_attempts": tries,
            "rungs": {r: {p: {"obs_identical_pairs": ident[(r, p)],
                              "counterexamples": found[(r, p)],
                              "examples": examples[(r, p)]}
                          for p in ("DEFENDER", "ENEMY")} for r in RUNGS}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scales", type=int, nargs="+", default=[4, 2])
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--tick-stride", type=int, default=6)
    ap.add_argument("--max-ticks", type=int, default=240)
    ap.add_argument("--promote", action="store_true")
    a = ap.parse_args()
    seeds = list(range(17600001, 17600001 + a.n_seeds))

    print(f"{LABEL}  {_now()}")
    print(f"  rungs: {RUNGS}")
    print(f"  perturbations: DEFENDER (in-cell), ENEMY (in-cell)")
    print(f"  a counterexample counts only if the AUGMENTED fingerprint is byte-identical")
    print(f"  PREDICTION: DEFENDER dies at +TEAMMATES; ENEMY survives it and dies "
          f"only at +TEAMMATES+ENEMIES\n", flush=True)

    t0 = time.time(); out = {}
    for n in a.scales:
        r = probe(n, seeds, a.device, a.tick_stride, a.max_ticks)
        out[f"{n}v{n}"] = r
        print(f"  {n}v{n} ({r['n_defenders']} defenders)  attempts="
              f"D:{r['perturbation_attempts']['DEFENDER']} E:{r['perturbation_attempts']['ENEMY']}")
        for rung in RUNGS:
            d = r["rungs"][rung]["DEFENDER"]; e = r["rungs"][rung]["ENEMY"]
            print(f"    {rung:<20s} DEFENDER: {d['counterexamples']:>3} ce "
                  f"({d['obs_identical_pairs']} id-pairs)   "
                  f"ENEMY: {e['counterexamples']:>3} ce ({e['obs_identical_pairs']} id-pairs)")
        print(flush=True)

    cfg = json.dumps({"scales": a.scales, "n_seeds": a.n_seeds,
                      "tick_stride": a.tick_stride, "max_ticks": a.max_ticks}, sort_keys=True)
    rid = f"{_now().replace(':','').replace('-','')}_{hashlib.sha256(cfg.encode()).hexdigest()[:8]}"
    rec = {"record": f"{LABEL} minimal observation repair selection",
           "status": "FROZEN_RESULT", "utc": _now(), "run_id": rid,
           "arm": "DIAGNOSTIC", "confirmatory": False,
           "study_class": "MECHANISTIC_FOLLOW_UP",
           "builds_on": "ARM1_TEST0_COUNTEREXAMPLE (observation defect, established)",
           "config_signature": json.loads(cfg), "elapsed_s": round(time.time() - t0, 1),
           "READING": "The lowest rung at which counterexamples vanish for BOTH "
                      "perturbation families is the minimal geometric repair. If they "
                      "survive TEAMMATES_ENEMIES, exact geometry is insufficient and an "
                      "explicit role/assignment signal r_i is required.",
           "results": out}
    p = SD / f"{LABEL}_{rid}_RESULT.json"
    p.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    if a.promote:
        (SD / f"{LABEL}_CANONICAL.json").write_text(json.dumps(
            {"record": f"CANONICAL {LABEL}", "points_to": p.name, "run_id": rid,
             "promoted_utc": _now()}, indent=2), encoding="utf-8")
    print(f"  -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
