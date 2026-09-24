r"""ARM 1 / TEST_0: observation-equivalence counterexample probe.

Implements ARM1_TEACHER_IDENTIFIABILITY_SPEC.json#TEST_0_OBSERVATION_EQUIVALENCE_COUNTEREXAMPLE.

Seeks a CONSTRUCTIVE proof that the scripted GUARD teacher is not a deterministic
function of the student's observation at N>=4:

    exists two physically valid states with BYTE-IDENTICAL student input for some
    agent i, whose projected teacher targets differ.

Mechanism (verified from the observation layout, see spec):
  * teammates reach the student ONLY via grid channel 1, a CELL-QUANTIZED
    set-scatter using `.round()`;
  * NO vec feature depends on teammate positions;
  * the teacher's assignment uses EXACT continuous defender coordinates
    (`dd.argmin`).
So moving defender j within its own grid cell is invisible to agent i's
observation but can flip which defender the teacher assigns to a threat.

Because quantization is round-to-nearest, positions 4.6 and 5.4 occupy the SAME
cell -- a perturbation up to ~0.9 cells wide, large enough to plausibly flip an
argmin, not a numerical epsilon.

    python experiments/eval_arm1_test0_counterexample.py --scales 2 4 6
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "ARM1_TEACHER_IDENTIFIABILITY_SPEC.json"
LABEL = "ARM1_TEST0_COUNTEREXAMPLE"


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _obs_bytes(obs, i: int, n: int) -> bytes:
    """Byte serialization of the COMPLETE per-agent policy input for agent i."""
    g = obs["grid"][0, i].detach().cpu().numpy().astype(np.float32)
    v = obs["vec"][0, i].detach().cpu().numpy().astype(np.float32)
    am = obs["agent_mask"][0].detach().cpu().numpy().astype(np.float32)
    m = obs["mask"][0].detach().cpu().numpy().astype(np.float32)
    per = m.shape[0] // n
    mi = m[i * per:(i + 1) * per]
    return (np.ascontiguousarray(g).tobytes() + np.ascontiguousarray(v).tobytes()
            + np.ascontiguousarray(am).tobytes() + np.ascontiguousarray(mi).tobytes())


def _project(tx: float, ty: float, W: np.ndarray) -> int:
    d2 = (W[:, 0] - tx) ** 2 + (W[:, 1] - ty) ** 2
    return int(np.argmin(d2))


def _capture(core, n: int, W: np.ndarray):
    obs = core.get_obs_tensors("blue")
    btx, bty = core._get_scripted_targets("blue")
    tx = btx[0].detach().cpu().numpy()
    ty = bty[0].detach().cpu().numpy()
    return (
        [_obs_bytes(obs, i, n) for i in range(n)],
        [(float(tx[i]), float(ty[i])) for i in range(n)],
        [_project(float(tx[i]), float(ty[i]), W) for i in range(n)],
    )


def probe_scale(scale: int, seeds: list[int], device: str, tick_stride: int,
                max_ticks: int) -> dict:
    import experiments.strategic_demand_searcher as S
    from experiments.opponent_spec import pole_A_genome
    from experiments.strategic_demand_searcher import apply_genome_to_core
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    from experiments.tqdm_loop import set_postfix, tqdm_iter

    S.AGENTS = scale
    genome = pole_A_genome(scale)
    n_def = (scale + 1) // 2
    def_lo = scale - n_def

    found: list[dict] = []
    n_attempts = 0
    n_obs_identical = 0
    n_no_intruder = 0
    n_with_intruder = 0

    bar = tqdm_iter(seeds, desc=f"TEST_0 {scale}v{scale}", unit="seed")
    for seed in bar:
        set_postfix(bar, f"found={len(found)} attempts={n_attempts}")
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
            core.drain_tag_events()
            W = core._macro_targets.detach().cpu().numpy()

            for tick in range(min(max_ticks, S.MAX_STEPS)):
                if tick % tick_stride == 0:
                    # A perturbation can only flip an assignment if there is at
                    # least one live intruder to assign. With zero intruders every
                    # defender holds at home regardless of position, so qA == qB
                    # trivially and the attempt is wasted.
                    intr = (core.red_alive[0] & (~core.red_tagged[0])
                            & core._is_on_home_side("blue", core.red_x)[0])
                    n_intr = int(intr.sum().item())
                    if n_intr < 1:
                        n_no_intruder += 1
                    else:
                        n_with_intruder += 1
                    # snapshot true positions
                    x0 = core.blue_x.clone()
                    y0 = core.blue_y.clone()
                    alive0 = core.blue_alive.clone()
                    tagged0 = core.blue_tagged.clone()
                if tick % tick_stride == 0 and n_intr >= 1:

                    for j in range(def_lo, scale):          # perturb a DEFENDER
                        if not bool(alive0[0, j]) or bool(tagged0[0, j]):
                            continue
                        cx = float(np.round(float(x0[0, j])))
                        cy = float(np.round(float(y0[0, j])))
                        # two in-cell variants, ~0.9 cells apart, both round to (cx,cy)
                        for (ax, ay), (bx, by) in (
                            ((cx - 0.45, cy), (cx + 0.45, cy)),
                            ((cx, cy - 0.45), (cx, cy + 0.45)),
                        ):
                            n_attempts += 1
                            core.blue_x[0, j] = ax; core.blue_y[0, j] = ay
                            obsA, tA, qA = _capture(core, scale, W)
                            core.blue_x[0, j] = bx; core.blue_y[0, j] = by
                            obsB, tB, qB = _capture(core, scale, W)
                            core.blue_x.copy_(x0); core.blue_y.copy_(y0)

                            for i in range(scale):
                                if i == j:
                                    continue            # perturbed agent excluded
                                if obsA[i] != obsB[i]:
                                    continue            # obs changed -> not in the class
                                n_obs_identical += 1
                                if qA[i] != qB[i]:
                                    found.append({
                                        "seed": seed, "tick": tick,
                                        "perturbed_defender_j": j,
                                        "subject_agent_i": i,
                                        "i_is_defender": bool(i >= def_lo),
                                        "j_pos_A": [ax, ay], "j_pos_B": [bx, by],
                                        "same_grid_cell": [cx, cy],
                                        "target_i_A": tA[i], "target_i_B": tB[i],
                                        "q_idx_A": qA[i], "q_idx_B": qB[i],
                                        "obs_bytes_identical_verified": True,
                                    })
                env.step_async(env.action_space.sample() * 0)
                _o, _r, done, _info = env.step_wait()
                if bool(np.asarray(done).any()):
                    break
        finally:
            env.close()
        if len(found) >= 25:
            break

    return {"scale": f"{scale}v{scale}", "n_defenders": n_def,
            "n_perturbation_attempts": n_attempts,
            "n_pairs_with_identical_obs": n_obs_identical,
            "n_tick_samples_with_intruder": n_with_intruder,
            "n_tick_samples_skipped_no_intruder": n_no_intruder,
            "n_counterexamples": len(found),
            "counterexamples": found[:10]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scales", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--n-seeds", type=int, default=4)
    ap.add_argument("--tick-stride", type=int, default=8)
    ap.add_argument("--max-ticks", type=int, default=120)
    ap.add_argument("--promote", action="store_true",
                    help="make this run the CANONICAL record (deliberate act; "
                         "only for a full matched-budget run)")
    a = ap.parse_args()

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: spec not frozen: {spec.get('status')!r}")
    lo, hi = spec["SEEDS"]["block"]
    seeds = list(range(lo, lo + a.n_seeds))

    print(f"{LABEL}  {_now()}")
    print(f"  implements  {SPEC.name}#TEST_0")
    print(f"  scales      {a.scales}   seeds {seeds[0]}..{seeds[-1]}")
    print(f"  mechanism   perturb a DEFENDER within its grid cell (round-to-nearest,")
    print(f"              so +-0.45 stays in cell); check another agent's obs is")
    print(f"              BYTE-identical while its projected teacher target differs")
    print(f"  PREDICTION  must NOT fire at 2v2 (argmin over 1 element)\n", flush=True)

    t0 = time.time()
    out = {}
    for n in a.scales:
        r = probe_scale(n, seeds, a.device, a.tick_stride, a.max_ticks)
        out[f"{n}v{n}"] = r
        verdict = ("COUNTEREXAMPLE FOUND" if r["n_counterexamples"] else "none found")
        print(f"  {n}v{n}: {r['n_counterexamples']} counterexample(s) from "
              f"{r['n_perturbation_attempts']} perturbations "
              f"({r['n_pairs_with_identical_obs']} obs-identical pairs)  -> {verdict}",
              flush=True)

    rec = {
        "record": f"{LABEL} observation-equivalence counterexample probe",
        "status": "FROZEN_RESULT", "one_shot": False, "utc": _now(),
        "arm": "DIAGNOSTIC", "confirmatory": False,
        "implements": f"{SPEC.name}#TEST_0_OBSERVATION_EQUIVALENCE_COUNTEREXAMPLE",
        "search_budget": {"n_seeds": a.n_seeds, "tick_stride": a.tick_stride,
                          "max_ticks": a.max_ticks},
        "elapsed_s": round(time.time() - t0, 1),
        "results": out,
        "READING": "a counterexample at N>=4 but NOT at N=2 supports the audit's claim "
                   "that the multi-defender assignment machinery requires information "
                   "erased by the student's observation mapping. Absence of a "
                   "counterexample is 'not constructed', NOT evidence of consistency.",
    }
    # ---- IMMUTABLE, RUN-SPECIFIC OUTPUT (never overwrite a scientific artifact) ----
    # A 6v6-only invocation once silently clobbered a combined 2v2/4v4/6v6 record.
    # This probe is legitimately re-runnable at different search budgets, so the
    # fix is not one-shot refusal but run-specific immutable files plus a pointer.
    import hashlib as _h
    cfg_sig = json.dumps({"scales": a.scales, "n_seeds": a.n_seeds,
                          "tick_stride": a.tick_stride, "max_ticks": a.max_ticks,
                          "device": a.device, "seed_lo": seeds[0]}, sort_keys=True)
    run_id = f"{_now().replace(':', '').replace('-', '')}_{_h.sha256(cfg_sig.encode()).hexdigest()[:8]}"
    rec["run_id"] = run_id
    rec["config_signature"] = json.loads(cfg_sig)
    p = SD / f"{LABEL}_{run_id}_RESULT.json"
    if p.exists():
        raise SystemExit(f"REFUSING: {p.name} already exists; immutable artifacts are "
                         f"never overwritten.")
    p.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    latest = SD / f"{LABEL}_CANONICAL.json"
    prior = []
    if latest.is_file():
        try:
            prior = json.loads(latest.read_text(encoding="utf-8")).get("history", [])
        except Exception:                                    # noqa: BLE001
            prior = []
    hist = (prior + [{"run_id": run_id, "file": p.name, "utc": _now(),
                      "scales": a.scales, "n_seeds": a.n_seeds,
                      "tick_stride": a.tick_stride, "max_ticks": a.max_ticks}])[-25:]

    # Promotion is a DELIBERATE ACT, never a side effect of running. "Most recent"
    # is the wrong rule: a 10-tick smoke run would silently displace a full
    # matched-budget result as the canonical record. --promote is required.
    if a.promote:
        latest.write_text(json.dumps({
            "record": f"CANONICAL {LABEL} run, explicitly promoted",
            "points_to": p.name, "run_id": run_id, "promoted_utc": _now(),
            "config_signature": json.loads(cfg_sig),
            "promotion_rule": "explicit --promote only. Promote the run covering ALL "
                              "scales at a MATCHED search budget; never a partial or "
                              "smoke run.",
            "history": hist,
        }, indent=2), encoding="utf-8")
        print(f"\n  -> {p}\n  -> {latest}  (PROMOTED to canonical)")
    else:
        if latest.is_file():
            doc = json.loads(latest.read_text(encoding="utf-8"))
            doc["history"] = hist
            latest.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        print(f"\n  -> {p}")
        print(f"  (not promoted; canonical pointer unchanged. Use --promote to "
              f"make this the canonical record.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
