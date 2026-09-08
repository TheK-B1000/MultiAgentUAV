"""Decision-leverage diagnostic, pass 2: matched counterfactual branches at selected decisions.

Implements DECISION_LEVERAGE_DIAGNOSTIC_SPEC.json#PASS_2_BRANCH. For each selected decision
state: rebuild the env at the same seed/pole, replay the student's recorded action prefix
(replay_prefix asserts live agreement at every replayed step and REFUSES on divergence), seed
core._rng identically for both branches so subsequent randomness is matched, then run

    BRANCH_S: the student's own action at this step, then student policy to episode end
    BRANCH_T: the counterfactual action at this step, then student policy to episode end

so only the single decision differs.

    mismatch states   -> counterfactual = the LATENT-matched TEACHER's action
                         (z0 -> pi_A, z1 -> pi_B, either pole; see
                          DLD_TEACHER_MATCHING_CLARIFICATION_AMENDMENT.json)
    agreement states  -> counterfactual = the student's own RUNNER-UP legal macro
                         (a decision-sensitivity reference, NOT a matched control --
                          see DLD_TERMINOLOGY_AND_LICENSED_READING_AMENDMENT.json)

Measurement only. No training, no gate.

Run:  python experiments/dld_pass2_branch.py --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.eval_hog_psp_v3 import _mean_ci

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "DECISION_LEVERAGE_DIAGNOSTIC_SPEC.json"
PASS1 = SD / "DLD_PASS1_RECORD_V2.json"
POWER_AMENDMENT = SD / "DLD_POWER_RESOLUTION_AMENDMENT.json"
LATENT_TEACHER_ARRAY = {0: "teacher_A_actions", 1: "teacher_B_actions"}
STUDENT_FROZEN = SD / "TEACHER_DISTILLATION_STUDENT_FROZEN.json"
OUT = SD / "DLD_RESULT.json"
ROWS = SD / "dld_branch_rows.csv"

N_AGENTS = 2
PER_CELL = 10          # frozen: 10 mismatch + 10 agreement per cell
MIN_REMAINING = 40     # frozen eligibility: >= 40 steps left before MAX_STEPS
MIN_STEP = 5           # frozen eligibility: not in the first 5 steps


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def branch_seed(seed: int, step: int) -> int:
    return int.from_bytes(hashlib.sha256(f"DLD|{seed}|{step}".encode()).digest()[:8], "big") % (2 ** 63 - 1)


def stride_sample(cands: list, k: int) -> list:
    """Deterministic evenly-spaced stride sample; no RNG, no outcome inspection."""
    if len(cands) <= k:
        return list(cands)
    idx = np.linspace(0, len(cands) - 1, k).round().astype(int)
    return [cands[i] for i in dict.fromkeys(idx.tolist())]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: spec not frozen: {spec['status']!r}")
    if args.device != spec["DEVICE"]["device"]:
        raise SystemExit(f"REFUSING: device {args.device!r} != frozen {spec['DEVICE']['device']!r}")
    if OUT.is_file() or ROWS.is_file():
        raise SystemExit("REFUSING: a DLD result already exists; one-shot")
    p1 = json.loads(PASS1.read_text(encoding="utf-8"))
    if p1["status"] != "FROZEN_DATASET":
        raise SystemExit("REFUSING: pass 1 not frozen")
    if "latent_matched" not in (p1.get("definitions") or {}):
        raise SystemExit("REFUSING: pass-1 record does not carry the latent-matched definition")
    _probe = np.load(ROOT / p1["episodes"][0]["file"])
    for _need in ("mismatch_latent", "teacher_A_actions", "teacher_B_actions"):
        if _need not in _probe.files:
            raise SystemExit(f"REFUSING: pass-1 shards lack {_need!r}; this is v1 data, which "
                             "would silently reintroduce the pole-matched category error")
    if not POWER_AMENDMENT.is_file():
        raise SystemExit("REFUSING: power-resolution amendment absent; the take-all rule is not authorised")

    import torch
    from experiments.ccp_s2_collect import replay_prefix, setup_env
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
    )
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.teacher_distillation import masked_heads

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("REFUSING: cuda required by the frozen spec but unavailable")
    device = args.device

    srec = json.loads(STUDENT_FROZEN.read_text(encoding="utf-8"))
    sck = ROOT / srec["TERMINAL_CHECKPOINT"]["path"]
    probe = R2.build_env(device, 11_940_001)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    student = load_custom_ppo_policy(str(sck), obs_space, act_space, device=device)

    # ------------------------------------------------------------ selection (frozen rule)
    selected = []
    shortfalls = {}
    for cell in sorted({e["cell"] for e in p1["episodes"]}):
        eps = sorted([e for e in p1["episodes"] if e["cell"] == cell],
                     key=lambda e: (e["episode"], e["seed"]))
        pools = {"mismatch": [], "agreement": []}
        for e in eps:
            d = np.load(ROOT / e["file"])
            dec, mm, n = d["decision_step"], d["mismatch_latent"], int(d["steps"])
            for t in range(n):
                if not dec[t] or t < MIN_STEP or (R2.MAX_STEPS - t) < MIN_REMAINING:
                    continue
                pools["mismatch" if mm[t] else "agreement"].append((e, t))
        for kind, pool in pools.items():
            take = stride_sample(pool, PER_CELL)
            if len(take) < PER_CELL:
                shortfalls[f"{cell}/{kind}"] = {"available": len(pool), "requested": PER_CELL}
            for e, t in take:
                selected.append({"cell": cell, "kind": kind, "seed": e["seed"], "step": t,
                                 "z": e["z"], "pole": e["pole"],
                                 "teacher": e.get("latent_teacher") or e.get("teacher"),
                                 "file": e["file"]})
    print(f"DLD PASS 2 (branch)  {_now()}  device={device}")
    print(f"  selected {len(selected)} branch points "
          f"({sum(s['kind']=='mismatch' for s in selected)} mismatch, "
          f"{sum(s['kind']=='agreement' for s in selected)} agreement)")
    if shortfalls:
        print(f"  SHORTFALLS (recorded, not backfilled): {shortfalls}")
    print(flush=True)

    # ------------------------------------------------------------ branching
    def rollout_from(seed: int, pole: str, z: int, prefix: list, first_action, bseed: int):
        env, obs, core = setup_env(R2, P0, phase_from_tag, install_keyed_opponent_overlays,
                                   pole_A_genome, assert_live_opponent_batch, device, seed, pole)
        try:
            student.fixed_latent_strategy = True
            student.fixed_latent_strategy_id = int(z)
            student.reset_strategy()
            obs = replay_prefix(R2, student, env, obs, prefix, z, f"dld|{seed}|{len(prefix)}")
            core._rng.manual_seed(int(bseed))          # matched randomness across both branches
            a = np.asarray(first_action, dtype=np.int64)
            env.step_async(a)
            obs, _r, done, info = env.step_wait()
            obs["global_state"] = env.state()
            steps = len(prefix) + 1
            terminal = None
            if bool(np.asarray(done).any()):
                i0 = info[0] if isinstance(info, (list, tuple)) else info
                res = (i0 or {}).get("episode_result") or {}
                terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
            while terminal is None and steps < R2.MAX_STEPS:
                act, _ = student.predict(obs, deterministic=True)
                env.step_async(act)
                obs, _r, done, info = env.step_wait()
                obs["global_state"] = env.state()
                steps += 1
                if bool(np.asarray(done).any()):
                    i0 = info[0] if isinstance(info, (list, tuple)) else info
                    res = (i0 or {}).get("episode_result") or {}
                    terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
            if terminal is None:
                terminal = (int(core.blue_score[0]), int(core.red_score[0]))
            return terminal, steps
        finally:
            env.close()

    def runner_up_action(seed: int, pole: str, z: int, prefix: list, s_act: list) -> list:
        """The student's own second-highest-probability legal macro at this state, for the
        free agent with the largest runner-up mass. Deterministic."""
        env, obs, core = setup_env(R2, P0, phase_from_tag, install_keyed_opponent_overlays,
                                   pole_A_genome, assert_live_opponent_batch, device, seed, pole)
        try:
            student.fixed_latent_strategy = True
            student.fixed_latent_strategy_id = int(z)
            student.reset_strategy()
            obs = replay_prefix(R2, student, env, obs, prefix, z, f"dld-ru|{seed}|{len(prefix)}")
            ot = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)
                  for k, v in obs.items()}
            zt = torch.zeros((1,), dtype=torch.long, device=device) + int(z)
            heads = masked_heads(student.model, ot, z_idx=zt)
            best_i, best_a, best_gap = None, None, -1.0
            for i in range(N_AGENTS):
                p = heads[i * 2].logits[0].softmax(-1)          # macro head for agent i
                top2 = torch.topk(p, k=min(2, p.numel()))
                if top2.values.numel() < 2:
                    continue
                alt = int(top2.indices[1])
                if alt == int(s_act[i * 2]):
                    continue
                if float(top2.values[1]) > best_gap:
                    best_gap, best_i, best_a = float(top2.values[1]), i, alt
            if best_i is None:
                return None
            out = list(s_act)
            out[best_i * 2] = best_a
            return out
        finally:
            env.close()

    rows = []
    for k, s in enumerate(selected, 1):
        d = np.load(ROOT / s["file"])
        prefix = [list(map(int, a)) for a in d["student_actions"][:s["step"]]]
        s_act = [int(x) for x in d["student_actions"][s["step"]]]
        if s["kind"] == "mismatch":
            cf = [int(x) for x in d[LATENT_TEACHER_ARRAY[int(s["z"])]][s["step"]]]
            cf_kind = f"latent_matched_teacher_action_{'pi_A' if int(s['z']) == 0 else 'pi_B'}"
        else:
            cf = runner_up_action(s["seed"], s["pole"], s["z"], prefix, s_act)
            cf_kind = "student_runner_up_macro"
            if cf is None:
                rows.append({**s, "skipped": "no_distinct_runner_up", "cf_kind": cf_kind})
                continue
        bs = branch_seed(s["seed"], s["step"])
        (sb, sr), s_steps = rollout_from(s["seed"], s["pole"], s["z"], prefix, s_act, bs)
        (tb, tr), t_steps = rollout_from(s["seed"], s["pole"], s["z"], prefix, cf, bs)
        row = {**{kk: s[kk] for kk in ("cell", "kind", "seed", "step", "z", "pole", "teacher")},
               "cf_kind": cf_kind,
               "branch_S_blue": sb, "branch_S_red": sr, "branch_S_steps": s_steps,
               "branch_T_blue": tb, "branch_T_red": tr, "branch_T_steps": t_steps,
               "L_win": int(tb > tr) - int(sb > sr),
               "L_margin": (tb - tr) - (sb - sr), "skipped": ""}
        rows.append(row)
        if k % 10 == 0 or k == len(selected):
            print(f"  {k}/{len(selected)} branch points done", flush=True)

    import csv
    done = [r for r in rows if not r.get("skipped")]
    with ROWS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def arr(kind, field):
        return np.array([r[field] for r in done if r["kind"] == kind], dtype=np.float64)

    res = {}
    for kind in ("mismatch", "agreement"):
        lm, lw = arr(kind, "L_margin"), arr(kind, "L_win")
        if lm.size == 0:
            res[kind] = {"n": 0}
            continue
        res[kind] = {"n": int(lm.size), "L_margin": _mean_ci(lm), "L_win": _mean_ci(lw),
                     "abs_L_margin": _mean_ci(np.abs(lm))}
    print("\n  DECISION LEVERAGE")
    for kind in ("mismatch", "agreement"):
        r = res[kind]
        if not r.get("n"):
            print(f"    {kind}: no branch points"); continue
        print(f"    {kind:10s} n={r['n']:3d}  L_margin {r['L_margin']['mean']:+.3f} "
              f"[{r['L_margin']['lcb95']:+.3f}, {r['L_margin']['ucb95']:+.3f}]  "
              f"L_win {r['L_win']['mean']:+.3f}  |L_margin| {r['abs_L_margin']['mean']:.3f}")

    OUT.write_text(json.dumps({
        "record": "Decision-leverage diagnostic result", "status": "FROZEN_RESULT",
        "utc": _now(), "device": device,
        "implements": "DECISION_LEVERAGE_DIAGNOSTIC_SPEC.json",
        "student": srec["TERMINAL_CHECKPOINT"]["sha256"],
        "selection": {"per_cell": PER_CELL, "min_remaining": MIN_REMAINING, "min_step": MIN_STEP,
                      "rule": "deterministic evenly-spaced stride over (episode, step)-sorted eligible pool",
                      "shortfalls": shortfalls, "n_selected": len(selected),
                      "n_completed": len(done), "n_skipped": len(rows) - len(done)},
        "counterfactuals": {"mismatch": "LATENT-matched teacher action (z0->pi_A, z1->pi_B)",
                            "agreement": "student's own runner-up legal macro -- a DECISION-SENSITIVITY REFERENCE, not a matched control"},
        "RESULTS": res,
        "bootstrap": {"samples": 20000, "alpha": 0.05, "rng_seed": 7, "unit": "branch point"},
        "IS_NOT_A_GATE": "measurement only; crossover remains the only scientific gate",
        "licensed_reading": "see DLD_TERMINOLOGY_AND_LICENSED_READING_AMENDMENT.json -- a large gap may mean mismatch decisions occur where swapping between two COMPETENT strategies has unusually large downstream consequences; it may NOT be read as 'mismatch states are more important than agreement states'",
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
