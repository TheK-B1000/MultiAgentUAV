"""CCP Phase 0: is (seed, action prefix) -> s_t a function?

Implements CCP_PHASE0_REPLAY_EQUIVALENCE_SPEC.json.

The whole causal-crossover program rests on matched causal branching, and there is no state
save/restore anywhere in gpu_env. Prefix replay is the only route to matched states that
adds no environment surface -- and nobody has ever tested whether it holds. If it does not,
every Phase 1 Q estimate would compare continuations from states that were never the same
state.

This feeds RECORDED actions rather than re-querying the policy, which is the point: Phase 1
will branch four different policies from one state, so what has to be deterministic is the
ENVIRONMENT given a seed and an action sequence.

  primary      EXACT bitwise equality on every observation field, global state, reward, done
  tolerance    NONE. Mismatches are reported with magnitudes, never absorbed.
  control      a deliberately perturbed prefix MUST produce a different state, or the whole
               result is void rather than merely failed

HARD GATE. A non-CONFIRMED verdict stops the program and returns to the PI.
Touches no EVAL block.

Run:  python experiments/ccp_phase0_replay_equivalence.py --device cuda
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

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "CCP_PHASE0_REPLAY_EQUIVALENCE_SPEC.json"
FROZEN_V4 = SD / "HOG_PSP_V4_MODEL_FROZEN.json"
OUT = SD / "CCP_PHASE0_REPLAY_EQUIVALENCE.json"

SEEDS = list(range(11_500_001, 11_500_009))
POLES = ("A", "B")
EVENT_HINTS = ("flag", "capture", "captured", "death", "died", "dead", "tag", "score")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _snap(obs: dict, gs, reward, done) -> dict:
    """Every field, not a convenient subset."""
    out = {f"obs.{k}": np.asarray(v).copy() for k, v in obs.items()}
    out["global_state"] = np.asarray(gs).copy()
    out["reward"] = np.asarray(reward).copy()
    out["done"] = np.asarray(done).copy()
    return out


def _compare(a: dict, b: dict) -> tuple[bool, list[dict]]:
    """Exact bitwise equality. Report magnitudes on mismatch; never absorb them."""
    diffs = []
    keys = sorted(set(a) | set(b))
    for k in keys:
        if k not in a or k not in b:
            diffs.append({"field": k, "issue": "present in only one snapshot"})
            continue
        x, y = a[k], b[k]
        if x.shape != y.shape:
            diffs.append({"field": k, "issue": f"shape {x.shape} vs {y.shape}"})
        elif not np.array_equal(x, y):
            d = np.abs(x.astype(np.float64) - y.astype(np.float64))
            diffs.append({"field": k, "max_abs_diff": float(d.max()),
                          "n_elements_differing": int((x != y).sum()),
                          "of": int(x.size)})
    return (not diffs), diffs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: Phase 0 spec is not frozen: {spec['status']!r}")

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
    )
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    frozen = json.loads(FROZEN_V4.read_text(encoding="utf-8"))
    ck = ROOT / frozen["TERMINAL_CHECKPOINT"]["path"]
    sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    if sha != frozen["TERMINAL_CHECKPOINT"]["sha256"]:
        raise SystemExit("REFUSING: action-source checkpoint sha mismatch")

    print(f"CCP PHASE 0  REPLAY EQUIVALENCE  {_now()}")
    print(f"  seeds {SEEDS[0]}..{SEEDS[-1]}  poles A,B   fresh 115xxxxx namespace")
    print(f"  primary criterion: EXACT bitwise equality, no tolerance\n", flush=True)

    probe = R2.build_env(device, SEEDS[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    policy = load_custom_ppo_policy(str(ck), obs_space, act_space, device=device)
    model = policy.model if hasattr(policy, "model") else policy
    model.eval()

    def setup(seed: int, pole: str):
        env = R2.build_env(device, seed)
        core = env.core
        core._bt_profile_override = None
        core._sds_opening_hold_steps = 0
        genomes = {"OP6": pole_A_genome()} if pole == "A" else {}
        install_keyed_opponent_overlays(core, genomes)
        key = P0.POLES[pole]
        env.env_method("set_phase", phase_from_tag(key))
        env.env_method("set_next_opponent", "SCRIPTED", key)
        obs = env.reset()
        obs["global_state"] = env.state()
        assert_live_opponent_batch(core, genomes, allowed_keys=(key,),
                                   context=f"ccp p0 {pole} seed {seed}")
        return env, obs

    def record(seed: int, pole: str, z: int = 0) -> dict:
        """Deterministic episode; states[i] is the state after i actions."""
        env, obs = setup(seed, pole)
        try:
            policy.fixed_latent_strategy = True
            policy.fixed_latent_strategy_id = int(z)
            policy.reset_strategy()
            states = [_snap(obs, obs["global_state"], np.zeros(1), np.zeros(1, dtype=bool))]
            actions, infos = [], []
            for _ in range(R2.MAX_STEPS):
                action, _ = policy.predict(obs, deterministic=True)
                actions.append(np.asarray(action).copy())
                env.step_async(action)
                obs, r, done, info = env.step_wait()
                obs["global_state"] = env.state()
                states.append(_snap(obs, obs["global_state"], r, done))
                infos.append(info)
                if bool(np.asarray(done).any()):
                    break
            return {"states": states, "actions": actions, "infos": infos, "T": len(actions)}
        finally:
            env.close()

    def replay(seed: int, pole: str, actions: list, k: int,
               perturb_at: int | None = None, perturb_with=None) -> dict:
        """FRESH env, same seed, first k recorded actions. Continuing the original tests nothing."""
        env, obs = setup(seed, pole)
        try:
            state = _snap(obs, obs["global_state"], np.zeros(1), np.zeros(1, dtype=bool))
            for i in range(k):
                a = np.asarray(actions[i]).copy()
                if perturb_at is not None and i == perturb_at:
                    a = np.asarray(perturb_with).copy()   # a DIFFERENT but valid recorded action
                env.step_async(a)
                obs, r, done, _info = env.step_wait()
                obs["global_state"] = env.state()
                state = _snap(obs, obs["global_state"], r, done)
                if bool(np.asarray(done).any()) and i < k - 1:
                    break
            return state
        finally:
            env.close()

    # ---- event-state availability, recorded rather than silently skipped ----
    probe_ep = record(SEEDS[0], "A")
    info0 = probe_ep["infos"][0] if probe_ep["infos"] else None
    info_keys = []
    if isinstance(info0, (list, tuple)) and info0 and isinstance(info0[0], dict):
        info_keys = sorted(info0[0].keys())
    elif isinstance(info0, dict):
        info_keys = sorted(info0.keys())
    event_keys = [k for k in info_keys if any(h in k.lower() for h in EVENT_HINTS)]
    event_limitation = None
    if not event_keys:
        event_limitation = (
            "The step info dict exposes no flag/capture/death/score key, so event states could "
            "not be added as extra prefix points. Recorded as a STATED LIMITATION of this Phase 0 "
            "result, not silently skipped. Observed info keys: " + (", ".join(info_keys) or "none")
        )
    print(f"  info keys: {info_keys or 'none'}")
    print(f"  event keys usable as extra prefix points: {event_keys or 'NONE (limitation recorded)'}\n",
          flush=True)

    # ------------------------------- primary: replay equivalence ------------
    results, all_diffs = [], []
    for seed in SEEDS:
        for pole in POLES:
            ep = record(seed, pole) if not (seed == SEEDS[0] and pole == "A") else probe_ep
            T = ep["T"]
            points = {0, max(1, round(0.10 * T)), round(0.50 * T),
                      round(0.90 * T), max(0, T - 1)}
            # Event states become extra prefix points. The event is the step where a counter
            # CHANGES, not every step where it is non-zero -- blue_score and red_score are
            # cumulative, so testing truthiness would mark most of the episode as an event.
            prev = None
            for step, inf in enumerate(ep["infos"]):
                rec = inf[0] if isinstance(inf, (list, tuple)) and inf else inf
                if not isinstance(rec, dict):
                    continue
                cur = {k: np.asarray(rec.get(k)).copy() for k in event_keys}
                if prev is not None and any(not np.array_equal(cur[k], prev[k]) for k in event_keys):
                    points.add(step + 1)
                prev = cur
            points = sorted(p for p in points if 0 <= p <= T)
            for k in points:
                if k > T:
                    continue
                got = replay(seed, pole, ep["actions"], k)
                ok, diffs = _compare(ep["states"][k], got)
                results.append({"seed": seed, "pole": pole, "T": T, "prefix_k": int(k),
                                "exact_match": bool(ok),
                                "fraction_of_episode": round(k / T, 3) if T else 0.0})
                if not ok:
                    all_diffs.append({"seed": seed, "pole": pole, "prefix_k": int(k),
                                      "fields": diffs[:6]})
            status = "OK " if all(r["exact_match"] for r in results if r["seed"] == seed
                                  and r["pole"] == pole) else "MISMATCH"
            print(f"  seed {seed} pole {pole}: T={T:4d}  prefixes {points}  {status}", flush=True)

    n_points = len(results)
    n_exact = sum(r["exact_match"] for r in results)
    near_terminal_ok = all(r["exact_match"] for r in results
                           if r["prefix_k"] == max(0, r["T"] - 1))

    # ------------------------------- mandatory negative control -------------
    cseed, cpole = SEEDS[0], "A"
    cep = probe_ep
    cT = cep["T"]
    ck_point = max(2, round(0.50 * cT))
    perturb_at = max(0, ck_point - 2)
    # substitute a DIFFERENT but genuinely valid action, so the control cannot fail merely by
    # feeding the env something out of range
    alt = next((np.asarray(a) for a in cep["actions"]
                if not np.array_equal(np.asarray(a), np.asarray(cep["actions"][perturb_at]))), None)
    if alt is None:
        raise SystemExit("REFUSING: every recorded action is identical; the negative control "
                         "cannot substitute a different valid action, so it would prove nothing")
    perturbed = replay(cseed, cpole, cep["actions"], ck_point,
                       perturb_at=perturb_at, perturb_with=alt)
    ctrl_ok, ctrl_diffs = _compare(cep["states"][ck_point], perturbed)
    control_detected = not ctrl_ok           # perturbation MUST show up
    control = {
        "design": "replay a prefix with one action deliberately changed; the state must DIFFER",
        "seed": cseed, "pole": cpole, "prefix_k": int(ck_point), "action_changed_at_step": int(perturb_at),
        "perturbation_detected": bool(control_detected),
        "fields_that_differed": [d["field"] for d in ctrl_diffs][:8],
        "why": ("A guard that cannot fail proves nothing. If the perturbed replay ALSO matched, the "
                "comparison would not be measuring what it claims and this whole result would be "
                "void rather than merely failed."),
    }
    print(f"\n  negative control: perturbation detected = {control_detected} "
          f"({len(ctrl_diffs)} fields differ)", flush=True)

    # ------------------------------- preregistered verdict ------------------
    if not control_detected:
        verdict = "REPLAY_EQUIVALENCE_FAILED"
        note = "VOID: the negative control did not detect a deliberately perturbed prefix."
    elif n_exact == n_points:
        verdict = "REPLAY_EQUIVALENCE_CONFIRMED"
        note = "Exact bitwise equality at every prefix point on every seed and both poles."
    elif n_exact > 0 and near_terminal_ok:
        verdict = "REPLAY_EQUIVALENCE_PARTIAL"
        note = "Exact at some prefix points and not others."
    else:
        verdict = "REPLAY_EQUIVALENCE_FAILED"
        note = "Mismatches are broad, or reach the near-terminal points Phase 1 most depends on."

    record_out = {
        "record": "CCP Phase 0 replay equivalence",
        "status": "FROZEN_RESULT", "one_shot": True, "utc": _now(),
        "implements": "CCP_PHASE0_REPLAY_EQUIVALENCE_SPEC.json",
        "program": "CAUSAL_CROSSOVER_PROGRAM_OPENING.json",
        "VERDICT": verdict, "note": note,
        "HARD_GATE": ("A non-CONFIRMED verdict stops the program and returns to the PI for a "
                      "decision about adding state save/restore to the environment."),
        "claim_under_test": "(seed, action prefix) -> s_t is a function",
        "criterion": "EXACT bitwise equality on every observation field, global state, reward and done",
        "tolerance_adopted": None,
        "summary": {"prefix_points_compared": n_points, "exact_matches": n_exact,
                    "seeds": len(SEEDS), "poles": list(POLES),
                    "near_terminal_all_exact": bool(near_terminal_ok)},
        "per_point": results,
        "mismatches": all_diffs,
        "negative_control": control,
        "event_state_coverage": {"info_keys_observed": info_keys,
                                 "event_keys_used": event_keys,
                                 "LIMITATION": event_limitation},
        "action_source": {"checkpoint": frozen["TERMINAL_CHECKPOINT"]["path"], "sha256": sha,
                          "why": ("recorded actions from a real policy reach realistic states; the "
                                  "claim under test is environment determinism given those actions, "
                                  "not policy determinism")},
        "EVAL_touched": False,
    }
    OUT.write_text(json.dumps(record_out, indent=2), encoding="utf-8")

    print(f"\n  {n_exact}/{n_points} prefix points exact")
    print(f"  VERDICT: {verdict}")
    print(f"  -> {OUT}")
    return 0 if verdict == "REPLAY_EQUIVALENCE_CONFIRMED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
