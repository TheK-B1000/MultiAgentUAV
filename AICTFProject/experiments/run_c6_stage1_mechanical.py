"""C6 Stage 1 — mechanical validation of the two opponent families.

Answers ONLY: does each family actually exert the mechanics its profile sets?
There is no carrier_survival analysis here, no response enumeration, and no
ranking. Stage 1 is engineering, and profiles MAY be corrected on a failure
precisely because nothing scientific has been measured.

Operationalization, declared before running. Blue attacks toward x > mid, so
red's home is x > mid and red's pressure on blue is red agents at x < mid:

    offensive_pressure   mean fraction of alive red agents in blue's half
    home_retention       fraction of STEPS with >=1 alive red in red's own half
    red_contest_fraction mean fraction of alive red agents in a CONTESTING role
                         (ROLE_DEFENDER / ROLE_INTERCEPTOR) from bt_red_role

red_contest_fraction replaces blue_denial, which was construct-invalid: it was
1 - blue's advance, an OUTCOME produced by both sides. C6A threatens blue's home,
blue stays back to defend, blue advances less, and "denial" rose -- while C6A
blocked nobody. See C6_STAGE1_METRIC_ERRATUM.json. Profiles and bars unchanged.

Each family is checked against the OP6-OP12 mixture mean on the knobs it
deliberately sets:

  C6A sets enable_defender=False, min_alive_for_defender=99,
  intercept_block_base=0.10. It must show offensive_pressure >= mixture,
  home_retention BELOW mixture (it really does abandon its half), and
  red_contest_fraction BELOW mixture (it really does decline to contest).

  C6B sets enable_defender=True, intercept_block_base=0.92, enable_2v1=True,
  lock_attacker=20. It must show offensive_pressure >= mixture (the pressure is
  genuinely sustained, not merely a wall), home_retention ABOVE mixture, and
  red_contest_fraction ABOVE mixture.

home_retention is MEASURED, not inferred from enable_defender. That knob is a
capability gate; whether it produces actual home allocation is a claim about BT
semantics, and checking it is exactly what Stage 1 is for.

An earlier version asked C6A for a CONDITIONAL rise in pressure with blue
commitment. That contradicted its own configuration -- an unconditionally
attacking family expresses "your exposed home is vulnerable" while failing such
a correlation -- and the correlation is confoundable by trajectory. The
conditional figure is still reported, DESCRIPTIVE only, never pass-critical.

Nothing here states which legal team response should win against either family.
That is prohibited by the freeze and is Stage 2's question.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

FROZEN = ROOT / "artifacts/c6_preregistration/C6_CONSTRUCTED_DEMAND_FROZEN.json"
OUT = ROOT / "artifacts/c6_stage1/C6_STAGE1_RESULT.json"

# Declared before the run. Each family is judged against the mixture mean.
MARGIN = 0.02
ROLE_DEFENDER, ROLE_INTERCEPTOR = 1, 3


def measure(policy, *, opponent, seed, device):
    from experiments.eval_v6i9_map_awareness import (
        _adapt_obs_for_policy, _done, _predict, _reset_obs, _unpack_step,
    )
    from experiments.run_g0_v2_evaluation import (
        AGENTS, CANONICAL_MAP, EPISODE_HORIZON, V2_RULES, legal_context,
    )
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    from rl.evaluation.opponent_resolution import (
        get_opponent_key, set_opponent, validate_opponent_name,
    )

    requested = validate_opponent_name(opponent)
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=AGENTS, max_red_agents=AGENTS,
        map_set="train", map_layout=CANONICAL_MAP,
        max_decision_steps=EPISODE_HORIZON, aquaticus_profile=True,
        rules_profile="OURS", device=device, seed=int(seed),
        obstacle_obs_channel=True, tag_telemetry_enabled=True, **V2_RULES,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    inner = policy.model if hasattr(policy, "model") else policy
    if hasattr(inner, "eval"):
        inner.eval()

    def _np(t):
        return t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t)

    rows = []
    try:
        set_opponent(env, requested)
        obs = _reset_obs(env.reset())
        if get_opponent_key(env) != requested:
            raise RuntimeError("opponent drift")
        core.drain_tag_events()
        mid_x = float(core.cols) * 0.5
        for _ in range(EPISODE_HORIZON + 8):
            ctx = legal_context(core)
            red_pos, blue_pos = _np(core.red_pos)[0], _np(core.blue_pos)[0]
            red_alive = _np(core.red_alive)[0].astype(bool)
            blue_alive = _np(core.blue_alive)[0].astype(bool)
            n_red = max(int(red_alive.sum()), 1)
            red_in_blue_half = (red_pos[:, 0] < mid_x) & red_alive
            red_at_home = (red_pos[:, 0] > mid_x) & red_alive
            # Red's OWN role choice, not blue's resulting position.
            roles = _np(core.bt_red_role)[0]
            contesting = np.isin(roles, (ROLE_DEFENDER, ROLE_INTERCEPTOR)) & red_alive
            rows.append({
                "blue_forward": int(ctx["agents_forward"]),
                "offensive_pressure": float(red_in_blue_half.sum()) / n_red,
                "home_present": 1.0 if int(red_at_home.sum()) >= 1 else 0.0,
                "red_contest_fraction": float(contesting.sum()) / n_red,
                "role_hist": roles.tolist(),
            })
            action = _predict(policy, _adapt_obs_for_policy(obs, policy))
            obs, _, done, _infos = _unpack_step(env.step(action))
            if _done(done):
                break
    finally:
        env.close()
    return rows


def summarize(rows):
    op = np.array([r["offensive_pressure"] for r in rows])
    hp = np.array([r["home_present"] for r in rows])
    bf = np.array([r["blue_forward"] for r in rows])
    rc = np.array([r["red_contest_fraction"] for r in rows])
    hi, lo = op[bf >= 2], op[bf == 0]
    cond = (float(hi.mean()) - float(lo.mean())) if hi.size and lo.size else None
    return {
        "n_steps": len(rows),
        "offensive_pressure": round(float(op.mean()), 4),
        "home_retention": round(float(hp.mean()), 4),
        "red_contest_fraction": round(float(rc.mean()), 4),
        # DESCRIPTIVE ONLY -- never a pass criterion; see module docstring.
        "descriptive_conditional_response": None if cond is None else round(cond, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--policy", type=int, default=3200001)
    args = ap.parse_args()

    from experiments.run_g0_v2_evaluation import resolve_cnn_channels
    from experiments.run_g0_v2_seed import OPPONENTS
    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.evaluation.checkpoint import load_policy

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    base = int(frozen["stage_1_mechanical_validation"]["seed_block"][0])

    tag = f"g0_v5_long_seed{args.policy}"
    ck = ROOT / "artifacts" / "g0_v5_long" / tag / "ckpts" / f"final_{tag}.zip"
    payload = read_checkpoint_payload(str(ck), map_location="cpu")
    pol = load_policy(str(ck), device=args.device,
                      num_cnn_channels=resolve_cnn_channels(payload, context=str(ck)))

    print("=" * 74)
    print("C6 STAGE 1 — mechanical validation (engineering only, no strategy claim)")
    print(f"  block {base}..{base + args.episodes - 1}  policy {args.policy}")
    print("=" * 74, flush=True)

    results = {}
    for opp in ["C6A", "C6B"] + list(OPPONENTS):
        rows = []
        for i in range(args.episodes):
            rows += measure(pol, opponent=opp, seed=base + i, device=args.device)
        results[opp] = s = summarize(rows)
        print(f"  {opp:5s} offence={s['offensive_pressure']:.3f} "
              f"home={s['home_retention']:.3f} contest={s['red_contest_fraction']:.3f}", flush=True)

    mix = [results[o] for o in OPPONENTS]
    mix_pressure = float(np.mean([m["offensive_pressure"] for m in mix]))
    mix_home = float(np.mean([m["home_retention"] for m in mix]))
    mix_contest = float(np.mean([m["red_contest_fraction"] for m in mix]))

    a, b = results["C6A"], results["C6B"]
    a_checks = {
        "sustains_offence": bool(a["offensive_pressure"] >= mix_pressure - MARGIN),
        "abandons_home": bool(a["home_retention"] < mix_home - MARGIN),
        "declines_to_contest": bool(a["red_contest_fraction"] < mix_contest - MARGIN),
    }
    b_checks = {
        "sustains_offence": bool(b["offensive_pressure"] >= mix_pressure - MARGIN),
        "retains_home": bool(b["home_retention"] > mix_home + MARGIN),
        "denies_advance": bool(b["red_contest_fraction"] > mix_contest + MARGIN),
    }
    a_ok, b_ok = all(a_checks.values()), all(b_checks.values())
    verdict = "STAGE1_PASS" if (a_ok and b_ok) else "STAGE1_FAIL"

    out = {
        "record": "C6 Stage 1 — mechanical validation",
        "verdict": verdict,
        "purpose": "confirm each family EXPRESSES the mechanics its profile sets. "
                   "No strategy or reversal claim is made or implied.",
        "criteria_declared_before_running": {
            "margin_vs_mixture_mean": MARGIN,
            "family_A": "offensive_pressure >= mixture AND home_retention < mixture "
                        "AND red_contest_fraction < mixture",
            "family_B": "offensive_pressure >= mixture AND home_retention > mixture "
                        "AND red_contest_fraction > mixture",
            "note": "each check validates a knob deliberately set, MEASURED rather than "
                    "inferred from the knob. descriptive_conditional_response is not a "
                    "criterion.",
        },
        "family_A_passed": bool(a_ok), "family_A_checks": a_checks,
        "family_B_passed": bool(b_ok), "family_B_checks": b_checks,
        "mixture_reference": {
            "offensive_pressure_mean": round(mix_pressure, 4),
            "home_retention_mean": round(mix_home, 4),
            "red_contest_fraction_mean": round(mix_contest, 4),
            "opponents": list(OPPONENTS),
        },
        "per_opponent": results,
        "seed_block": [base, base + args.episodes - 1],
        "policy": args.policy,
        "stage_2_authorized": verdict == "STAGE1_PASS",
        "on_fail": "the PROFILE may be corrected and Stage 1 re-run. No reversal or utility "
                   "number exists at Stage 1 and Stage 2 has not been touched.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nVERDICT: {verdict}")
    print(f"  mixture: offence={mix_pressure:.3f} home={mix_home:.3f} contest={mix_contest:.3f}")
    print(f"  C6A pass={a_ok}  {a_checks}")
    print(f"  C6B pass={b_ok}  {b_checks}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
