"""Behavioral diagnostic for the 4v4 exploratory specialists.

Implements 4V4_EXPLORATORY_AS_DIAGNOSTIC_PROTOCOL.json#MEASUREMENT_CATEGORIES.

DIAGNOSTIC, NOT A GATE. Deliberately separate from eval_specialist_crossover_scaled.py so a
bug here can never corrupt or block that one-shot sealed record. Deliberately REUSES the same
14400001..14400128 seed block the crossover eval spends -- re-deriving mechanism from a
published verdict, the same convention DEFENDER_STACKING_DIAGNOSTIC.json used. No new seed is
spent; this script writes no verdict-bearing record.

These are DISCOVERY measurements. None of them may be read as evidence for or against the
crossover gate itself, and none may be used to justify why the closed V1/V2 scripted probe
passed or failed. They exist only to inform the design of a possible future 4v4 confirmatory
strategic-demand specification.

Measured (concrete operational proxies for the protocol's stated categories):
  own_half_frac         fraction of ticks each agent is on its own side (defensive posture)
  n_intruders           simultaneous legal red intruders (alive, untagged, on blue's side)
  distinct_response_frac  of ticks with k>=2 intruders, fraction where the nearest-blue-agent
                        assignment to each intruder is INJECTIVE (distinct agents cover
                        distinct intruders) rather than converging on one
  defensive_spread      std-dev of position among agents currently on their own half
  carrier_support_dist  mean distance from the flag carrier to its nearest teammate, while
                        carrying
  time_past_midfield    fraction of ticks each agent spends on the ENEMY side
  first_flag_pressure_tick  first tick any blue agent is within gate2b_defender_hold_radius of
                        the enemy flag (a proxy for "arrived to press")

Run:  python experiments/diagnose_4v4_exploratory_specialist_behavior.py \
          --pi-a-path <ckpt> --pi-b-path <ckpt> --n-seeds 16 --device cpu

B3 track: Pole B is a candidate genome, not canonical ``pole_B_genome(4)``. Pass
``--pole-b-genome-json`` (and the track's own ``--seed-base`` / ``--out``) or the
diagnostic would profile behaviour against the WRONG Pole B while reporting B3
checkpoints. The seed-reuse convention above is unchanged: the block passed to
``--seed-base`` is the track's already-spent sealed crossover block, so the
behaviour observed corresponds to the very episodes the sealed cells summarize.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts" / "strategic_demand" / "sppo" / "4V4_EXPLORATORY_SPECIALIST_BEHAVIOR_DIAGNOSTIC.json"
SEED_BASE = 14_400_001  # the SAME block the sealed crossover eval spends; reused deliberately
N_AGENTS = 4
BASE_KEY = {"A": "OP6", "B": "OP7"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def diagnose_episode(env, core, policy) -> dict:
    import torch

    own_half_ticks = np.zeros(N_AGENTS)
    enemy_half_ticks = np.zeros(N_AGENTS)
    total_ticks = 0
    n_intruders_hist = []
    distinct_ok, distinct_total = 0, 0
    spread_samples = []
    carrier_dists = []
    first_pressure_tick = None
    terminal = None

    import experiments.r2_learned_crossover as R2

    obs = env.reset()
    for t in range(R2.MAX_STEPS):
        blue_on_home = core._is_on_home_side("blue", core.blue_x)[0].cpu().numpy()
        own_half_ticks += blue_on_home
        enemy_half_ticks += (~blue_on_home)
        total_ticks += 1

        red_on_blue = core._is_on_home_side("blue", core.red_x)[0]
        intruders = (core.red_alive[0] & (~core.red_tagged[0]) & red_on_blue)
        k = int(intruders.sum().item())
        n_intruders_hist.append(k)

        if k >= 2:
            distinct_total += 1
            bx, by = core.blue_x[0].cpu().numpy(), core.blue_y[0].cpu().numpy()
            rx, ry = core.red_x[0].cpu().numpy(), core.red_y[0].cpu().numpy()
            idxs = np.flatnonzero(intruders.cpu().numpy())
            nearest = []
            for i in idxs:
                d = np.sqrt((bx - rx[i]) ** 2 + (by - ry[i]) ** 2)
                nearest.append(int(np.argmin(d)))
            if len(set(nearest)) == len(nearest):
                distinct_ok += 1

        if blue_on_home.any():
            bx, by = core.blue_x[0].cpu().numpy(), core.blue_y[0].cpu().numpy()
            m = blue_on_home
            if m.sum() >= 2:
                spread_samples.append(float(np.sqrt(bx[m].var() + by[m].var())))

        carrying = core.blue_carrying[0].cpu().numpy()
        if carrying.any():
            bx, by = core.blue_x[0].cpu().numpy(), core.blue_y[0].cpu().numpy()
            ci = int(np.argmax(carrying))
            others = [j for j in range(N_AGENTS) if j != ci]
            if others:
                d = min(float(np.sqrt((bx[ci] - bx[j]) ** 2 + (by[ci] - by[j]) ** 2))
                        for j in others)
                carrier_dists.append(d)

        if first_pressure_tick is None:
            from gpu_env._core._scripted_blue_styles import gate2b_defender_hold_radius
            fx, fy = core.red_flag_pos[0, 0].item(), core.red_flag_pos[0, 1].item()
            bx, by = core.blue_x[0].cpu().numpy(), core.blue_y[0].cpu().numpy()
            radius = gate2b_defender_hold_radius(core.cfg)
            if (np.sqrt((bx - fx) ** 2 + (by - fy) ** 2) <= radius).any():
                first_pressure_tick = t

        with torch.no_grad():
            action, _ = policy.predict(obs, deterministic=True)
        env.step_async(action)
        obs, _r, done, info = env.step_wait()
        obs["global_state"] = env.state()
        if bool(np.asarray(done).any()):
            i0 = info[0] if isinstance(info, (list, tuple)) else info
            res = (i0 or {}).get("episode_result") or {}
            terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
            break

    if terminal is None:
        terminal = (int(core.blue_score[0]), int(core.red_score[0]))
    blue, red = terminal
    return {
        "own_half_frac": (own_half_ticks / max(1, total_ticks)).tolist(),
        "time_past_midfield": (enemy_half_ticks / max(1, total_ticks)).tolist(),
        "mean_n_intruders": float(np.mean(n_intruders_hist)) if n_intruders_hist else 0.0,
        "max_n_intruders": int(max(n_intruders_hist)) if n_intruders_hist else 0,
        "distinct_response_frac": (distinct_ok / distinct_total) if distinct_total else None,
        "distinct_total_ticks": distinct_total,
        "defensive_spread_mean": float(np.mean(spread_samples)) if spread_samples else None,
        "carrier_support_dist_mean": float(np.mean(carrier_dists)) if carrier_dists else None,
        "first_flag_pressure_tick": first_pressure_tick,
        "episode_ticks": total_ticks,
        "win": int(blue > red), "blue": blue, "red": red, "margin": blue - red,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi-a-path", required=True)
    ap.add_argument("--pi-b-path", required=True)
    ap.add_argument("--n-seeds", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed-base", type=int, default=SEED_BASE,
                    help="seed block to reuse; defaults to the exploratory track's 14400001. "
                         "A different track must pass its OWN already-spent sealed crossover "
                         "block so the behaviour matches that track's sealed episodes.")
    ap.add_argument("--pole-b-genome-json", default=None,
                    help="JSON SDSGenome candidate used INSTEAD of canonical pole_B_genome(N). "
                         "REQUIRED for any track whose Pole B is a candidate (e.g. B3-3).")
    ap.add_argument("--out", default=None,
                    help="output JSON path; defaults to the exploratory track's record")
    args = ap.parse_args()

    import experiments.r2_learned_crossover as R2
    from experiments.opponent_spec import (
        _with_full_team_defender_gate,
        assert_live_opponent_batch, install_keyed_opponent_overlays,
        pole_A_genome, pole_B_genome,
    )
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    R2.AGENTS = N_AGENTS
    device = args.device
    seed_base = int(args.seed_base)
    out_path = Path(args.out) if args.out else OUT

    if args.pole_b_genome_json:
        from experiments.sds_genome import SDSGenome

        pole_b = _with_full_team_defender_gate(
            SDSGenome.from_dict(
                json.loads(Path(args.pole_b_genome_json).read_text(encoding="utf-8"))
            ),
            N_AGENTS,
        )
        pole_b_source = f"CANDIDATE_OVERRIDE:{pole_b.genome_id} from {args.pole_b_genome_json}"
    else:
        pole_b = pole_B_genome(N_AGENTS)
        pole_b_source = "canonical_pole_B_genome"
    genomes_by_pole = {"A": {"OP6": pole_A_genome(N_AGENTS)}, "B": {"OP7": pole_b}}

    probe = R2.build_env(device, seed_base)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()

    policies = {"pi_A": load_custom_ppo_policy(str(args.pi_a_path), obs_space, act_space, device=device),
                "pi_B": load_custom_ppo_policy(str(args.pi_b_path), obs_space, act_space, device=device)}

    print(f"4V4 SPECIALIST BEHAVIOR DIAGNOSTIC  {_now()}")
    print(f"  DIAGNOSTIC, NOT A GATE. Reuses seeds {seed_base}+ deliberately (that track's "
          f"sealed crossover block); spends nothing new.")
    print(f"  pole_B  {pole_b_source}")
    print(f"  out     {out_path}\n", flush=True)

    from experiments.tqdm_loop import set_postfix, tqdm_iter

    results = {"pi_A": {"A": [], "B": []}, "pi_B": {"A": [], "B": []}}
    cells = [(name, pole, seed_base + i)
             for name in ("pi_A", "pi_B")
             for pole in ("A", "B")
             for i in range(args.n_seeds)]
    bar = tqdm_iter(cells, desc="4v4 behavior diagnostic", unit="ep")
    for name, pole, seed in bar:
        set_postfix(bar, f"{name}@Pole{pole} seed={seed}")
        env = R2.build_env(device, seed)
        core = env.core
        try:
            policies[name].reset_strategy()
            core._bt_profile_override = None
            core._sds_opening_hold_steps = 0
            genomes = genomes_by_pole[pole]
            install_keyed_opponent_overlays(core, genomes)
            key = BASE_KEY[pole]
            env.env_method("set_phase", phase_from_tag(key))
            env.env_method("set_next_opponent", "SCRIPTED", key)
            assert_live_opponent_batch(core, genomes, allowed_keys=(key,),
                                       context=f"4v4 behavior diagnostic {name}@{pole} seed {seed}")
            results[name][pole].append(diagnose_episode(env, core, policies[name]))
        finally:
            env.close()
        if seed == seed_base + args.n_seeds - 1:
            print(f"  {name} on Pole {pole}: {len(results[name][pole])} episodes diagnosed", flush=True)

    def agg(key, path):
        vals = []
        for ep in path:
            v = ep
            for p in key.split("."):
                v = v[p] if not isinstance(v, list) else v
            if v is not None:
                vals.append(v)
        return vals

    summary = {}
    for name in ("pi_A", "pi_B"):
        for pole in ("A", "B"):
            eps = results[name][pole]
            own_half = np.array([e["own_half_frac"] for e in eps])
            summary[f"{name}@{pole}"] = {
                "n_episodes": len(eps),
                "mean_own_half_frac_per_agent": own_half.mean(axis=0).tolist() if len(eps) else None,
                "mean_n_intruders": float(np.mean([e["mean_n_intruders"] for e in eps])) if eps else None,
                "max_n_intruders_seen": max((e["max_n_intruders"] for e in eps), default=0),
                "distinct_response_frac": (
                    float(np.mean([e["distinct_response_frac"] for e in eps
                                   if e["distinct_response_frac"] is not None]))
                    if any(e["distinct_response_frac"] is not None for e in eps) else None),
                "defensive_spread_mean": (
                    float(np.mean([e["defensive_spread_mean"] for e in eps
                                   if e["defensive_spread_mean"] is not None]))
                    if any(e["defensive_spread_mean"] is not None for e in eps) else None),
                "carrier_support_dist_mean": (
                    float(np.mean([e["carrier_support_dist_mean"] for e in eps
                                   if e["carrier_support_dist_mean"] is not None]))
                    if any(e["carrier_support_dist_mean"] is not None for e in eps) else None),
                "mean_first_flag_pressure_tick": (
                    float(np.mean([e["first_flag_pressure_tick"] for e in eps
                                   if e["first_flag_pressure_tick"] is not None]))
                    if any(e["first_flag_pressure_tick"] is not None for e in eps) else None),
            }

    print("\n  SUMMARY (descriptive only -- no gate)")
    for k, v in summary.items():
        print(f"    {k}: {json.dumps(v)}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "record": "4v4 specialist behavior diagnostic",
        "status": "DIAGNOSTIC_NOT_A_GATE", "utc": _now(),
        "implements": "4V4_EXPLORATORY_AS_DIAGNOSTIC_PROTOCOL.json#MEASUREMENT_CATEGORIES",
        "seeds_reused_from": f"that track's own already-spent sealed crossover block "
                             f"({seed_base}+), deliberately, no new seed spent",
        "seed_base": seed_base,
        "pole_b_source": pole_b_source,
        "n_seeds_diagnosed_per_cell": args.n_seeds,
        "checkpoints": {"pi_A": str(args.pi_a_path), "pi_B": str(args.pi_b_path)},
        "summary": summary,
        "raw_per_episode": results,
    }, indent=2, default=str), encoding="utf-8")
    print(f"\n  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
