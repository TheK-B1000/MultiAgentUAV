"""Blue-side closed-loop behavior capture for frozen 4v4 B3 specialists.

Implements 4V4_B3_BLUE_SIDE_BEHAVIOR_CAPTURE_PROTOCOL.json.

DIAGNOSTIC, NOT A GATE. Replays already-spent sealed confirmatory seeds
(16700001+) so that no new experimental seed budget is consumed. Matched
(seed, pole) pairs for pi_A3 and pi_B3 answer:

  Are the specialists strategically different over full trajectories, or do
  their per-state JSD differences wash out into the same closed-loop role
  pattern?

Uses existing rl.behavior_telemetry (num_attackers/defenders, role buckets,
macro counts) rather than inventing a parallel metric vocabulary.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from macro_actions import MacroAction
from rl.behavior_telemetry import (
    BEHAVIOR_TELEMETRY_NAMES,
    N_ROLE_BUCKET_MI,
    bucket_ids_from_telemetry,
    compute_behavior_telemetry_batch,
)

N_AGENTS = 4
BASE_KEY = {"A": "OP6", "B": "OP7"}
MACRO_NAMES = {int(m): m.name for m in MacroAction}
ROLE4_NAMES = {
    0: "all_push",
    1: "three_attack_one_defend",
    2: "two_attack_two_defend",
    3: "one_attack_three_defend",
    4: "escort_pair",
    5: "intercept_pair",
    6: "turtle_defense",
}

DEFAULT_PI_A = (
    "artifacts/scale_4v4_specialists/pi_A_specialist_4v4_b3/"
    "ckpts/final_pi_A_specialist_4v4_b3.zip"
)
DEFAULT_PI_B = (
    "artifacts/scale_4v4_specialists/pi_B_specialist_4v4_b3/"
    "ckpts/final_pi_B_specialist_4v4_b3.zip"
)
DEFAULT_POLE_B = (
    "artifacts/strategic_demand/sppo/pole_b2_candidates/B3-3_lockdef10_2v1.json"
)
DEFAULT_SEALED_CSV = (
    "artifacts/strategic_demand/sppo/"
    "confirmatory_b3_3_4v4_specialist_crossover_eval_rows.csv"
)
DEFAULT_OUT = (
    "artifacts/strategic_demand/sppo/4V4_B3_BLUE_SIDE_BEHAVIOR_CAPTURE.json"
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sealed_terminal(csv_path: Path, seed: int, policy: str, pole: str) -> tuple[int, int]:
    with csv_path.open(encoding="utf-8") as fh:
        rows = [
            r for r in csv.DictReader(fh)
            if int(r["seed"]) == seed and r.get("pole") == pole and r.get("policy") == policy
        ]
    if len(rows) != 1:
        raise SystemExit(
            f"FIDELITY: expected exactly 1 sealed row for "
            f"policy={policy} pole={pole} seed={seed}, got {len(rows)}"
        )
    return int(rows[0]["blue"]), int(rows[0]["red"])


def _pole_genomes(n: int, pole_b_json: Path | None) -> dict:
    from experiments.opponent_spec import (
        _with_full_team_defender_gate,
        pole_A_genome,
        pole_B_genome,
    )

    if pole_b_json is None:
        pole_b = pole_B_genome(n)
        src = "canonical_pole_B_genome"
    else:
        from experiments.sds_genome import SDSGenome

        pole_b = _with_full_team_defender_gate(
            SDSGenome.from_dict(json.loads(pole_b_json.read_text(encoding="utf-8"))),
            n,
        )
        src = f"CANDIDATE_OVERRIDE:{pole_b.genome_id} from {pole_b_json}"
    return {
        "A": {"OP6": pole_A_genome(n)},
        "B": {"OP7": pole_b},
        "_pole_b_source": src,
    }


def capture_episode(env, core, policy, max_steps: int, obs=None) -> dict:
    """Closed-loop blue-side telemetry for one episode (pre-step state each tick).

    If ``obs`` is provided, the env is assumed already reset (and asserted); do not
    reset again or the sealed-seed RNG advances and fidelity breaks.
    """
    if obs is None:
        obs = env.reset()
        obs["global_state"] = env.state()
    elif "global_state" not in obs:
        obs["global_state"] = env.state()

    n_ticks = 0
    own_half = np.zeros(N_AGENTS, dtype=np.float64)
    enemy_half = np.zeros(N_AGENTS, dtype=np.float64)
    max_penetration = np.zeros(N_AGENTS, dtype=np.float64)
    sum_penetration = np.zeros(N_AGENTS, dtype=np.float64)

    role_hist = np.zeros(N_ROLE_BUCKET_MI, dtype=np.int64)
    role_switches = 0
    prev_role = None
    att_def_series = []
    tel_sum = np.zeros(len(BEHAVIOR_TELEMETRY_NAMES), dtype=np.float64)
    macro_counts = Counter()

    carry_ticks = 0
    carry_started = False
    carry_events = 0
    returns = 0
    blue_score_prev = int(core.blue_score[0].item())

    mine_grab_events = 0
    mine_place_events = 0
    prev_mine_charges = None
    if hasattr(core, "blue_mine_charges"):
        prev_mine_charges = core.blue_mine_charges[0].cpu().numpy().copy()

    first_flag_pressure_tick = None
    role_timeline = []
    terminal = None

    for t in range(max_steps):
        with torch.no_grad():
            action, _ = policy.predict(obs, deterministic=True)
        act = np.asarray(action)
        if act.ndim == 1:
            act_t = torch.as_tensor(act[None, :], device=core.device, dtype=torch.long)
        else:
            act_t = torch.as_tensor(act, device=core.device, dtype=torch.long)

        tel = compute_behavior_telemetry_batch(core, act_t)[0].detach().cpu().numpy()
        _sb, rb, _pb, _adb = bucket_ids_from_telemetry(
            torch.as_tensor(tel[None, ...], device=core.device), act_t, core
        )
        role = int(rb[0].item())
        role_hist[role] += 1
        role_timeline.append(role)
        if prev_role is not None and role != prev_role:
            role_switches += 1
        prev_role = role

        n_att = float(tel[1])
        n_def = float(tel[2])
        att_def_series.append((n_att, n_def))
        tel_sum += tel

        macros = act.reshape(-1)[0::2].astype(int)
        for m in macros.tolist():
            macro_counts[int(m)] += 1

        blue_on_home = core._is_on_home_side("blue", core.blue_x)[0].cpu().numpy()
        own_half += blue_on_home
        enemy_half += ~blue_on_home
        bx = core.blue_x[0].cpu().numpy()
        mid = float(core.cfg.map_cols) * 0.5
        pen = np.maximum(0.0, bx - mid)
        pen = np.where(~blue_on_home, pen, 0.0)
        max_penetration = np.maximum(max_penetration, pen)
        sum_penetration += pen

        carrying = core.blue_carrying[0].cpu().numpy().astype(bool)
        if carrying.any():
            carry_ticks += 1
            if not carry_started:
                carry_started = True
                carry_events += 1
        else:
            carry_started = False
        blue_score = int(core.blue_score[0].item())
        if blue_score > blue_score_prev:
            returns += blue_score - blue_score_prev
        blue_score_prev = blue_score

        if prev_mine_charges is not None and hasattr(core, "blue_mine_charges"):
            charges = core.blue_mine_charges[0].cpu().numpy()
            delta = charges - prev_mine_charges
            mine_grab_events += int(np.maximum(delta, 0).sum())
            mine_place_events += int(np.maximum(-delta, 0).sum())
            prev_mine_charges = charges.copy()

        if first_flag_pressure_tick is None:
            fx = core.red_flag_pos[0, 0].item()
            fy = core.red_flag_pos[0, 1].item()
            by = core.blue_y[0].cpu().numpy()
            if (np.sqrt((bx - fx) ** 2 + (by - fy) ** 2) < 6.0).any():
                first_flag_pressure_tick = t

        n_ticks += 1

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

    ticks = max(1, n_ticks)
    att_arr = np.asarray(att_def_series, dtype=np.float64) if att_def_series else np.zeros((0, 2))
    role_occ = (role_hist / ticks).tolist()
    return {
        "n_ticks": n_ticks,
        "terminal": list(terminal),
        "win": int(terminal[0] > terminal[1]),
        "blue": terminal[0],
        "red": terminal[1],
        "own_half_frac_per_agent": (own_half / ticks).tolist(),
        "enemy_half_frac_per_agent": (enemy_half / ticks).tolist(),
        "mean_own_half_frac": float(own_half.mean() / ticks),
        "mean_enemy_half_frac": float(enemy_half.mean() / ticks),
        "max_penetration_per_agent": max_penetration.tolist(),
        "mean_penetration_per_agent": (sum_penetration / ticks).tolist(),
        "max_penetration": float(max_penetration.max()),
        "mean_penetration": float(sum_penetration.mean() / ticks),
        "mean_num_attackers": float(att_arr[:, 0].mean()) if len(att_arr) else 0.0,
        "mean_num_defenders": float(att_arr[:, 1].mean()) if len(att_arr) else 0.0,
        "attacker_defender_split_mean": [
            float(att_arr[:, 0].mean()) if len(att_arr) else 0.0,
            float(att_arr[:, 1].mean()) if len(att_arr) else 0.0,
        ],
        "role_occupancy": {ROLE4_NAMES[i]: role_occ[i] for i in range(N_ROLE_BUCKET_MI)},
        "role_switches": int(role_switches),
        "role_timeline": role_timeline,
        "telemetry_mean": {
            BEHAVIOR_TELEMETRY_NAMES[i]: float(tel_sum[i] / ticks)
            for i in range(len(BEHAVIOR_TELEMETRY_NAMES))
        },
        "macro_counts": {MACRO_NAMES.get(k, str(k)): int(v) for k, v in sorted(macro_counts.items())},
        "macro_frac": {
            MACRO_NAMES.get(k, str(k)): float(v) / max(1, sum(macro_counts.values()))
            for k, v in sorted(macro_counts.items())
        },
        "carry_ticks": int(carry_ticks),
        "carry_tick_frac": float(carry_ticks) / ticks,
        "carry_events": int(carry_events),
        "flag_returns": int(returns),
        "mine_grab_events": int(mine_grab_events),
        "mine_place_events": int(mine_place_events),
        "first_flag_pressure_tick": first_flag_pressure_tick,
    }


def first_meaningful_divergence(ep_a: dict, ep_b: dict) -> dict:
    ta = ep_a["role_timeline"]
    tb = ep_b["role_timeline"]
    n = min(len(ta), len(tb))
    first_role = None
    for t in range(n):
        if ta[t] != tb[t]:
            first_role = t
            break
    return {
        "first_role_bucket_divergence_tick": first_role,
        "diverged_within_overlap": first_role is not None,
        "overlap_ticks": n,
        "role_hamming_frac": (
            float(sum(1 for t in range(n) if ta[t] != tb[t])) / n if n else None
        ),
        "same_dominant_role": (
            max(ep_a["role_occupancy"], key=ep_a["role_occupancy"].get)
            == max(ep_b["role_occupancy"], key=ep_b["role_occupancy"].get)
        ),
    }


def _agg_cell(eps: list[dict]) -> dict:
    if not eps:
        return {}

    def mean_key(k):
        vals = [e[k] for e in eps if e.get(k) is not None]
        return float(np.mean(vals)) if vals else None

    role_keys = list(ROLE4_NAMES.values())
    role_means = {
        rk: float(np.mean([e["role_occupancy"][rk] for e in eps])) for rk in role_keys
    }
    macro_keys = sorted({m for e in eps for m in e["macro_frac"]})
    macro_means = {
        m: float(np.mean([e["macro_frac"].get(m, 0.0) for e in eps])) for m in macro_keys
    }
    return {
        "n_episodes": len(eps),
        "win_rate": mean_key("win"),
        "mean_num_attackers": mean_key("mean_num_attackers"),
        "mean_num_defenders": mean_key("mean_num_defenders"),
        "mean_own_half_frac": mean_key("mean_own_half_frac"),
        "mean_enemy_half_frac": mean_key("mean_enemy_half_frac"),
        "mean_penetration": mean_key("mean_penetration"),
        "max_penetration_mean": mean_key("max_penetration"),
        "mean_role_switches": mean_key("role_switches"),
        "mean_carry_tick_frac": mean_key("carry_tick_frac"),
        "mean_carry_events": mean_key("carry_events"),
        "mean_flag_returns": mean_key("flag_returns"),
        "mean_mine_grab_events": mean_key("mine_grab_events"),
        "mean_mine_place_events": mean_key("mine_place_events"),
        "mean_first_flag_pressure_tick": mean_key("first_flag_pressure_tick"),
        "role_occupancy_mean": role_means,
        "dominant_role": max(role_means, key=role_means.get),
        "macro_frac_mean": macro_means,
        "telemetry_mean": {
            name: float(np.mean([e["telemetry_mean"][name] for e in eps]))
            for name in BEHAVIOR_TELEMETRY_NAMES
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pi-a-path", default=DEFAULT_PI_A)
    ap.add_argument("--pi-b-path", default=DEFAULT_PI_B)
    ap.add_argument("--pole-b-genome-json", default=DEFAULT_POLE_B)
    ap.add_argument("--sealed-csv", default=DEFAULT_SEALED_CSV)
    ap.add_argument("--seed-base", type=int, default=16_700_001)
    ap.add_argument("--n-seeds", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--max-steps", type=int, default=300)
    args = ap.parse_args()

    import experiments.r2_learned_crossover as R2
    from experiments.opponent_spec import (
        assert_live_opponent_batch,
        install_keyed_opponent_overlays,
    )
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    R2.AGENTS = N_AGENTS
    device = args.device
    sealed_csv = Path(args.sealed_csv)
    out_path = Path(args.out)
    genomes = _pole_genomes(
        N_AGENTS, Path(args.pole_b_genome_json) if args.pole_b_genome_json else None
    )
    pole_b_source = genomes.pop("_pole_b_source")

    probe = R2.build_env(device, int(args.seed_base))
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()

    policies = {
        "pi_A": load_custom_ppo_policy(str(args.pi_a_path), obs_space, act_space, device=device),
        "pi_B": load_custom_ppo_policy(str(args.pi_b_path), obs_space, act_space, device=device),
    }

    seeds = [int(args.seed_base) + i for i in range(int(args.n_seeds))]
    print(f"4V4 B3 BLUE-SIDE BEHAVIOR CAPTURE  {_now()}")
    print("  DIAGNOSTIC_NOT_A_GATE. Reuses sealed seeds; spends nothing new.")
    print(f"  seeds  {seeds[0]}..{seeds[-1]}  (n={len(seeds)})")
    print(f"  pole_B {pole_b_source}")
    print(f"  out    {out_path}\n", flush=True)

    episodes: dict[str, dict[str, dict[int, dict]]] = {
        "pi_A": {"A": {}, "B": {}},
        "pi_B": {"A": {}, "B": {}},
    }
    fidelity_failures = []

    from experiments.tqdm_loop import set_postfix, tqdm_iter

    for pole in ("A", "B"):
        bar = tqdm_iter(
            [(seed, name) for seed in seeds for name in ("pi_A", "pi_B")],
            desc=f"blue-side capture Pole{pole}",
            unit="ep",
        )
        for seed, name in bar:
            set_postfix(bar, f"{name} seed={seed}")
            env = R2.build_env(device, seed)
            core = env.core
            try:
                policies[name].reset_strategy()
                core._bt_profile_override = None
                core._sds_opening_hold_steps = 0
                gmap = genomes[pole]
                install_keyed_opponent_overlays(core, gmap)
                key = BASE_KEY[pole]
                env.env_method("set_phase", phase_from_tag(key))
                env.env_method("set_next_opponent", "SCRIPTED", key)
                initial_obs = env.reset()
                initial_obs["global_state"] = env.state()
                assert_live_opponent_batch(
                    core, gmap, allowed_keys=(key,),
                    context=f"blue-side capture {name}@{pole} seed {seed}",
                )
                ep = capture_episode(
                    env, core, policies[name], args.max_steps, obs=initial_obs
                )
                expected = _sealed_terminal(sealed_csv, seed, name, pole)
                got = tuple(ep["terminal"])
                if got != expected:
                    fidelity_failures.append(
                        {
                            "policy": name,
                            "pole": pole,
                            "seed": seed,
                            "expected": list(expected),
                            "got": list(got),
                        }
                    )
                    print(
                        f"  FIDELITY MISMATCH {name}@{pole} seed={seed} "
                        f"expected={expected} got={got}",
                        flush=True,
                    )
                episodes[name][pole][seed] = ep
            finally:
                env.close()
        print(f"  Pole {pole}: {len(seeds)} seeds x 2 policies done", flush=True)

    summary = {}
    for name in ("pi_A", "pi_B"):
        for pole in ("A", "B"):
            eps = [episodes[name][pole][s] for s in seeds]
            summary[f"{name}@{pole}"] = _agg_cell(eps)

    divergence = {}
    for pole in ("A", "B"):
        per_seed = []
        for seed in seeds:
            d = first_meaningful_divergence(
                episodes["pi_A"][pole][seed], episodes["pi_B"][pole][seed]
            )
            d["seed"] = seed
            per_seed.append(d)
        ticks = [
            d["first_role_bucket_divergence_tick"]
            for d in per_seed
            if d["first_role_bucket_divergence_tick"] is not None
        ]
        divergence[f"pi_A_vs_pi_B@pole_{pole}"] = {
            "n_seeds": len(per_seed),
            "frac_diverged": float(np.mean([d["diverged_within_overlap"] for d in per_seed])),
            "mean_first_role_divergence_tick": float(np.mean(ticks)) if ticks else None,
            "median_first_role_divergence_tick": float(np.median(ticks)) if ticks else None,
            "mean_role_hamming_frac": float(
                np.mean(
                    [
                        d["role_hamming_frac"]
                        for d in per_seed
                        if d["role_hamming_frac"] is not None
                    ]
                )
            ),
            "frac_same_dominant_role": float(
                np.mean([d["same_dominant_role"] for d in per_seed])
            ),
            "per_seed": per_seed,
        }

    contrast = {}
    for pole in ("A", "B"):
        a = summary[f"pi_A@{pole}"]
        b = summary[f"pi_B@{pole}"]
        contrast[pole] = {
            "delta_mean_num_attackers": a["mean_num_attackers"] - b["mean_num_attackers"],
            "delta_mean_num_defenders": a["mean_num_defenders"] - b["mean_num_defenders"],
            "delta_mean_enemy_half_frac": a["mean_enemy_half_frac"] - b["mean_enemy_half_frac"],
            "delta_mean_penetration": a["mean_penetration"] - b["mean_penetration"],
            "delta_role_switches": a["mean_role_switches"] - b["mean_role_switches"],
            "dominant_role_pi_A": a["dominant_role"],
            "dominant_role_pi_B": b["dominant_role"],
            "same_dominant_role": a["dominant_role"] == b["dominant_role"],
            "role_occupancy_L1": float(
                sum(
                    abs(a["role_occupancy_mean"][k] - b["role_occupancy_mean"][k])
                    for k in ROLE4_NAMES.values()
                )
            ),
        }

    raw_compact = {
        name: {
            pole: {
                str(seed): {k: v for k, v in ep.items() if k != "role_timeline"}
                for seed, ep in episodes[name][pole].items()
            }
            for pole in ("A", "B")
        }
        for name in ("pi_A", "pi_B")
    }

    payload = {
        "record": "4V4_B3_BLUE_SIDE_BEHAVIOR_CAPTURE",
        "status": "DIAGNOSTIC_NOT_A_GATE",
        "utc": _now(),
        "implements": "4V4_B3_BLUE_SIDE_BEHAVIOR_CAPTURE_PROTOCOL.json",
        "question": (
            "Are pi_A3 and pi_B3 strategically different over full trajectories, "
            "or are their per-state JSD differences washing out into the same "
            "closed-loop role pattern?"
        ),
        "seed_base": int(args.seed_base),
        "n_seeds": int(args.n_seeds),
        "seeds": seeds,
        "seed_policy": "REUSE sealed confirmatory block; no new experimental seed budget",
        "pole_b_source": pole_b_source,
        "checkpoints": {"pi_A": str(args.pi_a_path), "pi_B": str(args.pi_b_path)},
        "sealed_csv": str(sealed_csv),
        "fidelity_failures": fidelity_failures,
        "fidelity_ok": len(fidelity_failures) == 0,
        "summary": summary,
        "matched_divergence": divergence,
        "contrast_pi_A_minus_pi_B": contrast,
        "raw_per_episode": raw_compact,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print("\n  SUMMARY")
    for k, v in summary.items():
        print(
            f"    {k}: attackers={v['mean_num_attackers']:.2f} "
            f"defenders={v['mean_num_defenders']:.2f} "
            f"enemy_half={v['mean_enemy_half_frac']:.3f} "
            f"pen={v['mean_penetration']:.2f} "
            f"dom={v['dominant_role']} "
            f"switches={v['mean_role_switches']:.1f}"
        )
    print("\n  MATCHED DIVERGENCE")
    for k, v in divergence.items():
        print(
            f"    {k}: frac_diverged={v['frac_diverged']:.2f} "
            f"mean_first_tick={v['mean_first_role_divergence_tick']} "
            f"hamming={v['mean_role_hamming_frac']:.3f} "
            f"same_dom={v['frac_same_dominant_role']:.2f}"
        )
    print(f"\n  fidelity_ok={payload['fidelity_ok']} failures={len(fidelity_failures)}")
    print(f"  -> {out_path}")
    return 0 if payload["fidelity_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
