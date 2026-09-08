"""Deterministic-replay qualitative capture: pairs an already-measured seed's exact frames
and behavioral tags with the quantitative result already on record for it.

Because the underlying policy is deterministic and the env is seeded, replaying (checkpoint,
pole, seed) exactly reproduces the sealed episode the win-rate numbers were computed from --
this is NOT a re-creation or approximation of the test condition, it IS that test condition,
verified by checking the replayed terminal score against the sealed CSV row before trusting
anything captured.

Captures, per episode:
  - RGB frames (core.render_rgb_array, already used nowhere else -- no simulator code touched)
    at a fixed tick interval PLUS at each detected "event" tick (first flag pressure, first
    counter-capture trigger, terminal tick)
  - a per-tick behavioral tag log: red role counts (defender/attacker/counter/interceptor),
    own-half occupancy, flag-carrier identity + nearest-teammate distance, blue-flag pressure

Run examples:
  python experiments/replay_capture_qualitative.py --mode specialist \\
      --pi-a-path <ckpt> --pi-b-path <ckpt> --policy pi_A --pole A --seed 16400001 \\
      --team-size 4 --sealed-csv <rows.csv> --out-dir <dir>
  python experiments/replay_capture_qualitative.py --mode scripted \\
      --style GUARD --genome-json <path> --pole A --seed 99995001 --team-size 4 --out-dir <dir>
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROLE_NAMES = {0: "ATTACKER", 1: "DEFENDER", 2: "ESCORT", 3: "INTERCEPTOR",
              4: "FLAG_RETR", 5: "COUNTER", 6: "2V1_WING"}
FRAME_EVERY = 10


def _save_png(frame: np.ndarray, path: Path) -> None:
    from PIL import Image
    Image.fromarray(frame).save(str(path))


def _sealed_terminal(csv_path: str, seed: int, policy: str | None, pole: str) -> tuple[int, int]:
    """FAIL CLOSED rather than silently matching the wrong policy's row.

    A CSV with a 'policy' column (e.g. the specialist crossover eval's rows) can have MULTIPLE
    rows sharing the same (seed, pole) -- one per policy. Matching on (seed, pole) alone, as an
    earlier version of this function did, silently compares against whichever policy's row
    happens to appear first in the file, which is wrong whenever that is not the policy being
    replayed. If the matches are ambiguous, --policy-label is REQUIRED.
    """
    with open(csv_path, encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if int(r["seed"]) == seed and r.get("pole") == pole]
    if not rows:
        raise SystemExit(f"FIDELITY CHECK: no row found in {csv_path} for seed={seed} pole={pole}")
    distinct_policies = {r.get("policy") for r in rows}
    if len(rows) > 1 and len(distinct_policies) > 1:
        if policy is None:
            raise SystemExit(
                f"FIDELITY CHECK: {len(rows)} rows match seed={seed} pole={pole} across "
                f"policies {sorted(distinct_policies)} -- --policy-label is REQUIRED to "
                f"disambiguate which one this replay corresponds to.")
        rows = [r for r in rows if r.get("policy") == policy]
        if not rows:
            raise SystemExit(f"FIDELITY CHECK: no row with policy={policy!r} for seed={seed} "
                             f"pole={pole}; available policies: {sorted(distinct_policies)}")
    return int(rows[0]["blue"]), int(rows[0]["red"])


def capture(env, core, policy_predict, out_dir: Path, tag: str, n_agents: int, initial_obs=None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(exist_ok=True)

    tick_log = []
    saved_frames = []
    first_flag_pressure_tick = None
    first_counter_tick = None
    terminal = None
    if initial_obs is not None:
        obs = initial_obs  # already reset + asserted by the caller; resetting again would
                           # advance the RNG and break bit-exact replay of the sealed seed
    else:
        obs = env.reset()
        obs["global_state"] = env.state()

    for t in range(300):
        roles = core.bt_red_role[0].tolist() if hasattr(core, "bt_red_role") else []
        role_counts = {ROLE_NAMES.get(r, r): roles.count(r) for r in set(roles)}
        carrying = core.blue_carrying[0].cpu().numpy() if hasattr(core, "blue_carrying") else np.zeros(n_agents, dtype=bool)
        carrier_dist = None
        if carrying.any():
            bx, by = core.blue_x[0].cpu().numpy(), core.blue_y[0].cpu().numpy()
            ci = int(np.argmax(carrying))
            others = [j for j in range(n_agents) if j != ci]
            if others:
                carrier_dist = min(float(np.sqrt((bx[ci]-bx[j])**2 + (by[ci]-by[j])**2)) for j in others)
        bx, by = core.blue_x[0].cpu().numpy(), core.blue_y[0].cpu().numpy()
        fx, fy = core.red_flag_pos[0, 0].item(), core.red_flag_pos[0, 1].item()
        pressure = bool((np.sqrt((bx-fx)**2 + (by-fy)**2) < 6.0).any())
        if pressure and first_flag_pressure_tick is None:
            first_flag_pressure_tick = t
        if 5 in role_counts and first_counter_tick is None:
            first_counter_tick = t

        event_tick = (t % FRAME_EVERY == 0) or (t == first_flag_pressure_tick) or (t == first_counter_tick)
        row = {"tick": t, "red_role_counts": role_counts, "carrying": bool(carrying.any()),
               "carrier_nearest_teammate_dist": carrier_dist, "blue_flag_pressure": pressure}
        tick_log.append(row)
        if event_tick:
            frame = env.render(mode="rgb_array")
            fname = f"{tag}_t{t:03d}.png"
            _save_png(frame, frames_dir / fname)
            saved_frames.append(fname)

        action, _ = policy_predict(obs)
        env.step_async(action)
        obs, _r, done, info = env.step_wait()
        obs["global_state"] = env.state()
        if bool(np.asarray(done).any()):
            i0 = info[0] if isinstance(info, (list, tuple)) else info
            res = (i0 or {}).get("episode_result") or {}
            terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
            frame = env.render(mode="rgb_array")
            fname = f"{tag}_FINAL_t{t:03d}.png"
            _save_png(frame, frames_dir / fname)
            saved_frames.append(fname)
            break
    if terminal is None:
        terminal = (int(core.blue_score[0]), int(core.red_score[0]))

    with (out_dir / f"{tag}_tick_log.json").open("w", encoding="utf-8") as fh:
        json.dump(tick_log, fh, indent=2)

    return {"tag": tag, "terminal": terminal, "win": int(terminal[0] > terminal[1]),
            "first_flag_pressure_tick": first_flag_pressure_tick,
            "first_counter_tick": first_counter_tick, "n_ticks": len(tick_log),
            "frames": saved_frames, "tick_log_file": f"{tag}_tick_log.json"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("specialist", "latent", "scripted"), required=True)
    ap.add_argument("--team-size", type=int, required=True)
    ap.add_argument("--pole", choices=("A", "B"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--pi-path", default=None, help="specialist checkpoint (mode=specialist)")
    ap.add_argument("--rung-checkpoint", default=None, help="Rung-N .pt (mode=latent)")
    ap.add_argument("--rung", type=int, default=1)
    ap.add_argument("--z", type=int, default=None)
    ap.add_argument("--style", choices=("GUARD", "BREACH"), default=None, help="mode=scripted")
    ap.add_argument("--genome-json", default=None, help="candidate Pole-B genome JSON, else canonical")
    ap.add_argument("--sealed-csv", default=None, help="if given, verify replay matches this row")
    ap.add_argument("--policy-label", default=None,
                    help="the 'policy' column value to match in --sealed-csv (e.g. 'pi_A', "
                         "'pi_B'). REQUIRED whenever --sealed-csv has a policy column with "
                         "more than one distinct value for this (seed, pole) -- otherwise the "
                         "check silently compares against whichever row happens to come "
                         "first, which can be a DIFFERENT policy's episode entirely.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    N = int(args.team_size)
    import experiments.r2_learned_crossover as R2
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays,
        pole_A_genome, pole_B_genome, _with_full_team_defender_gate,
    )
    from rl.curriculum import phase_from_tag
    R2.AGENTS = N
    BASE_KEY = {"A": "OP6", "B": "OP7"}

    device = args.device
    env = R2.build_env(device, args.seed)
    core = env.core

    if args.mode == "scripted":
        import experiments.strategic_demand_searcher as S
        from experiments.sds_genome import SDSGenome, apply_genome_to_core
        S.AGENTS = N
        if args.genome_json:
            g = _with_full_team_defender_gate(SDSGenome.from_dict(json.loads(Path(args.genome_json).read_text())), N)
        else:
            g = pole_A_genome(N) if args.pole == "A" else pole_B_genome(N)
        opp = g.base_opponent
        env.env_method("set_phase", opp)
        env.env_method("set_next_opponent", "SCRIPTED", opp)
        apply_genome_to_core(core, g)
        core.blue_scripted = True
        core.set_blue_style(args.style)

        def predict(obs):
            return env.action_space.sample() * 0, None
    else:
        from rl.custom_ppo import load_custom_ppo_policy
        if args.pole == "A":
            genomes = {"OP6": pole_A_genome(N)}
        elif args.genome_json:
            from experiments.sds_genome import SDSGenome
            g = _with_full_team_defender_gate(SDSGenome.from_dict(json.loads(Path(args.genome_json).read_text())), N)
            genomes = {"OP7": g}
            print(f"  Pole B SOURCE  candidate genome {g.genome_id!r} from {args.genome_json} (NOT canonical)")
        else:
            genomes = {"OP7": pole_B_genome(N)} if N != 2 else {}
        obs_space, act_space = env.observation_space, env.action_space
        if args.mode == "specialist":
            policy = load_custom_ppo_policy(args.pi_path, obs_space, act_space, device=device)
        else:
            from rl import ladder_rung1 as L1
            model, branch_cfg, _ = L1.load_rung(args.rung, args.rung_checkpoint, obs_space, act_space, device=device)
            policy = L1.make_dispatch_policy(model, branch_cfg, device=device)
            policy.fixed_latent_strategy = True
            policy.fixed_latent_strategy_id = int(args.z)
        policy.reset_strategy()

        # Order matters (matches every other evaluator in this program): clear any stale
        # override FIRST, THEN install the real overlay, THEN reset, THEN assert on the LIVE
        # post-reset state. Installing before the clear would have the clear wipe it out again.
        core._bt_profile_override = None
        core._sds_opening_hold_steps = 0
        install_keyed_opponent_overlays(core, genomes)
        key = BASE_KEY[args.pole]
        env.env_method("set_phase", phase_from_tag(key))
        env.env_method("set_next_opponent", "SCRIPTED", key)
        initial_obs = env.reset()
        initial_obs["global_state"] = env.state()
        assert_live_opponent_batch(core, genomes, allowed_keys=(key,), context=f"qualitative capture seed {args.seed}")

        def predict(obs):
            a, _ = policy.predict(obs, deterministic=True)
            return a, None

    tag = args.tag or f"{args.mode}_pole{args.pole}_seed{args.seed}"
    out_dir = Path(args.out_dir)
    result = capture(env, core, predict, out_dir, tag, N,
                     initial_obs=initial_obs if args.mode != "scripted" else None)
    env.close()

    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))
    print(f"  {len(result['frames'])} frames -> {out_dir / 'frames'}")

    if args.sealed_csv:
        expected = _sealed_terminal(args.sealed_csv, args.seed, args.policy_label, args.pole)
        ok = tuple(result["terminal"]) == expected
        print(f"  FIDELITY CHECK vs {args.sealed_csv} (policy_label={args.policy_label!r}): "
              f"expected={expected} replayed={tuple(result['terminal'])} "
              f"{'MATCH' if ok else 'MISMATCH -- DO NOT TRUST THIS CAPTURE'}")
        if not ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
