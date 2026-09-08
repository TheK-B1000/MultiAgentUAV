"""What do z0 and z1 actually DO, behaviorally, against each pole?

Exploratory, not a gate. Answers a narrower and more useful question than any win-rate delta:
not "does z0 beat z1 on Pole A", but "what tactics does each latent actually deploy, and does
that tactic change depending on which opponent it's facing, or is it a fixed playstyle?"

Forces z in {0, 1} against Pole A and Pole B (4 conditions), runs N episodes per condition, and
averages rl.behavior_telemetry.compute_behavior_telemetry_batch's per-step feature vector over
every decision step of every episode. That function is not new or untested -- it's the exact
feature set the live training diagnostic printer already uses every update
(rl/custom_ppo/rollout/collector.py), reused here for a clean forced-z breakdown it doesn't
otherwise get (training never forces z1 against Pole A or z0 against Pole B; z0=Pole A/z1=Pole B
is fixed by the winner-directed routing, so this is the only way to see the OFF-diagonal cells
behaviorally).

Runs on CPU by default specifically so it does not contend with an in-flight GPU training run.

Run:  python experiments/characterize_z0_z1_behavior.py --checkpoint <path> --episodes 32
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--base-seed", type=int, default=11_950_001)
    args = ap.parse_args()

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
    )
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.behavior_telemetry import BEHAVIOR_TELEMETRY_NAMES, compute_behavior_telemetry_batch
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    device = args.device
    print(f"CHARACTERIZE z0/z1 BEHAVIOR  checkpoint={ckpt.name}  device={device}")
    print(f"  {args.episodes} episodes per (z, pole) cell, base_seed={args.base_seed}\n", flush=True)

    probe = R2.build_env(device, args.base_seed)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()

    policy = load_custom_ppo_policy(str(ckpt), obs_space, act_space, device=device)
    if not (policy.model.uses_latent_strategy and policy.model.latent_k == 2):
        raise SystemExit("REFUSING: checkpoint is not a K=2 latent policy")

    results: dict[tuple[int, str], list[np.ndarray]] = defaultdict(list)
    outcomes: dict[tuple[int, str], list[int]] = defaultdict(list)

    for z in (0, 1):
        for pole in ("A", "B"):
            for ep in range(args.episodes):
                seed = args.base_seed + z * 1000 + (0 if pole == "A" else 500) + ep
                env = R2.build_env(device, seed)
                core = env.core
                try:
                    policy.fixed_latent_strategy = True
                    policy.fixed_latent_strategy_id = z
                    policy.reset_strategy()
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
                                               context=f"characterize z{z} pole{pole} seed {seed}")
                    step_feats = []
                    terminal = None
                    for _ in range(R2.MAX_STEPS):
                        action, _ = policy.predict(obs, deterministic=True)
                        act_t = torch.as_tensor(action, dtype=torch.long, device=core.device)
                        if act_t.ndim == 1:
                            act_t = act_t.unsqueeze(0)
                        with torch.no_grad():
                            beh = compute_behavior_telemetry_batch(core, act_t)
                        step_feats.append(beh[0].cpu().numpy())
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
                    results[(z, pole)].append(np.mean(step_feats, axis=0) if step_feats
                                              else np.zeros(len(BEHAVIOR_TELEMETRY_NAMES)))
                    outcomes[(z, pole)].append(int(terminal[0] > terminal[1]))
                finally:
                    env.close()
            wr = float(np.mean(outcomes[(z, pole)]))
            print(f"  z{z} vs Pole {pole}: {args.episodes} episodes, win rate {wr:.3f}", flush=True)

    summary = {}
    for (z, pole), vecs in results.items():
        mean_vec = np.mean(vecs, axis=0)
        summary[f"z{z}_pole{pole}"] = {
            name: float(mean_vec[i]) for i, name in enumerate(BEHAVIOR_TELEMETRY_NAMES)
        }
        summary[f"z{z}_pole{pole}"]["win_rate"] = float(np.mean(outcomes[(z, pole)]))

    print("\n=== Per-condition behavior means ===")
    header = f"{'feature':32s}" + "".join(f"{k:>16s}" for k in summary)
    print(header)
    for name in BEHAVIOR_TELEMETRY_NAMES + ("win_rate",):
        row = f"{name:32s}" + "".join(f"{summary[k][name]:16.4f}" for k in summary)
        print(row)

    print("\n=== Does z's behavior change with the opponent, or is it a fixed playstyle? ===")
    for z in (0, 1):
        a, b = summary[f"z{z}_poleA"], summary[f"z{z}_poleB"]
        deltas = {name: a[name] - b[name] for name in BEHAVIOR_TELEMETRY_NAMES}
        biggest = sorted(deltas.items(), key=lambda kv: -abs(kv[1]))[:3]
        print(f"  z{z}: biggest A-vs-B behavior shifts: "
              + ", ".join(f"{n}={d:+.3f}" for n, d in biggest))

    print("\n=== Does z0 differ from z1 on the SAME opponent (the crossover-relevant question)? ===")
    for pole in ("A", "B"):
        z0, z1 = summary[f"z0_pole{pole}"], summary[f"z1_pole{pole}"]
        deltas = {name: z0[name] - z1[name] for name in BEHAVIOR_TELEMETRY_NAMES}
        biggest = sorted(deltas.items(), key=lambda kv: -abs(kv[1]))[:3]
        print(f"  Pole {pole}: biggest z0-vs-z1 behavior shifts: "
              + ", ".join(f"{n}={d:+.3f}" for n, d in biggest))

    out_path = ROOT / "artifacts" / "strategic_demand" / "sppo" / f"Z0_Z1_BEHAVIOR_CHARACTERIZATION_{ckpt.stem}.json"
    out_path.write_text(json.dumps({
        "record": "z0/z1 behavior characterization (exploratory, not a gate)",
        "checkpoint": str(ckpt), "episodes_per_cell": args.episodes,
        "base_seed": args.base_seed, "device": device,
        "feature_names": list(BEHAVIOR_TELEMETRY_NAMES),
        "per_condition_means": summary,
        "status": "EXPLORATORY_DIAGNOSTIC_NOT_A_GATE",
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
