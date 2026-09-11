"""Collect frozen GUARD/BREACH role-target statistics for 4v4 B3 role preservation.

Implements 4V4_B3_ROLE_PRESERVATION_SPEC.json#SINGLE_AXIS_INTERVENTION.target_source.

Uses ONLY the existing role vocabulary:
  - rl.behavior_telemetry.role_bucket_detailed_id (7-d 4v4 occupancy)
  - BEHAVIOR_TELEMETRY num_attackers / num_defenders / attack_defense_ratio
  - MacroAction GET_FLAG / GO_HOME

Seeds reuse the already-spent disagreement-collection prefix 16810001+.
DIAGNOSTIC artifact: not a gate.
"""
from __future__ import annotations

import argparse
import json
import sys
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
STYLES = {"GUARD": "BLUE_ONE_DEFENDER_V2", "BREACH": "BLUE_BOTH_ATTACK_V2"}
ROLE4_NAMES = {
    0: "all_push",
    1: "three_attack_one_defend",
    2: "two_attack_two_defend",
    3: "one_attack_three_defend",
    4: "escort_pair",
    5: "intercept_pair",
    6: "turtle_defense",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_scripted_episode(env, core, style: str, max_steps: int) -> dict:
    """Collect role stats under a scripted blue style.

    Per STYLE_INTENT_PROJECTION_AMENDMENT: project telemetry macros from the
    scripted style's own allocation intent (not half-plane occupancy):
      GUARD  → last ceil(N/2) agents GO_HOME, others GET_FLAG
      BREACH → all alive agents GET_FLAG
    """
    core.blue_scripted = True
    core.set_blue_style(style)
    obs = env.reset()
    core.blue_scripted = True
    core.set_blue_style(style)
    obs["global_state"] = env.state()

    role_hist = np.zeros(N_ROLE_BUCKET_MI, dtype=np.float64)
    feat_sum = np.zeros(3, dtype=np.float64)
    n_ticks = 0
    n_def_guard = (N_AGENTS + 1) // 2
    guard_def_lo = N_AGENTS - n_def_guard

    for _t in range(max_steps):
        action = env.action_space.sample() * 0
        alive = core.blue_alive[0]
        macros = torch.full(
            (N_AGENTS,), int(MacroAction.GO_TO), device=core.device, dtype=torch.long
        )
        if style == "BLUE_BOTH_ATTACK_V2":
            macros = torch.where(
                alive,
                torch.full_like(macros, int(MacroAction.GET_FLAG)),
                macros,
            )
        elif style == "BLUE_ONE_DEFENDER_V2":
            att = torch.full_like(macros, int(MacroAction.GET_FLAG))
            deff = torch.full_like(macros, int(MacroAction.GO_HOME))
            idx = torch.arange(N_AGENTS, device=core.device)
            intended = torch.where(idx >= guard_def_lo, deff, att)
            macros = torch.where(alive, intended, macros)
        else:
            raise RuntimeError(f"unsupported style for target collection: {style}")

        flat = torch.zeros(N_AGENTS * 2, device=core.device, dtype=torch.long)
        flat[0::2] = macros
        act_t = flat.unsqueeze(0)
        tel = compute_behavior_telemetry_batch(core, act_t)
        _sb, rb, _pb, _adb = bucket_ids_from_telemetry(tel, act_t, core)
        role_hist[int(rb[0].item())] += 1
        tel_np = tel[0].detach().cpu().numpy()
        feat_sum += np.array([tel_np[1], tel_np[2], tel_np[12]], dtype=np.float64)
        n_ticks += 1

        env.step_async(action)
        obs, _r, done, _info = env.step_wait()
        obs["global_state"] = env.state()
        if bool(np.asarray(done).any()):
            break

    ticks = max(1, n_ticks)
    return {
        "n_ticks": n_ticks,
        "role_occupancy": (role_hist / ticks).tolist(),
        "mean_num_attackers": float(feat_sum[0] / ticks),
        "mean_num_defenders": float(feat_sum[1] / ticks),
        "mean_attack_defense_ratio": float(feat_sum[2] / ticks),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-size", type=int, default=4)
    ap.add_argument("--seed-base", type=int, default=16_810_001)
    ap.add_argument("--n-episodes-per-style-per-pole", type=int, default=16)
    ap.add_argument("--pole-b-genome-json", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-steps", type=int, default=300)
    args = ap.parse_args()
    if int(args.team_size) != N_AGENTS:
        raise SystemExit("this collector is frozen for team-size 4")

    import experiments.r2_learned_crossover as R2
    from experiments.opponent_spec import (
        _with_full_team_defender_gate,
        assert_live_opponent_batch,
        install_keyed_opponent_overlays,
        pole_A_genome,
    )
    from experiments.sds_genome import SDSGenome
    from rl.curriculum import phase_from_tag

    R2.AGENTS = N_AGENTS
    pole_b = _with_full_team_defender_gate(
        SDSGenome.from_dict(
            json.loads(Path(args.pole_b_genome_json).read_text(encoding="utf-8"))
        ),
        N_AGENTS,
    )
    genomes = {
        "A": {"OP6": pole_A_genome(N_AGENTS)},
        "B": {"OP7": pole_b},
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = int(args.n_episodes_per_style_per_pole)
    print(f"4V4 B3 ROLE TARGET COLLECTION  {_now()}")
    print(f"  styles GUARD={STYLES['GUARD']}  BREACH={STYLES['BREACH']}")
    print(f"  {n} episodes/style/pole on seeds {args.seed_base}+ (reused spent block)")
    print(f"  pole_B {pole_b.genome_id}\n", flush=True)

    from experiments.tqdm_loop import set_postfix, tqdm_iter

    raw = {style: {"A": [], "B": []} for style in STYLES}
    seed = int(args.seed_base)
    cells = [(style_name, style_id, pole)
             for style_name, style_id in STYLES.items()
             for pole in ("A", "B")
             for _i in range(n)]
    bar = tqdm_iter(cells, desc="role target collect", unit="ep")
    for style_name, style_id, pole in bar:
        set_postfix(bar, f"{style_name}@{pole} seed={seed}")
        env = R2.build_env(args.device, seed)
        core = env.core
        try:
            core._bt_profile_override = None
            core._sds_opening_hold_steps = 0
            install_keyed_opponent_overlays(core, genomes[pole])
            key = BASE_KEY[pole]
            env.env_method("set_phase", phase_from_tag(key))
            env.env_method("set_next_opponent", "SCRIPTED", key)
            _ = env.reset()
            assert_live_opponent_batch(
                core, genomes[pole], allowed_keys=(key,),
                context=f"role-target {style_name}@{pole} seed {seed}",
            )
            ep = _run_scripted_episode(env, core, style_id, args.max_steps)
            ep["seed"] = seed
            ep["pole"] = pole
            raw[style_name][pole].append(ep)
        finally:
            env.close()
        seed += 1
        if len(raw[style_name][pole]) == n:
            print(
                f"  {style_name}@{pole}: {len(raw[style_name][pole])} episodes",
                flush=True,
            )

    targets = {}
    for style_name in STYLES:
        eps = raw[style_name]["A"] + raw[style_name]["B"]
        occ = np.mean([e["role_occupancy"] for e in eps], axis=0)
        targets[style_name] = {
            "style_id": STYLES[style_name],
            "n_episodes": len(eps),
            "role_occupancy": {
                ROLE4_NAMES[i]: float(occ[i]) for i in range(N_ROLE_BUCKET_MI)
            },
            "role_occupancy_vector": occ.tolist(),
            "features": {
                "mean_num_attackers": float(np.mean([e["mean_num_attackers"] for e in eps])),
                "mean_num_defenders": float(np.mean([e["mean_num_defenders"] for e in eps])),
                "mean_attack_defense_ratio": float(
                    np.mean([e["mean_attack_defense_ratio"] for e in eps])
                ),
            },
            "dominant_role": ROLE4_NAMES[int(np.argmax(occ))],
        }

    payload = {
        "record": "4V4_B3_ROLE_TARGETS_GUARD_BREACH",
        "status": "FROZEN_TARGETS",
        "utc": _now(),
        "implements": "4V4_B3_ROLE_PRESERVATION_SPEC.json#target_source",
        "vocabulary": {
            "role_buckets": ROLE4_NAMES,
            "macros": {
                "attack": MacroAction.GET_FLAG.name,
                "defend": MacroAction.GO_HOME.name,
            },
            "telemetry": list(BEHAVIOR_TELEMETRY_NAMES),
        },
        "seed_base": int(args.seed_base),
        "n_episodes_per_style_per_pole": n,
        "pole_b_genome_id": pole_b.genome_id,
        "targets": targets,
        "raw_per_episode": raw,
    }
    out_path = out_dir / "role_targets.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("\n  FROZEN TARGETS")
    for style_name, t in targets.items():
        print(
            f"    {style_name}: dom={t['dominant_role']} "
            f"att={t['features']['mean_num_attackers']:.2f} "
            f"def={t['features']['mean_num_defenders']:.2f} "
            f"adr={t['features']['mean_attack_defense_ratio']:.3f}"
        )
        print(f"      occ={{{', '.join(f'{k}:{v:.2f}' for k,v in t['role_occupancy'].items() if v>0.05)}}}")
    print(f"\n  -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
