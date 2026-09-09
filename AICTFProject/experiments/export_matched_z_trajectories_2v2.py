"""Export matched z0/z1 trajectories for the sealed 2v2 Rung-1 qualitative seed.

Replays seed 11960003 under forced z for poles A and B against the sealed
Share-Encoder checkpoint, verifies terminal scores against rung1_ladder_eval_rows.csv,
and writes per-tick blue/red positions + actions so the paper can plot:

  same initial state -> overlay paths -> mark first decision / position divergence.

Outputs under artifacts/qualitative_capture/2v2_rung1_trajectory_strip/:
  trajectories.json   (all cells + divergence summary)
  *.npz.csv        (optional per-cell traces)

Run (from AICTFProject, CUDA preferred to match sealed fidelity):
  python experiments/export_matched_z_trajectories_2v2.py --device cuda
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

SEED = 11960003
CKPT = ROOT / "artifacts/strategic_demand/sppo/sharing_ladder/rung1/ckpts/final_rung1.pt"
SEALED_CSV = ROOT / "artifacts/strategic_demand/sppo/rung1_ladder_eval_rows.csv"
OUT = ROOT / "artifacts/qualitative_capture/2v2_rung1_trajectory_strip"

CELLS = [
    {"z": 0, "pole": "A"},
    {"z": 1, "pole": "A"},
    {"z": 0, "pole": "B"},
    {"z": 1, "pole": "B"},
]


def _sealed_terminal(z: int, pole: str) -> tuple[int, int]:
    with SEALED_CSV.open(encoding="utf-8") as fh:
        rows = [
            r for r in csv.DictReader(fh)
            if int(r["seed"]) == SEED and r["pole"] == pole and r["z"] == f"z{z}"
        ]
    if len(rows) != 1:
        raise SystemExit(f"REFUSING: expected 1 sealed row for z{z}/pole{pole}/seed{SEED}, got {len(rows)}")
    return int(rows[0]["blue"]), int(rows[0]["red"])


def _replay_cell(z: int, pole: str, device: str) -> dict:
    import experiments.r2_learned_crossover as R2
    from experiments.opponent_spec import (
        assert_live_opponent_batch,
        install_keyed_opponent_overlays,
        pole_A_genome,
    )
    from rl.curriculum import phase_from_tag
    from rl import ladder_rung1 as L1

    R2.AGENTS = 2
    env = R2.build_env(device, SEED)
    core = env.core
    # Match experiments/replay_capture_qualitative.py: Pole A installs OP6 genome;
    # 2v2 Pole B uses default OP7 scripted path with empty overlay.
    genomes = {"OP6": pole_A_genome(2)} if pole == "A" else {}

    obs_space, act_space = env.observation_space, env.action_space
    model, branch_cfg, _ = L1.load_rung(1, str(CKPT), obs_space, act_space, device=device)
    policy = L1.make_dispatch_policy(model, branch_cfg, device=device)
    policy.fixed_latent_strategy = True
    policy.fixed_latent_strategy_id = int(z)
    policy.reset_strategy()

    core._bt_profile_override = None
    core._sds_opening_hold_steps = 0
    install_keyed_opponent_overlays(core, genomes)
    key = "OP6" if pole == "A" else "OP7"
    env.env_method("set_phase", phase_from_tag(key))
    env.env_method("set_next_opponent", "SCRIPTED", key)
    obs = env.reset()
    obs["global_state"] = env.state()
    if genomes:
        assert_live_opponent_batch(core, genomes, allowed_keys=(key,), context=f"traj z{z} pole{pole}")

    ticks = []
    terminal = None
    for t in range(300):
        bx = core.blue_x[0].detach().cpu().numpy().astype(float).tolist()
        by = core.blue_y[0].detach().cpu().numpy().astype(float).tolist()
        rx = core.red_x[0].detach().cpu().numpy().astype(float).tolist()
        ry = core.red_y[0].detach().cpu().numpy().astype(float).tolist()
        balive = core.blue_alive[0].detach().cpu().numpy().astype(int).tolist()
        ralive = core.red_alive[0].detach().cpu().numpy().astype(int).tolist()
        bcarry = core.blue_carrying[0].detach().cpu().numpy().astype(int).tolist()
        action, _ = policy.predict(obs, deterministic=True)
        a = np.asarray(action).reshape(-1).astype(int).tolist()
        ticks.append({
            "tick": t,
            "blue_x": bx, "blue_y": by, "blue_alive": balive, "blue_carrying": bcarry,
            "red_x": rx, "red_y": ry, "red_alive": ralive,
            "action": a,
            "blue_score": int(core.blue_score[0]),
            "red_score": int(core.red_score[0]),
        })
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
    env.close()

    expected = _sealed_terminal(z, pole)
    match = terminal == expected
    return {
        "z": z,
        "pole": pole,
        "seed": SEED,
        "terminal": {"blue": terminal[0], "red": terminal[1]},
        "sealed_terminal": {"blue": expected[0], "red": expected[1]},
        "fidelity": "MATCH" if match else "MISMATCH",
        "n_ticks": len(ticks),
        "ticks": ticks,
    }


def _first_divergence(a_ticks: list[dict], b_ticks: list[dict], eps: float = 1e-4) -> dict:
    n = min(len(a_ticks), len(b_ticks))
    first_action = None
    first_pos = None
    for t in range(n):
        if first_action is None and a_ticks[t]["action"] != b_ticks[t]["action"]:
            first_action = t
        if first_pos is None:
            ax = np.asarray(a_ticks[t]["blue_x"]) - np.asarray(b_ticks[t]["blue_x"])
            ay = np.asarray(a_ticks[t]["blue_y"]) - np.asarray(b_ticks[t]["blue_y"])
            if float(np.max(np.abs(ax))) > eps or float(np.max(np.abs(ay))) > eps:
                first_pos = t
        if first_action is not None and first_pos is not None:
            break
    return {
        "first_action_divergence_tick": first_action,
        "first_position_divergence_tick": first_pos,
        "mark_tick": first_action if first_action is not None else first_pos,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not CKPT.exists():
        raise SystemExit(f"REFUSING: missing checkpoint {CKPT}")

    OUT.mkdir(parents=True, exist_ok=True)
    cells = []
    for c in CELLS:
        print(f"replaying z{c['z']} pole{c['pole']} ...")
        cell = _replay_cell(c["z"], c["pole"], args.device)
        print(f"  terminal={cell['terminal']} fidelity={cell['fidelity']} n={cell['n_ticks']}")
        if cell["fidelity"] != "MATCH":
            raise SystemExit("REFUSING: fidelity MISMATCH -- do not build figure from this export")
        # write flat csv
        csv_path = OUT / f"z{c['z']}_pole{c['pole']}_seed{SEED}_traj.csv"
        with csv_path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow([
                "tick", "b0_x", "b0_y", "b1_x", "b1_y",
                "r0_x", "r0_y", "r1_x", "r1_y",
                "a0_macro", "a0_tgt", "a1_macro", "a1_tgt",
                "blue_score", "red_score",
            ])
            for row in cell["ticks"]:
                a = row["action"] + [0] * (4 - len(row["action"]))
                w.writerow([
                    row["tick"],
                    row["blue_x"][0], row["blue_y"][0], row["blue_x"][1], row["blue_y"][1],
                    row["red_x"][0], row["red_y"][0], row["red_x"][1], row["red_y"][1],
                    a[0], a[1], a[2], a[3],
                    row["blue_score"], row["red_score"],
                ])
        cells.append(cell)

    by_pole = {}
    for pole in ("A", "B"):
        z0 = next(c for c in cells if c["pole"] == pole and c["z"] == 0)
        z1 = next(c for c in cells if c["pole"] == pole and c["z"] == 1)
        by_pole[pole] = _first_divergence(z0["ticks"], z1["ticks"])

    payload = {
        "record": "Matched z0/z1 trajectory strip source (2v2 Rung-1 Share-Encoder)",
        "seed": SEED,
        "checkpoint": str(CKPT.relative_to(ROOT)).replace("\\", "/"),
        "device": args.device,
        "fidelity": "ALL_MATCH",
        "divergence_by_pole": by_pole,
        "cells": [
            {k: v for k, v in c.items() if k != "ticks"} | {
                "ticks_file": f"z{c['z']}_pole{c['pole']}_seed{SEED}_traj.csv",
                "ticks": c["ticks"],
            }
            for c in cells
        ],
    }
    out_json = OUT / "trajectories.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_json), "divergence_by_pole": by_pole}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
