"""Export matched A' vs pi_B trajectories + role telemetry for 4v4 scaffold PASS*.

Replays one sealed confirmatory seed under A' (pi_A + forced 2D) and native pi_B
on both certified poles; verifies terminals against SCAFFOLDED_A_CROSSOVER_BRIDGE
rows; also aggregates position-near-home role allocation over a fixed seed subset.

Outputs under artifacts/qualitative_capture/4v4_scaffold_trajectory_strip/:
  trajectories.json
  role_allocation.json
  *_traj.csv

Run (from AICTFProject, CUDA to match sealed fidelity):
  ./.venv/Scripts/python.exe experiments/export_matched_scaffold_trajectories_4v4.py --device cuda
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

import experiments.probe_learned_composition as P  # noqa: E402
import experiments.run_scaffolded_a_crossover_bridge_4v4 as S  # noqa: E402

SEED = 21_100_005  # sealed row: A' wins A / loses B; pi_B wins B
ROLE_SEEDS = list(range(21_100_001, 21_100_033))  # first 32 of sealed block
R_HOME = 4.5  # same radius family as composition-probe position instrument
OUT = ROOT / "artifacts" / "qualitative_capture" / "4v4_scaffold_trajectory_strip"
SEALED_CSV = S.ROWS_PATH
SPEC_PATH = S.SPEC_PATH


def _sealed_terminal(arm: str, pole: str, seed: int) -> tuple[int, int, int]:
    with SEALED_CSV.open(encoding="utf-8") as fh:
        rows = [
            r for r in csv.DictReader(fh)
            if int(r["seed"]) == seed and r["arm"] == arm and r["pole"] == pole
        ]
    if len(rows) != 1:
        raise SystemExit(f"REFUSING: expected 1 sealed row {arm}/{pole}/{seed}, got {len(rows)}")
    return int(rows[0]["blue"]), int(rows[0]["red"]), int(rows[0]["win"])


def _replay(
    setup: dict,
    policies: dict,
    arm: str,
    pole: str,
    seed: int,
    device: str,
) -> dict:
    from experiments.opponent_spec import assert_live_opponent_batch
    from gpu_env._core._entity_obs import augment_obs_with_entities

    R2 = setup["R2"]
    policy = policies[S.ARM_POLICY[arm]]
    forced = S.pair_for_seed(seed) if arm == "A_prime" else ()
    env = R2.build_env(device, seed)
    core = env.core
    ticks: list[dict] = []
    try:
        policy.reset_strategy()
        gen, key = S._open_opponent(env, core, setup["genomes"], pole, "qual")
        obs = env.reset()
        obs["global_state"] = env.state()
        obs = augment_obs_with_entities(obs, core, side="blue")
        assert_live_opponent_batch(core, gen, allowed_keys=(key,), context=f"qual {arm} {pole}")
        for i in forced:
            P.install_forced_defend_target(core, int(i))

        terminal = None
        for t in range(R2.MAX_STEPS):
            snap = P._snap(core)
            action, _ = policy.predict(obs, deterministic=True)
            bx = snap["pos"][:, 0].astype(float).tolist()
            by = snap["pos"][:, 1].astype(float).tolist()
            d_home = np.linalg.norm(snap["pos"] - snap["flag_home"][None, :], axis=-1)
            active = snap["alive"] & ~snap["tagged"]
            near_home = active & ~snap["carrying"] & (d_home <= R_HOME)
            ticks.append({
                "tick": t,
                "blue_x": bx,
                "blue_y": by,
                "blue_alive": snap["alive"].astype(int).tolist(),
                "blue_carrying": snap["carrying"].astype(int).tolist(),
                "blue_tagged": snap["tagged"].astype(int).tolist(),
                "forced_defend": [1 if i in forced else 0 for i in range(S.N)],
                "near_home": near_home.astype(int).tolist(),
                "n_near_home": int(near_home.sum()),
                "n_active": int(active.sum()),
                "n_attack_like": int((active & ~near_home).sum()),
                "action": np.asarray(action).reshape(-1).astype(int).tolist(),
                "blue_score": int(core.blue_score[0]),
                "red_score": int(core.red_score[0]),
                "flag_home": snap["flag_home"].astype(float).tolist(),
            })
            env.step_async(action)
            obs, _r, done, info = env.step_wait()
            obs["global_state"] = env.state()
            obs = augment_obs_with_entities(obs, core, side="blue")
            if bool(np.asarray(done).any()):
                i0 = info[0] if isinstance(info, (list, tuple)) else info
                res = (i0 or {}).get("episode_result") or {}
                terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                break
        if terminal is None:
            terminal = (int(core.blue_score[0]), int(core.red_score[0]))
    finally:
        env.close()

    exp_b, exp_r, exp_w = _sealed_terminal(arm, pole, seed)
    match = terminal == (exp_b, exp_r)
    return {
        "arm": arm,
        "pole": pole,
        "seed": seed,
        "forced_ids": list(forced),
        "terminal": {"blue": terminal[0], "red": terminal[1]},
        "sealed_terminal": {"blue": exp_b, "red": exp_r, "win": exp_w},
        "fidelity": "MATCH" if match else "MISMATCH",
        "n_ticks": len(ticks),
        "ticks": ticks,
        "mean_n_near_home": float(np.mean([t["n_near_home"] for t in ticks])) if ticks else 0.0,
        "mean_n_attack_like": float(np.mean([t["n_attack_like"] for t in ticks])) if ticks else 0.0,
        "mean_attack_defense_ratio": float(
            np.mean([
                (t["n_attack_like"] / t["n_near_home"]) if t["n_near_home"] > 0 else float(t["n_attack_like"])
                for t in ticks
            ])
        ) if ticks else 0.0,
    }


def _first_divergence(a_ticks: list[dict], b_ticks: list[dict], eps: float = 1e-4) -> dict:
    n = min(len(a_ticks), len(b_ticks))
    first_action = first_pos = None
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


def _write_csv(cell: dict, path: Path) -> None:
    n = S.N
    fields = (
        ["tick", "blue_score", "red_score", "n_near_home", "n_attack_like"]
        + [f"blue_x{i}" for i in range(n)]
        + [f"blue_y{i}" for i in range(n)]
        + [f"forced{i}" for i in range(n)]
        + [f"near_home{i}" for i in range(n)]
    )
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for t in cell["ticks"]:
            row = {
                "tick": t["tick"],
                "blue_score": t["blue_score"],
                "red_score": t["red_score"],
                "n_near_home": t["n_near_home"],
                "n_attack_like": t["n_attack_like"],
            }
            for i in range(n):
                row[f"blue_x{i}"] = t["blue_x"][i]
                row[f"blue_y{i}"] = t["blue_y"][i]
                row[f"forced{i}"] = t["forced_defend"][i]
                row[f"near_home{i}"] = t["near_home"][i]
            w.writerow(row)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--role-n", type=int, default=32, help="seeds for role aggregate")
    args = ap.parse_args()

    spec = S._load_json(SPEC_PATH)
    setup = S._setup_poles()
    paths = S._checkpoint_paths(spec)
    policies = S._load_policies(setup["R2"], paths, args.device)

    OUT.mkdir(parents=True, exist_ok=True)
    cells = []
    for arm in ("A_prime", "pi_B"):
        for pole in ("A", "B"):
            print(f"replay {arm} pole {pole} seed {args.seed} ...", flush=True)
            cell = _replay(setup, policies, arm, pole, args.seed, args.device)
            tag = f"{arm}_pole{pole}_seed{args.seed}"
            csv_name = f"{tag}_traj.csv"
            _write_csv(cell, OUT / csv_name)
            slim = dict(cell)
            slim["ticks_file"] = csv_name
            cells.append(slim)
            print(f"  fidelity={cell['fidelity']} terminal={cell['terminal']} n={cell['n_ticks']}", flush=True)

    by = {(c["arm"], c["pole"]): c for c in cells}
    divergence = {
        pole: _first_divergence(by[("A_prime", pole)]["ticks"], by[("pi_B", pole)]["ticks"])
        for pole in ("A", "B")
    }
    fidelities = [c["fidelity"] for c in cells]
    blob = {
        "record": "Matched A'/pi_B trajectory strip (4v4 scaffolded crossover PASS*)",
        "seed": args.seed,
        "r_home": R_HOME,
        "device": args.device,
        "fidelity": "ALL_MATCH" if all(f == "MATCH" for f in fidelities) else "HAS_MISMATCH",
        "divergence_by_pole": divergence,
        "cells": cells,
        "claim_guard": S.GUARD_SENTENCE,
    }
    (OUT / "trajectories.json").write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")

    # Role allocation aggregate (descriptive; not a gate)
    role_means: dict[str, dict[str, float]] = {}
    role_seeds = list(range(S.SEED_BASE, S.SEED_BASE + int(args.role_n)))
    for arm in ("A_prime", "pi_B"):
        for pole in ("A", "B"):
            key = f"{arm}_pole{pole}"
            defs, atks, ratios = [], [], []
            for seed in role_seeds:
                print(f"role {key} seed {seed} ...", flush=True)
                cell = _replay(setup, policies, arm, pole, seed, args.device)
                if cell["fidelity"] != "MATCH":
                    raise SystemExit(f"REFUSING: role aggregate fidelity fail {key} seed {seed}")
                defs.append(cell["mean_n_near_home"])
                atks.append(cell["mean_n_attack_like"])
                ratios.append(cell["mean_attack_defense_ratio"])
            role_means[key] = {
                "num_defenders": float(np.mean(defs)),
                "num_attackers": float(np.mean(atks)),
                "attack_defense_ratio": float(np.mean(ratios)),
                "n_episodes": len(role_seeds),
            }

    role_blob = {
        "record": "4v4 scaffold role allocation (position-near-home, exploratory)",
        "r_home": R_HOME,
        "seeds": role_seeds,
        "instrument": (
            "ACTIVE & not carrying & dist(own flag home) <= r_home => defender-like; "
            "else ACTIVE => attacker-like. A' also has two forced DEFEND targets."
        ),
        "per_condition_means": role_means,
        "note": "Descriptive companion; not a specialization gate. Coarse telemetry can be pole-driven.",
        "claim_guard": S.GUARD_SENTENCE,
    }
    (OUT / "role_allocation.json").write_text(json.dumps(role_blob, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {OUT / 'trajectories.json'} fidelity={blob['fidelity']}")
    print(f"wrote {OUT / 'role_allocation.json'}")
    return 0 if blob["fidelity"] == "ALL_MATCH" else 2


if __name__ == "__main__":
    raise SystemExit(main())
