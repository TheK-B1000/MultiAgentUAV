"""Replay sealed role-allocated (Ours) crossover episodes and record agent movement.

For each team size this replays, with the exact nominal per-episode loop of
experiments/eval_specialist_crossover_scaled.py (same checkpoints, CLOSEST_DEFENDS split
policy, roles fixed for the episode, certified poles, deterministic actions), a set of
(policy, pole, seed) cells drawn from the sealed confirmatory block, and records every
blue/red position, role and status per decision tick. Each replayed terminal score is
checked against the sealed row (fidelity MATCH / MISMATCH).

Cells per team size:
  * trajectory seed: the lowest-ID sealed seed with the full crossover pattern
    (pi_A wins A, pi_B loses A, pi_B wins B, pi_A loses B); fallback = highest
    per-seed (Delta_A + Delta_B), lowest ID.
  * occupancy seeds: the first --occupancy-n seeds of the sealed block.
  Every seed is run for pi_A and pi_B on Pole A and Pole B.

Descriptive only: no seed is spent, nothing sealed is modified.

Outputs: artifacts/qualitative_capture/paper_suite_trajectories/<N>v<N>/*.npz + manifest.json

Run (from AICTFProject):
  ./.venv/Scripts/python.exe experiments/export_paper_trajectories.py --device cuda
"""
from __future__ import annotations

import argparse
import csv
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
OUT = ROOT / "artifacts" / "qualitative_capture" / "paper_suite_trajectories"
BASE_KEY = {"A": "OP6", "B": "OP7"}

SUITE = {
    2: {
        "rows": "standardized_2v2_split_k1_confirmatory_specialist_crossover_eval_rows.csv",
        "pi_D": "artifacts/scale_2v2_specialists/pi_A_specialist_2v2_std_split_defend_k1/ckpts/final_pi_A_specialist_2v2_std_split_defend_k1.zip",
        "attack": "artifacts/scale_2v2_specialists/pi_A_specialist_2v2_std_entity_repair/ckpts/final_pi_A_specialist_2v2_std_entity_repair.zip",
        "pi_B": "artifacts/scale_2v2_specialists/pi_B_specialist_2v2_std_entity_repair/ckpts/final_pi_B_specialist_2v2_std_entity_repair.zip",
        "k_defend": 0,
        "sha": {"pi_D": "e75b27751757", "attack": "858805dde358", "pi_B": "9ee024ad6356"},
    },
    4: {
        "rows": "defend_attack_split_policy_a_v1_confirmatory_v1_specialist_crossover_eval_rows.csv",
        "pi_D": "artifacts/scale_4v4_specialists/pi_A_specialist_4v4_a4_split_attack_defend_v1/ckpts/final_pi_A_specialist_4v4_a4_split_attack_defend_v1.zip",
        "attack": "artifacts/scale_4v4_specialists/pi_A_specialist_4v4_b3_entity_repair/ckpts/final_pi_A_specialist_4v4_b3_entity_repair.zip",
        "pi_B": "artifacts/scale_4v4_specialists/pi_B_specialist_4v4_b3_entity_repair_corrected/ckpts/final_pi_B_specialist_4v4_b3_entity_repair_corrected.zip",
        "k_defend": 0,
        "sha": {"pi_D": "b538cdf4ba5d", "attack": "94dde69d091a", "pi_B": "021342c84bbe"},
    },
    6: {
        "rows": "standardized_6v6_split_k1_confirmatory_specialist_crossover_eval_rows.csv",
        "pi_D": "artifacts/scale_6v6_specialists/pi_A_specialist_6v6_split_defend_k1_v1/ckpts/final_pi_A_specialist_6v6_split_defend_k1_v1.zip",
        "attack": "artifacts/scale_6v6_specialists/pi_A_specialist_6v6_c2_entity_repair/ckpts/final_pi_A_specialist_6v6_c2_entity_repair.zip",
        "pi_B": "artifacts/scale_6v6_specialists/pi_B_specialist_6v6_c2_entity_repair/ckpts/final_pi_B_specialist_6v6_c2_entity_repair.zip",
        "k_defend": 1,
        "sha": {"pi_D": "77261f919009", "attack": "3298000480ac", "pi_B": "fc0043235d3a"},
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sealed_rows(n: int) -> dict[tuple[str, str, int], dict]:
    with (SD / SUITE[n]["rows"]).open(newline="", encoding="utf-8") as fh:
        return {(r["policy"], r["pole"], int(r["seed"])): r for r in csv.DictReader(fh)}


def pick_trajectory_seed(rows: dict) -> tuple[int, str]:
    seeds = sorted({s for (_, _, s) in rows})
    w = lambda p, q, s: int(rows[(p, q, s)]["win"])  # noqa: E731
    full = [s for s in seeds
            if w("pi_A", "A", s) == 1 and w("pi_B", "A", s) == 0
            and w("pi_B", "B", s) == 1 and w("pi_A", "B", s) == 0]
    if full:
        return full[0], "full_crossover_lowest_id"
    score = lambda s: (w("pi_A", "A", s) - w("pi_B", "A", s)) + (w("pi_B", "B", s) - w("pi_A", "B", s))  # noqa: E731
    best = max(score(s) for s in seeds)
    return min(s for s in seeds if score(s) == best), f"max_delta_sum={best}_lowest_id"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--scales", type=int, nargs="+", default=[2, 4, 6], choices=(2, 4, 6))
    ap.add_argument("--occupancy-n", type=int, default=16)
    ap.add_argument("--max-cells", type=int, default=0, help="smoke: stop after this many new cells")
    args = ap.parse_args()

    import torch
    import experiments.r2_learned_crossover as R2
    from experiments.opponent_spec import assert_live_opponent_batch, install_keyed_opponent_overlays
    from experiments.pole_attestation import (
        assert_resolved_matches_certification, governing_certification, resolve_pole_genome,
    )
    from experiments.tqdm_loop import set_postfix, tqdm_iter
    from gpu_env._core._entity_obs import augment_obs_with_entities
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.custom_ppo.rule_role_assignment import RoleHoldState, roles_from_core
    from rl.custom_ppo.split_attack_defend import splice_actions

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    plan: list[tuple[int, str, str, int, str]] = []
    meta: dict[int, dict] = {}
    for n in args.scales:
        rows = sealed_rows(n)
        block = sorted({s for (_, _, s) in rows})
        traj_seed, rule = pick_trajectory_seed(rows)
        occ = block[: int(args.occupancy_n)]
        meta[n] = {"trajectory_seed": traj_seed, "trajectory_rule": rule, "occupancy_seeds": occ,
                   "sealed_rows": SUITE[n]["rows"], "k_defend": SUITE[n]["k_defend"] or n // 2}
        for seed in [traj_seed] + [s for s in occ if s != traj_seed]:
            kind = "trajectory" if seed == traj_seed else "occupancy"
            for policy in ("pi_A", "pi_B"):
                for pole in ("A", "B"):
                    plan.append((n, policy, pole, seed, kind))

    loaded: dict[int, dict] = {}

    def setup(n: int) -> dict:
        if n in loaded:
            return loaded[n]
        loaded.clear()
        R2.AGENTS = n
        cfg = SUITE[n]
        paths = {k: ROOT / cfg[k] for k in ("pi_D", "attack", "pi_B")}
        for k, p in paths.items():
            if not _sha(p).startswith(cfg["sha"][k]):
                raise SystemExit(f"REFUSING: {k} checkpoint hash mismatch at {n}v{n}: {p}")
        _verdict, cert = governing_certification(n)
        genomes = {}
        for pol in ("A", "B"):
            g = resolve_pole_genome(pol, n, None)
            assert_resolved_matches_certification(pol, n, cert, g, is_smoke=False)
            genomes[pol] = {BASE_KEY[pol]: g}
        probe = R2.build_env(device, 99_990_000 + n)
        obs_space, act_space = probe.observation_space, probe.action_space
        probe.close()
        pols = {k: load_custom_ppo_policy(str(p), obs_space, act_space, device=device) for k, p in paths.items()}
        if not bool(getattr(pols["pi_D"].model, "role_conditioning_enabled", False)):
            raise SystemExit(f"REFUSING: {n}v{n} pi_D is not role-conditioned")
        loaded[n] = {"pols": pols, "genomes": genomes, "rows": sealed_rows(n)}
        return loaded[n]

    def replay(n: int, policy_name: str, pole: str, seed: int) -> dict:
        s = setup(n)
        pols = s["pols"]
        policy = pols["pi_D"] if policy_name == "pi_A" else pols["pi_B"]
        attack = pols["attack"] if policy_name == "pi_A" else None
        env = R2.build_env(device, seed)
        core = env.core
        hold = None
        if bool(getattr(policy.model, "role_conditioning_enabled", False)):
            hold_ticks = int(getattr(policy.model, "role_hold_ticks", 0) or 0) or 8
            k = int(SUITE[n]["k_defend"])
            hold = RoleHoldState(int(env.num_envs), int(policy.model.n_agents), hold_ticks=hold_ticks,
                                 device=device, fixed_for_episode=True, k_defend=(k if k > 0 else None))

        def attach(obs, force):
            if hold is None:
                return obs
            out = dict(obs)
            out["roles"] = roles_from_core(core, hold, force=force, advance_age=True).detach().cpu().numpy().astype(np.float32)
            return out

        def act(obs):
            if attack is None:
                a, _ = policy.predict(obs, deterministic=True)
                return a
            ta, _ = policy.predict(obs, deterministic=True)
            aa, _ = attack.predict(obs, deterministic=True)
            na, hpa = int(policy.model.n_agents), int(policy.model.heads_per_agent)
            tt = torch.as_tensor(np.asarray(ta), dtype=torch.long).reshape(1, -1)
            at = torch.as_tensor(np.asarray(aa), dtype=torch.long).reshape(1, -1)
            rt = torch.as_tensor(np.asarray(obs["roles"]), dtype=torch.float32).reshape(1, na)
            return splice_actions(tt, at, rt < 0.5, hpa).reshape(-1).numpy().astype(np.int64)

        c = lambda x: x.detach().cpu().numpy()[0].astype(np.float32)  # noqa: E731
        rec = {k: [] for k in ("blue_x", "blue_y", "red_x", "red_y", "blue_alive", "blue_tagged",
                               "blue_carrying", "red_alive", "red_tagged", "red_carrying", "roles")}
        try:
            policy.reset_strategy()
            core._bt_profile_override = None
            core._sds_opening_hold_steps = 0
            genomes = s["genomes"][pole]
            install_keyed_opponent_overlays(core, genomes)
            key = BASE_KEY[pole]
            env.env_method("set_phase", phase_from_tag(key))
            env.env_method("set_next_opponent", "SCRIPTED", key)
            obs = env.reset()
            obs["global_state"] = env.state()
            obs = augment_obs_with_entities(obs, core, side="blue")
            obs = attach(obs, True)
            assert_live_opponent_batch(core, genomes, allowed_keys=(key,), context=f"traj {n}v{n} {pole} {seed}")
            got = core._bt_resolved_profile_tensors().get("min_alive_for_defender")
            got = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
            if got != n:
                raise SystemExit(f"FAIL-CLOSED: pole {pole} min_alive_for_defender={got} != {n}")
            home_b, home_r = c(core.blue_flag_home).reshape(-1)[:2], c(core.red_flag_home).reshape(-1)[:2]
            bounds = (float(core.cols - 1), float(core.rows - 1))
            terminal = None
            for _ in range(R2.MAX_STEPS):
                for k in ("blue_x", "blue_y", "red_x", "red_y"):
                    rec[k].append(c(getattr(core, k)))
                for k in ("blue_alive", "blue_tagged", "blue_carrying", "red_alive", "red_tagged", "red_carrying"):
                    rec[k].append(c(getattr(core, k)))
                rec["roles"].append(np.asarray(obs["roles"], dtype=np.float32).reshape(-1)
                                    if "roles" in obs else np.full(n, np.nan, np.float32))
                env.step_async(act(obs))
                obs, _r, done, info = env.step_wait()
                obs["global_state"] = env.state()
                obs = augment_obs_with_entities(obs, core, side="blue")
                obs = attach(obs, False)
                if bool(np.asarray(done).any()):
                    i0 = info[0] if isinstance(info, (list, tuple)) else info
                    res = (i0 or {}).get("episode_result") or {}
                    terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                    break
            if terminal is None:
                terminal = (int(core.blue_score[0]), int(core.red_score[0]))
        finally:
            env.close()
        sealed = s["rows"][(policy_name, pole, seed)]
        match = terminal == (int(sealed["blue"]), int(sealed["red"]))
        return {
            **{k: np.stack(v) for k, v in rec.items()},
            "flag_home_blue": home_b, "flag_home_red": home_r, "bounds": np.asarray(bounds, np.float32),
            "terminal": np.asarray(terminal, np.int32),
            "sealed_terminal": np.asarray((int(sealed["blue"]), int(sealed["red"])), np.int32),
            "fidelity_match": np.asarray(match),
        }

    manifests: dict[int, dict] = {}
    for n in meta:
        mp = OUT / f"{n}v{n}" / "manifest.json"
        mp.parent.mkdir(parents=True, exist_ok=True)
        m = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {"cells": {}}
        m.update({
            "record": "Paper agent-movement replays of sealed role-allocated (Ours) crossover episodes",
            "descriptive_only": True, "device": device, "updated_utc": _now(), **meta[n],
            "loop": "experiments/eval_specialist_crossover_scaled.py run_cell (nominal), reproduced verbatim",
        })
        manifests[n] = m
    new = 0
    bar = tqdm_iter(plan, desc=f"paper trajectories {'/'.join(f'{n}v{n}' for n in meta)}", unit="ep")
    for n, policy_name, pole, seed, kind in bar:
        set_postfix(bar, f"{n}v{n} {policy_name}@{pole} {seed}")
        name = f"{policy_name}_pole{pole}_{seed}.npz"
        path = OUT / f"{n}v{n}" / name
        manifest = manifests[n]
        if path.is_file() and manifest["cells"].get(name, {}).get("fidelity") == "MATCH":
            continue
        if args.max_cells and new >= args.max_cells:
            break
        cell = replay(n, policy_name, pole, seed)
        np.savez_compressed(path, **cell)
        fid = "MATCH" if bool(cell["fidelity_match"]) else "MISMATCH"
        manifest["cells"][name] = {"team_size": n, "policy": policy_name, "pole": pole, "seed": seed,
                                   "kind": kind, "ticks": int(cell["blue_x"].shape[0]),
                                   "terminal": cell["terminal"].tolist(),
                                   "sealed_terminal": cell["sealed_terminal"].tolist(), "fidelity": fid}
        (OUT / f"{n}v{n}" / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        new += 1
        if fid != "MATCH":
            print(f"  MISMATCH {n}v{n}/{name}: replay {cell['terminal'].tolist()} vs sealed "
                  f"{cell['sealed_terminal'].tolist()}", flush=True)
    fids = [c["fidelity"] for m in manifests.values() for c in m["cells"].values()]
    print(f"cells={len(fids)} match={fids.count('MATCH')} mismatch={fids.count('MISMATCH')}")
    return 0 if all(f == "MATCH" for f in fids) else 2


if __name__ == "__main__":
    raise SystemExit(main())
