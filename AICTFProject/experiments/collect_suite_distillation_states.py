"""Collect SUITE_DISTILLATION_<N>V<N> under CLOSEST_DEFENDS(k), for any suite team size.

Pole A acts with the sealed DEFEND_ATTACK_SPLIT composite (pi_D + frozen attack)
so the CLOSEST_DEFENDS allocator is live in the state distribution. Pole B acts
with entity-repair pi_B. Entity tensors and roles are stored so entity-repair
KL teachers can be queried at distillation time.

Implements SUITE_DISTILLATION_<N>V<N>_SPEC.json. One-shot: refuses if the dataset
manifest already exists.

Run:
  python experiments/collect_suite_distillation_states.py --team-size 4 --device cuda
  python experiments/collect_suite_distillation_states.py --team-size 2 --device cpu --smoke
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
#: One collector for every suite scale: team size is an argument, and every path and knob
#: below is derived from it. There is no module-level team size
#: (CROSS_SCALE_CANONICAL_RECIPE_V1.json#STAGE_IMPLEMENTATIONS_required).
#: CLOSEST_DEFENDS defender count per scale -- the one scale knob besides N.
K_DEFEND_BY_SCALE = {2: 1, 4: 2, 6: 1}
SUPPORTED_TEAM_SIZES = (2, 4, 6)


def _spec_path(n: int) -> Path:
    return SD / f"SUITE_DISTILLATION_{n}V{n}_SPEC.json"


def _out_dir(n: int, smoke: bool) -> Path:
    stem = f"suite_distillation_{n}v{n}" + ("_SMOKE" if smoke else "")
    return SD / stem / "states"


def _manifest(n: int, smoke: bool) -> Path:
    return SD / f"SUITE_DISTILLATION_{n}V{n}_DATASET{'_SMOKE' if smoke else ''}.json"


ENTITY_KEYS = ("teammates", "teammates_valid", "enemies", "enemies_valid")
STORE_KEYS = (
    "grid", "vec", "agent_mask", "mask", "global_state",
    *ENTITY_KEYS, "roles", "decision_mask", "step",
)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _composite_predict(trained_policy, attack_policy, obs) -> np.ndarray:
    from rl.custom_ppo.split_attack_defend import splice_actions
    import torch

    trained_action, _ = trained_policy.predict(obs, deterministic=True)
    attack_action, _ = attack_policy.predict(obs, deterministic=True)
    n_agents = int(trained_policy.model.n_agents)
    heads_per_agent = int(trained_policy.model.heads_per_agent)
    trained_t = torch.as_tensor(np.asarray(trained_action), dtype=torch.long).reshape(1, -1)
    attack_t = torch.as_tensor(np.asarray(attack_action), dtype=torch.long).reshape(1, -1)
    roles_t = torch.as_tensor(np.asarray(obs["roles"]), dtype=torch.float32).reshape(1, n_agents)
    is_defend = roles_t < 0.5
    exec_t = splice_actions(trained_t, attack_t, is_defend, heads_per_agent)
    return exec_t.reshape(-1).numpy().astype(np.int64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="NON-SCIENTIFIC plumbing smoke: disposable 999xxxxx seeds, "
             "_SMOKE paths, never touches the real manifest.",
    )
    ap.add_argument("--n-per-pole", type=int, default=None)
    ap.add_argument("--team-size", type=int, required=True, choices=SUPPORTED_TEAM_SIZES)
    args = ap.parse_args()
    device = args.device
    smoke = bool(args.smoke)
    N_AGENTS = int(args.team_size)
    K_DEFEND = K_DEFEND_BY_SCALE[N_AGENTS]
    SPEC_PATH = _spec_path(N_AGENTS)

    if not SPEC_PATH.is_file():
        raise SystemExit(
            f"FAIL-CLOSED: {SPEC_PATH.name} not found. Each scale needs its own frozen "
            f"collection spec naming the teachers, acting policies, seeds and poles."
        )
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: {SPEC_PATH.name} not frozen: {spec.get('status')!r}")

    n_per_pole = int(args.n_per_pole) if args.n_per_pole else int(spec["DATASET"]["n_per_pole"])
    if smoke:
        n_per_pole = max(min(n_per_pole, 12), 10)
        seed_a_base, seed_b_base = 99_920_001, 99_920_101
    else:
        seeds_spec = spec["SEEDS"]
        seed_a_base = int(str(seeds_spec["collection_A"]).split("..")[0])
        seed_b_base = int(str(seeds_spec["collection_B"]).split("..")[0])
    out_dir = _out_dir(N_AGENTS, smoke)
    manifest = _manifest(N_AGENTS, smoke)

    seeds = {
        "A": list(range(seed_a_base, seed_a_base + n_per_pole)),
        "B": list(range(seed_b_base, seed_b_base + n_per_pole)),
    }

    if manifest.is_file():
        raise SystemExit(
            f"REFUSING: {manifest.name} exists; "
            f"{'smoke rerun' if smoke else 'the suite dataset is collected once'}"
        )
    if smoke and out_dir.is_dir():
        import shutil
        shutil.rmtree(out_dir)

    # Propagate team size by name (same trap as collect_distillation_states_scale).
    import experiments.collect_distillation_states as C
    import experiments.r2_learned_crossover as R2
    C.N_AGENTS = N_AGENTS
    R2.AGENTS = N_AGENTS

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch,
        install_keyed_opponent_overlays,
        pole_A_genome,
        pole_B_genome,
    )
    import experiments.phase0_collect_scorer_data as P0
    from experiments.tqdm_loop import tqdm_iter
    from gpu_env._core._entity_obs import augment_obs_with_entities
    from rl.causal_supervision import decision_mask_from_core
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.custom_ppo.rule_role_assignment import RoleHoldState, roles_from_core

    act = spec["ACTING_DEPLOYMENT_locked"]
    kl = spec["KL_TEACHERS_locked"]
    pins = {
        "pi_D": act["Pole_A"]["pi_D"],
        "frozen_attack": act["Pole_A"]["frozen_attack_pi_A"],
        "pi_B_act": act["Pole_B"],
        "pi_A_kl": kl["pi_A"],
        "pi_B_kl": kl["pi_B"],
    }
    paths = {}
    for name, pin in pins.items():
        p = ROOT / pin["path"]
        if not p.is_file() or _sha(p) != pin["sha256"]:
            raise SystemExit(f"REFUSING: {name} missing or sha mismatch: {p}")
        paths[name] = p
    # KL and acting attack/B share pins; refuse drift.
    if pins["frozen_attack"]["sha256"] != pins["pi_A_kl"]["sha256"]:
        raise SystemExit("REFUSING: frozen_attack sha != KL pi_A sha")
    if pins["pi_B_act"]["sha256"] != pins["pi_B_kl"]["sha256"]:
        raise SystemExit("REFUSING: acting pi_B sha != KL pi_B sha")

    probe = R2.build_env(device, seeds["A"][0])
    obs_space, act_space = probe.observation_space, probe.action_space
    grid_dim = int(obs_space.spaces["grid"].shape[0])
    probe.close()
    if grid_dim != N_AGENTS:
        raise SystemExit(f"FAIL-CLOSED: probe grid dim {grid_dim} != {N_AGENTS}")

    pi_D = load_custom_ppo_policy(str(paths["pi_D"]), obs_space, act_space, device=device)
    frozen_attack = load_custom_ppo_policy(
        str(paths["frozen_attack"]), obs_space, act_space, device=device
    )
    pi_B = load_custom_ppo_policy(str(paths["pi_B_act"]), obs_space, act_space, device=device)

    if not bool(getattr(pi_D.model, "role_conditioning_enabled", False)):
        raise SystemExit("REFUSING: pi_D must be role-conditioned")
    if bool(getattr(frozen_attack.model, "role_conditioning_enabled", False)):
        raise SystemExit("REFUSING: frozen attack must NOT be role-conditioned")
    if getattr(pi_D.model, "entity_encoder", None) is None:
        raise SystemExit("REFUSING: pi_D expects entity repair")
    if getattr(frozen_attack.model, "entity_encoder", None) is None:
        raise SystemExit("REFUSING: frozen attack expects entity repair")
    if getattr(pi_B.model, "entity_encoder", None) is None:
        raise SystemExit("REFUSING: pi_B expects entity repair")
    for name, pol in (("pi_D", pi_D), ("frozen_attack", frozen_attack), ("pi_B", pi_B)):
        if bool(getattr(pol.model, "uses_latent_strategy", False)):
            raise SystemExit(f"REFUSING: {name} is latent-conditioned")

    alloc = spec["ALLOCATOR_locked"]
    if int(alloc["k_defend"]) != K_DEFEND:
        raise SystemExit(f"REFUSING: SPEC k_defend={alloc['k_defend']} != {K_DEFEND}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"SUITE DISTILLATION {N_AGENTS}V{N_AGENTS} COLLECT  {_now()}  device={device}"
          f"{'  [SMOKE]' if smoke else ''}")
    print(f"  CLOSEST_DEFENDS k={K_DEFEND} fixed_for_episode={alloc['fixed_for_episode']}")
    print(f"  n_per_pole={n_per_pole}  A={seeds['A'][0]}..{seeds['A'][-1]}  "
          f"B={seeds['B'][0]}..{seeds['B'][-1]}\n", flush=True)

    def _attach_roles(obs, core, hold: RoleHoldState, *, force: bool):
        roles = roles_from_core(core, hold, force=force, advance_age=True)
        out = dict(obs)
        out["roles"] = roles.detach().cpu().numpy().astype(np.float32)
        return out

    shards, totals = [], {
        "A": {"episodes": 0, "steps": 0, "decision_rows": 0, "wins": 0},
        "B": {"episodes": 0, "steps": 0, "decision_rows": 0, "wins": 0},
    }

    for pole in ("A", "B"):
        ep_iter = tqdm_iter(
            list(enumerate(seeds[pole])),
            desc=f"suite{N_AGENTS}v{N_AGENTS} pole {pole}",
            total=n_per_pole,
            unit="ep",
        )
        for ep_i, seed in ep_iter:
            env = R2.build_env(device, seed)
            core = env.core
            role_hold = RoleHoldState(
                int(env.num_envs),
                N_AGENTS,
                hold_ticks=int(alloc["hold_ticks_H_r"]),
                device=device,
                fixed_for_episode=bool(alloc["fixed_for_episode"]),
                k_defend=K_DEFEND,
            )
            try:
                pi_D.reset_strategy()
                frozen_attack.reset_strategy()
                pi_B.reset_strategy()
                core._bt_profile_override = None
                core._sds_opening_hold_steps = 0
                genomes = (
                    {"OP6": pole_A_genome(N_AGENTS)} if pole == "A"
                    else {"OP7": pole_B_genome(N_AGENTS)}
                )
                install_keyed_opponent_overlays(core, genomes)
                key = P0.POLES[pole]
                env.env_method("set_phase", phase_from_tag(key))
                env.env_method("set_next_opponent", "SCRIPTED", key)
                obs = env.reset()
                obs["global_state"] = env.state()
                obs = augment_obs_with_entities(obs, core, side="blue")
                # Roles always attached under CD so Pole A composite + stored
                # allocator audit see the same geometric assignment.
                obs = _attach_roles(obs, core, role_hold, force=True)
                assert_live_opponent_batch(
                    core, genomes, allowed_keys=(key,),
                    context=f"suite distill {N_AGENTS}v{N_AGENTS} {pole} seed {seed}",
                )
                resolved = core._bt_resolved_profile_tensors()
                got = resolved.get("min_alive_for_defender")
                got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
                if got_val != N_AGENTS:
                    raise SystemExit(
                        f"FAIL-CLOSED: pole {pole} min_alive_for_defender="
                        f"{got_val}, expected {N_AGENTS}"
                    )

                rows = {k: [] for k in STORE_KEYS}
                steps, terminal = 0, None
                for t in range(R2.MAX_STEPS):
                    d = decision_mask_from_core(core, N_AGENTS, side="blue")
                    d_np = np.asarray(d.detach().cpu())[0].copy()
                    if d_np.any():
                        for k in ("grid", "vec", "agent_mask", "mask", "global_state",
                                  *ENTITY_KEYS, "roles"):
                            rows[k].append(np.asarray(obs[k])[0].copy())
                        rows["decision_mask"].append(d_np)
                        rows["step"].append(t)

                    if pole == "A":
                        action = _composite_predict(pi_D, frozen_attack, obs)
                    else:
                        action, _ = pi_B.predict(obs, deterministic=True)
                    env.step_async(action)
                    obs, _r, done, info = env.step_wait()
                    obs["global_state"] = env.state()
                    obs = augment_obs_with_entities(obs, core, side="blue")
                    obs = _attach_roles(obs, core, role_hold, force=False)
                    steps += 1
                    if bool(np.asarray(done).any()):
                        i0 = info[0] if isinstance(info, (list, tuple)) else info
                        res = (i0 or {}).get("episode_result") or {}
                        terminal = (
                            int(res.get("blue_score", 0)),
                            int(res.get("red_score", 0)),
                        )
                        break
                if terminal is None:
                    terminal = (int(core.blue_score[0]), int(core.red_score[0]))
            finally:
                env.close()

            n_rows = len(rows["step"])
            shard = out_dir / f"{pole}_{seed}.npz"
            save = {
                "grid": np.asarray(rows["grid"], dtype=np.float32),
                "vec": np.asarray(rows["vec"], dtype=np.float32),
                "agent_mask": np.asarray(rows["agent_mask"], dtype=np.float32),
                "mask": np.asarray(rows["mask"], dtype=np.float32),
                "global_state": np.asarray(rows["global_state"], dtype=np.float32),
                "teammates": np.asarray(rows["teammates"], dtype=np.float32),
                "teammates_valid": np.asarray(rows["teammates_valid"], dtype=bool),
                "enemies": np.asarray(rows["enemies"], dtype=np.float32),
                "enemies_valid": np.asarray(rows["enemies_valid"], dtype=bool),
                "roles": np.asarray(rows["roles"], dtype=np.float32),
                "decision_mask": np.asarray(rows["decision_mask"], dtype=bool),
                "step": np.asarray(rows["step"], dtype=np.int32),
                "pole": np.full((n_rows,), 0 if pole == "A" else 1, dtype=np.int8),
                "episode": np.full((n_rows,), ep_i, dtype=np.int32),
                "seed": np.full((n_rows,), seed, dtype=np.int64),
            }
            np.savez_compressed(shard, **save)
            shards.append({
                "pole": pole, "episode": ep_i, "seed": seed, "steps": steps,
                "decision_rows": n_rows, "blue": terminal[0], "red": terminal[1],
                "file": str(shard.relative_to(ROOT)),
            })
            tt = totals[pole]
            tt["episodes"] += 1
            tt["steps"] += steps
            tt["decision_rows"] += n_rows
            tt["wins"] += int(terminal[0] > terminal[1])

    for pole in ("A", "B"):
        if totals[pole]["decision_rows"] <= 0:
            raise SystemExit(f"REFUSING: pole {pole} produced zero decision-bearing rows")

    manifest.write_text(json.dumps({
        "record": f"Suite distillation state set ({N_AGENTS}v{N_AGENTS})"
                  + (" -- NON-SCIENTIFIC SMOKE" if smoke else ""),
        "status": "SMOKE_NOT_SCIENTIFIC" if smoke else "FROZEN_DATASET",
        "utc": _now(),
        "implements": f"{SPEC_PATH.name}#DATASET",
        "team_size": N_AGENTS,
        "allocator": {
            "rule": "CLOSEST_DEFENDS",
            "k_defend": K_DEFEND,
            "fixed_for_episode": bool(alloc["fixed_for_episode"]),
            "hold_ticks_H_r": int(alloc["hold_ticks_H_r"]),
        },
        "acting": {
            "Pole_A": "DEFEND_ATTACK_SPLIT(pi_D + frozen_attack)",
            "Pole_B": "pi_B entity-repair",
            "pi_D": {"path": str(pins["pi_D"]["path"]), "sha256": pins["pi_D"]["sha256"]},
            "frozen_attack": {
                "path": str(pins["frozen_attack"]["path"]),
                "sha256": pins["frozen_attack"]["sha256"],
            },
            "pi_B": {"path": str(pins["pi_B_act"]["path"]), "sha256": pins["pi_B_act"]["sha256"]},
        },
        "teachers": {
            "pi_A": {"path": str(pins["pi_A_kl"]["path"]), "sha256": pins["pi_A_kl"]["sha256"]},
            "pi_B": {"path": str(pins["pi_B_kl"]["path"]), "sha256": pins["pi_B_kl"]["sha256"]},
        },
        "seeds": {k: [v[0], v[-1]] for k, v in seeds.items()},
        "device": device,
        "decision_rows_only": True,
        "stored_entity_tensors": True,
        "stored_roles": True,
        "totals": {
            p: {**t, "teacher_deployment_win_rate": t["wins"] / max(1, t["episodes"])}
            for p, t in totals.items()
        },
        "shards": shards,
    }, indent=2), encoding="utf-8")
    print(f"\n  A: {totals['A']}\n  B: {totals['B']}\n  -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
