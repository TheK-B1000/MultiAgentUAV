"""On-policy (student-state) collection for one DAgger round.

Implements ONPOLICY_TEACHER_DISTILLATION_SPEC.json#DESIGN.student_state_collection_per_round.
The CURRENT student (round 1: the offline TD student; round 2: the round-1 student) is deployed
with forced z in all four cells (z0@A, z1@A, z0@B, z1@B), 32 episodes each, deterministic --
the exact deployment convention the sealed eval uses. Every decision-bearing pre-action state
is stored with the exact observation dict and the live decision mask. Teacher labels are NOT
cached here; both teachers are queried on every state at training time.

Per-cell win rates are recorded as provenance only (non-sealed seeds); the protocol is fixed
and nothing about the run depends on them.

Run:  python experiments/collect_student_states.py --round 1 --device cpu
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
SPEC = SD / "ONPOLICY_TEACHER_DISTILLATION_SPEC.json"
N_PER_CELL = 32
N_AGENTS = 2
CELLS = ((0, "A"), (1, "A"), (0, "B"), (1, "B"))
ROUND_BASE = {1: 11_931_001, 2: 11_932_001}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def starting_student_record(rnd: int) -> Path:
    return (SD / "TEACHER_DISTILLATION_STUDENT_FROZEN.json" if rnd == 1
            else SD / f"ONPOLICY_TD_ROUND{rnd - 1}_STUDENT_FROZEN.json")


def cell_seeds(rnd: int) -> dict[tuple[int, str], list[int]]:
    base = ROUND_BASE[rnd]
    return {cell: list(range(base + i * N_PER_CELL, base + (i + 1) * N_PER_CELL))
            for i, cell in enumerate(CELLS)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True, choices=(1, 2))
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    rnd = args.round
    manifest = SD / f"ONPOLICY_TD_ROUND{rnd}_STATES.json"
    out_dir = SD / "teacher_distillation" / f"onpolicy_round{rnd}" / "states"
    if manifest.is_file():
        raise SystemExit(f"REFUSING: {manifest.name} exists; collection is one-shot per round")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: spec not frozen: {spec['status']!r}")
    rec_path = starting_student_record(rnd)
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    if rec["status"] != "FROZEN_STUDENT":
        raise SystemExit(f"REFUSING: starting student {rec_path.name} is {rec['status']!r}")
    ck = ROOT / rec["TERMINAL_CHECKPOINT"]["path"]
    if not ck.is_file() or _sha(ck) != rec["TERMINAL_CHECKPOINT"]["sha256"]:
        raise SystemExit("REFUSING: starting student checkpoint missing or sha mismatch")

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
    )
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.causal_supervision import decision_mask_from_core
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    device = args.device
    seeds = cell_seeds(rnd)
    probe = R2.build_env(device, seeds[CELLS[0]][0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    student = load_custom_ppo_policy(str(ck), obs_space, act_space, device=device)
    if not (student.model.uses_latent_strategy and student.model.latent_k == 2):
        raise SystemExit("REFUSING: starting student is not a K=2 latent policy")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"COLLECT STUDENT STATES  round {rnd}  {_now()}  device={device}")
    print(f"  student: {ck.name} (sha {rec['TERMINAL_CHECKPOINT']['sha256'][:12]}...)")
    print(f"  {N_PER_CELL} episodes x 4 cells, deterministic forced z, decision rows only\n", flush=True)

    shards, totals = [], {}
    for (z, pole) in CELLS:
        key = f"z{z}_pole{pole}"
        totals[key] = {"episodes": 0, "steps": 0, "decision_rows": 0, "wins": 0, "hit_cap": 0}
        for ep_i, seed in enumerate(seeds[(z, pole)]):
            env = R2.build_env(device, seed)
            core = env.core
            try:
                student.fixed_latent_strategy = True
                student.fixed_latent_strategy_id = int(z)
                student.reset_strategy()
                core._bt_profile_override = None
                core._sds_opening_hold_steps = 0
                genomes = {"OP6": pole_A_genome()} if pole == "A" else {}
                install_keyed_opponent_overlays(core, genomes)
                opp = P0.POLES[pole]
                env.env_method("set_phase", phase_from_tag(opp))
                env.env_method("set_next_opponent", "SCRIPTED", opp)
                obs = env.reset()
                obs["global_state"] = env.state()
                assert_live_opponent_batch(core, genomes, allowed_keys=(opp,),
                                           context=f"student collect r{rnd} z{z} {pole} seed {seed}")
                rows = {k: [] for k in ("grid", "vec", "agent_mask", "mask", "global_state",
                                        "decision_mask", "step")}
                steps, terminal = 0, None
                for t in range(R2.MAX_STEPS):
                    d = decision_mask_from_core(core, N_AGENTS, side="blue")
                    d_np = np.asarray(d.detach().cpu())[0].copy()
                    if d_np.any():
                        rows["grid"].append(np.asarray(obs["grid"])[0].copy())
                        rows["vec"].append(np.asarray(obs["vec"])[0].copy())
                        rows["agent_mask"].append(np.asarray(obs["agent_mask"])[0].copy())
                        rows["mask"].append(np.asarray(obs["mask"])[0].copy())
                        rows["global_state"].append(np.asarray(obs["global_state"])[0].copy())
                        rows["decision_mask"].append(d_np)
                        rows["step"].append(t)
                    action, _ = student.predict(obs, deterministic=True)
                    env.step_async(action)
                    obs, _r, done, info = env.step_wait()
                    obs["global_state"] = env.state()
                    steps += 1
                    if bool(np.asarray(done).any()):
                        i0 = info[0] if isinstance(info, (list, tuple)) else info
                        res = (i0 or {}).get("episode_result") or {}
                        terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                        break
                hit_cap = terminal is None
                if terminal is None:
                    terminal = (int(core.blue_score[0]), int(core.red_score[0]))
            finally:
                env.close()
            n_rows = len(rows["step"])
            shard = out_dir / f"{key}_{seed}.npz"
            np.savez_compressed(
                shard,
                grid=np.asarray(rows["grid"], dtype=np.float32),
                vec=np.asarray(rows["vec"], dtype=np.float32),
                agent_mask=np.asarray(rows["agent_mask"], dtype=np.float32),
                mask=np.asarray(rows["mask"], dtype=np.float32),
                global_state=np.asarray(rows["global_state"], dtype=np.float32),
                decision_mask=np.asarray(rows["decision_mask"], dtype=bool),
                step=np.asarray(rows["step"], dtype=np.int32),
                pole=np.full((n_rows,), 0 if pole == "A" else 1, dtype=np.int8),
                z_used=np.full((n_rows,), z, dtype=np.int8),
                episode=np.full((n_rows,), ep_i, dtype=np.int32),
                seed=np.full((n_rows,), seed, dtype=np.int64),
            )
            shards.append({"cell": key, "z": z, "pole": pole, "episode": ep_i, "seed": seed,
                           "steps": steps, "hit_cap": bool(hit_cap), "decision_rows": n_rows,
                           "blue": terminal[0], "red": terminal[1],
                           "file": str(shard.relative_to(ROOT))})
            tt = totals[key]
            tt["episodes"] += 1; tt["steps"] += steps; tt["decision_rows"] += n_rows
            tt["wins"] += int(terminal[0] > terminal[1]); tt["hit_cap"] += int(hit_cap)
        tt = totals[key]
        print(f"  {key}: {tt['episodes']} eps, {tt['decision_rows']} decision rows, "
              f"mean steps {tt['steps']/tt['episodes']:.0f}, hit_cap {tt['hit_cap']}/{tt['episodes']}, "
              f"win rate {tt['wins']/tt['episodes']:.3f} (provenance only)", flush=True)

    for key, tt in totals.items():
        if tt["decision_rows"] <= 0:
            raise SystemExit(f"REFUSING: cell {key} produced zero decision-bearing rows")
    manifest.write_text(json.dumps({
        "record": f"On-policy student-state set, round {rnd}", "status": "FROZEN_DATASET",
        "utc": _now(), "implements": "ONPOLICY_TEACHER_DISTILLATION_SPEC.json#DESIGN",
        "round": rnd, "generating_student": {"record": rec_path.name,
                                             "sha256": rec["TERMINAL_CHECKPOINT"]["sha256"]},
        "seeds": {f"z{k[0]}_pole{k[1]}": [v[0], v[-1]] for k, v in seeds.items()},
        "device": device, "decision_rows_only": True, "deterministic_actions": True,
        "totals": totals,
        "win_rates_are_provenance_only": "non-sealed seeds; never a gate, never a stopping criterion",
        "shards": shards,
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
