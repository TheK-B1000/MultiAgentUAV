"""Collect the distillation state set: pi_A deployed on Pole A, pi_B deployed on Pole B.

Implements TEACHER_DISTILLATION_SPEC.json#DATASET. Every pre-action state a teacher visits in
its own deployment is stored with the exact observation dict the policies consume plus the live
decision mask (rl.causal_supervision.decision_mask_from_core -- fatal if the predicate is
missing). Only decision-bearing rows (some agent free to commit) are kept: rows with no free
agent contribute nothing to the loss by construction.

Teacher targets are NOT cached here -- both teachers are queried on every stored state at
training time through the same masked-head code path the student uses.

Runs on CPU by default so it does not contend with an in-flight GPU eval.

Run:  python experiments/collect_distillation_states.py --device cpu
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
SPEC = SD / "TEACHER_DISTILLATION_SPEC.json"
TEACHER_SPEC = SD / "SPECIALIST_BASELINE_SPEC.json"
OUT_DIR = SD / "teacher_distillation" / "states"
MANIFEST = SD / "TEACHER_DISTILLATION_DATASET.json"

N_PER_POLE = 96
SEEDS = {"A": list(range(11_921_001, 11_921_001 + N_PER_POLE)),
         "B": list(range(11_921_101, 11_921_101 + N_PER_POLE))}
N_AGENTS = 2


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if MANIFEST.is_file():
        raise SystemExit(f"REFUSING: {MANIFEST.name} exists; the dataset is collected once")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: spec not frozen: {spec['status']!r}")
    tspec = json.loads(TEACHER_SPEC.read_text(encoding="utf-8"))["MODELS_UNDER_TEST"]

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
    paths = {}
    for name in ("pi_A", "pi_B"):
        ck = ROOT / tspec[name]["path"]
        if not ck.is_file() or _sha(ck) != tspec[name]["sha256"]:
            raise SystemExit(f"REFUSING: {name} checkpoint missing or sha mismatch")
        paths[name] = ck

    probe = R2.build_env(device, SEEDS["A"][0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    teachers = {n: load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
                for n, p in paths.items()}
    for n, pol in teachers.items():
        if getattr(pol.model, "uses_latent_strategy", False):
            raise SystemExit(f"REFUSING: {n} is latent-conditioned; teachers must be single-strategy")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"COLLECT DISTILLATION STATES  {_now()}  device={device}")
    print(f"  {N_PER_POLE} episodes per pole, decision-bearing rows only\n", flush=True)

    shards, totals = [], {"A": {"episodes": 0, "steps": 0, "decision_rows": 0, "wins": 0},
                          "B": {"episodes": 0, "steps": 0, "decision_rows": 0, "wins": 0}}
    for pole in ("A", "B"):
        teacher = teachers["pi_A" if pole == "A" else "pi_B"]
        for ep_i, seed in enumerate(SEEDS[pole]):
            env = R2.build_env(device, seed)
            core = env.core
            try:
                teacher.reset_strategy()
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
                                           context=f"distill collect {pole} seed {seed}")
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
                    action, _ = teacher.predict(obs, deterministic=True)
                    env.step_async(action)
                    obs, _r, done, info = env.step_wait()
                    obs["global_state"] = env.state()
                    steps += 1
                    if bool(np.asarray(done).any()):
                        i0 = info[0] if isinstance(info, (list, tuple)) else info
                        res = (i0 or {}).get("episode_result") or {}
                        terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                        break
                if terminal is None:
                    terminal = (int(core.blue_score[0]), int(core.red_score[0]))
            finally:
                env.close()
            n_rows = len(rows["step"])
            shard = OUT_DIR / f"{pole}_{seed}.npz"
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
                episode=np.full((n_rows,), ep_i, dtype=np.int32),
                seed=np.full((n_rows,), seed, dtype=np.int64),
            )
            shards.append({"pole": pole, "episode": ep_i, "seed": seed, "steps": steps,
                           "decision_rows": n_rows, "blue": terminal[0], "red": terminal[1],
                           "file": str(shard.relative_to(ROOT))})
            tt = totals[pole]
            tt["episodes"] += 1; tt["steps"] += steps; tt["decision_rows"] += n_rows
            tt["wins"] += int(terminal[0] > terminal[1])
            if (ep_i + 1) % 16 == 0:
                print(f"  pole {pole}: {ep_i + 1}/{N_PER_POLE} episodes, "
                      f"{tt['decision_rows']} decision rows so far", flush=True)

    for pole in ("A", "B"):
        tt = totals[pole]
        if tt["decision_rows"] <= 0:
            raise SystemExit(f"REFUSING: pole {pole} produced zero decision-bearing rows")
    MANIFEST.write_text(json.dumps({
        "record": "Teacher-distillation state set", "status": "FROZEN_DATASET", "utc": _now(),
        "implements": "TEACHER_DISTILLATION_SPEC.json#DATASET",
        "teachers": {n: {"path": tspec[n]["path"], "sha256": tspec[n]["sha256"]} for n in paths},
        "seeds": {k: [v[0], v[-1]] for k, v in SEEDS.items()},
        "device": device, "decision_rows_only": True,
        "totals": {p: {**t, "teacher_deployment_win_rate": t["wins"] / max(1, t["episodes"])}
                   for p, t in totals.items()},
        "shards": shards,
    }, indent=2), encoding="utf-8")
    print(f"\n  A: {totals['A']}\n  B: {totals['B']}\n  -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
