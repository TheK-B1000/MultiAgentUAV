"""Decision-leverage diagnostic, pass 1: record student rollouts with teacher actions per state.

Implements DECISION_LEVERAGE_DIAGNOSTIC_SPEC.json#PASS_1_RECORD. Rolls the FROZEN offline-TD
student (the exact checkpoint the Compression Crossover seal scored) with forced z, deterministic,
on CUDA -- the same instrument the seal used. At every decision-bearing step it records the
student's action, the pole-matched teacher's action at that same state, and the decision mask,
plus the full student action sequence so pass 2 can replay to any step.

Measurement only. No training, no gate, non-sealed seed block.

Run:  python experiments/dld_pass1_record.py --device cuda
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
SPEC = SD / "DECISION_LEVERAGE_DIAGNOSTIC_SPEC.json"
STUDENT_FROZEN = SD / "TEACHER_DISTILLATION_STUDENT_FROZEN.json"
TEACHER_DATASET = SD / "TEACHER_DISTILLATION_DATASET.json"
OUT_DIR = SD / "decision_leverage" / "pass1"
MANIFEST = SD / "DLD_PASS1_RECORD.json"

N_PER_CELL = 8
N_AGENTS = 2
CELLS = ((0, "A"), (1, "A"), (0, "B"), (1, "B"))
SEED_BASE = 11_940_001
POLE_TEACHER = {"A": "pi_A", "B": "pi_B"}   # pole-matched teacher; z0<->A, z1<->B per the program


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def cell_seeds() -> dict:
    return {c: list(range(SEED_BASE + i * N_PER_CELL, SEED_BASE + (i + 1) * N_PER_CELL))
            for i, c in enumerate(CELLS)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: spec not frozen: {spec['status']!r}")
    if args.device != spec["DEVICE"]["device"]:
        raise SystemExit(f"REFUSING: device {args.device!r} != frozen {spec['DEVICE']['device']!r}")
    if MANIFEST.is_file():
        raise SystemExit(f"REFUSING: {MANIFEST.name} exists; pass 1 is one-shot")

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
    )
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.causal_supervision import decision_mask_from_core
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("REFUSING: cuda required by the frozen spec but unavailable")
    device = args.device

    srec = json.loads(STUDENT_FROZEN.read_text(encoding="utf-8"))
    sck = ROOT / srec["TERMINAL_CHECKPOINT"]["path"]
    if _sha(sck) != srec["TERMINAL_CHECKPOINT"]["sha256"]:
        raise SystemExit("REFUSING: student checkpoint sha mismatch")
    tspec = json.loads(TEACHER_DATASET.read_text(encoding="utf-8"))["teachers"]

    seeds = cell_seeds()
    probe = R2.build_env(device, seeds[CELLS[0]][0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    student = load_custom_ppo_policy(str(sck), obs_space, act_space, device=device)
    teachers = {}
    for n in ("pi_A", "pi_B"):
        p = ROOT / tspec[n]["path"]
        if _sha(p) != tspec[n]["sha256"]:
            raise SystemExit(f"REFUSING: {n} sha mismatch")
        teachers[n] = load_custom_ppo_policy(str(p), obs_space, act_space, device=device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"DLD PASS 1 (record)  {_now()}  device={device}")
    print(f"  student: {sck.name} (sha {srec['TERMINAL_CHECKPOINT']['sha256'][:12]}...)")
    print(f"  {N_PER_CELL} episodes x 4 cells, forced z, deterministic\n", flush=True)

    episodes, totals = [], {}
    for (z, pole) in CELLS:
        key = f"z{z}_pole{pole}"
        tname = POLE_TEACHER[pole]
        totals[key] = {"episodes": 0, "steps": 0, "decision_steps": 0, "mismatch_steps": 0, "wins": 0}
        for ep_i, seed in enumerate(seeds[(z, pole)]):
            env = R2.build_env(device, seed)
            core = env.core
            try:
                student.fixed_latent_strategy = True
                student.fixed_latent_strategy_id = int(z)
                student.reset_strategy()
                teacher = teachers[tname]
                if getattr(teacher, "fixed_latent_strategy", None) is not None:
                    teacher.fixed_latent_strategy = False
                teacher.reset_strategy()
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
                                           context=f"dld pass1 z{z} {pole} seed {seed}")
                s_actions, t_actions, dmasks, mism = [], [], [], []
                steps, terminal = 0, None
                for _ in range(R2.MAX_STEPS):
                    d = np.asarray(decision_mask_from_core(core, N_AGENTS, side="blue").detach().cpu())[0]
                    a_s, _ = student.predict(obs, deterministic=True)
                    a_t, _ = teacher.predict(obs, deterministic=True)
                    s_vec = [int(x) for x in np.asarray(a_s).ravel()]
                    t_vec = [int(x) for x in np.asarray(a_t).ravel()]
                    # mismatch on a free agent's MACRO (even index), per the frozen definition
                    m = bool(any(d[i] and s_vec[i * 2] != t_vec[i * 2] for i in range(N_AGENTS)))
                    s_actions.append(s_vec); t_actions.append(t_vec)
                    dmasks.append(d.copy()); mism.append(m)
                    env.step_async(a_s)
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
            dm = np.asarray(dmasks, dtype=bool)
            mm = np.asarray(mism, dtype=bool)
            dec = dm.any(axis=1)
            f = OUT_DIR / f"{key}_{seed}.npz"
            np.savez_compressed(f, student_actions=np.asarray(s_actions, dtype=np.int64),
                                teacher_actions=np.asarray(t_actions, dtype=np.int64),
                                decision_mask=dm, mismatch=mm, decision_step=dec,
                                blue=np.int64(terminal[0]), red=np.int64(terminal[1]),
                                steps=np.int64(steps))
            episodes.append({"cell": key, "z": z, "pole": pole, "teacher": tname,
                             "episode": ep_i, "seed": seed, "steps": steps,
                             "decision_steps": int(dec.sum()),
                             "mismatch_steps": int((mm & dec).sum()),
                             "blue": terminal[0], "red": terminal[1],
                             "file": str(f.relative_to(ROOT))})
            tt = totals[key]
            tt["episodes"] += 1; tt["steps"] += steps
            tt["decision_steps"] += int(dec.sum()); tt["mismatch_steps"] += int((mm & dec).sum())
            tt["wins"] += int(terminal[0] > terminal[1])
        tt = totals[key]
        rate = tt["mismatch_steps"] / max(1, tt["decision_steps"])
        print(f"  {key} (teacher {tname}): {tt['episodes']} eps, {tt['decision_steps']} decision steps, "
              f"{tt['mismatch_steps']} mismatched ({rate:.3f}), win rate "
              f"{tt['wins']/tt['episodes']:.3f} (provenance only)", flush=True)

    MANIFEST.write_text(json.dumps({
        "record": "DLD pass 1 -- student rollouts with pole-matched teacher actions per state",
        "status": "FROZEN_DATASET", "utc": _now(), "device": device,
        "implements": "DECISION_LEVERAGE_DIAGNOSTIC_SPEC.json#PASS_1_RECORD",
        "student": {"path": srec["TERMINAL_CHECKPOINT"]["path"], "sha256": srec["TERMINAL_CHECKPOINT"]["sha256"]},
        "teachers": tspec, "seeds": {f"z{c[0]}_pole{c[1]}": [v[0], v[-1]] for c, v in seeds.items()},
        "mismatch_definition": "decision-bearing step where the pole-matched teacher's MACRO differs from the student's on at least one agent free to commit",
        "totals": totals, "episodes": episodes,
        "win_rates_are_provenance_only": "non-sealed diagnostic block; no gate",
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
