"""Decision-leverage diagnostic, pass 1 v2: record BOTH teachers at every state.

Implements DECISION_LEVERAGE_DIAGNOSTIC_SPEC.json#PASS_1_RECORD as corrected by
DLD_TEACHER_MATCHING_CLARIFICATION_AMENDMENT.json.

v1 compared each latent against the POLE-matched teacher, so in the two off-diagonal cells it
measured the intended strategic difference between the latents rather than student imitation
error. v2 stores BOTH pi_A's and pi_B's argmax actions at every recorded state, so:

    LATENT-matched (the quantity the diagnostic is about):  z0 -> pi_A, z1 -> pi_B, either pole
    POLE-matched   (what v1 recorded, kept for continuity): Pole A -> pi_A, Pole B -> pi_B

are both derivable afterwards and the ambiguity cannot recur on this data.

Same non-sealed seeds as v1 (11940001..11940032), same frozen student, same device, same forced
z, deterministic -- the trajectories are identical by construction; only the recorded labels are
corrected. Rationale frozen in the amendment BEFORE this rerun.

Run:  python experiments/dld_pass1_record_v2.py --device cuda
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
AMENDMENT = SD / "DLD_TEACHER_MATCHING_CLARIFICATION_AMENDMENT.json"
STUDENT_FROZEN = SD / "TEACHER_DISTILLATION_STUDENT_FROZEN.json"
TEACHER_DATASET = SD / "TEACHER_DISTILLATION_DATASET.json"
OUT_DIR = SD / "decision_leverage" / "pass1_v2"
MANIFEST = SD / "DLD_PASS1_RECORD_V2.json"

N_PER_CELL = 8
N_AGENTS = 2
CELLS = ((0, "A"), (1, "A"), (0, "B"), (1, "B"))
SEED_BASE = 11_940_001
LATENT_TEACHER = {0: "pi_A", 1: "pi_B"}      # corrected: by LATENT, regardless of pole
POLE_TEACHER = {"A": "pi_A", "B": "pi_B"}    # v1's rule, retained for continuity only


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: spec not frozen: {spec['status']!r}")
    amend = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if amend["status"] != "FROZEN_APPEND_ONLY_AMENDMENT":
        raise SystemExit("REFUSING: clarification amendment not frozen before the rerun")
    if args.device != spec["DEVICE"]["device"]:
        raise SystemExit(f"REFUSING: device {args.device!r} != frozen {spec['DEVICE']['device']!r}")
    if MANIFEST.is_file():
        raise SystemExit(f"REFUSING: {MANIFEST.name} exists; v2 is one-shot")

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

    seeds = {c: list(range(SEED_BASE + i * N_PER_CELL, SEED_BASE + (i + 1) * N_PER_CELL))
             for i, c in enumerate(CELLS)}
    probe = R2.build_env(device, seeds[CELLS[0]][0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    student = load_custom_ppo_policy(str(sck), obs_space, act_space, device=device)
    teachers = {}
    for n in ("pi_A", "pi_B"):
        p = ROOT / tspec[n]["path"]
        if _sha(p) != tspec[n]["sha256"]:
            raise SystemExit(f"REFUSING: {n} sha mismatch")
        pol = load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
        if getattr(pol, "fixed_latent_strategy", None) is not None:
            pol.fixed_latent_strategy = False
        teachers[n] = pol

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"DLD PASS 1 v2 (both teachers recorded)  {_now()}  device={device}")
    print(f"  student: {sck.name} (sha {srec['TERMINAL_CHECKPOINT']['sha256'][:12]}...)")
    print(f"  same seeds as v1; LATENT-matched mismatch is the corrected quantity\n", flush=True)

    episodes, totals = [], {}
    for (z, pole) in CELLS:
        key = f"z{z}_pole{pole}"
        lt, pt = LATENT_TEACHER[z], POLE_TEACHER[pole]
        totals[key] = {"episodes": 0, "decision_steps": 0, "mismatch_latent": 0,
                       "mismatch_pole": 0, "wins": 0, "latent_teacher": lt, "pole_teacher": pt}
        for ep_i, seed in enumerate(seeds[(z, pole)]):
            env = R2.build_env(device, seed)
            core = env.core
            try:
                student.fixed_latent_strategy = True
                student.fixed_latent_strategy_id = int(z)
                student.reset_strategy()
                for t in teachers.values():
                    t.reset_strategy()
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
                                           context=f"dld pass1v2 z{z} {pole} seed {seed}")
                s_acts, a_acts, b_acts, dmasks = [], [], [], []
                steps, terminal = 0, None
                for _ in range(R2.MAX_STEPS):
                    d = np.asarray(decision_mask_from_core(core, N_AGENTS, side="blue").detach().cpu())[0]
                    a_s, _ = student.predict(obs, deterministic=True)
                    a_a, _ = teachers["pi_A"].predict(obs, deterministic=True)
                    a_b, _ = teachers["pi_B"].predict(obs, deterministic=True)
                    s_acts.append([int(x) for x in np.asarray(a_s).ravel()])
                    a_acts.append([int(x) for x in np.asarray(a_a).ravel()])
                    b_acts.append([int(x) for x in np.asarray(a_b).ravel()])
                    dmasks.append(d.copy())
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

            S = np.asarray(s_acts, dtype=np.int64)
            A = np.asarray(a_acts, dtype=np.int64)
            B = np.asarray(b_acts, dtype=np.int64)
            dm = np.asarray(dmasks, dtype=bool)
            dec = dm.any(axis=1)

            def macro_mismatch(T):
                out = np.zeros(S.shape[0], dtype=bool)
                for i in range(N_AGENTS):
                    out |= dm[:, i] & (S[:, i * 2] != T[:, i * 2])
                return out

            mm_lat = macro_mismatch(A if lt == "pi_A" else B)
            mm_pole = macro_mismatch(A if pt == "pi_A" else B)
            f = OUT_DIR / f"{key}_{seed}.npz"
            np.savez_compressed(f, student_actions=S, teacher_A_actions=A, teacher_B_actions=B,
                                decision_mask=dm, decision_step=dec,
                                mismatch_latent=mm_lat, mismatch_pole=mm_pole,
                                blue=np.int64(terminal[0]), red=np.int64(terminal[1]),
                                steps=np.int64(steps))
            episodes.append({"cell": key, "z": z, "pole": pole, "latent_teacher": lt,
                             "pole_teacher": pt, "episode": ep_i, "seed": seed, "steps": steps,
                             "decision_steps": int(dec.sum()),
                             "mismatch_latent": int((mm_lat & dec).sum()),
                             "mismatch_pole": int((mm_pole & dec).sum()),
                             "blue": terminal[0], "red": terminal[1],
                             "file": str(f.relative_to(ROOT))})
            tt = totals[key]
            tt["episodes"] += 1; tt["decision_steps"] += int(dec.sum())
            tt["mismatch_latent"] += int((mm_lat & dec).sum())
            tt["mismatch_pole"] += int((mm_pole & dec).sum())
            tt["wins"] += int(terminal[0] > terminal[1])
        tt = totals[key]
        rl_ = tt["mismatch_latent"] / max(1, tt["decision_steps"])
        rp = tt["mismatch_pole"] / max(1, tt["decision_steps"])
        print(f"  {key}: {tt['decision_steps']} decision steps | LATENT-matched vs {lt}: "
              f"{tt['mismatch_latent']} ({rl_:.4f}) | pole-matched vs {pt}: {tt['mismatch_pole']} ({rp:.4f})",
              flush=True)

    MANIFEST.write_text(json.dumps({
        "record": "DLD pass 1 v2 -- both teachers recorded at every state",
        "status": "FROZEN_DATASET", "utc": _now(), "device": device,
        "implements": "DECISION_LEVERAGE_DIAGNOSTIC_SPEC.json#PASS_1_RECORD as corrected by DLD_TEACHER_MATCHING_CLARIFICATION_AMENDMENT.json",
        "supersedes_for_analysis": "DLD_PASS1_RECORD.json (v1, retained; its off-diagonal cells measured latent strategic difference, not imitation error)",
        "student": {"path": srec["TERMINAL_CHECKPOINT"]["path"], "sha256": srec["TERMINAL_CHECKPOINT"]["sha256"]},
        "teachers": tspec,
        "seeds": {f"z{c[0]}_pole{c[1]}": [v[0], v[-1]] for c, v in seeds.items()},
        "seed_reuse_rationale": "same non-sealed block as v1; identical deterministic trajectories regenerated solely to record a label that should have been recorded the first time -- rationale frozen in the amendment before this rerun",
        "definitions": {
            "latent_matched": "z0 -> pi_A, z1 -> pi_B regardless of pole; THE quantity the diagnostic is about",
            "pole_matched": "Pole A -> pi_A, Pole B -> pi_B; v1's rule, retained for continuity",
            "macro_only": "mismatch is on the MACRO head of an agent free to commit"},
        "totals": totals, "episodes": episodes,
        "win_rates_are_provenance_only": "non-sealed diagnostic block; no gate",
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
