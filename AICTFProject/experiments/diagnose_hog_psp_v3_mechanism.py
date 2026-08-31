"""H-OG-PSP V3 mechanism diagnostic. Implements HOG_PSP_V3_MECHANISM_DIAGNOSTIC_SPEC.json.

Answers ONE question: did V3 learn distinct trajectory-level latent identities?

    Layer 1  state-level, OG-PSP's ruler UNCHANGED, so V3 is comparable to
             V1 (0.006589 nats) and OG-PSP (0.042102 nats)
    Layer 2  PRIMARY -- trajectory identity, both latents on BOTH poles
    Layer 3  teacher-relative ratio, interpretation only

STRICTLY NOT A GATE. It does not select checkpoints, alter the treatment, or decide
whether EVAL runs. The terminal checkpoint goes to EVAL regardless of what this says.

EVAL 11300101..11300132 is never touched.

Run:  python experiments/diagnose_hog_psp_v3_mechanism.py --device cuda
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

import experiments.probe_teacher_trajectory_separability as P

SD = ROOT / "artifacts" / "strategic_demand"
SPEC = SD / "sppo" / "HOG_PSP_V3_MECHANISM_DIAGNOSTIC_SPEC.json"
FROZEN = SD / "sppo" / "HOG_PSP_V3_MODEL_FROZEN.json"
OUT = SD / "sppo" / "HOG_PSP_V3_MECHANISM_DIAGNOSTIC.json"

CALIB = list(range(10_700_097, 10_700_129))
V3_EVAL = range(11_300_101, 11_300_133)
N_BOOT, ALPHA, BOOT_SEED = 20_000, 0.05, 7
POLES = ("A", "B")

V1_CALIB_JSD = 0.006589
OG_PSP_CALIB_JSD = 0.042102


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _paired_ci(values: np.ndarray) -> dict:
    """Paired percentile bootstrap, seed as the resampling unit. Project convention."""
    v = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, len(v), size=(N_BOOT, len(v)))
    boot = v[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"mean": float(v.mean()), "lcb95": float(lo), "ucb95": float(hi), "n": int(len(v))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; this diagnostic is one-shot")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_TRAINING_COMPLETES":
        raise SystemExit(f"REFUSING: diagnostic spec is not frozen: {spec['status']!r}")

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    ck = ROOT / frozen["TERMINAL_CHECKPOINT"]["path"]
    actual = hashlib.sha256(ck.read_bytes()).hexdigest()
    if actual != frozen["TERMINAL_CHECKPOINT"]["sha256"]:
        raise SystemExit(f"REFUSING: checkpoint sha mismatch\n  {actual}")
    if frozen["TERMINAL_RECORD_VALIDITY"]["verdict"].split()[0] != "VALID":
        raise SystemExit("REFUSING: the run was not established VALID")

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
    )
    import experiments.diagnose_oracle_gated_k2_fit_calib as DG
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.trajectory_identity import FrozenDiscriminators, POLE_NAME

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    D = FrozenDiscriminators(verify=True)

    # Sign convention RE-ASSERTED at runtime, never assumed. A silent sklearn class
    # reordering would invert every Layer 2 conclusion while looking entirely plausible.
    for pole in ("A", "B"):
        classes = [int(c) for c in D.models[pole]["clf"].classes_]
        if classes != [P.PI_A, P.PI_B]:
            raise SystemExit(f"REFUSING: D_{pole} classes_ = {classes}, expected [0, 1]; "
                             "the positive=pi_B-like convention would be inverted")
    print(f"H-OG-PSP V3 MECHANISM DIAGNOSTIC  {_now()}")
    print(f"  checkpoint sha verified; run established VALID")
    print(f"  sign convention re-asserted: decision_function positive = pi_B-like")
    print(f"  CALIB {CALIB[0]}..{CALIB[-1]}   V3 EVAL {V3_EVAL.start}..{V3_EVAL.stop-1} SEALED\n",
          flush=True)

    probe = R2.build_env(device, CALIB[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    policy = load_custom_ppo_policy(str(ck), obs_space, act_space, device=device)
    model = policy.model if hasattr(policy, "model") else policy
    model.eval()

    # ------------------------------------------------- LAYER 1: OG-PSP's ruler
    print("  LAYER 1  state-level, unchanged ruler", flush=True)
    calib = DG.load_split(10_700_097, 10_700_128)
    n = len(calib["delta"])
    z0a = np.zeros(n); z1a = np.zeros(n); z1b = np.zeros(n); z0b = np.zeros(n); jsd = np.zeros(n)
    for s in range(0, n, 128):
        idx = np.arange(s, min(s + 128, n))
        obs = DG._obs_batch(calib, idx, device)
        pa, pb = calib["pi_a"][idx], calib["pi_b"][idx]
        z0a[idx] = DG._agreement(model, obs, pa, 0, device).cpu().numpy()
        z1a[idx] = DG._agreement(model, obs, pa, 1, device).cpu().numpy()
        z1b[idx] = DG._agreement(model, obs, pb, 1, device).cpu().numpy()
        z0b[idx] = DG._agreement(model, obs, pb, 0, device).cpu().numpy()
        jsd[idx] = DG._jsd_mean(model, obs, device)
    ap_, bp_ = calib["a_preferred"], calib["b_preferred"]
    layer1 = {
        "n_resolvable": n,
        "z0_z1_jsd_nats": float(jsd.mean()),
        "crossed_teacher_match": {
            "A_preferred": {"z0_own": float(z0a[ap_].mean()), "z1_crossed": float(z1a[ap_].mean()),
                            "gap_pp": float((z0a[ap_].mean() - z1a[ap_].mean()) * 100), "n": int(ap_.sum())},
            "B_preferred": {"z1_own": float(z1b[bp_].mean()), "z0_crossed": float(z0b[bp_].mean()),
                            "gap_pp": float((z1b[bp_].mean() - z0b[bp_].mean()) * 100), "n": int(bp_.sum())},
        },
        "reference": {"V1_CALIB_jsd_nats": V1_CALIB_JSD, "OG_PSP_CALIB_jsd_nats": OG_PSP_CALIB_JSD},
    }
    l1a = layer1["crossed_teacher_match"]["A_preferred"]
    l1b = layer1["crossed_teacher_match"]["B_preferred"]
    print(f"    JSD {layer1['z0_z1_jsd_nats']:.6f} nats  "
          f"(V1 {V1_CALIB_JSD:.6f}, OG-PSP {OG_PSP_CALIB_JSD:.6f})")
    print(f"    A-pref gap {l1a['gap_pp']:+.2f} pp   B-pref gap {l1b['gap_pp']:+.2f} pp\n", flush=True)

    # ------------------------------------ LAYER 2: trajectory identity, full cross
    print("  LAYER 2  trajectory identity, both latents on BOTH poles", flush=True)

    def roll(z: int, pole: str, seed: int) -> dict:
        """One deterministic episode, capturing obs_vec and actions for featurisation."""
        env = R2.build_env(device, seed)
        core = env.core
        try:
            policy.fixed_latent_strategy = True
            policy.fixed_latent_strategy_id = int(z)
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
                                       context=f"v3 diag {pole} z{z} seed {seed}")
            vecs, acts = [], []
            for _ in range(R2.MAX_STEPS):
                action, _ = policy.predict(obs, deterministic=True)
                vecs.append(np.asarray(obs["vec"])[0].copy())
                acts.append(np.asarray(action).ravel().copy())
                env.step_async(action)
                obs, _r, done, _info = env.step_wait()
                obs["global_state"] = env.state()
                if bool(np.asarray(done).any()):
                    break
            return {"vec": np.stack(vecs), "act": np.stack(acts)}
        finally:
            env.close()

    scores: dict[tuple[int, str], dict[int, float]] = {}
    for z in (0, 1):
        for pole in POLES:
            per_seed = {}
            for seed in CALIB:
                ep = roll(z, pole, seed)
                feats = P.featurise({"vec": ep["vec"], "act": ep["act"]})
                art = D.models[pole]
                x = art["scaler"].transform(feats.reshape(1, -1))
                per_seed[seed] = float(art["clf"].decision_function(x)[0])   # RAW logit
            scores[(z, pole)] = per_seed
            m = float(np.mean(list(per_seed.values())))
            print(f"    z{z} | Pole {pole}: mean raw D score {m:+.4f}", flush=True)

    def gap(pole: str) -> dict:
        z0 = np.array([scores[(0, pole)][s] for s in CALIB])
        z1 = np.array([scores[(1, pole)][s] for s in CALIB])
        return _paired_ci(z1 - z0)          # positive = z1 more pi_B-like than z0

    delta_A, delta_B = gap("A"), gap("B")
    delta_A["passes"] = delta_A["lcb95"] > 0
    delta_B["passes"] = delta_B["lcb95"] > 0

    def label_rates() -> dict:
        out = {}
        for z in (0, 1):
            for pole in POLES:
                v = np.array([scores[(z, pole)][s] for s in CALIB])
                out[f"z{z}|{pole}"] = {
                    "mean_raw_score": float(v.mean()),
                    "pct_classified_pi_B_like": float((v > 0).mean() * 100),
                    "pct_classified_pi_A_like": float((v <= 0).mean() * 100),
                }
        for pole in POLES:
            z0 = np.array([scores[(0, pole)][s] for s in CALIB])
            z1 = np.array([scores[(1, pole)][s] for s in CALIB])
            out[f"pct_seeds_different_label_pole_{pole}"] = float(((z0 > 0) != (z1 > 0)).mean() * 100)
        return out

    layer2 = {"delta_tau_A": delta_A, "delta_tau_B": delta_B,
              "descriptive": label_rates(),
              "per_seed_scores": {f"z{z}|{p}": scores[(z, p)] for z in (0, 1) for p in POLES}}
    print(f"    delta_tau_A {delta_A['mean']:+.4f} [{delta_A['lcb95']:+.4f}, {delta_A['ucb95']:+.4f}]"
          f"  {'PASS' if delta_A['passes'] else 'fail'}")
    print(f"    delta_tau_B {delta_B['mean']:+.4f} [{delta_B['lcb95']:+.4f}, {delta_B['ucb95']:+.4f}]"
          f"  {'PASS' if delta_B['passes'] else 'fail'}\n", flush=True)

    # -------------------------------- LAYER 3: teacher reference, context only
    print("  LAYER 3  teacher reference (context only)", flush=True)
    teacher_eps = P.load(10_700_097, 10_700_128)
    teacher_gap = {}
    for pole_idx, pole in ((P.POLE_A, "A"), (P.POLE_B, "B")):
        art = D.models[pole]
        vals = {}
        for pol_id in (P.PI_A, P.PI_B):
            sel = [e for e in teacher_eps if e["pole"] == pole_idx and e["policy"] == pol_id]
            f = np.stack([P.featurise(e) for e in sel])
            vals[pol_id] = float(art["clf"].decision_function(art["scaler"].transform(f)).mean())
        teacher_gap[pole] = {"pi_A_mean": vals[P.PI_A], "pi_B_mean": vals[P.PI_B],
                             "delta_tau_teacher": vals[P.PI_B] - vals[P.PI_A],
                             "n_episodes": len(sel)}
    layer3 = {"teacher": teacher_gap,
              "ratio_R": {"A": (delta_A["mean"] / teacher_gap["A"]["delta_tau_teacher"]
                                if teacher_gap["A"]["delta_tau_teacher"] else None),
                          "B": (delta_B["mean"] / teacher_gap["B"]["delta_tau_teacher"]
                                if teacher_gap["B"]["delta_tau_teacher"] else None)},
              "R_equals_1_is_NOT_required": True}
    print(f"    teacher delta_tau  A {teacher_gap['A']['delta_tau_teacher']:+.4f}   "
          f"B {teacher_gap['B']['delta_tau_teacher']:+.4f}")
    print(f"    R                  A {layer3['ratio_R']['A']:.4f}   B {layer3['ratio_R']['B']:.4f}\n",
          flush=True)

    # ------------------------------------------------- preregistered reading
    if delta_A["passes"] and delta_B["passes"]:
        reading = "TRAJECTORY_IDENTITY_CONFIRMED"
    elif (delta_A["passes"] or delta_B["passes"]) or (delta_A["mean"] > 0 and delta_B["mean"] > 0):
        reading = "TRAJECTORY_IDENTITY_PARTIAL"
    else:
        reading = "TRAJECTORY_IDENTITY_NOT_CONFIRMED"
    meaning = spec["PREREGISTERED_READINGS"][reading]

    OUT.write_text(json.dumps({
        "record": "H-OG-PSP V3 mechanism diagnostic",
        "status": "FROZEN_RESULT", "one_shot": True, "utc": _now(),
        "implements": "HOG_PSP_V3_MECHANISM_DIAGNOSTIC_SPEC.json (frozen 8b29fc5c)",
        "checkpoint": {"path": frozen["TERMINAL_CHECKPOINT"]["path"], "sha256": actual},
        "STRICTLY_NOT_A_GATE": ("Does not select checkpoints, alter the treatment, or decide "
                                "whether EVAL runs. The terminal checkpoint goes to EVAL "
                                "regardless of this reading."),
        "READING": reading, "meaning": meaning,
        "sign_convention_reasserted_at_runtime": "decision_function positive = pi_B-like; classes_ verified [0,1] for both D",
        "LAYER_1_state_level": layer1,
        "LAYER_2_trajectory_identity": layer2,
        "LAYER_3_teacher_reference": layer3,
        "rollouts": {"seeds": [CALIB[0], CALIB[-1]], "n_seeds": len(CALIB),
                     "cells": ["z0|A", "z1|A", "z0|B", "z1|B"],
                     "episodes": 4 * len(CALIB),
                     "why_full_cross": ("scoring only z0-on-A and z1-on-B would tangle opponent "
                                        "back into identity; the probe measured that shortcut "
                                        "at exactly 1.0000")},
        "bootstrap": {"procedure": "paired percentile bootstrap over seed-level scores",
                      "samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOT_SEED, "unit": "CALIB seed"},
        "feature_space_note": ("N_ACTION_BINS=48 truncates head 1's alphabet (0..49). D was "
                               "FITTED under this truncation and reached 0.9531/0.9688. The "
                               "diagnostic reproduces the frozen feature space exactly; it was "
                               "NOT altered now that the truncation is known."),
        "V3_EVAL_touched": False,
        "authorizes": "nothing; opening EVAL 11300101..11300132 requires a separate PI decision",
    }, indent=2), encoding="utf-8")

    print(f"  READING: {reading}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
