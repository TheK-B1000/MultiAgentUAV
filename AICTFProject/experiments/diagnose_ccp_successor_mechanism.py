"""CCP successor mechanism diagnostic. Implements CCP_SUCCESSOR_MECHANISM_DIAGNOSTIC_SPEC.json.

Answers ONE question: did the causal-supervision mechanism survive training, independent of
and before any payoff reading?

    Layer 1  state-level, V1-V4's ruler UNCHANGED, comparable to the whole ladder
    Layer 2  PRIMARY -- trajectory identity, both latents on BOTH poles
    Layer 3  teacher-relative ratio, interpretation only

Reuses experiments/diagnose_hog_psp_v4_mechanism.py as a live code import -- the same chain
V4 used against V3, V3 against OG-PSP. Nothing is measured here that OG-PSP's diagnostic did
not already measure. STRICTLY NOT A GATE per the frozen spec's rule 3: a weak or surprising
result does not prevent EVAL, which remains a separate explicit decision either way.

EVAL 11600101..11600132 is never touched.

Run:  python experiments/diagnose_ccp_successor_mechanism.py --device cuda
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
import experiments.diagnose_hog_psp_v4_mechanism as V4D      # the ruler, imported not copied

SD = ROOT / "artifacts" / "strategic_demand"
SPEC = SD / "sppo" / "CCP_SUCCESSOR_MECHANISM_DIAGNOSTIC_SPEC.json"
FROZEN = SD / "sppo" / "CCP_SUCCESSOR_MODEL_FROZEN.json"
V3_RESULT = SD / "sppo" / "HOG_PSP_V3_MECHANISM_DIAGNOSTIC.json"
V4_RESULT = SD / "sppo" / "HOG_PSP_V4_MECHANISM_DIAGNOSTIC.json"
PHASE1_RESULT = SD / "sppo" / "CCP_PHASE1_CAUSAL_BRANCHING.json"
OUT = SD / "sppo" / "CCP_SUCCESSOR_MECHANISM_DIAGNOSTIC.json"

# Hard-bound in the SCRIPT as well as the record: editing the record alone cannot
# legitimise a different checkpoint.
TERMINAL_SHA = "7164b662ad3fa6ce130350d10c25ae3a9e61f76e28a30b5911cca9d0c1b7a7f0"

CALIB = V4D.V3D.CALIB                  # 10700097..10700128, imported through the whole chain
POLES = V4D.V3D.POLES
_paired_ci = V4D.V3D._paired_ci        # the same bootstrap code object the whole ladder used
CCP_EVAL = range(11_600_101, 11_600_133)

V1_CALIB_JSD = V4D.V3D.V1_CALIB_JSD
OG_PSP_CALIB_JSD = V4D.V3D.OG_PSP_CALIB_JSD
V3_CALIB_JSD = V4D.V3_CALIB_JSD
V4_CALIB_JSD = 0.08192162030890326      # V4's own measured value; V4D has no such constant


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fail(msg: str) -> None:
    raise SystemExit(f"INTEGRITY DEFECT -- REFUSING: {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; this diagnostic is one-shot")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_MEASUREMENT":
        raise SystemExit(f"REFUSING: diagnostic spec is not frozen: {spec['status']!r}")

    if (V4D.V3D.N_BOOT, V4D.V3D.ALPHA, V4D.V3D.BOOT_SEED) != (20_000, 0.05, 7):
        _fail(f"ruler bootstrap constants drifted: "
              f"{(V4D.V3D.N_BOOT, V4D.V3D.ALPHA, V4D.V3D.BOOT_SEED)}")
    if (CALIB[0], CALIB[-1], len(CALIB)) != (10_700_097, 10_700_128, 32):
        _fail(f"CALIB block drifted: {CALIB[0]}..{CALIB[-1]} n={len(CALIB)}")
    if set(CALIB) & set(CCP_EVAL):
        _fail("CALIB overlaps EVAL 11600101..11600132")

    # --- integrity: wrong checkpoint -----------------------------------------
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen["TERMINAL_CHECKPOINT"]["sha256"] != TERMINAL_SHA:
        _fail("CCP_SUCCESSOR_MODEL_FROZEN.json disagrees with the sha hard-bound in this script")
    ck = ROOT / frozen["TERMINAL_CHECKPOINT"]["path"]
    actual = hashlib.sha256(ck.read_bytes()).hexdigest()
    if actual != TERMINAL_SHA:
        _fail(f"checkpoint bytes sha mismatch\n  on disk  {actual}\n  expected {TERMINAL_SHA}")
    if frozen["TERMINAL_RECORD_VALIDITY"]["verdict"] != "VALID":
        _fail("the successor run was not established VALID")

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
    )
    import experiments.diagnose_oracle_gated_k2_fit_calib as DG
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.trajectory_identity import FrozenDiscriminators

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    # --- integrity: broken discriminator provenance --------------------------
    D = FrozenDiscriminators(verify=True)
    for pole in POLES:
        classes = [int(c) for c in D.models[pole]["clf"].classes_]
        if classes != [P.PI_A, P.PI_B]:
            _fail(f"D_{pole} classes_ = {classes}, expected [0, 1]; the positive=pi_B-like "
                  "convention would be inverted")

    print(f"CCP SUCCESSOR MECHANISM DIAGNOSTIC  {_now()}")
    print(f"  checkpoint sha hard-bound and verified; run established VALID")
    print(f"  D_A/D_B sha verified; sign convention re-asserted: positive = pi_B-like")
    print(f"  CALIB {CALIB[0]}..{CALIB[-1]}   CCP EVAL {CCP_EVAL.start}..{CCP_EVAL.stop-1} SEALED",
          flush=True)

    probe = R2.build_env(device, CALIB[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    policy = load_custom_ppo_policy(str(ck), obs_space, act_space, device=device)
    model = policy.model if hasattr(policy, "model") else policy
    model.eval()

    # --- integrity: wrong architecture loaded (private critic heads must be live) --
    critic = getattr(model, "critic", None)
    if critic is None:
        _fail("the loaded model exposes no .critic; cannot verify the architecture")
    if getattr(critic, "private_z_heads", None) is not True:
        _fail(f"loaded critic private_z_heads = {getattr(critic, 'private_z_heads', None)!r}, "
              "expected True. This checkpoint does not carry the frozen architecture.")
    print(f"  V4-inherited architecture present: private_z_heads=True\n", flush=True)

    # ------------------------------------------------- LAYER 1: unchanged ruler
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
        "reference": {"V1_CALIB_jsd_nats": V1_CALIB_JSD, "OG_PSP_CALIB_jsd_nats": OG_PSP_CALIB_JSD,
                      "V3_CALIB_jsd_nats": V3_CALIB_JSD, "V4_CALIB_jsd_nats": V4_CALIB_JSD},
    }
    l1a = layer1["crossed_teacher_match"]["A_preferred"]
    l1b = layer1["crossed_teacher_match"]["B_preferred"]
    print(f"    JSD {layer1['z0_z1_jsd_nats']:.6f} nats  (V1 {V1_CALIB_JSD:.6f}, "
          f"OG-PSP {OG_PSP_CALIB_JSD:.6f}, V3 {V3_CALIB_JSD:.6f}, V4 0.081922)")
    print(f"    A-pref gap {l1a['gap_pp']:+.2f} pp   B-pref gap {l1b['gap_pp']:+.2f} pp\n", flush=True)

    # ------------------------------------ LAYER 2: trajectory identity, full cross
    print("  LAYER 2  trajectory identity, both latents on BOTH poles", flush=True)
    routing_verified = 0

    def roll(z: int, pole: str, seed: int) -> dict:
        nonlocal routing_verified
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
                                       context=f"ccp successor diag {pole} z{z} seed {seed}")
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

            info = policy.strategy_info()
            if not info:
                _fail(f"strategy_info() empty after z{z}|{pole} seed {seed}")
            if info.get("strategy_fixed") is not True:
                _fail(f"z{z}|{pole} seed {seed}: latent was not forced")
            if int(info["strategy"]) != int(z):
                _fail(f"z{z}|{pole} seed {seed}: requested z{z}, ran z{int(info['strategy'])}")
            if any(int(v) != int(z) for v in info.get("strategy_batch", [])):
                _fail(f"z{z}|{pole} seed {seed}: strategy_batch not uniformly z{z}")
            if not vecs:
                _fail(f"z{z}|{pole} seed {seed}: zero-length episode")
            routing_verified += 1
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
                val = float(art["clf"].decision_function(x)[0])
                if not np.isfinite(val):
                    _fail(f"non-finite D score at z{z}|{pole} seed {seed}")
                per_seed[seed] = val
            scores[(z, pole)] = per_seed
            m = float(np.mean(list(per_seed.values())))
            print(f"    z{z} | Pole {pole}: mean raw D score {m:+.4f}", flush=True)

    if routing_verified != 4 * len(CALIB):
        _fail(f"expected {4 * len(CALIB)} verified episodes, got {routing_verified}")
    D.assert_still_frozen()

    def gap(pole: str) -> dict:
        z0 = np.array([scores[(0, pole)][s] for s in CALIB])
        z1 = np.array([scores[(1, pole)][s] for s in CALIB])
        return _paired_ci(z1 - z0)

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

    # ------------------------------- new interpretive note, descriptive only
    # Consults the Phase 1 causal bank for which segments are near this CALIB rollout's
    # poles, purely as descriptive context. Changes no threshold, no category.
    phase1 = json.loads(PHASE1_RESULT.read_text(encoding="utf-8"))
    segment_signs = {}
    for k, v in phase1["contrasts"].items():
        if v["mode"] == "full_takeover" and v["delta_Q_hat"] != 0:
            segment_signs.setdefault(v["pole"], []).append(
                {"state_id": v["state_id"], "estimand": v["estimand"],
                 "delta_Q_hat": v["delta_Q_hat"]})
    interpretive_note = {
        "applies_regardless_of_the_categorical_reading_above": True,
        "training_objective_was_localised_not_uniform": (
            "unlike V1-V4, this successor was never trained to maximise a uniform A-identity "
            "for z0 or B-identity for z1 across the state space; it was trained on 7 specific "
            "measured causal segments, one of which (11500105|A|83) has a NEGATIVE delta_Q, "
            "meaning z0 was deliberately taught toward pi_B there."),
        "nonzero_segments_by_pole": segment_signs,
        "reading_this_note_supports": (
            "a CALIB-level result weaker than V3/V4's under the SAME unchanged ruler is "
            "consistent with either (a) the mechanism failing to preserve anything, or (b) the "
            "mechanism correctly localising behaviour to measured segments rather than "
            "generalising a global identity the objective never asked for. This diagnostic "
            "cannot distinguish (a) from (b) on CALIB alone; it is recorded as an open "
            "question for the paper, not resolved here."),
    }

    OUT.write_text(json.dumps({
        "record": "CCP successor mechanism diagnostic",
        "status": "FROZEN_RESULT", "one_shot": True, "utc": _now(),
        "implements": "CCP_SUCCESSOR_MECHANISM_DIAGNOSTIC_SPEC.json",
        "checkpoint": {"path": frozen["TERMINAL_CHECKPOINT"]["path"], "sha256": actual,
                       "hard_bound_in_script": True},
        "STRICTLY_NOT_A_GATE": ("Does not select checkpoints, alter the treatment, or decide "
                                "whether EVAL runs. A weak or surprising reading does NOT "
                                "withhold the payoff EVAL, per the frozen spec's rule 3."),
        "READING": reading, "meaning": meaning,
        "INTERPRETIVE_NOTE_DESCRIPTIVE_ONLY": interpretive_note,
        "sign_convention_reasserted_at_runtime": "decision_function positive = pi_B-like; classes_ verified [0,1] for both D",
        "architecture_verified": {"private_z_heads": True},
        "LAYER_1_state_level": layer1,
        "LAYER_2_trajectory_identity": layer2,
        "LAYER_3_teacher_reference": layer3,
        "rollouts": {"seeds": [CALIB[0], CALIB[-1]], "n_seeds": len(CALIB),
                     "cells": ["z0|A", "z1|A", "z0|B", "z1|B"],
                     "episodes": 4 * len(CALIB)},
        "bootstrap": {"procedure": "paired percentile bootstrap over seed-level scores",
                      "samples": V4D.V3D.N_BOOT, "alpha": V4D.V3D.ALPHA, "rng_seed": V4D.V3D.BOOT_SEED,
                      "unit": "CALIB seed",
                      "code_object": "imported through diagnose_hog_psp_v4_mechanism -> "
                                     "diagnose_hog_psp_v3_mechanism, not copied"},
        "CCP_EVAL_touched": False,
        "authorizes": "nothing; opening EVAL 11600101..11600132 requires a separate PI decision",
    }, indent=2), encoding="utf-8")

    print(f"  READING: {reading}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
