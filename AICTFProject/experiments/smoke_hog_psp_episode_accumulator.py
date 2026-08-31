"""H-OG-PSP smoke 3: does streaming accumulation reproduce the validated feature map?

The regulariser needs FULL episodes, but episodes span several rollout boundaries. The
accumulator carries only sufficient statistics across those boundaries. That is only
sound if the reconstructed feature vector is the SAME OBJECT the offline probe produced
-- otherwise "trajectory identity" would silently come to mean something else, and D's
0.9531 / 0.9688 validation would no longer apply to what training actually optimises.

Acceptance, on real episodes:

    offline probe featuriser  ==  streaming accumulator      (within tolerance)
    D_A(offline) == D_A(stream),  D_B(offline) == D_B(stream)

Then the cross-boundary lifecycle, feeding one episode in three chunks:

    exactly one flush          correct env index / length
    accumulator resets         next episode starts clean
    no cross-env contamination no duplicate flush on terminal-plus-reset

Diagnostic. Authorizes nothing. EVAL untouched.

Run:  python experiments/smoke_hog_psp_episode_accumulator.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.probe_teacher_trajectory_separability as P
from rl.trajectory_identity import (
    EpisodeFeatureAccumulator, FrozenDiscriminators, LATENT_TO_TARGET,
)

OUT = ROOT / "artifacts" / "strategic_demand" / "sppo" / "HOG_PSP_EPISODE_ACCUMULATOR_SMOKE.json"

# PRIMARY correctness: against a float64 reference, at machine tolerance. This is the
# strictest check available and the one that decides the verdict.
TOL = 1e-12
# SECONDARY: the stored obs_vec is float32, so the ORIGINAL offline featuriser rounds at
# float32 precision. Bound derived a priori from float32 eps (1.19e-7) accumulated over
# a few hundred steps: eps * sqrt(n) * |x|max, generously rounded up. Derived from
# precision, never from the observed differences.
TOL_F32 = 1e-5


def featurise64(vec, act):
    """The probe's own feature map on a float64 copy -- an exact reference."""
    import numpy as _np
    return P.featurise({"vec": _np.asarray(vec, dtype=_np.float64), "act": act})


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def real_episodes(n: int) -> list[dict]:
    """Whole teacher episodes, selected by (policy, pole) together."""
    eps = []
    for seed in range(P.FIT_LO, P.FIT_LO + 30):
        path = P.DATA / f"seed_{seed}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as z:
            policy, pole = z["plain_policy"], z["plain_pole"]
            vec, act = z["plain_obs_vec"], z["plain_action"]
            for pol_id, q, z_id in ((P.PI_A, P.POLE_A, 0), (P.PI_B, P.POLE_B, 1)):
                sel = np.nonzero((policy == pol_id) & (pole == q))[0]
                if sel.size < 24:
                    continue
                eps.append({"vec": vec[sel][:, 0], "act": act[sel],
                            "z": z_id, "pole": int(q), "seed": seed})
                if len(eps) >= n:
                    return eps
    return eps


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; this smoke is one-shot")

    D = FrozenDiscriminators(verify=True)
    episodes = real_episodes(8)
    if len(episodes) < 4:
        raise SystemExit(f"REFUSING: only {len(episodes)} episodes available")
    n_agents, vec_dim = episodes[0]["vec"].shape[1], episodes[0]["vec"].shape[2]
    failures: list[str] = []
    print(f"H-OG-PSP EPISODE ACCUMULATOR SMOKE  {_now()}")
    print(f"  {len(episodes)} real episodes, obs_vec ({n_agents}, {vec_dim})")

    # ---------------------------------------------- 1. exact feature equivalence
    equiv = []
    for i, ep in enumerate(episodes):
        offline32 = P.featurise({"vec": ep["vec"], "act": ep["act"]})
        offline = featurise64(ep["vec"], ep["act"])

        acc = EpisodeFeatureAccumulator(1, n_agents, vec_dim)
        for t in range(len(ep["vec"])):                 # one timestep at a time
            acc.observe(0, ep["vec"][t], ep["act"][t])
        flushed = acc.flush(0)
        stream = flushed["features"]

        max_abs = float(np.max(np.abs(offline - stream)))          # vs float64 reference
        f32_gap = float(np.max(np.abs(offline32 - stream)))        # vs float32 original
        target = LATENT_TO_TARGET[ep["z"]]
        s_off = D.score(offline, ep["pole"], target)
        s_str = D.score(stream, ep["pole"], target)
        equiv.append({
            "episode": i, "n_steps": int(flushed["n_steps"]),
            "steps_match": flushed["n_steps"] == len(ep["vec"]),
            "max_abs_diff_vs_float64_reference": max_abs,
            "exact_to_machine_precision": max_abs <= TOL,
            "max_abs_diff_vs_float32_original": f32_gap,
            "within_float32_bound": f32_gap <= TOL_F32,
            "D_score_float64_reference": s_off, "D_score_streaming": s_str,
            "scores_equal": abs(s_off - s_str) <= TOL,
        })
        if max_abs > TOL:
            failures.append(f"episode {i}: differs from the float64 reference by {max_abs:.3e}")
        if f32_gap > TOL_F32:
            failures.append(f"episode {i}: exceeds the float32 bound at {f32_gap:.3e}")
        if abs(s_off - s_str) > TOL:
            failures.append(f"episode {i}: D scores differ by {abs(s_off - s_str):.3e}")
    worst = max(e["max_abs_diff_vs_float64_reference"] for e in equiv)
    worst32 = max(e["max_abs_diff_vs_float32_original"] for e in equiv)
    print(f"  vs float64 reference: worst {worst:.3e} (machine precision)")
    print(f"  vs float32 original : worst {worst32:.3e} (float32 rounding in the reference)")
    print(f"  D scores equal: {all(e['scores_equal'] for e in equiv)}")

    # ------------------------------------------- 2. across rollout boundaries
    ep = episodes[0]
    T = len(ep["vec"])
    cuts = [T // 3, 2 * T // 3]
    acc = EpisodeFeatureAccumulator(4, n_agents, vec_dim)
    chunks, mid_flush = [(0, cuts[0]), (cuts[0], cuts[1]), (cuts[1], T)], []
    for ci, (lo, hi) in enumerate(chunks):
        for t in range(lo, hi):
            acc.observe(2, ep["vec"][t], ep["act"][t])          # env index 2
        if ci < len(chunks) - 1:
            mid_flush.append(acc.steps(2))                       # still open
    spanning = acc.flush(2)
    offline = featurise64(ep["vec"], ep["act"])
    span_diff = float(np.max(np.abs(offline - spanning["features"])))

    boundary = {
        "chunks": [[int(a), int(b)] for a, b in chunks],
        "steps_open_at_each_boundary": mid_flush,
        "flushes_total": acc.flushes,
        "exactly_one_flush": acc.flushes == 1,
        "correct_env_index": spanning["env_index"] == 2,
        "episode_length": int(spanning["n_steps"]),
        "length_correct": spanning["n_steps"] == T,
        "max_abs_diff_vs_offline": span_diff,
        "identical_to_single_pass": span_diff <= TOL,
    }
    if not boundary["exactly_one_flush"]:
        failures.append(f"{acc.flushes} flushes for one spanning episode; expected 1")
    if not boundary["length_correct"]:
        failures.append(f"spanning episode length {spanning['n_steps']} != {T}")
    if span_diff > TOL:
        failures.append(f"episode split across boundaries differs by {span_diff:.3e}")
    print(f"  across 3 rollout chunks: one flush={boundary['exactly_one_flush']}, "
          f"length {spanning['n_steps']}/{T}, diff {span_diff:.3e}")

    # ------------------------------------- 3. reset, duplicates, contamination
    after_reset_steps = acc.steps(2)
    dup = acc.flush(2)                                   # terminal + reset must not double
    ep2 = episodes[1]
    for t in range(len(ep2["vec"])):
        acc.observe(2, ep2["vec"][t], ep2["act"][t])
    second = acc.flush(2)
    second_clean = float(np.max(np.abs(
        featurise64(ep2["vec"], ep2["act"]) - second["features"])))

    acc2 = EpisodeFeatureAccumulator(4, n_agents, vec_dim)
    for t in range(len(ep["vec"])):
        acc2.observe(0, ep["vec"][t], ep["act"][t])
    for t in range(len(ep2["vec"])):
        acc2.observe(1, ep2["vec"][t], ep2["act"][t])
    iso_a = acc2.flush(0)
    iso_b = acc2.flush(1)
    iso_a_ok = float(np.max(np.abs(featurise64(ep["vec"], ep["act"])
                                   - iso_a["features"]))) <= TOL
    iso_b_ok = float(np.max(np.abs(featurise64(ep2["vec"], ep2["act"])
                                   - iso_b["features"]))) <= TOL

    hygiene = {
        "steps_after_flush": after_reset_steps,
        "resets_to_zero": after_reset_steps == 0,
        "duplicate_flush_returns_none": dup is None,
        "next_episode_clean": second_clean <= TOL,
        "next_episode_diff": second_clean,
        "two_envs_do_not_contaminate": bool(iso_a_ok and iso_b_ok),
    }
    if after_reset_steps != 0:
        failures.append("accumulator did not reset after flush")
    if dup is not None:
        failures.append("a duplicate flush manufactured a phantom episode")
    if second_clean > TOL:
        failures.append(f"next episode carried state; diff {second_clean:.3e}")
    if not (iso_a_ok and iso_b_ok):
        failures.append("state leaked between environments")
    print(f"  hygiene: reset={hygiene['resets_to_zero']}, "
          f"dup_flush_none={hygiene['duplicate_flush_returns_none']}, "
          f"next_clean={hygiene['next_episode_clean']}, "
          f"env_isolation={hygiene['two_envs_do_not_contaminate']}")

    verdict = "PASS" if not failures else "FAIL"
    OUT.write_text(json.dumps({
        "record": "H-OG-PSP smoke 3: streaming episode feature accumulator",
        "status": "SMOKE_RESULT", "utc": _now(), "VERDICT": verdict,
        "question": ("Does streaming accumulation across rollout boundaries reproduce the "
                     "EXACT full-episode feature vector the probe validated and D was fitted "
                     "on?"),
        "why_it_matters": ("If it does not, 'trajectory identity' silently becomes a different "
                           "quantity, and D's 0.9531 / 0.9688 validation stops applying to what "
                           "training actually optimises."),
        "tolerance": {
            "primary_vs_float64_reference": TOL,
            "secondary_vs_float32_original": TOL_F32,
            "why_two": ("The stored obs_vec is float32, so the ORIGINAL offline featuriser "
                        "rounds at float32 precision (eps 1.19e-7) and cannot represent "
                        "agreement below that. Correctness is therefore decided against an "
                        "exact float64 reference at machine tolerance, which is STRICTER, not "
                        "looser. The float32 bound is derived a priori from eps*sqrt(n_steps), "
                        "never from the observed differences."),
        },
        "worst_diff_vs_float32_original": worst32,
        "feature_equivalence": equiv,
        "worst_feature_diff": worst,
        "across_rollout_boundaries": boundary,
        "hygiene": hygiene,
        "no_raw_trajectory_storage": ("Only sum, sum-of-squares, action counts and n are "
                                      "carried; the accumulator is not an episode replay "
                                      "buffer."),
        "rollout_window_shortcut_rejected": ("Featurising a rollout fragment was available and "
                                             "was NOT taken. D was validated on full episodes; "
                                             "a fragment is a different object of unknown "
                                             "separability."),
        "failures": failures,
        "authorizes": "nothing; the combined treatment smoke remains",
        "EVAL_touched": False,
    }, indent=2), encoding="utf-8")

    print(f"\n  VERDICT: {verdict}")
    for f in failures:
        print(f"    FAIL: {f}")
    print(f"  -> {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
