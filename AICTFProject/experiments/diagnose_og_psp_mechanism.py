"""OG-PSP mechanism diagnostic: did the paired objective force z to carry identity?

Implements OG_PSP_MECHANISM_DIAGNOSTIC_SPEC.json (frozen 5191370e, amended 0c9c4ff4).

Answers whether OG-PSP's objective actually forced z to encode specialist identity,
SEPARATELY from whether that identity clears the held-out crossover gate. V1 conflated
these: it looked like specialisation had failed to transfer when in fact no
specialisation ever existed.

The crossed comparison is the point. Asking only "does z0 match pi_A?" cannot
distinguish "the latents specialised" from "a shared latent-independent state->action
map that happens to match both teachers". V1 scored 94.7% on that question and still
had z0-z1 JSD of 0.0051 -- because z1 matched pi_A almost as well (93.0%).

STRICTLY NOT A GATE. It does not authorise EVAL, does not alter the model, and cannot
substitute for the frozen crossover gate. A strong mechanism result with a failed
crossover is still CROSSOVER_NOT_CONFIRMED.

EVAL 11200001..11200032 is never opened.

Run:  python experiments/diagnose_og_psp_mechanism.py --device cuda
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

SD = ROOT / "artifacts" / "strategic_demand"
SPEC = SD / "sppo" / "OG_PSP_MECHANISM_DIAGNOSTIC_SPEC.json"
FROZEN = SD / "sppo" / "OG_PSP_MODEL_FROZEN.json"
OUT = SD / "sppo" / "OG_PSP_MECHANISM_DIAGNOSTIC.json"
V2_PARENT = SD / "sppo" / "oracle_gated_k2_v2_bank_data"

# FIT is the TRAINING bank: legacy FIT shards plus the V2 collection.
LEGACY_FIT_LO, LEGACY_FIT_HI = 10_700_001, 10_700_096
V2_FIT_LO, V2_FIT_HI = 11_000_001, 11_000_320
CALIB_LO, CALIB_HI = 10_700_097, 10_700_128
EVAL_LO, EVAL_HI = 11_200_001, 11_200_032
LEGACY_EVAL_LO, LEGACY_EVAL_HI = 10_700_129, 10_700_160

BATCH = 128
LN2 = float(np.log(2.0))

# --------------------------------------------------------------------------------
# Thresholds operationalising the spec's QUALITATIVE readings. Frozen here, in code,
# before the diagnostic is run.
#
# DISCLOSURE, because thresholds chosen after any related observation are not blind:
# while monitoring the run for operational health I saw the training log's
# "[Actor Z] sep_JSD" field, whose tail values were ~0.021. That is a DIFFERENT
# quantity from what this script measures -- sep_JSD is computed on live rollout
# states, this is computed on held-out and bank branch states -- but it is related
# enough that the exposure must be recorded rather than assumed harmless.
#
# The thresholds below are therefore anchored to quantities fixed BEFORE the run:
# V1's measured baseline and the measured teacher contrast, at round multiples.
# "Substantially above V1" is read as an order of magnitude on the contrast scale;
# "still near V1" is read as within 3x of V1.
# --------------------------------------------------------------------------------
V1_JSD_NATS = 0.0051                      # V1 measured, same _jsd_from_logits path
TEACHER_CONTRAST_BITS = 0.3919            # measured pi_A vs pi_B, log2 path
TEACHER_CONTRAST_NATS = TEACHER_CONTRAST_BITS * LN2

JSD_FAILED_AT_NATS = 3.0 * V1_JSD_NATS                  # <= this: still near V1
JSD_WORKED_AT_NATS = 0.25 * TEACHER_CONTRAST_NATS       # >= this: substantial
V1_CROSSED_GAP_PP = 2.0                                 # V1 measured +1.67 / +2.09
GAP_FAILED_AT_PP = 2.0 * V1_CROSSED_GAP_PP              # <= this: still near V1
GAP_WORKED_AT_PP = 5.0 * V1_CROSSED_GAP_PP              # >= this: substantial
CALIB_SURVIVAL_FRAC = 0.50                              # CALIB keeps >= half of FIT


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_fit_split(DG) -> dict:
    """FIT = legacy shards + V2 shards, parsed by V1's own loader.

    V1's load_split hardcodes its data directory, and the V2 collection lives
    elsewhere. Rather than reimplement the parsing (and risk drift from the code
    that produced V1's comparable numbers), the module constant is redirected for
    the V2 range and restored, with the loaded seed range verified afterwards.
    """
    legacy = DG.load_split(LEGACY_FIT_LO, LEGACY_FIT_HI)

    original_data = DG.DATA
    try:
        DG.DATA = V2_PARENT
        new = DG.load_split(V2_FIT_LO, V2_FIT_HI)
    finally:
        DG.DATA = original_data
    if DG.DATA != original_data:
        raise SystemExit("REFUSING: V1 loader's DATA constant was not restored")

    lo, hi = int(new["seed"].min()), int(new["seed"].max())
    if lo < V2_FIT_LO or hi > V2_FIT_HI:
        raise SystemExit(f"REFUSING: V2 load returned seeds [{lo}, {hi}] outside the range")

    merged = {}
    for k in legacy:
        merged[k] = np.concatenate([legacy[k], new[k]])
    merged["a_preferred"] = merged["delta"] < 0
    merged["b_preferred"] = merged["delta"] > 0
    return merged


def assert_eval_untouched(*splits: dict) -> None:
    """Absence of an EVAL seed must be verified, never assumed.

    Both sealed blocks are checked as explicit bands: OG-PSP's own EVAL, and the
    legacy V1 EVAL block, which shares the 10700xxx prefix with FIT and CALIB and
    is therefore the easier of the two to leak into a range by accident.
    """
    bands = {"OG-PSP EVAL": (EVAL_LO, EVAL_HI),
             "legacy V1 EVAL": (LEGACY_EVAL_LO, LEGACY_EVAL_HI)}
    for sp in splits:
        seeds = sp["seed"]
        for name, (lo, hi) in bands.items():
            hit = np.unique(seeds[(seeds >= lo) & (seeds <= hi)])
            if hit.size:
                raise SystemExit(
                    f"REFUSING: {name} seeds entered the diagnostic: {hit[:5].tolist()}")


def score_split(DG, model, split: dict, device: str) -> dict:
    """All four crossed teacher-match cells plus latent divergence."""
    n = len(split["delta"])
    z0_a = np.zeros(n)      # z0 vs pi_A
    z1_a = np.zeros(n)      # z1 vs pi_A   <- crossed
    z1_b = np.zeros(n)      # z1 vs pi_B
    z0_b = np.zeros(n)      # z0 vs pi_B   <- crossed
    jsd = np.zeros(n)

    for start in range(0, n, BATCH):
        idx = np.arange(start, min(start + BATCH, n))
        obs = DG._obs_batch(split, idx, device)
        pi_a, pi_b = split["pi_a"][idx], split["pi_b"][idx]
        z0_a[idx] = DG._agreement(model, obs, pi_a, 0, device).cpu().numpy()
        z1_a[idx] = DG._agreement(model, obs, pi_a, 1, device).cpu().numpy()
        z1_b[idx] = DG._agreement(model, obs, pi_b, 1, device).cpu().numpy()
        z0_b[idx] = DG._agreement(model, obs, pi_b, 0, device).cpu().numpy()
        jsd[idx] = DG._jsd_mean(model, obs, device)

    a_pref, b_pref = split["a_preferred"], split["b_preferred"]

    def _cell(mask, vals):
        return {"n": int(mask.sum()),
                "mean": float(vals[mask].mean()) if mask.any() else None}

    own_a, crossed_a = _cell(a_pref, z0_a), _cell(a_pref, z1_a)
    own_b, crossed_b = _cell(b_pref, z1_b), _cell(b_pref, z0_b)
    gap_a = (own_a["mean"] - crossed_a["mean"]) * 100.0 if own_a["mean"] is not None else None
    gap_b = (own_b["mean"] - crossed_b["mean"]) * 100.0 if own_b["mean"] is not None else None

    return {
        "n_resolvable": n,
        "n_A_preferred": int(a_pref.sum()),
        "n_B_preferred": int(b_pref.sum()),
        "crossed_teacher_match": {
            "A_preferred_states": {
                "z0_match_pi_A_own": own_a,
                "z1_match_pi_A_crossed": crossed_a,
                "gap_pp": gap_a,
            },
            "B_preferred_states": {
                "z1_match_pi_B_own": own_b,
                "z0_match_pi_B_crossed": crossed_b,
                "gap_pp": gap_b,
            },
        },
        "latent_divergence": {
            "z0_z1_jsd_nats": float(jsd.mean()),
            "z0_z1_jsd_bits": float(jsd.mean() / LN2),
            "frac_of_teacher_contrast": float((jsd.mean() / LN2) / TEACHER_CONTRAST_BITS),
            "n": n,
        },
    }


def classify(fit: dict, calib: dict) -> dict:
    """Apply the spec's three preregistered readings via the frozen thresholds."""
    fj = fit["latent_divergence"]["z0_z1_jsd_nats"]
    cj = calib["latent_divergence"]["z0_z1_jsd_nats"]
    fga = fit["crossed_teacher_match"]["A_preferred_states"]["gap_pp"]
    fgb = fit["crossed_teacher_match"]["B_preferred_states"]["gap_pp"]
    cga = calib["crossed_teacher_match"]["A_preferred_states"]["gap_pp"]
    cgb = calib["crossed_teacher_match"]["B_preferred_states"]["gap_pp"]

    fit_gap_min, fit_gap_max = min(fga, fgb), max(fga, fgb)
    calib_gap_min = min(cga, cgb)

    fit_strong = fj >= JSD_WORKED_AT_NATS and fit_gap_min >= GAP_WORKED_AT_PP
    fit_near_v1 = fj <= JSD_FAILED_AT_NATS and fit_gap_max <= GAP_FAILED_AT_PP
    calib_survives = (cj >= CALIB_SURVIVAL_FRAC * fj
                      and calib_gap_min >= CALIB_SURVIVAL_FRAC * fit_gap_min)

    if fit_strong and calib_survives:
        reading = "MECHANISM_WORKED"
        meaning = ("Latent divergence and crossed teacher-match gaps are both "
                   "substantially above V1 and both survive on CALIB. z carries "
                   "specialist identity, and that identity generalises beyond the bank.")
    elif fit_strong and not calib_survives:
        reading = "MEMORISED_PAIRING"
        meaning = ("Strong differentiation on the training bank that collapses on "
                   "unseen states. The network learned the bank's pairing rather than "
                   "specialist identity.")
    elif fit_near_v1:
        reading = "MECHANISM_FAILED"
        meaning = ("Paired supervision did not produce meaningful differentiation "
                   "UNDER THE FROZEN OG-PSP TREATMENT, BANK, ARCHITECTURE, AND 1M "
                   "BUDGET. This says nothing about whether some larger bank or "
                   "longer budget could.")
    else:
        reading = "PARTIAL_DIFFERENTIATION_HUMAN_DECISION_REQUIRED"
        meaning = ("Differentiation is materially above V1 but below the frozen "
                   "'substantial' bar, or the two measures disagree. The spec froze "
                   "qualitative readings; this outcome sits between them and is NOT "
                   "binned autonomously.")

    return {
        "READING": reading,
        "meaning": meaning,
        "fit_strong": fit_strong,
        "fit_near_v1": fit_near_v1,
        "calib_survives": calib_survives,
        "measured": {
            "jsd_nats": {"FIT": fj, "CALIB": cj, "V1": V1_JSD_NATS},
            "crossed_gap_pp": {
                "FIT_A": fga, "FIT_B": fgb, "CALIB_A": cga, "CALIB_B": cgb,
                "V1_A": 1.67, "V1_B": 2.09,
            },
        },
        "thresholds_frozen_before_running": {
            "jsd_worked_at_nats": JSD_WORKED_AT_NATS,
            "jsd_failed_at_nats": JSD_FAILED_AT_NATS,
            "gap_worked_at_pp": GAP_WORKED_AT_PP,
            "gap_failed_at_pp": GAP_FAILED_AT_PP,
            "calib_survival_frac": CALIB_SURVIVAL_FRAC,
        },
    }


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
    ck_path = ROOT / frozen["TERMINAL_CHECKPOINT"]["path"]
    expected = frozen["TERMINAL_CHECKPOINT"]["sha256"]
    actual = _sha256(ck_path)
    if actual != expected:
        raise SystemExit(
            f"REFUSING: checkpoint sha mismatch\n  expected {expected}\n  actual   {actual}")

    import torch
    import experiments.diagnose_oracle_gated_k2_fit_calib as DG
    import experiments.r2_learned_crossover as R2
    from rl.custom_ppo import load_custom_ppo_policy

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    probe = R2.build_env(device, LEGACY_FIT_LO)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    policy = load_custom_ppo_policy(str(ck_path), obs_space, act_space, device=device)
    model = policy.model if hasattr(policy, "model") else policy
    model.eval()

    print(f"OG-PSP MECHANISM DIAGNOSTIC  {_now()}")
    print(f"  terminal checkpoint sha256 verified against OG_PSP_MODEL_FROZEN")
    print(f"  FIT   {LEGACY_FIT_LO}..{LEGACY_FIT_HI} + {V2_FIT_LO}..{V2_FIT_HI}")
    print(f"  CALIB {CALIB_LO}..{CALIB_HI}  (never in the OG-PSP bank)")
    print(f"  EVAL  {EVAL_LO}..{EVAL_HI} sealed\n", flush=True)

    fit = load_fit_split(DG)
    calib = DG.load_split(CALIB_LO, CALIB_HI)
    assert_eval_untouched(fit, calib)
    print(f"  FIT   {len(fit['delta'])} resolvable states", flush=True)
    print(f"  CALIB {len(calib['delta'])} resolvable states\n", flush=True)

    fit_scores = score_split(DG, model, fit, device)
    calib_scores = score_split(DG, model, calib, device)
    verdict = classify(fit_scores, calib_scores)

    record = {
        "record": "OG-PSP mechanism diagnostic (FIT vs CALIB)",
        "status": "FROZEN_RESULT",
        "one_shot": True,
        "utc": _now(),
        "implements": "OG_PSP_MECHANISM_DIAGNOSTIC_SPEC.json",
        "checkpoint": {"path": str(ck_path.relative_to(ROOT)), "sha256": actual},
        "STRICTLY_NOT_A_GATE": (
            "Does not authorise EVAL, does not alter the model, cannot substitute for "
            "the frozen crossover gate. A strong mechanism result with a failed "
            "crossover is still CROSSOVER_NOT_CONFIRMED."),
        "UNITS_CORRECTION": {
            "problem": (
                "The spec places v1_baseline 0.0051 and teacher_reference_contrast "
                "0.3919 side by side and derives '1.3% of available teacher contrast' "
                "from their ratio, but they are in different units."),
            "detail": (
                "_jsd_from_logits (V1's path, reused here) uses natural log, so all "
                "model JSD figures including V1's 0.0051 are NATS. "
                "diagnose_teacher_contrast.py used log2, so 0.3919 is BITS."),
            "consequence": (
                "V1-to-OG-PSP JSD comparison is unaffected: both come from the same "
                "function and are both nats. Only the ratio against teacher contrast "
                "was wrong. Correctly converted, V1 reproduced "
                f"{(V1_JSD_NATS / LN2) / TEACHER_CONTRAST_BITS:.4f} of teacher "
                "contrast, not 0.013."),
            "handling": "Both nats and bits are reported for every JSD figure below.",
        },
        "THRESHOLD_DISCLOSURE": (
            "Thresholds operationalising the spec's qualitative readings were frozen "
            "in code before this ran, anchored to V1's baseline and the measured "
            "teacher contrast. While monitoring the run for operational health I had "
            "seen the training log's [Actor Z] sep_JSD field (~0.021 in the tail), a "
            "different quantity measured on live rollout states rather than these "
            "branch states. Disclosed because thresholds set after any related "
            "observation are not fully blind."),
        "splits": {
            "FIT": {"ranges": [[LEGACY_FIT_LO, LEGACY_FIT_HI], [V2_FIT_LO, V2_FIT_HI]],
                    "n_resolvable": fit_scores["n_resolvable"],
                    "is_the_training_bank": True},
            "CALIB": {"ranges": [[CALIB_LO, CALIB_HI]],
                      "n_resolvable": calib_scores["n_resolvable"],
                      "never_trained_on": True},
            "EVAL": {"ranges": [[EVAL_LO, EVAL_HI]], "touched": False},
        },
        "FIT": fit_scores,
        "CALIB": calib_scores,
        "verdict": verdict,
        "v1_reference": {
            "z0_z1_jsd_nats": V1_JSD_NATS,
            "crossed_gap_A_pp": 1.67,
            "crossed_gap_B_pp": 2.09,
            "v1_eval_verdict": "ORACLE_GATED_K2_CROSSOVER_NOT_CONFIRMED",
        },
        "authorizes": "nothing; opening EVAL requires a separate PI decision",
    }
    OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")

    def _show(name, s):
        ca = s["crossed_teacher_match"]["A_preferred_states"]
        cb = s["crossed_teacher_match"]["B_preferred_states"]
        d = s["latent_divergence"]
        print(f"  {name}")
        print(f"    A-pref  z0->pi_A {ca['z0_match_pi_A_own']['mean']:.4f}   "
              f"z1->pi_A {ca['z1_match_pi_A_crossed']['mean']:.4f}   "
              f"gap {ca['gap_pp']:+.2f} pp")
        print(f"    B-pref  z1->pi_B {cb['z1_match_pi_B_own']['mean']:.4f}   "
              f"z0->pi_B {cb['z0_match_pi_B_crossed']['mean']:.4f}   "
              f"gap {cb['gap_pp']:+.2f} pp")
        print(f"    JSD     {d['z0_z1_jsd_nats']:.6f} nats "
              f"({d['z0_z1_jsd_bits']:.6f} bits, "
              f"{d['frac_of_teacher_contrast']*100:.2f}% of teacher contrast)")

    _show("FIT  ", fit_scores)
    _show("CALIB", calib_scores)
    print(f"\n  V1 reference: JSD {V1_JSD_NATS:.6f} nats, gaps +1.67 / +2.09 pp")
    print(f"\n  READING: {verdict['READING']}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
