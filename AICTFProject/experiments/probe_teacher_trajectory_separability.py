"""Feasibility probe: are pi_A and pi_B trajectories distinguishable at trajectory level?

This is the PREREQUISITE for any trajectory-grounded latent objective (H-OG-PSP).
If a classifier cannot tell the two PROVEN specialists apart from their trajectories,
then there is no trajectory-level strategy signal to impose on z, and the proposed
V3 objective has no target. Answering this costs no rollouts and no EVAL.

THE CONFOUND THIS PROBE EXISTS TO AVOID
---------------------------------------
The tempting comparison is pi_A-on-Pole-A vs pi_B-on-Pole-B. A discriminator can
score near-perfectly on that by identifying THE OPPONENT (OP6+overlay vs OP7) while
learning nothing about specialist strategy. That would hand us a beautiful number
and a worthless training target -- the same class of error as the decision-masking
bug, where a metric looked right and measured something else.

So the question is asked PER POLE, with both specialists on the SAME opponent:

    Pole A:  pi_A trajectories  vs  pi_B trajectories     <- the real question
    Pole B:  pi_A trajectories  vs  pi_B trajectories     <- the real question
    A@A vs B@B                                            <- reported ONLY to
                                                             quantify the trap

Train on FIT seeds, test on held-out CALIB seeds. EVAL is never touched.

A shuffled-label control runs alongside: with ~190 training episodes and 128
features, overfitting is a live risk, and a real signal must beat its own shuffle.

Threshold frozen BEFORE reading, at the project's existing tau=0.70 convention.

Diagnostic. Authorizes nothing. Gates the CONTENT of a future V3 spec.

Run:  python experiments/probe_teacher_trajectory_separability.py
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

SD = ROOT / "artifacts" / "strategic_demand"
DATA = SD / "stratified_regime_data" / "seed_shards"
OUT = SD / "sppo" / "TEACHER_TRAJECTORY_SEPARABILITY_PROBE.json"

FIT_LO, FIT_HI = 10_700_001, 10_700_096
CALIB_LO, CALIB_HI = 10_700_097, 10_700_128
EVAL_LO, EVAL_HI = 10_700_129, 10_700_160

# Encoding verified in collect_stratified_regime_data.py:383-388
PI_A, PI_B = 0, 1
POLE_A, POLE_B = 0, 1

# ---- frozen BEFORE any number was read -------------------------------------
TRAJECTORY_SIGNAL_BAR = 0.70   # held-out balanced accuracy, project tau convention
SHUFFLE_MARGIN = 0.10          # real must beat its own shuffled control by this
N_ACTION_BINS = 48             # plain_action observed range is 0..47
L2_C = 0.1                     # strong regularisation; n ~ 190, d = 128
RANDOM_STATE = 11
# ----------------------------------------------------------------------------


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def episodes_from_shard(path: Path) -> list[dict]:
    """Split a shard's plain_* stream into per-(policy, pole) episodes."""
    out = []
    with np.load(path, allow_pickle=False) as z:
        policy, pole, step = z["plain_policy"], z["plain_pole"], z["plain_step"]
        vec, act = z["plain_obs_vec"], z["plain_action"]
        for p in (PI_A, PI_B):
            for q in (POLE_A, POLE_B):
                sel = np.nonzero((policy == p) & (pole == q))[0]
                if sel.size == 0:
                    continue
                st = step[sel]
                # a step that does not advance marks a new episode
                breaks = np.nonzero(np.diff(st) <= 0)[0] + 1
                for seg in np.split(sel, breaks):
                    if seg.size < 8:            # too short to characterise
                        continue
                    out.append({"policy": int(p), "pole": int(q),
                                "vec": vec[seg][:, 0], "act": act[seg]})
    return out


def featurise(ep: dict) -> np.ndarray:
    """Frozen feature map: per-agent obs_vec mean and std, plus action histogram.

    Deliberately does NOT include pole, seed, episode length, or any identifier
    that could let the classifier shortcut around behaviour.
    """
    v = ep["vec"]                                  # (T, n_agents, 20)
    mean = v.mean(axis=0).ravel()                  # 40
    std = v.std(axis=0).ravel()                    # 40
    hist = np.bincount(ep["act"].ravel(), minlength=N_ACTION_BINS)[:N_ACTION_BINS]
    hist = hist.astype(np.float64) / max(1, hist.sum())        # 48
    return np.concatenate([mean, std, hist])


def load(lo: int, hi: int) -> list[dict]:
    eps = []
    for s in range(lo, hi + 1):
        p = DATA / f"seed_{s}.npz"
        if not p.is_file():
            continue
        for ep in episodes_from_shard(p):
            ep["seed"] = s
            eps.append(ep)
    if not eps:
        raise SystemExit(f"REFUSING: no episodes in [{lo}, {hi}]")
    return eps


def evaluate(train: list[dict], test: list[dict], label: str) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    Xtr = np.stack([featurise(e) for e in train])
    ytr = np.array([e["policy"] for e in train])
    Xte = np.stack([featurise(e) for e in test])
    yte = np.array([e["policy"] for e in test])

    if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
        raise SystemExit(f"REFUSING: {label} has only one class present")

    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=L2_C, max_iter=5000, random_state=RANDOM_STATE)
    clf.fit(sc.transform(Xtr), ytr)
    acc = balanced_accuracy_score(yte, clf.predict(sc.transform(Xte)))

    rng = np.random.default_rng(RANDOM_STATE)
    sh = rng.permutation(ytr)
    clf_s = LogisticRegression(C=L2_C, max_iter=5000, random_state=RANDOM_STATE)
    clf_s.fit(sc.transform(Xtr), sh)
    acc_shuf = balanced_accuracy_score(yte, clf_s.predict(sc.transform(Xte)))

    return {
        "comparison": label,
        "n_train": len(train), "n_test": len(test),
        "train_class_balance": {"pi_A": int((ytr == PI_A).sum()),
                                "pi_B": int((ytr == PI_B).sum())},
        "test_class_balance": {"pi_A": int((yte == PI_A).sum()),
                               "pi_B": int((yte == PI_B).sum())},
        "held_out_balanced_accuracy": float(acc),
        "shuffled_label_control": float(acc_shuf),
        "beats_shuffle_by": float(acc - acc_shuf),
        "clears_bar": bool(acc >= TRAJECTORY_SIGNAL_BAR
                           and acc - acc_shuf >= SHUFFLE_MARGIN),
    }


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; this probe is one-shot")

    fit, calib = load(FIT_LO, FIT_HI), load(CALIB_LO, CALIB_HI)
    for name, eps in (("FIT", fit), ("CALIB", calib)):
        seeds = {e["seed"] for e in eps}
        if any(EVAL_LO <= s <= EVAL_HI for s in seeds):
            raise SystemExit(f"REFUSING: an EVAL seed entered {name}")

    print(f"TEACHER TRAJECTORY SEPARABILITY PROBE  {_now()}")
    print(f"  FIT   {len(fit)} episodes from seeds {FIT_LO}..{FIT_HI}")
    print(f"  CALIB {len(calib)} episodes from seeds {CALIB_LO}..{CALIB_HI}")
    print(f"  bar: held-out balanced accuracy >= {TRAJECTORY_SIGNAL_BAR}, "
          f"beating its shuffle by >= {SHUFFLE_MARGIN}\n", flush=True)

    results = {}
    for pole, name in ((POLE_A, "Pole A"), (POLE_B, "Pole B")):
        tr = [e for e in fit if e["pole"] == pole]
        te = [e for e in calib if e["pole"] == pole]
        results[name] = evaluate(tr, te, f"pi_A vs pi_B, both on {name}")

    # The confounded comparison, reported ONLY to quantify the trap it sets.
    tr_c = [e for e in fit if (e["policy"] == PI_A) == (e["pole"] == POLE_A)]
    te_c = [e for e in calib if (e["policy"] == PI_A) == (e["pole"] == POLE_A)]
    confounded = evaluate(tr_c, te_c, "CONFOUNDED: pi_A@PoleA vs pi_B@PoleB")

    clean = [results["Pole A"], results["Pole B"]]
    both_clear = all(r["clears_bar"] for r in clean)
    any_clear = any(r["clears_bar"] for r in clean)
    verdict = ("TRAJECTORY_SIGNAL_PRESENT" if both_clear else
               "TRAJECTORY_SIGNAL_PARTIAL" if any_clear else
               "NO_TRAJECTORY_SIGNAL")

    meaning = {
        "TRAJECTORY_SIGNAL_PRESENT": (
            "On both poles, held-out trajectories of the two proven specialists are "
            "separable with the opponent held fixed. A trajectory-grounded latent "
            "objective has a real target to aim at."),
        "TRAJECTORY_SIGNAL_PARTIAL": (
            "Separable on one pole but not the other. A trajectory objective would be "
            "training against a signal that exists in only half the strategic space, "
            "which is precisely the asymmetry OG-PSP already produced. This needs a "
            "PI decision before any V3 spec is frozen."),
        "NO_TRAJECTORY_SIGNAL": (
            "The two specialists are NOT separable from their trajectories under this "
            "representation, with the opponent controlled. The proposed V3 trajectory "
            "objective has no target. Either the strategic distinction lives somewhere "
            "this representation does not capture, or it is not a trajectory-level "
            "property at all. Redesign the representation BEFORE spending a 1M run."),
    }[verdict]

    record = {
        "record": "pi_A vs pi_B trajectory separability probe",
        "status": "FROZEN_RESULT", "one_shot": True, "utc": _now(),
        "purpose": ("Prerequisite for any trajectory-grounded latent objective. "
                    "Gates the CONTENT of a future V3 spec; authorizes nothing."),
        "VERDICT": verdict,
        "meaning": meaning,
        "thresholds_frozen_before_reading": {
            "trajectory_signal_bar": TRAJECTORY_SIGNAL_BAR,
            "shuffle_margin": SHUFFLE_MARGIN,
            "l2_C": L2_C,
            "rationale": ("0.70 is the project's existing tau convention. The shuffle "
                          "margin exists because n~190 with d=128 makes overfitting a "
                          "live risk, so a real signal must beat its own shuffle."),
        },
        "confound_control": {
            "why": ("pi_A@PoleA vs pi_B@PoleB lets a classifier win by identifying the "
                    "OPPONENT rather than the strategy. Both specialists are therefore "
                    "compared on the SAME pole."),
            "confounded_comparison_for_reference_only": confounded,
            "is_not_the_result": True,
        },
        "clean_per_pole_results": results,
        "feature_map": {
            "per_agent_obs_vec_mean": 40,
            "per_agent_obs_vec_std": 40,
            "action_histogram": N_ACTION_BINS,
            "total_dim": 80 + N_ACTION_BINS,
            "deliberately_excluded": ["pole", "seed", "episode length", "any identifier"],
        },
        "splits": {"FIT": [FIT_LO, FIT_HI], "CALIB": [CALIB_LO, CALIB_HI],
                   "EVAL_touched": False},
        "authorizes": "nothing; a V3 spec requires a separate PI decision",
    }
    OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")

    for name in ("Pole A", "Pole B"):
        r = results[name]
        print(f"  {name}: held-out {r['held_out_balanced_accuracy']:.4f}  "
              f"shuffled {r['shuffled_label_control']:.4f}  "
              f"(n_train {r['n_train']}, n_test {r['n_test']})  "
              f"{'CLEARS' if r['clears_bar'] else 'below bar'}")
    print(f"\n  [confounded, reference only] {confounded['held_out_balanced_accuracy']:.4f}"
          f"  <- what the opponent shortcut would have scored")
    print(f"\n  VERDICT: {verdict}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
