"""Fit and FREEZE the two pole-specific trajectory discriminators for H-OG-PSP V3.

Implements HOG_PSP_V3_SPEC.json#AMENDMENT_1_COMPONENT_2_MECHANISM.

    D_A : full-episode trajectory on Pole A -> {pi_A, pi_B}
    D_B : full-episode trajectory on Pole B -> {pi_A, pi_B}

Two separate discriminators, never one shared. A single classifier spanning both poles
can win by rediscovering OP6-vs-OP7, and the probe measured exactly how easy that
shortcut is: the confounded comparison scored 1.0000.

Trained ONCE on frozen SAPPO teacher trajectories, then held FIXED for the whole V3
run. A co-adapted discriminator can be satisfied by arbitrary difference, and our own
data holds the counterexample: OG-PSP's z1 earned a +10.0 pp crossed gap and converted
it to exactly zero payoff.

Feature map and hyperparameters are IDENTICAL to the validated probe -- imported from
it rather than copied, so the two cannot drift apart.

EVAL 11300101..11300132 is never touched. CALIB is used only to report held-out
accuracy; it does not select anything.

Run:  python experiments/fit_trajectory_discriminators.py
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.probe_teacher_trajectory_separability as P

SD = ROOT / "artifacts" / "strategic_demand"
SPEC = SD / "sppo" / "HOG_PSP_V3_SPEC.json"
OUT_DIR = SD / "sppo" / "hog_psp_v3_discriminators"
RECORD = SD / "sppo" / "HOG_PSP_TRAJECTORY_DISCRIMINATORS.json"

# The probe's own bar, reused unchanged. A discriminator that cannot clear the bar the
# probe set is not a usable training target.
MIN_HELD_OUT_ACCURACY = P.TRAJECTORY_SIGNAL_BAR


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fit_one(train: list[dict], test: list[dict], pole_name: str) -> tuple[dict, dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler

    Xtr = np.stack([P.featurise(e) for e in train])
    ytr = np.array([e["policy"] for e in train])
    Xte = np.stack([P.featurise(e) for e in test])
    yte = np.array([e["policy"] for e in test])

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=P.L2_C, max_iter=5000, random_state=P.RANDOM_STATE)
    clf.fit(scaler.transform(Xtr), ytr)
    acc = float(balanced_accuracy_score(yte, clf.predict(scaler.transform(Xte))))

    # classes_ order matters: the runner must map pi_A/pi_B to the right column.
    classes = [int(c) for c in clf.classes_]
    if sorted(classes) != [P.PI_A, P.PI_B]:
        raise SystemExit(f"REFUSING: {pole_name} classes are {classes}, expected [0, 1]")

    return ({"scaler": scaler, "clf": clf, "classes": classes,
             "feature_dim": int(Xtr.shape[1]), "pole": pole_name},
            {"pole": pole_name, "n_train": len(train), "n_test": len(test),
             "held_out_balanced_accuracy": acc,
             "class_index_of_pi_A": classes.index(P.PI_A),
             "class_index_of_pi_B": classes.index(P.PI_B),
             "clears_bar": bool(acc >= MIN_HELD_OUT_ACCURACY)})


def main() -> int:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: V3 spec is not frozen: {spec['status']!r}")
    if "AMENDMENT_1_COMPONENT_2_MECHANISM" not in spec:
        raise SystemExit("REFUSING: the PG-regulariser amendment is not in the frozen spec")
    if RECORD.is_file():
        raise SystemExit(f"REFUSING: {RECORD} exists; discriminators are frozen once")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fit_eps, calib_eps = P.load(P.FIT_LO, P.FIT_HI), P.load(P.CALIB_LO, P.CALIB_HI)
    for name, eps in (("FIT", fit_eps), ("CALIB", calib_eps)):
        seeds = {e["seed"] for e in eps}
        if any(11_300_101 <= s <= 11_300_132 for s in seeds):
            raise SystemExit(f"REFUSING: a V3 EVAL seed entered {name}")
        if any(P.EVAL_LO <= s <= P.EVAL_HI for s in seeds):
            raise SystemExit(f"REFUSING: a legacy EVAL seed entered {name}")

    print(f"H-OG-PSP TRAJECTORY DISCRIMINATORS  {_now()}")
    print(f"  feature map imported from the validated probe: {P.featurise.__module__}")
    print(f"  bar: held-out balanced accuracy >= {MIN_HELD_OUT_ACCURACY}\n")

    failures, reports, artifacts = [], {}, {}
    for pole, pole_name in ((P.POLE_A, "A"), (P.POLE_B, "B")):
        tr = [e for e in fit_eps if e["pole"] == pole]
        te = [e for e in calib_eps if e["pole"] == pole]
        art, rep = fit_one(tr, te, pole_name)
        artifacts[pole_name], reports[pole_name] = art, rep
        if not rep["clears_bar"]:
            failures.append(f"D_{pole_name} held-out accuracy {rep['held_out_balanced_accuracy']:.4f} "
                            f"is below the probe's own bar {MIN_HELD_OUT_ACCURACY}")
        print(f"  D_{pole_name}: held-out {rep['held_out_balanced_accuracy']:.4f}  "
              f"(train {rep['n_train']}, test {rep['n_test']})  "
              f"{'CLEARS' if rep['clears_bar'] else 'BELOW BAR'}")

    if failures:
        for f in failures:
            print(f"    FAIL: {f}")
        raise SystemExit("REFUSING to freeze discriminators that do not clear the bar")

    shas = {}
    for pole_name, art in artifacts.items():
        path = OUT_DIR / f"D_{pole_name}.pkl"
        path.write_bytes(pickle.dumps(art))
        shas[pole_name] = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"  froze D_{pole_name} -> {path.name}  sha256 {shas[pole_name][:16]}")

    RECORD.write_text(json.dumps({
        "record": "H-OG-PSP V3 frozen trajectory discriminators",
        "status": "FROZEN_ARTIFACT", "utc": _now(),
        "implements": "HOG_PSP_V3_SPEC.json#AMENDMENT_1_COMPONENT_2_MECHANISM",
        "two_discriminators_not_one": {
            "rule": "D_A scores Pole A trajectories, D_B scores Pole B trajectories",
            "why": ("A single classifier across both poles can rediscover OP6-vs-OP7. The "
                    "probe's confounded comparison scored 1.0000, measuring exactly how easy "
                    "that shortcut is."),
        },
        "frozen_for_the_whole_run": {
            "rule": "these parameters do not change during PPO",
            "why": ("A co-adapted discriminator can be satisfied by arbitrary difference. "
                    "OG-PSP's z1 earned a +10.0 pp crossed gap and converted it to exactly "
                    "zero payoff -- difference without value is a demonstrated failure mode "
                    "in our own data, not a hypothetical one."),
        },
        "unit_of_credit": "FULL EPISODE; no segment horizon is introduced",
        "feature_map": {
            "source": "imported from probe_teacher_trajectory_separability, not copied",
            "dim": artifacts["A"]["feature_dim"],
            "components": "per-agent obs_vec mean and std, plus action histogram",
        },
        "per_pole": reports,
        "sha256": shas,
        "splits": {"train": [P.FIT_LO, P.FIT_HI], "held_out": [P.CALIB_LO, P.CALIB_HI],
                   "V3_EVAL_touched": False, "legacy_EVAL_touched": False},
        "held_out_selects_nothing": ("CALIB is reported to show the discriminators generalise. "
                                     "No hyperparameter was chosen using it."),
        "authorizes": "nothing; the liveness smoke and a full treatment smoke remain",
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
