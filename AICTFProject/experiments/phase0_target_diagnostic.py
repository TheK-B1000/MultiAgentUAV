"""Phase 0 -- scorer-target diagnostic (TRAIN SEEDS ONLY).

Provenance for PHASE0_TARGET_UNDERSPECIFICATION.json. This is the diagnostic
that revealed the frozen protocol never designated a Q_psi regression target,
and that the three recorded candidates do not agree.

It reads ONLY the 160 training seeds (6500001..6500160). The 96 held-out seeds
are never opened, so Gate 0B remains prospective. The script asserts this rather
than relying on care.

Reported per pole, on MATCHED branch states (both teachers branched from the
identical restored state), with the contrast oriented so POSITIVE always means
"the strategically appropriate teacher for this pole is favoured":

    pole A:  d = X(pi_A) - X(pi_B)
    pole B:  d = X(pi_B) - X(pi_A)

for each candidate target X in {Monte-Carlo return, terminal win margin,
terminal win flag}. Bootstrap is BY SEED (rng 7, n_boot 20000, alpha 0.05),
matching the Gate 0B procedure, because branch states within a seed are
correlated.

This is a TARGET-SELECTION diagnostic computed before any scorer exists. It is
NOT Gate 0B and must never be reported as such.

Run:  python experiments/phase0_target_diagnostic.py
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts/strategic_demand"
COLL = SD / "phase0_scorer_data/full_collection_rebuild_per_branch"
OUT = SD / "PHASE0_TARGET_DIAGNOSTIC_TRAIN_ONLY.json"

SEED_BASE, N_TRAIN = 6_500_001, 160
TRAIN = set(range(SEED_BASE, SEED_BASE + N_TRAIN))
N_BOOT, RNG, ALPHA = 20_000, 7, 0.05


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _boot_lcb(per_seed_means: np.ndarray) -> tuple[float, float]:
    r = np.random.default_rng(RNG)
    idx = r.integers(0, len(per_seed_means), size=(N_BOOT, len(per_seed_means)))
    bs = per_seed_means[idx].mean(1)
    return float(per_seed_means.mean()), float(np.percentile(bs, 100 * ALPHA / 2))


def main() -> int:
    shards = sorted(glob.glob(str(COLL / "seed_shards" / "*.npz")))
    per_seed: dict[str, dict[int, list]] = {"A": {}, "B": {}}
    margin_support, n_opened = Counter(), 0

    for p in shards:
        seed = int(Path(p).stem.split("seed_")[-1])
        if seed not in TRAIN:
            continue                      # held-out never opened
        n_opened += 1
        z = np.load(p, allow_pickle=True)
        rA, rB = z["branch_pi_A_return"].ravel(), z["branch_pi_B_return"].ravel()
        mA = z["branch_pi_A_blue"].astype(int) - z["branch_pi_A_red"].astype(int)
        mB = z["branch_pi_B_blue"].astype(int) - z["branch_pi_B_red"].astype(int)
        wA = (z["branch_pi_A_blue"] > z["branch_pi_A_red"]).astype(int)
        wB = (z["branch_pi_B_blue"] > z["branch_pi_B_red"]).astype(int)
        margin_support.update(mA.tolist()); margin_support.update(mB.tolist())
        for i, pl in enumerate(z["branch_pole"]):
            k = "A" if int(pl) == 0 else "B"
            s = 1.0 if k == "A" else -1.0          # orient toward the apt teacher
            per_seed[k].setdefault(seed, []).append(
                (s * (rA[i] - rB[i]), s * float(mA[i] - mB[i]), s * float(wA[i] - wB[i]))
            )

    if n_opened != N_TRAIN:
        raise SystemExit(f"REFUSING: opened {n_opened} train shards, expected {N_TRAIN}")

    rec = {
        "record": "PHASE0 scorer-target diagnostic (TRAIN SEEDS ONLY)",
        "utc": _now(),
        "purpose": "provenance for the target under-specification finding; NOT Gate 0B",
        "data_scope": f"{SEED_BASE}..{SEED_BASE + N_TRAIN - 1} (160 train seeds)",
        "held_out_seeds_opened": 0,
        "bootstrap": {"unit": "seed", "n_boot": N_BOOT, "rng": RNG, "alpha": ALPHA},
        "orientation": "positive = the strategically appropriate teacher for that pole is favoured",
        "terminal_margin_support_train": dict(sorted(margin_support.items())),
        "poles": {},
    }
    print("PHASE 0 SCORER-TARGET DIAGNOSTIC (train seeds only)")
    print(f"  margin support (train): {dict(sorted(margin_support.items()))}\n")
    for k in ("A", "B"):
        seeds = sorted(per_seed[k])
        arr = {j: np.array([np.mean([v[j] for v in per_seed[k][s]]) for s in seeds])
               for j in range(3)}
        n_states = sum(len(v) for v in per_seed[k].values())
        cell = {"n_matched_states": n_states, "n_seeds": len(seeds), "targets": {}}
        print(f"  pole {k}  (n={n_states} matched states, {len(seeds)} seeds)")
        for j, name in ((0, "monte_carlo_return"), (1, "terminal_win_margin"), (2, "terminal_win_flag")):
            mean, lcb = _boot_lcb(arr[j])
            cell["targets"][name] = {"mean": round(mean, 6), "LCB95": round(lcb, 6),
                                     "sign_correct": bool(lcb > 0)}
            print(f"     {name:20s} mean {mean:+.4f}   LCB95 {lcb:+.4f}   "
                  f"{'correct' if lcb > 0 else 'WRONG WAY'}")
        rec["poles"][k] = cell
        print()

    rec["finding"] = (
        "Monte-Carlo return is oriented AGAINST the strategic payoff ordering on both "
        "poles; terminal win margin and terminal win flag are oriented with it. The "
        "frozen protocol recorded all three but designated none as the regression target."
    )
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
