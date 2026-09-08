"""Phase 0 -- GATE 0B. One shot, on the 96 held-out seeds.

    d_A(o) = V_hat(o, pi_A, A) - V_hat(o, pi_B, A)      on pole-A matched states
    d_B(o) = V_hat(o, pi_B, B) - V_hat(o, pi_A, B)      on pole-B matched states

    requires   LCB95(E[d_A]) > 0   AND   LCB95(E[d_B]) > 0

Both must pass. If either fails: STOP, SPPPO is not built. No alternative
target, no second scorer, no larger fit set.

V_hat is the ANALYTIC expectation under each frozen teacher's masked action
distribution at the SAME restored state -- not a sampled estimate. The
equivalence to brute-force enumeration is unit-tested in
tests/test_qpsi_scorer.py.

BOOTSTRAP UNIT IS THE SEED. The three branch points inside an episode are
correlated, so each draw resamples the 96 held-out seeds with replacement and
retains ALL branch points belonging to each selected seed, exactly as frozen in
PHASE0_DATA_BUDGET_FROZEN.json. The scientific unit is 96 seeds, not 288 states.

This script refuses to run if:
  - a gate0b record already exists (the gate is one-shot),
  - the scorer weights do not hash to the SHA in PHASE0_SCORER_FROZEN.json,
  - a teacher query fails to reproduce recorded deterministic actions.

Run:  python experiments/phase0_gate0b.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase0_scorer_common import (              # noqa: E402
    HELDOUT_SEEDS, SD, assert_teacher_query_valid, load_split, sha256_file,
    teacher_action_dists,
)
from experiments.phase0_fit_qpsi import TEACHERS            # noqa: E402
from rl.scorer.qpsi import QPsi, QPsiConfig                 # noqa: E402

FROZEN = SD / "PHASE0_SCORER_FROZEN.json"
WEIGHTS = SD / "phase0_scorer_data" / "qpsi_frozen.pt"
OUT = SD / "phase0_scorer_data" / "gate0b.json"

N_BOOT, RNG, ALPHA = 20_000, 7, 0.05


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _bootstrap_by_seed(d: np.ndarray, seeds: np.ndarray):
    """Resample SEEDS with replacement, retaining all branch points per seed."""
    uniq = np.unique(seeds)
    by_seed = [d[seeds == s] for s in uniq]
    rng = np.random.default_rng(RNG)
    idx = rng.integers(0, len(uniq), size=(N_BOOT, len(uniq)))
    draws = np.empty(N_BOOT, dtype=np.float64)
    for i in range(N_BOOT):
        draws[i] = np.concatenate([by_seed[j] for j in idx[i]]).mean()
    return float(np.percentile(draws, 100 * ALPHA / 2)), draws, uniq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists. Gate 0B is ONE SHOT.")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    declared = frozen["weights"]["sha256"]
    actual = sha256_file(WEIGHTS)
    if actual != declared:
        raise SystemExit(f"REFUSING: weights sha {actual[:16]} != frozen {declared[:16]}")

    device = a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"
    print(f"PHASE 0 -- GATE 0B  {_now()}")
    print(f"  scorer   sha256 {actual[:16]}...  (matches frozen record)")
    print(f"  held-out {len(HELDOUT_SEEDS)} seeds  {HELDOUT_SEEDS[0]}..{HELDOUT_SEEDS[-1]}")
    print(f"  bootstrap by SEED, n={N_BOOT}, rng={RNG}, alpha={ALPHA}\n")

    ckpt = torch.load(WEIGHTS, map_location=device, weights_only=False)
    model = QPsi(QPsiConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    from rl.custom_ppo import load_custom_ppo_policy
    import experiments.r2_learned_crossover as R2
    probe = R2.build_env(device, HELDOUT_SEEDS[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    teachers = {k: load_custom_ppo_policy(str(v), obs_space, act_space, device=device)
                for k, v in TEACHERS.items()}

    split = load_split(HELDOUT_SEEDS, want_plain=False)
    if split.heldout_opened != len(HELDOUT_SEEDS):
        raise SystemExit("REFUSING: held-out shard count mismatch")
    validity = {tag: assert_teacher_query_valid(teachers[tag], split, ti, device)
                for ti, tag in enumerate(("pi_A", "pi_B"))}
    print(f"  teacher query validity: "
          + ", ".join(f"{k} {v['agent1_argmax_agreement']:.3f}/{v['agent2_argmax_agreement']:.3f}"
                      for k, v in validity.items()))

    # one row per matched state (branch rows duplicate the state per teacher)
    keep = np.nonzero(split.b_teacher == 0)[0]
    t = lambda arr, dt: torch.as_tensor(arr, dtype=dt)
    V = {}
    with torch.no_grad():
        for tag in ("pi_A", "pi_B"):
            vals = []
            for i in range(0, len(keep), 512):
                j = keep[i:i + 512]
                g = t(split.b_grid[j], torch.float32).to(device)
                v = t(split.b_vec[j], torch.float32).to(device)
                am = t(split.b_amask[j], torch.float32).to(device)
                mk = t(split.b_mask[j], torch.float32).to(device)
                pl = t(split.b_pole[j], torch.long).to(device)
                p1, p2 = teacher_action_dists(teachers[tag], g, v, am, mk)
                vals.append(model.expected_value(g, v, am, pl, p1, p2).cpu().numpy())
            V[tag] = np.concatenate(vals)

    pole, seeds = split.b_pole[keep], split.b_seed[keep]
    results, verdicts = {}, {}
    for name, sel, sign in (("A", pole == 0, 1.0), ("B", pole == 1, -1.0)):
        d = sign * (V["pi_A"][sel] - V["pi_B"][sel])
        sd = seeds[sel]
        lcb, draws, uniq = _bootstrap_by_seed(d, sd)
        per_seed = np.array([d[sd == s].mean() for s in uniq])
        results[f"d_{name}"] = {
            "mean": round(float(d.mean()), 6),
            "LCB95": round(lcb, 6),
            "bootstrap_mean": round(float(draws.mean()), 6),
            "n_matched_states": int(sel.sum()),
            "n_seeds": int(len(uniq)),
            "per_seed_mean": round(float(per_seed.mean()), 6),
            "per_seed_sd": round(float(per_seed.std(ddof=1)), 6),
            "frac_seeds_positive": round(float((per_seed > 0).mean()), 6),
        }
        verdicts[f"LCB95(E[d_{name}]) > 0"] = bool(lcb > 0)
        print(f"\n  d_{name}: mean {d.mean():+.6f}   LCB95 {lcb:+.6f}   "
              f"{int(sel.sum())} states / {len(uniq)} seeds   "
              f"{'PASS' if lcb > 0 else 'FAIL'}")

    passed = all(verdicts.values())
    rec = {
        "record": "PHASE0 GATE 0B -- action-conditioned scorer validity",
        "status": "FROZEN_RESULT",
        "utc": _now(),
        "one_shot": True,
        "protocol_ref": "PHASE0_ACTION_CONDITIONED_SCORER_PROTOCOL.json::GATE_0B_action_conditioned_scorer_validity",
        "scorer": {"weights_sha256": actual, "record": "PHASE0_SCORER_FROZEN.json"},
        "target_amendment": "PHASE0_SCORER_TARGET_AMENDMENT.json (terminal win margin)",
        "held_out_block": [HELDOUT_SEEDS[0], HELDOUT_SEEDS[-1], len(HELDOUT_SEEDS)],
        "contrasts": {
            "d_A": "V_hat(o, pi_A, A) - V_hat(o, pi_B, A)",
            "d_B": "V_hat(o, pi_B, B) - V_hat(o, pi_A, B)",
        },
        "expectation": "ANALYTIC over masked teacher action distributions, not sampled",
        "bootstrap": {"unit": "SEED", "n_boot": N_BOOT, "rng": RNG, "alpha": ALPHA,
                      "rule": "resample the 96 seeds with replacement, retain ALL branch points of each"},
        "scientific_unit": "96 seeds, NOT 288 branch states",
        "teacher_query_validity": validity,
        "results": results,
        "requirement": verdicts,
        "VERDICT": "PASS" if passed else "FAIL",
        "consequence": ("Phase 0 complete; SPPPO is scientifically licensed at the frozen "
                        "1M terminal-only budget" if passed else
                        "STOP. SPPPO is not built. No alternative target, no second scorer, "
                        "no larger fit set."),
    }
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n{'='*60}\n  GATE 0B VERDICT: {rec['VERDICT']}\n  -> {OUT}")
    if not passed:
        print("  STOP. SPPPO is not built.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
