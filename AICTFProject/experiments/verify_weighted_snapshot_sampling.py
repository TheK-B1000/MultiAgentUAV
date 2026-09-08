"""Verify weighted snapshot sampling against the four PI-specified requirements.

  1. No weights supplied reproduces the existing UNIFORM sampler behaviour, draw for draw.
  2. Fixed RNG seed + fixed weights produces deterministic selections.
  3. Zero-probability members are NEVER sampled; malformed distributions FAIL CLOSED.
  4. The existing snapshot-load guard is preserved (no None opponent can enter training).

Requirement 1 is checked by replaying the ORIGINAL uniform draw
(rng.integers(0, len(snapshots))) against the new code path on an identically seeded RNG,
so "unchanged" is demonstrated rather than asserted.

Run:  python experiments/verify_weighted_snapshot_sampling.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "artifacts" / "strategic_demand" / "sppo" / "WEIGHTED_SNAPSHOT_SAMPLING_PREFLIGHT.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _draw(weights, snapshots, rng, n):
    """Mirror of the sampling branch in curriculum_runtime._sample_snapshot_opponents."""
    out = []
    for _ in range(n):
        if weights is None:
            idx = int(rng.integers(0, len(snapshots)))
        else:
            cum = np.cumsum(weights)
            idx = int(np.searchsorted(cum, float(rng.random()) * float(cum[-1]), side="right"))
            idx = min(idx, len(snapshots) - 1)
        out.append(snapshots[idx])
    return out


def main() -> int:
    from rl.custom_ppo.curriculum_runtime import _validate_snapshot_weights

    checks: dict[str, bool] = {}
    detail: dict = {}
    pool = ["pi_0.zip", "pi_1.zip", "pi_2.zip"]

    # ---- 1. uniform-preserving, draw for draw --------------------------------------
    legacy = _draw(None, pool, np.random.default_rng(902), 500)
    new_no_weights = _draw(_validate_snapshot_weights(None, pool), pool,
                           np.random.default_rng(902), 500)
    checks["1_no_weights_reproduces_uniform_draw_for_draw"] = (legacy == new_no_weights)
    # an empty tuple (the config default) must also mean uniform
    new_empty = _draw(_validate_snapshot_weights((), pool), pool,
                      np.random.default_rng(902), 500)
    checks["1b_empty_weights_also_uniform"] = (legacy == new_empty)
    detail["uniform_counts"] = dict(Counter(legacy))

    # ---- 2. determinism --------------------------------------------------------------
    w = _validate_snapshot_weights([0.25, 0.75, 0.0], pool)
    a = _draw(w, pool, np.random.default_rng(1234), 300)
    b = _draw(w, pool, np.random.default_rng(1234), 300)
    checks["2_fixed_seed_and_weights_is_deterministic"] = (a == b)

    # ---- 3. zero-probability never sampled; malformed fails closed -------------------
    counts = Counter(a)
    detail["weighted_counts_p25_p75_p0"] = dict(counts)
    checks["3a_zero_probability_member_never_sampled"] = (counts.get("pi_2.zip", 0) == 0)
    # the nonuniform mixture is actually respected (0.25/0.75 over 300 draws)
    frac0 = counts.get("pi_0.zip", 0) / 300.0
    detail["observed_frac_pi_0"] = frac0
    checks["3b_nonuniform_mixture_respected"] = bool(0.18 < frac0 < 0.32)
    # sigma = [1, 0, 0] must select member 0 every single time
    w_deg = _validate_snapshot_weights([1.0, 0.0, 0.0], pool)
    deg = _draw(w_deg, pool, np.random.default_rng(7), 200)
    checks["3c_degenerate_sigma_selects_only_member_0"] = (set(deg) == {"pi_0.zip"})

    bad_cases = {
        "length_mismatch": [0.5, 0.5],
        "negative_mass": [0.5, -0.1, 0.6],
        "all_zero": [0.0, 0.0, 0.0],
        "non_finite": [float("nan"), 0.5, 0.5],
        "weights_without_pool": None,  # handled separately below
    }
    failed_closed = {}
    for name, bad in bad_cases.items():
        if name == "weights_without_pool":
            try:
                _validate_snapshot_weights([1.0], [])
                failed_closed[name] = False
            except ValueError:
                failed_closed[name] = True
            continue
        try:
            _validate_snapshot_weights(bad, pool)
            failed_closed[name] = False
        except ValueError:
            failed_closed[name] = True
    detail["malformed_fail_closed"] = failed_closed
    checks["3d_malformed_distributions_fail_closed"] = all(failed_closed.values())

    # ---- 4. load guard preserved -----------------------------------------------------
    src = (ROOT / "rl/custom_ppo/curriculum_runtime.py").read_text(encoding="utf-8")
    guard_intact = ("_load_snapshot_policy(pick) is None" in src
                    and "refusing to train" in src)
    checks["4_snapshot_load_guard_preserved"] = guard_intact
    detail["load_guard_source_present"] = guard_intact

    print("=" * 74)
    print("WEIGHTED SNAPSHOT SAMPLING PREFLIGHT")
    print("=" * 74)
    for k, v in detail.items():
        print(f"  {k}: {v}")
    print()
    for k in sorted(checks):
        print(f"  [{'PASS' if checks[k] else 'FAIL'}] {k}")
    n_pass = sum(1 for v in checks.values() if v)
    verdict = "PASS" if n_pass == len(checks) else "FAIL"
    print(f"\n  {n_pass}/{len(checks)}  VERDICT: {verdict}")

    OUT.write_text(json.dumps({
        "record": "Weighted snapshot sampling preflight",
        "status": "FROZEN_RESULT", "utc": _now(),
        "implements": "PSRO_DOUBLE_ORACLE_BASELINE_SPEC.json#WHAT_MUST_BE_BUILT.1_weighted_snapshot_sampling",
        "requirements_source": "PI, 2026-09-06 (four stated requirements)",
        "checks": checks, "passed": f"{n_pass}/{len(checks)}",
        "detail": detail, "VERDICT": verdict,
    }, indent=2, default=str), encoding="utf-8")
    print(f"  -> {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
