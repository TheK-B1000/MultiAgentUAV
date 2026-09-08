"""OG-PSP paired rehearsal smoke on the REAL 1804-state bank.

Stricter than V1's rehearsal smoke, because V1's could not have detected V1's actual
failure. It verified both latents received *some* pressure; it never verified they
received *contradictory* pressure on the *same* state. That distinction is the entire
mechanism change.

Proves, using the real loader and a fresh step-0 K=2 model:

    bank loaded                 1804 eligible
    same state -> z0 / pi_A     observed
    same state -> z1 / pi_B     observed
    z0 exposures == z1 exposures
    masked teacher disagreement > 0 in the sampled positive control
    tied exposures              0
    legacy one-sided path       0 invocations
    loss aggregation            MEAN, not sum
    intended actor params       move
    unreached params            unchanged

Diagnostic. Authorizes nothing. EVAL untouched.

Run:  python experiments/smoke_paired_rehearsal.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand"
SPEC = SD / "sppo" / "ORACLE_GATED_PAIRED_SPECIALIST_PRESERVATION_SPEC.json"
OUT = SD / "sppo" / "OG_PSP_PAIRED_REHEARSAL_SMOKE.json"

SMOKE_SEED, BATCH_STATES, LR, N_BATCHES = 5150, 64, 1e-4, 6


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from rl.paired_rehearsal import load_paired_bank, paired_rehearsal_loss
    import rl.oracle_rehearsal as V1
    from experiments.oracle_rehearsal_smoke import build_fresh_k2

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN -- SPEC_FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: OG-PSP spec is not frozen: {spec['status']!r}")
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; this smoke is one-shot")

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    failures: list[str] = []

    # --- legacy one-sided path must never fire -------------------------------
    v1_calls = {"n": 0}
    _orig_sample = V1.RehearsalBank.sample

    def _tripwire(self, *a, **k):
        v1_calls["n"] += 1
        return _orig_sample(self, *a, **k)
    V1.RehearsalBank.sample = _tripwire

    try:
        bank = load_paired_bank(include_v2=True, rng_seed=SMOKE_SEED)
        comp = bank.composition()
        print(f"OG-PSP PAIRED REHEARSAL SMOKE  {_now()}")
        print(f"  bank: {comp['eligible']} eligible "
              f"({comp['A_preferred']} A-pref, {comp['B_preferred']} B-pref), "
              f"{comp['tied_excluded_from_sampling']} tied excluded")
        print(f"  masked teacher disagreement: {comp['teacher_disagreement_frac']:.4f}")
        if comp["eligible"] < 1500:
            failures.append(f"bank has {comp['eligible']} eligible, below the frozen 1500")

        cfg, model = build_fresh_k2(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  fresh K=2 model: {n_params:,} params, weights NOT loaded\n")

        before = {n: p.detach().cpu().numpy().copy() for n, p in model.named_parameters()}
        opt = torch.optim.Adam(model.parameters(), lr=LR)

        pairing_ok = True
        pos_control_disagree = 0
        losses = []
        for _ in range(N_BATCHES):
            b = bank.sample(BATCH_STATES)

            # same state -> both latents, with the correct teacher each
            for sid in np.unique(b["state_id"]):
                rows = np.nonzero(b["state_id"] == sid)[0]
                if len(rows) != 2 or sorted(b["z_idx"][rows].tolist()) != [0, 1]:
                    pairing_ok = False
                    break
                r0 = rows[b["z_idx"][rows] == 0][0]
                r1 = rows[b["z_idx"][rows] == 1][0]
                if not np.array_equal(b["teacher_action"][r0], bank.pi_a_action[sid]):
                    pairing_ok = False
                if not np.array_equal(b["teacher_action"][r1], bank.pi_b_action[sid]):
                    pairing_ok = False
            pos_control_disagree += int(b["teachers_disagree"].sum())

            loss = paired_rehearsal_loss(model, b, device=device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))

        if not pairing_ok:
            failures.append("a sampled state did not yield exactly (z0->pi_A, z1->pi_B)")
        if pos_control_disagree == 0:
            failures.append("no sampled state had masked teacher disagreement; the "
                            "positive control found no contrastive signal")

        # --- loss aggregation must be MEAN, not sum --------------------------
        scale = {}
        for n_states in (16, 32, 64):
            probe = load_paired_bank(include_v2=True, rng_seed=SMOKE_SEED + 1)
            with torch.no_grad():
                scale[n_states] = float(
                    paired_rehearsal_loss(model, probe.sample(n_states), device=device))
        vals = list(scale.values())
        ratio = max(vals) / max(1e-9, min(vals))
        if ratio > 2.0:
            failures.append(
                f"loss scales with batch size (ratio {ratio:.2f} across 32/64/128 pairs); "
                "a MEAN must stay flat, a SUM would scale ~4x")

        # --- parameter movement ---------------------------------------------
        after = {n: p.detach().cpu().numpy() for n, p in model.named_parameters()}
        changed = sorted(n for n in after if not np.array_equal(before[n], after[n]))
        unchanged = sorted(n for n in after if n not in changed)
        grad_none = sorted(n for n, p in model.named_parameters() if p.grad is None)
        if not changed:
            failures.append("no parameter changed; the optimizer path is dead")
        if grad_none != unchanged:
            failures.append("unchanged parameters do not match the no-gradient set")

        # --- invariants -------------------------------------------------------
        try:
            bank.assert_invariants()
        except Exception as exc:                                  # noqa: BLE001
            failures.append(str(exc))
        tel = bank.telemetry()
        if v1_calls["n"] != 0:
            failures.append(f"legacy V1 one-sided rehearsal path fired {v1_calls['n']} times")
    finally:
        V1.RehearsalBank.sample = _orig_sample

    print(f"  pairing verified per state      : {pairing_ok}")
    print(f"  positive control disagreements  : {pos_control_disagree}")
    print(f"  z0 / z1 exposures               : {tel['latent_exposures']}")
    print(f"  tied exposures                  : {tel['tied_exposures']}")
    print(f"  legacy one-sided path calls     : {v1_calls['n']}")
    print(f"  loss by pairs {dict(scale)}  ratio {ratio:.2f}")
    print(f"  params moved {len(changed)} of {len(after)}   "
          f"unchanged == no-grad set: {grad_none == unchanged}")

    verdict = "PASS" if not failures else "FAIL"
    OUT.write_text(json.dumps({
        "record": "OG-PSP paired rehearsal smoke on the real bank",
        "status": "SMOKE_RESULT", "utc": _now(), "VERDICT": verdict,
        "implements": "ORACLE_GATED_PAIRED_SPECIALIST_PRESERVATION_SPEC.json",
        "meaning": ("Proves the paired treatment path exists: same state -> two "
                    "contradictory specialist targets -> loss -> optimizer -> parameter "
                    "update. Says NOTHING about whether learning works."),
        "bank": comp,
        "model": {"fresh_step_0": True, "params": n_params, "seed": SMOKE_SEED},
        "pairing_verified_per_state": pairing_ok,
        "positive_control_disagreement_states": pos_control_disagree,
        "latent_exposures": tel["latent_exposures"],
        "latent_exposures_equal": tel["latent_exposures"].get("z0") == tel["latent_exposures"].get("z1"),
        "tied_exposures": tel["tied_exposures"],
        "legacy_one_sided_path_calls": v1_calls["n"],
        "loss_aggregation": {"by_pairs": {str(k * 2): v for k, v in scale.items()},
                             "max_min_ratio": round(ratio, 3),
                             "verdict": "MEAN" if ratio <= 2.0 else "SUM-LIKE"},
        "parameters": {"changed": len(changed), "unchanged": len(unchanged),
                       "changed_groups": sorted({n.split('.')[0] for n in changed}),
                       "unchanged_groups": sorted({n.split('.')[0] for n in unchanged}),
                       "unchanged_equals_no_gradient_set": grad_none == unchanged},
        "loss_first_last": [losses[0], losses[-1]],
        "failures": failures,
        "authorizes": "nothing; the rollout smoke and a fresh PI decision remain",
        "EVAL_touched": False,
    }, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}")
    for f in failures:
        print(f"    FAIL: {f}")
    print(f"  -> {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
