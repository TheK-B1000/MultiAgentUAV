"""Flatten the Stage-B trajectories into the array rl/causal_sequence_runner.py consumes.

Pure transformation of frozen artifacts: reads CCP_S2_CAUSAL_BANK_ROUTING.json (Stage A) and
CCP_S2_CAUSAL_BANK_STAGE_B.json (Stage B) plus the per-unit .npz trajectories, and emits one
flat array in the EXACT schema CausalSequenceRunner already expects (the same seam V1 through
V4 validated). No rollout, no re-measurement, no choice left open -- every value is copied.

Deliberately unfiltered: every recorded timestep is emitted, including committed ticks and
zero-weight agents. causal_supervision_loss already makes those inert (they contribute to
neither numerator nor denominator), so filtering here would be an unfrozen choice that changes
sampling density without changing the objective.

bank_hash is computed over the frozen routing decisions themselves -- (unit, pole, estimand,
t*, w, r_bank, latent) sorted -- so the training run can recompute it from the frozen artifacts
and refuse a stale or hand-edited array, exactly as the predecessor's segment_bank_hash check
does.

Run:  python experiments/ccp_s2_build_training_array.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
ROUTING = SD / "CCP_S2_CAUSAL_BANK_ROUTING.json"
STAGE_B = SD / "CCP_S2_CAUSAL_BANK_STAGE_B.json"
NPZ_OUT = SD / "ccp_s2_causal_bank.npz"
META_OUT = SD / "CCP_S2_CAUSAL_BANK_ARRAY.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def bank_hash(stage_b: dict) -> str:
    """Deterministic hash of the frozen routing decisions the array encodes."""
    canon = sorted(
        f"{u['unit']}|{u['pole']}|{u['estimand']}|{u['t_star']}|{u['w']:.10g}|"
        f"{u['r_bank']}|{u['latent_supervised']}"
        for u in stage_b["units"])
    return hashlib.sha256("\n".join(canon).encode()).hexdigest()


def main() -> int:
    if META_OUT.is_file():
        raise SystemExit(f"REFUSING: {META_OUT} exists; one-shot")
    routing = json.loads(ROUTING.read_text(encoding="utf-8"))
    stage_b = json.loads(STAGE_B.read_text(encoding="utf-8"))
    for name, rec in (("Stage A routing", routing), ("Stage B", stage_b)):
        if rec["status"] != "FROZEN_RESULT":
            raise SystemExit(f"REFUSING: {name} not frozen: {rec['status']!r}")

    units = stage_b["units"]
    cols: dict = {k: [] for k in ("grid", "vec", "agent_mask", "mask", "global_state",
                                 "actions", "z_idx", "decision_mask", "weight", "segment_idx")}
    seg_meta = []
    for s_i, u in enumerate(units):
        z = np.load(ROOT / u["trajectory_file"])
        n = int(z["decision_mask"].shape[0])
        if n != u["steps"]:
            raise SystemExit(f"REFUSING: {u['unit']} has {n} rows, record says {u['steps']}")
        for k in ("grid", "vec", "agent_mask", "mask", "global_state"):
            cols[k].append(z[k])
        cols["actions"].append(z["teacher_actions"])          # the teacher IS the target
        cols["z_idx"].append(z["z_idx"])
        cols["decision_mask"].append(z["decision_mask"])
        cols["weight"].append(z["weights"])
        cols["segment_idx"].append(np.full((n,), s_i, dtype=np.int64))
        seg_meta.append({"segment_idx": s_i, "unit": u["unit"], "pole": u["pole"],
                         "estimand": u["estimand"], "teacher": u["t_star"], "weight": u["w"],
                         "latent": u["latent_supervised"], "rows": n,
                         "supervised_decisions": u["n_supervised_decisions"],
                         "supervised_with_disagreement":
                             u["n_supervised_decisions_with_disagreement"]})

    arrays = {k: np.concatenate(v, axis=0) for k, v in cols.items()}
    n_rows = arrays["z_idx"].shape[0]
    for k, a in arrays.items():
        if a.shape[0] != n_rows:
            raise SystemExit(f"REFUSING: column {k} has {a.shape[0]} rows, expected {n_rows}")

    sup = arrays["decision_mask"] & (arrays["weight"] > 0)
    if int(sup.sum()) != stage_b["N_usable_commitment_level_supervision_targets"]:
        raise SystemExit(
            f"REFUSING: flattened array has {int(sup.sum())} supervision targets, Stage B "
            f"recorded {stage_b['N_usable_commitment_level_supervision_targets']}")

    np.savez_compressed(NPZ_OUT, **arrays)
    npz_sha = hashlib.sha256(NPZ_OUT.read_bytes()).hexdigest()
    bh = bank_hash(stage_b)

    z0_rows = int((arrays["z_idx"] == 0).sum())
    z1_rows = int((arrays["z_idx"] == 1).sum())
    META_OUT.write_text(json.dumps({
        "record": "CCP-S2 causal bank training array", "status": "FROZEN_ARTIFACT",
        "utc": _now(),
        "derived_from": {"stage_a": "CCP_S2_CAUSAL_BANK_ROUTING.json",
                         "stage_b": "CCP_S2_CAUSAL_BANK_STAGE_B.json"},
        "segment_bank_hash": bh,
        "npz_sha256": npz_sha, "npz": str(NPZ_OUT.relative_to(ROOT)),
        "n_rows": n_rows,
        "nonzero_segments_rolled_out": len(units),
        "total_segments_in_causal_bank": len(routing["units"]),
        "N_usable_commitment_level_supervision_targets": int(sup.sum()),
        "N_supervision_targets_with_teacher_disagreement":
            stage_b["N_supervision_targets_with_teacher_disagreement"],
        "rows_by_latent": {"z0": z0_rows, "z1": z1_rows},
        "unfiltered": "every recorded timestep is present; committed ticks and zero-weight "
                      "agents are inert by construction in causal_supervision_loss, not "
                      "removed here",
        "segments": seg_meta,
    }, indent=2), encoding="utf-8")

    print(f"CCP-S2 CAUSAL BANK ARRAY  {_now()}")
    print(f"  rows                {n_rows}  (z0 {z0_rows} / z1 {z1_rows})")
    print(f"  segments            {len(units)} nonzero of {len(routing['units'])} active")
    print(f"  supervision targets {int(sup.sum())}")
    print(f"  bank_hash           {bh}")
    print(f"  npz sha256          {npz_sha[:16]}...")
    print(f"\n  -> {NPZ_OUT}\n  -> {META_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
