"""Read-only audit: does the SAME methodology actually hold at 2v2 / 4v4 / 6v6?

Implements the verification side of
CROSS_SCALE_BASELINE_SUITE_V1_SPEC.json#METHODOLOGY_IDENTITY_2026_09_26, which locks
"one methodology, team size is the only independent variable; only N and k may differ".

It does NOT trust run_config.json or a manifest field. For every teacher/specialist it
LOADS the checkpoint and reads the cfg and parameter names off the real object, because
a manifest can omit a setting entirely (the 6v6 specialists carry no run_config at all)
and `sappo_anchor_lambda` defaults to 0.10 whether or not anchoring ran.

Two axes the user flagged, plus what they imply:

  SA-PPO anchor   gate is cfg["sappo_anchor_dataset"] -- a NON-EMPTY path means the
                  anchor loss was active. The lambda alone proves nothing.
  entity repair   presence of entity_* parameters in the state dict.
  distillation    teachers / allocator / acting policy / stored tensors / device of
                  each scale's frozen dataset manifest.

Writes CROSS_SCALE_METHODOLOGY_IDENTITY_AUDIT.json. Exit 1 if the scales disagree.

Run: python experiments/audit_cross_scale_methodology_identity.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SUITE_SPEC = SD / "CROSS_SCALE_BASELINE_SUITE_V1_SPEC.json"
OUT = SD / "CROSS_SCALE_METHODOLOGY_IDENTITY_AUDIT.json"

#: The KL teachers each scale's distillation dataset actually names, plus the
#: Separated-arm pieces. Paths are the ones pinned in the frozen dataset manifests.
MODELS = {
    "2v2": {
        "pi_A": "artifacts/strategic_demand/sappo_continuation/sappo_pi_A_specialist_1p5M_seed7100001/ckpts/final_sappo_pi_A_specialist_1p5M_seed7100001.zip",
        "pi_B": "artifacts/strategic_demand/sappo_continuation/sappo_pi_B_specialist_1p5M_seed7200001/ckpts/final_sappo_pi_B_specialist_1p5M_seed7200001.zip",
    },
    "4v4": {
        "pi_A": "artifacts/scale_4v4_specialists/pi_A_specialist_4v4_b3_entity_repair/ckpts/final_pi_A_specialist_4v4_b3_entity_repair.zip",
        "pi_B": "artifacts/scale_4v4_specialists/pi_B_specialist_4v4_b3_entity_repair_corrected/ckpts/final_pi_B_specialist_4v4_b3_entity_repair_corrected.zip",
        "pi_D_split": "artifacts/scale_4v4_specialists/pi_A_specialist_4v4_a4_split_attack_defend_v1/ckpts/final_pi_A_specialist_4v4_a4_split_attack_defend_v1.zip",
    },
    "6v6": {
        "pi_A": "artifacts/scale_6v6_specialists/pi_A_specialist_6v6/ckpts/final_pi_A_specialist_6v6.zip",
        "pi_B": "artifacts/scale_6v6_specialists/pi_B_specialist_6v6/ckpts/final_pi_B_specialist_6v6.zip",
    },
}

DATASETS = {
    "2v2": "TEACHER_DISTILLATION_DATASET.json",
    "4v4": "SUITE_DISTILLATION_4V4_DATASET.json",
    "6v6": "TEACHER_DISTILLATION_6V6_DATASET.json",
}

#: Axes that the locked identity says must NOT vary by scale.
INVARIANT_AXES = ("sappo_anchor_on", "entity_repair_on", "total_timesteps")
#: Only the FOUNDATION specialists are comparable across scales on these axes. pi_D is a
#: different stage (200k split PPO) and pooling its budget with a 1M foundation would
#: manufacture a disagreement that is not one. It is reported on its own instead.
FOUNDATION_ROLES = ("pi_A", "pi_B")
DATASET_INVARIANT_AXES = ("allocator", "acting_policy_documented", "stored_entity_tensors",
                          "stored_roles", "decision_rows_only", "n_per_pole")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def inspect_checkpoint(path: Path) -> dict:
    """Load the real object. Never infer a setting from a sibling manifest."""
    import torch

    if not path.is_file():
        return {"present": False, "path": str(path), "error": "MISSING"}
    blob = torch.load(str(path), map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        return {"present": True, "path": str(path), "error": f"unexpected payload {type(blob)!r}"}
    cfg = blob.get("cfg") or {}
    sd = blob.get("model_state_dict") or blob.get("state_dict") or {}
    names = list(sd) if isinstance(sd, dict) else []
    entity = [k for k in names if "entity" in k.lower()]
    anchor_ds = str(cfg.get("sappo_anchor_dataset", "") or "")
    return {
        "present": True,
        "path": str(path.relative_to(ROOT)),
        "sha256": _sha(path),
        # The gate is the dataset path, not the lambda: lambda defaults to 0.10 always.
        "sappo_anchor_on": bool(anchor_ds),
        "sappo_anchor_dataset": anchor_ds,
        "sappo_anchor_lambda": cfg.get("sappo_anchor_lambda"),
        "sappo_anchor_cadence": cfg.get("sappo_anchor_cadence"),
        "entity_repair_on": bool(entity),
        "n_entity_params": len(entity),
        "entity_hidden_dim": cfg.get("entity_hidden_dim"),
        "role_conditioning_enabled": cfg.get("role_conditioning_enabled"),
        "role_k_defend": cfg.get("role_k_defend"),
        "split_attack_defend_enabled": cfg.get("split_attack_defend_enabled"),
        "defend_teacher_lambda": cfg.get("defend_teacher_lambda"),
        "total_timesteps": cfg.get("total_timesteps"),
        "n_params": len(names),
    }


def inspect_dataset(name: str) -> dict:
    p = SD / name
    if not p.is_file():
        return {"present": False, "manifest": name, "error": "MISSING"}
    d = json.loads(p.read_text(encoding="utf-8"))
    tot = d.get("totals") or {}
    n_per_pole = sorted({int(v.get("episodes", -1)) for v in tot.values()}) if tot else []
    return {
        "present": True,
        "manifest": name,
        "sha256": _sha(p),
        "status": d.get("status"),
        "team_size_declared": d.get("team_size", "ABSENT"),
        "allocator": d.get("allocator", "ABSENT"),
        "acting_policy_documented": "acting" in d,
        "stored_entity_tensors": d.get("stored_entity_tensors", "ABSENT"),
        "stored_roles": d.get("stored_roles", "ABSENT"),
        "decision_rows_only": d.get("decision_rows_only"),
        "device": d.get("device"),
        "teachers": {k: (v or {}).get("path") for k, v in (d.get("teachers") or {}).items()},
        "seeds": d.get("seeds"),
        "n_per_pole": n_per_pole[0] if len(n_per_pole) == 1 else n_per_pole,
        "n_shards": len(d.get("shards") or []),
    }


def main() -> int:
    spec = json.loads(SUITE_SPEC.read_text(encoding="utf-8"))
    ident = spec.get("METHODOLOGY_IDENTITY_2026_09_26")
    if ident is None:
        raise SystemExit("FAIL-CLOSED: suite spec carries no METHODOLOGY_IDENTITY_2026_09_26 section")
    if str(ident.get("status")) != "LOCKED":
        raise SystemExit(f"FAIL-CLOSED: methodology identity status {ident.get('status')!r}, expected LOCKED")

    models = {s: {r: inspect_checkpoint(ROOT / p) for r, p in m.items()} for s, m in MODELS.items()}
    datasets = {s: inspect_dataset(n) for s, n in DATASETS.items()}

    print(f"CROSS-SCALE METHODOLOGY IDENTITY AUDIT  {_now()}")
    print("  (settings read from the loaded checkpoints, not from manifests)\n")
    print(f"  {'scale/role':<26}{'anchor':<9}{'entity':<9}{'steps':<11}params")
    for scale, roles in models.items():
        for role, info in roles.items():
            if not info.get("present"):
                print(f"  {scale+'/'+role:<26}{'MISSING -- ' + info['path']}")
                continue
            print(f"  {scale+'/'+role:<26}"
                  f"{'ON' if info['sappo_anchor_on'] else 'off':<9}"
                  f"{'ON' if info['entity_repair_on'] else 'off':<9}"
                  f"{str(info['total_timesteps']):<11}{info['n_params']}")

    print(f"\n  {'scale':<8}{'allocator':<26}{'entity+roles':<14}{'device':<7}manifest")
    for scale, d in datasets.items():
        if not d.get("present"):
            print(f"  {scale:<8}MISSING {d['manifest']}")
            continue
        er = f"{d['stored_entity_tensors']}/{d['stored_roles']}"
        print(f"  {scale:<8}{str(d['allocator']):<26}{er:<14}{str(d['device']):<7}{d['manifest']}")

    # --- disagreements on axes the locked identity says must not vary -------------
    findings = []
    for axis in INVARIANT_AXES:
        vals = {}
        for scale, roles in models.items():
            present = [i for r, i in roles.items()
                       if i.get("present") and r in FOUNDATION_ROLES]
            if not present:
                continue
            vals[scale] = sorted({json.dumps(i.get(axis)) for i in present})
        flat = {s: v for s, v in vals.items()}
        distinct = {tuple(v) for v in flat.values()}
        if len(distinct) > 1:
            findings.append({"axis": axis, "kind": "model_training", "by_scale": flat})
    for axis in DATASET_INVARIANT_AXES:
        flat = {s: d.get(axis) for s, d in datasets.items() if d.get("present")}
        if len({json.dumps(v) for v in flat.values()}) > 1:
            findings.append({"axis": axis, "kind": "distillation_dataset", "by_scale": flat})

    print(f"\n  {len(findings)} axis/axes disagree across scales:")
    for f in findings:
        print(f"    [{f['kind']}] {f['axis']}: {json.dumps(f['by_scale'])}")

    OUT.write_text(json.dumps({
        "record": "cross-scale methodology identity audit (2v2 / 4v4 / 6v6)",
        "utc": _now(),
        "scope": "READ-ONLY. Verifies CROSS_SCALE_BASELINE_SUITE_V1_SPEC.json#METHODOLOGY_IDENTITY_2026_09_26 "
                 "against the artifacts on disk. Changes nothing and authorizes nothing.",
        "method": "every setting is read from the LOADED checkpoint (cfg + parameter names). "
                  "The SA-PPO gate is a non-empty cfg['sappo_anchor_dataset']; sappo_anchor_lambda "
                  "defaults to 0.10 regardless of whether anchoring ran, so it is not the gate. "
                  "entity repair is the presence of entity_* parameters.",
        "identity_source": {"file": SUITE_SPEC.name, "section": "METHODOLOGY_IDENTITY_2026_09_26",
                            "status": ident.get("status")},
        "models_by_scale": models,
        "distillation_datasets_by_scale": datasets,
        "cross_scale_disagreements": findings,
        "identity_holds": not findings,
        "comparison_rule": {
            "foundation_roles_compared": list(FOUNDATION_ROLES),
            "why": "pi_D is the 200k split stage, not a foundation specialist. Pooling its budget "
                   "with a 1M foundation would report a disagreement that is not one. Its own "
                   "settings are still recorded under models_by_scale.",
        },
        "what_this_does_not_check": [
            "that the code path executed at each scale was byte-identical (only recorded cfg/params)",
            "pole overlay equality between scales",
            "whether any disagreement changes a scientific conclusion",
        ],
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0 if not findings else 1


if __name__ == "__main__":
    raise SystemExit(main())
