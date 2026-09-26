"""Identity attestation: one methodology at 2v2 / 4v4 / 6v6, emitted as a parity table.

Required by CROSS_SCALE_CANONICAL_RECIPE_V1.json#IDENTITY_ATTESTATION_required. Every row
except N, k, seeds, paths and mechanically-derived tensor dimensions must match across scales.

Three rules this file exists to enforce:

  1. The required values come from the FROZEN recipe record, not from this file. Editing the
     target means editing the frozen record, which is a visible act.
  2. Settings are read from the LOADED checkpoint (cfg + parameter names). A missing
     run_config.json means UNKNOWN -- never "same config". The 6v6 historical specialists
     carry no run_config at all, which is exactly how a manifest-only parity test passes
     while the artifacts disagree.
  3. The SA-PPO gate is a non-empty cfg["sappo_anchor_dataset"]. sappo_anchor_lambda is 0.10
     by PPOConfig default at every scale, so it is evidence of nothing.

Cell values: the measured value, PENDING (artifact not built yet), or UNKNOWN (present but
unreadable). A scale whose artifacts are all PENDING is reported as such and does not
count as a mismatch -- it counts as not yet attestable, which is a different claim.

Exit 0 only if every built scale matches the frozen recipe on every invariant row.

Run: python experiments/attest_cross_scale_identity.py [--json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
RECIPE = SD / "CROSS_SCALE_CANONICAL_RECIPE_V1.json"
SUITE_SPEC = SD / "CROSS_SCALE_BASELINE_SUITE_V1_SPEC.json"
OUT = SD / "CROSS_SCALE_IDENTITY_ATTESTATION.json"

PENDING, UNKNOWN = "PENDING", "UNKNOWN"

#: Per scale: the SUITE artifacts (not the historical ones that share a scale) and the
#: suite dataset manifest. Paths for unbuilt scales are the ones their pipeline promises,
#: so the attestation names what is missing instead of silently skipping it.
SCALES = {
    "2v2": {
        "N": 2, "k": 1,
        "base": {},
        "split": {},
        "dataset": None,
        "pipeline": "SAME_METHODOLOGY_2V2_PORT_PLAN.json (nothing built yet; fresh certification first)",
    },
    "4v4": {
        "N": 4, "k": 2,
        "base": {
            "pi_A": "artifacts/scale_4v4_specialists/pi_A_specialist_4v4_b3_entity_repair/ckpts/final_pi_A_specialist_4v4_b3_entity_repair.zip",
            "pi_B": "artifacts/scale_4v4_specialists/pi_B_specialist_4v4_b3_entity_repair_corrected/ckpts/final_pi_B_specialist_4v4_b3_entity_repair_corrected.zip",
        },
        "split": {
            "pi_D": "artifacts/scale_4v4_specialists/pi_A_specialist_4v4_a4_split_attack_defend_v1/ckpts/final_pi_A_specialist_4v4_a4_split_attack_defend_v1.zip",
        },
        "dataset": "SUITE_DISTILLATION_4V4_DATASET.json",
        "pipeline": "REFERENCE (built)",
    },
    "6v6": {
        "N": 6, "k": 1,
        "base": {
            "pi_A": "artifacts/scale_6v6_specialists/pi_A_specialist_6v6_c2_entity_repair/ckpts/final_pi_A_specialist_6v6_c2_entity_repair.zip",
            "pi_B": "artifacts/scale_6v6_specialists/pi_B_specialist_6v6_c2_entity_repair/ckpts/final_pi_B_specialist_6v6_c2_entity_repair.zip",
        },
        "split": {
            "pi_D": "artifacts/scale_6v6_specialists/pi_A_specialist_6v6_split_defend_k1_v1/ckpts/final_pi_A_specialist_6v6_split_defend_k1_v1.zip",
        },
        "dataset": None,
        "pipeline": "SCHOOL_PC_6V6_LOCKED_PIPELINE.json (final repaired artifacts, not the historical 6v6 specialists)",
    },
}

ROW_ORDER = ("N", "k", "anchor_enabled", "foundation_steps", "entity_repair", "split_steps",
             "Nprime_teacher", "dataset_allocator", "dataset_entities", "dataset_roles",
             "episodes_per_pole", "decision_only")
#: N and k are the declared independent variables; everything else must match.
FREE_ROWS = ("N", "k")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_cfg(rel: str) -> tuple[str, dict]:
    """Return (state, info) reading the real object. Never infers from a sibling manifest."""
    p = ROOT / rel
    if not p.is_file():
        return PENDING, {"path": rel}
    try:
        import torch
        blob = torch.load(str(p), map_location="cpu", weights_only=False)
    except Exception as exc:                                   # noqa: BLE001
        return UNKNOWN, {"path": rel, "error": f"unreadable: {exc}"}
    if not isinstance(blob, dict):
        return UNKNOWN, {"path": rel, "error": f"unexpected payload {type(blob)!r}"}
    cfg = blob.get("cfg") or {}
    sd = blob.get("model_state_dict") or blob.get("state_dict") or {}
    names = list(sd) if isinstance(sd, dict) else []
    if not cfg:
        # Present but carrying no cfg: that is UNKNOWN, not "matches".
        return UNKNOWN, {"path": rel, "sha256": _sha(p), "error": "checkpoint carries no cfg block"}
    anchor_ds = str(cfg.get("sappo_anchor_dataset", "") or "")
    return "OK", {
        "path": rel, "sha256": _sha(p),
        "anchor_enabled": bool(anchor_ds),
        "anchor_dataset": anchor_ds,
        "total_timesteps": cfg.get("total_timesteps"),
        "entity_repair": bool([k for k in names if "entity" in k.lower()]),
        "entity_hidden_dim": cfg.get("entity_hidden_dim"),
        "split_attack_defend_enabled": bool(cfg.get("split_attack_defend_enabled")),
        "defend_teacher_lambda": cfg.get("defend_teacher_lambda"),
        "role_k_defend": cfg.get("role_k_defend"),
    }


SCALE_TOKENS = ("2v2", "4v4", "6v6", "2V2", "4V4", "6V6")
SUITE_SCALES = {2, 4, 6}


def _static_scale_facts(path: Path) -> dict:
    """Static read of a stage module: does it fork by scale, and which sizes does it accept?

    Uses ast, not regex, so a team-size constant inside a string or comment cannot be
    mistaken for a declaration.
    """
    import ast

    out: dict = {"hardcoded_team_size": None, "declared_team_sizes": None,
                 "team_size_branches": [], "parse_error": None}
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception as exc:                                   # noqa: BLE001
        out["parse_error"] = str(exc)
        return out

    def _ints(node):
        if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
            vals = [e.value for e in node.elts
                    if isinstance(e, ast.Constant) and isinstance(e.value, int)]
            return sorted(set(vals)) or None
        return None

    for node in ast.walk(tree):
        # module-level N_AGENTS / AGENTS = <int>  -> a fixed team size baked into the module
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in ("N_AGENTS", "AGENTS"):
                    if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                        out["hardcoded_team_size"] = node.value.value
                if isinstance(t, ast.Name) and t.id == "SUPPORTED_TEAM_SIZES":
                    got = _ints(node.value)
                    if got:
                        out["declared_team_sizes"] = got
        # add_argument("--team-size", ..., choices=(2, 4, 6))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "add_argument":
            first = node.args[0] if node.args else None
            if isinstance(first, ast.Constant) and first.value == "--team-size":
                for kw in node.keywords:
                    if kw.arg == "choices":
                        got = _ints(kw.value)
                        if got:
                            out["declared_team_sizes"] = got
                        elif isinstance(kw.value, ast.Name) and kw.value.id == "SUPPORTED_TEAM_SIZES":
                            pass            # picked up by the assignment branch above
        # `if n == 2:` / `n != 2` -- a methodology branch on team size. A declared
        # choices=(2,4,6) is worthless if the body then forks on the value, which is
        # exactly the legacy-dataset routing this attestation exists to catch.
        if isinstance(node, ast.Compare) and isinstance(node.left, ast.Name) \
                and node.left.id in ("n", "N", "n_agents", "N_AGENTS", "team_size", "AGENTS") \
                and len(node.ops) == 1 and isinstance(node.ops[0], (ast.Eq, ast.NotEq)) \
                and isinstance(node.comparators[0], ast.Constant) \
                and node.comparators[0].value in SUITE_SCALES:
            op = "==" if isinstance(node.ops[0], ast.Eq) else "!="
            out["team_size_branches"].append(
                f"line {node.lineno}: {node.left.id} {op} {node.comparators[0].value}")
    return out


def attest_implementations(required: dict) -> dict:
    """Layer 2: every stage must resolve to ONE module serving all three scales."""
    stages: dict = {}
    for stage, spec in (required.get("stages") or {}).items():
        rel = spec.get("module")
        p = ROOT / rel
        rec: dict = {"module": rel, "declared_state": spec.get("state", "declared compliant")}
        if not p.is_file():
            rec.update(verdict="MISSING", detail=f"module not found: {rel}")
            stages[stage] = rec
            continue
        rec["sha256"] = _sha(p)
        facts = _static_scale_facts(p)
        rec.update(facts)
        reasons = []
        if any(tok in Path(rel).name for tok in SCALE_TOKENS):
            reasons.append("filename encodes a scale (per-scale fork)")
        if facts["hardcoded_team_size"] is not None:
            reasons.append(f"module-level team size hardcoded to {facts['hardcoded_team_size']}")
        declared = facts["declared_team_sizes"]
        if declared is not None and not SUITE_SCALES.issubset(set(declared)):
            missing = sorted(SUITE_SCALES - set(declared))
            reasons.append(f"declared team sizes {declared} omit suite scale(s) {missing}")
        for br in facts["team_size_branches"]:
            reasons.append(f"methodology branch on team size -- {br}")
        rec["verdict"] = "COMPLIANT" if not reasons else "FORKED"
        rec["reasons"] = reasons
        # One module serving every scale is the whole point, so record it explicitly.
        rec["same_module_all_scales"] = not reasons
        stages[stage] = rec
    return stages


def _agree(infos: list[dict], field: str):
    """One value if every reading agrees, else a conflict marker."""
    vals = {json.dumps(i.get(field)) for i in infos}
    if len(vals) == 1:
        return json.loads(vals.pop())
    return f"CONFLICT{sorted(vals)}"


def collect(scale: str, cfgdef: dict) -> dict:
    row: dict = {"N": cfgdef["N"], "k": cfgdef["k"]}
    ev: dict = {"pipeline": cfgdef["pipeline"], "base": {}, "split": {}, "dataset": None}

    base_ok = []
    for role, rel in (cfgdef["base"] or {}).items():
        state, info = _load_cfg(rel)
        ev["base"][role] = {"state": state, **info}
        if state == "OK":
            base_ok.append(info)
    if not cfgdef["base"]:
        row["anchor_enabled"] = row["foundation_steps"] = row["entity_repair"] = PENDING
    elif not base_ok:
        st = {ev["base"][r]["state"] for r in ev["base"]}
        mark = PENDING if st == {PENDING} else UNKNOWN
        row["anchor_enabled"] = row["foundation_steps"] = row["entity_repair"] = mark
    else:
        row["anchor_enabled"] = _agree(base_ok, "anchor_enabled")
        row["foundation_steps"] = _agree(base_ok, "total_timesteps")
        row["entity_repair"] = _agree(base_ok, "entity_repair")

    split_ok = []
    for role, rel in (cfgdef["split"] or {}).items():
        state, info = _load_cfg(rel)
        ev["split"][role] = {"state": state, **info}
        if state == "OK":
            split_ok.append(info)
    if not split_ok:
        st = {ev["split"][r]["state"] for r in ev["split"]} if ev["split"] else {PENDING}
        mark = PENDING if st == {PENDING} else UNKNOWN
        row["split_steps"] = row["Nprime_teacher"] = mark
    else:
        row["split_steps"] = _agree(split_ok, "total_timesteps")
        lam = _agree(split_ok, "defend_teacher_lambda")
        row["Nprime_teacher"] = bool(lam) if isinstance(lam, (int, float)) else lam

    if cfgdef["dataset"]:
        p = SD / cfgdef["dataset"]
        if not p.is_file():
            row["dataset_allocator"] = row["dataset_entities"] = UNKNOWN
            row["dataset_roles"] = row["episodes_per_pole"] = row["decision_only"] = UNKNOWN
            ev["dataset"] = {"state": UNKNOWN, "manifest": cfgdef["dataset"], "error": "missing"}
        else:
            d = json.loads(p.read_text(encoding="utf-8"))
            alloc = d.get("allocator")
            tot = d.get("totals") or {}
            eps = sorted({int(v.get("episodes", -1)) for v in tot.values()})
            row["dataset_allocator"] = (alloc or {}).get("rule") if isinstance(alloc, dict) else (alloc or UNKNOWN)
            row["dataset_entities"] = d.get("stored_entity_tensors", UNKNOWN)
            row["dataset_roles"] = d.get("stored_roles", UNKNOWN)
            row["episodes_per_pole"] = eps[0] if len(eps) == 1 else f"CONFLICT{eps}"
            row["decision_only"] = d.get("decision_rows_only", UNKNOWN)
            ev["dataset"] = {"state": "OK", "manifest": cfgdef["dataset"], "sha256": _sha(p),
                             "allocator": alloc, "n_shards": len(d.get("shards") or []),
                             "device": d.get("device")}
    else:
        for r in ("dataset_allocator", "dataset_entities", "dataset_roles",
                  "episodes_per_pole", "decision_only"):
            row[r] = PENDING
        ev["dataset"] = {"state": PENDING, "manifest": None}
    return {"row": row, "evidence": ev}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="print the record instead of the table")
    args = ap.parse_args()

    recipe = json.loads(RECIPE.read_text(encoding="utf-8"))
    if str(recipe.get("status")) != "FROZEN_RECIPE":
        raise SystemExit(f"FAIL-CLOSED: recipe status {recipe.get('status')!r}, expected FROZEN_RECIPE")
    ident = json.loads(SUITE_SPEC.read_text(encoding="utf-8")).get("METHODOLOGY_IDENTITY_2026_09_26") or {}
    if str(ident.get("status")) != "LOCKED":
        raise SystemExit("FAIL-CLOSED: suite methodology identity is not LOCKED")

    cr = recipe["COMMON_RECIPE_locked"]
    required = {
        "anchor_enabled": cr["sappo_anchor"]["value"],
        "foundation_steps": cr["base_specialist_total_timesteps"],
        "entity_repair": cr["entity_repair_enabled"],
        "split_steps": cr["split_defender_stage"]["total_timesteps"],
        "Nprime_teacher": True,
        "dataset_allocator": cr["dataset_collection"]["allocator"].split(",")[0],
        "dataset_entities": cr["dataset_collection"]["stored_entity_tensors"],
        "dataset_roles": cr["dataset_collection"]["stored_roles"],
        "episodes_per_pole": cr["dataset_collection"]["episodes_per_pole"],
        "decision_only": cr["dataset_collection"]["decision_rows_only"],
    }

    got = {s: collect(s, d) for s, d in SCALES.items()}
    scales = list(SCALES)

    def cell(v) -> str:
        if v is True:
            return "true"
        if v is False:
            return "false"
        if isinstance(v, int) and not isinstance(v, bool):
            if v >= 1_000_000 and v % 1_000_000 == 0:
                return f"{v // 1_000_000}M"
            return f"{v // 1000}k" if v >= 1000 and v % 1000 == 0 else str(v)
        return str(v)

    print(f"CROSS-SCALE IDENTITY ATTESTATION  {_now()}")
    print(f"  required values from {RECIPE.name} (FROZEN_RECIPE)")
    print(f"  settings read from loaded checkpoints; UNKNOWN != match\n")
    w = 22
    # Width from the widest cell actually present, so a long value (CLOSEST_DEFENDS,
    # CONFLICT[...]) cannot run into the next column and misread as a different value.
    cw = max(10, *(len(cell(got[s]["row"].get(r))) + 2 for s in scales for r in ROW_ORDER),
             *(len(cell(v)) + 2 for v in required.values()))
    print("  " + "row".ljust(w) + "".join(s.rjust(cw) for s in scales) + "     required")
    mismatches, pending_scales = [], []
    for r in ROW_ORDER:
        vals = [got[s]["row"].get(r) for s in scales]
        req = "-" if r in FREE_ROWS else cell(required.get(r))
        print("  " + r.ljust(w) + "".join(cell(v).rjust(cw) for v in vals) + f"     {req}")
        if r in FREE_ROWS:
            continue
        for s, v in zip(scales, vals):
            if v in (PENDING,):
                continue
            if v == UNKNOWN or str(v).startswith("CONFLICT") or v != required.get(r):
                mismatches.append({"scale": s, "row": r, "measured": v, "required": required.get(r)})
    for s in scales:
        rows = [v for k, v in got[s]["row"].items() if k not in FREE_ROWS]
        if all(v == PENDING for v in rows):
            pending_scales.append(s)

    # ---- layer 2: implementation path --------------------------------------------
    impl_req = recipe.get("STAGE_IMPLEMENTATIONS_required")
    if impl_req is None:
        raise SystemExit("FAIL-CLOSED: frozen recipe carries no STAGE_IMPLEMENTATIONS_required")
    impls = attest_implementations(impl_req)
    forked = {k: v for k, v in impls.items() if v.get("verdict") != "COMPLIANT"}

    print(f"\n  LAYER 2 -- implementation path (one module per stage, all three scales)")
    sw = max(len(k) for k in impls) + 2
    for stage, rec in impls.items():
        mark = "ok  " if rec["verdict"] == "COMPLIANT" else "FORK"
        print(f"    [{mark}] {stage.ljust(sw)}{rec['module']}")
        for r in rec.get("reasons", []):
            print(f"           -> {r}")
        if rec["verdict"] == "MISSING":
            print(f"           -> {rec['detail']}")

    print()
    if pending_scales:
        print(f"  NOT YET ATTESTABLE (nothing built): {', '.join(pending_scales)}")
        for s in pending_scales:
            print(f"    {s}: {got[s]['evidence']['pipeline']}")
    if mismatches:
        print(f"\n  LAYER 1: {len(mismatches)} mismatch(es) against the frozen recipe:")
        for m in mismatches:
            print(f"    {m['scale']:<5} {m['row']:<22} measured={cell(m['measured'])}  required={cell(m['required'])}")
    else:
        print("  LAYER 1: every BUILT scale matches the frozen recipe on every invariant row.")
    if forked:
        print(f"  LAYER 2: {len(forked)}/{len(impls)} stage(s) are NOT a single cross-scale implementation.")
    else:
        print(f"  LAYER 2: all {len(impls)} stages resolve to one cross-scale implementation.")

    record = {
        "record": "cross-scale identity attestation",
        "utc": _now(),
        "scope": "READ-ONLY preflight. Changes nothing, authorizes nothing.",
        "required_from": {"file": RECIPE.name, "sha256": _sha(RECIPE), "status": recipe["status"]},
        "identity_from": {"file": SUITE_SPEC.name, "section": "METHODOLOGY_IDENTITY_2026_09_26"},
        "required_values": required,
        "free_rows": list(FREE_ROWS),
        "table": {s: got[s]["row"] for s in scales},
        "evidence": {s: got[s]["evidence"] for s in scales},
        "mismatches": mismatches,
        "not_yet_attestable": pending_scales,
        "layer_2_implementations": impls,
        "layer_2_forked_stages": sorted(forked),
        "attested": not mismatches and not pending_scales and not forked,
        "attested_requires": "layer 1 (configuration semantics) AND layer 2 (implementation path). "
                             "Layer 1 alone can pass while two scales reach the same numbers "
                             "through different code.",
        "semantics": {
            "PENDING": "artifact not built yet; not a mismatch, and not an attestation either",
            "UNKNOWN": "artifact present but its setting could not be read from the loaded object; treated as a mismatch",
            "CONFLICT[...]": "roles within one scale disagree",
        },
        "what_this_does_not_check": [
            "that the executed code path was byte-identical across scales",
            "pole overlay equality between scales",
            "that the N' teacher schedule/cadence/formula match beyond the lambda being present",
        ],
    }
    OUT.write_text(json.dumps(record, indent=2), encoding="utf-8")
    if args.json:
        print(json.dumps(record, indent=2))
    print(f"\n  -> {OUT}")
    return 0 if record["attested"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
