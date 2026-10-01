r"""Cross-scale audit of the standardized suite distillation datasets -- GREEN before any student trains.

    python experiments/audit_suite_datasets_cross_scale.py [--write]

Datasets (PI, 2026-09-28): SUITE_DISTILLATION_2V2 and the corrected SUITE_DISTILLATION_4V4_V2. Each is
checked against its OWN frozen spec, the governing certification and the files on disk; then the pair is
checked for recipe identity -- only N, k, seeds, checkpoint pins, certified pole identity, paths and tensor
shapes that follow from N may differ. Any failed check -> RED, and the record says so; nothing downstream
may consume a dataset while the audit is RED.

Per dataset:
  manifest FROZEN_DATASET; spec FROZEN and its sha256 equals the manifest's; collector module and fingerprint
  present; allocator CLOSEST_DEFENDS, fixed-for-episode, H_r=8, k == K_DEFEND_BY_SCALE[N]; pole hashes equal a
  fresh resolution through pole_attestation (the certification), and the spec's overlays agree; the five
  acting/teacher pins equal the spec and the files on disk; frozen attack == KL pi_A, acting pi_B == KL pi_B;
  shards: exactly n_per_pole per pole, seeds exactly the registered blocks, every file present; registry blocks
  SPENT; in EVERY stored row: exactly k defenders, a non-empty decision mask, entity tensors with N-1
  teammates and N enemies.
Across datasets:
  same scientific implementation (identical collector blob AND no diff under the scientific paths of
  experiments/code_identity.py; both real git shas reported, and differing shas need a FROZEN
  COLLECTOR_COMMIT_EQUIVALENCE attestation equal to the live recomputation), same manifest key set, same fingerprint key set,
  same allocator except k, same stored keys, same n_per_pole, decision-rows-only / entities / roles flags.
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

from experiments import code_identity as CI  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
DATASETS = {
    "2v2": ("SUITE_DISTILLATION_2V2_SPEC.json", "SUITE_DISTILLATION_2V2_DATASET.json"),
    "4v4": ("SUITE_DISTILLATION_4V4_V2_SPEC.json", "SUITE_DISTILLATION_4V4_V2_DATASET.json"),
    "6v6": ("SUITE_DISTILLATION_6V6_SPEC.json", "SUITE_DISTILLATION_6V6_DATASET.json"),
    # Symmetric role construction (SYMMETRIC_ROLE_TOP50_DIAGNOSTIC_SPEC.json): both poles act with a
    # split composite (pi_DA + frozen pi_A on A, pi_DB + frozen pi_B on B), k = ceil(N/3). Audited as
    # their own family with 2v2_sym as the reference; never mixed with the asymmetric datasets.
    **{f"{n}v{n}_sym": (f"SUITE_DISTILLATION_{n}V{n}_SYM_SPEC.json", f"SUITE_DISTILLATION_{n}V{n}_SYM_DATASET.json")
       for n in (2, 4, 6)},
}
#: The CLOSEST_DEFENDS scale knob (CROSS_SCALE_CANONICAL_RECIPE_V1 ALLOWED_TO_DIFFER).
K_BY_SCALE = {"2v2": 1, "4v4": 2, "6v6": 1, "2v2_sym": 1, "4v4_sym": 2, "6v6_sym": 2}
OUT = SD / "SUITE_DATASETS_CROSS_SCALE_AUDIT.json"
#: Reviewed changes to files the collector executes, allowed between the reference (2v2) collection
#: and a later one (PI 2026-09-29: identity = the executed code). Each entry pins exact blobs.
EXECUTED_CODE_ATTESTATION = SD / "COLLECTOR_EXECUTED_CODE_ATTESTATION.json"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def shard_collector_sha(z) -> str | None:
    """The collector commit embedded in a loaded shard's fingerprint; None when the shard carries none."""
    if "fingerprint" not in z.files:
        return None
    return json.loads(str(z["fingerprint"])).get("collector_git_sha")


def audit_one(scale: str, spec_name: str, man_name: str, checks: list) -> dict:
    from experiments.collect_suite_distillation_states import K_DEFEND_BY_SCALE
    from experiments.pole_attestation import pole_identity, resolve_pole_genome
    from experiments import seed_registry as SR

    def check(name, ok, detail=None):
        checks.append({"scale": scale, "check": name, "ok": bool(ok), "detail": detail})
        return bool(ok)

    sp, mp = SD / spec_name, SD / man_name
    if not check("manifest_exists", mp.is_file(), man_name) or not check("spec_exists", sp.is_file(), spec_name):
        return {}
    spec, man = json.loads(sp.read_text(encoding="utf-8")), json.loads(mp.read_text(encoding="utf-8"))
    n = int(man.get("team_size", -1))
    symmetric = scale.endswith("_sym")
    k = K_BY_SCALE[scale] if symmetric else K_DEFEND_BY_SCALE.get(n)
    check("construction_matches_family",
          (spec.get("ACTING_DEPLOYMENT_locked") or {}).get("construction", "asymmetric")
          == (man.get("acting") or {}).get("construction", "asymmetric")
          == ("symmetric" if symmetric else "asymmetric"),
          (man.get("acting") or {}).get("construction"))
    check("manifest_FROZEN_DATASET", man.get("status") == "FROZEN_DATASET", man.get("status"))
    check("spec_FROZEN", str(spec.get("status", "")).startswith("FROZEN"), spec.get("status"))
    col = man.get("collector") or {}
    check("collector_recorded", col.get("module") == "experiments/collect_suite_distillation_states.py"
          and bool(col.get("git_sha")), col)
    check("spec_sha_matches_manifest", col.get("spec_sha256") == _sha(sp), col.get("spec_sha256"))
    check("fingerprint_recorded", isinstance(man.get("fingerprint"), dict))

    al = man.get("allocator") or {}
    check("allocator_CD_k_fixed_H8", al == {"rule": "CLOSEST_DEFENDS", "k_defend": k, "fixed_for_episode": True,
                                            "hold_ticks_H_r": 8}, al)
    check("spec_allocator_k", int(spec["ALLOCATOR_locked"]["k_defend"]) == k, spec["ALLOCATOR_locked"])

    for p in ("A", "B"):
        live = pole_identity(p, n, resolve_pole_genome(p, n))
        rec = (man.get("poles") or {}).get(p) or {}
        check(f"pole_{p}_hash_equals_certification", rec.get("pole_config_hash") == live["pole_config_hash"],
              [rec.get("pole_config_hash"), live["pole_config_hash"]])
        want = {kk: (int(v) if isinstance(v, float) and float(v).is_integer() else v)
                for kk, v in sorted(spec["POLES"][p]["overlay"].items())}
        check(f"pole_{p}_spec_overlay_equals_certification", want == live["overlay"], [want, live["overlay"]])

    act, kl = spec["ACTING_DEPLOYMENT_locked"], spec["KL_TEACHERS_locked"]
    pins = {"pi_D": act["Pole_A"]["pi_D"], "frozen_attack": act["Pole_A"]["frozen_attack_pi_A"],
            "pi_B_act": act["Pole_B"]["frozen_attack_pi_B"] if symmetric else act["Pole_B"],
            "pi_A_kl": kl["pi_A"], "pi_B_kl": kl["pi_B"]}
    mact, mteach = man.get("acting") or {}, man.get("teachers") or {}
    mpins = {"pi_D": mact.get("pi_D"), "frozen_attack": mact.get("frozen_attack"), "pi_B_act": mact.get("pi_B"),
             "pi_A_kl": mteach.get("pi_A"), "pi_B_kl": mteach.get("pi_B")}
    if symmetric:
        pins["pi_DB"], mpins["pi_DB"] = act["Pole_B"]["pi_D"], mact.get("pi_DB")
    for name, pin in pins.items():
        f = ROOT / pin["path"]
        check(f"pin_{name}_file_sha", f.is_file() and _sha(f) == pin["sha256"], pin["sha256"][:16])
        check(f"pin_{name}_manifest_equals_spec", (mpins[name] or {}).get("sha256") == pin["sha256"])
    check("frozen_attack_is_KL_pi_A", pins["frozen_attack"]["sha256"] == pins["pi_A_kl"]["sha256"])
    check("acting_pi_B_is_KL_pi_B", pins["pi_B_act"]["sha256"] == pins["pi_B_kl"]["sha256"])

    npp = int(spec["DATASET"]["n_per_pole"])
    shards = man.get("shards") or []
    reg = man.get("seed_registry") or {}
    for blk, pole in (("collection_A", "A"), ("collection_B", "B")):
        lo, hi = (int(x) for x in str(spec["SEEDS"][blk]).split(".."))
        got = sorted(s_["seed"] for s_ in shards if s_["pole"] == pole)
        check(f"shards_{pole}_exact_registered_block", got == list(range(lo, hi + 1)) and len(got) == npp,
              [len(got), lo, hi])
        rid = spec["SEEDS"]["registry_experiment_ids"][blk]
        b = next((x for x in SR.load()["blocks"] if x["experiment_id"] == rid), None)
        check(f"registry_{blk}_SPENT", b is not None and b["status"] == "SPENT" and (b["lo"], b["hi"]) == (lo, hi),
              None if b is None else [b["status"], b["lo"], b["hi"]])
        check(f"manifest_registry_{blk}", (reg.get(blk) or {}).get("experiment_id") == rid)

    sha = col.get("git_sha")
    check("manifest_fingerprint_collector_sha", (man.get("fingerprint") or {}).get("collector_git_sha") == sha,
          [(man.get("fingerprint") or {}).get("collector_git_sha"), sha])
    check("collector_clean_at_collection", col.get("git_dirty") is False, col.get("git_dirty"))
    if "scientific_tree_sha256" in col:   # recorded from 2026-09-28 on; older manifests are recomputed from git_sha
        check("manifest_scientific_tree_matches_git_sha",
              col["scientific_tree_sha256"] == CI.scientific_tree_sha256(sha), col["scientific_tree_sha256"][:16])
    bad_def, bad_dec, bad_ent, missing, rows, foreign = 0, 0, 0, 0, 0, 0
    for s_ in shards:
        f = ROOT / s_["file"]
        if not f.is_file():
            missing += 1
            continue
        z = np.load(f, allow_pickle=False)
        if not sha or shard_collector_sha(z) != sha:
            foreign += 1
        roles, dm = z["roles"], z["decision_mask"]
        rows += roles.shape[0]
        bad_def += int(((roles < 0.5).sum(axis=1) != k).sum())
        bad_dec += int((~dm.any(axis=1)).sum())
        tm, en = z["teammates"], z["enemies"]
        if tm.shape[1:3] != (n, n - 1) or en.shape[1:3] != (n, n):
            bad_ent += 1
    check("all_shard_files_present", missing == 0, missing)
    check("every_shard_from_the_manifest_collector_commit", foreign == 0, {"foreign_shards": foreign, "sha": sha})
    check("every_row_exactly_k_defenders", bad_def == 0 and rows > 0, {"rows": rows, "violations": bad_def})
    check("every_row_decision_eligible", bad_dec == 0, bad_dec)
    check("entity_tensor_shapes_follow_N", bad_ent == 0, bad_ent)
    check("flags_decision_only_entities_roles", man.get("decision_rows_only") is True
          and man.get("stored_entity_tensors") is True and man.get("stored_roles") is True)
    return {"spec": spec, "manifest": man, "n": n, "k": k, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--scales", default="2v2,4v4",
                    help="datasets to audit; 2v2 is the reference every other scale is compared with. "
                         "Default 2v2,4v4 reproduces the original audit exactly.")
    ap.add_argument("--out", default=None, help="record path (default: the original 2v2/4v4 record)")
    a = ap.parse_args()
    scales = [s.strip() for s in a.scales.split(",") if s.strip()]
    ref = scales[0]
    # A single symmetric dataset may be audited on its own (per-dataset checks only): the school PC
    # collects 6v6_sym without the 2v2_sym shards on disk.
    single_sym = len(scales) == 1 and ref.endswith("_sym")
    if ((ref not in ("2v2", "2v2_sym") and not single_sym) or any(s not in DATASETS for s in scales)
            or any(s.endswith("_sym") != ref.endswith("_sym") for s in scales)):
        raise SystemExit(f"--scales must start with 2v2 (or 2v2_sym), stay in one family, and name only {list(DATASETS)}")
    out_path = Path(a.out) if a.out else OUT
    checks: list = []
    got = {sc: audit_one(sc, *DATASETS[sc], checks) for sc in scales}

    def cross(tag, name, ok, detail=None):
        checks.append({"scale": tag, "check": name, "ok": bool(ok), "detail": detail})

    if all(got.values()):
        ref_m, ref_s = got[ref]["manifest"], got[ref]["spec"]
        sha_ref = ref_m["collector"]["git_sha"]
        for sc in scales[1:]:
            tag = f"x{sc}"
            m, s = got[sc]["manifest"], got[sc]["spec"]
            sha = m["collector"]["git_sha"]
            # Identity is the executed code, not HEAD. Both real shas are reported, never rewritten.
            if sc == "4v4":
                # 2v2 and 4v4 were collected by the same code; the original whole-tree rule applies.
                try:
                    eq = CI.code_equivalence(sha_ref, sha)
                except RuntimeError as exc:
                    eq = {"equivalent": False, "error": str(exc)}
                cross(tag, "same_scientific_implementation", eq["equivalent"],
                      {"git_sha_2v2": sha_ref, f"git_sha_{sc}": sha, "collector_identical": eq.get("collector_identical"),
                       "scientific_changed_files": eq.get("scientific_changed_files"), "error": eq.get("error")})
                if sha_ref != sha:
                    att_p = SD / CI.attestation_name(sha_ref, sha)
                    att = json.loads(att_p.read_text(encoding="utf-8")) if att_p.is_file() else {}
                    cross(tag, "differing_shas_have_frozen_equivalence_attestation",
                          att.get("status") == "FROZEN" and att.get("equivalence") == eq, att_p.name)
            else:
                # Later scales: the collector's executed import closure must be identical, or differ only
                # by reviewed, blob-pinned changes (EXECUTED_CODE_ATTESTATION).
                att = (json.loads(EXECUTED_CODE_ATTESTATION.read_text(encoding="utf-8"))
                       if EXECUTED_CODE_ATTESTATION.is_file() else {})
                try:
                    eq = CI.executed_code_equivalence(sha_ref, sha, att.get("reviewed_changes", [])
                                                      if att.get("status") == "FROZEN" else [])
                except RuntimeError as exc:
                    eq = {"equivalent": False, "error": str(exc)}
                cross(tag, "same_executed_collector_code", eq["equivalent"],
                      {"git_sha_2v2": sha_ref, f"git_sha_{sc}": sha, "closure_files": eq.get("n_closure_files"),
                       "differing": [d["file"] for d in eq.get("differing", [])],
                       "unreviewed": eq.get("unreviewed"), "error": eq.get("error")})
            cross(tag, "same_manifest_keys", set(ref_m) == set(m), sorted(set(ref_m) ^ set(m)))
            cross(tag, "same_fingerprint_keys", set(ref_m["fingerprint"]) == set(m["fingerprint"]))
            ar = {kk: v for kk, v in ref_m["allocator"].items() if kk != "k_defend"}
            am = {kk: v for kk, v in m["allocator"].items() if kk != "k_defend"}
            cross(tag, "same_allocator_except_k", ar == am, [ar, am])
            cross(tag, "same_stored_keys", ref_s["DATASET"]["stored_obs_keys"] == s["DATASET"]["stored_obs_keys"])
            cross(tag, "same_n_per_pole", ref_s["DATASET"]["n_per_pole"] == s["DATASET"]["n_per_pole"])
            cross(tag, "k_is_the_scale_knob", (got[ref]["k"], got[sc]["k"]) == (K_BY_SCALE[ref], K_BY_SCALE[sc]))
    green = bool(checks) and all(c["ok"] for c in checks)
    for c in checks:
        print(f"  [{'OK' if c['ok'] else 'FAIL'}] {c['scale']:5s} {c['check']:44s} {'' if c['detail'] is None else str(c['detail'])[:110]}")
    print(f"\n{sum(c['ok'] for c in checks)}/{len(checks)} -> {'GREEN' if green else 'RED'}")
    if a.write:
        out_path.write_text(json.dumps({
            "record_id": out_path.stem,
            "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "verdict": "GREEN" if green else "RED",
            "datasets": {sc: DATASETS[sc][1] for sc in scales},
            "rows": {sc: (g or {}).get("rows") for sc, g in got.items()},
            "checks": checks,
        }, indent=2) + "\n", encoding="utf-8")
        print(f"-> {out_path}")
    return 0 if green else 1


if __name__ == "__main__":
    raise SystemExit(main())
