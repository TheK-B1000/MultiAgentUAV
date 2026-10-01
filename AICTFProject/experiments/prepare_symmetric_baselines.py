r"""Freeze the symmetric-role baseline suite, one scale at a time (2v2, 4v4, 6v6; N is an argument).

The symmetric family re-runs the sharing baselines on a state set visited by the SYMMETRIC system:
both poles act with a split composite under CLOSEST_DEFENDS(k = ceil(N/3)) -- pi_DA + frozen pi_A on
Pole A, pi_DB + frozen pi_B on Pole B. The defenders only choose which states are visited; the KL
teachers stay the repaired pi_A / pi_B. Every arm is then scored on the frozen Ours top-50 seeds as
a POST-HOC diagnostic (SYMMETRIC_ROLE_TOP50_DIAGNOSTIC_SPEC.json); nothing here is confirmatory.

Stages (each writes one frozen spec and refuses to overwrite a different one):

  --reserve                 register every collection / student-init block for all three scales
                            (fixed ranges, so two machines never allocate; idempotent)
  --team-size N --collection  SUITE_DISTILLATION_<N>V<N>_SYM_SPEC.json, bound to the SEALED
                            TOP50_<N>V<N>_SYMMETRIC_OURS record (exactly the defenders evaluated)
  --team-size N --sharing     STANDARDIZED_<N>V<N>_SYM_SHARING_SPEC.json, bound to the FROZEN
                            symmetric dataset and its GREEN audit
  --team-size N --eval        STANDARDIZED_<N>V<N>_SYM_SHARING_EVAL_SPEC.json, bound to the three
                            frozen students; also authorizes the robustness runs on symmetric Ours

  python experiments/prepare_symmetric_baselines.py --reserve
  python experiments/prepare_symmetric_baselines.py --team-size 6 --collection
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
DIAG_SPEC = SD / "SYMMETRIC_ROLE_TOP50_DIAGNOSTIC_SPEC.json"
TAG = "SYM"
COLLECTION_PARENT = {2: "SUITE_DISTILLATION_2V2_SPEC.json", 4: "SUITE_DISTILLATION_4V4_V2_SPEC.json",
                     6: "SUITE_DISTILLATION_6V6_SPEC.json"}
ARMS = ("share_encoder", "fully_shared", "generalist")
ARM_TAG = {"share_encoder": "share_encoder", "fully_shared": "fully_shared_z", "generalist": "generalist"}
EVAL_ARM_KEY = {"share_encoder": "share_encoder", "fully_shared": "fully_shared_z", "generalist": "generalist"}
N_INIT = {"share_encoder": 2, "fully_shared": 1, "generalist": 1}
#: Robustness on symmetric Ours: the frozen medium tier of each family (as STANDARDIZED_*_NOISE_SPEC).
#: Nominal is TOP50_<N>V<N>_SYMMETRIC_OURS itself -- same seeds, deterministic, so re-running it
#: would reproduce it row for row.
ROBUSTNESS = {"LOCALIZATION": "localization_noise", "MOTION": "motion_error", "DELAY": "control_delay"}
ROBUSTNESS_SEVERITY = "medium"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rel(p) -> str:
    return str(p).replace("\\", "/")


def k_sym(n: int) -> int:
    return -(-n // 3)


def blocks(n: int) -> dict:
    """Fixed registry blocks of the symmetric suite at scale N: one 100k stride per purpose, one
    1000-seed lane per scale inside it. Pre-registered so the school PC (6v6) and this PC (2v2/4v4)
    never allocate concurrently."""
    lane = {2: 0, 4: 1000, 6: 2000}[n]
    out = {
        "collection_A": (f"SYMMETRIC_{n}V{n}_SUITE_DATASET_POLE_A", 26_300_001 + lane, 96),
        "collection_B": (f"SYMMETRIC_{n}V{n}_SUITE_DATASET_POLE_B", 26_400_001 + lane, 96),
        "share_encoder": (f"SYMMETRIC_{n}V{n}_SHARE_ENCODER_TRAINING", 26_500_001 + lane, N_INIT["share_encoder"]),
        "fully_shared": (f"SYMMETRIC_{n}V{n}_FULLY_SHARED_Z_TRAINING", 26_600_001 + lane, N_INIT["fully_shared"]),
        "generalist": (f"SYMMETRIC_{n}V{n}_GENERALIST_TRAINING", 26_700_001 + lane, N_INIT["generalist"]),
    }
    return {k: (eid, lo, lo + cnt - 1) for k, (eid, lo, cnt) in out.items()}


def _rng(b) -> str:
    return f"{b[1]}..{b[2]}"


def reserve() -> list[str]:
    from experiments import seed_registry as SR
    done = []
    reg = {b["experiment_id"]: b for b in SR.load()["blocks"]}
    for n in (2, 4, 6):
        for purpose, (eid, lo, hi) in blocks(n).items():
            if eid in reg:
                if (reg[eid]["lo"], reg[eid]["hi"]) != (lo, hi):
                    raise SystemExit(f"REFUSING: {eid} registered as {reg[eid]['lo']}..{reg[eid]['hi']}, not {lo}..{hi}")
                continue
            SR.allocate(eid, lo, hi, "exploratory",
                        purpose=f"symmetric-role baseline suite {n}v{n}: {purpose}",
                        spec="experiments/prepare_symmetric_baselines.py")
            done.append(eid)
    return done


def _load(p: Path) -> dict:
    if not p.is_file():
        raise SystemExit(f"FAIL-CLOSED: {p.name} missing")
    return json.loads(p.read_text(encoding="utf-8"))


def _write_frozen(path: Path, doc: dict) -> Path:
    """One-shot: an existing spec is accepted only if its content (minus utc) is identical."""
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        strip = lambda d: {k: v for k, v in d.items() if k != "utc"}  # noqa: E731
        if strip(old) != strip(doc):
            raise SystemExit(f"REFUSING: {path.name} exists with different content; specs are frozen once")
        return path
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def _diag_entry(n: int) -> dict:
    d = _load(DIAG_SPEC)
    e = (d.get("POST_HOC_MATCHED_ROLE_ABLATIONS") or {}).get(f"TOP50_{n}V{n}_SYMMETRIC_OURS")
    if not isinstance(e, dict):
        raise SystemExit(f"FAIL-CLOSED: TOP50_{n}V{n}_SYMMETRIC_OURS not in {DIAG_SPEC.name}")
    if int(e["role_k_defend"]) != k_sym(n):
        raise SystemExit(f"FAIL-CLOSED: diagnostic k={e['role_k_defend']} != ceil({n}/3)")
    return e


def sealed_ours(n: int) -> dict:
    """The defenders of the SEALED symmetric Ours top-50 record, verified on disk."""
    rec = _load(SD / f"TOP50_{n}V{n}_SYMMETRIC_OURS_SPECIALIST_CROSSOVER_EVAL_RESULT.json")
    if rec.get("status") != "SEALED":
        raise SystemExit(f"REFUSING: TOP50_{n}V{n}_SYMMETRIC_OURS is {rec.get('status')!r}, not SEALED")
    sa, sb = rec.get("split_policy_pi_A"), rec.get("split_policy_pi_B")
    if not (isinstance(sa, dict) and isinstance(sb, dict)):
        raise SystemExit(f"REFUSING: TOP50_{n}V{n}_SYMMETRIC_OURS is not split on both sides")
    e = _diag_entry(n)
    out = {
        "pi_DA": {"path": _rel(sa["pi_D_path"]), "sha256": rec["checkpoints"]["pi_A"]},
        "pi_DB": {"path": _rel(sb["pi_D_path"]), "sha256": rec["checkpoints"]["pi_B"]},
        "pi_A": {"path": _rel(sa["frozen_attack_path"]), "sha256": sa["frozen_attack_sha256"]},
        "pi_B": {"path": _rel(sb["frozen_attack_path"]), "sha256": sb["frozen_attack_sha256"]},
    }
    for name, pin in out.items():
        f = ROOT / pin["path"]
        if not f.is_file() or _sha(f) != pin["sha256"]:
            raise SystemExit(f"REFUSING: {name} {pin['path']} missing or sha != the sealed record")
    if out["pi_A"] != {"path": _rel(e["frozen_attack_A"]["path"]), "sha256": e["frozen_attack_A"]["sha256"]} or \
            out["pi_B"] != {"path": _rel(e["frozen_attack_B"]["path"]), "sha256": e["frozen_attack_B"]["sha256"]}:
        raise SystemExit("REFUSING: the sealed record's frozen attackers differ from the diagnostic spec's")
    return {"pins": out, "record": rec, "entry": e}


def collection_spec(n: int) -> Path:
    from experiments import seed_registry as SR
    parent_name = COLLECTION_PARENT[n]
    parent = _load(SD / parent_name)
    ours = sealed_ours(n)
    pins = ours["pins"]
    kl = parent["KL_TEACHERS_locked"]
    for side in ("A", "B"):
        if kl[f"pi_{side}"]["sha256"] != pins[f"pi_{side}"]["sha256"]:
            raise SystemExit(f"REFUSING: {parent_name} KL teacher pi_{side} != the symmetric frozen attacker; "
                             f"the symmetric suite keeps the same teachers")
    reg = {b["experiment_id"]: b for b in SR.load()["blocks"]}
    bl = blocks(n)
    for key in ("collection_A", "collection_B"):
        b = reg.get(bl[key][0])
        if b is None or b["status"] != "RESERVED":
            raise SystemExit(f"REFUSING: {bl[key][0]} not RESERVED (run --reserve)")
    name = f"SUITE_DISTILLATION_{n}V{n}_{TAG}_SPEC.json"
    doc = copy.deepcopy(parent)
    doc.update({
        "record_id": name[:-5],
        "status": "FROZEN_BEFORE_COLLECTION",
        "utc": _now(),
        "classification": "DIAGNOSTIC dataset (collection only), symmetric role construction. Post-hoc; "
                          "not confirmatory. Not PAPER-FAITHFUL.",
        "authorized_by": "PI, 2026-10-01: symmetric-role baseline suite at 2v2/4v4/6v6 (one pipeline, N an argument)",
        "parent": [parent_name, DIAG_SPEC.name,
                   f"TOP50_{n}V{n}_SYMMETRIC_OURS_SPECIALIST_CROSSOVER_EVAL_RESULT.json (sealed; defenders pinned from it)"],
        "derived_from": parent_name,
        "ALLOCATOR_locked": {**parent["ALLOCATOR_locked"], "k_defend": k_sym(n),
                             "source": "k = ceil(N/3) (SYMMETRIC_ROLE_TOP50_DIAGNOSTIC_SPEC.json)"},
        "ACTING_DEPLOYMENT_locked": {
            "construction": "symmetric",
            "why": "both poles are visited by the symmetric system: the same split construction on each side",
            "Pole_A": {"policy": "DEFEND_ATTACK_SPLIT composite", "pi_D": pins["pi_DA"],
                       "frozen_attack_pi_A": pins["pi_A"],
                       "splice": "rl.custom_ppo.split_attack_defend.splice_actions; deterministic=True"},
            "Pole_B": {"policy": "DEFEND_ATTACK_SPLIT composite", "pi_D": pins["pi_DB"],
                       "frozen_attack_pi_B": pins["pi_B"],
                       "splice": "rl.custom_ppo.split_attack_defend.splice_actions; deterministic=True"},
            "defenders_are_not_teachers": True,
        },
        "SEEDS": {"collection_A": _rng(bl["collection_A"]), "collection_B": _rng(bl["collection_B"]),
                  "registry_experiment_ids": {"collection_A": bl["collection_A"][0],
                                              "collection_B": bl["collection_B"][0]},
                  "seed_class": "exploratory",
                  "chosen_by": "prepare_symmetric_baselines.blocks (pre-registered, fixed lanes)",
                  "student_init_and_shuffle": "allocated by the symmetric sharing spec, not here"},
        "LAUNCH": {"command": f".venv/Scripts/python.exe experiments/collect_suite_distillation_states.py "
                              f"--team-size {n} --dataset-tag {TAG} --device cuda --resume"},
        "OUTPUTS": {"dataset_manifest": f"SUITE_DISTILLATION_{n}V{n}_{TAG}_DATASET.json",
                    "state_shards": f"artifacts/strategic_demand/sppo/suite_distillation_symmetric/{n}v{n}/states/"},
        "SEQUENCE": "freeze -> smoke -> collect once -> dataset -> audit (--scales <N>v<N>_sym) -> symmetric sharing spec",
        "NOT_AUTHORIZED_BY_THIS_SPEC": ["training any student (collection only)",
                                        "reusing these seeds for any evaluation",
                                        "using pi_DA or pi_DB as a KL teacher",
                                        "claiming confirmatory or PAPER-FAITHFUL status"],
    })
    for k in ("SAME_RECIPE_AS_2V2", "SMOKE_before_collection"):
        doc.pop(k, None)
    doc["SAME_RECIPE_AS_PARENT"] = (f"identical to {parent_name} except: Pole B acts with the split composite "
                                    f"(pi_DB + frozen pi_B), k = ceil(N/3) = {k_sym(n)}, Pole A's defender is the "
                                    f"sealed symmetric pi_DA, seeds, output paths")
    return _write_frozen(SD / name, doc)


def _split_rows(dataset: Path) -> dict:
    import numpy as np
    from experiments.run_suite_sharing_distillation import _load_scale
    _man, arr, hold = _load_scale(dataset)
    tr = np.where(~hold)[0]
    return {"train_pole_A": int((arr["pole"][tr] == 0).sum()), "train_pole_B": int((arr["pole"][tr] == 1).sum()),
            "holdout": int(hold.sum())}


def audit_record(n: int) -> Path:
    return SD / f"SUITE_DATASETS_{TAG}_AUDIT_{n}V{n}.json"


def sharing_spec(n: int) -> Path:
    from experiments.run_suite_sharing_distillation import dataset_content_sha256
    parent_name = f"STANDARDIZED_{n}V{n}_SHARING_SPEC.json"
    parent = _load(SD / parent_name)
    man_p = SD / f"SUITE_DISTILLATION_{n}V{n}_{TAG}_DATASET.json"
    man = _load(man_p)
    if man.get("status") != "FROZEN_DATASET" or man.get("dataset_mode") != "symmetric_roles":
        raise SystemExit(f"REFUSING: {man_p.name} is not a FROZEN symmetric dataset")
    aud = _load(audit_record(n))
    if aud.get("verdict") != "GREEN":
        raise SystemExit(f"REFUSING: {audit_record(n).name} is {aud.get('verdict')!r}")
    bl = blocks(n)
    rows = _split_rows(man_p)
    name = f"STANDARDIZED_{n}V{n}_{TAG}_SHARING_SPEC.json"
    doc = copy.deepcopy(parent)
    arms = {}
    for a in ARMS:
        eid, lo, hi = bl[a]
        arms[a] = {**parent["ARMS_locked"][a], "seed_block": eid,
                   "init": f"fresh; seeds {lo}..{hi} ({eid})"}
    doc.update({
        "record_id": name[:-5],
        "status": "FROZEN_BEFORE_TRAINING",
        "utc": _now(),
        "classification": "DIAGNOSTIC (symmetric role construction; post-hoc top-50 evaluation). Not PAPER-FAITHFUL.",
        "decided_by": "PI, 2026-10-01: rerun the sharing baselines on the symmetric state distribution",
        "scope": f"{n}v{n} symmetric family only. Same teachers, recipe, split rule and arm order as {parent_name}; "
                 f"only the dataset (symmetric acting) and the seeds differ.",
        "derived_from": parent_name,
        "SCALE": {**parent["SCALE"], "k_defend": k_sym(n)},
        "DATASET_locked": {**parent["DATASET_locked"],
                           "manifest": _rel(man_p.relative_to(ROOT)),
                           "manifest_sha256": _sha(man_p),
                           "content_sha256": dataset_content_sha256(man),
                           "collected_at": man["collector"]["git_sha"],
                           "decision_rows": sum(int(s["decision_rows"]) for s in man["shards"]),
                           "audit_record": _rel(audit_record(n).relative_to(ROOT))},
        "SPLIT_AND_SAMPLING_locked": {**parent["SPLIT_AND_SAMPLING_locked"], "rows": rows},
        "ARMS_locked": arms,
        "SEEDS_locked": {
            "training": {bl[a][0]: _rng(bl[a]) for a in ARMS},
            "evaluation": {"mode": "POST-HOC on the frozen Ours top-50 seeds "
                                   f"(STANDARDIZED_{n}V{n}_{TAG}_SHARING_EVAL_SPEC.json); no fresh evaluation block"},
            "all_must_be": "training blocks registered and RESERVED at preflight",
        },
        "OUTPUTS": {"root": f"artifacts/strategic_demand/sppo/suite_sharing_std/{n}v{n}_sym/<arm tag>/"},
        "LAUNCH": {a: f".venv/Scripts/python.exe experiments/run_suite_sharing_distillation.py --arm {a} "
                      f"--team-size {n} --spec-tag {TAG} --device cuda" for a in ARMS},
        "NOT_AUTHORIZED_BY_THIS_SPEC": ["changing teachers, recipe, split or arm order",
                                        "using pi_DA / pi_DB as teachers",
                                        "claiming confirmatory or PAPER-FAITHFUL status"],
    })
    doc.pop("EVALUATION_planned", None)
    doc.pop("frozen_after", None)
    return _write_frozen(SD / name, doc)


def labels(n: int) -> dict:
    p = f"TOP50_{n}V{n}_{TAG}"
    return {"share_encoder": f"{p}_SHARE_ENCODER", "fully_shared": f"{p}_FULLY_SHARED_Z",
            "generalist": f"{p}_GENERALIST",
            **{f"robust_{k.lower()}": f"{p}_OURS_{k}_{ROBUSTNESS_SEVERITY.upper()}" for k in ROBUSTNESS}}


def eval_spec(n: int) -> Path:
    parent_name = f"STANDARDIZED_{n}V{n}_SHARING_EVAL_SPEC.json"
    parent = _load(SD / parent_name)
    ours = sealed_ours(n)
    e = ours["entry"]
    seeds_file = ROOT / e["seed_ids_file"]
    lo, hi = (int(x) for x in str(e["block"]).split(".."))
    seed_ids = sorted(int(s) for s in json.loads(seeds_file.read_text(encoding="utf-8")))
    lab = labels(n)
    arms = {}
    for a in ARMS:
        fz = SD / "suite_sharing_std" / f"{n}v{n}_sym" / ARM_TAG[a] / "STUDENT_FROZEN.json"
        if not fz.is_file():
            raise SystemExit(f"REFUSING: {a} not frozen ({fz.relative_to(ROOT)})")
        ck = SD / "suite_sharing_std" / f"{n}v{n}_sym" / ARM_TAG[a] / "ckpts" / f"final_{ARM_TAG[a]}_{n}v{n}.pt"
        if not ck.is_file():
            raise SystemExit(f"REFUSING: {ck.relative_to(ROOT)} missing")
        arms[EVAL_ARM_KEY[a]] = {"label": lab[a], "checkpoint": _rel(ck.relative_to(ROOT)), "sha256": _sha(ck),
                                 "format": parent["ARMS"][EVAL_ARM_KEY[a]]["format"]}
    base = {"registry_experiment_id": e["registry_experiment_id"], "block": e["block"],
            "primary_record": e["primary_record"], "seed_ids": seed_ids, "seed_ids_file": e["seed_ids_file"]}
    pins = ours["pins"]
    post_hoc = {lab[a]: {**base, "system": a} for a in ARMS}
    for key, fam in ROBUSTNESS.items():
        post_hoc[lab[f"robust_{key.lower()}"]] = {**base, "system": "symmetric Ours", "perturbation": fam,
                                                   "severity": ROBUSTNESS_SEVERITY}
    common = (f".venv/Scripts/python.exe experiments/eval_specialist_crossover_scaled.py --team-size {n} "
              f"--spec artifacts/strategic_demand/sppo/STANDARDIZED_{n}V{n}_{TAG}_SHARING_EVAL_SPEC.json "
              f"--post-hoc-ablation-spec artifacts/strategic_demand/sppo/STANDARDIZED_{n}V{n}_{TAG}_SHARING_EVAL_SPEC.json "
              f"--seed-list {e['seed_ids_file']} --registry-experiment-id {e['registry_experiment_id']} "
              f"--device cuda --pi-a-path {pins['pi_DA']['path']} --pi-b-path {pins['pi_DB']['path']} "
              f"--role-fixed-for-episode --role-k-defend {k_sym(n)} "
              f"--frozen-attack-path {pins['pi_A']['path']} --frozen-attack-path-sha256 {pins['pi_A']['sha256']} "
              f"--frozen-attack-path-b {pins['pi_B']['path']} --frozen-attack-path-b-sha256 {pins['pi_B']['sha256']} "
              f"--resume")
    name = f"STANDARDIZED_{n}V{n}_{TAG}_SHARING_EVAL_SPEC.json"
    doc = {
        "record_id": name[:-5],
        "status": "FROZEN_BEFORE_EVAL",
        "arm": "POST_HOC_ABLATION",
        "confirmatory": False,
        "utc": _now(),
        "classification": "DIAGNOSTIC (post-hoc, frozen Ours top-50 seeds). Not confirmatory. Not PAPER-FAITHFUL.",
        "decided_by": "PI, 2026-10-01: symmetric baselines on the same 50 scenarios as symmetric Ours",
        "parent": [parent_name, f"STANDARDIZED_{n}V{n}_{TAG}_SHARING_SPEC.json", DIAG_SPEC.name],
        "claim_boundary": f"{n}v{n} symmetric-role baselines on the Ours top-50 seeds; post-hoc diagnostic with a "
                          f"selection bias toward the old system; mean effects only",
        "ARMS": arms,
        "OURS_SYMMETRIC_REFERENCE": {
            "label": f"TOP50_{n}V{n}_SYMMETRIC_OURS", "sealed_record":
                f"TOP50_{n}V{n}_SYMMETRIC_OURS_SPECIALIST_CROSSOVER_EVAL_RESULT.json",
            "pins": pins, "k_defend": k_sym(n),
            "why": "Delta_G and every comparison pair by seed against this sealed run; it is also the nominal "
                   "reference of the robustness runs (deterministic on the same seeds)"},
        "SEEDS": {"registry_experiment_id": e["registry_experiment_id"], "block": e["block"], "n": len(seed_ids),
                  "seed_class": "post_hoc", "seed_ids_file": e["seed_ids_file"],
                  "seed_ids_sha256": _sha(seeds_file)},
        "POST_HOC_MATCHED_ROLE_ABLATIONS": post_hoc,
        "POLES": parent["POLES"],
        "EVALUATION": {**parent["EVALUATION"],
                       "robustness": f"symmetric Ours under each family at the frozen {ROBUSTNESS_SEVERITY} tier "
                                     f"(DEPLOYMENT_ROBUSTNESS_SPEC.json#TIERS); paired within seed against the "
                                     f"nominal symmetric Ours rows",
                       "score_margin": "mean (blue - red) per cell, read from the sealed rows; no new episodes"},
        "LAUNCH": {
            **{a: f".venv/Scripts/python.exe experiments/eval_suite_sharing_crossover.py --team-size {n} "
                  f"--arm {a} --spec-tag {TAG} --device cuda --resume" for a in ARMS},
            **{f"robust_{k.lower()}": f"{common} --label {lab[f'robust_{k.lower()}']} --perturbation {fam} "
                                      f"--severity {ROBUSTNESS_SEVERITY}" for k, fam in ROBUSTNESS.items()},
        },
        "NOT_AUTHORIZED_BY_THIS_SPEC": ["fresh seed spend", "re-running any label", "low/high tiers",
                                        "tuning anything after a result", "claiming confirmatory status"],
    }
    if lo > min(seed_ids) or hi < max(seed_ids):
        raise SystemExit("FAIL-CLOSED: seed_ids outside the block")
    return _write_frozen(SD / name, doc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team-size", type=int, choices=(2, 4, 6))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--reserve", action="store_true")
    g.add_argument("--collection", action="store_true")
    g.add_argument("--sharing", action="store_true")
    g.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.reserve:
        print("reserved:", reserve() or "nothing new")
        return 0
    if a.team_size is None:
        raise SystemExit("--team-size is required")
    fn = collection_spec if a.collection else sharing_spec if a.sharing else eval_spec
    print(f"-> {fn(a.team_size)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
