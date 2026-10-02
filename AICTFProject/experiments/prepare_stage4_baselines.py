r"""Freeze Stage-4 (z vs r) sharing suite specs under dual-branch teachers.

Stage 4 starts only after dual-branch teachers seal. Teachers ARE the dual-branch
composites (role-gated ATTACK+DEFEND), not the 1M foundations. Students:

  Share-Encoder, Fully Shared+z+r := pi(a|o,z,r), Role-only := pi(a|o,r)

    python experiments/prepare_stage4_baselines.py --reserve
    python experiments/prepare_stage4_baselines.py --team-size 6 --collection
    python experiments/prepare_stage4_baselines.py --team-size 6 --sharing
    python experiments/prepare_stage4_baselines.py --team-size 6 --eval

Authorized per scale by STAGE4_<N>V<N>_FULL_SUITE_SPEC.json (6v6: STAGE4_6V6_SCHOOL_SUITE_SPEC.json).
Same ladder at 2v2, 4v4, and 6v6. Only N, k=ceil(N/3), checkpoints, and seeds differ.
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
TAG = "STAGE4"
COLLECTION_PARENT = {
    2: "SUITE_DISTILLATION_2V2_SPEC.json",
    4: "SUITE_DISTILLATION_4V4_V2_SPEC.json",
    6: "SUITE_DISTILLATION_6V6_SPEC.json",
}
ARMS = ("share_encoder", "fully_shared", "role_only")
N_INIT = {"share_encoder": 2, "fully_shared": 1, "role_only": 1}
INTERP = SD / "STAGE4_SHARING_INTERPRETATION_V1.json"
AUTH_FOR = {
    2: SD / "STAGE4_2V2_FULL_SUITE_SPEC.json",
    4: SD / "STAGE4_4V4_FULL_SUITE_SPEC.json",
    6: SD / "STAGE4_6V6_SCHOOL_SUITE_SPEC.json",
}
# Spent confirmatory block the frozen top-50 is a subset of. Student evals are post-hoc
# on that block; the primary record is the full-block confirmatory seal (its seeds.block
# equals the registry range). The dual-branch top-50 result is a subset and cannot be
# the primary.
SPENT_EVAL = {
    2: (
        "STANDARDIZED_2V2_SPLIT_K1_CONFIRMATORY_SPECIALIST_CROSSOVER",
        "STANDARDIZED_2V2_SPLIT_K1_CONFIRMATORY_SPECIALIST_CROSSOVER_EVAL_RESULT.json",
    ),
    4: (
        "DEFEND_ATTACK_SPLIT_POLICY_A_V1_CONFIRMATORY_V1_EVAL",
        "DEFEND_ATTACK_SPLIT_POLICY_A_V1_CONFIRMATORY_V1_SPECIALIST_CROSSOVER_EVAL_RESULT.json",
    ),
    6: (
        "STANDARDIZED_6V6_SEPARATED_EVAL",
        "STANDARDIZED_6V6_SPLIT_K1_CONFIRMATORY_SPECIALIST_CROSSOVER_EVAL_RESULT.json",
    ),
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _rel(p) -> str:
    return str(p).replace("\\", "/")


def k_sym(n: int) -> int:
    return -(-n // 3)


def blocks(n: int) -> dict:
    """Fixed registry blocks for Stage 4 at scale N (100k strides; lane by N)."""
    lane = {2: 0, 4: 1000, 6: 2000}[n]
    out = {
        "collection_A": (f"STAGE4_{n}V{n}_SUITE_DATASET_POLE_A", 27_000_001 + lane, 96),
        "collection_B": (f"STAGE4_{n}V{n}_SUITE_DATASET_POLE_B", 27_100_001 + lane, 96),
        "share_encoder": (f"STAGE4_{n}V{n}_SHARE_ENCODER_TRAINING", 27_200_001 + lane, N_INIT["share_encoder"]),
        "fully_shared": (f"STAGE4_{n}V{n}_FULLY_SHARED_ZR_TRAINING", 27_300_001 + lane, N_INIT["fully_shared"]),
        "role_only": (f"STAGE4_{n}V{n}_ROLE_ONLY_TRAINING", 27_400_001 + lane, N_INIT["role_only"]),
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
                    raise SystemExit(
                        f"REFUSING: {eid} registered as {reg[eid]['lo']}..{reg[eid]['hi']}, not {lo}..{hi}"
                    )
                continue
            SR.allocate(
                eid, lo, hi, "exploratory",
                purpose=f"Stage-4 dual-branch sharing suite {n}v{n}: {purpose}",
                spec="experiments/prepare_stage4_baselines.py",
            )
            done.append(eid)
    return done


def _load(p: Path) -> dict:
    if not p.is_file():
        raise SystemExit(f"FAIL-CLOSED: {p.name} missing")
    return json.loads(p.read_text(encoding="utf-8"))


def _write_frozen(path: Path, doc: dict) -> Path:
    if path.is_file():
        old = json.loads(path.read_text(encoding="utf-8"))
        strip = lambda d: {k: v for k, v in d.items() if k != "utc"}  # noqa: E731
        if strip(old) != strip(doc):
            raise SystemExit(f"REFUSING: {path.name} exists with different content; specs are frozen once")
        return path
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path


def sealed_teachers(n: int) -> dict:
    """Dual-branch teachers from the school deploy manifest (or SEALED Stage-4 teacher record)."""
    seal = SD / f"STAGE4_{n}V{n}_TEACHERS_SEALED.json"
    if seal.is_file():
        rec = _load(seal)
        if rec.get("status") != "SEALED":
            raise SystemExit(f"REFUSING: {seal.name} is {rec.get('status')!r}")
        pins = rec["pins"]
    else:
        man_p = ROOT / f"{n}v{n}" / "dual_branch_deploy_manifest.json"
        if not man_p.is_file():
            raise SystemExit(
                f"REFUSING: need {seal.name} or {man_p.relative_to(ROOT)} after dual-branch export"
            )
        man = _load(man_p)
        if man.get("architecture") != "DUAL_BRANCH_ROLE_COMPOSITE_V1":
            raise SystemExit(f"REFUSING: {man_p.name} is not DUAL_BRANCH_ROLE_COMPOSITE_V1")
        pins = {
            "pi_A_defend": man["pi_A_defend"],
            "pi_A_attack": man["pi_A_attack"],
            "pi_B_defend": man["pi_B_defend"],
            "pi_B_attack": man["pi_B_attack"],
        }
    for name, pin in pins.items():
        f = ROOT / pin["path"]
        if not f.is_file() or _sha(f) != pin["sha256"]:
            raise SystemExit(f"REFUSING: {name} {pin['path']} missing or sha mismatch")
    return pins


def collection_spec(n: int) -> Path:
    from experiments import seed_registry as SR
    if n not in COLLECTION_PARENT:
        raise SystemExit(f"REFUSING: Stage-4 collection freeze currently authorized for N in "
                         f"{sorted(COLLECTION_PARENT)}, got {n}")
    parent = _load(SD / COLLECTION_PARENT[n])
    pins = sealed_teachers(n)
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
        "classification": "DIAGNOSTIC Stage-4 dataset under dual-branch teachers. Post-hoc; "
                          "not confirmatory. Not PAPER-FAITHFUL.",
        "authorized_by": f"{AUTH_FOR[n].name} + {INTERP.name}",
        "parent": [COLLECTION_PARENT[n], INTERP.name, AUTH_FOR[n].name,
                   f"{n}v{n}/dual_branch_deploy_manifest.json"],
        "derived_from": COLLECTION_PARENT[n],
        "ALLOCATOR_locked": {
            **parent["ALLOCATOR_locked"],
            "k_defend": k_sym(n),
            "source": "k = ceil(N/3) (DUAL_BRANCH_ROLE_COMPOSITE_V1)",
        },
        "ACTING_DEPLOYMENT_locked": {
            "construction": "dual_branch",
            "why": "both poles visited by dual-branch composites; KL teachers are those composites",
            "Pole_A": {
                "policy": "DEFEND_ATTACK_SPLIT composite",
                "pi_D": pins["pi_A_defend"],
                "frozen_attack_pi_A": pins["pi_A_attack"],
                "splice": "rl.custom_ppo.split_attack_defend.splice_actions; deterministic=True",
            },
            "Pole_B": {
                "policy": "DEFEND_ATTACK_SPLIT composite",
                "pi_D": pins["pi_B_defend"],
                "frozen_attack_pi_B": pins["pi_B_attack"],
                "splice": "rl.custom_ppo.split_attack_defend.splice_actions; deterministic=True",
            },
            "teachers_are_the_acting_composites": True,
        },
        "KL_TEACHERS_locked": {
            "mode": "dual_branch_role_gated",
            "note": "Stage 4: role-gated dual-branch composites are the KL targets "
                    "(not the 1M foundation specialists).",
            "pi_A": {"defend": pins["pi_A_defend"], "attack": pins["pi_A_attack"], "z": 0},
            "pi_B": {"defend": pins["pi_B_defend"], "attack": pins["pi_B_attack"], "z": 1},
            "teacher_targets_not_cached": True,
        },
        "SEEDS": {
            "collection_A": _rng(bl["collection_A"]),
            "collection_B": _rng(bl["collection_B"]),
            "registry_experiment_ids": {
                "collection_A": bl["collection_A"][0],
                "collection_B": bl["collection_B"][0],
            },
            "seed_class": "exploratory",
            "chosen_by": "prepare_stage4_baselines.blocks",
        },
        "LAUNCH": {
            "command": (
                f".venv/Scripts/python.exe experiments/collect_suite_distillation_states.py "
                f"--team-size {n} --dataset-tag {TAG} --device cuda --resume"
            ),
        },
        "OUTPUTS": {
            "dataset_manifest": f"SUITE_DISTILLATION_{n}V{n}_{TAG}_DATASET.json",
            "state_shards": f"artifacts/strategic_demand/sppo/suite_distillation_stage4/{n}v{n}/states/",
        },
        "NOT_AUTHORIZED_BY_THIS_SPEC": [
            "using foundation specialists as KL teachers",
            "presenting historical Fully Shared+z / Generalist as Stage 4",
            "claiming confirmatory or PAPER-FAITHFUL status",
        ],
    })
    for k in ("SAME_RECIPE_AS_2V2", "SMOKE_before_collection"):
        doc.pop(k, None)
    return _write_frozen(SD / name, doc)


def audit_record(n: int) -> Path:
    return SD / f"SUITE_DATASETS_{TAG}_AUDIT_{n}V{n}.json"


def sharing_spec(n: int) -> Path:
    from experiments.run_suite_sharing_distillation import dataset_content_sha256
    parent_name = f"STANDARDIZED_{n}V{n}_SHARING_SPEC.json"
    parent = _load(SD / parent_name)
    man_p = SD / f"SUITE_DISTILLATION_{n}V{n}_{TAG}_DATASET.json"
    man = _load(man_p)
    if man.get("status") != "FROZEN_DATASET" or man.get("dataset_mode") != "dual_branch_teachers":
        raise SystemExit(f"REFUSING: {man_p.name} is not a FROZEN dual_branch Stage-4 dataset")
    aud = _load(audit_record(n))
    if aud.get("verdict") != "GREEN":
        raise SystemExit(f"REFUSING: {audit_record(n).name} is {aud.get('verdict')!r}")
    bl = blocks(n)
    pins = sealed_teachers(n)
    import numpy as np
    from experiments.run_suite_sharing_distillation import _load_scale
    _man, arr, hold = _load_scale(man_p)
    tr = np.where(~hold)[0]
    rows = {
        "train_pole_A": int((arr["pole"][tr] == 0).sum()),
        "train_pole_B": int((arr["pole"][tr] == 1).sum()),
        "holdout": int(hold.sum()),
    }
    name = f"STANDARDIZED_{n}V{n}_{TAG}_SHARING_SPEC.json"
    doc = copy.deepcopy(parent)
    arms = {
        "share_encoder": {
            **parent["ARMS_locked"]["share_encoder"],
            "seed_block": bl["share_encoder"][0],
            "init": f"fresh; seeds {_rng(bl['share_encoder'])}",
            "role_conditioning": True,
            "arch_source": "pi_A_defend (dual-branch)",
        },
        "fully_shared": {
            "label": "Fully Shared+z+r",
            "builder": "rl/suite_fully_shared_distill.py::build_fully_shared_student(role_conditioning=True)",
            "policy": "pi(a | o, z, r)",
            "seed_block": bl["fully_shared"][0],
            "init": f"fresh; seeds {_rng(bl['fully_shared'])}",
        },
        "role_only": {
            "label": "Role-only",
            "builder": "rl/suite_fully_shared_distill.py::build_role_only_student",
            "policy": "pi(a | o, r)",
            "seed_block": bl["role_only"][0],
            "init": f"fresh; seeds {_rng(bl['role_only'])}",
            "controlled_difference_vs_fully_shared": "z removed",
        },
    }
    doc.update({
        "record_id": name[:-5],
        "status": "FROZEN_BEFORE_TRAINING",
        "utc": _now(),
        "classification": "DIAGNOSTIC Stage-4 sharing ladder under dual-branch teachers. Not PAPER-FAITHFUL.",
        "decided_by": f"PI via {INTERP.name} + {AUTH_FOR[n].name}",
        "scope": f"{n}v{n} Stage-4 family. Teachers = dual-branch composites; arms = Share-Encoder, "
                 f"Fully Shared+z+r, Role-only.",
        "derived_from": parent_name,
        "SCALE": {**parent["SCALE"], "k_defend": k_sym(n)},
        "ARM_ORDER": list(ARMS),
        "DATASET_locked": {
            **parent["DATASET_locked"],
            "manifest": _rel(man_p.relative_to(ROOT)),
            "manifest_sha256": _sha(man_p),
            "content_sha256": dataset_content_sha256(man),
            "collected_at": man["collector"]["git_sha"],
            "decision_rows": sum(int(s["decision_rows"]) for s in man["shards"]),
            "audit_record": _rel(audit_record(n).relative_to(ROOT)),
        },
        "TEACHERS_locked": {
            "mode": "dual_branch_role_gated",
            "pi_A": {"defend": pins["pi_A_defend"], "attack": pins["pi_A_attack"]},
            "pi_B": {"defend": pins["pi_B_defend"], "attack": pins["pi_B_attack"]},
            "must_equal": "the dataset manifest's teachers block",
        },
        "SPLIT_AND_SAMPLING_locked": {**parent["SPLIT_AND_SAMPLING_locked"], "rows": rows},
        "ARMS_locked": arms,
        "SEEDS_locked": {
            "training": {bl[a][0]: _rng(bl[a]) for a in ARMS},
            "evaluation": {
                "mode": (
                    f"POST-HOC on the frozen Ours top-50 seeds "
                    f"(STANDARDIZED_{n}V{n}_{TAG}_SHARING_EVAL_SPEC.json); no fresh evaluation block"
                ),
            },
            "all_must_be": "training blocks registered and RESERVED at preflight",
        },
        "OUTPUTS": {"root": f"artifacts/strategic_demand/sppo/suite_sharing_std/{n}v{n}_stage4/<arm tag>/"},
        "LAUNCH": {
            a: (
                f".venv/Scripts/python.exe experiments/run_suite_sharing_distillation.py "
                f"--arm {a} --team-size {n} --spec-tag {TAG} --device cuda"
            )
            for a in ARMS
        },
        "NOT_AUTHORIZED_BY_THIS_SPEC": [
            "Generalist pi(a|o) as a Stage-4 rung",
            "foundation specialists as teachers",
            "claiming confirmatory or PAPER-FAITHFUL status",
        ],
    })
    doc.pop("EVALUATION_planned", None)
    doc.pop("frozen_after", None)
    return _write_frozen(SD / name, doc)


def labels(n: int) -> dict:
    p = f"TOP50_{n}V{n}_{TAG}"
    return {
        "share_encoder": f"{p}_SHARE_ENCODER",
        "fully_shared": f"{p}_FULLY_SHARED_ZR",
        "role_only": f"{p}_ROLE_ONLY",
    }


def eval_spec(n: int) -> Path:
    """Post-hoc top-50 eval for Stage-4 students (diagnostic). Pins must already be frozen."""
    parent_name = f"STANDARDIZED_{n}V{n}_SHARING_EVAL_SPEC.json"
    parent = _load(SD / parent_name)
    # Same frozen Ours top-50 seed list the dual-branch diagnostic uses.
    seeds_file = ROOT / f"artifacts/strategic_demand/sppo/symmetric_role_top50/{n}v{n}_ours_top50_seed_ids.json"
    if not seeds_file.is_file():
        raise SystemExit(f"REFUSING: {seeds_file} missing")
    seed_ids = sorted(int(s) for s in json.loads(seeds_file.read_text(encoding="utf-8")))
    reg_id, primary = SPENT_EVAL[n]
    # Block range from the SPENT Separated eval registry entry.
    from experiments import seed_registry as SR
    b = next((x for x in SR.load()["blocks"] if x["experiment_id"] == reg_id), None)
    if b is None:
        raise SystemExit(f"REFUSING: {reg_id} not registered")
    block = f"{b['lo']}..{b['hi']}"
    lab = labels(n)
    arm_dir = {
        "share_encoder": "share_encoder",
        "fully_shared": "fully_shared_z_r",
        "role_only": "role_only",
    }
    arm_key = {
        "share_encoder": "share_encoder",
        "fully_shared": "fully_shared_z_r",
        "role_only": "role_only",
    }
    fmt = {
        "share_encoder": "sharing_ladder_rung1_v1",
        "fully_shared": "suite_fully_shared_z_r_v1",
        "role_only": "suite_role_only_v1",
    }
    arms = {}
    for a in ARMS:
        tag = arm_dir[a]
        fz = SD / "suite_sharing_std" / f"{n}v{n}_stage4" / tag / "STUDENT_FROZEN.json"
        if not fz.is_file():
            raise SystemExit(f"REFUSING: {a} not frozen ({fz.relative_to(ROOT)})")
        ck = SD / "suite_sharing_std" / f"{n}v{n}_stage4" / tag / "ckpts" / f"final_{tag}_{n}v{n}.pt"
        if not ck.is_file():
            raise SystemExit(f"REFUSING: {ck.relative_to(ROOT)} missing")
        arms[arm_key[a]] = {
            "label": lab[a],
            "checkpoint": _rel(ck.relative_to(ROOT)),
            "sha256": _sha(ck),
            "format": fmt[a],
        }
    base = {
        "registry_experiment_id": reg_id,
        "block": block,
        "primary_record": primary,
        "seed_ids": seed_ids,
        "seed_ids_file": _rel(seeds_file.relative_to(ROOT)),
    }
    post_hoc = {lab[a]: {**base, "system": a} for a in ARMS}
    name = f"STANDARDIZED_{n}V{n}_{TAG}_SHARING_EVAL_SPEC.json"
    doc = {
        "record_id": name[:-5],
        "status": "FROZEN_BEFORE_EVAL",
        "arm": "POST_HOC_ABLATION",
        "confirmatory": False,
        "utc": _now(),
        "classification": "DIAGNOSTIC Stage-4 student crossover on frozen top-50 seeds. Not confirmatory.",
        "decided_by": AUTH_FOR[n].name,
        "parent": [parent_name, f"STANDARDIZED_{n}V{n}_{TAG}_SHARING_SPEC.json", AUTH_FOR[n].name],
        "ARMS": arms,
        "SEEDS": {
            "registry_experiment_id": reg_id,
            "block": block,
            "n": len(seed_ids),
            "seed_class": "post_hoc",
            "seed_ids_file": _rel(seeds_file.relative_to(ROOT)),
            "seed_ids_sha256": _sha(seeds_file),
        },
        "POST_HOC_MATCHED_ROLE_ABLATIONS": post_hoc,
        "POLES": parent["POLES"],
        "EVALUATION": parent.get("EVALUATION", {}),
        "LAUNCH": {
            a: (
                f".venv/Scripts/python.exe experiments/eval_suite_sharing_crossover.py "
                f"--team-size {n} --arm {a} --spec-tag {TAG} --device cuda --resume"
            )
            for a in ARMS
        },
        "NOT_AUTHORIZED_BY_THIS_SPEC": [
            "fresh seed spend",
            "Generalist as a Stage-4 rung",
            "claiming confirmatory status",
        ],
    }
    return _write_frozen(SD / name, doc)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reserve", action="store_true")
    ap.add_argument("--team-size", type=int, choices=(2, 4, 6), default=None)
    ap.add_argument("--collection", action="store_true")
    ap.add_argument("--sharing", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.reserve:
        done = reserve()
        print(f"reserved {len(done)} blocks: {done or '(already present)'}")
        return 0
    if a.team_size is None:
        raise SystemExit("need --team-size N with --collection / --sharing / --eval, or --reserve")
    n = int(a.team_size)
    if a.collection:
        p = collection_spec(n)
        print(f"froze {p.name}")
    if a.sharing:
        p = sharing_spec(n)
        print(f"froze {p.name}")
    if a.eval:
        p = eval_spec(n)
        print(f"froze {p.name}")
    if not (a.collection or a.sharing or a.eval):
        raise SystemExit("nothing to do")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
