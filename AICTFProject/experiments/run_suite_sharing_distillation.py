"""Suite distillation: Share-Encoder, Fully Shared+z and the Generalist, one scale at a time.

Same teachers, same frozen state set, same KL objective, same budget
(20 epochs, batch 256, Adam 3e-4, grad-norm clip 1.0, no weight decay).
Only the sharing structure changes.

  python experiments/run_suite_sharing_distillation.py --arm share_encoder --team-size 2 --preflight --device cuda
  python experiments/run_suite_sharing_distillation.py --arm share_encoder --team-size 2 --device cuda

Everything scale-specific -- dataset, teachers, poles, seeds, arm order -- comes from the
scale's frozen STANDARDIZED_<N>V<N>_SHARING_SPEC.json (2026-09-28); a scale without one refuses.
Seeds are spent only out of RESERVED registry blocks the spec names. The preflight verifies
the spec's PREFLIGHT_REQUIRED list, and a real launch re-verifies it before training.
"""
from __future__ import annotations

import argparse
import csv
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
EPOCHS, BATCH, LR, CLIP, WEIGHT_DECAY = 20, 256, 3e-4, 1.0, 0.0
FIT_MIN = 0.50
LADDER_RUNG = {
    "share_encoder": 1,
    "share_backbone": 2,
    "share_macro": 3,
}
#: Seeds each arm's builder consumes: ladder arms draw one fresh branch per z.
N_INIT_SEEDS = {"fully_shared": 1, "generalist": 1, "role_only": 1, "share_encoder": 2, "share_backbone": 2, "share_macro": 2}
ARM_TAG = {
    "fully_shared": "fully_shared_z",
    "generalist": "generalist",
    "role_only": "role_only",
    "share_encoder": "share_encoder",
    "share_backbone": "share_backbone",
    "share_macro": "share_macro",
}
ARM_LABEL = {
    "fully_shared": "FULLY SHARED+z",
    "generalist": "GENERALIST",
    "role_only": "ROLE-ONLY",
    "share_encoder": "SHARE-ENCODER",
    "share_backbone": "SHARE-BACKBONE",
    "share_macro": "SHARE-MACRO",
}
#: Legacy output root: legacy-dataset 2v2 and invalidated 4v4 students. Never read or written.
LEGACY_OUT = SD / "suite_sharing"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


#: --spec-tag SYM selects the symmetric-role family; STAGE4 selects dual-branch teachers + z+r / role-only.
SPEC_TAGS = ("", "SYM", "STAGE4")


def _family(n: int, tag: str = "") -> str:
    if tag == "STAGE4":
        return f"{n}v{n}_stage4"
    return f"{n}v{n}" + (f"_{tag.lower()}" if tag else "")


def spec_path(n: int, tag: str = "") -> Path:
    return SD / f"STANDARDIZED_{n}V{n}{'_' + tag if tag else ''}_SHARING_SPEC.json"


def load_spec(n: int, arm: str, tag: str = "") -> dict:
    """The scale's frozen sharing spec. The dataset path comes from here, never from N: an
    N-derived name resolved 4v4 to the invalidated SUITE_DISTILLATION_4V4_DATASET.json."""
    p = spec_path(n, tag)
    if not p.is_file():
        raise SystemExit(f"FAIL-CLOSED: {p.name} missing; every scale needs its own frozen sharing spec")
    spec = json.loads(p.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: {p.name} not frozen: {spec.get('status')!r}")
    if arm not in spec.get("ARM_ORDER", []) or arm not in spec.get("ARMS_locked", {}):
        raise SystemExit(f"REFUSING: {arm} is not an arm of {p.name} ({spec.get('ARM_ORDER')})")
    return spec


def dataset_content_sha256(man: dict) -> str:
    """sha256 over '<file>\\t<sha256>' of every listed shard -- binds a spec to the exact bytes."""
    lines = sorted(f"{s['file']}\t{_sha(ROOT / s['file'])}" for s in man["shards"])
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def arm_seeds(spec: dict, arm: str) -> tuple[str, list[int]]:
    """The arm's registry block and its init seeds (every seed of the block, in order)."""
    from experiments import seed_registry as SR
    eid = spec["ARMS_locked"][arm]["seed_block"]
    b = next((x for x in SR.load()["blocks"] if x["experiment_id"] == eid), None)
    if b is None:
        raise SystemExit(f"FAIL-CLOSED: seed block {eid} is not registered")
    seeds = list(range(int(b["lo"]), int(b["hi"]) + 1))
    if len(seeds) != N_INIT_SEEDS[arm] or spec["SEEDS_locked"]["training"].get(eid) != f"{b['lo']}..{b['hi']}":
        raise SystemExit(f"FAIL-CLOSED: {eid} {b['lo']}..{b['hi']} disagrees with the spec or the arm's "
                         f"{N_INIT_SEEDS[arm]} init seed(s)")
    return eid, seeds


def spec_checks(spec: dict, arm: str, n: int, dataset: Path, man: dict, n_rows: dict, tag: str = "") -> dict:
    """PREFLIGHT_REQUIRED items that do not need the model. A real launch re-runs these."""
    from experiments import audit_suite_datasets_cross_scale as AU
    from experiments import code_identity as CI
    from experiments import seed_registry as SR
    ds, c = spec["DATASET_locked"], {}
    c["dataset_is_the_spec_manifest"] = dataset.resolve() == (ROOT / ds["manifest"]).resolve()
    c["dataset_not_forbidden"] = not any(dataset.name in f for f in spec["FORBIDDEN_INPUTS"])
    c["dataset_FROZEN"] = man.get("status") == ds["status_required"]
    c["dataset_manifest_sha256"] = _sha(dataset) == ds["manifest_sha256"]
    c["dataset_content_sha256"] = dataset_content_sha256(man) == ds["content_sha256"]
    audit = json.loads((ROOT / ds["audit_record"]).read_text(encoding="utf-8"))
    c["audit_record_GREEN"] = audit.get("verdict") == "GREEN"
    scale = _family(n, tag)
    live: list = []
    AU.audit_one(scale, *AU.DATASETS[scale], live)
    c["live_dataset_audit_all_pass"] = (AU.DATASETS[scale][1] == dataset.name
                                        and bool(live) and all(x["ok"] for x in live))
    c["pole_hashes_equal_spec"] = all(
        (man.get("poles") or {}).get(p, {}).get("pole_config_hash") == spec["POLES_locked"][p]["pole_config_hash"]
        for p in ("A", "B"))
    t = spec["TEACHERS_locked"]
    if t.get("mode") == "dual_branch_role_gated" or (
        isinstance(t.get("pi_A"), dict) and "defend" in (t.get("pi_A") or {})
    ):
        ok = man.get("teachers", {}).get("mode") == "dual_branch_role_gated"
        for side in ("pi_A", "pi_B"):
            for half in ("defend", "attack"):
                mp = (man.get("teachers") or {}).get(side, {}).get(half) or {}
                sp = t[side][half]
                ok = ok and mp.get("sha256") == sp["sha256"] and mp.get("path") == sp["path"]
                ok = ok and _sha(ROOT / sp["path"]) == sp["sha256"]
        c["teachers_equal_spec_and_manifest"] = bool(ok)
        c["teachers_are_dual_branch"] = True
    else:
        c["teachers_equal_spec_and_manifest"] = all(
            man["teachers"][k]["sha256"] == t[k]["sha256"] and man["teachers"][k]["path"] == t[k]["path"]
            and _sha(ROOT / t[k]["path"]) == t[k]["sha256"] for k in ("pi_A", "pi_B"))
        c["teachers_are_dual_branch"] = False
    c["N_and_k"] = (int(man.get("team_size", -1)) == n == spec["SCALE"]["N"]
                    and (man.get("allocator") or {}).get("k_defend") == spec["SCALE"]["k_defend"])
    r = spec["RECIPE_locked"]
    c["recipe_equals_spec"] = (r["epochs"], r["batch"], r["lr"], r["grad_clip_norm"], r["weight_decay"]) == \
        (EPOCHS, BATCH, LR, CLIP, WEIGHT_DECAY) and r["optimizer"] == "Adam"
    c["split_rows_equal_spec"] = n_rows == spec["SPLIT_AND_SAMPLING_locked"]["rows"]
    c["scientific_tree_clean"] = not CI.scientific_dirty(ROOT)
    ev = spec["SEEDS_locked"]["evaluation"]              # block ids map to "lo..hi"; other keys are notes
    blocks = [arm_seeds(spec, arm)[0], *(k for k, v in ev.items() if isinstance(v, str) and ".." in v
                                         and v.replace("..", "").isdigit())]
    reg = {x["experiment_id"]: x for x in SR.load()["blocks"]}
    c["seed_blocks_RESERVED"] = all(reg.get(b, {}).get("status") == "RESERVED" for b in blocks)
    if arm == "generalist":
        d = ROOT / spec["ARMS_locked"][arm]["definition"]
        c["generalist_definition_sha256"] = d.is_file() and _sha(d) == spec["ARMS_locked"][arm]["definition_sha256"]
    if tag == "STAGE4" and arm == "generalist":
        c["stage4_forbids_generalist"] = False
    elif tag == "STAGE4":
        c["stage4_forbids_generalist"] = arm != "generalist"
    return c


def _paths(arm: str, n: int, spec_tag: str = ""):
    tag = ARM_TAG[arm]
    if spec_tag == "STAGE4" and arm == "fully_shared":
        tag = "fully_shared_z_r"
    out = SD / "suite_sharing_std" / _family(n, spec_tag) / tag
    return {
        "out": out,
        "ckpt": out / "ckpts" / f"final_{tag}_{n}v{n}.pt",
        "metrics": out / "metrics.csv",
        "preflight": out / "preflight.json",
        "frozen": out / "STUDENT_FROZEN.json",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        required=True,
        choices=tuple(ARM_TAG),
    )
    ap.add_argument("--team-size", type=int, required=True, choices=(2, 4, 6))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--spec-tag", default="", choices=SPEC_TAGS,
                    help="SYM = symmetric-role family; STAGE4 = dual-branch teachers + z+r/role-only")
    args = ap.parse_args()
    stag = str(args.spec_tag)
    n = int(args.team_size)
    arm = str(args.arm)
    if stag == "STAGE4" and arm == "generalist":
        raise SystemExit("REFUSING: Stage 4 forbids Generalist pi(a|o); use role_only")
    device = args.device
    paths = _paths(arm, n, stag)
    is_ladder = arm in LADDER_RUNG

    # Which arms exist at a scale, their order, seeds and dataset are the scale's frozen spec --
    # not tables here. A scale or arm the spec does not name refuses.
    spec = load_spec(n, arm, stag)
    seed_block, init_seeds = arm_seeds(spec, arm)
    dataset = ROOT / spec["DATASET_locked"]["manifest"]
    if not dataset.is_file():
        raise SystemExit(f"FAIL-CLOSED: {dataset.name} (the spec's dataset) is missing")
    if not args.preflight:
        order = spec["ARM_ORDER"]
        missing = [a for a in order[:order.index(arm)] if not _paths(a, n, stag)["frozen"].is_file()]
        if missing:
            raise SystemExit(f"REFUSING: arm order {order}; freeze {missing} before {arm}")

    import torch
    import experiments.r2_learned_crossover as R2
    import experiments.run_teacher_distillation as RTD
    from rl import teacher_distillation as TD
    from rl.custom_ppo import load_custom_ppo_policy

    # One loader and one agent-count propagation at every scale (the former `n == 2` route
    # used RTD.load_dataset(), which reads the legacy 2v2 manifest layout).
    import experiments.collect_distillation_states as C
    C.N_AGENTS = n
    R2.AGENTS = n

    man, arr, hold = _load_scale(dataset)
    train_idx = np.where(~hold)[0]
    hold_idx = np.where(hold)[0]
    tspec = man["teachers"]

    n_rows = {"train_pole_A": int((arr["pole"][train_idx] == 0).sum()),
              "train_pole_B": int((arr["pole"][train_idx] == 1).sum()), "holdout": int(hold_idx.size)}
    gate = spec_checks(spec, arm, n, dataset, man, n_rows, stag)
    if not args.preflight and not all(gate.values()):
        raise SystemExit(f"REFUSING: spec checks fail at launch: {[k for k, v in gate.items() if not v]}")

    probe = R2.build_env(device, init_seeds[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()

    dual_teachers = (
        tspec.get("mode") == "dual_branch_role_gated"
        or (isinstance(tspec.get("pi_A"), dict) and "defend" in tspec["pi_A"])
    )
    teachers: dict = {"mode": "dual_branch_role_gated"} if dual_teachers else {}
    if dual_teachers:
        for name in ("pi_A", "pi_B"):
            teachers[name] = {}
            for half in ("defend", "attack"):
                ck = ROOT / tspec[name][half]["path"]
                if _sha(ck) != tspec[name][half]["sha256"]:
                    raise SystemExit(f"REFUSING: {name}.{half} sha mismatch")
                pol = load_custom_ppo_policy(str(ck), obs_space, act_space, device=device)
                pol.model.eval()
                for p in pol.model.parameters():
                    p.requires_grad_(False)
                teachers[name][half] = pol.model
        arch_src = str(ROOT / tspec["pi_A"]["defend"]["path"])
    else:
        for name in ("pi_A", "pi_B"):
            ck = ROOT / tspec[name]["path"]
            if _sha(ck) != tspec[name]["sha256"]:
                raise SystemExit(f"REFUSING: {name} sha mismatch")
            pol = load_custom_ppo_policy(str(ck), obs_space, act_space, device=device)
            pol.model.eval()
            for p in pol.model.parameters():
                p.requires_grad_(False)
            teachers[name] = pol.model
        arch_src = str(ROOT / tspec["pi_A"]["path"])

    if not args.preflight and paths["frozen"].is_file():
        raise SystemExit(f"REFUSING: {paths['frozen']} exists; one-shot")
    if not args.preflight and not paths["preflight"].is_file():
        raise SystemExit("REFUSING: run --preflight first")

    if arm in ("fully_shared", "generalist", "role_only"):
        from rl import suite_fully_shared_distill as FS
        seed = init_seeds[0]
        branch_seeds = None
        rung = None
        role_on = bool(stag == "STAGE4" and arm == "fully_shared")

        if arm == "role_only":
            build_fn, save_fn, load_fn = FS.build_role_only_student, FS.save_role_only, FS.load_role_only
        elif arm == "fully_shared":
            build_fn, save_fn, load_fn = FS.build_fully_shared_student, FS.save_fully_shared, FS.load_fully_shared
        else:
            build_fn, save_fn, load_fn = FS.build_generalist_student, FS.save_generalist, FS.load_generalist

        def build_student():
            if arm == "fully_shared" and role_on:
                return build_fn(arch_src, obs_space, act_space, seed=seed, device=device,
                                role_conditioning=True)
            return build_fn(arch_src, obs_space, act_space, seed=seed, device=device)

        def actor_params(m):
            return TD.actor_parameters(m)

        def critic_params(m):
            return TD.critic_parameters(m)

        def save_student(m, cfg, kw, path, prov):
            save_fn(m, cfg, kw, path, {**prov, "suite_arm": arm, "team_size": n,
                                       "stage4": stag == "STAGE4"})

        def load_student(path):
            out = load_fn(path, obs_space, act_space, device=device)
            # load_role_only -> (model, cfg, payload); load_fully_shared/generalist -> (model, payload).
            if len(out) == 3:
                model, _cfg, payload = out
                return model, payload
            return out

        def unique_count(m, actor):
            return sum(int(p.numel()) for _, p in actor)

        def shared_params(m):
            return []

        def private_params(m, z):
            return []

        def share_report(m):
            return None

    else:
        from rl import ladder_rung1 as L1
        rung = int(LADDER_RUNG[arm])
        branch_seeds = tuple(init_seeds)
        seed = branch_seeds[0]

        def build_student():
            if rung == 1:
                model, cfg, kw, _ref = L1.build_rung1(
                    arch_src, obs_space, act_space,
                    seeds=branch_seeds, device=device,
                )
            else:
                model, cfg, kw, _ref = L1.build_rung(
                    rung, arch_src, obs_space, act_space,
                    seeds=branch_seeds, device=device,
                )
            return model, cfg, kw

        def actor_params(m):
            return L1.actor_parameters(m)

        def critic_params(m):
            return L1.critic_parameters(m)

        def save_student(m, cfg, kw, path, prov):
            payload_prov = {**prov, "suite_arm": arm, "team_size": n, "rung": rung,
                            "stage4": stag == "STAGE4"}
            if rung == 1:
                L1.save_rung1(m, cfg, kw, path, payload_prov)
            else:
                L1.save_rung(rung, m, cfg, kw, path, payload_prov)

        def load_student(path):
            if rung == 1:
                model, cfg, payload = L1.load_rung1(path, obs_space, act_space, device=device)
            else:
                model, cfg, payload = L1.load_rung(rung, path, obs_space, act_space, device=device)
            return model, payload

        def unique_count(m, actor):
            if rung == 1:
                return int(L1.sharing_arithmetic(m)["n_unique"])
            return int(L1.sharing_arithmetic_generic(m)["n_unique"])

        def shared_params(m):
            if rung == 1:
                return L1.shared_parameters(m)
            return L1.shared_module_parameters(m)

        def private_params(m, z):
            if rung == 1:
                return L1.private_actor_parameters(m, z)
            return L1.private_actor_parameters_generic(m, z)

        def share_report(m):
            if rung == 1:
                return L1.sharing_arithmetic(m)
            return L1.sharing_arithmetic_generic(m)

    model, spec_cfg, kw = build_student()
    model.train()
    actor = actor_params(model)
    critic = critic_params(model)
    for _, p in critic:
        p.requires_grad_(False)
    opt = torch.optim.Adam([p for _, p in actor], lr=LR, weight_decay=WEIGHT_DECAY)
    n_unique = unique_count(model, actor)

    label = ARM_LABEL[arm]
    print(f"SUITE {label} {n}v{n} {'PREFLIGHT' if args.preflight else 'TRAIN'} {_now()} device={device}")
    print(f"  rows train={train_idx.size} holdout={hold_idx.size} unique actor params={n_unique:,}")
    if is_ladder:
        share = share_report(model)
        mods = share.get("shared_modules") if share else None
        print(
            f"  rung={rung} shared={mods or 'actor_cnn'} arithmetic_ok={share['ok']} "
            f"(unique={share['n_unique']:,})",
            flush=True,
        )

    def fidelity(idx) -> dict:
        model.eval()
        acc, tot = {}, 0
        with torch.no_grad():
            for s in range(0, int(idx.size), 256):
                j = idx[s:s + 256]
                obs, dm = RTD.to_torch(arr, j, device)
                d = TD.fidelity_diagnostics(model, teachers, obs, dm)
                for k, v in d.items():
                    acc[k] = acc.get(k, 0.0) + v * int(j.size)
                tot += int(j.size)
        model.train()
        return {k: v / max(1, tot) for k, v in acc.items()}

    idx0 = train_idx[:64]
    obs0, dm0 = RTD.to_torch(arr, idx0, device)

    if args.preflight:
        checks = {f"spec:{k}": bool(v) for k, v in gate.items()}
        checks["dataset_both_poles"] = bool(
            train_idx.size > 0 and hold_idx.size > 0
            and (arr["pole"][train_idx] == 0).any() and (arr["pole"][train_idx] == 1).any()
        )
        if dual_teachers:
            checks["dataset_has_roles"] = bool("roles" in arr)
            checks["student_role_conditioning"] = bool(
                getattr(model, "role_conditioning_enabled", False)
                or (hasattr(model, "branch") and all(
                    getattr(model.branch[z], "role_conditioning_enabled", False) for z in ("z0", "z1")
                ))
            )
        with torch.no_grad():
            if dual_teachers:
                la = TD.dual_branch_composite_logits(
                    teachers["pi_A"]["defend"], teachers["pi_A"]["attack"], obs0
                )
            else:
                la = TD.head_logits(teachers["pi_A"], obs0)
            self_kl, _ = TD.masked_mean(TD.kl_per_head(la, la), dm0)
            z0 = torch.zeros((int(dm0.shape[0]),), dtype=torch.long, device=device)
            z1 = torch.ones((int(dm0.shape[0]),), dtype=torch.long, device=device)
            if arm == "role_only":
                a = torch.cat([t.reshape(t.shape[0], -1) for t in TD.head_logits(model, obs0)], -1)
                b = a
            else:
                a = torch.cat([t.reshape(t.shape[0], -1) for t in TD.head_logits(model, obs0, z_idx=z0)], -1)
                b = torch.cat([t.reshape(t.shape[0], -1) for t in TD.head_logits(model, obs0, z_idx=z1)], -1)
        checks["teacher_self_kl_zero"] = bool(float(self_kl) == 0.0)
        if arm in ("generalist", "role_only"):
            checks["z_has_no_pathway"] = bool(float((a - b).abs().max()) == 0.0)
        else:
            checks["z_changes_logits"] = bool(float((a - b).abs().max()) > 0.0)
        if arm in ("fully_shared", "generalist", "role_only"):
            checks["single_actor_no_second_branch"] = bool(not hasattr(model, "branch"))
        else:
            from rl import ladder_rung1 as L1
            share = share_report(model)
            if rung == 1:
                checks["encoder_shared_identity_and_arithmetic"] = bool(
                    model.encoder_is_shared() and share["ok"]
                )
            else:
                checks["shared_modules_identity_and_arithmetic"] = bool(
                    model.modules_are_shared() and share["ok"]
                )
            if rung == 3:
                h0 = model.branch["z0"].latent_actor.action_head
                h1 = model.branch["z1"].latent_actor.action_head
                checks["macro_head_tied_target_head_private"] = bool(
                    getattr(h0, "macro_head", None) is getattr(h1, "macro_head", None)
                    and getattr(h0, "target_head", None) is not getattr(h1, "target_head", None)
                )
                checks["shared_set_proper_superset_of_rung2"] = bool(
                    set(L1.SHARED_BY_RUNG[2]).issubset(set(L1.SHARED_BY_RUNG[3]))
                    and len(L1.SHARED_BY_RUNG[3]) > len(L1.SHARED_BY_RUNG[2])
                )
        loss, _diag = TD.distillation_loss(model, teachers, obs0, dm0)
        checks["loss_finite_positive"] = bool(torch.isfinite(loss) and float(loss.detach()) > 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g = lambda ps: any(p.grad is not None and float(p.grad.abs().max()) > 0 for _, p in ps)
        if arm in ("fully_shared", "generalist", "role_only"):
            checks["grad_actor_not_critic"] = bool(g(actor) and not g(critic))
        else:
            shared = shared_params(model)
            priv0 = private_params(model, 0)
            priv1 = private_params(model, 1)
            checks["grad_shared_and_both_private_none_critic"] = bool(
                g(shared) and g(priv0) and g(priv1) and not g(critic)
            )
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            torch.nn.utils.clip_grad_norm_([p for _, p in actor], CLIP)
            opt.step()
            opt.zero_grad(set_to_none=True)
            moved = lambda ps: any(not torch.equal(before[name], p.detach()) for name, p in ps)
            still = lambda ps: all(torch.equal(before[name], p.detach()) for name, p in ps)
            checks["step_moves_shared_and_private_critic_still"] = bool(
                moved(shared) and moved(priv0) and moved(priv1) and still(critic)
            )
        paths["out"].mkdir(parents=True, exist_ok=True)
        tmp = paths["out"] / "_preflight_roundtrip.pt"
        save_student(model, spec_cfg, kw, str(tmp), {"preflight": True})
        loaded, _ = load_student(str(tmp))
        worst = 0.0
        with torch.no_grad():
            for z in (0, 1):
                zt = torch.full((int(dm0.shape[0]),), z, dtype=torch.long, device=device)
                if arm == "role_only":
                    xs = TD.head_logits(model, obs0)
                    ys = TD.head_logits(loaded, obs0)
                else:
                    xs = TD.head_logits(model, obs0, z_idx=zt)
                    ys = TD.head_logits(loaded, obs0, z_idx=zt)
                for x, y in zip(xs, ys):
                    worst = max(worst, float((x - y).abs().max()))
                if arm == "role_only":
                    break
        tmp.unlink(missing_ok=True)
        checks["roundtrip"] = bool(worst <= 1e-5)
        n_pass = sum(bool(v) for v in checks.values())
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        paths["preflight"].write_text(json.dumps({
            "record": f"suite {arm} {n}v{n} preflight",
            "utc": _now(), "spec": spec_path(n, stag).name, "spec_sha256": _sha(spec_path(n, stag)),
            "seed_block": seed_block, "init_seeds": init_seeds,
            "checks": checks, "passed": f"{n_pass}/{len(checks)}",
            "initial_loss": float(loss.detach()), "unique_actor_params": n_unique,
            "branch_seeds": list(branch_seeds) if branch_seeds else None,
            "rung": rung,
            "stage4": stag == "STAGE4",
            "VERDICT": "PASS" if n_pass == len(checks) else "FAIL",
        }, indent=2), encoding="utf-8")
        print(f"  {n_pass}/{len(checks)} -> {paths['preflight']}")
        return 0 if n_pass == len(checks) else 1

    if json.loads(paths["preflight"].read_text(encoding="utf-8")).get("VERDICT") != "PASS":
        raise SystemExit("REFUSING: preflight did not pass")
    # Rebuild a fresh student; preflight may have stepped weights.
    model, spec_cfg, kw = build_student()
    model.train()
    actor = actor_params(model)
    critic = critic_params(model)
    for _, p in critic:
        p.requires_grad_(False)
    opt = torch.optim.Adam([p for _, p in actor], lr=LR, weight_decay=WEIGHT_DECAY)
    n_unique = unique_count(model, actor)

    batches = RTD.Batches(arr, train_idx, batch=BATCH, seed=seed)
    print(f"  epochs={EPOCHS} batch={BATCH} lr={LR} clip={CLIP} updates/epoch={batches.n_per_epoch()}",
          flush=True)
    from experiments.tqdm_loop import set_postfix, tqdm_iter
    rows = []
    epoch_bar = tqdm_iter(range(EPOCHS), desc=f"distill {arm} {n}v{n}", total=EPOCHS, unit="epoch")
    for ep in epoch_bar:
        tr_a = tr_b = 0.0
        n_b = 0
        for idx in batches.epoch():
            obs, dm = RTD.to_torch(arr, idx, device)
            loss, diag = TD.distillation_loss(model, teachers, obs, dm)
            if not torch.isfinite(loss):
                raise SystemExit(f"REFUSING: non-finite loss at epoch {ep}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in actor], CLIP)
            opt.step()
            tr_a += diag["kl_A"]
            tr_b += diag["kl_B"]
            n_b += 1
        h = fidelity(hold_idx)
        row = {
            "epoch": ep + 1,
            "train_kl_A": tr_a / max(1, n_b),
            "train_kl_B": tr_b / max(1, n_b),
            **{f"holdout_{k}": v for k, v in h.items()},
        }
        rows.append(row)
        set_postfix(
            epoch_bar,
            f"KL {row['train_kl_A']:.3f}/{row['train_kl_B']:.3f} "
            f"agree {h['agree_z0_vs_piA']:.3f}/{h['agree_z1_vs_piB']:.3f}",
        )
        print(
            f"  epoch {ep+1:2d}  train KL {row['train_kl_A']:.4f}/{row['train_kl_B']:.4f}  "
            f"agree {h['agree_z0_vs_piA']:.3f}/{h['agree_z1_vs_piB']:.3f}",
            flush=True,
        )
    paths["out"].mkdir(parents=True, exist_ok=True)
    paths["ckpt"].parent.mkdir(parents=True, exist_ok=True)
    with paths["metrics"].open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    final = rows[-1]
    fit_ok = (
        final["holdout_agree_z0_vs_piA"] >= FIT_MIN
        and final["holdout_agree_z1_vs_piB"] >= FIT_MIN
    )
    save_student(model, spec_cfg, kw, str(paths["ckpt"]), {
        "arm": arm, "team_size": n, "seed": seed,
        "branch_seeds": list(branch_seeds) if branch_seeds else None,
        "rung": rung,
        "epochs": EPOCHS, "batch": BATCH, "lr": LR, "clip": CLIP,
        "dataset": str(dataset.relative_to(ROOT)), "fit_ok": fit_ok, "final": final,
    })
    paths["frozen"].write_text(json.dumps({
        "record": f"suite {arm} {n}v{n}",
        "utc": _now(), "status": "FROZEN_STUDENT" if fit_ok else "FIT_FAILED",
        "checkpoint": str(paths["ckpt"].relative_to(ROOT)),
        "sha256": _sha(paths["ckpt"]),
        "unique_actor_params": n_unique,
        "branch_seeds": list(branch_seeds) if branch_seeds else None,
        "rung": rung,
        "final_holdout": final,
        "recipe": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "clip": CLIP, "weight_decay": WEIGHT_DECAY},
        "dataset": str(dataset.relative_to(ROOT)),
        "dataset_manifest_sha256": _sha(dataset),
        "spec": spec_path(n, stag).name, "spec_sha256": _sha(spec_path(n, stag)),
        "seed_block": seed_block, "init_seeds": init_seeds,
    }, indent=2), encoding="utf-8")
    from experiments import seed_registry as SR
    SR.set_status(seed_block, "SPENT", f"{paths['frozen'].name} written ({arm} {n}v{n})")
    print(f"  -> {paths['frozen']} fit_ok={fit_ok}; {seed_block} SPENT")
    return 0 if fit_ok else 2


def _load_scale(dataset_path: Path):
    import experiments.run_teacher_distillation as RTD
    man = json.loads(dataset_path.read_text(encoding="utf-8"))
    if man.get("status") != "FROZEN_DATASET":
        raise SystemExit(f"REFUSING: dataset status {man.get('status')!r}")
    optional = []
    for sh in man["shards"]:
        d0 = np.load(ROOT / sh["file"])
        if int(d0["step"].shape[0]) == 0:
            continue
        optional = [k for k in RTD.OPTIONAL_OBS_KEYS if k in d0.files]
        break
    parts = {k: [] for k in RTD.OBS_KEYS + tuple(optional) + ("decision_mask", "pole", "episode")}
    for sh in man["shards"]:
        d = np.load(ROOT / sh["file"])
        if int(d["step"].shape[0]) == 0:
            continue
        for k in parts:
            parts[k].append(d[k])
    arr = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    keep = arr["decision_mask"].any(axis=1)
    if not keep.all():
        arr = {k: v[keep] for k, v in arr.items()}
    hold = (arr["episode"] % 10) == 9
    return man, arr, hold


if __name__ == "__main__":
    raise SystemExit(main())
