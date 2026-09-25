"""Cross-scale suite distillation: Fully Shared+z and ladder sharing arms.

Same teachers, same frozen state set, same KL objective, same budget
(20 epochs, batch 256, Adam 3e-4, grad-norm clip 1.0, no weight decay).
Only the sharing structure changes.

  python experiments/run_suite_sharing_distillation.py --arm fully_shared --team-size 2 --preflight --device cpu
  python experiments/run_suite_sharing_distillation.py --arm fully_shared --team-size 4 --device cuda
  python experiments/run_suite_sharing_distillation.py --arm share_encoder --team-size 4 --preflight --device cuda
  python experiments/run_suite_sharing_distillation.py --arm share_backbone --team-size 4 --preflight --device cuda
  python experiments/run_suite_sharing_distillation.py --arm share_macro --team-size 4 --preflight --device cuda

2v2 Share-Encoder is the sealed Rung-1 student (reused, not retrained).
4v4 Share-Encoder / Backbone / Macro require their construction amendments
and SUITE_DISTILLATION_4V4_DATASET.json.
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
EPOCHS, BATCH, LR, CLIP = 20, 256, 3e-4, 1.0
FIT_MIN = 0.50
# Fully Shared+z: single init seed. Ladder arms: (z0, z1) branch seeds.
SEEDS_FULLY = {2: 11_980_001, 4: 22_580_001, 6: 22_680_001}
SEEDS_LADDER = {
    "share_encoder": {
        2: (11_961_001, 11_961_002),  # sealed 2v2 Rung-1 (not retrained here)
        4: (22_581_001, 22_581_002),
        6: (22_681_001, 22_681_002),
    },
    "share_backbone": {
        4: (22_582_001, 22_582_002),
    },
    "share_macro": {
        4: (22_583_001, 22_583_002),
    },
}
LADDER_RUNG = {
    "share_encoder": 1,
    "share_backbone": 2,
    "share_macro": 3,
}
LADDER_AMEND = {
    "share_encoder": {4: SD / "SUITE_SHARE_ENCODER_4V4_CONSTRUCTION_AMENDMENT.json"},
    "share_backbone": {4: SD / "SUITE_SHARE_BACKBONE_4V4_CONSTRUCTION_AMENDMENT.json"},
    "share_macro": {4: SD / "SUITE_SHARE_MACRO_4V4_CONSTRUCTION_AMENDMENT.json"},
}
ARM_TAG = {
    "fully_shared": "fully_shared_z",
    "share_encoder": "share_encoder",
    "share_backbone": "share_backbone",
    "share_macro": "share_macro",
}
ARM_LABEL = {
    "fully_shared": "FULLY SHARED+z",
    "share_encoder": "SHARE-ENCODER",
    "share_backbone": "SHARE-BACKBONE",
    "share_macro": "SHARE-MACRO",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _dataset_for(n: int) -> Path:
    if n == 2:
        return SD / "TEACHER_DISTILLATION_DATASET.json"
    return SD / f"SUITE_DISTILLATION_{n}V{n}_DATASET.json"


def _paths(arm: str, n: int):
    tag = ARM_TAG[arm]
    out = SD / "suite_sharing" / f"{n}v{n}" / tag
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
        choices=("fully_shared", "share_encoder", "share_backbone", "share_macro"),
    )
    ap.add_argument("--team-size", type=int, required=True, choices=(2, 4, 6))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    n = int(args.team_size)
    arm = str(args.arm)
    device = args.device
    paths = _paths(arm, n)
    is_ladder = arm in LADDER_RUNG

    if arm == "share_encoder" and n == 2:
        sealed = SD / "sharing_ladder" / "rung1" / "ckpts" / "final_rung1.pt"
        rec = SD / "RUNG1_STUDENT_FROZEN.json"
        if not sealed.is_file() or not rec.is_file():
            raise SystemExit("REFUSING: 2v2 Share-Encoder seal missing")
        print(f"2v2 Share-Encoder already sealed; reuse {sealed}")
        print(f"  record {rec.name}")
        return 0

    if is_ladder and arm != "share_encoder" and n != 4:
        raise SystemExit(
            f"REFUSING: {arm} suite distill is only authorized at 4v4 "
            f"(got team-size={n})."
        )

    if is_ladder:
        amend_path = LADDER_AMEND.get(arm, {}).get(n)
        if amend_path is None or not amend_path.is_file():
            raise SystemExit(
                f"FAIL-CLOSED: {arm} at {n}v{n} needs a frozen construction "
                f"amendment (missing {amend_path.name if amend_path else 'record'})."
            )
        amend = json.loads(amend_path.read_text(encoding="utf-8"))
        if not str(amend.get("status", "")).startswith("FROZEN"):
            raise SystemExit(f"REFUSING: {amend_path.name} not frozen: {amend.get('status')!r}")

    dataset = _dataset_for(n)
    if not dataset.is_file():
        raise SystemExit(
            f"FAIL-CLOSED: {dataset.name} is missing. "
            f"{n}v{n} suite distillation needs the matched teacher-state set "
            f"(4v4/6v6: collected under CLOSEST_DEFENDS k={'2' if n == 4 else '1'})."
        )

    import torch
    import experiments.r2_learned_crossover as R2
    import experiments.run_teacher_distillation as RTD
    from rl import teacher_distillation as TD
    from rl.custom_ppo import load_custom_ppo_policy

    if n != 2:
        import experiments.collect_distillation_states as C
        C.N_AGENTS = n
        R2.AGENTS = n

    man, arr, hold = RTD.load_dataset() if n == 2 else _load_scale(dataset)
    train_idx = np.where(~hold)[0]
    hold_idx = np.where(hold)[0]
    tspec = man["teachers"]

    probe_seed = (
        SEEDS_FULLY[n] if arm == "fully_shared" else SEEDS_LADDER[arm][n][0]
    )
    probe = R2.build_env(device, probe_seed)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()

    teachers = {}
    for name in ("pi_A", "pi_B"):
        ck = ROOT / tspec[name]["path"]
        if _sha(ck) != tspec[name]["sha256"]:
            raise SystemExit(f"REFUSING: {name} sha mismatch")
        pol = load_custom_ppo_policy(str(ck), obs_space, act_space, device=device)
        pol.model.eval()
        for p in pol.model.parameters():
            p.requires_grad_(False)
        teachers[name] = pol.model

    if not args.preflight and paths["frozen"].is_file():
        raise SystemExit(f"REFUSING: {paths['frozen']} exists; one-shot")
    if not args.preflight and not paths["preflight"].is_file():
        raise SystemExit("REFUSING: run --preflight first")

    if arm == "fully_shared":
        from rl.suite_fully_shared_distill import (
            build_fully_shared_student, load_fully_shared, save_fully_shared,
        )
        seed = SEEDS_FULLY[n]
        branch_seeds = None
        rung = None

        def build_student():
            return build_fully_shared_student(
                str(ROOT / tspec["pi_A"]["path"]), obs_space, act_space,
                seed=seed, device=device,
            )

        def actor_params(m):
            return TD.actor_parameters(m)

        def critic_params(m):
            return TD.critic_parameters(m)

        def save_student(m, cfg, kw, path, prov):
            save_fully_shared(m, cfg, kw, path, {**prov, "suite_arm": arm, "team_size": n})

        def load_student(path):
            return load_fully_shared(path, obs_space, act_space, device=device)

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
        branch_seeds = SEEDS_LADDER[arm][n]
        seed = branch_seeds[0]

        def build_student():
            if rung == 1:
                model, cfg, kw, _ref = L1.build_rung1(
                    str(ROOT / tspec["pi_A"]["path"]), obs_space, act_space,
                    seeds=branch_seeds, device=device,
                )
            else:
                model, cfg, kw, _ref = L1.build_rung(
                    rung, str(ROOT / tspec["pi_A"]["path"]), obs_space, act_space,
                    seeds=branch_seeds, device=device,
                )
            return model, cfg, kw

        def actor_params(m):
            return L1.actor_parameters(m)

        def critic_params(m):
            return L1.critic_parameters(m)

        def save_student(m, cfg, kw, path, prov):
            payload_prov = {**prov, "suite_arm": arm, "team_size": n, "rung": rung}
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
    opt = torch.optim.Adam([p for _, p in actor], lr=LR)
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
        checks = {}
        checks["dataset_both_poles"] = bool(
            train_idx.size > 0 and hold_idx.size > 0
            and (arr["pole"][train_idx] == 0).any() and (arr["pole"][train_idx] == 1).any()
        )
        with torch.no_grad():
            la = TD.head_logits(teachers["pi_A"], obs0)
            self_kl, _ = TD.masked_mean(TD.kl_per_head(la, la), dm0)
            z0 = torch.zeros((int(dm0.shape[0]),), dtype=torch.long, device=device)
            z1 = torch.ones((int(dm0.shape[0]),), dtype=torch.long, device=device)
            a = torch.cat([t.reshape(t.shape[0], -1) for t in TD.head_logits(model, obs0, z_idx=z0)], -1)
            b = torch.cat([t.reshape(t.shape[0], -1) for t in TD.head_logits(model, obs0, z_idx=z1)], -1)
        checks["teacher_self_kl_zero"] = bool(float(self_kl) == 0.0)
        checks["z_changes_logits"] = bool(float((a - b).abs().max()) > 0.0)
        if arm == "fully_shared":
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
        if arm == "fully_shared":
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
                for x, y in zip(
                    TD.head_logits(model, obs0, z_idx=zt),
                    TD.head_logits(loaded, obs0, z_idx=zt),
                ):
                    worst = max(worst, float((x - y).abs().max()))
        tmp.unlink(missing_ok=True)
        checks["roundtrip"] = bool(worst <= 1e-5)
        n_pass = sum(bool(v) for v in checks.values())
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        paths["preflight"].write_text(json.dumps({
            "record": f"suite {arm} {n}v{n} preflight",
            "utc": _now(), "checks": checks, "passed": f"{n_pass}/{len(checks)}",
            "initial_loss": float(loss.detach()), "unique_actor_params": n_unique,
            "branch_seeds": list(branch_seeds) if branch_seeds else None,
            "rung": rung,
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
    opt = torch.optim.Adam([p for _, p in actor], lr=LR)
    n_unique = unique_count(model, actor)

    batches = RTD.Batches(arr, train_idx, batch=BATCH, seed=seed)
    print(f"  epochs={EPOCHS} batch={BATCH} lr={LR} clip={CLIP} updates/epoch={batches.n_per_epoch()}",
          flush=True)
    rows = []
    for ep in range(EPOCHS):
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
        "recipe": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "clip": CLIP, "weight_decay": 0.0},
        "dataset": str(dataset.relative_to(ROOT)),
    }, indent=2), encoding="utf-8")
    print(f"  -> {paths['frozen']} fit_ok={fit_ok}")
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
