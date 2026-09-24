"""Cross-scale suite distillation: Fully Shared+z and Share-Encoder.

Same teachers, same frozen state set, same KL objective, same budget
(20 epochs, batch 256, Adam 3e-4, grad-norm clip 1.0, no weight decay).
Only the sharing structure changes.

  python experiments/run_suite_sharing_distillation.py --arm fully_shared --team-size 2 --preflight --device cpu
  python experiments/run_suite_sharing_distillation.py --arm fully_shared --team-size 2 --device cpu
  python experiments/run_suite_sharing_distillation.py --arm share_encoder --team-size 2

2v2 Share-Encoder is the sealed Rung-1 student (reused, not retrained).
4v4/6v6 require SUITE_DISTILLATION_{n}V{n}_DATASET.json (matched collection
under CLOSEST_DEFENDS). They are fail-closed until that manifest exists.
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
SEEDS = {2: 11_980_001, 4: 22_580_001, 6: 22_680_001}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _dataset_for(n: int) -> Path:
    if n == 2:
        return SD / "TEACHER_DISTILLATION_DATASET.json"
    return SD / f"SUITE_DISTILLATION_{n}V{n}_DATASET.json"


def _paths(arm: str, n: int):
    tag = "fully_shared_z" if arm == "fully_shared" else "share_encoder"
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
    ap.add_argument("--arm", required=True, choices=("fully_shared", "share_encoder"))
    ap.add_argument("--team-size", type=int, required=True, choices=(2, 4, 6))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    n = int(args.team_size)
    arm = str(args.arm)
    device = args.device
    paths = _paths(arm, n)

    if arm == "share_encoder" and n == 2:
        sealed = SD / "sharing_ladder" / "rung1" / "ckpts" / "final_rung1.pt"
        rec = SD / "RUNG1_STUDENT_FROZEN.json"
        if not sealed.is_file() or not rec.is_file():
            raise SystemExit("REFUSING: 2v2 Share-Encoder seal missing")
        print(f"2v2 Share-Encoder already sealed; reuse {sealed}")
        print(f"  record {rec.name}")
        return 0

    dataset = _dataset_for(n)
    if not dataset.is_file():
        raise SystemExit(
            f"FAIL-CLOSED: {dataset.name} is missing. "
            f"{n}v{n} suite distillation needs the matched teacher-state set "
            f"(4v4/6v6: collected under CLOSEST_DEFENDS k={'2' if n==4 else '1'})."
        )

    import torch
    import experiments.r2_learned_crossover as R2
    import experiments.run_teacher_distillation as RTD
    from rl import teacher_distillation as TD
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.suite_fully_shared_distill import (
        build_fully_shared_student, load_fully_shared, save_fully_shared,
    )

    if n != 2:
        import experiments.collect_distillation_states as C
        C.N_AGENTS = n
        R2.AGENTS = n

    man, arr, hold = RTD.load_dataset() if n == 2 else _load_scale(dataset)
    train_idx = np.where(~hold)[0]
    hold_idx = np.where(hold)[0]
    tspec = man["teachers"]
    seed = SEEDS[n]

    probe = R2.build_env(device, seed)
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

    if arm != "fully_shared":
        raise SystemExit(
            f"FAIL-CLOSED: Share-Encoder at {n}v{n} is not launched by this runner "
            "until SUITE collection exists and a construction amendment is frozen. "
            "2v2 reuses the sealed Rung-1 student."
        )

    if not args.preflight and paths["frozen"].is_file():
        raise SystemExit(f"REFUSING: {paths['frozen']} exists; one-shot")
    if not args.preflight and not paths["preflight"].is_file():
        raise SystemExit("REFUSING: run --preflight first")

    model, spec_cfg, kw = build_fully_shared_student(
        str(ROOT / tspec["pi_A"]["path"]), obs_space, act_space, seed=seed, device=device,
    )
    model.train()
    actor = TD.actor_parameters(model)
    critic = TD.critic_parameters(model)
    for _, p in critic:
        p.requires_grad_(False)
    opt = torch.optim.Adam([p for _, p in actor], lr=LR)
    n_unique = sum(int(p.numel()) for _, p in actor)

    print(f"SUITE FULLY SHARED+z {n}v{n} {'PREFLIGHT' if args.preflight else 'TRAIN'} {_now()} device={device}")
    print(f"  rows train={train_idx.size} holdout={hold_idx.size} unique actor params={n_unique:,}")

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
        checks["single_actor_no_second_branch"] = bool(not hasattr(model, "branch"))
        loss, diag = TD.distillation_loss(model, teachers, obs0, dm0)
        checks["loss_finite_positive"] = bool(torch.isfinite(loss) and float(loss.detach()) > 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g_actor = any(p.grad is not None and float(p.grad.abs().max()) > 0 for _, p in actor)
        g_critic = any(p.grad is not None and float(p.grad.abs().max()) > 0 for _, p in critic)
        checks["grad_actor_not_critic"] = bool(g_actor and not g_critic)
        paths["out"].mkdir(parents=True, exist_ok=True)
        tmp = paths["out"] / "_preflight_roundtrip.pt"
        save_fully_shared(model, spec_cfg, kw, str(tmp), {"preflight": True})
        loaded, _ = load_fully_shared(str(tmp), obs_space, act_space, device=device)
        worst = 0.0
        with torch.no_grad():
            for z in (0, 1):
                zt = torch.full((int(dm0.shape[0]),), z, dtype=torch.long, device=device)
                for x, y in zip(TD.head_logits(model, obs0, z_idx=zt), TD.head_logits(loaded, obs0, z_idx=zt)):
                    worst = max(worst, float((x - y).abs().max()))
        tmp.unlink(missing_ok=True)
        checks["roundtrip"] = bool(worst <= 1e-5)
        n_pass = sum(bool(v) for v in checks.values())
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        paths["preflight"].write_text(json.dumps({
            "record": f"suite fully_shared_z {n}v{n} preflight",
            "utc": _now(), "checks": checks, "passed": f"{n_pass}/{len(checks)}",
            "initial_loss": float(loss.detach()), "unique_actor_params": n_unique,
            "VERDICT": "PASS" if n_pass == len(checks) else "FAIL",
        }, indent=2), encoding="utf-8")
        print(f"  {n_pass}/{len(checks)} -> {paths['preflight']}")
        return 0 if n_pass == len(checks) else 1

    if json.loads(paths["preflight"].read_text(encoding="utf-8")).get("VERDICT") != "PASS":
        raise SystemExit("REFUSING: preflight did not pass")
    # Rebuild a fresh student; preflight stepped the in-memory weights.
    model, spec_cfg, kw = build_fully_shared_student(
        str(ROOT / tspec["pi_A"]["path"]), obs_space, act_space, seed=seed, device=device,
    )
    model.train()
    actor = TD.actor_parameters(model)
    critic = TD.critic_parameters(model)
    for _, p in critic:
        p.requires_grad_(False)
    opt = torch.optim.Adam([p for _, p in actor], lr=LR)

    batches = RTD.Batches(arr, train_idx, batch=BATCH, seed=seed)
    print(f"  epochs={EPOCHS} batch={BATCH} lr={LR} clip={CLIP} updates/epoch={batches.n_per_epoch()}", flush=True)
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
        row = {"epoch": ep + 1, "train_kl_A": tr_a / max(1, n_b), "train_kl_B": tr_b / max(1, n_b),
               **{f"holdout_{k}": v for k, v in h.items()}}
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
    fit_ok = (final["holdout_agree_z0_vs_piA"] >= FIT_MIN and final["holdout_agree_z1_vs_piB"] >= FIT_MIN)
    save_fully_shared(model, spec_cfg, kw, str(paths["ckpt"]), {
        "arm": "fully_shared_z", "team_size": n, "seed": seed,
        "epochs": EPOCHS, "batch": BATCH, "lr": LR, "clip": CLIP,
        "dataset": str(dataset.relative_to(ROOT)), "fit_ok": fit_ok, "final": final,
    })
    paths["frozen"].write_text(json.dumps({
        "record": f"suite fully_shared_z {n}v{n}",
        "utc": _now(), "status": "FROZEN_STUDENT" if fit_ok else "FIT_FAILED",
        "checkpoint": str(paths["ckpt"].relative_to(ROOT)),
        "sha256": _sha(paths["ckpt"]),
        "unique_actor_params": n_unique,
        "final_holdout": final,
        "recipe": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "clip": CLIP, "weight_decay": 0.0},
    }, indent=2), encoding="utf-8")
    print(f"  -> {paths['frozen']} fit_ok={fit_ok}")
    return 0 if fit_ok else 2


def _load_scale(dataset_path: Path):
    import experiments.run_teacher_distillation as RTD
    man = json.loads(dataset_path.read_text(encoding="utf-8"))
    if man.get("status") != "FROZEN_DATASET":
        raise SystemExit(f"REFUSING: dataset status {man.get('status')!r}")
    parts = {k: [] for k in RTD.OBS_KEYS + ("decision_mask", "pole", "episode")}
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
