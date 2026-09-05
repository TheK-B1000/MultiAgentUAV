"""Sharing-ladder Rung 1: preflight (8 checks) and the frozen 20-epoch distillation.

Implements RUNG1_CONSTRUCTION_AMENDMENT.json. Same loss, data, schedule and fit check as the
offline TD run (TEACHER_DISTILLATION_SPEC.json); the only change is the architecture -- a shared
CNN encoder with everything else private per z (rl/ladder_rung1.py).

    --preflight   8 mechanical checks incl. encoder-sharing identity + arithmetic, gradients
                  reaching the SHARED encoder and BOTH private stacks with NONE on any critic,
                  and a save/load round-trip; trains nothing
    (default)     20 fixed epochs, fit check, freeze (sha) -> RUNG1_STUDENT_FROZEN.json

Run:  python experiments/run_ladder_rung1_distillation.py --preflight --device cpu
      python experiments/run_ladder_rung1_distillation.py --device cpu
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

from experiments.run_teacher_distillation import Batches, load_dataset, to_torch

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
AMEND_BY_RUNG = {1: SD / "RUNG1_CONSTRUCTION_AMENDMENT.json",
                 2: SD / "RUNG2_CONSTRUCTION_AND_PREDICTION.json"}
SEEDS_BY_RUNG = {1: (11_961_001, 11_961_002), 2: (11_963_001, 11_963_002)}
LADDER = SD / "SHARING_LADDER_SPEC.json"
SPECIALISTS = SD / "SPECIALIST_BASELINE_SPEC.json"
EPOCHS, BATCH, LR, CLIP = 20, 256, 3e-4, 1.0
FIT_MIN_AGREEMENT = 0.50


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rung", type=int, default=1, choices=(1, 2))
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    device = args.device
    RUNG = int(args.rung)
    SEEDS = SEEDS_BY_RUNG[RUNG]
    AMEND = AMEND_BY_RUNG[RUNG]
    OUT_DIR = SD / "sharing_ladder" / f"rung{RUNG}"
    CKPT = OUT_DIR / "ckpts" / f"final_rung{RUNG}.pt"
    METRICS = OUT_DIR / f"rung{RUNG}_metrics.csv"
    PREFLIGHT_OUT = SD / f"RUNG{RUNG}_PREFLIGHT.json"
    FROZEN_OUT = SD / f"RUNG{RUNG}_STUDENT_FROZEN.json"

    amend = json.loads(AMEND.read_text(encoding="utf-8"))
    if amend["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: Rung {RUNG} construction record not frozen: {amend['status']!r}")
    if json.loads(LADDER.read_text(encoding="utf-8"))["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit("REFUSING: ladder spec not frozen")
    if not args.preflight and FROZEN_OUT.is_file():
        raise SystemExit(f"REFUSING: {FROZEN_OUT.name} exists; Rung {RUNG} training is one-shot")
    if not args.preflight and not PREFLIGHT_OUT.is_file():
        raise SystemExit("REFUSING: run --preflight first")

    import torch
    import experiments.r2_learned_crossover as R2
    from rl import ladder_rung1 as L1
    from rl import teacher_distillation as TD
    from rl.custom_ppo import load_custom_ppo_policy

    man, arr, hold = load_dataset()
    train_idx, hold_idx = np.where(~hold)[0], np.where(hold)[0]
    tspec = json.loads(SPECIALISTS.read_text(encoding="utf-8"))["MODELS_UNDER_TEST"]
    tdata = man["teachers"]

    probe = R2.build_env(device, 11_961_001)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()

    teachers = {}
    for n in ("pi_A", "pi_B"):
        p = ROOT / tspec[n]["path"]
        if _sha(p) != tspec[n]["sha256"] or tdata[n]["sha256"] != tspec[n]["sha256"]:
            raise SystemExit(f"REFUSING: {n} sha mismatch")
        pol = load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
        pol.model.eval()
        for q in pol.model.parameters():
            q.requires_grad_(False)
        teachers[n] = pol.model

    model, branch_cfg, kwargs, pi_a_sd = L1.build_rung(RUNG, str(ROOT / tspec["pi_A"]["path"]), obs_space,
                                                       act_space, seeds=SEEDS, device=device)
    model.train()
    actor = L1.actor_parameters(model)
    critic = L1.critic_parameters(model)
    shared = L1.shared_module_parameters(model)
    priv0, priv1 = L1.private_actor_parameters_generic(model, 0), L1.private_actor_parameters_generic(model, 1)
    for _, p in actor:
        p.requires_grad_(True)
    for _, p in critic:
        p.requires_grad_(False)
    opt = torch.optim.Adam([p for _, p in actor], lr=LR)
    share = L1.sharing_arithmetic_generic(model)

    print(f"RUNG {RUNG} {'PREFLIGHT' if args.preflight else 'DISTILLATION'}  {_now()}  device={device}")
    print(f"  rows train={train_idx.size} holdout={hold_idx.size}")
    print(f"  params: shared {share['shared_modules']} {len(shared)} tensors ({share['n_shared']:,} values), private z0 {len(priv0)}, "
          f"private z1 {len(priv1)}, critic {len(critic)} (excluded); unique={share['n_unique']:,} "
          f"expected={share['expected_unique']:,} -> {'OK' if share['ok'] else 'MISMATCH'}")

    def fidelity(idx) -> dict:
        model.eval()
        acc, n = {}, 0
        with torch.no_grad():
            for s in range(0, idx.size, 512):
                j = idx[s:s + 512]
                obs, dm = to_torch(arr, j, device)
                d = TD.fidelity_diagnostics(model, teachers, obs, dm)
                for k, v in d.items():
                    acc[k] = acc.get(k, 0.0) + v * int(j.size)
                n += int(j.size)
        model.train()
        return {k: v / max(1, n) for k, v in acc.items()}

    # ---------------------------------------------------------------- preflight
    if args.preflight:
        checks = {}
        checks["1_dataset"] = bool(train_idx.size > 0 and hold_idx.size > 0
                                   and (arr["pole"][train_idx] == 0).any() and (arr["pole"][train_idx] == 1).any())
        checks["2_branches_match_specialist_arch_and_differ_from_piA"] = True   # build_rung1 raised otherwise
        checks["3_shared_modules_identity_and_arithmetic"] = bool(model.modules_are_shared() and share["ok"])
        # mixed-z batch: half the rows as z0, half as z1 -- both branches must receive gradient
        idx = train_idx[:64]
        obs, dm = to_torch(arr, idx, device)
        with torch.no_grad():
            la = TD.head_logits(teachers["pi_A"], obs)
            self_kl, _ = TD.masked_mean(TD.kl_per_head(la, la), dm)
        checks["4_teacher_self_kl_zero"] = bool(float(self_kl) == 0.0)
        loss, diag = TD.distillation_loss(model, teachers, obs, dm)
        checks["5_initial_loss_finite_positive"] = bool(torch.isfinite(loss) and float(loss.detach()) > 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        g = lambda ps: any(p.grad is not None and float(p.grad.abs().max()) > 0 for _, p in ps)
        checks["6_grad_on_shared_and_both_private_none_on_critic"] = bool(
            g(shared) and g(priv0) and g(priv1) and not g(critic))
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        torch.nn.utils.clip_grad_norm_([p for _, p in actor], CLIP)
        opt.step(); opt.zero_grad(set_to_none=True)
        moved = lambda ps: any(not torch.equal(before[n], p.detach()) for n, p in ps)
        still = lambda ps: all(torch.equal(before[n], p.detach()) for n, p in ps)
        checks["7_step_moves_shared_and_both_private_critic_still"] = bool(
            moved(shared) and moved(priv0) and moved(priv1) and still(critic))
        tmp = OUT_DIR / "ckpts" / "_preflight_roundtrip.pt"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        L1.save_rung(RUNG, model, branch_cfg, kwargs, str(tmp), {"preflight": True})
        loaded, _, _ = L1.load_rung(RUNG, str(tmp), obs_space, act_space, device=device)
        worst = 0.0
        for z in (0, 1):
            for a, b in zip(L1.logits_for_z(model, obs, z, device), L1.logits_for_z(loaded, obs, z, device)):
                worst = max(worst, float((a - b).abs().max()))
        tmp.unlink(missing_ok=True)
        checks["8_save_load_roundtrip_identical_logits_both_z"] = bool(worst <= 1e-6)
        n_pass = sum(checks.values())
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        print(f"  initial loss {float(loss.detach()):.4f} (kl_A {diag['kl_A']:.4f}, kl_B {diag['kl_B']:.4f})  roundtrip max|dlogit|={worst:.2e}")
        PREFLIGHT_OUT.write_text(json.dumps({
            "record": f"Rung {RUNG} mechanical preflight", "utc": _now(), "device": device, "rung": RUNG,
            "implements": f"{AMEND.name}#PREFLIGHT",
            "checks": checks, "passed": f"{n_pass}/{len(checks)}", "sharing_arithmetic": share,
            "initial_loss": float(loss.detach()), "initial_diag": diag, "roundtrip_max_abs_logit_diff": worst,
            "VERDICT": "PASS" if n_pass == len(checks) else "FAIL",
        }, indent=2), encoding="utf-8")
        print(f"  -> {PREFLIGHT_OUT}  {n_pass}/{len(checks)}")
        return 0 if n_pass == len(checks) else 1

    # ---------------------------------------------------------------- training
    if json.loads(PREFLIGHT_OUT.read_text(encoding="utf-8")).get("VERDICT") != "PASS":
        raise SystemExit("REFUSING: preflight did not pass 8/8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    batches = Batches(arr, train_idx, batch=BATCH, seed=SEEDS[0])
    print(f"  epochs={EPOCHS} batch={BATCH} lr={LR} clip={CLIP}  updates/epoch={batches.n_per_epoch()}\n", flush=True)
    rows = []
    for ep in range(EPOCHS):
        tr_a = tr_b = 0.0; n_b = 0
        for idx in batches.epoch():
            obs, dm = to_torch(arr, idx, device)
            loss, diag = TD.distillation_loss(model, teachers, obs, dm)
            if not torch.isfinite(loss):
                raise SystemExit(f"REFUSING: non-finite loss at epoch {ep}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in actor], CLIP)
            opt.step()
            tr_a += diag["kl_A"]; tr_b += diag["kl_B"]; n_b += 1
        h = fidelity(hold_idx)
        row = {"epoch": ep + 1, "train_kl_A": tr_a / max(1, n_b), "train_kl_B": tr_b / max(1, n_b),
               **{f"holdout_{k}": v for k, v in h.items()}}
        rows.append(row)
        print(f"  epoch {ep + 1:2d}  train KL {row['train_kl_A']:.4f}/{row['train_kl_B']:.4f}  "
              f"holdout KL {h['kl_A']:.4f}/{h['kl_B']:.4f}  agree z0~A {h['agree_z0_vs_piA']:.3f}  "
              f"z1~B {h['agree_z1_vs_piB']:.3f}  z0/z1 JSD {h['student_z0_z1_jsd']:.4f}  teacher JSD {h['teacher_A_B_jsd']:.4f}",
              flush=True)
    with METRICS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    final = rows[-1]
    fit_ok = (final["holdout_agree_z0_vs_piA"] >= FIT_MIN_AGREEMENT
              and final["holdout_agree_z1_vs_piB"] >= FIT_MIN_AGREEMENT)
    if not model.modules_are_shared():
        raise SystemExit("REFUSING: module sharing was lost during training")
    provenance = {"spec": AMEND.name, "rung": RUNG, "seeds": list(SEEDS), "epochs": EPOCHS,
                  "batch": BATCH, "lr": LR, "clip": CLIP, "teachers": tdata, "final_holdout": final,
                  "sharing_arithmetic": share, "utc": _now()}
    L1.save_rung(RUNG, model, branch_cfg, kwargs, str(CKPT), provenance)
    loaded, _, _ = L1.load_rung(RUNG, str(CKPT), obs_space, act_space, device=device)
    obs, _ = to_torch(arr, hold_idx[:64], device)
    worst = 0.0
    for z in (0, 1):
        for a, b in zip(L1.logits_for_z(model, obs, z, device), L1.logits_for_z(loaded, obs, z, device)):
            worst = max(worst, float((a - b).abs().max()))
    if worst > 1e-6:
        raise SystemExit(f"REFUSING: saved Rung {RUNG} does not reproduce the trained model (max|dlogit|={worst})")

    status = "FROZEN_STUDENT" if fit_ok else "FIT_FAILED"
    FROZEN_OUT.write_text(json.dumps({
        "record_id": f"RUNG{RUNG}_STUDENT_FROZEN", "status": status, "utc": _now(), "rung": RUNG,
        "implements": AMEND.name,
        "architecture": f"shared {list(share['shared_modules'])}; everything else private per z",
        "TERMINAL_CHECKPOINT": {"path": str(CKPT.relative_to(ROOT)), "sha256": _sha(CKPT), "bytes": CKPT.stat().st_size,
                                "format": f"sharing_ladder_rung{RUNG}_v1"},
        "sharing_arithmetic": share, "modules_shared_after_training": True,
        "training": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "clip": CLIP, "seeds": list(SEEDS),
                     "updates_per_epoch": batches.n_per_epoch(), "optimized": "shared modules + both private stacks; critics excluded"},
        "final_holdout": final, "fit_check": {"min_agreement": FIT_MIN_AGREEMENT, "passed": bool(fit_ok)},
        "roundtrip_max_abs_logit_diff": worst,
        "EVAL": {"rule": "LADDER_MATCHED_EVALUATION_AMENDMENT.json", "seeds": "the 128 matched seeds in RUNG0_LADDER_REFERENCE.json", "touched": False},
        "NEXT": "matched 128-seed eval on cuda" if fit_ok else "do NOT evaluate: fit check failed",
    }, indent=2), encoding="utf-8")
    print(f"\n  fit check: {'PASS' if fit_ok else 'FAIL'}  (agree z0~A {final['holdout_agree_z0_vs_piA']:.3f}, "
          f"z1~B {final['holdout_agree_z1_vs_piB']:.3f}, floor {FIT_MIN_AGREEMENT})")
    print(f"  -> {CKPT}\n  -> {FROZEN_OUT}  [{status}]")
    return 0 if fit_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
