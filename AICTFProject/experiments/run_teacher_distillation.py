"""Phase 1 -- pure compression: distil pi_A / pi_B into a fresh K=2 latent student.

Implements TEACHER_DISTILLATION_SPEC.json#TRAINING. A standalone supervised runner: no PPO
objective, no critic update, no task reward, no advantage, no rollout-policy optimisation.
Only rl.teacher_distillation.distillation_loss over the student's ACTOR parameters.

    --preflight   runs the 7 mechanical checks the spec requires, writes
                  TEACHER_DISTILLATION_PREFLIGHT.json, trains nothing
    (default)     fixed 20-epoch training, fit check, freeze the student (sha), write
                  TEACHER_DISTILLATION_STUDENT_FROZEN.json

Run:  python experiments/run_teacher_distillation.py --preflight --device cpu
      python experiments/run_teacher_distillation.py --device cpu
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
SPEC = SD / "TEACHER_DISTILLATION_SPEC.json"
DATASET = SD / "TEACHER_DISTILLATION_DATASET.json"
INCUMBENT_FROZEN = SD / "CCP_SUCCESSOR_MODEL_FROZEN.json"
PREFLIGHT_OUT = SD / "TEACHER_DISTILLATION_PREFLIGHT.json"
OUT_DIR = SD / "teacher_distillation"
CKPT = OUT_DIR / "ckpts" / "final_teacher_distillation_student.zip"
METRICS = OUT_DIR / "phase1_metrics.csv"
FROZEN_OUT = SD / "TEACHER_DISTILLATION_STUDENT_FROZEN.json"

SEED = 11_920_001
EPOCHS = 20
BATCH = 256
LR = 3e-4
CLIP = 1.0
FIT_MIN_AGREEMENT = 0.50
OBS_KEYS = ("grid", "vec", "agent_mask", "mask", "global_state")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_dataset():
    man = json.loads(DATASET.read_text(encoding="utf-8"))
    if man["status"] != "FROZEN_DATASET":
        raise SystemExit(f"REFUSING: dataset not frozen: {man['status']!r}")
    parts = {k: [] for k in OBS_KEYS + ("decision_mask", "pole", "episode")}
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
    # holdout: every 10th episode per pole, by episode id
    hold = (arr["episode"] % 10) == 9
    return man, arr, hold


class Batches:
    """Stratified by state-origin pole: half the rows from A-deployment states, half from B."""

    def __init__(self, arr, idx, *, batch: int, seed: int):
        self.arr = arr
        self.a = idx[arr["pole"][idx] == 0]
        self.b = idx[arr["pole"][idx] == 1]
        if self.a.size == 0 or self.b.size == 0:
            raise SystemExit("REFUSING: a pole has zero training rows; cannot stratify")
        self.batch = int(batch)
        self.rng = np.random.default_rng(int(seed))

    def epoch(self):
        half = self.batch // 2
        pa, pb = self.rng.permutation(self.a), self.rng.permutation(self.b)
        n = min(pa.size // half, pb.size // half)
        for i in range(n):
            yield np.concatenate([pa[i * half:(i + 1) * half], pb[i * half:(i + 1) * half]])

    def n_per_epoch(self) -> int:
        half = self.batch // 2
        return min(self.a.size // half, self.b.size // half)


def to_torch(arr, idx, device):
    import torch
    obs = {k: torch.as_tensor(arr[k][idx], dtype=torch.float32, device=device) for k in OBS_KEYS}
    dm = torch.as_tensor(arr["decision_mask"][idx], dtype=torch.bool, device=device)
    return obs, dm


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    device = args.device

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: spec not frozen: {spec['status']!r}")
    if not args.preflight and FROZEN_OUT.is_file():
        raise SystemExit(f"REFUSING: {FROZEN_OUT.name} exists; Phase 1 is one-shot")
    if not args.preflight and not PREFLIGHT_OUT.is_file():
        raise SystemExit("REFUSING: run --preflight first; no production run without 7/7")

    import torch
    import experiments.r2_learned_crossover as R2
    from rl import teacher_distillation as TD
    from rl.custom_ppo import load_custom_ppo_policy

    man, arr, hold = load_dataset()
    train_idx = np.where(~hold)[0]
    hold_idx = np.where(hold)[0]
    tspec = man["teachers"]

    probe = R2.build_env(device, 11_921_001)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()

    teachers = {}
    for n in ("pi_A", "pi_B"):
        ck = ROOT / tspec[n]["path"]
        if _sha(ck) != tspec[n]["sha256"]:
            raise SystemExit(f"REFUSING: {n} sha mismatch vs the frozen dataset manifest")
        pol = load_custom_ppo_policy(str(ck), obs_space, act_space, device=device)
        pol.model.eval()
        for p in pol.model.parameters():
            p.requires_grad_(False)
        teachers[n] = pol.model

    inc_rec = json.loads(INCUMBENT_FROZEN.read_text(encoding="utf-8"))
    inc_path = ROOT / inc_rec["TERMINAL_CHECKPOINT"]["path"]
    if _sha(inc_path) != inc_rec["TERMINAL_CHECKPOINT"]["sha256"]:
        raise SystemExit("REFUSING: incumbent checkpoint sha mismatch")
    student, inc_payload = TD.build_fresh_student(str(inc_path), obs_space, act_space,
                                                  seed=SEED, device=device)
    student.train()
    actor = TD.actor_parameters(student)
    critic = TD.critic_parameters(student)
    opt = torch.optim.Adam([p for _, p in actor], lr=LR)

    print(f"TEACHER DISTILLATION {'PREFLIGHT' if args.preflight else 'PHASE 1'}  {_now()}  device={device}")
    print(f"  rows: train={train_idx.size} holdout={hold_idx.size}  "
          f"actor params={len(actor)} critic params={len(critic)} (excluded)")

    # ------------------------------------------------------------------ preflight
    if args.preflight:
        checks = {}
        checks["1_dataset"] = bool(train_idx.size > 0 and hold_idx.size > 0
                                   and (arr["pole"][train_idx] == 0).any()
                                   and (arr["pole"][train_idx] == 1).any())
        ref = inc_payload["model_state_dict"]
        sd = student.state_dict()
        diff = max(float((sd[k].cpu().float() - ref[k].cpu().float()).abs().max()) for k in sd)
        checks["2_fresh_student_same_arch_different_weights"] = bool(
            list(sd.keys()) == list(ref.keys()) and diff > 0.0)
        idx = train_idx[:min(64, train_idx.size)]
        obs, dm = to_torch(arr, idx, device)
        with torch.no_grad():
            la = TD.head_logits(teachers["pi_A"], obs)
            self_kl, _ = TD.masked_mean(TD.kl_per_head(la, la), dm)
        checks["3_teacher_self_kl_is_zero"] = bool(float(self_kl) == 0.0)
        loss, diag = TD.distillation_loss(student, teachers, obs, dm)
        checks["4_initial_loss_finite_positive"] = bool(torch.isfinite(loss) and float(loss) > 0.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        actor_grad = any(p.grad is not None and float(p.grad.abs().max()) > 0 for _, p in actor)
        critic_grad = any(p.grad is not None and float(p.grad.abs().max()) > 0 for _, p in critic)
        checks["5_grad_on_actor_not_critic"] = bool(actor_grad and not critic_grad)
        before = {n: p.detach().clone() for n, p in student.named_parameters()}
        torch.nn.utils.clip_grad_norm_([p for _, p in actor], CLIP)
        opt.step()
        opt.zero_grad(set_to_none=True)
        actor_moved = any(not torch.equal(before[n], p.detach()) for n, p in actor)
        critic_still = all(torch.equal(before[n], p.detach()) for n, p in critic)
        checks["6_step_moves_actor_only"] = bool(actor_moved and critic_still)
        tmp = OUT_DIR / "ckpts" / "_preflight_roundtrip.zip"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        TD.write_student_checkpoint(inc_payload, student, str(tmp), {"preflight": True})
        worst = TD.verify_roundtrip(str(tmp), student, obs_space, act_space, obs, device=device)
        tmp.unlink(missing_ok=True)
        checks["7_checkpoint_roundtrip_identical_logits"] = bool(worst <= 1e-6)
        n_pass = sum(checks.values())
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        print(f"  initial loss {float(loss):.4f}  (kl_A {diag['kl_A']:.4f}, kl_B {diag['kl_B']:.4f})  "
              f"roundtrip max|dlogit|={worst:.2e}")
        PREFLIGHT_OUT.write_text(json.dumps({
            "record": "Teacher-distillation mechanical preflight", "utc": _now(),
            "implements": "TEACHER_DISTILLATION_SPEC.json#MECHANICAL_PREFLIGHT_REQUIRED_BEFORE_TRAINING",
            "checks": checks, "passed": f"{n_pass}/{len(checks)}",
            "initial_loss": float(loss), "initial_diag": diag, "roundtrip_max_abs_logit_diff": worst,
            "rows": {"train": int(train_idx.size), "holdout": int(hold_idx.size)},
            "VERDICT": "PASS" if n_pass == len(checks) else "FAIL",
        }, indent=2), encoding="utf-8")
        print(f"  -> {PREFLIGHT_OUT}  {n_pass}/{len(checks)}")
        return 0 if n_pass == len(checks) else 1

    # ------------------------------------------------------------------ Phase 1 training
    pre = json.loads(PREFLIGHT_OUT.read_text(encoding="utf-8"))
    if pre.get("VERDICT") != "PASS":
        raise SystemExit("REFUSING: preflight did not pass 7/7")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    batches = Batches(arr, train_idx, batch=BATCH, seed=SEED)
    print(f"  epochs={EPOCHS} batch={BATCH} lr={LR} clip={CLIP}  "
          f"updates/epoch={batches.n_per_epoch()}\n", flush=True)

    def holdout_eval() -> dict:
        student.eval()
        acc, n = {}, 0
        with torch.no_grad():
            for s in range(0, hold_idx.size, 512):
                idx = hold_idx[s:s + 512]
                obs, dm = to_torch(arr, idx, device)
                d = TD.fidelity_diagnostics(student, teachers, obs, dm)
                w = int(idx.size)
                for k, v in d.items():
                    acc[k] = acc.get(k, 0.0) + v * w
                n += w
        student.train()
        return {k: v / max(1, n) for k, v in acc.items()}

    rows = []
    for ep in range(EPOCHS):
        tr_a, tr_b, n_b = 0.0, 0.0, 0
        for idx in batches.epoch():
            obs, dm = to_torch(arr, idx, device)
            loss, diag = TD.distillation_loss(student, teachers, obs, dm)
            if not torch.isfinite(loss):
                raise SystemExit(f"REFUSING: non-finite loss at epoch {ep}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in actor], CLIP)
            opt.step()
            tr_a += diag["kl_A"]; tr_b += diag["kl_B"]; n_b += 1
        h = holdout_eval()
        row = {"epoch": ep + 1, "train_kl_A": tr_a / max(1, n_b), "train_kl_B": tr_b / max(1, n_b),
               **{f"holdout_{k}": v for k, v in h.items()}}
        rows.append(row)
        print(f"  epoch {ep + 1:2d}  train KL A/B {row['train_kl_A']:.4f}/{row['train_kl_B']:.4f}  "
              f"holdout KL {h['kl_A']:.4f}/{h['kl_B']:.4f}  agree z0~A {h['agree_z0_vs_piA']:.3f}  "
              f"z1~B {h['agree_z1_vs_piB']:.3f}  student JSD {h['student_z0_z1_jsd']:.4f}  "
              f"teacher JSD {h['teacher_A_B_jsd']:.4f}", flush=True)
    with METRICS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    final = rows[-1]
    fit_ok = (final["holdout_agree_z0_vs_piA"] >= FIT_MIN_AGREEMENT
              and final["holdout_agree_z1_vs_piB"] >= FIT_MIN_AGREEMENT)
    provenance = {"spec": "TEACHER_DISTILLATION_SPEC.json", "seed": SEED, "epochs": EPOCHS,
                  "batch": BATCH, "lr": LR, "clip": CLIP, "dataset": str(DATASET.relative_to(ROOT)),
                  "teachers": tspec, "final_holdout": {k: v for k, v in final.items()},
                  "fit_check": {"min_agreement": FIT_MIN_AGREEMENT, "passed": bool(fit_ok)},
                  "utc": _now()}
    TD.write_student_checkpoint(inc_payload, student, str(CKPT), provenance)
    idx = hold_idx[:min(64, hold_idx.size)]
    obs, _ = to_torch(arr, idx, device)
    worst = TD.verify_roundtrip(str(CKPT), student, obs_space, act_space, obs, device=device)
    if worst > 1e-6:
        raise SystemExit(f"REFUSING: written checkpoint does not reproduce the student (max|dlogit|={worst})")

    status = "FROZEN_STUDENT" if fit_ok else "FIT_FAILED"
    FROZEN_OUT.write_text(json.dumps({
        "record_id": "TEACHER_DISTILLATION_STUDENT_FROZEN", "status": status, "utc": _now(),
        "implements": "TEACHER_DISTILLATION_SPEC.json#TRAINING",
        "TERMINAL_CHECKPOINT": {"path": str(CKPT.relative_to(ROOT)), "sha256": _sha(CKPT),
                                "bytes": CKPT.stat().st_size},
        "architecture_source": {"incumbent": str(inc_path.relative_to(ROOT)),
                                "incumbent_sha256": inc_rec["TERMINAL_CHECKPOINT"]["sha256"],
                                "weights_loaded_from_incumbent": False, "fresh_init_seed": SEED},
        "training": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "clip": CLIP,
                     "updates_per_epoch": batches.n_per_epoch(),
                     "optimized": "actor parameters only", "n_actor_params": len(actor),
                     "n_critic_params_excluded": len(critic)},
        "final_holdout": final, "fit_check": provenance["fit_check"],
        "roundtrip_max_abs_logit_diff": worst,
        "EVAL_STATE_AT_FREEZE": {"block": spec["EVAL_PROTOCOL_COMPRESSION_CROSSOVER"]["block"],
                                 "touched": False},
        "NO_MODEL_SELECTION": "fixed epochs; terminal weights only; holdout used for diagnostics and the fit check, never for selection",
        "NEXT": ("open the Compression Crossover sealed EVAL" if fit_ok else
                 "do NOT open the eval: the student did not imitate its teachers (fit check failed)"),
    }, indent=2), encoding="utf-8")
    print(f"\n  fit check: {'PASS' if fit_ok else 'FAIL'}  (agree z0~A {final['holdout_agree_z0_vs_piA']:.3f}, "
          f"z1~B {final['holdout_agree_z1_vs_piB']:.3f}, floor {FIT_MIN_AGREEMENT})")
    print(f"  -> {CKPT}\n  -> {FROZEN_OUT}  [{status}]")
    return 0 if fit_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
