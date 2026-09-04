"""One DAgger round of on-policy teacher distillation.

Implements ONPOLICY_TEACHER_DISTILLATION_SPEC.json#DESIGN.training_per_round and
#THE_MISSING_DIAGNOSTIC_ON_POLICY_FIDELITY.

    --preflight   7 mechanical checks on this round's data + starting student; trains nothing
    (default)     measure the PRE-training on-policy fidelity (the starting student on the
                  states it generated itself -- the "red-handed" number), then 10 fixed epochs
                  over teacher states UNION all student-state rounds so far, actor params only,
                  then freeze (sha) and record every fidelity diagnostic

No PPO, no reward, no critic, no retention, no separation loss. Identical objective to
TEACHER_DISTILLATION_SPEC.json; only the state distribution changes.

Run:  python experiments/run_onpolicy_teacher_distillation.py --round 1 --preflight --device cpu
      python experiments/run_onpolicy_teacher_distillation.py --round 1 --device cpu
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

from experiments.run_teacher_distillation import OBS_KEYS, Batches, load_dataset, to_torch

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "ONPOLICY_TEACHER_DISTILLATION_SPEC.json"
TEACHER_DATASET = SD / "TEACHER_DISTILLATION_DATASET.json"
INCUMBENT_FROZEN = SD / "CCP_SUCCESSOR_MODEL_FROZEN.json"

EPOCHS = 10
BATCH = 256
LR = 3e-4
CLIP = 1.0
FIT_MIN_AGREEMENT = 0.50
STUDENT_HOLDOUT_MOD = 8   # episode % 8 == 7 held out, per cell


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def starting_student_record(rnd: int) -> Path:
    return (SD / "TEACHER_DISTILLATION_STUDENT_FROZEN.json" if rnd == 1
            else SD / f"ONPOLICY_TD_ROUND{rnd - 1}_STUDENT_FROZEN.json")


def load_student_round(rnd: int):
    man = json.loads((SD / f"ONPOLICY_TD_ROUND{rnd}_STATES.json").read_text(encoding="utf-8"))
    if man["status"] != "FROZEN_DATASET":
        raise SystemExit(f"REFUSING: round {rnd} student states not frozen")
    parts = {k: [] for k in OBS_KEYS + ("decision_mask", "pole", "episode", "z_used")}
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
    hold = (arr["episode"] % STUDENT_HOLDOUT_MOD) == (STUDENT_HOLDOUT_MOD - 1)
    return man, arr, hold


def concat(arrs: list[dict], keys) -> dict:
    return {k: np.concatenate([a[k] for a in arrs], axis=0) for k in keys}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True, choices=(1, 2))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--preflight", action="store_true")
    args = ap.parse_args()
    rnd, device = args.round, args.device

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: spec not frozen: {spec['status']!r}")
    out_dir = SD / "teacher_distillation" / f"onpolicy_round{rnd}"
    preflight_out = SD / f"ONPOLICY_TD_ROUND{rnd}_PREFLIGHT.json"
    frozen_out = SD / f"ONPOLICY_TD_ROUND{rnd}_STUDENT_FROZEN.json"
    ckpt = out_dir / "ckpts" / f"final_onpolicy_td_round{rnd}_student.zip"
    metrics_csv = out_dir / f"round{rnd}_metrics.csv"
    if not args.preflight and frozen_out.is_file():
        raise SystemExit(f"REFUSING: {frozen_out.name} exists; each round is one-shot")
    if not args.preflight and not preflight_out.is_file():
        raise SystemExit("REFUSING: run --preflight for this round first")

    import torch
    import experiments.r2_learned_crossover as R2
    from rl import teacher_distillation as TD
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.custom_ppo.checkpoints.archive import read_checkpoint_payload

    # ---------------------------------------------------------------- data
    t_man, t_arr, t_hold = load_dataset()
    tspec = t_man["teachers"]
    keys = OBS_KEYS + ("decision_mask", "pole", "episode")
    rounds = {r: load_student_round(r) for r in range(1, rnd + 1)}
    this_man, this_arr, this_hold = rounds[rnd]

    train_parts, train_src = [{k: t_arr[k][~t_hold] for k in keys}], ["teacher"]
    for r, (_, a, h) in rounds.items():
        train_parts.append({k: a[k][~h] for k in keys}); train_src.append(f"student_r{r}")
    train = concat(train_parts, keys)
    train_idx = np.arange(train["pole"].shape[0])
    n_rows = {s: int(p["pole"].shape[0]) for s, p in zip(train_src, train_parts)}

    # ---------------------------------------------------------------- models
    probe = R2.build_env(device, 11_931_001)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    teachers = {}
    for n in ("pi_A", "pi_B"):
        p = ROOT / tspec[n]["path"]
        if _sha(p) != tspec[n]["sha256"]:
            raise SystemExit(f"REFUSING: {n} sha mismatch vs the frozen dataset manifest")
        pol = load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
        pol.model.eval()
        for q in pol.model.parameters():
            q.requires_grad_(False)
        teachers[n] = pol.model

    rec_path = starting_student_record(rnd)
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    if rec["status"] != "FROZEN_STUDENT":
        raise SystemExit(f"REFUSING: starting student {rec_path.name} is {rec['status']!r}")
    start_ck = ROOT / rec["TERMINAL_CHECKPOINT"]["path"]
    if _sha(start_ck) != rec["TERMINAL_CHECKPOINT"]["sha256"]:
        raise SystemExit("REFUSING: starting student sha mismatch")
    if this_man["generating_student"]["sha256"] != rec["TERMINAL_CHECKPOINT"]["sha256"]:
        raise SystemExit("REFUSING: this round's states were not generated by the starting student")
    student = load_custom_ppo_policy(str(start_ck), obs_space, act_space, device=device).model
    envelope = read_checkpoint_payload(str(start_ck), map_location="cpu")
    inc_ref = read_checkpoint_payload(
        str(ROOT / json.loads(INCUMBENT_FROZEN.read_text(encoding="utf-8"))["TERMINAL_CHECKPOINT"]["path"]),
        map_location="cpu")["model_state_dict"]
    sd = student.state_dict()
    if list(sd.keys()) != list(inc_ref.keys()) or any(tuple(sd[k].shape) != tuple(inc_ref[k].shape) for k in sd):
        raise SystemExit("REFUSING: student architecture does not match the incumbent's")
    student.train()
    actor = TD.actor_parameters(student)
    critic = TD.critic_parameters(student)
    for _, p in actor:
        p.requires_grad_(True)
    for _, p in critic:
        p.requires_grad_(False)
    opt = torch.optim.Adam([p for _, p in actor], lr=LR)

    def fidelity_over(arr, idx, model) -> dict:
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

    print(f"ON-POLICY TD ROUND {rnd} {'PREFLIGHT' if args.preflight else ''}  {_now()}  device={device}")
    print(f"  starting student: {start_ck.name}")
    print(f"  train rows: {n_rows}  (total {int(train_idx.size)})   "
          f"teacher holdout {int(t_hold.sum())}   round-{rnd} student holdout {int(this_hold.sum())}")

    # ---------------------------------------------------------------- preflight
    if args.preflight:
        checks = {}
        checks["1_datasets_load_both_poles_holdouts"] = bool(
            train_idx.size > 0 and (train["pole"] == 0).any() and (train["pole"] == 1).any()
            and t_hold.sum() > 0 and this_hold.sum() > 0)
        checks["2_starting_student_sha_and_arch"] = True  # reaching here means both checks above passed
        idx = np.arange(min(64, this_arr["pole"].shape[0]))
        obs, dm = to_torch(this_arr, idx, device)
        with torch.no_grad():
            la = TD.head_logits(teachers["pi_A"], obs)
            self_kl, _ = TD.masked_mean(TD.kl_per_head(la, la), dm)
            ha = TD.masked_heads(teachers["pi_A"], obs)
            agree_self, _ = TD.masked_mean(torch.stack(
                [(h.logits.argmax(-1) == g.logits.argmax(-1)).float() for h, g in zip(ha, ha)], dim=1), dm)
        checks["3_teacher_self_agreement_1_and_self_kl_0_on_student_states"] = bool(
            float(self_kl) == 0.0 and float(agree_self) == 1.0)
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
        checks["6_step_moves_actor_only"] = bool(
            any(not torch.equal(before[n], p.detach()) for n, p in actor)
            and all(torch.equal(before[n], p.detach()) for n, p in critic))
        tmp = out_dir / "ckpts" / "_preflight_roundtrip.zip"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        TD.write_student_checkpoint(envelope, student, str(tmp), {"preflight": True, "round": rnd})
        worst = TD.verify_roundtrip(str(tmp), student, obs_space, act_space, obs, device=device)
        tmp.unlink(missing_ok=True)
        checks["7_checkpoint_roundtrip_identical_logits"] = bool(worst <= 1e-6)
        n_pass = sum(checks.values())
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        print(f"  initial loss on student states {float(loss.detach()):.4f} "
              f"(kl_A {diag['kl_A']:.4f}, kl_B {diag['kl_B']:.4f})  roundtrip max|dlogit|={worst:.2e}")
        preflight_out.write_text(json.dumps({
            "record": f"On-policy TD round {rnd} mechanical preflight", "utc": _now(),
            "implements": "ONPOLICY_TEACHER_DISTILLATION_SPEC.json#MECHANICAL_PREFLIGHT_REQUIRED_BEFORE_EACH_ROUND",
            "checks": checks, "passed": f"{n_pass}/{len(checks)}",
            "initial_loss_on_student_states": float(loss.detach()), "initial_diag": diag,
            "roundtrip_max_abs_logit_diff": worst, "train_rows": n_rows,
            "VERDICT": "PASS" if n_pass == len(checks) else "FAIL",
        }, indent=2), encoding="utf-8")
        print(f"  -> {preflight_out}  {n_pass}/{len(checks)}")
        return 0 if n_pass == len(checks) else 1

    # ---------------------------------------------------------------- the red-handed number
    pre = json.loads(preflight_out.read_text(encoding="utf-8"))
    if pre.get("VERDICT") != "PASS":
        raise SystemExit("REFUSING: preflight did not pass 7/7")
    all_this = np.arange(this_arr["pole"].shape[0])
    onpolicy_pre = fidelity_over(this_arr, all_this, student)
    teacher_hold_pre = fidelity_over(t_arr, np.where(t_hold)[0], student)
    print(f"\n  PRE-TRAINING FIDELITY of the starting student")
    print(f"    on its OWN round-{rnd} states (n={all_this.size}):  agree z0~A {onpolicy_pre['agree_z0_vs_piA']:.3f}  "
          f"z1~B {onpolicy_pre['agree_z1_vs_piB']:.3f}  KL {onpolicy_pre['kl_A']:.4f}/{onpolicy_pre['kl_B']:.4f}")
    print(f"    on teacher holdout states (n={int(t_hold.sum())}): agree z0~A {teacher_hold_pre['agree_z0_vs_piA']:.3f}  "
          f"z1~B {teacher_hold_pre['agree_z1_vs_piB']:.3f}  KL {teacher_hold_pre['kl_A']:.4f}/{teacher_hold_pre['kl_B']:.4f}")
    print(f"    (diagnostic only -- never a gate)\n", flush=True)

    # ---------------------------------------------------------------- train
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    batches = Batches(train, train_idx, batch=BATCH, seed=11_930_001 + rnd)
    print(f"  epochs={EPOCHS} batch={BATCH} lr={LR} clip={CLIP}  updates/epoch={batches.n_per_epoch()}\n", flush=True)
    rows = []
    for ep in range(EPOCHS):
        tr_a = tr_b = 0.0; n_b = 0
        for idx in batches.epoch():
            obs, dm = to_torch(train, idx, device)
            loss, diag = TD.distillation_loss(student, teachers, obs, dm)
            if not torch.isfinite(loss):
                raise SystemExit(f"REFUSING: non-finite loss at epoch {ep}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in actor], CLIP)
            opt.step()
            tr_a += diag["kl_A"]; tr_b += diag["kl_B"]; n_b += 1
        th = fidelity_over(t_arr, np.where(t_hold)[0], student)
        sh = fidelity_over(this_arr, np.where(this_hold)[0], student)
        row = {"epoch": ep + 1, "train_kl_A": tr_a / max(1, n_b), "train_kl_B": tr_b / max(1, n_b),
               **{f"teacher_holdout_{k}": v for k, v in th.items()},
               **{f"student_r{rnd}_holdout_{k}": v for k, v in sh.items()}}
        rows.append(row)
        print(f"  epoch {ep + 1:2d}  train KL {row['train_kl_A']:.4f}/{row['train_kl_B']:.4f}  "
              f"teacher-holdout agree {th['agree_z0_vs_piA']:.3f}/{th['agree_z1_vs_piB']:.3f}  "
              f"student-holdout agree {sh['agree_z0_vs_piA']:.3f}/{sh['agree_z1_vs_piB']:.3f}  "
              f"student-holdout KL {sh['kl_A']:.4f}/{sh['kl_B']:.4f}  JSD {sh['student_z0_z1_jsd']:.4f}", flush=True)
    with metrics_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    final = rows[-1]
    fit_ok = (final["teacher_holdout_agree_z0_vs_piA"] >= FIT_MIN_AGREEMENT
              and final["teacher_holdout_agree_z1_vs_piB"] >= FIT_MIN_AGREEMENT)
    provenance = dict(envelope.get("teacher_distillation") or {})
    provenance[f"onpolicy_round{rnd}"] = {
        "spec": "ONPOLICY_TEACHER_DISTILLATION_SPEC.json", "starting_student_sha256": rec["TERMINAL_CHECKPOINT"]["sha256"],
        "train_rows": n_rows, "epochs": EPOCHS, "batch": BATCH, "lr": LR, "clip": CLIP,
        "pre_training_onpolicy_fidelity": onpolicy_pre, "pre_training_teacher_holdout_fidelity": teacher_hold_pre,
        "final": final, "utc": _now()}
    TD.write_student_checkpoint(envelope, student, str(ckpt), provenance)
    idx = np.where(this_hold)[0][:64]
    obs, _ = to_torch(this_arr, idx, device)
    worst = TD.verify_roundtrip(str(ckpt), student, obs_space, act_space, obs, device=device)
    if worst > 1e-6:
        raise SystemExit(f"REFUSING: written checkpoint does not reproduce the student (max|dlogit|={worst})")

    status = "FROZEN_STUDENT" if fit_ok else "FIT_FAILED"
    frozen_out.write_text(json.dumps({
        "record_id": f"ONPOLICY_TD_ROUND{rnd}_STUDENT_FROZEN", "status": status, "utc": _now(),
        "implements": "ONPOLICY_TEACHER_DISTILLATION_SPEC.json#DESIGN",
        "round": rnd, "starting_student": {"record": rec_path.name, "sha256": rec["TERMINAL_CHECKPOINT"]["sha256"]},
        "TERMINAL_CHECKPOINT": {"path": str(ckpt.relative_to(ROOT)), "sha256": _sha(ckpt), "bytes": ckpt.stat().st_size},
        "training": {"epochs": EPOCHS, "batch": BATCH, "lr": LR, "clip": CLIP, "train_rows": n_rows,
                     "updates_per_epoch": batches.n_per_epoch(), "optimized": "actor parameters only"},
        "DIAGNOSTICS_NOT_GATES": {
            "pre_training_onpolicy_fidelity_of_starting_student_on_its_own_states": onpolicy_pre,
            "pre_training_teacher_holdout_fidelity_of_starting_student": teacher_hold_pre,
            "post_training_teacher_holdout": {k: v for k, v in final.items() if k.startswith("teacher_holdout_")},
            "post_training_student_holdout_note": "states generated by the STARTING student, not the trained one -- slightly off-policy for the new student",
            "post_training_student_holdout": {k: v for k, v in final.items() if k.startswith(f"student_r{rnd}_holdout_")},
        },
        "fit_check": {"min_agreement": FIT_MIN_AGREEMENT, "on": "teacher-state holdout", "passed": bool(fit_ok)},
        "roundtrip_max_abs_logit_diff": worst,
        "EVAL_STATE_AT_FREEZE": {"block": spec["EVAL_PROTOCOL_ONPOLICY_COMPRESSION_CROSSOVER"]["block"], "touched": False},
        "NEXT": (f"round {rnd + 1} collection from this student" if rnd < 2 else
                 ("open the ON-POLICY COMPRESSION CROSSOVER sealed eval (device per the frozen contingent rule)"
                  if fit_ok else "do NOT open the eval: fit check failed")),
    }, indent=2), encoding="utf-8")
    print(f"\n  fit check: {'PASS' if fit_ok else 'FAIL'}  (teacher-holdout agree "
          f"{final['teacher_holdout_agree_z0_vs_piA']:.3f}/{final['teacher_holdout_agree_z1_vs_piB']:.3f}, floor {FIT_MIN_AGREEMENT})")
    print(f"  -> {ckpt}\n  -> {frozen_out}  [{status}]")
    return 0 if fit_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
