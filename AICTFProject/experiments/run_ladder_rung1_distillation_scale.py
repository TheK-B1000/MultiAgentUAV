"""Team-size-parameterized wrapper around run_ladder_rung1_distillation.py.

Implements RUNG1_CONSTRUCTION_{n}V{n}_AMENDMENT.json. Same loss, data, schedule, fit check,
and 8-check preflight as the 2v2 Rung 1 run (rl/ladder_rung1.py) -- the model/architecture layer
is already team-size generic (it derives its shapes from the teacher checkpoint's own saved
metadata and from observation_space/action_space, not from any hardcoded agent count). Only the
ORCHESTRATION script's dataset path, teacher source, seeds, and output paths are 2v2-specific in
the original and are parameterized here, by name, exactly like the collection wrapper.

Deliberately does NOT reproduce eval_ladder_rung1_matched.py's ladder-relative D_A/D_B
comparison against a Rung-0 reference -- there is no Rung-0-at-scale, and building one would be
the sharing-ladder rediscovery work this track is explicitly skipping. The crossover evaluation
for a scaled Rung-1 student is the separate, simpler eval_rung1_crossover_scaled.py.

Run:
  python experiments/run_ladder_rung1_distillation_scale.py --team-size 6 --preflight --device cpu
  python experiments/run_ladder_rung1_distillation_scale.py --team-size 6 --device cuda
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
SUPPORTED_TEAM_SIZES = (4, 6)
EPOCHS, BATCH, LR, CLIP = 20, 256, 3e-4, 1.0
FIT_MIN_AGREEMENT = 0.50


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_dataset_scale(dataset_path: Path, allow_smoke: bool):
    """Local copy of run_teacher_distillation.load_dataset()'s body, with one difference:
    the frozen-status check also accepts 'SMOKE_NOT_SCIENTIFIC' when allow_smoke=True.

    A separate function rather than editing run_teacher_distillation.py's shared
    load_dataset(), which the real 2v2 path also calls and which must keep enforcing
    FROZEN_DATASET unconditionally -- loosening it there would let a non-frozen dataset slip
    into a real run by accident. This wrapper's own real (non-smoke) path enforces the exact
    same FROZEN_DATASET-only rule.
    """
    import numpy as _np
    import experiments.run_teacher_distillation as _RTD

    man = json.loads(dataset_path.read_text(encoding="utf-8"))
    allowed = {"FROZEN_DATASET"} | ({"SMOKE_NOT_SCIENTIFIC"} if allow_smoke else set())
    if man["status"] not in allowed:
        raise SystemExit(f"REFUSING: dataset status {man['status']!r} not in {sorted(allowed)}")
    parts = {k: [] for k in _RTD.OBS_KEYS + ("decision_mask", "pole", "episode")}
    for sh in man["shards"]:
        d = _np.load(ROOT / sh["file"])
        if int(d["step"].shape[0]) == 0:
            continue
        for k in parts:
            parts[k].append(d[k])
    arr = {k: _np.concatenate(v, axis=0) for k, v in parts.items()}
    keep = arr["decision_mask"].any(axis=1)
    if not keep.all():
        arr = {k: v[keep] for k, v in arr.items()}
    hold = (arr["episode"] % 10) == 9
    return man, arr, hold


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-size", type=int, required=True, choices=SUPPORTED_TEAM_SIZES)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="NON-SCIENTIFIC plumbing smoke: uses disposable seeds, reads the "
                         "_SMOKE-suffixed dataset (collect_distillation_states_scale.py "
                         "--smoke), trains only 2 epochs, and writes to _SMOKE-suffixed "
                         "output paths that can never collide with or block the real "
                         "one-shot run. Use this to exercise the full training loop against "
                         "a placeholder dataset before real teachers/data exist.")
    args = ap.parse_args()
    n = int(args.team_size)
    device = args.device
    epochs = EPOCHS

    AMEND = SD / f"RUNG1_CONSTRUCTION_{n}V{n}_AMENDMENT.json"
    if not AMEND.is_file():
        raise SystemExit(f"FAIL-CLOSED: {AMEND.name} not found. Preregister-before-train: "
                         f"freeze this team size's Rung-1 construction amendment (seeds, "
                         f"output paths) before any training epoch runs.")
    amend = json.loads(AMEND.read_text(encoding="utf-8"))
    if not str(amend.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: {AMEND.name} not frozen: {amend.get('status')!r}")

    if args.smoke:
        DATASET_MANIFEST = SD / f"TEACHER_DISTILLATION_{n}V{n}_DATASET_SMOKE.json"
        SEEDS = (99_900_000 + 1000 * n + 900, 99_900_000 + 1000 * n + 901)
        OUT_DIR = ROOT / f"artifacts/strategic_demand/sppo/sharing_ladder_{n}v{n}_SMOKE/rung1/ckpts"
        CKPT = OUT_DIR / f"final_rung1_{n}v{n}_smoke.pt"
        METRICS = SD / f"sharing_ladder_{n}v{n}_SMOKE" / "rung1_metrics.csv"
        PREFLIGHT_OUT = SD / f"RUNG1_{n}V{n}_PREFLIGHT_SMOKE.json"
        FROZEN_OUT = SD / f"RUNG1_{n}V{n}_STUDENT_FROZEN_SMOKE.json"
        epochs = 2
    else:
        DATASET_MANIFEST = SD / f"TEACHER_DISTILLATION_{n}V{n}_DATASET.json"
        SEEDS = tuple(amend["SEEDS"]["training"])
        OUT_DIR = ROOT / amend["OUTPUT_PATHS"]["checkpoint_dir"]
        CKPT = OUT_DIR / f"final_rung1_{n}v{n}.pt"
        METRICS = SD / f"sharing_ladder_{n}v{n}" / "rung1_metrics.csv"
        PREFLIGHT_OUT = SD / amend["OUTPUT_PATHS"]["preflight_record"]
        FROZEN_OUT = SD / amend["OUTPUT_PATHS"]["frozen_record"]

    if not DATASET_MANIFEST.is_file():
        raise SystemExit(f"FAIL-CLOSED: {DATASET_MANIFEST.name} not found -- collect the "
                         f"dataset first (collect_distillation_states_scale.py --team-size {n}"
                         f"{' --smoke' if args.smoke else ''}).")

    if not args.preflight and FROZEN_OUT.is_file():
        raise SystemExit(f"REFUSING: {FROZEN_OUT.name} exists; {'smoke rerun' if args.smoke else f'Rung 1 training at {n}v{n} is one-shot'}")
    if not args.preflight and not PREFLIGHT_OUT.is_file():
        raise SystemExit("REFUSING: run --preflight first")

    import torch
    import experiments.r2_learned_crossover as R2
    from rl import ladder_rung1 as L1
    from rl import teacher_distillation as TD
    from rl.custom_ppo import load_custom_ppo_policy
    import experiments.run_teacher_distillation as RTD  # noqa: N812 -- module, not a class

    # By-name propagation: R2.AGENTS is what R2.build_env feeds into GPUFieldConfig. Set
    # BEFORE the probe env is built so obs/act spaces reflect the live team size.
    R2.AGENTS = n

    man, arr, hold = _load_dataset_scale(DATASET_MANIFEST, allow_smoke=args.smoke)
    if int(man.get("team_size", -1)) != n:
        raise SystemExit(f"FAIL-CLOSED: dataset manifest team_size={man.get('team_size')} != {n}")
    train_idx, hold_idx = np.where(~hold)[0], np.where(hold)[0]
    tdata = man["teachers"]

    probe = R2.build_env(device, SEEDS[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    grid_dim = int(obs_space.spaces["grid"].shape[0])
    probe.close()
    if grid_dim != n:
        raise SystemExit(f"FAIL-CLOSED: probe env grid dim {grid_dim} != team size {n}")

    teachers = {}
    for name in ("pi_A", "pi_B"):
        p = ROOT / tdata[name]["path"]
        if _sha(p) != tdata[name]["sha256"]:
            raise SystemExit(f"REFUSING: {name} sha mismatch against the dataset manifest -- "
                             f"the checkpoint on disk is not the one the dataset was collected from")
        pol = load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
        pol.model.eval()
        for q in pol.model.parameters():
            q.requires_grad_(False)
        teachers[name] = pol.model

    model, branch_cfg, kwargs, pi_a_sd = L1.build_rung(1, str(ROOT / tdata["pi_A"]["path"]), obs_space,
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

    print(f"RUNG 1 AT {n}v{n}  {'PREFLIGHT' if args.preflight else 'DISTILLATION'}  {_now()}  device={device}")
    print(f"  dataset {DATASET_MANIFEST.name}  rows train={train_idx.size} holdout={hold_idx.size}")
    print(f"  params: shared {share['shared_modules']} {len(shared)} tensors ({share['n_shared']:,} values), "
          f"private z0 {len(priv0)}, private z1 {len(priv1)}, critic {len(critic)} (excluded); "
          f"unique={share['n_unique']:,} expected={share['expected_unique']:,} "
          f"-> {'OK' if share['ok'] else 'MISMATCH'}", flush=True)

    def fidelity(idx) -> dict:
        model.eval()
        acc, cnt = {}, 0
        with torch.no_grad():
            for s in range(0, idx.size, 512):
                j = idx[s:s + 512]
                obs, dm = RTD.to_torch(arr, j, device)
                d = TD.fidelity_diagnostics(model, teachers, obs, dm)
                for k, v in d.items():
                    acc[k] = acc.get(k, 0.0) + v * int(j.size)
                cnt += int(j.size)
        model.train()
        return {k: v / max(1, cnt) for k, v in acc.items()}

    # ---------------------------------------------------------------- preflight
    if args.preflight:
        checks = {}
        checks["1_dataset"] = bool(train_idx.size > 0 and hold_idx.size > 0
                                   and (arr["pole"][train_idx] == 0).any() and (arr["pole"][train_idx] == 1).any())
        checks["2_branches_match_specialist_arch_and_differ_from_piA"] = True  # build_rung raised otherwise
        checks["3_shared_modules_identity_and_arithmetic"] = bool(model.modules_are_shared() and share["ok"])
        idx = train_idx[:min(64, train_idx.size)]
        obs, dm = RTD.to_torch(arr, idx, device)
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
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        torch.nn.utils.clip_grad_norm_([p for _, p in actor], CLIP)
        opt.step(); opt.zero_grad(set_to_none=True)
        moved = lambda ps: any(not torch.equal(before[name], p.detach()) for name, p in ps)
        still = lambda ps: all(torch.equal(before[name], p.detach()) for name, p in ps)
        checks["7_step_moves_shared_and_both_private_critic_still"] = bool(
            moved(shared) and moved(priv0) and moved(priv1) and still(critic))
        tmp = OUT_DIR / "_preflight_roundtrip.pt"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        L1.save_rung(1, model, branch_cfg, kwargs, str(tmp), {"preflight": True})
        loaded, _, _ = L1.load_rung(1, str(tmp), obs_space, act_space, device=device)
        worst = 0.0
        for z in (0, 1):
            for a, b in zip(L1.logits_for_z(model, obs, z, device), L1.logits_for_z(loaded, obs, z, device)):
                worst = max(worst, float((a - b).abs().max()))
        tmp.unlink(missing_ok=True)
        checks["8_save_load_roundtrip_identical_logits_both_z"] = bool(worst <= 1e-6)
        n_pass = sum(checks.values())
        for k, v in checks.items():
            print(f"  [{'PASS' if v else 'FAIL'}] {k}")
        print(f"  initial loss {float(loss.detach()):.4f} (kl_A {diag['kl_A']:.4f}, kl_B {diag['kl_B']:.4f})  "
              f"roundtrip max|dlogit|={worst:.2e}")
        PREFLIGHT_OUT.write_text(json.dumps({
            "record": f"Rung 1 at {n}v{n} mechanical preflight", "utc": _now(), "device": device, "rung": 1,
            "team_size": n, "implements": f"{AMEND.name}#PREFLIGHT",
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
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    batches = RTD.Batches(arr, train_idx, batch=BATCH, seed=SEEDS[0])
    print(f"  epochs={epochs} batch={BATCH} lr={LR} clip={CLIP}  updates/epoch={batches.n_per_epoch()}\n", flush=True)
    rows = []
    for ep in range(epochs):
        tr_a = tr_b = 0.0; n_b = 0
        for idx in batches.epoch():
            obs, dm = RTD.to_torch(arr, idx, device)
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
              f"z1~B {h['agree_z1_vs_piB']:.3f}  z0/z1 JSD {h['student_z0_z1_jsd']:.4f}  "
              f"teacher JSD {h['teacher_A_B_jsd']:.4f}", flush=True)
    with METRICS.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    final = rows[-1]
    fit_ok = (final["holdout_agree_z0_vs_piA"] >= FIT_MIN_AGREEMENT
              and final["holdout_agree_z1_vs_piB"] >= FIT_MIN_AGREEMENT)
    if not model.modules_are_shared():
        raise SystemExit("REFUSING: module sharing was lost during training")
    provenance = {"spec": AMEND.name, "rung": 1, "team_size": n, "seeds": list(SEEDS), "epochs": epochs,
                  "batch": BATCH, "lr": LR, "clip": CLIP, "teachers": tdata, "final_holdout": final,
                  "sharing_arithmetic": share, "utc": _now()}
    L1.save_rung(1, model, branch_cfg, kwargs, str(CKPT), provenance)
    loaded, _, _ = L1.load_rung(1, str(CKPT), obs_space, act_space, device=device)
    obs, _ = RTD.to_torch(arr, hold_idx[:min(64, hold_idx.size)], device)
    worst = 0.0
    for z in (0, 1):
        for a, b in zip(L1.logits_for_z(model, obs, z, device), L1.logits_for_z(loaded, obs, z, device)):
            worst = max(worst, float((a - b).abs().max()))
    if worst > 1e-6:
        raise SystemExit(f"REFUSING: saved Rung 1 does not reproduce the trained model (max|dlogit|={worst})")

    status = "FROZEN_STUDENT" if fit_ok else "FIT_FAILED"
    FROZEN_OUT.write_text(json.dumps({
        "record_id": f"RUNG1_{n}V{n}_STUDENT_FROZEN", "status": status, "utc": _now(), "rung": 1, "team_size": n,
        "implements": AMEND.name,
        "architecture": f"shared {list(share['shared_modules'])}; everything else private per z",
        "TERMINAL_CHECKPOINT": {"path": str(CKPT.relative_to(ROOT)), "sha256": _sha(CKPT), "bytes": CKPT.stat().st_size,
                                "format": "sharing_ladder_rung1_v1"},
        "sharing_arithmetic": share, "modules_shared_after_training": True,
        "training": {"epochs": epochs, "batch": BATCH, "lr": LR, "clip": CLIP, "seeds": list(SEEDS),
                     "updates_per_epoch": batches.n_per_epoch(), "optimized": "shared modules + both private stacks; critics excluded"},
        "final_holdout": final, "fit_check": {"min_agreement": FIT_MIN_AGREEMENT, "passed": bool(fit_ok)},
        "roundtrip_max_abs_logit_diff": worst,
        "fidelity_band": "NONE -- not inherited from the 2v2 ladder, per RUNG1_CONSTRUCTION_{}V{}_AMENDMENT.json".format(n, n),
        "NEXT": "sealed crossover eval on a fresh block via eval_rung1_crossover_scaled.py" if fit_ok
               else "do NOT evaluate: fit check failed",
    }, indent=2), encoding="utf-8")
    print(f"\n  fit check: {'PASS' if fit_ok else 'FAIL'}  (agree z0~A {final['holdout_agree_z0_vs_piA']:.3f}, "
          f"z1~B {final['holdout_agree_z1_vs_piB']:.3f}, floor {FIT_MIN_AGREEMENT})")
    print(f"  -> {CKPT}\n  -> {FROZEN_OUT}  [{status}]")
    return 0 if fit_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
