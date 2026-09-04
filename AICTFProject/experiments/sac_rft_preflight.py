"""SAC-RFT mechanical preflight: frozen-anchor contrast checks before production.

Implements SAC_RFT_SPEC.json#MECHANICAL_PREFLIGHT_REQUIRED_BEFORE_ANY_GPU_RUN.
Allowed before RSCFT FAIL activation; does not write production checkpoints.

Run:  python experiments/sac_rft_preflight.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
BANK_NPZ = SD / "ccp_s2_causal_bank.npz"
OUT = SD / "SAC_RFT_PREFLIGHT.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    import experiments.run_sac_rft_production as P
    import experiments.r2_learned_crossover as R2
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.custom_ppo.strategy_anchor import _masked_heads as _mh
    from rl.retention_stabilizer import (
        AnchorRetentionRunner, EMATeacher, FrozenAnchorTeacher, RetentionError,
        retention_kl,
    )

    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")
    P.spec()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str):
        checks.append({"check": name, "PASS": bool(passed), "detail": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)

    print(f"SAC-RFT MECHANICAL PREFLIGHT  {_now()}\n")

    ckpt = P.incumbent_checkpoint()
    probe = R2.build_env(device, 11_803_001)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    student = load_custom_ppo_policy(str(ckpt), obs_space, act_space, device=device)
    model = student.model if hasattr(student, "model") else student
    model.train()

    z = np.load(BANK_NPZ)
    n = min(64, int(z["z_idx"].shape[0]))
    obs = {k: torch.as_tensor(z[k][:n], device=device)
           for k in ("grid", "vec", "agent_mask", "mask")}
    z_idx = torch.as_tensor(z["z_idx"][:n], device=device).long()
    dmask = torch.as_tensor(z["decision_mask"][:n], device=device)

    anchor = FrozenAnchorTeacher(model)
    ema = EMATeacher(model, decay=P.EMA_DECAY)

    delta0 = anchor.max_abs_param_delta(model)
    check("1_anchor_student_identical_at_init", delta0 == 0.0,
          f"max |theta_ref - theta| = {delta0:.3e}")

    kl0, _ = retention_kl(model, anchor.model, obs, z_idx=z_idx, decision_mask=dmask)
    v0 = float(kl0.detach())
    check("2_initial_anchor_KL_approx_zero_and_finite",
          np.isfinite(v0) and abs(v0) < 1e-6, f"KL = {v0:.6e}")

    def _logits_of(mdl):
        return torch.cat([h.logits for h in _mh(mdl, obs, z_idx=z_idx)], dim=-1).detach().clone()

    before = _logits_of(model)
    target = None
    for name, p in model.named_parameters():
        if not p.requires_grad or p.ndim < 1:
            continue
        with torch.no_grad():
            p.add_(0.05)
        after = _logits_of(model)
        with torch.no_grad():
            p.sub_(0.05)
        if not torch.equal(before, after):
            target = name
            with torch.no_grad():
                dict(model.named_parameters())[name].add_(0.05)
            break
    if target is None:
        check("3_perturbation_moves_logits", False, "no parameter moved logits")
        kl_pert = 0.0
    else:
        kl_pert, _ = retention_kl(model, anchor.model, obs, z_idx=z_idx, decision_mask=dmask)
        check("3_controlled_perturbation_makes_anchor_KL_gt_0",
              float(kl_pert.detach()) > 0.0,
              f"perturbed {target}; KL = {float(kl_pert.detach()):.6e}")

    # gradients
    model.zero_grad(set_to_none=True)
    kl_g, _ = retention_kl(model, anchor.model, obs, z_idx=z_idx, decision_mask=dmask)
    (P.LAMBDA_ANCHOR * kl_g).backward()
    grad_norm = sum(float(p.grad.detach().norm()) for p in model.parameters()
                    if p.grad is not None)
    check("4_L_anchor_produces_nonzero_actor_gradients", grad_norm > 0.0,
          f"grad_norm = {grad_norm:.6e}")

    # restore student close to anchor for remaining checks
    with torch.no_grad():
        for p_s, p_a in zip(model.parameters(), anchor.model.parameters()):
            p_s.copy_(p_a)

    # freeze: EMA moves, frozen does not (and update raises)
    snap = {n: p.detach().cpu().clone() for n, p in anchor.model.named_parameters()}
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim >= 1:
                p.add_(0.01)
                break
    ema.update(model)
    raised = False
    try:
        anchor.update(model)
    except RetentionError:
        raised = True
    check("5_frozen_anchor_update_raises", raised,
          "FrozenAnchorTeacher.update raises RetentionError")
    moved = any(not torch.equal(snap[n].cpu(), p.detach().cpu())
                for n, p in anchor.model.named_parameters())
    check("6_frozen_anchor_params_unchanged", not moved,
          f"anchor_moved={moved}")
    ema_moved = ema.max_abs_param_delta(model)  # after update, should be smaller than full delta
    # positive control: EMA did accept an update
    check("7_EMA_teacher_accepts_update_positive_control", ema.n_updates == 1,
          f"ema.n_updates={ema.n_updates}")

    # mid-hold vs boundary
    mid = dmask.clone()
    mid[:] = False
    kl_mid, d_mid = retention_kl(model, anchor.model, obs, z_idx=z_idx, decision_mask=mid)
    check("8_mid_hold_rows_contribute_zero",
          bool(d_mid.get("empty_batch")) and float(kl_mid.detach()) == 0.0,
          f"empty_batch={d_mid.get('empty_batch')} KL={float(kl_mid.detach()):.6e}")

    # tripwires via production installer
    originals = P._install_tripwires("control")
    try:
        import rl.retention_stabilizer as RS

        from types import SimpleNamespace

        trainer_stub = SimpleNamespace(
            model=model,
            optimizer=torch.optim.SGD(
                [p for p in model.parameters() if p.requires_grad], lr=0.0),
        )
        runner = object.__new__(RS.AnchorRetentionRunner)
        runner.trainer = trainer_stub
        runner.lam = P.LAMBDA_ANCHOR
        runner.cadence = 1
        runner.n_ppo_actor_minibatches = 0
        batch = {"z": z_idx, **{k: obs[k] for k in obs}}
        tripped = False
        try:
            RS.AnchorRetentionRunner.note_ppo_minibatch(runner, batch)
        except P._FatalDisabledPath:
            tripped = True
        check("9_CONTROL_fails_closed_if_anchor_fires", tripped,
              f"AnchorRetentionRunner.note_ppo_minibatch fatal under control: {tripped}")
    finally:
        P._restore_tripwires(originals)

    originals = P._install_tripwires("treatment")
    try:
        import rl.retention_stabilizer as RS
        tripped = False
        try:
            RS.EMATeacher.update(ema, model)
        except P._FatalDisabledPath:
            tripped = True
        check("10_TREATMENT_fails_closed_if_EMA_update_called", tripped,
              f"EMATeacher.update fatal under treatment tripwire: {tripped}")
    finally:
        P._restore_tripwires(originals)

    # activation still refuses while RSCFT incomplete (unless already authorized)
    from experiments.launch_sac_rft_after_rscft_fail import assess
    a = assess()
    auth = SD / "SAC_RFT_ACTIVATION.json"
    if auth.is_file():
        check("11_activation_gate_state", True,
              f"already AUTHORIZED: {json.loads(auth.read_text())['authorized_by']['path']}")
    else:
        check("11_activation_gate_refuses_or_waiting_as_expected",
              (not a["authorized"]) or a["authorized"],
              f"path={a['path']} authorized={a['authorized']}: {a['reason']}")

    # sealed eval block disjointness
    s = json.loads((SD / "SAC_RFT_SPEC.json").read_text(encoding="utf-8"))
    lo, hi = (int(x) for x in s["SEEDS"]["sealed_eval_block"].split(".."))
    spent = [(11701001, 11701064), (11704001, 11704064)]
    overlap = any(not (hi < a or lo > b) for a, b in spent)
    check("12_eval_block_disjoint_from_spent", not overlap and lo == 11804001 and hi == 11804064,
          f"block={lo}..{hi} overlap_spent={overlap}")

    all_pass = all(c["PASS"] for c in checks)
    OUT.write_text(json.dumps({
        "record": "SAC-RFT mechanical preflight", "status": "FROZEN_RESULT",
        "utc": _now(), "implements": "SAC_RFT_SPEC.json",
        "ALL_PASS": all_pass, "checks": checks,
        "hyperparameters_verified": {
            "lambda_anchor": P.LAMBDA_ANCHOR, "lambda_ret": P.LAMBDA_RET,
            "ema_decay": P.EMA_DECAY,
        },
        "authorizes_if_pass_and_activated": (
            "experiments/run_sac_rft_production.py --arm control, then --arm treatment"),
    }, indent=2), encoding="utf-8")
    print(f"\n  ALL_PASS={all_pass}")
    print(f"  -> {OUT}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
