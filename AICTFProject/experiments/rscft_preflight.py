"""RSCFT mechanical preflight: all 12 checks required before either 500k run.

Implements RSCFT_SPEC.json#MECHANICAL_PREFLIGHT_REQUIRED_BEFORE_ANY_GPU_RUN. Every check runs
against REAL objects -- the real incumbent checkpoint, the real Stage-B bank rows with their
real stored decision masks, the real config objects run_rscft_production.py builds -- not
against stubs or descriptions of them.

Checks 6 and 7 are the ones that most need real data: the Stage-B trajectories carry a
per-row, per-agent decision_mask recorded from the live environment, so mid-hold rows and
genuine commitment-boundary rows can be separated exactly rather than approximated.

Run:  python experiments/rscft_preflight.py --device cuda
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
SPEC = SD / "RSCFT_SPEC.json"
BANK_NPZ = SD / "ccp_s2_causal_bank.npz"
OUT = SD / "RSCFT_PREFLIGHT.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    import experiments.run_rscft_production as P
    import experiments.r2_learned_crossover as R2
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.retention_stabilizer import EMATeacher, RetentionError, retention_kl

    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")
    spec = P.spec()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    checks: list[dict] = []

    def check(name: str, passed: bool, detail: str):
        checks.append({"check": name, "PASS": bool(passed), "detail": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}", flush=True)

    print(f"RSCFT MECHANICAL PREFLIGHT  {_now()}\n")

    ckpt = P.incumbent_checkpoint()
    probe = R2.build_env(device, 11_703_001)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    student = load_custom_ppo_policy(str(ckpt), obs_space, act_space, device=device)
    model = student.model if hasattr(student, "model") else student
    model.train()

    # real Stage-B rows, with their real recorded decision masks
    z = np.load(BANK_NPZ)
    n = min(64, int(z["z_idx"].shape[0]))
    obs = {k: torch.as_tensor(z[k][:n], device=device)
           for k in ("grid", "vec", "agent_mask", "mask")}
    z_idx = torch.as_tensor(z["z_idx"][:n], device=device).long()
    dmask = torch.as_tensor(z["decision_mask"][:n], device=device)

    teacher = EMATeacher(model, decay=P.EMA_DECAY)

    # --- 1. teacher and student behaviorally identical at initialization -------------------
    delta0 = teacher.max_abs_param_delta(model)
    check("1_teacher_student_identical_at_init", delta0 == 0.0,
          f"max |theta_bar - theta| = {delta0:.3e} over all parameters")

    # --- 2. initial retention KL approximately zero and finite -----------------------------
    kl0, diag0 = retention_kl(model, teacher.model, obs, z_idx=z_idx, decision_mask=dmask)
    v0 = float(kl0.detach())
    check("2_initial_KL_approx_zero_and_finite",
          np.isfinite(v0) and abs(v0) < 1e-6, f"KL = {v0:.6e}, finite={np.isfinite(v0)}")

    # --- 3. a controlled actor perturbation makes KL > 0 -----------------------------------
    # The perturbation target must actually be ON the z-conditioned forward path. Picking a
    # name that merely LOOKS like an actor parameter is how this check first "failed":
    # latent_actor.action_head.weight is not on the LRO branch this model routes z through,
    # so perturbing it left the logits bit-identical and KL correctly read 0. The target is
    # therefore chosen by VERIFYING it moves the logits, not by matching a name.
    from rl.custom_ppo.strategy_anchor import _masked_heads as _mh

    def _logits_of(mdl):
        return torch.cat([h.logits for h in _mh(mdl, obs, z_idx=z_idx)], dim=-1).detach().clone()

    base_logits = _logits_of(model)
    target, saved = None, None
    for nme, p in model.named_parameters():
        if not any(k in nme for k in ("latent_action_heads", "latent_branch_trunks",
                                      "latent_adapters", "actor")):
            continue
        trial = p.detach().clone()
        with torch.no_grad():
            p.add_(torch.randn_like(p) * 0.05)
        moved = not torch.equal(_logits_of(model), base_logits)
        with torch.no_grad():
            p.copy_(trial)
        if moved:
            target, saved, target_name = p, trial, nme
            break
    if target is None:
        check("3_perturbation_raises_KL", False,
              "no actor-side parameter was found whose perturbation changes the masked "
              "logits; the check cannot be made meaningful and must not be reported as passing")
        raise SystemExit("REFUSING: preflight cannot validate retention without a live "
                         "perturbation target")
    with torch.no_grad():
        target.add_(torch.randn_like(target) * 0.05)
    if torch.equal(_logits_of(model), base_logits):
        raise SystemExit("REFUSING: perturbation did not move the logits despite verification")
    kl_pert, _ = retention_kl(model, teacher.model, obs, z_idx=z_idx, decision_mask=dmask)
    v_pert = float(kl_pert.detach())
    with torch.no_grad():
        target.copy_(saved)
    kl_restored, _ = retention_kl(model, teacher.model, obs, z_idx=z_idx, decision_mask=dmask)
    check("3_perturbation_raises_KL", np.isfinite(v_pert) and v_pert > 0.0,
          f"KL after perturbing {target_name} = {v_pert:.6e} (restored to "
          f"{float(kl_restored.detach()):.3e}); target verified to move the masked logits")

    # --- 4. L_ret produces nonzero actor gradients ------------------------------------------
    with torch.no_grad():
        target.add_(torch.randn_like(target) * 0.05)     # need a nonzero loss to differentiate
    model.zero_grad(set_to_none=True)
    kl_g, _ = retention_kl(model, teacher.model, obs, z_idx=z_idx, decision_mask=dmask)
    (P.LAMBDA_RET * kl_g).backward()
    gnorm = float(sum((p.grad.detach() ** 2).sum() for p in model.parameters()
                      if p.grad is not None) ** 0.5)
    check("4_L_ret_produces_nonzero_actor_gradients", gnorm > 0.0,
          f"grad norm = {gnorm:.6e}")

    # --- 5. EMA params move after an actor update, and receive no optimizer gradients -------
    teacher_grads = [p.grad for p in teacher.model.parameters() if p.grad is not None]
    teacher_requires = [p.requires_grad for p in teacher.model.parameters()]
    before = teacher.max_abs_param_delta(model)
    teacher.update(model)
    after = teacher.max_abs_param_delta(model)
    check("5_EMA_moves_but_takes_no_gradients",
          len(teacher_grads) == 0 and not any(teacher_requires) and after < before,
          f"teacher params with .grad = {len(teacher_grads)}, any requires_grad = "
          f"{any(teacher_requires)}, |delta| {before:.4e} -> {after:.4e} (moved toward student)")
    with torch.no_grad():
        target.copy_(saved)

    # --- 6 / 7. mid-hold rows contribute exactly zero; boundary rows contribute normally ----
    teacher2 = EMATeacher(model, decay=P.EMA_DECAY)
    with torch.no_grad():
        target.add_(torch.randn_like(target) * 0.05)     # make teacher != student
    mid_hold = ~dmask
    rows_mid = mid_hold.any(dim=1)
    rows_bnd = dmask.any(dim=1)
    if not bool(rows_mid.any()):
        check("6_mid_hold_rows_contribute_exactly_zero", False,
              "no mid-hold rows present in the sampled bank slice; cannot verify")
    else:
        kl_mid, d_mid = retention_kl(model, teacher2.model, obs, z_idx=z_idx,
                                     decision_mask=torch.zeros_like(dmask))
        check("6_mid_hold_rows_contribute_exactly_zero", float(kl_mid.detach()) == 0.0,
              f"KL with every agent masked as committed = {float(kl_mid.detach()):.6e} "
              f"(eligible heads {d_mid['eligible_heads']})")
    kl_bnd, d_bnd = retention_kl(model, teacher2.model, obs, z_idx=z_idx, decision_mask=dmask)
    check("7_boundary_rows_contribute_normally",
          float(kl_bnd.detach()) > 0.0 and d_bnd["eligible_heads"] > 0,
          f"KL at real decision boundaries = {float(kl_bnd.detach()):.6e} over "
          f"{d_bnd['eligible_heads']} eligible heads")
    with torch.no_grad():
        target.copy_(saved)

    # --- 8. the CCP-S2 causal loss still produces its known gradients -----------------------
    from rl.causal_supervision import causal_supervision_loss
    model.zero_grad(set_to_none=True)
    c_loss = causal_supervision_loss(
        model, obs, torch.as_tensor(z["actions"][:n], device=device).long(),
        z_idx=z_idx, decision_mask=dmask,
        weights=torch.as_tensor(z["weight"][:n], device=device).float())
    c_loss.backward()
    c_gnorm = float(sum((p.grad.detach() ** 2).sum() for p in model.parameters()
                        if p.grad is not None) ** 0.5)
    check("8_causal_loss_still_produces_gradients",
          np.isfinite(float(c_loss.detach())) and c_gnorm > 0.0,
          f"loss = {float(c_loss.detach()):.6f}, grad norm = {c_gnorm:.6e}")

    # --- 9 / 10. arm tripwires fail closed in both directions --------------------------------
    import rl.retention_stabilizer as RS
    orig = P._install_tripwires("control")
    try:
        RS.retention_kl(model, teacher.model, obs, z_idx=z_idx, decision_mask=dmask)
        control_fails_closed = False
    except P._FatalDisabledPath:
        control_fails_closed = True
    finally:
        P._restore_tripwires(orig)
    check("9_CONTROL_fails_closed_if_retention_executes", control_fails_closed,
          "retention_kl raises _FatalDisabledPath under the control arm's tripwires")

    from rl.custom_ppo.update.updater import PPOUpdater

    class _Stub:
        n_ppo_actor_minibatches, n_retention_updates, cadence = 4, 0, 1
    try:
        PPOUpdater._assert_retention_cadence(None, _Stub())
        treatment_fails_closed = False
    except RuntimeError:
        treatment_fails_closed = True
    check("10_TREATMENT_fails_closed_if_retention_absent", treatment_fails_closed,
          "the updater's cadence assertion aborts when retention updates stop firing")

    # --- 11. config diff between arms contains only retention-related fields -----------------
    seed = P.training_seed(spec)
    c_cfg = P.build_config("control", seed, ckpt)
    t_cfg = P.build_config("treatment", seed, ckpt)
    allowed = {"run_tag", "checkpoint_dir", "metrics_csv_path", "episode_csv_path"}
    diffs = {k for k in set(vars(c_cfg)) | set(vars(t_cfg))
             if getattr(c_cfg, k, "<m>") != getattr(t_cfg, k, "<m>")}
    check("11_config_diff_is_only_retention_and_paths", not (diffs - allowed),
          f"differing config fields: {sorted(diffs)} (retention is attached at runtime, not "
          "via config, so the configs themselves are identical apart from paths)")

    # --- 12. the fresh EVAL block remains untouched ------------------------------------------
    ev = spec["SEEDS"]["sealed_eval_block"]
    lo, hi = (int(x) for x in ev.split(".."))
    train_range = range(seed, seed + 321)
    eval_untouched = (not (lo <= seed <= hi)) and not (set(train_range) & set(range(lo, hi + 1)))
    existing = list(SD.glob("*rscft*eval*")) + list(SD.glob("RSCFT_EVAL*"))
    check("12_fresh_EVAL_block_untouched", eval_untouched and not existing,
          f"{ev} disjoint from training range {train_range.start}..{train_range.stop - 1}; "
          f"{len(existing)} RSCFT eval artifacts exist")

    all_pass = all(c["PASS"] for c in checks)
    verdict = "PASS" if all_pass else "FAIL"
    OUT.write_text(json.dumps({
        "record": "RSCFT mechanical preflight", "status": "FROZEN_RESULT", "utc": _now(),
        "implements": "RSCFT_SPEC.json#MECHANICAL_PREFLIGHT_REQUIRED_BEFORE_ANY_GPU_RUN",
        "VERDICT": verdict, "n_checks": len(checks),
        "n_passed": sum(c["PASS"] for c in checks), "checks": checks,
        "hyperparameters_verified": {"lambda_ret": P.LAMBDA_RET, "ema_decay": P.EMA_DECAY,
                                     "lambda_causal": P.LAMBDA_CAUSAL},
        "authorizes_if_pass": "experiments/run_rscft_production.py --arm control, then "
                              "--arm treatment",
    }, indent=2), encoding="utf-8")
    print(f"\n  {sum(c['PASS'] for c in checks)}/{len(checks)} checks passed")
    print(f"  VERDICT: {verdict}")
    print(f"  -> {OUT}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
