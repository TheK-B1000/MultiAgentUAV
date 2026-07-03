#!/usr/bin/env python3
"""A/B micro-experiment: BPTT-PPO control vs arc-credit running-mean treatment.

Both arms use the same **recurrent (GRU) V6I9 router** inherited from
``v6i9_mapaware_router_sparse_hardpool``.  The only resolved-config deltas are
the credit channel fields.

Credit channel comparison
-------------------------
Control:
    latent_strategy_ppo_coef  = 0.10   (BPTT PPO active)
    latent_arc_credit_enabled = False
    Advantage source: router_advantages (sparse-reward GAE from router critic)

Treatment:
    latent_strategy_ppo_coef  = 0.0    (BPTT PPO disabled)
    latent_arc_credit_enabled = True
    latent_arc_credit_baseline = "running_mean"
    Advantage source: arc_return - EMA_running_mean (auto-centered)

Architecture fields (must be identical between arms):
    recurrent_selector_hidden_dim, recurrent_seq_len, recurrent_burn_in,
    router_chunks_per_batch, strategy_interval, router_ent_coef, latent_lam_p,
    router_freeze_actor, router_reward_enabled, opponent_pool, latent_k,
    learning rate, clip_range, ...

Example::

    # Treatment:
    python experiments/run_ab_router_credit.py --arm treatment \\
      --checkpoint checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip \\
      --n-updates 5 --device cuda --seed 1 \\
      --out-dir artifacts/ab_router_credit

    # Control:
    python experiments/run_ab_router_credit.py --arm control \\
      --checkpoint checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip \\
      --n-updates 5 --device cuda --seed 1 \\
      --out-dir artifacts/ab_router_credit
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.dump_router_rollout_audit import _build_audit_trainer  # noqa: E402
from rl.custom_ppo.diagnostics.arc_credit_smoke import (  # noqa: E402
    frozen_actor_z_fingerprint,
    router_fingerprint,
)
from rl.custom_ppo.diagnostics.router_rollout_dump import (  # noqa: E402
    file_sha256,
    git_commit_hash,
)
from rl.config.ppo_config import PPOConfig  # noqa: E402
from rl.presets import apply_preset  # noqa: E402
from rl.training.config_validation import normalize_and_validate_training_config  # noqa: E402

_CONTROL_PRESET = "v6i9_mapaware_router_sparse_hardpool"
_TREATMENT_PRESET = "v6i9_arc_credit_running_mean_hardpool"

# Fields that MUST differ between control and treatment (not a failure).
_ALLOWED_DIFFS = frozenset({
    "latent_arc_credit_enabled",
    "latent_arc_credit_baseline",
    "latent_arc_credit_coef",
    "latent_arc_credit_min_len",
    "latent_strategy_ppo_coef",
    "run_tag",
    # Derived config fields that cascade from the above.
    "latent_arc_credit_n_epochs",
    "latent_arc_credit_clip_eps",
    "latent_arc_credit_return_norm",
    "latent_arc_credit_coef",
})

# Architecture + training fields that must be identical (checked explicitly).
_ARCHITECTURE_FIELDS = [
    "recurrent_selector_hidden_dim",
    "recurrent_seq_len",
    "recurrent_burn_in",
    "router_chunks_per_batch",
    "strategy_interval",
    "router_ent_coef",
    "latent_lam_p",
    "latent_lam_h",
    "latent_k",
    "router_freeze_actor",
    "router_reward_enabled",
    "router_reward_win_weight",
    "router_reward_flag_cap_weight",
    "router_reward_sparse_weight",
    "router_reward_scale",
    "router_reward_normalize",
    "latent_assignment_mode",
    "v6i9_training_stage",
    "opponent_pool",
    "map_name",
    "clip_range",
    "clip_range_vf",
    "learning_rate",
    "n_envs",
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "normalize_returns",
    "seed",
]

_TELEMETRY_KEYS = [
    "latent_arc_count",
    "latent_arc_finalized_count",
    "latent_arc_mean_return",
    "latent_arc_baseline_mean",
    "latent_arc_raw_advantage_mean",
    "latent_arc_raw_advantage_std",
    "latent_arc_advantage_mean",
    "latent_arc_advantage_std",
    "latent_arc_positive_fraction",
    "latent_arc_running_mean_value",
    "latent_arc_running_mean_count",
    "latent_arc_grad_norm",
    "latent_arc_credit_coef",
    "latent_arc_raw_adv_mean_z0",
    "latent_arc_raw_adv_mean_z1",
    "latent_arc_raw_adv_mean_z2",
    "latent_arc_raw_adv_mean_z3",
    "latent_arc_count_z0",
    "latent_arc_count_z1",
    "latent_arc_count_z2",
    "latent_arc_count_z3",
    "latent_arc_raw_adv_z_spread",
    "router_selected_z_occupancy_z0",
    "router_selected_z_occupancy_z1",
    "router_selected_z_occupancy_z2",
    "router_selected_z_occupancy_z3",
    "router_selected_z_occupancy_max",
    "router_selected_z_unique_count",
    "router_selected_z_dominant",
    "router_selected_z_decision_count",
    "q_phi_grad_norm",
    "q_phi_strategy_encoder_grad_norm",
    "router_bptt_ppo_loss",
    "router_bptt_ent_loss",
    "router_bptt_persist_loss",
    "router_bptt_decision_count",
    "strategy_policy_loss",
    "strategy_grad_norm",
    "router_advantage_mean",
    "router_advantage_std",
    "router_advantage_positive_fraction",
    "strategy_entropy",
    "latent_mi_z_phase_nats",
    "latent_mi_z_outcome_nats",
    "latent_mi_z_opponent_nats",
    "latent_mi_z_flag_state_nats",
    "policy_loss",
    "value_loss",
    "entropy",
    "grad_norm",
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="A/B router credit micro-experiment")
    p.add_argument("--arm", choices=["control", "treatment"], required=True)
    p.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/2v2/"
            "final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip"
        ),
    )
    p.add_argument("--n-updates", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out-dir", default="artifacts/ab_router_credit")
    p.add_argument("--preset", default=None, help="Override preset (default: arm-specific)")
    p.add_argument(
        "--extra-allowed-diffs",
        nargs="+",
        default=[],
        metavar="FIELD",
        help=(
            "Additional config fields allowed to differ between arms without "
            "raising an architecture mismatch error. Use when testing entropy or "
            "coverage hyperparameters alongside the credit channel (e.g. "
            "router_ent_coef latent_lam_h)."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a previously completed arm directory",
    )
    return p.parse_args()


def _cfg_as_dict(cfg: PPOConfig) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def _resolve_cfg(preset: str) -> dict[str, Any]:
    """Return the fully normalized config dict for a preset (no trainer/env needed)."""
    cfg = PPOConfig()
    cfg = apply_preset(cfg, preset)
    cfg = normalize_and_validate_training_config(cfg)
    return _cfg_as_dict(cfg)


def _assert_preset_diff(
    control_preset: str,
    treatment_preset: str,
    *,
    architecture_fields: list[str],
    allowed_diffs: frozenset[str],
) -> tuple[dict, dict, list[str]]:
    """Resolve both presets and assert the A/B diff is clean.

    Returns (ctrl_dict, treat_dict, unexpected_diffs).
    Raises RuntimeError if any architecture field differs.
    """
    print("[ab-preset-diff] Resolving control preset…")
    ctrl = _resolve_cfg(control_preset)
    print("[ab-preset-diff] Resolving treatment preset…")
    treat = _resolve_cfg(treatment_preset)

    all_keys = sorted(set(ctrl) | set(treat))
    differing = [k for k in all_keys if ctrl.get(k) != treat.get(k)]
    unexpected = [k for k in differing if k not in allowed_diffs]
    expected_missing = [k for k in allowed_diffs if k not in differing]

    print("\n[ab-preset-diff] Confirmed delta (allowed):")
    for k in sorted(allowed_diffs):
        if k in differing:
            print(f"    {k}: {ctrl.get(k)!r}  ->  {treat.get(k)!r}")
        elif k in all_keys:
            print(f"    {k}: (same={ctrl.get(k)!r})")
        else:
            print(f"    {k}: (not in either config)")

    print("\n[ab-preset-diff] Architecture fields (must be identical):")
    arch_bad = []
    for f in architecture_fields:
        cv = ctrl.get(f, "<missing>")
        tv = treat.get(f, "<missing>")
        ok = cv == tv
        tag = "OK" if ok else "MISMATCH"
        print(f"    [{tag}] {f}: {cv!r}" + (f"  vs  {tv!r}" if not ok else ""))
        if not ok:
            arch_bad.append(f)

    if arch_bad:
        raise RuntimeError(
            f"Architecture mismatch between A/B arms — fields: {arch_bad}\n"
            "Fix the presets so these are identical before running."
        )

    if unexpected:
        raise RuntimeError(
            f"Unexpected config differences between arms: {unexpected}\n"
            "Only credit-channel fields should differ. "
            "Add to _ALLOWED_DIFFS if intentional."
        )

    return ctrl, treat, differing


def _derive_advantage_source(cfg: PPOConfig) -> str:
    """Return a human-readable tag for the advantage source used by the BPTT router."""
    arc_enabled = bool(getattr(cfg, "latent_arc_credit_enabled", False))
    strategy_ppo_coef = float(getattr(cfg, "latent_strategy_ppo_coef", 0.10) or 0.0)
    router_reward = bool(getattr(cfg, "router_reward_enabled", False))
    arc_baseline = str(getattr(cfg, "latent_arc_credit_baseline", "") or "")

    if arc_enabled and strategy_ppo_coef == 0.0:
        return f"arc_return_{arc_baseline}_baseline (BPTT PPO disabled)"
    if router_reward:
        return "router_advantages (sparse-reward GAE from router critic)"
    return "advantages (actor-GAE — fallback)"


def _extract_telemetry(stats: dict, keys: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in keys:
        v = stats.get(k)
        if v is not None:
            try:
                f = float(v)
                out[k] = float("nan") if not math.isfinite(f) else f
            except (TypeError, ValueError):
                out[k] = float("nan")
        else:
            out[k] = float("nan")
    return out


def _logit_diagnostics_from_buffer(buffer, latent_k: int = 4) -> dict[str, float]:
    """Compute per-update logit diagnostics from the rollout buffer.

    Uses stored ``z_logits`` (shape [T, n_envs, K] or [T*n_envs, K]) masked by
    ``router_decision_valid`` to isolate actual routing decisions.

    Returns keys:
      logit_diag_n_decisions         -- number of router decisions in buffer
      logit_diag_marginal_entropy_nats -- H(q_bar)
      logit_diag_conditional_entropy_nats -- mean H(z|context)
      logit_diag_mi_proxy_nats       -- H(q_bar) - H(z|context)
      logit_diag_top1_top2_margin    -- mean(argmax_logit - 2nd_logit)
      logit_diag_q_bar_z{i}          -- marginal softmax prob per z
      logit_diag_argmax_frac_z{i}    -- fraction of decisions argmaxed to z
      logit_diag_logit_std_z{i}      -- std of raw logit_z across decisions
    """
    try:
        import torch
    except ImportError:
        return {}

    fields = getattr(buffer, "fields", {})
    z_logits = fields.get("z_logits")
    rdv = fields.get("router_decision_valid")
    if z_logits is None:
        return {}

    pos = int(getattr(buffer, "pos", z_logits.shape[0]))
    logits = z_logits[:pos]

    # Flatten to [N, K] regardless of buffer shape.
    if logits.dim() == 3:
        T, E, K = logits.shape
        logits_flat = logits.reshape(T * E, K)
        if rdv is not None:
            mask = rdv[:pos].reshape(T * E).bool()
        else:
            mask = torch.ones(T * E, dtype=torch.bool, device=logits_flat.device)
    else:
        logits_flat = logits
        if rdv is not None:
            mask = rdv[:pos].bool()
        else:
            mask = torch.ones(len(logits_flat), dtype=torch.bool, device=logits_flat.device)

    decision_logits = logits_flat[mask]
    n = int(decision_logits.shape[0])
    if n == 0:
        return {"logit_diag_n_decisions": 0.0}

    with torch.no_grad():
        probs = torch.softmax(decision_logits, dim=-1)
        q_bar = probs.mean(dim=0)
        log_p = torch.log(probs.clamp_min(1e-8))
        cond_entropy = -(probs * log_p).sum(dim=-1).mean()
        q_bar_log = torch.log(q_bar.clamp_min(1e-8))
        marginal_entropy = -(q_bar * q_bar_log).sum()
        mi_proxy = marginal_entropy - cond_entropy
        sorted_logits, _ = torch.sort(decision_logits, dim=-1, descending=True)
        margin = (sorted_logits[:, 0] - sorted_logits[:, 1]).mean()
        argmax_z = decision_logits.argmax(dim=-1)
        logit_std = decision_logits.std(dim=0)

    result: dict[str, float] = {
        "logit_diag_n_decisions": float(n),
        "logit_diag_marginal_entropy_nats": float(marginal_entropy.item()),
        "logit_diag_conditional_entropy_nats": float(cond_entropy.item()),
        "logit_diag_mi_proxy_nats": float(mi_proxy.item()),
        "logit_diag_top1_top2_margin": float(margin.item()),
    }
    for zi in range(latent_k):
        result[f"logit_diag_q_bar_z{zi}"] = float(q_bar[zi].item())
        result[f"logit_diag_argmax_frac_z{zi}"] = float((argmax_z == zi).float().mean().item())
        result[f"logit_diag_logit_std_z{zi}"] = float(logit_std[zi].item())
    return result


def _detect_actual_advantage_source(buffer) -> str:
    """Inspect the rollout buffer to report which advantage field will be used."""
    fields = getattr(buffer, "fields", {})
    if "router_advantages" in fields:
        return "router_advantages"
    if "option_advantages" in fields:
        return "option_advantages"
    return "advantages"


def main() -> None:
    args = _parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    control_preset = _CONTROL_PRESET
    treatment_preset = (
        args.preset if (args.preset and args.arm == "treatment") else _TREATMENT_PRESET
    )
    arm_preset = treatment_preset if args.arm == "treatment" else control_preset

    out_dir = Path(args.out_dir) / args.arm
    done_marker = out_dir / "summary.json"
    if done_marker.exists() and not args.force:
        print(
            f"[ab-router-credit] ERROR: {done_marker} already exists. "
            "Pass --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"[ab-router-credit] arm          = {args.arm}")
    print(f"[ab-router-credit] preset       = {arm_preset}")
    print(f"[ab-router-credit] n_updates    = {args.n_updates}")
    print(f"[ab-router-credit] checkpoint   = {checkpoint}")
    print(f"[ab-router-credit] output dir   = {out_dir}")
    print("=" * 72)

    # --- Preset diff assertion ---
    # Extra diffs allow entropy/coverage fields to vary when testing hyperparameter
    # changes alongside the credit channel (e.g. specialize preset).
    effective_allowed_diffs = _ALLOWED_DIFFS | frozenset(args.extra_allowed_diffs)
    effective_arch_fields = [f for f in _ARCHITECTURE_FIELDS if f not in effective_allowed_diffs]
    ctrl_dict, treat_dict, differing = _assert_preset_diff(
        control_preset,
        treatment_preset,
        architecture_fields=effective_arch_fields,
        allowed_diffs=effective_allowed_diffs,
    )

    # Save resolved configs for both arms.
    (out_dir / "resolved_config_control.json").write_text(
        json.dumps(ctrl_dict, indent=2, default=str), encoding="utf-8"
    )
    (out_dir / "resolved_config_treatment.json").write_text(
        json.dumps(treat_dict, indent=2, default=str), encoding="utf-8"
    )
    # Also save this arm's config separately.
    arm_dict = treat_dict if args.arm == "treatment" else ctrl_dict
    (out_dir / "resolved_config.json").write_text(
        json.dumps(arm_dict, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n[ab-router-credit] Building trainer for {args.arm} arm…")
    cfg, resolved, env, trainer = _build_audit_trainer(
        preset=arm_preset,
        checkpoint=str(checkpoint),
        device=args.device,
        seed=args.seed,
    )

    # --- Architecture banner ---
    recurrent_dim = int(getattr(cfg, "recurrent_selector_hidden_dim", 0) or 0)
    router_type = f"recurrent-GRU (hidden_dim={recurrent_dim})" if recurrent_dim > 0 else "feedforward-MLP"
    adv_source = _derive_advantage_source(cfg)

    print("\n" + "=" * 72)
    print(f"  Router type              : {router_type}")
    print(f"  strategy_interval        : {getattr(cfg, 'strategy_interval', '?')}")
    print(f"  recurrent_seq_len        : {getattr(cfg, 'recurrent_seq_len', '?')}")
    print(f"  recurrent_burn_in        : {getattr(cfg, 'recurrent_burn_in', '?')}")
    print(f"  router_chunks_per_batch  : {getattr(cfg, 'router_chunks_per_batch', '?')}")
    print(f"  router_ent_coef          : {getattr(cfg, 'router_ent_coef', '?')}")
    print(f"  latent_lam_p             : {getattr(cfg, 'latent_lam_p', '?')}")
    print(f"  latent_lam_h             : {getattr(cfg, 'latent_lam_h', '?')}")
    print(f"  latent_strategy_ppo_coef : {getattr(cfg, 'latent_strategy_ppo_coef', '?')}")
    print(f"  latent_arc_credit_enabled: {getattr(cfg, 'latent_arc_credit_enabled', '?')}")
    print(f"  latent_arc_credit_baseline: {getattr(cfg, 'latent_arc_credit_baseline', '?')}")
    print(f"  router_reward_enabled    : {getattr(cfg, 'router_reward_enabled', '?')}")
    print(f"  router_freeze_actor      : {getattr(cfg, 'router_freeze_actor', '?')}")
    print(f"  Advantage source         : {adv_source}")
    print("=" * 72 + "\n")

    # Presets must not be feedforward if arc-credit treatment (recurrent is fine).
    if args.arm == "treatment":
        if not bool(getattr(cfg, "latent_arc_credit_enabled", False)):
            raise RuntimeError("Treatment preset must set latent_arc_credit_enabled=True")
        if float(getattr(cfg, "latent_strategy_ppo_coef", 1.0) or 0.0) != 0.0:
            raise RuntimeError("Treatment must zero latent_strategy_ppo_coef")
        if str(getattr(cfg, "latent_arc_credit_baseline", "")) != "running_mean":
            raise RuntimeError("Treatment baseline must be running_mean")

    # --- Checkpoint metadata ---
    ckpt_hash = file_sha256(checkpoint)
    src_commit = git_commit_hash(PROJECT_ROOT)

    # --- Initial parameter snapshots (before any update) ---
    # Both arms load the same checkpoint with the same seed, so these hashes
    # must be identical across arms. Save them now so a post-hoc diff is possible.
    frozen_hash_before = frozen_actor_z_fingerprint(trainer.model)
    router_hash_before = router_fingerprint(trainer.model)

    # --- Optimizer freshness check ---
    # The router uses latent_router_optimizer when present, else falls back to
    # trainer.optimizer. The repertoire checkpoint (Stage 2) never trained the
    # router, so router params should have zero accumulated optimizer state.
    # Both arms load the same checkpoint → identical optimizer state.
    router_opt = getattr(trainer, "latent_router_optimizer", None)
    main_opt = getattr(trainer, "optimizer", None)
    _check_opt = router_opt if router_opt is not None else main_opt
    opt_state_keys = len(_check_opt.state) if _check_opt is not None else -1
    router_opt_name = "latent_router_optimizer" if router_opt is not None else "optimizer (fallback)"
    fresh_optimizer = opt_state_keys == 0

    print("\n[ab-router-credit] Initial tensor hashes:")
    print(f"  initial router hash      : {router_hash_before[:16]}…")
    print(f"  initial frozen actor hash: {frozen_hash_before[:16]}…")
    print(f"  fresh optimizer          : {fresh_optimizer}  ({router_opt_name}.state has {opt_state_keys} entries)")
    if not fresh_optimizer:
        print("  NOTE: optimizer carries prior momentum — same for both arms (loaded from identical checkpoint).")

    meta = {
        "arm": args.arm,
        "preset": arm_preset,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": ckpt_hash,
        "source_commit": src_commit,
        "seed": args.seed,
        "n_updates": args.n_updates,
        "router_type": router_type,
        "advantage_source_cfg": adv_source,
        "initial_router_hash": router_hash_before,
        "initial_frozen_actor_hash": frozen_hash_before,
        "fresh_optimizer": fresh_optimizer,
        "optimizer_checked": router_opt_name,
        "optimizer_state_entry_count": opt_state_keys,
    }
    (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    per_update_metrics: list[dict] = []

    try:
        for update_idx in range(args.n_updates):
            print(f"[ab-router-credit] --- update {update_idx + 1}/{args.n_updates} ---")

            buffer = trainer.collect_rollout()

            # Report actual buffer advantage field after first rollout.
            if update_idx == 0:
                actual_adv_field = _detect_actual_advantage_source(buffer)
                print(f"  Buffer advantage field   : {actual_adv_field}  (actual, from buffer)")
                meta["advantage_source_buffer"] = actual_adv_field
                (out_dir / "run_meta.json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )

            global_step = int(getattr(trainer, "global_step", 0)) + buffer.pos

            # Logit diagnostics from buffer (before update() may clear it).
            latent_k = int(getattr(cfg, "latent_k", 4) or 4)
            logit_diag = _logit_diagnostics_from_buffer(buffer, latent_k=latent_k)

            stats = trainer.update(buffer, total_timesteps=global_step)

            tel = _extract_telemetry(stats, _TELEMETRY_KEYS)
            tel.update(logit_diag)

            raw_mean = tel.get("latent_arc_raw_advantage_mean", float("nan"))
            pos_frac = tel.get("latent_arc_positive_fraction", float("nan"))
            q_phi = tel.get("q_phi_grad_norm", float("nan"))
            arc_n = tel.get("latent_arc_count", float("nan"))
            entropy = tel.get("strategy_entropy", float("nan"))

            def _fmt(v: float) -> str:
                return f"{v:.4f}" if math.isfinite(v) else "nan"

            print(
                f"  arc_raw_adv_mean={_fmt(raw_mean)}"
                f"  pos_frac={_fmt(pos_frac)}"
                f"  q_phi_grad={_fmt(q_phi)}"
                f"  arc_count={arc_n:.0f}"
                f"  z_entropy={_fmt(entropy)}"
            )

            per_z_adv = [tel.get(f"latent_arc_raw_adv_mean_z{zi}", float("nan")) for zi in range(4)]
            per_z_cnt = [tel.get(f"latent_arc_count_z{zi}", 0.0) for zi in range(4)]
            print(f"  per-z adv  : {[_fmt(v) for v in per_z_adv]}")
            print(f"  per-z count: {[int(v) for v in per_z_cnt]}")
            sel_occ = [tel.get(f"router_selected_z_occupancy_z{zi}", float("nan")) for zi in range(4)]
            print(
                f"  sel-z occ  : {[_fmt(v) for v in sel_occ]}"
                f"  dominant=z{int(tel.get('router_selected_z_dominant', -1))}"
                f"  unique={int(tel.get('router_selected_z_unique_count', 0))}"
            )

            # Logit-level diagnostics (context specialization vs marginal coverage).
            n_dec = tel.get("logit_diag_n_decisions", float("nan"))
            h_marg = tel.get("logit_diag_marginal_entropy_nats", float("nan"))
            h_cond = tel.get("logit_diag_conditional_entropy_nats", float("nan"))
            mi_p = tel.get("logit_diag_mi_proxy_nats", float("nan"))
            margin = tel.get("logit_diag_top1_top2_margin", float("nan"))
            q_bar = [tel.get(f"logit_diag_q_bar_z{zi}", float("nan")) for zi in range(latent_k)]
            argmax_frac = [tel.get(f"logit_diag_argmax_frac_z{zi}", float("nan")) for zi in range(latent_k)]
            logit_std = [tel.get(f"logit_diag_logit_std_z{zi}", float("nan")) for zi in range(latent_k)]
            print(
                f"  logit_diag : n={n_dec:.0f}"
                f"  H_marg={_fmt(h_marg)}"
                f"  H_cond={_fmt(h_cond)}"
                f"  MI={_fmt(mi_p)}"
                f"  margin={_fmt(margin)}"
            )
            print(f"  q_bar      : {[_fmt(v) for v in q_bar]}")
            print(f"  argmax_frac: {[_fmt(v) for v in argmax_frac]}")
            print(f"  logit_std  : {[_fmt(v) for v in logit_std]}")

            record = {
                "update_idx": update_idx,
                "global_step": global_step,
                "arm": args.arm,
                "telemetry": tel,
            }
            per_update_metrics.append(record)
            (out_dir / f"update_{update_idx:04d}.json").write_text(
                json.dumps(record, indent=2), encoding="utf-8"
            )

        # --- Save final checkpoint ---
        final_ckpt_path = out_dir / f"final_{args.arm}.zip"
        trainer.save(str(final_ckpt_path))
        print(f"[ab-router-credit] Saved final checkpoint -> {final_ckpt_path}")

        # --- Frozen / router parameter audit after all updates ---
        frozen_hash_after = frozen_actor_z_fingerprint(trainer.model)
        router_hash_after = router_fingerprint(trainer.model)
        frozen_report = {
            # Hashes at startup (before first update) — compare across arms to
            # confirm both loaded identical weights from the same checkpoint.
            "initial_router_hash": router_hash_before,
            "initial_frozen_actor_hash": frozen_hash_before,
            "fresh_optimizer": fresh_optimizer,
            # Hashes after all N updates — integrity audit.
            "frozen_actor_z_hash_before": frozen_hash_before,
            "frozen_actor_z_hash_after": frozen_hash_after,
            "frozen_actor_z_unchanged": frozen_hash_before == frozen_hash_after,
            "router_hash_before": router_hash_before,
            "router_hash_after": router_hash_after,
            "router_moved": router_hash_before != router_hash_after,
        }
        (out_dir / "frozen_hash_report.json").write_text(
            json.dumps(frozen_report, indent=2), encoding="utf-8"
        )
        print("[ab-router-credit] Frozen-actor check :", "PASS" if frozen_report["frozen_actor_z_unchanged"] else "FAIL")
        print("[ab-router-credit] Router moved        :", "YES" if frozen_report["router_moved"] else "NO — router did not update")

        # --- Final summary ---
        summary = {
            **meta,
            "final_checkpoint": str(final_ckpt_path),
            "final_checkpoint_sha256": file_sha256(final_ckpt_path),
            "frozen_hash_report": frozen_report,
            "updates": per_update_metrics,
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[ab-router-credit] Summary written -> {out_dir / 'summary.json'}")

    finally:
        if hasattr(env, "close"):
            env.close()


if __name__ == "__main__":
    main()
