"""Shared helpers for the PPO update loop."""

from __future__ import annotations

from typing import Any

import torch


class StrictFaithfulDictWrapper(dict):
    """Block privileged obs keys from the forced-z actor separation path only."""

    disallowed_keys = {
        "opponent_id", "phase_id", "phase", "outcome_id", "role_bucket_id",
        "spread_bucket_id", "pressure_bucket_id", "attack_defense_ratio_bucket_id",
        "role_bucket", "spread_bucket", "pressure_bucket", "attack_defense_ratio_bucket",
        "opponent", "outcome",
    }

    def __getitem__(self, key):
        if key in self.disallowed_keys:
            raise AssertionError(
                f"Leakage detected! Disallowed key '{key}' accessed inside separation loss."
            )
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key in self.disallowed_keys:
            raise AssertionError(
                f"Leakage detected! Disallowed key '{key}' accessed inside separation loss."
            )
        return super().get(key, default)


def warmup_ramp_value(
    *,
    global_step: int,
    warmup_steps: int,
    ramp_steps: int,
    start_value: float,
    target_value: float,
) -> float:
    """Return zero before warmup, then linearly ramp start to target."""
    step = int(global_step)
    warmup = max(0, int(warmup_steps))
    ramp = max(0, int(ramp_steps))
    start = max(0.0, float(start_value))
    target = max(0.0, float(target_value))
    if step < warmup:
        return 0.0
    if ramp <= 0:
        return target
    progress = max(0.0, min(1.0, (step - warmup) / float(ramp)))
    return start + progress * (target - start)


def assert_finite_loss(total_loss: torch.Tensor, *, epoch_idx: int, mb_idx: int) -> None:
    if not torch.isfinite(total_loss).all():
        raise FloatingPointError(
            f"Non-finite PPO loss at epoch={epoch_idx}, minibatch={mb_idx}"
        )


def assert_finite_gradients(model: torch.nn.Module, *, epoch_idx: int, mb_idx: int) -> None:
    for name, param in model.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            raise FloatingPointError(
                f"Non-finite gradient in {name} at epoch={epoch_idx}, minibatch={mb_idx}"
            )


def populate_main_loop_qphi_telemetry(
    row: dict[str, float],
    *,
    cfg: Any,
    hparams: Any,
    runtime: Any,
    latent_lam_h: float,
) -> None:
    """Expose main-loop q_phi activity in the generic q_phi smoke fields."""
    active = (
        bool(getattr(hparams, "use_latent_strategy", False))
        and not bool(getattr(hparams, "fixed_latent_strategy", False))
        and getattr(runtime, "latent_router_optimizer", None) is None
        and (
            float(getattr(hparams, "latent_strategy_ppo_coef", 0.0) or 0.0) > 0.0
            or float(getattr(hparams, "latent_lam_p", 0.0) or 0.0) > 0.0
            or float(latent_lam_h or 0.0) > 0.0
            or float(getattr(hparams, "latent_kl_consecutive", 0.0) or 0.0) > 0.0
        )
    )
    grad = float(row.get("strategy_grad_norm", 0.0) or 0.0)
    row["main_loop_q_phi_train_active"] = 1.0 if active else 0.0
    row["main_loop_q_phi_grad_norm"] = grad
    if active and float(row.get("latent_q_phi_train_active", 0.0) or 0.0) <= 0.0:
        row["latent_q_phi_train_active"] = 1.0
    if grad > 0.0 and float(row.get("q_phi_grad_norm", 0.0) or 0.0) <= 0.0:
        row["q_phi_grad_norm"] = grad
        row["q_phi_strategy_encoder_grad_norm"] = grad


def tensor_stat(value: Any, *, default: float = 0.0) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    if value is None:
        return float(default)
    return float(value)
