"""Cross-field invariant validation for resolved preset configs.

All checks operate on a fully resolved ``PPOConfig`` instance (after the preset
apply function has run).  Validation is intentionally non-fatal by default:
callers receive a list of ``PresetValidationError`` instances and decide whether
to abort or log-and-continue.

Design rules
------------
* Never import from ``rl.presets.__init__`` — use ``rl.presets.models`` only.
* All checks must be stateless and side-effect-free.
* Every check documents the invariant it enforces.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Sequence

from rl.presets.models import PresetValidationError

if TYPE_CHECKING:
    from rl.train_ppo import PPOConfig

# ---------------------------------------------------------------------------
# Individual invariant checks
# ---------------------------------------------------------------------------
# Each check function has the signature:
#   (cfg: PPOConfig, preset_name: str) -> list[PresetValidationError]
# It returns an empty list if the invariant holds.

_Check = Callable[["PPOConfig", str], list[PresetValidationError]]

_CHECKS: list[_Check] = []


def _register_check(fn: _Check) -> _Check:
    _CHECKS.append(fn)
    return fn


@_register_check
def _check_latent_k_positive(cfg: "PPOConfig", name: str) -> list[PresetValidationError]:
    """latent_k must be > 0 when latent variables are enabled."""
    if not getattr(cfg, "use_latent_variable", False):
        return []
    k = getattr(cfg, "latent_k", None)
    if k is None or k <= 0:
        return [
            PresetValidationError(
                "latent_k must be > 0 when use_latent_variable=True",
                preset_name=name,
                field_path="latent_k",
                observed=k,
                constraint="latent_k > 0",
            )
        ]
    return []


@_register_check
def _check_router_requires_latent(cfg: "PPOConfig", name: str) -> list[PresetValidationError]:
    """Router training cannot be enabled when latent variables are disabled."""
    router_enabled = getattr(cfg, "router_reward_enabled", False)
    latent_enabled = getattr(cfg, "use_latent_variable", False)
    if router_enabled and not latent_enabled:
        return [
            PresetValidationError(
                "router_reward_enabled=True requires use_latent_variable=True",
                preset_name=name,
                field_path="router_reward_enabled",
                observed=router_enabled,
                constraint="use_latent_variable must be True when router_reward_enabled=True",
            )
        ]
    return []


@_register_check
def _check_op4_not_in_pool(cfg: "PPOConfig", name: str) -> list[PresetValidationError]:
    """OP4 (fixed strategy oracle) is evaluation-only and must not appear in opponent pools."""
    pool = getattr(cfg, "opponent_pool", ())
    if pool is None:
        return []
    if "op4" in pool or 4 in pool:
        return [
            PresetValidationError(
                "OP4 is evaluation-only and must not appear in opponent_pool",
                preset_name=name,
                field_path="opponent_pool",
                observed=pool,
                constraint="'op4' / 4 not in opponent_pool",
            )
        ]
    return []


@_register_check
def _check_reward_coefs_finite(cfg: "PPOConfig", name: str) -> list[PresetValidationError]:
    """All named reward coefficients must be finite (not NaN or ±inf)."""
    import math

    coef_fields = [
        "vf_coef",
        "ent_coef",
        "latent_strategy_ppo_coef",
        "latent_lam_p",
        "latent_lam_h",
        "latent_cf_separation_coef",
        "latent_kl_consecutive",
        "latent_strategy_aux_predict_phase_coef",
        "latent_strategy_aux_return_coef",
    ]
    errors: list[PresetValidationError] = []
    for field in coef_fields:
        val = getattr(cfg, field, None)
        if val is not None and not math.isfinite(val):
            errors.append(
                PresetValidationError(
                    f"Reward coefficient {field!r} is non-finite: {val!r}",
                    preset_name=name,
                    field_path=field,
                    observed=val,
                    constraint="must be finite",
                )
            )
    return errors


@_register_check
def _check_learning_rate_positive(cfg: "PPOConfig", name: str) -> list[PresetValidationError]:
    """Learning rate must be positive."""
    lr = getattr(cfg, "learning_rate", None)
    if lr is not None and lr <= 0:
        return [
            PresetValidationError(
                f"learning_rate must be positive, got {lr!r}",
                preset_name=name,
                field_path="learning_rate",
                observed=lr,
                constraint="learning_rate > 0",
            )
        ]
    return []


@_register_check
def _check_v6i9_stage_field(cfg: "PPOConfig", name: str) -> list[PresetValidationError]:
    """v6i9_training_stage must be one of the known stage identifiers (when set)."""
    stage = getattr(cfg, "v6i9_training_stage", "")
    if not stage:
        return []
    allowed = {"stage1_mapaware_generalist", "stage2_repertoire", "stage3_router"}
    if stage not in allowed:
        return [
            PresetValidationError(
                f"Unknown v6i9_training_stage {stage!r}",
                preset_name=name,
                field_path="v6i9_training_stage",
                observed=stage,
                constraint=f"one of {sorted(allowed)}",
            )
        ]
    return []


@_register_check
def _check_residual_requires_latent(cfg: "PPOConfig", name: str) -> list[PresetValidationError]:
    """enable_latent_z_residual=True requires latent variables to be enabled."""
    residual = getattr(cfg, "enable_latent_z_residual", False)
    latent = getattr(cfg, "use_latent_variable", False)
    if residual and not latent:
        return [
            PresetValidationError(
                "enable_latent_z_residual=True requires use_latent_variable=True",
                preset_name=name,
                field_path="enable_latent_z_residual",
                observed=residual,
                constraint="use_latent_variable must be True",
            )
        ]
    return []


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def validate_preset(cfg: "PPOConfig", preset_name: str = "") -> list[PresetValidationError]:
    """Run all registered invariant checks against a resolved ``PPOConfig``.

    Returns a (possibly empty) list of ``PresetValidationError`` instances.
    The caller decides whether to raise, log, or ignore them.
    """
    errors: list[PresetValidationError] = []
    for check in _CHECKS:
        errors.extend(check(cfg, preset_name))
    return errors


def assert_preset_valid(cfg: "PPOConfig", preset_name: str = "") -> None:
    """Validate and raise ``PresetValidationError`` on the first failure."""
    errors = validate_preset(cfg, preset_name)
    if errors:
        raise errors[0]


__all__ = [
    "validate_preset",
    "assert_preset_valid",
]
