"""Resolved per-update context (constant across epoch/minibatch loops)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rl.custom_ppo.update.pair_utils import latent_pair_count, validate_v6_protocol_latent_k
from rl.custom_ppo.update.phase_policy import PhaseTrainingPolicy, resolve_training_phase


@dataclass(frozen=True)
class LearningRateState:
    primary: float
    actor: float | None = None
    critic: float | None = None
    router: float | None = None


@dataclass(frozen=True)
class PPOUpdateContext:
    global_step: int
    total_timesteps: int
    progress_remaining: float
    phase: str
    phase_policy: PhaseTrainingPolicy

    actor_step_enabled: bool
    critic_step_enabled: bool
    router_step_enabled: bool

    learning_rates: LearningRateState
    entropy_coefficient: float
    latent_entropy_coefficient: float
    separation_coefficient: float
    rollout_usage_coefficient: float
    adapter_scale: float

    use_latent_strategy: bool
    fixed_latent_strategy: bool
    has_dedicated_router_optimizer: bool
    macro_router_active: bool

    target_action_kl: float | None
    target_strategy_kl: float | None
    latent_k: int
    pair_count: int

    has_resample_rows: bool
    apply_main_loop_qphi_loss: bool
    action_kl_stop_enabled: bool
    strategy_kl_stop_enabled: bool


class PPOUpdateContextBuilder:
    """Build :class:`PPOUpdateContext` once at the start of ``update()``."""

    def __init__(self, *, cfg: Any, hparams: Any, runtime: Any) -> None:
        self.cfg = cfg
        self.hparams = hparams
        self.runtime = runtime

    def build(
        self,
        *,
        total_timesteps: int,
        primary_lr: float,
        latent_lam_h: float,
        curr_sep_coef: float,
        curr_adapter_scale: float,
        v6i1_usage_coef: float = 0.0,
        v6i1_lr_stats: dict[str, float] | None = None,
        buffer: Any | None = None,
    ) -> PPOUpdateContext:
        runtime = self.runtime
        hparams = self.hparams
        cfg = self.cfg
        global_step = int(runtime.global_step)
        progress_remaining = max(
            0.0, 1.0 - float(global_step) / max(1.0, float(total_timesteps))
        )
        latent_k = int(hparams.latent_k)
        validate_v6_protocol_latent_k(cfg, latent_k)
        pair_count = latent_pair_count(latent_k)

        phase = resolve_training_phase(runtime, global_step=global_step)
        phase_policy = PhaseTrainingPolicy.from_phase(phase)

        has_dedicated_router_optimizer = (
            getattr(runtime, "latent_router_optimizer", None) is not None
        )
        from rl.custom_ppo.v6i1_phase_runtime import v6i1_macro_router_active

        macro_router_active = bool(v6i1_macro_router_active(runtime))
        apply_main_loop_qphi_loss = (
            float(hparams.latent_strategy_ppo_coef or 0.0) > 0.0
            and not has_dedicated_router_optimizer
            and not macro_router_active
        )

        has_resample_rows = False
        if (
            buffer is not None
            and hparams.use_latent_strategy
            and int(buffer.pos) > 0
            and "z_resampled" in buffer.fields
        ):
            rs = buffer.fields["z_resampled"][: int(buffer.pos)].reshape(-1).bool()
            has_resample_rows = bool(rs.any())

        target_kl = getattr(cfg, "target_kl", None)
        strategy_target_kl = getattr(cfg, "strategy_target_kl", None)
        if strategy_target_kl is None:
            strategy_target_kl = target_kl

        ent_coef = (
            hparams.ent_coef if progress_remaining > 0.75 else 0.5 * hparams.ent_coef
        )

        lr_stats = v6i1_lr_stats or {}
        learning_rates = LearningRateState(
            primary=float(primary_lr),
            actor=float(lr_stats["actor_lr"]) if "actor_lr" in lr_stats else None,
            critic=float(lr_stats["critic_lr"]) if "critic_lr" in lr_stats else None,
            router=float(lr_stats["router_lr"]) if "router_lr" in lr_stats else None,
        )

        return PPOUpdateContext(
            global_step=global_step,
            total_timesteps=int(total_timesteps),
            progress_remaining=float(progress_remaining),
            phase=phase,
            phase_policy=phase_policy,
            actor_step_enabled=phase_policy.actor_step_enabled,
            critic_step_enabled=phase_policy.critic_step_enabled,
            router_step_enabled=phase_policy.router_step_enabled,
            learning_rates=learning_rates,
            entropy_coefficient=float(ent_coef),
            latent_entropy_coefficient=float(latent_lam_h),
            separation_coefficient=float(curr_sep_coef),
            rollout_usage_coefficient=float(v6i1_usage_coef),
            adapter_scale=float(curr_adapter_scale),
            use_latent_strategy=bool(hparams.use_latent_strategy),
            fixed_latent_strategy=bool(hparams.fixed_latent_strategy),
            has_dedicated_router_optimizer=has_dedicated_router_optimizer,
            macro_router_active=macro_router_active,
            target_action_kl=(
                float(target_kl) if target_kl is not None else None
            ),
            target_strategy_kl=(
                float(strategy_target_kl) if strategy_target_kl is not None else None
            ),
            latent_k=latent_k,
            pair_count=pair_count,
            has_resample_rows=has_resample_rows,
            apply_main_loop_qphi_loss=apply_main_loop_qphi_loss,
            action_kl_stop_enabled=phase_policy.actor_step_enabled,
            strategy_kl_stop_enabled=(
                apply_main_loop_qphi_loss and has_resample_rows
            ),
        )
