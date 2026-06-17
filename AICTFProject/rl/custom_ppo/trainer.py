from __future__ import annotations

import os
import sys
import warnings
from dataclasses import asdict
from functools import partial
from typing import Any, Optional

import numpy as np
import torch
from rl.global_state import GLOBAL_STATE_DIM
from rl.latent_marl import TemporalStateTracker
from rl.ppo_core import TensorDictRolloutBuffer

from rl.custom_ppo.policy import SharedActorCentralizedCritic
from rl.custom_ppo.inference import (
    _torch_load_checkpoint,
    _assert_compatible_global_state_dim,
    apply_deterministic_sampling_generators,
    _load_model_state_dict_compat,
    CUSTOM_PPO_FORMAT,
    CUSTOM_PPO_LATENT_FORMAT,
    CUSTOM_PPO_ACTOR_ARCH,
    CUSTOM_PPO_VEC_SCHEMA_VERSION,
)
from rl.custom_ppo.curriculum_runtime import _hook_sample_training_opponent_before_reset
from rl.custom_ppo.episode_stats import EpisodeStats
from rl.custom_ppo.latent_behavior_contrast import BehaviorContrastMemory
from rl.custom_ppo.latent_strategy_state import LatentStrategyState
from rl.custom_ppo.ppo_updater import PPOUpdater
from rl.custom_ppo.return_normalization import ReturnNormalizer
from rl.custom_ppo.rollout_collector import RolloutCollector
from rl.custom_ppo.router_distill_hook import PeriodicRouterDistillHook
from rl.custom_ppo.trainer_audit import log_decentralized_actor_contract_once
from rl.custom_ppo.trainer_config import TrainerHyperparams, build_model_kwargs
from rl.custom_ppo.training_telemetry import TrainingTelemetry
# Re-exported for back-compat (``rl.custom_ppo._compose_training_reward_components``).
from rl.custom_ppo.reward_composition import _compose_training_reward_components  # noqa: F401


def _tqdm_for_sb3_progress() -> Any:
    """Match Stable-Baselines3 ``ProgressBarCallback``: prefer ``tqdm.rich.tqdm`` when available."""
    try:
        from tqdm import TqdmExperimentalWarning

        warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    except Exception:
        pass
    try:
        from tqdm.rich import tqdm  # type: ignore[import-not-found]
    except ImportError:
        from tqdm import tqdm  # type: ignore[import-not-found]
    return tqdm


def _open_sb3_style_progress(
    cfg: Any,
    *,
    total_timesteps: int,
    current_num_timesteps: int,
) -> Any:
    if not bool(getattr(cfg, "enable_progress_bar", True)):
        return None
    rem = int(total_timesteps) - int(current_num_timesteps)
    if rem <= 0:
        return None
    try:
        tqdm = _tqdm_for_sb3_progress()
    except ImportError:
        print(
            "[PPO] Install tqdm and rich for the SB3-style bar:  pip install tqdm rich",
            file=sys.stderr,
        )
        return None
    return tqdm(
        total=rem,
        dynamic_ncols=True,
        file=sys.stderr,
        mininterval=0.2,
    )


class CustomPPOTrainer:
    """Small PPO trainer that owns rollout, GAE, and update math locally."""

    def __init__(
        self,
        env,
        cfg,
        *,
        learning_rate: float,
        clip_range: float,
        ent_coef: float,
        n_epochs: int,
        batch_size: int,
        value_clip_range: Optional[float] = None,
        curriculum: Optional[Any] = None,
    ) -> None:
        """Construct a trainer.

        Hyperparameter resolution (~50 ``getattr(cfg, ..., default)`` calls
        in the legacy ``__init__``) has been extracted into
        :class:`~rl.custom_ppo.trainer_config.TrainerHyperparams`. The
        explicit kwargs (``learning_rate`` / ``clip_range`` / ``ent_coef``
        / ``n_epochs`` / ``batch_size`` / ``value_clip_range``) stay on
        the signature for backward compatibility with existing call sites
        (tests, ``tools/critic_ceiling.py``); see :meth:`from_config` for
        the ergonomic single-arg factory used by ``train_ppo``.
        """
        hparams = TrainerHyperparams.from_ppo_config(
            cfg,
            env,
            learning_rate=learning_rate,
            clip_range=clip_range,
            ent_coef=ent_coef,
            n_epochs=n_epochs,
            batch_size=batch_size,
            value_clip_range=value_clip_range,
            curriculum=curriculum,
        )
        self.env = env
        self.cfg = cfg
        self.hparams = hparams
        self.curriculum = curriculum
        self.device = torch.device(str(cfg.device))

        # Bulk-copy the resolved hyperparameters onto historical
        # ``self.<name>`` attributes — every downstream module reads
        # ``trainer.use_latent_strategy``, ``trainer.latent_k`` etc.,
        # so keeping those names live preserves the public surface.
        self.use_latent_strategy = hparams.use_latent_strategy
        self.latent_k = hparams.latent_k
        self.latent_resample_every_n = hparams.latent_resample_every_n
        self.fixed_latent_strategy = hparams.fixed_latent_strategy
        self.latent_gae_reset_on_z_change = hparams.latent_gae_reset_on_z_change
        self.latent_bootstrap_z_deterministic = hparams.latent_bootstrap_z_deterministic
        self.fixed_latent_strategy_id = hparams.fixed_latent_strategy_id

        self.model = SharedActorCentralizedCritic(
            env.observation_space, env.action_space, **build_model_kwargs(cfg, hparams)
        ).to(self.device)
        apply_deterministic_sampling_generators(
            self.model, int(getattr(cfg, "seed", 0) or 0), device=self.device
        )
        from rl.custom_ppo.curriculum_gates import is_staged_v6i1_curriculum

        if is_staged_v6i1_curriculum(cfg):
            from rl.custom_ppo.v6i1_phase_runtime import build_v6i1_optimizers

            build_v6i1_optimizers(self, base_lr=hparams.learning_rate)
            self.base_learning_rate = hparams.learning_rate
        else:
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=hparams.learning_rate, eps=1e-5
            )
            self.base_learning_rate = hparams.learning_rate
            self.v6i1_three_optimizer_mode = False
            self.actor_optimizer = self.optimizer
            self.critic_optimizer = self.optimizer
            self.router_optimizer = None

        # Dedicated optimizer for the q_phi strategy encoder and the
        # episode_strategy_value_head when ``latent_episode_strategy_lr`` is set.
        # Rationale: the shared optimizer's LR is calibrated for the noisy actor;
        # q_phi's per-update gradient (post-marginal-baseline) is real but small
        # and cannot move logits enough at the actor's LR in 15 update cycles.
        # A separate AdamW with a higher LR (e.g. 1e-3 to 1e-2) gives the router
        # the larger effective step it needs. Under Fix 5, the shared optimizer
        # never updates the strategy_encoder anyway (zero gradient via the
        # main-loop coef-gate), so there is no double-stepping concern.
        self.latent_router_optimizer: Optional[torch.optim.Optimizer] = None
        if (
            not getattr(self, "v6i1_three_optimizer_mode", False)
            and hparams.latent_episode_strategy_lr is not None
            and hparams.use_latent_strategy
            and not hparams.fixed_latent_strategy
        ):
            router_params: list[torch.nn.Parameter] = []
            strategy_encoder = getattr(self.model, "strategy_encoder", None)
            if strategy_encoder is not None:
                router_params.extend(p for p in strategy_encoder.parameters() if p.requires_grad)
            value_head = getattr(self.model, "episode_strategy_value_head", None)
            if value_head is not None:
                router_params.extend(p for p in value_head.parameters() if p.requires_grad)
            if router_params:
                self.latent_router_optimizer = torch.optim.AdamW(
                    router_params,
                    lr=float(hparams.latent_episode_strategy_lr),
                    eps=1e-5,
                )
        self.clip_range = hparams.clip_range
        self.ent_coef = hparams.ent_coef
        self.vf_coef = hparams.vf_coef
        self.n_epochs = hparams.n_epochs
        self.batch_size = hparams.batch_size
        self.value_clip_range = hparams.value_clip_range
        self.normalize_returns = hparams.normalize_returns

        # Running return-normalization stats live on these two sub-components.
        # ``strategy_return_norm`` is always materialized; its update site
        # (``_update_strategy_return_stats``) gates on
        # ``latent_strategy_aux_return_head`` so an instance with the default
        # mean=0 / var=1 simply passes values through when the aux head is off.
        self.return_norm = ReturnNormalizer(enabled=hparams.normalize_returns)
        self.strategy_return_norm = ReturnNormalizer(enabled=True)

        self.latent_strategy_ppo_coef = hparams.latent_strategy_ppo_coef
        self.latent_episode_strategy_ppo = hparams.latent_episode_strategy_ppo
        self.latent_episode_strategy_coef = hparams.latent_episode_strategy_coef
        self.latent_episode_strategy_clip_eps = hparams.latent_episode_strategy_clip_eps
        self.latent_episode_strategy_value_coef = hparams.latent_episode_strategy_value_coef
        self.latent_episode_strategy_return_norm = hparams.latent_episode_strategy_return_norm
        self.latent_episode_strategy_warmup_decision_steps = (
            hparams.latent_episode_strategy_warmup_decision_steps
        )
        self.latent_episode_strategy_n_epochs = hparams.latent_episode_strategy_n_epochs
        self.latent_episode_strategy_lr = hparams.latent_episode_strategy_lr
        # v3i19 arc-credit channel.
        self.latent_arc_credit_enabled = hparams.latent_arc_credit_enabled
        self.latent_arc_credit_coef = hparams.latent_arc_credit_coef
        self.latent_arc_credit_n_epochs = hparams.latent_arc_credit_n_epochs
        self.latent_arc_credit_clip_eps = hparams.latent_arc_credit_clip_eps
        self.latent_arc_credit_return_norm = hparams.latent_arc_credit_return_norm
        self.latent_arc_credit_baseline = hparams.latent_arc_credit_baseline
        self.latent_arc_credit_min_len = hparams.latent_arc_credit_min_len
        self.latent_q_phi_marginal_baseline = hparams.latent_q_phi_marginal_baseline
        self.latent_q_phi_bucket_baseline = hparams.latent_q_phi_bucket_baseline
        self.latent_q_phi_bucket_baseline_ema = hparams.latent_q_phi_bucket_baseline_ema
        self.latent_q_phi_bucket_baseline_min_count = hparams.latent_q_phi_bucket_baseline_min_count
        self.latent_forced_z_episode_frac = hparams.latent_forced_z_episode_frac
        self.latent_behavior_contrast_coef = hparams.latent_behavior_contrast_coef
        self.latent_behavior_contrast_margin = hparams.latent_behavior_contrast_margin
        self.latent_behavior_contrast_ema = hparams.latent_behavior_contrast_ema
        self.latent_behavior_contrast_anneal_after_steps = (
            hparams.latent_behavior_contrast_anneal_after_steps
        )
        self.latent_behavior_contrast_anneal_to = hparams.latent_behavior_contrast_anneal_to
        self.latent_usage_balance_coef = hparams.latent_usage_balance_coef
        self.latent_q_phi_train_after_steps = hparams.latent_q_phi_train_after_steps
        self.latent_preference_coef = hparams.latent_preference_coef
        self.latent_preference_temperature = hparams.latent_preference_temperature
        self.latent_preference_min_bucket_count = hparams.latent_preference_min_bucket_count
        self.latent_preference_min_distinct_z = hparams.latent_preference_min_distinct_z
        self.latent_preference_opponent_balanced = hparams.latent_preference_opponent_balanced
        self.latent_preference_log_opponent_targets = hparams.latent_preference_log_opponent_targets
        self.latent_preference_confidence_scale = hparams.latent_preference_confidence_scale
        self.latent_preference_commit_coef = hparams.latent_preference_commit_coef
        self.latent_awrd_enabled = hparams.latent_awrd_enabled
        self.latent_awrd_coef = hparams.latent_awrd_coef
        self.latent_awrd_temperature = hparams.latent_awrd_temperature
        self.latent_awrd_min_bucket_count = hparams.latent_awrd_min_bucket_count
        self.latent_awrd_min_distinct_z = hparams.latent_awrd_min_distinct_z
        self.latent_awrd_margin_threshold = hparams.latent_awrd_margin_threshold
        self.latent_awrd_margin_scale = hparams.latent_awrd_margin_scale
        self.latent_awrd_min_margin = hparams.latent_awrd_min_margin
        self.latent_awrd_soft_margin_gating = hparams.latent_awrd_soft_margin_gating
        self.latent_awrd_warmup_steps = hparams.latent_awrd_warmup_steps
        self.latent_awrd_ramp_steps = hparams.latent_awrd_ramp_steps
        self.latent_specialist_router_enabled = hparams.latent_specialist_router_enabled
        self.latent_marginal_balance_coef = hparams.latent_marginal_balance_coef
        self.latent_conditional_entropy_min_coef = hparams.latent_conditional_entropy_min_coef
        self.latent_conditional_entropy_min_coef_start = (
            hparams.latent_conditional_entropy_min_coef_start
        )
        self.latent_specialist_conditional_entropy_scope = (
            hparams.latent_specialist_conditional_entropy_scope
        )
        self.latent_context_mi_coef = hparams.latent_context_mi_coef
        self.latent_specialist_warmup_steps = hparams.latent_specialist_warmup_steps
        self.latent_specialist_ramp_steps = hparams.latent_specialist_ramp_steps
        self.latent_specialist_min_bucket_count = hparams.latent_specialist_min_bucket_count
        self.latent_specialist_context_key_mode = hparams.latent_specialist_context_key_mode
        self.latent_specialist_use_rollout_states = (
            hparams.latent_specialist_use_rollout_states
        )
        self.latent_specialist_rollout_max_samples = (
            hparams.latent_specialist_rollout_max_samples
        )
        self.late_entropy_floor = hparams.late_entropy_floor
        self.commitment_type = hparams.commitment_type
        self.latent_event_refresh_enabled = hparams.latent_event_refresh_enabled
        self.latent_event_refresh_min_gap_steps = hparams.latent_event_refresh_min_gap_steps
        self.latent_event_refresh_max_per_episode = hparams.latent_event_refresh_max_per_episode
        self.latent_event_refresh_use_q_phi = hparams.latent_event_refresh_use_q_phi
        self.latent_event_refresh_force_roles = hparams.latent_event_refresh_force_roles
        self.latent_sparse_tactical_refresh_enabled = (
            hparams.latent_sparse_tactical_refresh_enabled
        )
        self.latent_sparse_tactical_refresh_interval_steps = (
            hparams.latent_sparse_tactical_refresh_interval_steps
        )
        self.latent_sparse_tactical_refresh_min_dwell_steps = (
            hparams.latent_sparse_tactical_refresh_min_dwell_steps
        )
        self.latent_v3i3_event_preference_enabled = hparams.latent_v3i3_event_preference_enabled
        self.latent_v3i3_event_preference_coef = hparams.latent_v3i3_event_preference_coef
        self.latent_v3i3_event_preference_temperature = hparams.latent_v3i3_event_preference_temperature
        self.latent_v3i3_event_preference_min_bucket_count = (
            hparams.latent_v3i3_event_preference_min_bucket_count
        )
        self.latent_v3i3_event_preference_min_distinct_z = (
            hparams.latent_v3i3_event_preference_min_distinct_z
        )
        self.latent_v3i3_event_preference_buffer_size = (
            hparams.latent_v3i3_event_preference_buffer_size
        )
        self.latent_v3i3_event_preference_warmup_steps = (
            hparams.latent_v3i3_event_preference_warmup_steps
        )
        self.latent_v3i3_event_preference_normalize = (
            hparams.latent_v3i3_event_preference_normalize
        )
        self.latent_v3i3_refresh_log_enabled = hparams.latent_v3i3_refresh_log_enabled
        self.latent_v3i3_refresh_log_path = hparams.latent_v3i3_refresh_log_path
        self.latent_event_preference_key_mode = hparams.latent_event_preference_key_mode

        # Bucket-baseline helper for v3d. Only constructed when the mode is
        # set; ``apply_episode_strategy_ppo`` falls back to the V-marginal /
        # legacy baseline when this is ``None``. State (per-bucket EMAs +
        # global mean) lives on the helper across rollouts.
        from rl.custom_ppo.latent_bucket_baseline import BucketBaseline
        self.latent_bucket_baseline: Optional[BucketBaseline] = None
        if self.latent_q_phi_bucket_baseline is not None:
            self.latent_bucket_baseline = BucketBaseline(
                ema=self.latent_q_phi_bucket_baseline_ema,
                min_count=self.latent_q_phi_bucket_baseline_min_count,
            )
        self.latent_behavior_contrast: Optional[BehaviorContrastMemory] = None
        if (
            hparams.use_latent_strategy
            and not hparams.fixed_latent_strategy
            and hparams.latent_behavior_contrast_coef > 0.0
            and hparams.latent_forced_z_episode_frac > 0.0
        ):
            self.latent_behavior_contrast = BehaviorContrastMemory(
                latent_k=hparams.latent_k,
                ema=hparams.latent_behavior_contrast_ema,
                margin=hparams.latent_behavior_contrast_margin,
                device=self.device,
            )
        self.latent_strategy_aux_return_coef = hparams.latent_strategy_aux_return_coef
        self.latent_strategy_aux_return_head = hparams.latent_strategy_aux_return_head
        self.latent_strategy_aux_predict_phase_coef = hparams.latent_strategy_aux_predict_phase_coef

        self.reward_shaping_coef_start = hparams.reward_shaping_coef_start
        self.reward_shaping_coef_end = hparams.reward_shaping_coef_end
        self.reward_shaping_decay_steps = hparams.reward_shaping_decay_steps
        self.reward_dense_weight = hparams.reward_dense_weight
        self.reward_scale = hparams.reward_scale
        self.reward_clip = hparams.reward_clip
        self.reward_stalemate_penalty = hparams.reward_stalemate_penalty

        self.periodic_checkpoint_steps = hparams.periodic_checkpoint_steps
        self._next_periodic_checkpoint_step = (
            self.periodic_checkpoint_steps if self.periodic_checkpoint_steps > 0 else 0
        )

        # v4i4post periodic router-distillation hook. Stays a no-op unless
        # the ``latent_router_distill_enabled`` flag (or the
        # ``v4i4post`` / legacy alias preset) turns it on. The canonical
        # v4i3 (Summer Proof Suite) leaves this off. The hook owns its own
        # cadence counter, subprocess
        # plumbing, and Adam-moment reset; the trainer just hands it a
        # checkpoint path after each periodic save.
        self.router_distill_hook = PeriodicRouterDistillHook(
            cfg=cfg,
            run_tag=str(getattr(cfg, "run_tag", "ppo")),
            checkpoint_dir=str(getattr(cfg, "checkpoint_dir", "checkpoints")),
        )

        self.global_step = 0
        self.last_stats: dict[str, float] = {}
        self.run_id = hparams.run_id
        self.run_pid = hparams.run_pid
        self._updates_completed = 0
        self.episode_stats = EpisodeStats(success_window=200)
        self.metrics_csv_path = hparams.metrics_csv_path
        self.episode_csv_path = hparams.episode_csv_path
        self.strategy_experience_csv_path = hparams.strategy_experience_csv_path

        self.telemetry = TrainingTelemetry(
            cfg=self.cfg,
            hparams=self.hparams,
            curriculum=self.curriculum,
            reward_shaping_coef=self._reward_shaping_coef,
            runtime=self,
        )
        if self.use_latent_strategy:
            self.temporal_tracker = TemporalStateTracker(
                num_envs=int(env.num_envs),
                state_dim=GLOBAL_STATE_DIM,
                device=self.device,
            )
        else:
            self.temporal_tracker = None
        self.latent_resample_on_flag = hparams.latent_resample_on_flag
        self.latent_kl_consecutive = hparams.latent_kl_consecutive
        self.latent_state = LatentStrategyState(self)
        self.episode_strategy_recorder = self.latent_state.episode_strategy_recorder
        if bool(getattr(cfg, "use_v6i1_curriculum", False)):
            from rl.custom_ppo.curriculum_gates import V6I1CurriculumController, is_staged_v6i1_curriculum

            if is_staged_v6i1_curriculum(cfg):
                from rl.custom_ppo.v6i1_phase_runtime import build_v6i1_optimizers

                self.v6i1_curriculum = V6I1CurriculumController(self)
                build_v6i1_optimizers(self, base_lr=hparams.learning_rate)
            else:
                self.v6i1_curriculum = None
        else:
            self.v6i1_curriculum = None
        self.rollout_collector = RolloutCollector(
            model=self.model,
            env=self.env,
            device=self.device,
            cfg=self.cfg,
            hparams=self.hparams,
            latent_state=self.latent_state,
            telemetry=self.telemetry,
            episode_stats=self.episode_stats,
            temporal_tracker=self.temporal_tracker,
            reward_shaping_coef=self._reward_shaping_coef,
            runtime=self,
        )
        self.updater = PPOUpdater(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            cfg=self.cfg,
            hparams=self.hparams,
            latent_state=self.latent_state,
            runtime=self,
        )
        self._sb3_rollout_pbar = None
        self._last_obs = None
        self._last_global_state = None
        self._last_context_state = None
        self._decentralized_actor_contract_logged = False

        self._opponent_randomize_training = hparams.opponent_randomize_training
        # ``list`` (not tuple) for back-compat with downstream callers that
        # historically appended/sliced this attribute.
        self._opponent_pool_tags = list(hparams.opponent_pool_tags)
        # ``None`` ⇒ uniform sampling. Otherwise a list of probabilities aligned
        # positionally with ``_opponent_pool_tags`` (already normalized upstream).
        weights = list(hparams.opponent_pool_weights) if hparams.opponent_pool_weights else None
        if weights is not None and len(weights) != len(self._opponent_pool_tags):
            raise ValueError(
                f"opponent_pool_weights length {len(weights)} does not match "
                f"opponent_pool_tags length {len(self._opponent_pool_tags)}."
            )
        self._opponent_pool_weights = weights
        self._rng_opponent = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 901)
        if self._opponent_randomize_training:
            if not self._opponent_pool_tags:
                raise ValueError(
                    "Opponent pool training (mode=OPPONENT_POOL or opponent_randomize) requires a non-empty "
                    "opponent_pool (e.g. OP1–OP3, OP5–OP7; OP4 optional with --allow-op4-in-training-pool)."
                )
            self.env._before_reset_indices_hook = partial(
                _hook_sample_training_opponent_before_reset, self
            )

    @classmethod
    def from_config(
        cls,
        env,
        cfg,
        *,
        curriculum: Optional[Any] = None,
    ) -> "CustomPPOTrainer":
        """Build a trainer using PPO hyperparameters from ``cfg`` directly.

        Eliminates the boilerplate where every caller re-passed
        ``cfg.learning_rate`` / ``cfg.clip_range`` / etc. as keyword args
        (``train_ppo.py``, ``tools/critic_ceiling.py``). The legacy
        kwargs constructor stays available for tests that override the
        defaults inline.
        """
        return cls(
            env,
            cfg,
            learning_rate=float(cfg.learning_rate),
            clip_range=float(cfg.clip_range),
            ent_coef=float(cfg.ent_coef),
            n_epochs=int(cfg.n_epochs),
            batch_size=int(cfg.batch_size),
            value_clip_range=getattr(cfg, "clip_range_vf", cfg.clip_range),
            curriculum=curriculum,
        )

    def _reward_shaping_coef(self) -> float:
        if self.reward_shaping_decay_steps <= 0:
            return float(self.reward_shaping_coef_start)
        frac = min(1.0, max(0.0, float(self.global_step) / float(self.reward_shaping_decay_steps)))
        return float(self.reward_shaping_coef_start + frac * (self.reward_shaping_coef_end - self.reward_shaping_coef_start))

    def collect_rollout(self) -> TensorDictRolloutBuffer:
        """Collect one rollout and compute advantages/returns.

        Thin delegate: actual stepping / buffer-filling / GAE / option-return
        logic lives in :class:`~rl.custom_ppo.rollout_collector.RolloutCollector`.
        """
        return self.rollout_collector.collect()

    def update(self, buffer: TensorDictRolloutBuffer, *, total_timesteps: int) -> dict[str, float]:
        """Run PPO epochs over one rollout.

        Thin delegate: schedules / minibatch loop / losses / optimizer
        step / KL early-stop / post-update diagnostics live in
        :class:`~rl.custom_ppo.ppo_updater.PPOUpdater`. The trainer still
        owns ``last_stats``, the optimizer, the model, and the latent /
        return-norm state that the updater mutates via the context object.
        """
        return self.updater.update(buffer, total_timesteps=total_timesteps)

    def _save_periodic_checkpoint(self) -> None:
        if self.periodic_checkpoint_steps <= 0:
            return
        while self.global_step >= self._next_periodic_checkpoint_step:
            ckpt_name = f"ckpt_{str(getattr(self.cfg, 'run_tag', 'ppo'))}_{int(self._next_periodic_checkpoint_step)}.zip"
            ckpt_path = os.path.join(str(getattr(self.cfg, "checkpoint_dir", "checkpoints")), ckpt_name)
            self.save(ckpt_path)
            print(f"[PPO] Periodic checkpoint saved: {ckpt_path}")
            # v4i4post hook (no-op unless ``latent_router_distill_enabled``):
            # run q_probe + router distillation against the just-saved checkpoint
            # and hot-swap the distilled ``strategy_encoder.*`` weights back
            # into the running model. Always best-effort: any failure here
            # cannot block training.
            self.router_distill_hook.maybe_run(
                trainer=self,
                ckpt_path=ckpt_path,
                global_step=int(self.global_step),
            )
            self._next_periodic_checkpoint_step += self.periodic_checkpoint_steps

    def learn(self, total_timesteps: int) -> dict[str, float]:
        """Train until at least ``total_timesteps`` environment transitions have been collected."""
        total = int(total_timesteps)
        self._sb3_rollout_pbar = _open_sb3_style_progress(
            self.cfg, total_timesteps=total, current_num_timesteps=self.global_step
        )
        try:
            while self.global_step < total:
                rollout = self.collect_rollout()
                stats = self.update(rollout, total_timesteps=total)
                self._updates_completed += 1
                row = self.telemetry.write_update_metrics(stats, rollout)
                self._save_periodic_checkpoint()
                self.telemetry.print_update_diagnostics(row, stats)
                if getattr(self, "v6i1_curriculum", None) is not None:
                    self.v6i1_curriculum.maybe_apply_phase_transitions()
                    self.v6i1_curriculum.check_and_run_gate()
                    self.v6i1_curriculum.check_terminal_failure()
        finally:
            if self._sb3_rollout_pbar is not None:
                self._sb3_rollout_pbar.refresh()  # type: ignore[union-attr]
                self._sb3_rollout_pbar.close()  # type: ignore[union-attr]
                self._sb3_rollout_pbar = None
            self.telemetry.close_e3_step_telemetry()
        return self.last_stats

    def save(self, path: str) -> None:
        """Save a torch checkpoint. The project keeps the historical ``.zip`` suffix."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        rn = self.return_norm.state_dict()
        srn = self.strategy_return_norm.state_dict()
        payload: dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "updates_completed": self._updates_completed,
            "return_norm_mean": rn["mean"],
            "return_norm_var": rn["var"],
            "return_norm_count": rn["count"],
            "strategy_return_mean": srn["mean"],
            "strategy_return_var": srn["var"],
            "strategy_return_count": srn["count"],
            "cfg": asdict(self.cfg),
            "last_stats": self.last_stats,
            "format": CUSTOM_PPO_LATENT_FORMAT if self.use_latent_strategy else CUSTOM_PPO_FORMAT,
            "actor_arch": CUSTOM_PPO_ACTOR_ARCH,
            "actor_cnn_feature_dim": int(self.model.actor_cnn_feature_dim),
            "global_state_dim": int(self.model.global_state_dim),
            "vec_schema_version": CUSTOM_PPO_VEC_SCHEMA_VERSION,
        }
        if getattr(self, "v6i1_three_optimizer_mode", False):
            from rl.custom_ppo.v6i1_phase_runtime import (
                latent_state_v6i1_checkpoint,
                v6i1_curriculum_state_dict,
            )

            payload["v6i1_three_optimizer_mode"] = True
            payload["actor_optimizer_state_dict"] = self.actor_optimizer.state_dict()
            payload["critic_optimizer_state_dict"] = self.critic_optimizer.state_dict()
            payload["router_optimizer_state_dict"] = self.router_optimizer.state_dict()
            if getattr(self, "v6i1_curriculum", None) is not None:
                payload["v6i1_curriculum_state"] = v6i1_curriculum_state_dict(self.v6i1_curriculum)
            payload["latent_state_v6i1"] = latent_state_v6i1_checkpoint(self.latent_state)
        torch.save(payload, path)

    def load(self, path: str) -> None:
        """Restore a checkpoint produced by :meth:`save`."""
        payload = _torch_load_checkpoint(path, map_location=self.device)
        _assert_compatible_global_state_dim(payload, path)
        _load_model_state_dict_compat(self.model, payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        v6i1_latent_payload: dict[str, Any] = dict(payload.get("latent_state_v6i1", {}) or {})
        if bool(payload.get("v6i1_three_optimizer_mode", False)):
            from rl.custom_ppo.v6i1_phase_runtime import (
                load_v6i1_curriculum_state,
                restore_latent_state_v6i1_checkpoint,
            )

            if getattr(self, "actor_optimizer", None) is not None and "actor_optimizer_state_dict" in payload:
                self.actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
            if getattr(self, "critic_optimizer", None) is not None and "critic_optimizer_state_dict" in payload:
                self.critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
            if getattr(self, "router_optimizer", None) is not None and "router_optimizer_state_dict" in payload:
                self.router_optimizer.load_state_dict(payload["router_optimizer_state_dict"])
                self.latent_router_optimizer = self.router_optimizer
            if getattr(self, "v6i1_curriculum", None) is not None and "v6i1_curriculum_state" in payload:
                load_v6i1_curriculum_state(self.v6i1_curriculum, payload["v6i1_curriculum_state"])
        self.global_step = int(payload.get("global_step", 0))
        self._updates_completed = int(payload.get("updates_completed", 0))
        self.return_norm.load_state_dict(
            {
                "mean": payload.get("return_norm_mean", 0.0),
                "var": payload.get("return_norm_var", 1.0),
                "count": payload.get("return_norm_count", 1e-4),
            }
        )
        self.strategy_return_norm.load_state_dict(
            {
                "mean": payload.get("strategy_return_mean", 0.0),
                "var": payload.get("strategy_return_var", 1.0),
                "count": payload.get("strategy_return_count", 1e-4),
            }
        )
        self.last_stats = dict(payload.get("last_stats", {}))
        self._last_obs = None
        self._last_global_state = None
        self.latent_state.current_z = None
        if self.use_latent_strategy:
            self.latent_state.reset()
        if v6i1_latent_payload:
            from rl.custom_ppo.v6i1_phase_runtime import restore_latent_state_v6i1_checkpoint

            restore_latent_state_v6i1_checkpoint(self.latent_state, v6i1_latent_payload)
