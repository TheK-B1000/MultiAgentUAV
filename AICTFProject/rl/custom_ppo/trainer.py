from __future__ import annotations

import os
import sys
import time
import warnings
from dataclasses import asdict, fields
from typing import Any, Callable, Optional

import torch
from rl.global_state import GLOBAL_STATE_DIM
from rl.latent_marl import TemporalStateTracker

from rl.custom_ppo.communication import CommRolloutRuntime, extend_observation_space_if_needed
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
from rl.custom_ppo.curriculum_runtime import TrainingOpponentPool
from rl.custom_ppo.episode_stats import EpisodeStats
from rl.custom_ppo.latent_behavior_contrast import BehaviorContrastMemory
from rl.custom_ppo.latent.state import LatentStrategyState
from rl.custom_ppo.ppo_updater import PPOUpdater
from rl.custom_ppo.return_normalization import ReturnNormalizer
from rl.custom_ppo.rollout_collector import RolloutCollector
from rl.custom_ppo.router_distill_hook import PeriodicRouterDistillHook
from rl.custom_ppo.trainer_config import TrainerHyperparams, build_model_kwargs
from rl.custom_ppo.trainer_optimizers import TrainerOptimizerBundle
from rl.custom_ppo.training_telemetry import TrainingTelemetry
# Re-exported for back-compat (``rl.custom_ppo._compose_training_reward_components``).
from rl.custom_ppo.reward_composition import _compose_training_reward_components  # noqa: F401

# Hyperparameter aliases not stored under the same name on TrainerHyperparams.
_HP_ATTR_ALIASES: dict[str, str] = {
    "base_learning_rate": "learning_rate",
}

# Backward-compatible optimizer attribute names → bundle accessors.
_OPTIMIZER_ATTR_ACCESSORS: dict[str, Callable[["CustomPPOTrainer"], Any]] = {
    "optimizer": lambda t: t.optimizers.primary,
    "actor_optimizer": lambda t: t.optimizers.actor,
    "actor_cf_optimizer": lambda t: t.optimizers.actor_cf,
    "critic_optimizer": lambda t: t.optimizers.critic,
    "router_optimizer": lambda t: t.optimizers.router,
    "latent_router_optimizer": lambda t: t.optimizers.latent_router_optimizer,
    "v6i1_three_optimizer_mode": lambda t: t.optimizers.v6i1_three_optimizer_mode,
}

# Legacy private opponent-pool fields used by older hooks / tests.
_OPPONENT_POOL_ATTR_ALIASES: dict[str, str] = {
    "_opponent_randomize_training": "enabled",
    "_opponent_pool_tags": "tags",
    "_opponent_pool_weights": "weights",
    "_rng_opponent": "rng",
}


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

        Resolved hyperparameters live on :attr:`hparams`. Legacy call sites
        that read ``trainer.latent_k`` etc. are served by :meth:`__getattr__`.
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

        policy_obs_space = extend_observation_space_if_needed(env.observation_space, cfg)
        self.model = SharedActorCentralizedCritic(
            policy_obs_space, env.action_space, **build_model_kwargs(cfg, hparams)
        ).to(self.device)
        self.comm_runtime = CommRolloutRuntime(cfg, device=self.device)
        self.comm_runtime.reset(
            batch_size=int(env.num_envs),
            num_agents=int(self.model.n_agents),
        )
        self.comm_runtime.bind_env_core(env.core)
        apply_deterministic_sampling_generators(
            self.model, int(getattr(cfg, "seed", 0) or 0), device=self.device
        )
        self.optimizers = TrainerOptimizerBundle.build(model=self.model, cfg=cfg, hparams=hparams)

        self.return_norm = ReturnNormalizer(enabled=hparams.normalize_returns)
        self.strategy_return_norm = ReturnNormalizer(enabled=True)
        self._init_optional_latent_helpers(hparams)

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
        self._next_periodic_checkpoint_step = (
            hparams.periodic_checkpoint_steps if hparams.periodic_checkpoint_steps > 0 else 0
        )

        self.opponent_pool = TrainingOpponentPool.from_hparams(cfg, hparams)
        self.opponent_pool.attach_before_reset_hook(self.env, self)

        self.telemetry = TrainingTelemetry(
            cfg=self.cfg,
            hparams=self.hparams,
            curriculum=self.curriculum,
            reward_shaping_coef=self._reward_shaping_coef,
            runtime=self,
        )
        if hparams.use_latent_strategy:
            self.temporal_tracker = TemporalStateTracker(
                num_envs=int(env.num_envs),
                state_dim=GLOBAL_STATE_DIM,
                device=self.device,
            )
        else:
            self.temporal_tracker = None
        self.latent_state = LatentStrategyState(self)
        self.episode_strategy_recorder = self.latent_state.episode_strategy_recorder
        self.v6i1_curriculum = self._build_v6i1_curriculum(cfg)
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
            optimizer=self.optimizers.primary,
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

    def _reset_module_parameters(self, module: torch.nn.Module) -> int:
        n = 0
        for child in module.modules():
            reset = getattr(child, "reset_parameters", None)
            if callable(reset):
                reset()
                n += 1
        return n

    def _clear_optimizer_state_for_params(self, params: list[torch.nn.Parameter]) -> int:
        wanted = {id(p) for p in params}
        cleared = 0
        for optimizer in (
            self.optimizers.primary,
            self.optimizers.actor,
            self.optimizers.critic,
            self.optimizers.router,
            self.optimizers.actor_cf,
        ):
            if optimizer is None:
                continue
            for param in list(optimizer.state.keys()):
                if id(param) in wanted:
                    optimizer.state.pop(param, None)
                    cleared += 1
        return cleared

    def _reinitialize_router_after_load(self) -> None:
        if not self.use_latent_strategy or not hasattr(self.model, "strategy_encoder"):
            return
        router_params = [p for p in self.model.strategy_encoder.parameters()]
        n_modules = self._reset_module_parameters(self.model.strategy_encoder)
        n_states = self._clear_optimizer_state_for_params(router_params)
        print(
            "[PPO] Router reinitialized after checkpoint load: "
            f"strategy_encoder_reset_modules={n_modules}, optimizer_states_cleared={n_states}"
        )

    def __getattr__(self, name: str) -> Any:
        if name in _OPTIMIZER_ATTR_ACCESSORS:
            return _OPTIMIZER_ATTR_ACCESSORS[name](self)
        if name in _OPPONENT_POOL_ATTR_ALIASES:
            return getattr(self.opponent_pool, _OPPONENT_POOL_ATTR_ALIASES[name])
        hp_name = _HP_ATTR_ALIASES.get(name, name)
        try:
            return getattr(self.hparams, hp_name)
        except AttributeError as exc:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from exc

    def __dir__(self) -> list[str]:
        return sorted(
            set(object.__dir__(self))
            | set(_OPTIMIZER_ATTR_ACCESSORS)
            | set(_OPPONENT_POOL_ATTR_ALIASES)
            | set(_HP_ATTR_ALIASES)
            | {f.name for f in fields(self.hparams)}
        )

    def _build_v6i1_curriculum(self, cfg: Any) -> Any:
        if not bool(getattr(cfg, "use_v6i1_curriculum", False)):
            return None
        from rl.custom_ppo.curriculum_gates import V6I1CurriculumController
        from rl.custom_ppo.gate_protocol import is_staged_v6_team_intent_curriculum

        if not is_staged_v6_team_intent_curriculum(cfg):
            return None
        return V6I1CurriculumController(self)

    def _init_optional_latent_helpers(self, hparams: TrainerHyperparams) -> None:
        from rl.custom_ppo.latent_bucket_baseline import BucketBaseline

        self.latent_bucket_baseline: Optional[BucketBaseline] = None
        if hparams.latent_q_phi_bucket_baseline is not None:
            self.latent_bucket_baseline = BucketBaseline(
                ema=hparams.latent_q_phi_bucket_baseline_ema,
                min_count=hparams.latent_q_phi_bucket_baseline_min_count,
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

    @classmethod
    def from_config(
        cls,
        env,
        cfg,
        *,
        curriculum: Optional[Any] = None,
    ) -> "CustomPPOTrainer":
        """Build a trainer using PPO hyperparameters from ``cfg`` directly."""
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
        return float(
            self.reward_shaping_coef_start
            + frac * (self.reward_shaping_coef_end - self.reward_shaping_coef_start)
        )

    def collect_rollout(self) -> TensorDictRolloutBuffer:
        """Collect one rollout and compute advantages/returns."""
        return self.rollout_collector.collect()

    def update(self, buffer: TensorDictRolloutBuffer, *, total_timesteps: int) -> dict[str, float]:
        """Run PPO epochs over one rollout."""
        return self.updater.update(buffer, total_timesteps=total_timesteps)

    def _save_periodic_checkpoint(self) -> None:
        if self.periodic_checkpoint_steps <= 0:
            return
        while self.global_step >= self._next_periodic_checkpoint_step:
            ckpt_name = f"ckpt_{str(getattr(self.cfg, 'run_tag', 'ppo'))}_{int(self._next_periodic_checkpoint_step)}.zip"
            ckpt_path = os.path.join(str(getattr(self.cfg, "checkpoint_dir", "checkpoints")), ckpt_name)
            self.save(ckpt_path)
            print(f"[PPO] Periodic checkpoint saved: {ckpt_path}")
            self.router_distill_hook.maybe_run(
                trainer=self,
                ckpt_path=ckpt_path,
                global_step=int(self.global_step),
            )
            self._next_periodic_checkpoint_step += self.periodic_checkpoint_steps
    def learn(self, total_timesteps: int) -> None:
        """Run training until global_step reaches ``total_timesteps``."""
        total = int(total_timesteps)
        self._sb3_rollout_pbar = _open_sb3_style_progress(
            self.cfg, total_timesteps=total, current_num_timesteps=self.global_step
        )
        train_start = time.perf_counter()
        self.telemetry.emit_training_started(
            total_timesteps=total,
            checkpoint_path=getattr(self.cfg, "load_path", None),
        )
        try:
            while self.global_step < total:
                rollout_start = time.perf_counter()
                rollout = self.collect_rollout()
                rollout_duration = max(0.0, time.perf_counter() - rollout_start)
                self.telemetry.emit_rollout_completed(
                    rollout,
                    duration_seconds=rollout_duration,
                )
                update_start = time.perf_counter()
                stats = self.update(rollout, total_timesteps=total)
                stats = dict(stats)
                stats["optimization_duration_seconds"] = max(0.0, time.perf_counter() - update_start)
                self._updates_completed += 1
                row = self.telemetry.write_update_metrics(stats, rollout)
                self.telemetry.emit_performance_sample(phase="update")
                self._save_periodic_checkpoint()
                self.telemetry.print_update_diagnostics(row, stats)
                if self.v6i1_curriculum is not None:
                    old_phase = self.v6i1_curriculum.phase
                    self.v6i1_curriculum.maybe_apply_phase_transitions()
                    if self.v6i1_curriculum.phase != old_phase:
                        if torch.cuda.is_available():
                            torch.cuda.reset_peak_memory_stats()
                    self.v6i1_curriculum.check_and_run_gate()
                    self.v6i1_curriculum.check_terminal_failure()
                    total = max(
                        total,
                        int(self.v6i1_curriculum.effective_training_terminal_step()),
                    )
            training_duration = max(0.0, time.perf_counter() - train_start)
            self.telemetry.emit_training_completed(
                total_timesteps=total,
                duration_seconds=training_duration,
            )
            self.telemetry.write_performance_summary(
                training_duration_seconds=training_duration,
            )
        except (KeyboardInterrupt, SystemExit) as exc:
            training_duration = max(0.0, time.perf_counter() - train_start)
            self.telemetry.emit_training_interrupted(
                total_timesteps=total,
                duration_seconds=training_duration,
                reason=type(exc).__name__,
            )
            self.telemetry.write_performance_summary(
                training_duration_seconds=training_duration,
            )
            raise
        except BaseException as exc:
            training_duration = max(0.0, time.perf_counter() - train_start)
            self.telemetry.emit_training_failed(
                total_timesteps=total,
                duration_seconds=training_duration,
                error=exc,
            )
            self.telemetry.write_performance_summary(
                training_duration_seconds=training_duration,
            )
            raise
        finally:
            self.telemetry.close_e3_step_telemetry()
            if self._sb3_rollout_pbar is not None:
                try:
                    self._sb3_rollout_pbar.refresh()
                except Exception:
                    pass
                try:
                    self._sb3_rollout_pbar.close()
                except Exception:
                    pass
                self._sb3_rollout_pbar = None

    def save(self, path: str) -> None:
        """Save a torch checkpoint. The project keeps the historical ``.zip`` suffix."""
        start = time.perf_counter()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        rn = self.return_norm.state_dict()
        srn = self.strategy_return_norm.state_dict()
        payload: dict[str, Any] = {
            "model_state_dict": self.model.state_dict(),
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
        self.optimizers.write_checkpoint(payload)
        if self.v6i1_curriculum is not None:
            from rl.custom_ppo.v6i1_phase_runtime import (
                latent_state_v6i1_checkpoint,
                v6i1_curriculum_state_dict,
            )

            payload["v6i1_curriculum_state"] = v6i1_curriculum_state_dict(self.v6i1_curriculum)
            payload["latent_state_v6i1"] = latent_state_v6i1_checkpoint(self.latent_state)
        if self.comm_runtime.enabled:
            payload["comm_runtime_state"] = self.comm_runtime.state_dict()
        payload["ppo_updater_state"] = self.updater.state_dict()
        torch.save(payload, path)
        self.telemetry.emit_checkpoint_saved(
            path=path,
            duration_seconds=max(0.0, time.perf_counter() - start),
        )

    def load(self, path: str) -> None:
        """Restore a checkpoint produced by :meth:`save`."""
        start = time.perf_counter()
        payload = _torch_load_checkpoint(path, map_location=self.device)
        _assert_compatible_global_state_dim(payload, path)
        _load_model_state_dict_compat(
            self.model,
            payload["model_state_dict"],
            allow_active_actor_migration=bool(getattr(self.cfg, "allow_active_actor_module_migration", False)),
            checkpoint_cfg=payload.get("cfg"),
            target_cfg=self.cfg,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
        )
        load_weights_only = bool(getattr(self.cfg, "load_weights_only", False))
        if load_weights_only:
            print("[PPO] Skipping checkpoint optimizer state: --load-weights-only was set.")
        else:
            reinit_router = bool(getattr(self.cfg, "router_reinitialize_on_load", False))
            if reinit_router:
                print("[PPO] Skipping checkpoint optimizer state: router_reinitialize_on_load=True")
                self._reinitialize_router_after_load()
            else:
                allow_migration = bool(getattr(self.cfg, "allow_active_actor_module_migration", False))
                self.optimizers.load_checkpoint(payload, allow_architecture_migration=allow_migration)
        v6i1_latent_payload: dict[str, Any] = dict(payload.get("latent_state_v6i1", {}) or {})
        if self.v6i1_curriculum is not None and "v6i1_curriculum_state" in payload:
            from rl.custom_ppo.v6i1_phase_runtime import load_v6i1_curriculum_state

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
        if self.comm_runtime.enabled and "comm_runtime_state" in payload:
            self.comm_runtime.load_state_dict(dict(payload.get("comm_runtime_state", {}) or {}))
        self.updater.load_state_dict(dict(payload.get("ppo_updater_state", {}) or {}))
        self.telemetry.emit_checkpoint_loaded(
            path=path,
            duration_seconds=max(0.0, time.perf_counter() - start),
        )
