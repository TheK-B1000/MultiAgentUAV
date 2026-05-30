from __future__ import annotations

import os
import sys
import warnings
from collections import deque
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
from rl.custom_ppo.csv_writers import SCRIPTED_OPPONENT_MI_COUNT
from rl.custom_ppo.curriculum_runtime import _hook_sample_training_opponent_before_reset
from rl.custom_ppo.latent_strategy_state import LatentStrategyState
from rl.custom_ppo.ppo_updater import PPOUpdater
from rl.custom_ppo.rollout_collector import RolloutCollector
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
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=hparams.learning_rate, eps=1e-5
        )
        self.base_learning_rate = hparams.learning_rate
        self.clip_range = hparams.clip_range
        self.ent_coef = hparams.ent_coef
        self.vf_coef = hparams.vf_coef
        self.n_epochs = hparams.n_epochs
        self.batch_size = hparams.batch_size
        self.value_clip_range = hparams.value_clip_range
        self.normalize_returns = hparams.normalize_returns

        self._return_norm_mean = 0.0
        self._return_norm_var = 1.0
        self._return_norm_count = 1e-4
        self._strategy_return_mean = 0.0
        self._strategy_return_var = 1.0
        self._strategy_return_count = 1e-4

        self.latent_strategy_ppo_coef = hparams.latent_strategy_ppo_coef
        self.latent_episode_strategy_ppo = hparams.latent_episode_strategy_ppo
        self.latent_episode_strategy_coef = hparams.latent_episode_strategy_coef
        self.latent_episode_strategy_clip_eps = hparams.latent_episode_strategy_clip_eps
        self.latent_episode_strategy_value_coef = hparams.latent_episode_strategy_value_coef
        self.latent_episode_strategy_return_norm = hparams.latent_episode_strategy_return_norm
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

        self.global_step = 0
        self.last_stats: dict[str, float] = {}
        self.run_id = hparams.run_id
        self.run_pid = hparams.run_pid
        self._updates_completed = 0
        self._ep_wins = 0
        self._ep_losses = 0
        self._ep_draws = 0
        self._episodes_completed = 0
        self._rollout_episode_records: list[dict[str, Any]] = []
        self._recent_episode_successes = deque(maxlen=200)
        self.metrics_csv_path = hparams.metrics_csv_path
        self.episode_csv_path = hparams.episode_csv_path
        self.strategy_experience_csv_path = hparams.strategy_experience_csv_path

        self.telemetry = TrainingTelemetry(self)
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
        self.rollout_collector = RolloutCollector(self)
        self.updater = PPOUpdater(self)
        self._sb3_rollout_pbar = None
        self._last_obs = None
        self._last_global_state = None
        self._last_context_state = None
        self._decentralized_actor_contract_logged = False

        self._opponent_randomize_training = hparams.opponent_randomize_training
        # ``list`` (not tuple) for back-compat with downstream callers that
        # historically appended/sliced this attribute.
        self._opponent_pool_tags = list(hparams.opponent_pool_tags)
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
                if row:
                    z_wr_parts: list[str] = []
                    z_occ_parts: list[str] = []
                    if self.use_latent_strategy:
                        for i in range(self.latent_k):
                            wr = row.get(f"episode_z_{i}_win_rate", "")
                            occ = row.get(f"strategy_occupancy_{i}", "")
                            z_wr_parts.append("-" if wr == "" else f"{float(wr):.3f}")
                            z_occ_parts.append("-" if occ == "" else f"{float(occ):.3f}")
                    z_entropy = float(row.get("strategy_entropy", 0.0) or 0.0)
                    z_entropy_frac = float(row.get("strategy_entropy_frac", 0.0) or 0.0)
                    z_wr_spread = float(row.get("strategy_wr_spread", 0.0) or 0.0)
                    opp_suffix = ""
                    if self.use_latent_strategy:
                        mi_z_o = float(row.get("latent_mi_z_opponent_nats", 0.0) or 0.0)
                        mi_z_p = float(row.get("latent_mi_z_phase_nats", 0.0) or 0.0)
                        mi_z_y = float(row.get("latent_mi_z_outcome_nats", 0.0) or 0.0)
                        mi_z_f = float(row.get("latent_mi_z_flag_state_nats", 0.0) or 0.0)
                        opp_diag_bits: list[str] = []
                        for o in range(SCRIPTED_OPPONENT_MI_COUNT):
                            occ_o = [
                                float(row.get(f"strategy_occupancy_op{o}_z{k}", 0.0) or 0.0) for k in range(self.latent_k)
                            ]
                            wr_o = [row.get(f"episode_opp{o}_z{k}_win_rate", "") for k in range(self.latent_k)]
                            if sum(occ_o) < 1e-9 and all(w == "" for w in wr_o):
                                continue
                            occ_s = ",".join(f"{x:.2f}" for x in occ_o)
                            wr_s = ",".join("-" if w == "" else f"{float(w):.2f}" for w in wr_o)
                            opp_diag_bits.append(f"o{o}:z_occ=[{occ_s}] z_wr=[{wr_s}]")
                        opp_suffix = (
                            f" MI_z_o={mi_z_o:.4f} MI_z_phase={mi_z_p:.4f} "
                            f"MI_z_flag={mi_z_f:.4f} MI_z_outcome={mi_z_y:.4f} | "
                            + " ".join(opp_diag_bits)
                        )
                    print(
                        f"[PPO|diag] steps={self.global_step} "
                        f"ev={row['explained_variance']:.3f} "
                        f"v_loss={row['value_loss']:.3f} "
                        f"shape/out={row['reward_shaping_mean']:.3f}/{row['reward_outcome_mean']:.3f} "
                        f"qphi_grad={row.get('strategy_grad_norm', 0.0):.3f} "
                        f"zH={z_entropy:.3f}({z_entropy_frac:.2f}) "
                        f"z_wr_spread={z_wr_spread:.3f} "
                        f"z_aux_ret={float(row.get('strategy_aux_return_loss', row.get('strategy_q_loss', 0.0))):.3f} "
                        f"z_pi={float(row.get('strategy_policy_loss', 0.0)):.3f} "
                        f"z_ratio={float(row.get('strategy_ratio_std', 0.0)):.3f} "
                        f"z_occ=[{','.join(z_occ_parts)}] "
                        f"z_wr=[{','.join(z_wr_parts)}]"
                        f"{opp_suffix}"
                    )
                    if self.use_latent_strategy:
                        sw_cap = float(row.get("latent_switch_near_capture_frac", 0.0) or 0.0)
                        sw_kill = float(row.get("latent_switch_near_kill_frac", 0.0) or 0.0)
                        sw_ret = float(row.get("latent_switch_near_return_frac", 0.0) or 0.0)
                        div_role = float(row.get("latent_role_diversity", 0.0) or 0.0)
                        div_spread = float(row.get("latent_spread_diversity", 0.0) or 0.0)
                        div_pres = float(row.get("latent_pressure_diversity", 0.0) or 0.0)
                        div_adr = float(row.get("latent_adr_diversity", 0.0) or 0.0)
                        print(
                            f"      [Switch Near] cap={sw_cap:.3f} kill={sw_kill:.3f} ret={sw_ret:.3f} | "
                            f"div_role={div_role:.3f} div_spread={div_spread:.3f} div_pressure={div_pres:.3f} div_adr={div_adr:.3f}"
                        )
                if self.normalize_returns:
                    print(
                        "[PPO|return_norm] "
                        f"update={self._updates_completed} "
                        f"mean={stats.get('return_norm_mean', 0.0):.4f} "
                        f"std={stats.get('return_norm_std', 0.0):.4f} "
                        f"count={stats.get('return_norm_count', 0.0):.0f}"
                    )
                if bool(getattr(self.cfg, "verbose_training", False)):
                    latent_bits = ""
                    if self.use_latent_strategy:
                        latent_bits = (
                            f" z_entropy={stats.get('strategy_entropy', 0.0):.4f} "
                            f"z_persist={stats.get('strategy_persist_loss', 0.0):.4f}"
                        )
                    print(
                        "[PPO|custom] "
                        f"steps={self.global_step} policy_loss={stats['policy_loss']:.4f} "
                        f"value_loss={stats['value_loss']:.4f} approx_kl={stats['approx_kl']:.5f}"
                        f"{latent_bits}"
                    )
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
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "global_step": self.global_step,
                "updates_completed": self._updates_completed,
                "return_norm_mean": float(self._return_norm_mean),
                "return_norm_var": float(self._return_norm_var),
                "return_norm_count": float(self._return_norm_count),
                "strategy_return_mean": float(self._strategy_return_mean),
                "strategy_return_var": float(self._strategy_return_var),
                "strategy_return_count": float(self._strategy_return_count),
                "cfg": asdict(self.cfg),
                "last_stats": self.last_stats,
                "format": CUSTOM_PPO_LATENT_FORMAT if self.use_latent_strategy else CUSTOM_PPO_FORMAT,
                "actor_arch": CUSTOM_PPO_ACTOR_ARCH,
                "actor_cnn_feature_dim": int(self.model.actor_cnn_feature_dim),
                "global_state_dim": int(self.model.global_state_dim),
                "vec_schema_version": CUSTOM_PPO_VEC_SCHEMA_VERSION,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Restore a checkpoint produced by :meth:`save`."""
        payload = _torch_load_checkpoint(path, map_location=self.device)
        _assert_compatible_global_state_dim(payload, path)
        _load_model_state_dict_compat(self.model, payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.global_step = int(payload.get("global_step", 0))
        self._updates_completed = int(payload.get("updates_completed", 0))
        self._return_norm_mean = float(payload.get("return_norm_mean", 0.0))
        self._return_norm_var = float(payload.get("return_norm_var", 1.0))
        self._return_norm_count = float(payload.get("return_norm_count", 1e-4))
        self._strategy_return_mean = float(payload.get("strategy_return_mean", 0.0))
        self._strategy_return_var = float(payload.get("strategy_return_var", 1.0))
        self._strategy_return_count = float(payload.get("strategy_return_count", 1e-4))
        self.last_stats = dict(payload.get("last_stats", {}))
        self._last_obs = None
        self._last_global_state = None
        self.latent_state.current_z = None
        if self.use_latent_strategy:
            self.latent_state.reset()
