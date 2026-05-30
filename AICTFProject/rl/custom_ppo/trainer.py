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
        self.env = env
        self.cfg = cfg
        self.curriculum = curriculum
        self.device = torch.device(str(cfg.device))
        self.use_latent_strategy = bool(getattr(cfg, "use_latent_strategy", False))
        self.latent_k = int(getattr(cfg, "latent_k", 4)) if self.use_latent_strategy else 0
        self.latent_resample_every_n = max(0, int(getattr(cfg, "latent_resample_every_n", 0) or 0))
        self.fixed_latent_strategy = self.use_latent_strategy and bool(
            getattr(cfg, "fixed_latent_strategy", False)
        )
        self.latent_gae_reset_on_z_change = bool(
            getattr(cfg, "latent_gae_reset_on_z_change", True)
        ) and (self.use_latent_strategy and not self.fixed_latent_strategy)
        self.latent_bootstrap_z_deterministic = bool(getattr(cfg, "latent_bootstrap_z_deterministic", True))
        self.fixed_latent_strategy_id = (
            max(0, min(int(getattr(cfg, "fixed_latent_strategy_id", 0) or 0), self.latent_k - 1))
            if self.use_latent_strategy
            else 0
        )
        model_kwargs: dict[str, Any] = {
            "actor_cnn_feature_dim": int(getattr(cfg, "actor_cnn_feature_dim", 128)),
        }
        if self.use_latent_strategy:
            model_kwargs.update(
                {
                    "latent_k": self.latent_k,
                    "z_embed_dim": int(getattr(cfg, "latent_z_embed_dim", 16)),
                    "strategy_hidden_dim": int(getattr(cfg, "latent_strategy_hidden", 128)),
                    "critic_hidden_dim": int(getattr(cfg, "latent_vf_hidden", 128)),
                    # Canonical attribute access. Legacy CLI / cfg-dict keys
                    # (``latent_strategy_q_head``) are folded into the
                    # canonical name at the load boundary (CLI parsing for
                    # PPOConfig, inference.canonicalize_latent_strategy_cfg
                    # for checkpoint dicts), so the trainer reads one name.
                    "use_strategy_aux_return_head": bool(
                        getattr(cfg, "latent_strategy_aux_return_head", False)
                    ),
                    "use_episode_strategy_value_head": bool(getattr(cfg, "latent_episode_strategy_ppo", False)),
                    "strategy_tau": max(1e-3, float(getattr(cfg, "latent_strategy_tau", 1.0) or 1.0)),
                }
            )
        self.model = SharedActorCentralizedCritic(env.observation_space, env.action_space, **model_kwargs).to(self.device)
        apply_deterministic_sampling_generators(
            self.model, int(getattr(cfg, "seed", 0) or 0), device=self.device
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(learning_rate), eps=1e-5)
        self.base_learning_rate = float(learning_rate)
        self.clip_range = float(clip_range)
        self.ent_coef = float(ent_coef)
        self.vf_coef = max(0.0, float(getattr(cfg, "vf_coef", 1.0) or 0.0))
        self.n_epochs = int(n_epochs)
        self.batch_size = int(batch_size)
        self.value_clip_range = None if value_clip_range is None else float(value_clip_range)
        self.normalize_returns = bool(getattr(cfg, "normalize_returns", False))
        self._return_norm_mean = 0.0
        self._return_norm_var = 1.0
        self._return_norm_count = 1e-4
        self.latent_strategy_ppo_coef = max(0.0, float(getattr(cfg, "latent_strategy_ppo_coef", 0.1) or 0.0))
        self.latent_episode_strategy_ppo = (
            self.use_latent_strategy
            and not self.fixed_latent_strategy
            and bool(getattr(cfg, "latent_episode_strategy_ppo", False))
        )
        self.latent_episode_strategy_coef = max(
            0.0, float(getattr(cfg, "latent_episode_strategy_coef", 0.0) or 0.0)
        )
        self.latent_episode_strategy_clip_eps = max(
            1e-6, float(getattr(cfg, "latent_episode_strategy_clip_eps", 0.2) or 0.2)
        )
        self.latent_episode_strategy_value_coef = max(
            0.0, float(getattr(cfg, "latent_episode_strategy_value_coef", 0.5) or 0.0)
        )
        self.latent_episode_strategy_return_norm = bool(
            getattr(cfg, "latent_episode_strategy_return_norm", True)
        )
        # Canonical attribute access only — legacy ``latent_strategy_q_*`` keys
        # are folded at the config-load boundary, see ``rl.custom_ppo.inference
        # .canonicalize_latent_strategy_cfg`` and the CLI argparse handler.
        self.latent_strategy_aux_return_coef = max(
            0.0, float(getattr(cfg, "latent_strategy_aux_return_coef", 0.0) or 0.0)
        )
        self.latent_strategy_aux_return_head = (
            self.use_latent_strategy
            and bool(getattr(cfg, "latent_strategy_aux_return_head", False))
        )
        self.latent_strategy_aux_predict_phase_coef = max(0.0, float(getattr(cfg, "latent_strategy_aux_predict_phase_coef", 0.0) or 0.0))
        self.reward_shaping_coef_start = float(getattr(cfg, "reward_shaping_coef_start", 1.0) or 1.0)
        self.reward_shaping_coef_end = float(getattr(cfg, "reward_shaping_coef_end", self.reward_shaping_coef_start))
        self.reward_shaping_decay_steps = max(0, int(getattr(cfg, "reward_shaping_decay_steps", 0) or 0))
        env_cfg = getattr(env, "cfg", None)
        self.reward_dense_weight = max(0.0, float(getattr(env_cfg, "dense_weight", 1.0) or 0.0))
        self.reward_scale = max(1e-6, float(getattr(env_cfg, "reward_scale", 1.0) or 1.0))
        self.reward_clip = max(1e-6, float(getattr(env_cfg, "reward_clip", 1.0) or 1.0))
        self.reward_stalemate_penalty = float(getattr(env_cfg, "stalemate_penalty", 0.0) or 0.0)
        self.periodic_checkpoint_steps = max(0, int(getattr(cfg, "periodic_checkpoint_steps", 0) or 0))
        self._next_periodic_checkpoint_step = (
            self.periodic_checkpoint_steps if self.periodic_checkpoint_steps > 0 else 0
        )
        self._strategy_return_mean = 0.0
        self._strategy_return_var = 1.0
        self._strategy_return_count = 1e-4
        self.global_step = 0
        self.last_stats: dict[str, float] = {}
        self.run_id = str(getattr(cfg, "run_id", "") or "")
        self.run_pid = int(getattr(cfg, "run_pid", os.getpid()) or os.getpid())
        self._updates_completed = 0
        self._ep_wins = 0
        self._ep_losses = 0
        self._ep_draws = 0
        self._episodes_completed = 0
        self._rollout_episode_records: list[dict[str, Any]] = []
        self._recent_episode_successes = deque(maxlen=200)
        self.metrics_csv_path = str(getattr(cfg, "metrics_csv_path", "") or "")
        self.episode_csv_path = str(getattr(cfg, "episode_csv_path", "") or "")
        self.strategy_experience_csv_path = str(getattr(cfg, "strategy_experience_csv_path", "") or "")
        self.telemetry = TrainingTelemetry(self)
        if self.use_latent_strategy:
            self.temporal_tracker = TemporalStateTracker(
                num_envs=int(env.num_envs),
                state_dim=GLOBAL_STATE_DIM,
                device=self.device,
            )
        else:
            self.temporal_tracker = None
        self.latent_resample_on_flag = (
            bool(getattr(cfg, "latent_resample_on_flag", False))
            and self.use_latent_strategy
            and not self.fixed_latent_strategy
        )
        self.latent_kl_consecutive = (
            max(0.0, float(getattr(cfg, "latent_kl_consecutive", 0.0) or 0.0))
            if self.use_latent_strategy and not self.fixed_latent_strategy
            else 0.0
        )
        self.latent_state = LatentStrategyState(self)
        self.rollout_collector = RolloutCollector(self)
        self.updater = PPOUpdater(self)
        self._sb3_rollout_pbar = None
        self._last_obs = None
        self._last_global_state = None
        self._last_context_state = None
        self._decentralized_actor_contract_logged = False
        mode_s = str(getattr(cfg, "mode", "") or "").strip().upper()
        self._opponent_randomize_training = (
            (mode_s == "OPPONENT_POOL" or bool(getattr(cfg, "opponent_randomize", False)))
            and curriculum is None
        )
        self._opponent_pool_tags = (
            [str(x).strip().upper() for x in getattr(cfg, "opponent_pool", ())] if self._opponent_randomize_training else []
        )
        self._rng_opponent = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 901)
        if self._opponent_randomize_training:
            if not self._opponent_pool_tags:
                raise ValueError(
                    "Opponent pool training (mode=OPPONENT_POOL or opponent_randomize) requires a non-empty "
                    "opponent_pool (e.g. OP1–OP3, OP5–OP7; OP4 optional with --allow-op4-in-training-pool)."
                )
            self.env._before_reset_indices_hook = partial(_hook_sample_training_opponent_before_reset, self)

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
