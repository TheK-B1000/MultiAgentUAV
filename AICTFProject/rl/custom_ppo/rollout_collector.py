"""Rollout collection for :class:`CustomPPOTrainer`.

This module owns the step-the-env loop and everything that happens inside it
that is *not* z-state machinery (PR-5), telemetry writing (PR-4), or the PPO
update (PR-7): the per-step act / step / reward composition / buffer fill,
the next-value bootstrap, the GAE compute, and the option-return compute.

Why this module exists
----------------------
Before extraction, ``CustomPPOTrainer.collect_rollout`` was ~290 lines and
co-located reset, per-step sampling, env stepping, reward composition,
telemetry triggers, buffer filling, GAE, option returns, and return-norm
update. Splitting it out lets the trainer become the conductor and lets the
collector be replaced or wrapped (e.g. for evaluation) without touching the
update or telemetry paths.

Internal structure
------------------
:meth:`RolloutCollector.collect` is intentionally a tiny conductor that calls,
in order:

1. :meth:`_initial_step_state` — reset / persist obs + global / context state.
2. :meth:`make_buffer` — register the field schema for one rollout.
3. :meth:`_step_once` for ``cfg.n_steps`` iterations — runs one env step,
   composes rewards, builds and records a ``StepFrame`` via
   :class:`~rl.custom_ppo.rollout_step_recorder.RolloutStepRecorder` (PR-9),
   and advances obs / global / context state.
4. :meth:`_finalize_buffer` — next-values alignment, GAE, option-returns,
   and return-norm stats update.

Static collaborators (``model`` / ``env`` / ``device`` / ``cfg`` /
``hparams`` / ``latent_state`` / ``telemetry`` / ``episode_stats`` /
``temporal_tracker`` / ``reward_shaping_coef``) are injected explicitly
into the constructor. A ``runtime`` back-reference to the trainer is
kept for the mutable cross-rollout cursor state that hasn't been
extracted into its own owner yet: ``global_step``, ``_last_obs`` /
``_last_global_state`` / ``_last_context_state``, ``_sb3_rollout_pbar``,
and ``_global_state_probe_rows``. It is also passed through to a handful
of legacy helpers (``_denormalize_values``, ``_update_return_norm_stats``,
``_update_curriculum_after_episode``, ``log_decentralized_actor_contract_once``)
that still take a trainer-shaped first arg.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from rl.behavior_telemetry import (
    N_TELEMETRY,
    bucket_ids_from_telemetry,
    compute_behavior_telemetry_batch,
)
from rl.global_state import (
    GLOBAL_STATE_DIM,
    GLOBAL_STATE_FLAG_TERRITORY_SLICE,
)
from rl.custom_ppo.latent.router_sampling import (
    build_current_plus_delta_router_context,
    router_current_plus_delta_enabled,
)
from rl.ppo_core import (
    TensorDictRolloutBuffer,
    align_next_values_to_rollout_actions,
)
from rl.csia import CSIARewardModel
from rl.custom_ppo.csv_writers import _opponent_id_int_from_info
from rl.custom_ppo.curriculum_runtime import _update_curriculum_after_episode
from rl.custom_ppo.option_returns import compute_option_returns
from rl.custom_ppo.return_normalization import (
    _denormalize_values,
    _update_return_norm_stats,
)
from rl.custom_ppo.reward_composition import _compose_training_reward_components
from rl.custom_ppo.rollout_step_recorder import RolloutStepRecorder, StepFrame
from rl.custom_ppo.trainer_audit import log_decentralized_actor_contract_once
from rl.custom_ppo.trainer_config import TrainerHyperparams

if TYPE_CHECKING:
    from rl.custom_ppo.latent_strategy_state import LatentStrategyState
    from rl.custom_ppo.training_telemetry import TrainingTelemetry
    from rl.custom_ppo.trainer import CustomPPOTrainer


class RolloutCollector:
    """Owns ``collect_rollout`` and its env-stepping helpers.

    Constructed once by the trainer and held as ``self.rollout_collector``.
    See the module docstring for the full dependency surface.
    """

    def __init__(
        self,
        *,
        model: Any,
        env: Any,
        device: Any,
        cfg: Any,
        hparams: TrainerHyperparams,
        latent_state: "LatentStrategyState",
        telemetry: "TrainingTelemetry",
        episode_stats: Any,
        temporal_tracker: Any,
        reward_shaping_coef: Any,
        runtime: "CustomPPOTrainer",
    ) -> None:
        self.model = model
        self.env = env
        self.device = device
        self.cfg = cfg
        self.hparams = hparams
        self.latent_state = latent_state
        self.telemetry = telemetry
        self.episode_stats = episode_stats
        self.temporal_tracker = temporal_tracker
        self._reward_shaping_coef_fn = reward_shaping_coef
        self.runtime = runtime
        self.step_recorder = RolloutStepRecorder(runtime)
        self.csia_reward_model = CSIARewardModel.from_config(cfg)

    # ------------------------------------------------------------------
    # Step-level helpers (also called from inside ``next_values``).
    # ------------------------------------------------------------------

    @staticmethod
    def flag_territory_features_changed(
        pre: torch.Tensor, post: torch.Tensor, *, eps: float = 1e-4
    ) -> torch.Tensor:
        """(B, 4) pre/post flag-sector slice; return (B,) bool: min distances or capture flags changed."""
        d0 = (pre[:, 0:2] - post[:, 0:2]).abs() > float(eps)
        ch_float = d0.any(dim=-1)
        ch_cap = (pre[:, 2:4] - post[:, 2:4]).abs() > 0.5
        ch_capt = ch_cap.any(dim=-1)
        return ch_float | ch_capt

    def tensor_obs(self, obs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        device = self.device
        return {
            "grid": torch.as_tensor(obs["grid"], dtype=torch.float32, device=device),
            "vec": torch.as_tensor(obs["vec"], dtype=torch.float32, device=device),
            "agent_mask": torch.as_tensor(obs["agent_mask"], dtype=torch.float32, device=device),
            "mask": torch.as_tensor(obs["mask"], dtype=torch.float32, device=device),
        }

    def on_sb3_rollout_env_step(self) -> None:
        p = self.runtime._sb3_rollout_pbar
        if p is None:
            return
        nenv = int(self.env.num_envs)
        try:
            rest = int(p.total) - int(p.n)  # type: ignore[attr-defined]
        except Exception:
            p.update(nenv)  # type: ignore[call-arg]
            return
        p.update(int(min(nenv, max(0, rest))))  # type: ignore[call-arg]

    def obs_rows_from_next(
        self,
        next_obs: Dict[str, np.ndarray],
        infos: list[dict],
    ) -> Dict[str, np.ndarray]:
        rows: dict[str, list[np.ndarray]] = {key: [] for key in ("grid", "vec", "agent_mask", "mask")}
        for env_i, info in enumerate(infos):
            use_terminal = bool(info.get("truncated", False)) and isinstance(
                info.get("terminal_observation"), dict
            )
            terminal_obs = info.get("terminal_observation") if use_terminal else {}
            for key in rows:
                source = terminal_obs.get(key, next_obs[key][env_i]) if isinstance(terminal_obs, dict) else next_obs[key][env_i]
                rows[key].append(np.asarray(source, dtype=np.float32))
        return {key: np.stack(values, axis=0) for key, values in rows.items()}

    def make_buffer(self, obs: Dict[str, np.ndarray]) -> TensorDictRolloutBuffer:
        cfg = self.cfg
        hparams = self.hparams
        n_steps = int(cfg.n_steps)
        n_envs = int(self.env.num_envs)
        buffer = TensorDictRolloutBuffer(n_steps, n_envs, device=self.device)
        buffer.register_field("obs_grid", tuple(obs["grid"].shape[1:]))
        buffer.register_field("obs_vec", tuple(obs["vec"].shape[1:]))
        buffer.register_field("obs_agent_mask", tuple(obs["agent_mask"].shape[1:]))
        buffer.register_field("obs_mask", tuple(obs["mask"].shape[1:]))
        buffer.register_field("global_state", (self.model.global_state_dim,))
        buffer.register_field("actions", (len(getattr(self.env.action_space, "nvec", [])),), dtype=torch.long)
        buffer.register_field("log_probs")
        buffer.register_field("values")
        buffer.register_field("values_norm")
        buffer.register_field("next_values")
        buffer.register_field("rewards")
        buffer.register_field("reward_terminal")
        buffer.register_field("reward_offense")
        buffer.register_field("reward_pbrs")
        buffer.register_field("reward_team")
        buffer.register_field("reward_sparse")
        buffer.register_field("reward_sparse_points")
        buffer.register_field("reward_failure")
        buffer.register_field("reward_behavior_contrast")
        buffer.register_field("reward_csia")
        buffer.register_field("reward_total")
        buffer.register_field("terminated", dtype=torch.bool)
        buffer.register_field("truncated", dtype=torch.bool)
        buffer.register_field("opponent_id", dtype=torch.long)
        if hparams.use_latent_strategy:
            buffer.register_field("z", dtype=torch.long)
            buffer.register_field("prev_z", dtype=torch.long)
            buffer.register_field("z_log_probs")
            buffer.register_field("z_logits", (hparams.latent_k,))
            buffer.register_field("z_resampled", dtype=torch.bool)
            buffer.register_field("z_resampled_actual", dtype=torch.bool)
            buffer.register_field("z_forced", dtype=torch.bool)
            buffer.register_field("z_persist_mask", dtype=torch.bool)
            buffer.register_field("phase_id", dtype=torch.long)
            buffer.register_field("outcome_id", dtype=torch.long)
            buffer.register_field("behavior_telemetry", (N_TELEMETRY,))
            buffer.register_field("spread_bucket_id", dtype=torch.long)
            buffer.register_field("role_bucket_id", dtype=torch.long)
            buffer.register_field("pressure_bucket_id", dtype=torch.long)
            buffer.register_field("attack_defense_ratio_bucket_id", dtype=torch.long)
            buffer.register_field("blue_ahead", dtype=torch.float32)
            if router_current_plus_delta_enabled(cfg):
                router_dim = int(getattr(cfg, "router_context_dimension", 0) or 0)
                if router_dim <= 0:
                    raise ValueError("router_context_mode=current_plus_delta requires router_context_dimension > 0")
                buffer.register_field("router_context", (router_dim,))
                buffer.register_field("prev_router_context", (router_dim,))
                buffer.register_field("persistence_valid", dtype=torch.bool)
                buffer.register_field("episode_id", dtype=torch.long)
                buffer.register_field("opportunity_index", dtype=torch.long)
                buffer.register_field("env_id", dtype=torch.long)
            if hparams.latent_kl_consecutive > 0.0:
                buffer.register_field("z_logits_prev", (hparams.latent_k,))
                buffer.register_field("z_kl_prev_valid")
            if bool(getattr(self.model, "use_recurrent_selector", False)):
                hidden_dim = int(getattr(self.model, "recurrent_selector_hidden_dim", 0) or 0)
                if hidden_dim > 0:
                    buffer.register_field("selector_hidden", (hidden_dim,))
        if bool(getattr(self.model, "communication_enabled", False)):
            n_agents = int(self.model.n_agents)
            buffer.register_field("message_symbols", (n_agents,), dtype=torch.long)
            buffer.register_field("message_log_probs")
            buffer.register_field("message_entropy")
            buffer.register_field("message_boundary_mask", dtype=torch.bool)
        return buffer

    def z_for_bootstrap(
        self,
        next_context_gs_t: torch.Tensor,
        z_t: torch.Tensor,
        dones: np.ndarray,
    ) -> torch.Tensor:
        """Strategy index for V(s', z') bootstrapping to match the start of the *next* decision."""
        hparams = self.hparams
        if not hparams.use_latent_strategy:
            raise RuntimeError("z_for_bootstrap requires latent strategy mode.")
        if hparams.fixed_latent_strategy:
            return torch.full_like(z_t, int(hparams.fixed_latent_strategy_id), dtype=torch.long)
        device = self.device
        done_t = torch.as_tensor(dones, dtype=torch.bool, device=device)
        age_next = self.latent_state.strategy_age + 1
        age_next = torch.where(done_t, torch.zeros_like(age_next), age_next)
        needs_next = self.latent_state.needs_strategy_sample.clone()
        if bool(done_t.any().item()):
            needs_next = needs_next.clone()
            needs_next[done_t] = bool(not hparams.fixed_latent_strategy)
        resample_next = needs_next.clone()
        if hparams.latent_resample_every_n > 0:
            resample_next = resample_next | (age_next >= int(hparams.latent_resample_every_n))
        resample_next = resample_next & (~done_t)
        z_next = z_t.long().clone()
        if bool(resample_next.any().item()):
            idx = torch.where(resample_next)[0]
            gs_sub = next_context_gs_t.index_select(0, idx)
            if router_current_plus_delta_enabled(self.cfg):
                previous = self.latent_state.previous_opportunity_features.index_select(0, idx).to(
                    device=gs_sub.device, dtype=gs_sub.dtype
                )
                gs_sub = build_current_plus_delta_router_context(gs_sub, previous)
            hidden_sub = None
            if bool(getattr(self.model, "use_recurrent_selector", False)):
                selector_hidden = getattr(self.latent_state, "selector_hidden", None)
                if selector_hidden is None:
                    raise RuntimeError("recurrent selector bootstrap requires latent_state.selector_hidden")
                hidden_sub = selector_hidden.index_select(0, idx)
            sampled_z, _, _, _, _ = self.model.sample_strategy(
                gs_sub,
                deterministic=bool(hparams.latent_bootstrap_z_deterministic),
                selector_hidden=hidden_sub,
            )
            z_next[idx] = sampled_z.long()
        return z_next

    def next_values(
        self,
        infos: list[dict],
        next_global_state: np.ndarray,
        next_obs: Optional[Dict[str, np.ndarray]] = None,
        prev_z: Optional[torch.Tensor] = None,
        dones: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        runtime = self.runtime
        device = self.device
        rows = []
        for env_i, info in enumerate(infos):
            if bool(info.get("terminated", False)):
                rows.append(np.asarray(next_global_state[env_i], dtype=np.float32))
            elif bool(info.get("truncated", False)):
                terminal_obs = info.get("terminal_observation") or {}
                rows.append(np.asarray(terminal_obs.get("global_state", next_global_state[env_i]), dtype=np.float32))
            else:
                rows.append(np.asarray(next_global_state[env_i], dtype=np.float32))
        gs = torch.as_tensor(np.stack(rows, axis=0), dtype=torch.float32, device=device)
        with torch.no_grad():
            if not self.hparams.use_latent_strategy:
                return _denormalize_values(runtime, self.model.values(gs))

            done_t = torch.as_tensor(dones, dtype=torch.bool, device=device) if dones is not None else None
            next_context_gs_t = self.temporal_tracker.update(gs, dones=done_t)
            runtime._last_context_state = next_context_gs_t

            if next_obs is None or prev_z is None:
                raise ValueError("latent next value bootstrap requires next_obs and prev_z.")
            obs_rows = self.obs_rows_from_next(next_obs, infos)
            comm = getattr(runtime, "comm_runtime", None)
            if comm is not None and comm.enabled:
                comm.bind_env_core(self.env.core)
                obs_rows = comm.prepare_obs(
                    obs_rows,
                    expected_grid_channels=int(self.model.grid_shape[0]),
                )
            next_obs_t = self.tensor_obs(obs_rows)
            if dones is None:
                raise ValueError("latent next value bootstrap requires dones for z lookahead.")
            next_z = self.z_for_bootstrap(
                next_context_gs_t,
                prev_z.long().reshape(-1),
                dones,
            )
            _, next_values, _, _ = self.model.act(
                next_obs_t,
                next_context_gs_t,
                deterministic=True,
                z_idx=next_z,
            )
            next_values = _denormalize_values(runtime, next_values)
            terminated = torch.as_tensor(
                [bool(info.get("terminated", False)) for info in infos],
                dtype=torch.bool,
                device=device,
            )
            return torch.where(terminated, torch.zeros_like(next_values), next_values)

    # ------------------------------------------------------------------
    # Episode boundary handling (per-env, per-step inside ``collect``).
    # ------------------------------------------------------------------

    def on_episode_done(
        self,
        info: dict[str, Any],
        *,
        timestep: Optional[int] = None,
        rollout_step: Optional[int] = None,
        latent_z: Optional[int] = None,
        env_index: Optional[int] = None,
    ) -> None:
        runtime = self.runtime
        er = info.get("episode_result")
        if isinstance(er, dict):
            bs = int(er.get("blue_score", 0))
            rs = int(er.get("red_score", 0))
        else:
            bs = int(info.get("blue_score", 0))
            rs = int(info.get("red_score", 0))
        self.episode_stats.record(
            blue_score=bs,
            red_score=rs,
            latent_z=latent_z,
            opponent_id=_opponent_id_int_from_info(self.cfg, info),
        )
        self.telemetry.write_episode_metrics(
            info,
            blue_score=bs,
            red_score=rs,
            timestep=int(timestep or runtime.global_step),
            rollout_step=rollout_step,
            latent_z=latent_z,
        )
        _update_curriculum_after_episode(runtime, info=info, blue_score=bs, red_score=rs, env_index=env_index)
        every = int(getattr(self.cfg, "episode_log_every", 0) or 0)
        if every > 0 and self.episode_stats.episodes_completed % every == 0:
            self.telemetry.print_episode_progress(info)

    # ------------------------------------------------------------------
    # Main entry point.
    # ------------------------------------------------------------------

    def collect(self) -> TensorDictRolloutBuffer:
        """Collect one rollout and compute advantages/returns.

        Conductor only: every named stage below corresponds to a private
        helper. Read top-to-bottom for the per-rollout flow; read each
        helper for the per-step / per-stage detail.
        """
        runtime = self.runtime
        log_decentralized_actor_contract_once(runtime)
        self.episode_stats.reset_rollout()
        self.latent_state.rollout_strategy_episode_records = []
        self.latent_state.reset_behavior_contrast_rollout_stats()
        self.latent_state.reset_event_refresh_rollout_stats()
        self.latent_state.reset_sparse_tactical_refresh_rollout_stats()
        obs, global_state, context_state = self._initial_step_state()
        comm = getattr(self.runtime, "comm_runtime", None)
        if comm is not None and comm.enabled:
            comm.bind_env_core(self.env.core)
            obs = comm.prepare_obs(
                obs,
                expected_grid_channels=int(self.model.grid_shape[0]),
            )
        buffer = self.make_buffer(obs)
        for step_idx in range(int(self.cfg.n_steps)):
            obs, global_state, context_state = self._step_once(
                buffer,
                step_idx=step_idx,
                obs=obs,
                global_state=global_state,
                context_state=context_state,
            )
        self._finalize_buffer(buffer)
        runtime._last_obs = obs
        runtime._last_global_state = global_state
        return buffer

    # ------------------------------------------------------------------
    # Rollout setup / teardown.
    # ------------------------------------------------------------------

    def _initial_step_state(
        self,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, torch.Tensor]:
        """Return ``(obs, global_state, context_state)`` for the first step.

        Either reuses persisted state from the previous rollout or, on the
        very first ``collect()`` call, resets the env (and the latent
        temporal tracker / latent state) and seeds the context.
        """
        runtime = self.runtime
        device = self.device
        use_latent = self.hparams.use_latent_strategy
        if runtime._last_obs is None or runtime._last_global_state is None:
            obs = self.env.reset()
            global_state = self.env.state().astype(np.float32)
            self.latent_state.reset()
            if getattr(runtime, "comm_runtime", None) is not None and runtime.comm_runtime.enabled:
                runtime.comm_runtime.reset(
                    batch_size=int(self.env.num_envs),
                    num_agents=int(self.model.n_agents),
                )
                runtime.comm_runtime.bind_env_core(self.env.core)
            if use_latent:
                gs_t = torch.as_tensor(global_state, dtype=torch.float32, device=device)
                context_state = self.temporal_tracker.update(gs_t)
            else:
                context_state = torch.as_tensor(global_state, dtype=torch.float32, device=device)
            return obs, global_state, context_state
        obs = runtime._last_obs
        global_state = runtime._last_global_state
        if use_latent:
            context_state = runtime._last_context_state
        else:
            context_state = torch.as_tensor(global_state, dtype=torch.float32, device=device)
        comm = getattr(runtime, "comm_runtime", None)
        if comm is not None and comm.enabled:
            comm.bind_env_core(self.env.core)
            obs = comm.prepare_obs(
                obs,
                expected_grid_channels=int(self.model.grid_shape[0]),
            )
        return obs, global_state, context_state

    def _finalize_buffer(self, buffer: TensorDictRolloutBuffer) -> None:
        """Post-loop: align ``next_values``, run GAE, option-returns, return-norm."""
        cfg = self.cfg
        hparams = self.hparams
        buffer.fields["next_values"][: int(buffer.pos)].copy_(
            align_next_values_to_rollout_actions(
                buffer.fields["values"][: int(buffer.pos)],
                buffer.fields["next_values"][: int(buffer.pos)],
                buffer.fields["terminated"][: int(buffer.pos)].bool(),
                buffer.fields["truncated"][: int(buffer.pos)].bool(),
            )
        )
        gae_kw: Dict[str, Any] = dict(
            gamma=float(cfg.gamma),
            gae_lambda=float(cfg.gae_lambda),
        )
        if hparams.latent_gae_reset_on_z_change:
            gae_kw["latent_z_field"] = "z"
            gae_kw["reset_gae_on_z_change"] = True
        buffer.compute_returns_and_advantages(**gae_kw)
        if hparams.use_latent_strategy:
            with torch.no_grad():
                option_returns, option_advantages = compute_option_returns(
                    rewards=buffer.fields["rewards"],
                    values=buffer.fields["values"],
                    next_values=buffer.fields["next_values"],
                    terminated=buffer.fields["terminated"],
                    truncated=buffer.fields["truncated"],
                    z_resampled=buffer.fields["z_resampled"],
                    gamma=float(cfg.gamma),
                )
                if "option_returns" not in buffer.fields:
                    buffer.register_field("option_returns")
                if "option_advantages" not in buffer.fields:
                    buffer.register_field("option_advantages")
                buffer.fields["option_returns"].copy_(option_returns)
                buffer.fields["option_advantages"].copy_(option_advantages)
        _update_return_norm_stats(self.runtime, buffer.fields["returns"][: int(buffer.pos)])

    # ------------------------------------------------------------------
    # Per-step pipeline.
    # ------------------------------------------------------------------

    def _step_once(
        self,
        buffer: TensorDictRolloutBuffer,
        *,
        step_idx: int,
        obs: Dict[str, np.ndarray],
        global_state: np.ndarray,
        context_state: torch.Tensor,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, torch.Tensor]:
        """Run one env step and write one row to ``buffer``.

        Sequence of named stages, each delegated to a small private helper:
        sample → (latent telemetry) → env step → episode-done bookkeeping →
        next-value bootstrap → reward composition → record (PR-9) →
        diagnostics → flag-resample trigger → state advance → end-of-step
        latent housekeeping → E3 telemetry append.
        """
        runtime = self.runtime
        env = self.env
        comm = getattr(runtime, "comm_runtime", None)
        decision_global_state_np = np.asarray(global_state, dtype=np.float32)
        if comm is not None and comm.enabled:
            comm.bind_env_core(env.core)
            obs = comm.prepare_obs(
                obs,
                expected_grid_channels=int(self.model.grid_shape[0]),
            )
        obs_t = self.tensor_obs(obs)
        comm_boundary = (
            comm.current_boundary_mask()
            if comm is not None and comm.enabled
            else None
        )
        with torch.no_grad():
            z_t, prev_z_t, strategy_aux = self.latent_state.strategy_for_step(context_state)
            message_aux = None
            if comm is not None and comm.enabled:
                assert comm_boundary is not None
                if bool(comm_boundary.any()):
                    message_aux = self.model._sample_messages(
                        obs_t,
                        z_idx=z_t,
                        comm_boundary_mask=comm_boundary,
                    )
                    comm.submit_sampled_messages(
                        symbols=message_aux["message_symbols"],
                        boundary_mask=message_aux["message_boundary_mask"],
                        env_core=env.core,
                    )
                else:
                    message_aux = comm.non_boundary_message_aux(
                        boundary_mask=comm_boundary,
                        num_agents=int(self.model.n_agents),
                    )
            actions_t, values_norm_t, log_probs_t, _ = self.model.act(
                obs_t, context_state, z_idx=z_t
            )
            values_t = _denormalize_values(runtime, values_norm_t)
        actions_np = actions_t.detach().cpu().numpy().astype(np.int64)

        beh_t, sb, rb, pb, adb, blue_ahead_t = self._pre_step_latent_telemetry(actions_t)

        env.step_async(actions_np)
        next_obs, _rewards, dones, infos = env.step_wait()
        if comm is not None and comm.enabled:
            comm.advance_after_step(env.core)
            if bool(np.asarray(dones).any()):
                comm.reset_env_indices(np.asarray(dones))
        step_after = runtime.global_step + int(env.num_envs)

        z_np = z_t.detach().cpu().numpy() if z_t is not None else None
        self._handle_episode_dones(
            dones=dones,
            infos=infos,
            z_np=z_np,
            step_idx=step_idx,
            step_after=step_after,
        )

        next_global_state = env.state().astype(np.float32)
        next_values_t = self.next_values(
            infos, next_global_state, next_obs=next_obs, prev_z=z_t, dones=dones
        )

        terminated = np.asarray(
            [bool(info.get("terminated", bool(done))) for info, done in zip(infos, dones)]
        )
        truncated = np.asarray([bool(info.get("truncated", False)) for info in infos])
        reward_component = self._compose_step_rewards(infos)
        if self.hparams.use_latent_strategy and beh_t is not None and z_t is not None:
            contrast_bonus = self.latent_state.record_behavior_contrast_step(
                behavior_telemetry=beh_t,
                z_idx=z_t,
                dones=dones,
            )
            reward_component["reward_behavior_contrast"] = contrast_bonus
            reward_component["reward_total"] = reward_component["reward_total"] + contrast_bonus
        else:
            reward_component["reward_behavior_contrast"] = torch.zeros(
                (int(env.num_envs),), dtype=torch.float32, device=self.device
            )

        opp_row = torch.as_tensor(
            [_opponent_id_int_from_info(self.cfg, dict(info)) for info in infos],
            dtype=torch.long,
            device=self.device,
        )
        reward_component["reward_csia"] = torch.zeros(
            (int(env.num_envs),), dtype=torch.float32, device=self.device
        )
        if self.hparams.use_latent_strategy and z_t is not None:
            csia_bonus = self.csia_reward_model.bonus(
                opp_row,
                z_t,
                device=self.device,
                update=int(runtime._updates_completed),
            )
            reward_component["reward_csia"] = csia_bonus
            reward_component["reward_total"] = reward_component["reward_total"] + csia_bonus

        if self.hparams.use_latent_strategy:
            self._update_latent_episode_returns(
                reward_component=reward_component, dones=dones, infos=infos
            )

        frame = StepFrame(
            obs=obs,
            context_state=context_state,
            decision_global_state_np=decision_global_state_np,
            actions_t=actions_t,
            log_probs_t=log_probs_t,
            values_t=values_t,
            values_norm_t=values_norm_t,
            next_values_t=next_values_t,
            reward_component=reward_component,
            terminated=terminated,
            truncated=truncated,
            opp_row=opp_row,
            infos=infos,
            strategy_aux=strategy_aux if self.hparams.use_latent_strategy else None,
            behavior_telemetry=beh_t,
            spread_bucket=sb,
            role_bucket=rb,
            pressure_bucket=pb,
            attack_defense_ratio_bucket=adb,
            blue_ahead=blue_ahead_t,
            message_aux=message_aux,
        )
        self.step_recorder.record(buffer, frame)

        self._append_global_state_probe_rows(decision_global_state_np, infos)
        if self.hparams.latent_resample_on_flag:
            self._apply_flag_resample_trigger(context_state, next_global_state)

        next_context_state = self._advance_context(next_global_state)
        runtime.global_step += int(env.num_envs)
        self.on_sb3_rollout_env_step()

        self._finalize_step_latent_state(strategy_aux, dones)
        self._append_e3_step_telemetry(
            step_idx=step_idx,
            decision_global_state_np=decision_global_state_np,
            z_t=z_t,
            prev_z_t=prev_z_t,
            strategy_aux=strategy_aux,
            infos=infos,
            beh_t=beh_t,
            sb=sb,
            rb=rb,
            pb=pb,
            adb=adb,
            blue_ahead_t=blue_ahead_t,
            context_state=context_state,
        )
        return next_obs, next_global_state, next_context_state

    # ------------------------------------------------------------------
    # Per-step helpers, in roughly the order ``_step_once`` calls them.
    # ------------------------------------------------------------------

    def _pre_step_latent_telemetry(
        self, actions_t: torch.Tensor
    ) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        """Compute ``(behavior_telemetry, sb, rb, pb, adb, blue_ahead)`` or all-``None``.

        Latent-only: returns six ``None`` when ``use_latent_strategy`` is
        ``False`` so call sites don't need to branch. The values are read
        from the *current* env core state (i.e. before ``step_async``).
        """
        if not self.hparams.use_latent_strategy:
            return None, None, None, None, None, None
        env_core = self.env.core
        beh_t = compute_behavior_telemetry_batch(env_core, actions_t)
        sb, rb, pb, adb = bucket_ids_from_telemetry(beh_t, actions_t, env_core)
        blue_ahead_t = (env_core.blue_score > env_core.red_score).to(
            dtype=torch.float32, device=self.device
        )
        return beh_t, sb, rb, pb, adb, blue_ahead_t

    def _handle_episode_dones(
        self,
        *,
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
        z_np: Optional[np.ndarray],
        step_idx: int,
        step_after: int,
    ) -> None:
        """Forward each finished episode to :meth:`on_episode_done`."""
        for env_i, (done_i, info) in enumerate(zip(dones, infos)):
            if not bool(done_i):
                continue
            latent_z = int(z_np[env_i]) if z_np is not None else None
            self.on_episode_done(
                dict(info),
                timestep=step_after,
                rollout_step=step_idx + 1,
                latent_z=latent_z,
                env_index=env_i,
            )

    def _compose_step_rewards(
        self, infos: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """Pull reward components from ``infos`` and run the training composer."""
        device = self.device
        hparams = self.hparams
        reward_component = {
            key: torch.as_tensor(
                [float(info.get(key, 0.0) or 0.0) for info in infos],
                dtype=torch.float32,
                device=device,
            )
            for key in (
                "reward_terminal",
                "reward_offense",
                "reward_pbrs",
                "reward_team",
                "reward_sparse",
                "reward_sparse_points",
                "reward_failure",
                "reward_total",
            )
        }
        shaping_coef = float(self._reward_shaping_coef_fn())
        stalemate = torch.as_tensor(
            [bool(info.get("stalemate_truncated", False)) for info in infos],
            dtype=torch.bool,
            device=device,
        )
        return _compose_training_reward_components(
            reward_component,
            dense_weight=hparams.reward_dense_weight,
            reward_scale=hparams.reward_scale,
            reward_clip=hparams.reward_clip,
            shaping_coef=shaping_coef,
            stalemate=stalemate,
            stalemate_penalty=hparams.reward_stalemate_penalty,
        )

    def _update_latent_episode_returns(
        self,
        *,
        reward_component: Dict[str, torch.Tensor],
        dones: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """Maintain ``episode_return_accum`` and flush on env-level dones.

        Used by the episode-strategy PPO path to score whole-episode returns
        per latent ``z``. Caller already gated this on ``use_latent_strategy``.
        """
        latent_state = self.latent_state
        done_t = torch.as_tensor(dones, dtype=torch.bool, device=self.device)
        reward_total = reward_component["reward_total"].detach()
        latent_state.episode_return_accum = (
            latent_state.episode_return_accum + reward_total
        )
        # v3i19 arc-credit reward accumulator. Independent of episode-credit:
        # arc accumulates over the open z-arc only and is finalized at z change
        # OR at episode end (handled below). No-op when arc credit is disabled.
        latent_state.arc_accumulate_step(reward_total)
        latent_state.macro_accumulate_step(reward_total)
        if not bool(done_t.any().item()):
            return
        episode_strategy_ppo_on = bool(self.hparams.latent_episode_strategy_ppo)
        from rl.custom_ppo.v6i1_phase_runtime import (
            is_v6i1_staged_trainer,
            resolve_v6i1_episode_forced_frac,
        )
        from rl.custom_ppo.schedules import resolve_latent_forced_z_frac

        if is_v6i1_staged_trainer(self.runtime):
            forced_z_logging_on = resolve_v6i1_episode_forced_frac(self.runtime) > 0.0
        else:
            forced_z_logging_on = (
                resolve_latent_forced_z_frac(
                    self.cfg,
                    global_step=int(getattr(self.runtime, "global_step", 0) or 0),
                )
                > 0.0
            )
        # v3i3 finalization is gated on either the preference loss or the
        # per-refresh CSV log being enabled. Independent of episode-credit so
        # the proof-layer log works even without ``latent_episode_strategy_ppo``.
        v3i3_finalize_on = bool(
            self.hparams.latent_v3i3_event_preference_enabled
            or self.hparams.latent_v3i3_refresh_log_enabled
        )
        if episode_strategy_ppo_on or v3i3_finalize_on or forced_z_logging_on:
            for env_i, done_i in enumerate(dones):
                if not bool(done_i):
                    continue
                info_dict = dict(infos[env_i])
                episode_return = float(
                    latent_state.episode_return_accum[env_i]
                    .detach()
                    .cpu()
                    .item()
                )
                if v3i3_finalize_on:
                    latent_state.finalize_v3i3_refresh_records(
                        env_i, info_dict, episode_return=episode_return
                    )
                if episode_strategy_ppo_on or forced_z_logging_on:
                    latent_state.record_episode_strategy_outcome(
                        env_i, info_dict, episode_return=episode_return
                    )
        latent_state.episode_return_accum[done_t] = 0.0
        latent_state.episode_strategy_has_start[done_t] = False
        # v3i19 arc-credit: finalize the still-open arc on episode termination.
        # The arc's accumulated reward is the segment-return for credit
        # attribution to the z that was active in those last steps. Must run
        # BEFORE the next ``strategy_for_step`` (which would otherwise see
        # ``arc_has_open=True`` and overwrite the snapshot).
        if bool(getattr(self.hparams, "latent_arc_credit_enabled", False)):
            latent_state.arc_finalize(done_t, reason="episode_end")
        latent_state.macro_finalize(done_t, reason="episode_end")

    def _append_global_state_probe_rows(
        self,
        decision_global_state_np: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """Append optional ``(global_state, score_diff, time_frac)`` probe rows."""
        probe_rows = getattr(self.runtime, "_global_state_probe_rows", None)
        if probe_rows is None:
            return
        env_cfg = self.env.cfg
        score_lim = max(1, int(getattr(env_cfg, "score_limit", 1)))
        max_dec = max(1, int(getattr(env_cfg, "max_decision_steps", 400)))
        for i, info in enumerate(infos):
            bs = int(info.get("blue_score", 0) or 0)
            rs = int(info.get("red_score", 0) or 0)
            ds = int(info.get("decision_steps", 0) or 0)
            probe_rows.append(
                {
                    "global_state": np.asarray(decision_global_state_np[i], dtype=np.float32).copy(),
                    "score_diff": float(bs - rs) / float(score_lim),
                    "time_frac": float(ds) / float(max_dec),
                }
            )

    def _apply_flag_resample_trigger(
        self,
        context_state: torch.Tensor,
        next_global_state: np.ndarray,
    ) -> None:
        """Mark envs whose flag-territory features changed for next-step resample."""
        prev_sec = context_state[:, GLOBAL_STATE_FLAG_TERRITORY_SLICE]
        nxt_sec = torch.as_tensor(
            next_global_state[:, GLOBAL_STATE_FLAG_TERRITORY_SLICE],
            dtype=torch.float32,
            device=self.device,
        )
        chg = self.flag_territory_features_changed(prev_sec, nxt_sec)
        self.latent_state.needs_strategy_sample[chg] = True

    def _advance_context(self, next_global_state: np.ndarray) -> torch.Tensor:
        """Return the context tensor for the *next* step.

        Latent mode: the temporal tracker was already advanced inside
        :meth:`next_values`, so we just hand back ``runtime._last_context_state``.
        Non-latent mode: context is just the (raw) global state on device.
        """
        if self.hparams.use_latent_strategy:
            return self.runtime._last_context_state
        return torch.as_tensor(next_global_state, dtype=torch.float32, device=self.device)

    def _finalize_step_latent_state(
        self,
        strategy_aux: Dict[str, torch.Tensor],
        dones: np.ndarray,
    ) -> None:
        """End-of-step latent housekeeping (KL-prev snapshot + strategy age)."""
        latent_state = self.latent_state
        hparams = self.hparams
        if (
            hparams.use_latent_strategy
            and hparams.latent_kl_consecutive > 0.0
            and latent_state.z_kl_first_in_ep is not None
        ):
            latent_state.prev_z_logits = strategy_aux["z_logits"].detach().clone()
            latent_state.z_kl_first_in_ep = torch.as_tensor(
                dones, dtype=torch.bool, device=self.device
            )
        latent_state.mark_strategy_step_done(dones)

    def _append_e3_step_telemetry(
        self,
        *,
        step_idx: int,
        decision_global_state_np: np.ndarray,
        z_t: Optional[torch.Tensor],
        prev_z_t: Optional[torch.Tensor],
        strategy_aux: Dict[str, torch.Tensor],
        infos: List[Dict[str, Any]],
        beh_t: Optional[torch.Tensor],
        sb: Optional[torch.Tensor],
        rb: Optional[torch.Tensor],
        pb: Optional[torch.Tensor],
        adb: Optional[torch.Tensor],
        blue_ahead_t: Optional[torch.Tensor],
        context_state: Optional[torch.Tensor] = None,
    ) -> None:
        """Forward one row to the E3 step telemetry CSV when enabled."""
        if not (self.telemetry.e3_step_telemetry_path and self.hparams.use_latent_strategy):
            return
        if z_t is None or prev_z_t is None:
            return
        if (
            beh_t is None
            or sb is None
            or rb is None
            or pb is None
            or adb is None
            or blue_ahead_t is None
        ):
            return
        self.telemetry.append_e3_step(
            rollout_step=step_idx,
            global_step_at_step_end=int(self.runtime.global_step),
            decision_global_state_np=decision_global_state_np,
            z_t=z_t,
            prev_z=prev_z_t,
            strategy_aux=strategy_aux,
            infos=infos,
            behavior_telemetry_np=beh_t.detach().cpu().numpy(),
            spread_bucket_np=sb.detach().cpu().numpy(),
            role_bucket_np=rb.detach().cpu().numpy(),
            pressure_bucket_np=pb.detach().cpu().numpy(),
            attack_defense_ratio_bucket_np=adb.detach().cpu().numpy(),
            blue_ahead_np=blue_ahead_t.detach().cpu().numpy(),
            context_state=context_state,
        )


__all__ = ["RolloutCollector"]
