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

Per the refactor plan we deliberately keep ``trainer`` as a context object
for this first pass — making the dependencies explicit later is a follow-up.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional

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
from rl.latent_phase_labels import (
    outcome_id_from_global_state,
    team_phase_id_from_global_state,
)
from rl.ppo_core import (
    TensorDictRolloutBuffer,
    align_next_values_to_rollout_actions,
)
from rl.custom_ppo.csv_writers import _opponent_id_int_from_info
from rl.custom_ppo.curriculum_runtime import _update_curriculum_after_episode
from rl.custom_ppo.option_returns import compute_option_returns
from rl.custom_ppo.return_normalization import (
    _denormalize_values,
    _update_return_norm_stats,
)
from rl.custom_ppo.reward_composition import _compose_training_reward_components
from rl.custom_ppo.trainer_audit import log_decentralized_actor_contract_once

if TYPE_CHECKING:
    from rl.custom_ppo.trainer import CustomPPOTrainer


class RolloutCollector:
    """Owns ``collect_rollout`` and its env-stepping helpers.

    Constructed once by the trainer and held as ``self.rollout_collector``.
    Reads trainer state (cfg, env, model, optimizer-free, global_step) and
    mutates trainer counters (``_episodes_completed``, ``global_step``,
    ``_last_obs``, ``_last_global_state``, ``_last_context_state``,
    ``_ep_wins/losses/draws``, ``_rollout_episode_records``,
    ``_recent_episode_successes``).
    """

    def __init__(self, trainer: "CustomPPOTrainer") -> None:
        self.trainer = trainer

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
        device = self.trainer.device
        return {
            "grid": torch.as_tensor(obs["grid"], dtype=torch.float32, device=device),
            "vec": torch.as_tensor(obs["vec"], dtype=torch.float32, device=device),
            "agent_mask": torch.as_tensor(obs["agent_mask"], dtype=torch.float32, device=device),
            "mask": torch.as_tensor(obs["mask"], dtype=torch.float32, device=device),
        }

    def on_sb3_rollout_env_step(self) -> None:
        trainer = self.trainer
        p = trainer._sb3_rollout_pbar
        if p is None:
            return
        nenv = int(trainer.env.num_envs)
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
        trainer = self.trainer
        n_steps = int(trainer.cfg.n_steps)
        n_envs = int(trainer.env.num_envs)
        buffer = TensorDictRolloutBuffer(n_steps, n_envs, device=trainer.device)
        buffer.register_field("obs_grid", tuple(obs["grid"].shape[1:]))
        buffer.register_field("obs_vec", tuple(obs["vec"].shape[1:]))
        buffer.register_field("obs_agent_mask", tuple(obs["agent_mask"].shape[1:]))
        buffer.register_field("obs_mask", tuple(obs["mask"].shape[1:]))
        buffer.register_field("global_state", (trainer.model.global_state_dim,))
        buffer.register_field("actions", (len(getattr(trainer.env.action_space, "nvec", [])),), dtype=torch.long)
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
        buffer.register_field("reward_total")
        buffer.register_field("terminated", dtype=torch.bool)
        buffer.register_field("truncated", dtype=torch.bool)
        buffer.register_field("opponent_id", dtype=torch.long)
        if trainer.use_latent_strategy:
            buffer.register_field("z", dtype=torch.long)
            buffer.register_field("prev_z", dtype=torch.long)
            buffer.register_field("z_log_probs")
            buffer.register_field("z_logits", (trainer.latent_k,))
            buffer.register_field("z_resampled", dtype=torch.bool)
            buffer.register_field("z_persist_mask", dtype=torch.bool)
            buffer.register_field("phase_id", dtype=torch.long)
            buffer.register_field("outcome_id", dtype=torch.long)
            buffer.register_field("behavior_telemetry", (N_TELEMETRY,))
            buffer.register_field("spread_bucket_id", dtype=torch.long)
            buffer.register_field("role_bucket_id", dtype=torch.long)
            buffer.register_field("pressure_bucket_id", dtype=torch.long)
            buffer.register_field("attack_defense_ratio_bucket_id", dtype=torch.long)
            buffer.register_field("blue_ahead", dtype=torch.float32)
            if trainer.latent_kl_consecutive > 0.0:
                buffer.register_field("z_logits_prev", (trainer.latent_k,))
                buffer.register_field("z_kl_prev_valid")
        return buffer

    def z_for_bootstrap(
        self,
        next_context_gs_t: torch.Tensor,
        z_t: torch.Tensor,
        dones: np.ndarray,
    ) -> torch.Tensor:
        """Strategy index for V(s', z') bootstrapping to match the start of the *next* decision."""
        trainer = self.trainer
        if not trainer.use_latent_strategy:
            raise RuntimeError("z_for_bootstrap requires latent strategy mode.")
        if trainer.fixed_latent_strategy:
            return torch.full_like(z_t, int(trainer.fixed_latent_strategy_id), dtype=torch.long)
        device = trainer.device
        done_t = torch.as_tensor(dones, dtype=torch.bool, device=device)
        age_next = trainer.latent_state.strategy_age + 1
        age_next = torch.where(done_t, torch.zeros_like(age_next), age_next)
        needs_next = trainer.latent_state.needs_strategy_sample.clone()
        if bool(done_t.any().item()):
            needs_next = needs_next.clone()
            needs_next[done_t] = bool(not trainer.fixed_latent_strategy)
        resample_next = needs_next.clone()
        if trainer.latent_resample_every_n > 0:
            resample_next = resample_next | (age_next >= int(trainer.latent_resample_every_n))
        resample_next = resample_next & (~done_t)
        z_next = z_t.long().clone()
        if bool(resample_next.any().item()):
            idx = torch.where(resample_next)[0]
            gs_sub = next_context_gs_t.index_select(0, idx)
            sampled_z, _, _, _ = trainer.model.sample_strategy(
                gs_sub,
                deterministic=bool(trainer.latent_bootstrap_z_deterministic),
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
        trainer = self.trainer
        rows = []
        for env_i, info in enumerate(infos):
            if bool(info.get("terminated", False)):
                rows.append(np.zeros((GLOBAL_STATE_DIM,), dtype=np.float32))
            elif bool(info.get("truncated", False)):
                terminal_obs = info.get("terminal_observation") or {}
                rows.append(np.asarray(terminal_obs.get("global_state", next_global_state[env_i]), dtype=np.float32))
            else:
                rows.append(np.asarray(next_global_state[env_i], dtype=np.float32))
        gs = torch.as_tensor(np.stack(rows, axis=0), dtype=torch.float32, device=trainer.device)
        with torch.no_grad():
            if not trainer.use_latent_strategy:
                return _denormalize_values(trainer, trainer.model.values(gs))

            done_t = torch.as_tensor(dones, dtype=torch.bool, device=trainer.device) if dones is not None else None
            next_context_gs_t = trainer.temporal_tracker.update(gs, dones=done_t)
            trainer._last_context_state = next_context_gs_t

            if next_obs is None or prev_z is None:
                raise ValueError("latent next value bootstrap requires next_obs and prev_z.")
            obs_rows = self.obs_rows_from_next(next_obs, infos)
            next_obs_t = self.tensor_obs(obs_rows)
            if dones is None:
                raise ValueError("latent next value bootstrap requires dones for z lookahead.")
            next_z = self.z_for_bootstrap(
                next_context_gs_t,
                prev_z.long().reshape(-1),
                dones,
            )
            _, next_values, _, _ = trainer.model.act(
                next_obs_t,
                next_context_gs_t,
                deterministic=True,
                z_idx=next_z,
            )
            next_values = _denormalize_values(trainer, next_values)
            terminated = torch.as_tensor(
                [bool(info.get("terminated", False)) for info in infos],
                dtype=torch.bool,
                device=trainer.device,
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
        trainer = self.trainer
        er = info.get("episode_result")
        if isinstance(er, dict):
            bs = int(er.get("blue_score", 0))
            rs = int(er.get("red_score", 0))
        else:
            bs = int(info.get("blue_score", 0))
            rs = int(info.get("red_score", 0))
        success = 1 if bs > rs else 0
        if bs > rs:
            trainer._ep_wins += 1
        elif bs < rs:
            trainer._ep_losses += 1
        else:
            trainer._ep_draws += 1
        trainer._episodes_completed += 1
        trainer._rollout_episode_records.append(
            {
                "blue_score": int(bs),
                "red_score": int(rs),
                "win_margin": int(bs) - int(rs),
                "success": success,
                "latent_z": latent_z,
                "opponent_id": int(_opponent_id_int_from_info(trainer.cfg, info)),
            }
        )
        trainer._recent_episode_successes.append(success)
        trainer.telemetry.write_episode_metrics(
            info,
            blue_score=bs,
            red_score=rs,
            timestep=int(timestep or trainer.global_step),
            rollout_step=rollout_step,
            latent_z=latent_z,
        )
        _update_curriculum_after_episode(trainer, info=info, blue_score=bs, red_score=rs, env_index=env_index)
        every = int(getattr(trainer.cfg, "episode_log_every", 0) or 0)
        if every > 0 and trainer._episodes_completed % every == 0:
            trainer.telemetry.print_episode_progress(info)

    # ------------------------------------------------------------------
    # Main entry point.
    # ------------------------------------------------------------------

    def collect(self) -> TensorDictRolloutBuffer:
        """Collect one rollout and compute advantages/returns."""
        trainer = self.trainer
        log_decentralized_actor_contract_once(trainer)
        trainer._rollout_episode_records = []
        trainer.latent_state.rollout_strategy_episode_records = []
        if trainer._last_obs is None or trainer._last_global_state is None:
            obs = trainer.env.reset()
            global_state = trainer.env.state().astype(np.float32)
            trainer.latent_state.reset()
            if trainer.use_latent_strategy:
                gs_t = torch.as_tensor(global_state, dtype=torch.float32, device=trainer.device)
                context_state = trainer.temporal_tracker.update(gs_t)
            else:
                context_state = torch.as_tensor(global_state, dtype=torch.float32, device=trainer.device)
        else:
            obs = trainer._last_obs
            global_state = trainer._last_global_state
            if trainer.use_latent_strategy:
                context_state = trainer._last_context_state
            else:
                context_state = torch.as_tensor(global_state, dtype=torch.float32, device=trainer.device)
        buffer = self.make_buffer(obs)
        for step_idx in range(int(trainer.cfg.n_steps)):
            decision_global_state_np = np.asarray(global_state, dtype=np.float32)
            obs_t = self.tensor_obs(obs)
            with torch.no_grad():
                z_t, prev_z_t, strategy_aux = trainer.latent_state.strategy_for_step(context_state)
                actions_t, values_norm_t, action_log_probs_t, _ = trainer.model.act(obs_t, context_state, z_idx=z_t)
                values_t = _denormalize_values(trainer, values_norm_t)
                log_probs_t = action_log_probs_t
            actions_np = actions_t.detach().cpu().numpy().astype(np.int64)
            beh_t = sb = rb = pb = adb = blue_ahead_t = None
            if trainer.use_latent_strategy:
                beh_t = compute_behavior_telemetry_batch(trainer.env.core, actions_t)
                sb, rb, pb, adb = bucket_ids_from_telemetry(beh_t, actions_t, trainer.env.core)
                blue_ahead_t = (trainer.env.core.blue_score > trainer.env.core.red_score).to(
                    dtype=torch.float32, device=trainer.device
                )
            trainer.env.step_async(actions_np)
            next_obs, rewards, dones, infos = trainer.env.step_wait()
            step_after = trainer.global_step + int(trainer.env.num_envs)
            z_np = z_t.detach().cpu().numpy() if z_t is not None else None
            for env_i, (done_i, info) in enumerate(zip(dones, infos)):
                if bool(done_i):
                    latent_z = int(z_np[env_i]) if z_np is not None else None
                    self.on_episode_done(
                        dict(info),
                        timestep=step_after,
                        rollout_step=step_idx + 1,
                        latent_z=latent_z,
                        env_index=env_i,
                    )
            next_global_state = trainer.env.state().astype(np.float32)
            next_values_t = self.next_values(infos, next_global_state, next_obs=next_obs, prev_z=z_t, dones=dones)
            terminated = np.asarray([bool(info.get("terminated", bool(done))) for info, done in zip(infos, dones)])
            truncated = np.asarray([bool(info.get("truncated", False)) for info in infos])
            reward_component = {
                key: torch.as_tensor(
                    [float(info.get(key, 0.0) or 0.0) for info in infos],
                    dtype=torch.float32,
                    device=trainer.device,
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
            shaping_coef = float(trainer._reward_shaping_coef())
            stalemate = torch.as_tensor(
                [bool(info.get("stalemate_truncated", False)) for info in infos],
                dtype=torch.bool,
                device=trainer.device,
            )
            reward_component = _compose_training_reward_components(
                reward_component,
                dense_weight=trainer.reward_dense_weight,
                reward_scale=trainer.reward_scale,
                reward_clip=trainer.reward_clip,
                shaping_coef=shaping_coef,
                stalemate=stalemate,
                stalemate_penalty=trainer.reward_stalemate_penalty,
            )
            if trainer.use_latent_strategy:
                done_t = torch.as_tensor(dones, dtype=torch.bool, device=trainer.device)
                trainer.latent_state.episode_return_accum = (
                    trainer.latent_state.episode_return_accum + reward_component["reward_total"].detach()
                )
                if bool(done_t.any().item()):
                    if trainer.latent_episode_strategy_ppo:
                        for env_i, done_i in enumerate(dones):
                            if bool(done_i):
                                trainer.latent_state.record_episode_strategy_outcome(
                                    env_i,
                                    dict(infos[env_i]),
                                    episode_return=float(
                                        trainer.latent_state.episode_return_accum[env_i].detach().cpu().item()
                                    ),
                                )
                    trainer.latent_state.episode_return_accum[done_t] = 0.0
                    trainer.latent_state.episode_strategy_has_start[done_t] = False

            opp_row = torch.as_tensor(
                [_opponent_id_int_from_info(trainer.cfg, dict(info)) for info in infos],
                dtype=torch.long,
                device=trainer.device,
            )

            add_items: dict[str, torch.Tensor] = dict(
                obs_grid=torch.as_tensor(obs["grid"], dtype=torch.float32, device=trainer.device),
                obs_vec=torch.as_tensor(obs["vec"], dtype=torch.float32, device=trainer.device),
                obs_agent_mask=torch.as_tensor(obs["agent_mask"], dtype=torch.float32, device=trainer.device),
                obs_mask=torch.as_tensor(obs["mask"], dtype=torch.float32, device=trainer.device),
                global_state=context_state,
                actions=actions_t,
                log_probs=log_probs_t,
                values=values_t,
                values_norm=values_norm_t,
                next_values=next_values_t,
                rewards=reward_component["reward_total"],
                reward_terminal=reward_component["reward_terminal"],
                reward_offense=reward_component["reward_offense"],
                reward_pbrs=reward_component["reward_pbrs"],
                reward_team=reward_component["reward_team"],
                reward_sparse=reward_component["reward_sparse"],
                reward_sparse_points=reward_component["reward_sparse_points"],
                reward_failure=reward_component["reward_failure"],
                reward_total=reward_component["reward_total"],
                terminated=torch.as_tensor(terminated, dtype=torch.bool, device=trainer.device),
                truncated=torch.as_tensor(truncated, dtype=torch.bool, device=trainer.device),
                opponent_id=opp_row,
            )
            if trainer.use_latent_strategy:
                n_e = int(trainer.env.num_envs)
                phase_list: list[int] = []
                outcome_list: list[int] = []
                for e in range(n_e):
                    info_e = dict(infos[e]) if e < len(infos) else {}
                    sf = float(info_e.get("stalemate_frac", 0.0) or 0.0)
                    phase_list.append(
                        int(team_phase_id_from_global_state(decision_global_state_np[e], stalemate_frac=sf))
                    )
                    outcome_list.append(int(outcome_id_from_global_state(decision_global_state_np[e])))
                add_items.update(
                    z=strategy_aux["z"],
                    prev_z=strategy_aux["prev_z"],
                    z_log_probs=strategy_aux["z_log_prob"],
                    z_logits=strategy_aux["z_logits"],
                    z_resampled=strategy_aux["z_resampled"],
                    z_persist_mask=strategy_aux["z_persist_mask"],
                    phase_id=torch.as_tensor(phase_list, dtype=torch.long, device=trainer.device),
                    outcome_id=torch.as_tensor(outcome_list, dtype=torch.long, device=trainer.device),
                    behavior_telemetry=beh_t,
                    spread_bucket_id=sb,
                    role_bucket_id=rb,
                    pressure_bucket_id=pb,
                    attack_defense_ratio_bucket_id=adb,
                    blue_ahead=blue_ahead_t,
                )
                if trainer.latent_kl_consecutive > 0.0 and trainer.latent_state.z_kl_first_in_ep is not None:
                    z_logits_cur = strategy_aux["z_logits"]
                    zlp = trainer.latent_state.prev_z_logits
                    if zlp is None:
                        zlp = torch.zeros_like(z_logits_cur)
                    add_items["z_logits_prev"] = zlp
                    add_items["z_kl_prev_valid"] = (~trainer.latent_state.z_kl_first_in_ep).to(dtype=torch.float32)
            buffer.add(**add_items)
            probe_rows = getattr(trainer, "_global_state_probe_rows", None)
            if probe_rows is not None:
                score_lim = max(1, int(getattr(trainer.env.cfg, "score_limit", 1)))
                max_dec = max(1, int(getattr(trainer.env.cfg, "max_decision_steps", 400)))
                gs_np = decision_global_state_np
                for i, info in enumerate(infos):
                    bs = int(info.get("blue_score", 0) or 0)
                    rs = int(info.get("red_score", 0) or 0)
                    ds = int(info.get("decision_steps", 0) or 0)
                    probe_rows.append(
                        {
                            "global_state": np.asarray(gs_np[i], dtype=np.float32).copy(),
                            "score_diff": float(bs - rs) / float(score_lim),
                            "time_frac": float(ds) / float(max_dec),
                        }
                    )
            if trainer.latent_resample_on_flag:
                prev_sec = context_state[:, GLOBAL_STATE_FLAG_TERRITORY_SLICE]
                nxt_sec = torch.as_tensor(
                    next_global_state[:, GLOBAL_STATE_FLAG_TERRITORY_SLICE],
                    dtype=torch.float32,
                    device=trainer.device,
                )
                chg = self.flag_territory_features_changed(prev_sec, nxt_sec)
                trainer.latent_state.needs_strategy_sample[chg] = True
            obs = next_obs
            global_state = next_global_state
            if trainer.use_latent_strategy:
                context_state = trainer._last_context_state
            else:
                context_state = torch.as_tensor(global_state, dtype=torch.float32, device=trainer.device)
            trainer.global_step += int(trainer.env.num_envs)
            self.on_sb3_rollout_env_step()
            if trainer.use_latent_strategy and trainer.latent_kl_consecutive > 0.0 and trainer.latent_state.z_kl_first_in_ep is not None:
                trainer.latent_state.prev_z_logits = strategy_aux["z_logits"].detach().clone()
                trainer.latent_state.z_kl_first_in_ep = torch.as_tensor(dones, dtype=torch.bool, device=trainer.device)
            trainer.latent_state.mark_strategy_step_done(dones)
            if trainer.telemetry.e3_step_telemetry_path and trainer.use_latent_strategy and z_t is not None and prev_z_t is not None:
                assert beh_t is not None and sb is not None and adb is not None
                trainer.telemetry.append_e3_step(
                    rollout_step=step_idx,
                    global_step_at_step_end=int(trainer.global_step),
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
                )

        buffer.fields["next_values"][: int(buffer.pos)].copy_(
            align_next_values_to_rollout_actions(
                buffer.fields["values"][: int(buffer.pos)],
                buffer.fields["next_values"][: int(buffer.pos)],
                buffer.fields["terminated"][: int(buffer.pos)].bool(),
                buffer.fields["truncated"][: int(buffer.pos)].bool(),
            )
        )
        gae_kw: dict[str, Any] = dict(
            gamma=float(trainer.cfg.gamma),
            gae_lambda=float(trainer.cfg.gae_lambda),
        )
        if trainer.latent_gae_reset_on_z_change:
            gae_kw["latent_z_field"] = "z"
            gae_kw["reset_gae_on_z_change"] = True
        buffer.compute_returns_and_advantages(**gae_kw)
        if trainer.use_latent_strategy:
            with torch.no_grad():
                option_returns, option_advantages = compute_option_returns(
                    rewards=buffer.fields["rewards"],
                    values=buffer.fields["values"],
                    next_values=buffer.fields["next_values"],
                    terminated=buffer.fields["terminated"],
                    truncated=buffer.fields["truncated"],
                    z_resampled=buffer.fields["z_resampled"],
                    gamma=float(trainer.cfg.gamma),
                )
                if "option_returns" not in buffer.fields:
                    buffer.register_field("option_returns")
                if "option_advantages" not in buffer.fields:
                    buffer.register_field("option_advantages")
                buffer.fields["option_returns"].copy_(option_returns)
                buffer.fields["option_advantages"].copy_(option_advantages)
        _update_return_norm_stats(trainer, buffer.fields["returns"][: int(buffer.pos)])
        trainer._last_obs = obs
        trainer._last_global_state = global_state
        return buffer


__all__ = ["RolloutCollector"]
