"""Per-step router sampling: sparse z resample, forced-z, refresh, behavior log-probs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch
from torch.distributions import Categorical

from rl.global_state import GLOBAL_STATE_DIM
from rl.custom_ppo.latent.behavior_policy import (
    behavior_log_prob_from_probs,
    epsilon_behavior_probs,
    resolve_action_sources,
)
from rl.custom_ppo.latent.types import RouterAction, RouterActionSource
from rl.custom_ppo.latent.context_buckets import (
    carrier_progress_bucket_ids,
    flag_state_bucket_ids,
    score_pressure_bucket_ids,
    strategy_experience_bucket_ids,
    team_phase_bucket_ids,
)
from rl.custom_ppo.schedules import resolve_latent_forced_z_frac

if TYPE_CHECKING:
    from rl.custom_ppo.latent.state import LatentStrategyState


class RouterSamplingState:
    def __init__(self, host: "LatentStrategyState") -> None:
        self.host = host

    @property
    def trainer(self):
        return self.host.trainer

    def strategy_for_step(
        self,
        global_state: torch.Tensor,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], dict[str, torch.Tensor]]:
        """Return current sparse strategy and sampling metadata for one rollout step."""
        trainer = self.host.trainer
        if not trainer.use_latent_strategy:
            return None, None, {}
        if self.host.current_z is None:
            self.host.reset()
        self.host.record_tactical_context_step(global_state)
        assert self.host.current_z is not None

        device = trainer.device
        if trainer.fixed_latent_strategy:
            batch = int(global_state.shape[0])
            z_idx = torch.full(
                (batch,), trainer.fixed_latent_strategy_id, dtype=torch.long, device=device
            )
            prev_z = self.host.current_z.clone()
            self.host.current_z = z_idx.clone()
            fixed_logits = torch.full(
                (batch, trainer.latent_k), -1.0e8, dtype=torch.float32, device=device
            )
            fixed_logits[:, trainer.fixed_latent_strategy_id] = 0.0
            false_mask = torch.zeros((batch,), dtype=torch.bool, device=device)
            aux = {
                "z": z_idx,
                "prev_z": prev_z,
                "z_log_prob": torch.zeros((batch,), dtype=torch.float32, device=device),
                "z_entropy": torch.zeros((batch,), dtype=torch.float32, device=device),
                "z_logits": fixed_logits,
                "z_resampled": false_mask,
                "z_forced": false_mask,
                "z_persist_mask": false_mask,
            }
            return z_idx, prev_z, aux

        episode_start_mask = self.host.needs_strategy_sample.clone()
        resample_mask = episode_start_mask.clone()
        if trainer.latent_resample_every_n > 0:
            resample_mask |= self.host.strategy_age >= trainer.latent_resample_every_n

        # v3i event refresh
        trigger_enemy_flag = torch.zeros_like(episode_start_mask)
        trigger_friendly_flag = torch.zeros_like(episode_start_mask)
        trigger_score = torch.zeros_like(episode_start_mask)
        trigger_near_base = torch.zeros_like(episode_start_mask)
        trigger_refresh = torch.zeros_like(episode_start_mask)
        sparse_reason_interval = torch.zeros_like(episode_start_mask)
        sparse_reason_flag = torch.zeros_like(episode_start_mask)
        sparse_reason_phase = torch.zeros_like(episode_start_mask)
        sparse_reason_score_pressure = torch.zeros_like(episode_start_mask)
        sparse_refresh_attempt = torch.zeros_like(episode_start_mask)
        sparse_refresh_accepted = torch.zeros_like(episode_start_mask)

        curr_gs = global_state[:, :GLOBAL_STATE_DIM].float().detach()

        if getattr(trainer, "latent_sparse_tactical_refresh_enabled", False):
            if self.host.prev_global_state is not None:
                active_envs = (
                    (~episode_start_mask)
                    & self.host.episode_strategy_committed
                    & (~self.host.episode_forced_z)
                )
                if bool(active_envs.any().item()):
                    prev_gs = self.host.prev_global_state
                    interval_steps = max(
                        1,
                        int(
                            getattr(
                                trainer,
                                "latent_sparse_tactical_refresh_interval_steps",
                                32,
                            )
                            or 32
                        ),
                    )
                    min_dwell_steps = max(
                        1,
                        int(
                            getattr(
                                trainer,
                                "latent_sparse_tactical_refresh_min_dwell_steps",
                                16,
                            )
                            or 16
                        ),
                    )
                    sparse_reason_interval = active_envs & (
                        self.host.steps_since_last_tactical_refresh >= interval_steps
                    )
                    sparse_reason_flag = active_envs & (
                        flag_state_bucket_ids(prev_gs)
                        != flag_state_bucket_ids(curr_gs)
                    )
                    sparse_reason_phase = active_envs & (
                        team_phase_bucket_ids(prev_gs)
                        != team_phase_bucket_ids(curr_gs)
                    )
                    sparse_reason_score_pressure = active_envs & (
                        score_pressure_bucket_ids(prev_gs)
                        != score_pressure_bucket_ids(curr_gs)
                    )
                    sparse_refresh_attempt = (
                        sparse_reason_interval
                        | sparse_reason_flag
                        | sparse_reason_phase
                        | sparse_reason_score_pressure
                    )
                    dwell_satisfied = (
                        self.host.steps_since_z_change >= min_dwell_steps
                    )
                    sparse_refresh_accepted = (
                        sparse_refresh_attempt & dwell_satisfied
                    )
                    sparse_refresh_rejected = (
                        sparse_refresh_attempt & (~dwell_satisfied)
                    )

                    self.host.rollout_sparse_refresh_attempt_count += int(
                        sparse_refresh_attempt.sum().item()
                    )
                    self.host.rollout_sparse_refresh_accept_count += int(
                        sparse_refresh_accepted.sum().item()
                    )
                    self.host.rollout_sparse_refresh_reject_dwell_count += int(
                        sparse_refresh_rejected.sum().item()
                    )
                    self.host.rollout_sparse_refresh_reason_interval += int(
                        sparse_reason_interval.sum().item()
                    )
                    self.host.rollout_sparse_refresh_reason_flag += int(
                        sparse_reason_flag.sum().item()
                    )
                    self.host.rollout_sparse_refresh_reason_phase += int(
                        sparse_reason_phase.sum().item()
                    )
                    self.host.rollout_sparse_refresh_reason_score_pressure += int(
                        sparse_reason_score_pressure.sum().item()
                    )
                    resample_mask |= sparse_refresh_accepted

        if getattr(trainer, "latent_event_refresh_enabled", False):
            self.host.rollout_refresh_total_steps += int(curr_gs.shape[0])
            if self.host.prev_global_state is not None:
                active_envs = ~episode_start_mask
                if bool(active_envs.any().item()):
                    prev_gs = self.host.prev_global_state

                    # 1. enemy captures/grabs flag (index 10)
                    trigger_enemy_flag = active_envs & (prev_gs[:, 10] <= 0.5) & (curr_gs[:, 10] > 0.5)
                    # 2. friendly captures/grabs flag (index 11)
                    trigger_friendly_flag = active_envs & (prev_gs[:, 11] <= 0.5) & (curr_gs[:, 11] > 0.5)
                    # 3. score changes (indices 14 and 15)
                    trigger_score = active_envs & ((prev_gs[:, 14] != curr_gs[:, 14]) | (prev_gs[:, 15] != curr_gs[:, 15]))

                    # 4. enemy carrier near base
                    enemy_near = (curr_gs[:, 10] > 0.5) & (curr_gs[:, 23] < 0.20)
                    enemy_near_prev = (prev_gs[:, 10] > 0.5) & (prev_gs[:, 23] < 0.20)
                    trigger_enemy_near = active_envs & enemy_near & ~enemy_near_prev

                    # 5. friendly carrier near base
                    friendly_near = (curr_gs[:, 11] > 0.5) & (curr_gs[:, 23] < 0.20)
                    friendly_near_prev = (prev_gs[:, 11] > 0.5) & (prev_gs[:, 23] < 0.20)
                    trigger_friendly_near = active_envs & friendly_near & ~friendly_near_prev

                    trigger_near_base = trigger_enemy_near | trigger_friendly_near

                    # Guardrails
                    event_refresh_allowed = (
                        (self.host.steps_since_last_refresh >= trainer.latent_event_refresh_min_gap_steps)
                        & (self.host.refresh_count_this_episode < trainer.latent_event_refresh_max_per_episode)
                    )

                    trigger_refresh = event_refresh_allowed & (
                        trigger_enemy_flag | trigger_friendly_flag | trigger_score | trigger_near_base
                    )
                    resample_mask |= trigger_refresh

        # Warmup: defer the committed z snapshot until ctx170 EMAs
        # have observed a few decision steps of opponent behavior. The provisional
        # z chosen at step 0 still drives actions during the warmup window, but we
        # force a resample at the commit step and snapshot/train on that committed
        # (context, z) pair instead. Without this guard, q_phi is fed a structurally
        # opponent-blind context (raw initial geometry + zeroed EMAs) at step 0.
        warmup = int(getattr(trainer, "latent_episode_strategy_warmup_decision_steps", 0) or 0)
        commit_now = torch.zeros_like(episode_start_mask)
        if warmup > 0:
            commit_now = (
                (self.host.steps_since_ep_start == warmup)
                & (~self.host.episode_strategy_committed)
                & (~episode_start_mask)  # never both on the same call
            )
            if bool(commit_now.any().item()):
                resample_mask = resample_mask | commit_now
                # Fix forced-z bucket alignment at warmup/commit step:
                forced_commit = commit_now & self.host.episode_forced_z
                if bool(forced_commit.any().item()):
                    f_idx = torch.where(forced_commit)[0]
                    self.host.episode_contrast_bucket[f_idx] = strategy_experience_bucket_ids(
                        global_state.index_select(0, f_idx)
                    ).detach()

        prev_z = self.host.current_z.clone()
        z_idx = self.host.current_z.clone()
        persist_mask = resample_mask & (~self.host.needs_strategy_sample) & (~commit_now)

        if bool(episode_start_mask.any().item()):
            self.host.begin_episodes(episode_start_mask)

        selector_hidden_pre = (
            self.host.selector_hidden.clone() if self.host.selector_hidden is not None else None
        )
        selector_hidden = self.host.selector_hidden
        if selector_hidden is not None:
            z_logits = trainer.model.strategy_logits(global_state, selector_hidden=selector_hidden)
        else:
            z_logits = trainer.model.strategy_logits(global_state)
        z_dist = Categorical(logits=z_logits)
        if bool(episode_start_mask.any().item()):
            start_idx = torch.where(episode_start_mask)[0]
            self.host.episode_forced_z[start_idx] = False
            self.host.v6i1_episode_rehearsal[start_idx] = False
            self.host.episode_behavior_sum[start_idx] = 0.0
            self.host.episode_behavior_count[start_idx] = 0
            self.host.episode_contrast_bucket[start_idx] = strategy_experience_bucket_ids(
                global_state.index_select(0, start_idx)
            ).detach()
            from rl.custom_ppo.v6i1_phase_runtime import (
                is_v6i1_staged_trainer,
                resolve_v6i1_episode_forced_frac,
                resolve_v6i1_episode_rehearsal_prob,
                v6i1_schedule_context,
            )

            gen = trainer.model._sampling_gen_strategy
            rand_kwargs = {"dtype": torch.float32, "device": device}
            if gen is not None:
                rand_kwargs["generator"] = gen
            if is_v6i1_staged_trainer(trainer):
                phase, _, _, _ = v6i1_schedule_context(trainer)
                rehearsal_prob = float(resolve_v6i1_episode_rehearsal_prob(trainer))
                if phase in ("B", "C") and rehearsal_prob > 0.0:
                    rehearsal_draw = torch.rand((int(start_idx.numel()),), **rand_kwargs)
                    rehearsal_local = rehearsal_draw < rehearsal_prob
                    if bool(rehearsal_local.any().item()):
                        rehearsal_idx = start_idx[rehearsal_local]
                        self.host.v6i1_episode_rehearsal[rehearsal_idx] = True
                        uniform_logits = torch.zeros(
                            (int(rehearsal_idx.numel()), trainer.latent_k),
                            dtype=torch.float32,
                            device=device,
                        )
                        uniform_dist = Categorical(logits=uniform_logits)
                        forced_z = trainer.model._categorical_argmax_or_sample(
                            uniform_dist,
                            deterministic=False,
                            generator=trainer.model._sampling_gen_strategy,
                        ).long()
                        self.host.episode_forced_z[rehearsal_idx] = True
                        self.host.episode_forced_z_id[rehearsal_idx] = forced_z
                router_idx = start_idx[~self.host.v6i1_episode_rehearsal[start_idx]]
                forced_frac = float(resolve_v6i1_episode_forced_frac(trainer))
            else:
                router_idx = start_idx
                forced_frac = resolve_latent_forced_z_frac(
                    trainer.cfg,
                    global_step=int(getattr(trainer, "global_step", 0) or 0),
                )
            if forced_frac > 0.0 and int(router_idx.numel()) > 0:
                forced_draw = torch.rand((int(router_idx.numel()),), **rand_kwargs)
                forced_mask_local = forced_draw < forced_frac
                if bool(forced_mask_local.any().item()):
                    forced_idx = router_idx[forced_mask_local]
                    uniform_logits = torch.zeros(
                        (int(forced_idx.numel()), trainer.latent_k),
                        dtype=torch.float32,
                        device=device,
                    )
                    uniform_dist = Categorical(logits=uniform_logits)
                    forced_z = trainer.model._categorical_argmax_or_sample(
                        uniform_dist,
                        deterministic=False,
                        generator=trainer.model._sampling_gen_strategy,
                    ).long()
                    self.host.episode_forced_z[forced_idx] = True
                    self.host.episode_forced_z_id[forced_idx] = forced_z

        forced_active = self.host.episode_forced_z.clone()
        proposed_z = z_idx.clone()
        resample_mask = resample_mask & (~forced_active)
        if bool(resample_mask.any().item()):
            idx = torch.where(resample_mask)[0]
            sampled_dist = Categorical(logits=z_logits.index_select(0, idx))
            sampled_z = trainer.model._categorical_argmax_or_sample(
                sampled_dist,
                deterministic=False,
                generator=trainer.model._sampling_gen_strategy,
            )
            
            # Telemetry for event refresh
            if getattr(trainer, "latent_event_refresh_enabled", False):
                event_resampled = trigger_refresh & resample_mask
                if bool(event_resampled.any().item()):
                    self.host.rollout_refresh_count += int(event_resampled.sum().item())
                    self.host.rollout_refresh_reason_enemy_flag += int(trigger_enemy_flag[event_resampled].sum().item())
                    self.host.rollout_refresh_reason_friendly_flag += int(trigger_friendly_flag[event_resampled].sum().item())
                    self.host.rollout_refresh_reason_score_change += int(trigger_score[event_resampled].sum().item())
                    self.host.rollout_refresh_reason_near_base += int(trigger_near_base[event_resampled].sum().item())

                    self.host.refresh_count_this_episode[event_resampled] += 1

            # v3i3 per-refresh capture: stash the (state_at_refresh, prev_z,
            # next_z, event_type, flag_state_bucket, decision_step,
            # return_at_refresh) tuple for every event-driven refresh that
            # actually fired this step. Opponent_id + future_return are filled
            # in on episode-done by ``_finalize_v3i3_refresh_records``. The
            # capture is gated on either of v3i3's two consumer features
            # (preference loss OR per-refresh CSV log) being enabled so
            # disabled runs pay zero overhead.
            v3i3_enabled = bool(
                getattr(trainer, "latent_v3i3_event_preference_enabled", False)
                or getattr(trainer, "latent_v3i3_refresh_log_enabled", False)
            )
            if v3i3_enabled and getattr(trainer, "latent_event_refresh_enabled", False):
                event_resampled = trigger_refresh & resample_mask
                if bool(event_resampled.any().item()):
                    # Primary event type per env when multiple triggers fire on
                    # the same step. Priority left-to-right:
                    #   enemy_flag (0) > friendly_flag (1) > score (2) > near_base (3)
                    event_type_t = torch.full(
                        (curr_gs.shape[0],), -1, dtype=torch.long, device=device
                    )
                    event_type_t = torch.where(
                        trigger_near_base, torch.full_like(event_type_t, 3), event_type_t
                    )
                    event_type_t = torch.where(
                        trigger_score, torch.full_like(event_type_t, 2), event_type_t
                    )
                    event_type_t = torch.where(
                        trigger_friendly_flag, torch.full_like(event_type_t, 1), event_type_t
                    )
                    event_type_t = torch.where(
                        trigger_enemy_flag, torch.full_like(event_type_t, 0), event_type_t
                    )
                    # 2*enemy_carries_our_flag + we_carry_enemy_flag, range 0..3.
                    enemy_has = (curr_gs[:, 10] > 0.5).long()
                    we_have = (curr_gs[:, 11] > 0.5).long()
                    flag_state_t = enemy_has * 2 + we_have

                    carrier_progress_bucket_t = carrier_progress_bucket_ids(curr_gs)

                    # The actual sampled z post-resample for the event-refreshed
                    # envs. ``z_idx`` still holds prev_z at this point in the
                    # method (the bulk ``z_idx[idx] = sampled_z`` happens below);
                    # construct next_z by indexing into sampled_z which is
                    # aligned with ``idx`` rows.
                    idx_to_pos = {int(v.item()): i for i, v in enumerate(idx)}
                    for env_i_t in torch.where(event_resampled)[0]:
                        env_i = int(env_i_t.item())
                        pos = idx_to_pos.get(env_i, None)
                        if pos is None:
                            continue
                        record = {
                            "env_id": env_i,
                            "episode_id": int(self.host.episode_id_per_env[env_i].item()),
                            "decision_step": int(self.host.steps_since_ep_start[env_i].item()),
                            "reason_id": int(event_type_t[env_i].item()),
                            "prev_z": int(prev_z[env_i].item()),
                            "next_z": int(sampled_z[pos].item()),
                            "flag_state_bucket": int(flag_state_t[env_i].item()),
                            "carrier_progress_bucket": int(carrier_progress_bucket_t[env_i].item()),
                            "return_at_refresh": float(
                                self.host.episode_return_accum[env_i].item()
                            ),
                            "refresh_state": global_state[env_i].detach().clone(),
                        }
                        self.host.pending_refresh_records.setdefault(env_i, []).append(record)

            z_idx[idx] = sampled_z
            proposed_z[idx] = sampled_z
            self.host.current_z = z_idx.clone()
            self.host.strategy_age[idx] = 0
            self.host.needs_strategy_sample[idx] = False
            self.host.steps_since_last_refresh[resample_mask] = 0
            from rl.custom_ppo.v6i1_phase_runtime import (
                is_v6i1_staged_trainer,
                resolve_v6i1_exploration_epsilon_current,
            )

            if is_v6i1_staged_trainer(trainer):
                epsilon = float(resolve_v6i1_exploration_epsilon_current(trainer))
                if epsilon > 0.0:
                    router_resample = resample_mask & (~self.host.v6i1_episode_rehearsal)
                    if bool(router_resample.any().item()):
                        ridx = torch.where(router_resample)[0]
                        gen = trainer.model._sampling_gen_strategy
                        rand_kwargs = {"dtype": torch.float32, "device": device}
                        if gen is not None:
                            rand_kwargs["generator"] = gen
                        explore_draw = torch.rand((int(ridx.numel()),), **rand_kwargs)
                        explore_local = explore_draw < epsilon
                        if bool(explore_local.any().item()):
                            explore_idx = ridx[explore_local]
                            uniform_logits = torch.zeros(
                                (int(explore_idx.numel()), trainer.latent_k),
                                dtype=torch.float32,
                                device=device,
                            )
                            uniform_dist = Categorical(logits=uniform_logits)
                            explore_z = trainer.model._categorical_argmax_or_sample(
                                uniform_dist,
                                deterministic=False,
                                generator=trainer.model._sampling_gen_strategy,
                            ).long()
                            z_idx[explore_idx] = explore_z
                            self.host.current_z = z_idx.clone()

        if bool(forced_active.any().item()):
            z_idx[forced_active] = self.host.episode_forced_z_id[forced_active]
            self.host.current_z = z_idx.clone()
            self.host.strategy_age[forced_active] = 0
            self.host.needs_strategy_sample[forced_active] = False
            self.host.steps_since_last_refresh[forced_active] = 0

        sparse_actual_changes = sparse_refresh_accepted & (z_idx != prev_z)
        if bool(sparse_refresh_accepted.any().item()):
            self.host.steps_since_last_tactical_refresh[
                sparse_refresh_accepted
            ] = 0
        if bool(sparse_actual_changes.any().item()):
            dwell_values = self.host.steps_since_z_change[
                sparse_actual_changes
            ].float()
            self.host.rollout_sparse_z_change_count += int(
                sparse_actual_changes.sum().item()
            )
            self.host.rollout_sparse_z_dwell_sum += float(
                dwell_values.sum().item()
            )
            self.host.rollout_sparse_z_dwell_count += int(
                sparse_actual_changes.sum().item()
            )

        actual_z_changes = resample_mask & (z_idx != prev_z)
        if bool(actual_z_changes.any().item()):
            self.host.steps_since_z_change[actual_z_changes] = 0
        committed_now = commit_now & resample_mask
        if bool(committed_now.any().item()):
            self.host.steps_since_z_change[committed_now] = 0
            self.host.steps_since_last_tactical_refresh[committed_now] = 0

        # Same-z sparse proposals are refreshes but not switches. Persistence
        # pressure and GAE boundaries only apply to actual executed z changes.
        persist_mask = (
            persist_mask & (~sparse_refresh_accepted)
        ) | sparse_actual_changes

        # Check actual z changes for event-refreshed envs
        if getattr(trainer, "latent_event_refresh_enabled", False):
            event_resampled = trigger_refresh & resample_mask
            if bool(event_resampled.any().item()):
                actual_changes = (z_idx != prev_z) & event_resampled
                self.host.rollout_refresh_z_changed_count += int(actual_changes.sum().item())
                
                # Track transitions
                for env_idx in torch.where(event_resampled)[0]:
                    pz_val = int(prev_z[env_idx].item())
                    nz_val = int(z_idx[env_idx].item())
                    latent_k = int(trainer.latent_k)
                    if 0 <= pz_val < latent_k and 0 <= nz_val < latent_k:
                        self.host.rollout_refresh_transitions[pz_val, nz_val] += 1.0

        from rl.custom_ppo.v6i1_phase_runtime import (
            is_v6i1_staged_trainer,
            resolve_v6i1_exploration_epsilon_current,
        )

        epsilon = 0.0
        if is_v6i1_staged_trainer(trainer):
            epsilon = float(resolve_v6i1_exploration_epsilon_current(trainer))
        router_probs = torch.softmax(z_logits.detach(), dim=-1)
        behavior_probs = router_probs.clone()
        if epsilon > 0.0:
            behavior_probs = epsilon_behavior_probs(
                router_probs, epsilon=epsilon, latent_k=int(trainer.latent_k)
            )
        if bool(forced_active.any().item()):
            behavior_probs[forced_active] = 1.0 / float(max(1, int(trainer.latent_k)))
        behavior_log_prob = behavior_log_prob_from_probs(behavior_probs, z_idx)
        router_log_prob = z_dist.log_prob(z_idx)
        z_log_prob = behavior_log_prob
        z_entropy = z_dist.entropy()

        # v3i19 arc-credit lifecycle hook. Every z-sample boundary (episode
        # start, sparse resample, event refresh, warmup commit) is treated as
        # the end of the previous arc and the start of a new one:
        #
        # * ``arc_finalize`` pushes the previous arc's (ctx, z, log_prob,
        #   arc_return, arc_length) snapshot into the rollout buffer if it's
        #   above ``latent_arc_credit_min_len``. Envs at episode-start have
        #   no open arc (cleared on episode-done) so this is a per-env no-op
        #   for them.
        # * ``arc_open`` snapshots the new arc's start state.
        #
        # Both are no-ops when ``latent_arc_credit_enabled`` is False, so legacy
        # presets pay zero overhead here.
        if bool(resample_mask.any().item()):
            self.host.arc_finalize(resample_mask, reason="z_change")
            self.host.arc_open(
                resample_mask,
                global_state=global_state,
                z_idx=z_idx,
                z_log_prob=z_log_prob,
                selector_hidden=selector_hidden_pre,
            )
        macro_boundary_mask = resample_mask & (~forced_active)
        if bool(macro_boundary_mask.any().item()):
            self.host.macro_finalize(macro_boundary_mask, reason="boundary")
            self.host.macro_open(
                macro_boundary_mask,
                global_state=global_state,
                z_idx=z_idx,
                z_log_prob=z_log_prob,
                selector_hidden=selector_hidden_pre,
            )
        if bool((resample_mask & (~forced_active)).any().item()):
            if self.host.selector_hidden is not None:
                _, h_new = trainer.model._forward_q_phi(global_state, self.host.selector_hidden)
                self.host.selector_hidden = h_new.detach()

        # Snapshot the q_phi training (state, z, log_prob) pair:
        # - warmup == 0: legacy behavior, snapshot at episode start (step 0)
        # - warmup  > 0: snapshot at the commit step, after the EMA window
        if warmup > 0:
            snapshot_mask = commit_now
        else:
            snapshot_mask = episode_start_mask
        snapshot_mask = snapshot_mask & (~forced_active)

        # Track the warmup bookkeeping.
        if bool(snapshot_mask.any().item()):
            self.host.episode_strategy_committed |= snapshot_mask
            self.host.first_z_sample_step = torch.where(
                snapshot_mask,
                self.host.steps_since_ep_start,
                self.host.first_z_sample_step,
            )
            if warmup > 0:
                self.host.episode_return_baseline_at_commit = torch.where(
                    snapshot_mask,
                    self.host.episode_return_accum,
                    self.host.episode_return_baseline_at_commit,
                )

        if getattr(trainer, "latent_sparse_tactical_refresh_enabled", False):
            agreement_mask = (
                self.host.episode_strategy_committed & (~forced_active)
            )
            if bool(agreement_mask.any().item()):
                q_phi_argmax = torch.argmax(z_logits, dim=-1)
                self.host.rollout_q_phi_argmax_executed_agree_count += int(
                    (q_phi_argmax[agreement_mask] == z_idx[agreement_mask])
                    .sum()
                    .item()
                )
                self.host.rollout_q_phi_argmax_executed_total += int(
                    agreement_mask.sum().item()
                )
        self.host.store_episode_strategy_start(
            start_mask=snapshot_mask,
            global_state=global_state,
            router_action=RouterAction(
                proposed_z=proposed_z,
                executed_z=z_idx,
                router_probs=router_probs,
                behavior_probs=behavior_probs,
                behavior_log_prob=behavior_log_prob,
                router_log_prob=router_log_prob,
                source=RouterActionSource.ROUTER,
            ),
            action_sources=resolve_action_sources(
                forced_mask=forced_active,
                rehearsal_mask=self.host.v6i1_episode_rehearsal,
                epsilon_override_mask=proposed_z != z_idx,
                event_refresh_mask=trigger_refresh & resample_mask,
                batch_size=int(z_idx.shape[0]),
                device=device,
            ),
            selector_hidden=selector_hidden_pre,
            z_logits=z_logits,
        )

        # Exclude step 0 from q_phi PPO training when warmup is active.
        # z_resampled means "eligible for q_phi training", not merely "sampled a latent"
        training_resample_mask = resample_mask.clone()
        if warmup > 0:
            training_resample_mask = training_resample_mask & (~episode_start_mask)
        training_resample_mask = training_resample_mask & (~forced_active)

        self.host.prev_global_state = curr_gs.clone()

        aux = {
            "z": z_idx,
            "prev_z": prev_z,
            "z_log_prob": z_log_prob,
            "z_entropy": z_entropy,
            "z_logits": z_logits,
            "z_resampled": training_resample_mask,
            "z_resampled_actual": resample_mask,
            "z_persist_mask": persist_mask,
            "z_forced": forced_active,
        }
        if selector_hidden_pre is not None:
            aux["selector_hidden"] = selector_hidden_pre.detach().clone()
        return z_idx, prev_z, aux

    def record_tactical_context_step(self, global_state: torch.Tensor) -> None:
        """Accumulate detached tactical occupancy for each active episode."""
        from rl.custom_ppo.latent.context_buckets import tactical_local_context_keys

        if global_state.dim() != 2:
            return
        keys = tactical_local_context_keys(global_state).detach().long()
        env_ids = torch.arange(int(keys.shape[0]), dtype=torch.long, device=keys.device)
        self.host.episode_tactical_bucket_counts[env_ids, keys] += 1

    def mark_strategy_step_done(self, dones: np.ndarray) -> None:
        """Advance per-env step counter; reset on env-level done."""
        trainer = self.host.trainer
        if not trainer.use_latent_strategy:
            return
        done_t = torch.as_tensor(dones, dtype=torch.bool, device=trainer.device)
        self.host.strategy_age += 1
        self.host.steps_since_ep_start += 1
        self.host.steps_since_last_refresh += 1
        self.host.steps_since_last_tactical_refresh += 1
        self.host.steps_since_z_change += 1
        if bool(done_t.any().item()):
            self.host.reset_completed_envs(done_t)
            self.host.strategy_age[done_t] = 0
            self.host.needs_strategy_sample[done_t] = not trainer.fixed_latent_strategy
            self.host.steps_since_ep_start[done_t] = 0
            self.host.episode_strategy_committed[done_t] = False
            self.host.episode_tactical_bucket_counts[done_t] = 0
            self.host.first_z_sample_step[done_t] = -1
            self.host.episode_return_baseline_at_commit[done_t] = 0.0
            self.host.episode_forced_z[done_t] = False
            self.host.episode_forced_z_id[done_t] = 0
            self.host.episode_contrast_bucket[done_t] = 0
            self.host.episode_behavior_sum[done_t] = 0.0
            self.host.episode_behavior_count[done_t] = 0
            self.host.steps_since_last_refresh[done_t] = 0
            self.host.refresh_count_this_episode[done_t] = 0
            self.host.steps_since_last_tactical_refresh[done_t] = 0
            self.host.steps_since_z_change[done_t] = 0
            if self.host.prev_global_state is not None:
                self.host.prev_global_state[done_t] = 0.0
            self.host.episode_id_per_env[done_t] += 1
            # Defensive: drop any v3i3 pending refresh records that weren't
            # finalized by ``_finalize_v3i3_refresh_records`` (shouldn't
            # happen in the normal rollout flow, but avoids leaking state
            # into the next episode if a caller forgets to wire the hook).
            for env_i, done_i in enumerate(dones):
                if bool(done_i) and self.host.pending_refresh_records.get(env_i):
                    self.host.pending_refresh_records[env_i] = []


    def representative_tactical_bucket(self, env_index: int) -> int:
        """Return the dominant meaningful tactical context for one episode."""
        counts = self.host.episode_tactical_bucket_counts[int(env_index)]
        if int(counts.sum().item()) <= 0:
            contrast_bucket = int(
                self.host.episode_contrast_bucket[int(env_index)].item()
            )
            if contrast_bucket != 0:
                return contrast_bucket
            strategy_bucket = int(
                self.host.episode_strategy_bucket[int(env_index)].item()
            )
            if strategy_bucket != 0:
                return strategy_bucket
            # Neutral phase, no flags taken, tied score is local bucket 1.
            return 1

        candidates = counts.clone()
        # phase=0, flags=(0,0), score=tied encodes to local key 1. Prefer a
        # context where something tactical happened whenever one exists.
        if int(candidates.sum().item() - candidates[1].item()) > 0:
            candidates[1] = 0
        return int(torch.argmax(candidates).detach().cpu().item())


