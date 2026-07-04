"""Arc-level router consequence credit (v3i19)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from rl.custom_ppo.latent.context_buckets import strategy_experience_bucket_ids
from rl.custom_ppo.latent.optimization.router_ppo import RouterPPOEngine
from rl.custom_ppo.latent.optimization.router_registry import LatentOptimizerRegistry
from rl.custom_ppo.latent.records import stack_selector_hidden_records
from rl.custom_ppo.latent.types import RouterPPOBatch, RouterPPOConfig

if TYPE_CHECKING:
    from rl.custom_ppo.latent.state import LatentStrategyState


class ArcCreditManager:
    """Per-arc accumulate / finalize / PPO."""

    def __init__(self, host: LatentStrategyState) -> None:
        self.host = host
        self._engine: RouterPPOEngine | None = None

    def _engine_for_host(self) -> RouterPPOEngine:
        if self._engine is None:
            registry = LatentOptimizerRegistry.from_trainer(self.host.trainer)
            self._engine = RouterPPOEngine(trainer=self.host.trainer, registry=registry)
        return self._engine

    def reset_arc_credit_rollout_state(self) -> None:
        """Drop the rollout's arc-credit buffer + telemetry counters.

        Does NOT touch the per-env open-arc state (``arc_open_*``,
        ``arc_return_accum``, ``arc_steps_accum``, ``arc_has_open``) because
        those reflect in-flight arcs that span the rollout boundary; only the
        finalized-record buffer + rollout-level counters are drained.
        """
        self.host.rollout_strategy_arc_records = []
        self.host.rollout_arc_finalized_count = 0
        self.host.rollout_arc_dropped_short_count = 0
        self.host.rollout_arc_length_sum = 0
        self.host.rollout_arc_return_sum = 0.0


    def arc_accumulate_step(self, rewards: torch.Tensor) -> None:
        """Add the env's per-step team-mean reward to each open arc's return.

        Called once per env step from the rollout loop, AFTER the env has
        stepped and rewards are available, BEFORE any z-resample for the
        next step. ``rewards`` may be shape ``(n_envs,)`` (already team-
        reduced) or ``(n_envs, n_agents)``; we coerce to ``(n_envs,)``.

        No-op when arc credit is disabled to keep zero overhead in legacy
        presets.
        """
        trainer = self.host.trainer
        if not getattr(trainer, "latent_arc_credit_enabled", False):
            return
        r = rewards.detach()
        if r.dim() > 1:
            r = r.mean(dim=tuple(range(1, r.dim())))
        # Only accumulate for envs that have an arc currently open.
        active = self.host.arc_has_open
        if not bool(active.any().item()):
            return
        self.host.arc_return_accum = torch.where(
            active, self.host.arc_return_accum + r.to(self.host.arc_return_accum), self.host.arc_return_accum
        )
        self.host.arc_steps_accum = torch.where(
            active, self.host.arc_steps_accum + 1, self.host.arc_steps_accum
        )


    def arc_finalize(
        self,
        finalize_mask: torch.Tensor,
        *,
        opponent_ids: Optional[torch.Tensor] = None,
        reason: str = "z_change",
    ) -> int:
        """Push open arcs into the rollout's arc record buffer.

        ``finalize_mask`` selects envs whose currently-open arc has just ended
        (because z is about to change OR because the episode terminated).
        Arcs shorter than ``latent_arc_credit_min_len`` are dropped from the
        PPO training buffer (still counted in the dropped-short telemetry).

        Returns the number of arcs pushed to the buffer.
        """
        trainer = self.host.trainer
        if not getattr(trainer, "latent_arc_credit_enabled", False):
            return 0
        eligible = finalize_mask & self.host.arc_has_open
        if not bool(eligible.any().item()):
            return 0
        idx = torch.where(eligible)[0]
        min_len = max(1, int(getattr(trainer, "latent_arc_credit_min_len", 32) or 1))
        pushed = 0
        for env_i in idx.detach().cpu().tolist():
            env_i = int(env_i)
            steps = int(self.host.arc_steps_accum[env_i].detach().cpu().item())
            arc_return = float(self.host.arc_return_accum[env_i].detach().cpu().item())
            self.host.rollout_arc_finalized_count += 1
            self.host.rollout_arc_length_sum += steps
            self.host.rollout_arc_return_sum += arc_return
            # Running-mean EMA over arc returns (used by the running_mean
            # baseline). Computed *before* the drop-short check so even
            # short arcs inform the EMA's drift.
            self.host.arc_return_running_count += 1
            alpha = 1.0 / float(self.host.arc_return_running_count)
            self.host.arc_return_running_mean += alpha * (arc_return - self.host.arc_return_running_mean)
            if steps < min_len:
                self.host.rollout_arc_dropped_short_count += 1
                continue
            rec_opp = (
                int(opponent_ids[env_i].detach().cpu().item())
                if opponent_ids is not None
                else int(self.host.arc_open_opponent_id[env_i].detach().cpu().item())
            )
            # Stable, monotonic arc id so downstream replay buffers can reject
            # duplicate insertions by identity (not by content hash, which two
            # legitimate episodes can occasionally collide on). The counter is
            # NOT reset per rollout — it is unique for the lifetime of this
            # LatentStrategyState — so ``arc_uid`` alone uniquely identifies a
            # finalized arc.
            arc_uid = int(getattr(self.host, "_arc_record_uid_counter", 0))
            self.host._arc_record_uid_counter = arc_uid + 1
            rec = {
                    "global_state_0": self.host.arc_open_ctx[env_i].detach().clone().cpu(),
                    "z": int(self.host.arc_open_z[env_i].detach().cpu().item()),
                    "z_logprob_old": float(self.host.arc_open_log_prob[env_i].detach().cpu().item()),
                    "arc_return": arc_return,
                    "arc_length": steps,
                    "opponent_id": rec_opp,
                    "bucket_id": int(self.host.arc_open_bucket_id[env_i].detach().cpu().item()),
                    "reason": str(reason),
                    "env_index": env_i,
                    "arc_uid": arc_uid,
                    "commit_step": int(self.host.arc_open_commit_step[env_i].detach().cpu().item()),
                }
            if str(getattr(trainer.cfg, "router_opening_context_mode", "") or ""):
                rec["opening_context"] = (
                    self.host.arc_open_opening_context[env_i].detach().clone().cpu()
                )
            if self.host.arc_open_selector_hidden is not None:
                rec["selector_hidden_0"] = (
                    self.host.arc_open_selector_hidden[env_i].detach().clone().cpu()
                )
            self.host.rollout_strategy_arc_records.append(rec)
            pushed += 1
        # Mark the open-arc slot as closed for these envs. The arc_open_*
        # snapshots remain in place but ``arc_has_open`` going False means
        # subsequent ``arc_accumulate_step`` calls skip them until a new arc
        # is opened.
        self.host.arc_has_open[idx] = False
        self.host.arc_return_accum[idx] = 0.0
        self.host.arc_steps_accum[idx] = 0
        return pushed


    def arc_open(
        self,
        open_mask: torch.Tensor,
        *,
        global_state: torch.Tensor,
        z_idx: torch.Tensor,
        z_log_prob: torch.Tensor,
        opponent_ids: Optional[torch.Tensor] = None,
        selector_hidden: torch.Tensor | None = None,
    ) -> int:
        """Snapshot a new arc start for each env in ``open_mask``.

        Must be called AFTER ``arc_finalize`` for the same envs (so the
        previous arc is already pushed before we overwrite the snapshot).
        Pairs with ``arc_finalize`` to form the per-arc lifecycle:
        finalize old -> open new on every z-resample or episode-start step.
        """
        trainer = self.host.trainer
        if not getattr(trainer, "latent_arc_credit_enabled", False):
            return 0
        if not bool(open_mask.any().item()):
            return 0
        idx = torch.where(open_mask)[0]
        gs = global_state.index_select(0, idx).detach()
        # Ensure stored ctx matches arc_open_ctx's expected dim. The model's
        # global_state_dim is the post-temporal-stacking width that q_phi
        # actually consumes; the env state passed in may be the same shape
        # (it is during normal MAPPO loops).
        target_dim = int(self.host.arc_open_ctx.shape[1])
        if gs.shape[1] >= target_dim:
            gs = gs[:, :target_dim]
        else:
            pad = torch.zeros(
                (gs.shape[0], target_dim - gs.shape[1]),
                dtype=gs.dtype,
                device=gs.device,
            )
            gs = torch.cat([gs, pad], dim=1)
        self.host.arc_open_ctx[idx] = gs
        self.host.arc_open_z[idx] = z_idx.index_select(0, idx).detach().long()
        self.host.arc_open_log_prob[idx] = z_log_prob.index_select(0, idx).detach().float()
        if selector_hidden is not None and self.host.arc_open_selector_hidden is not None:
            self.host.arc_open_selector_hidden[idx] = selector_hidden.index_select(0, idx).detach()
        buckets = strategy_experience_bucket_ids(gs).detach()
        self.host.arc_open_bucket_id[idx] = buckets
        self.host.arc_open_commit_step[idx] = self.host.steps_since_ep_start.index_select(0, idx).detach().long()
        mode = str(getattr(trainer.cfg, "router_opening_context_mode", "") or "").strip().lower()
        if mode in {"initial_commit_delta", "state0_commit_delta", "opening_summary"}:
            init = self.host.episode_initial_global_state.index_select(0, idx).detach()
            if init.shape[1] >= target_dim:
                init = init[:, :target_dim]
            elif init.shape[1] < target_dim:
                pad = torch.zeros(
                    (init.shape[0], target_dim - init.shape[1]),
                    dtype=init.dtype,
                    device=init.device,
                )
                init = torch.cat([init, pad], dim=1)
            opening = torch.cat([init, gs, gs - init], dim=1)
            self.host.arc_open_opening_context[idx] = opening
        elif mode:
            raise ValueError(
                f"Unknown router_opening_context_mode {mode!r}; expected "
                "'initial_commit_delta' or empty string"
            )
        if opponent_ids is not None:
            self.host.arc_open_opponent_id[idx] = opponent_ids.index_select(0, idx).detach().long()
        self.host.arc_has_open[idx] = True
        self.host.arc_return_accum[idx] = 0.0
        self.host.arc_steps_accum[idx] = 0
        return int(idx.numel())


    @staticmethod
    def empty_arc_strategy_stats() -> dict[str, float]:
        """Default arc-credit telemetry slot (zeroed when arc credit is off).

        The ``q_phi_*`` and ``latent_arc_grad_norm`` fields are the smoke
        alarm for v3i19: if ``latent_arc_credit_coef`` is 1.0 but
        ``q_phi_grad_norm`` stays tiny, the consequence channel is decorative
        and v3i19 will collapse to v3i18 silently. Logged unconditionally so
        every rollout's CSV exposes the gradient-flow check.
        """
        return {
            "latent_arc_count": 0.0,
            "latent_arc_finalized_count": 0.0,
            "latent_arc_dropped_short_count": 0.0,
            "latent_arc_mean_length": 0.0,
            "latent_arc_mean_return": 0.0,
            "latent_arc_advantage_mean": 0.0,
            "latent_arc_advantage_std": 0.0,
            "latent_arc_baseline_mean": 0.0,
            "latent_arc_raw_advantage_mean": 0.0,
            "latent_arc_raw_advantage_std": 0.0,
            "latent_arc_positive_fraction": 0.0,
            **{f"latent_arc_raw_adv_mean_z{_zi}": 0.0 for _zi in range(4)},
            **{f"latent_arc_count_z{_zi}": 0.0 for _zi in range(4)},
            # Separation of per-z raw advantage means (max - min over z's that
            # received >=1 arc). A centered signal with zero spread gives every
            # z the same expected credit and CANNOT teach routing, no matter how
            # healthy the aggregate positive_fraction looks.
            "latent_arc_raw_adv_z_spread": 0.0,
            "latent_arc_running_mean_count": 0.0,
            "latent_arc_running_mean_value": 0.0,
            "latent_arc_policy_loss": 0.0,
            "latent_arc_value_loss": 0.0,
            "latent_arc_clipfrac": 0.0,
            "latent_arc_approx_kl": 0.0,
            "latent_arc_credit_coef": 0.0,
            "latent_arc_grad_norm": 0.0,
            "q_phi_grad_norm": 0.0,
            # Split grad-norm diagnostic. Sanity check:
            # sqrt(strategy_encoder_grad_norm^2 + value_head_grad_norm^2 +
            #      other_grad_norm^2) approx q_phi_grad_norm.
            # If strategy_encoder portion stays tiny while value_head
            # dominates, arc credit is training only the baseline and the
            # router will stay glued to uniform.
            "q_phi_strategy_encoder_grad_norm": 0.0,
            "q_phi_value_head_grad_norm": 0.0,
            "q_phi_other_grad_norm": 0.0,
            "q_phi_entropy": 0.0,
            "q_phi_mean_max_prob": 0.0,
        }


    def apply_arc_strategy_ppo(self) -> dict[str, float]:
        """Per-arc PPO update on q_phi (v3i19 Summer-faithful consequence credit).

        Mirrors ``apply_episode_strategy_ppo`` but operates on per-arc records
        (one per z-arc, not one per episode). Each arc contributes its own
        (ctx_at_arc_start, z, log_prob, arc_return) tuple. The advantage is
        ``arc_return - baseline``, where baseline is either ``V_phi(ctx, z)``
        (context_value mode, reusing the existing strategy value head) or a
        detached EMA of arc returns (running_mean mode, no V dependency).

        Plan-faithful: arc_return is summed env reward over the arc only.
        No labels, no semantic heads, no opponent ID seen by q_phi or the
        value head. The optimizer used is the same dedicated router
        optimizer (or shared optimizer) as ``apply_episode_strategy_ppo``.
        """
        trainer = self.host.trainer
        stats = ArcCreditManager.empty_arc_strategy_stats()
        stats["latent_arc_finalized_count"] = float(self.host.rollout_arc_finalized_count)
        stats["latent_arc_dropped_short_count"] = float(self.host.rollout_arc_dropped_short_count)
        if self.host.rollout_arc_finalized_count > 0:
            stats["latent_arc_mean_length"] = float(
                self.host.rollout_arc_length_sum / self.host.rollout_arc_finalized_count
            )
            stats["latent_arc_mean_return"] = float(
                self.host.rollout_arc_return_sum / self.host.rollout_arc_finalized_count
            )

        if not getattr(trainer, "latent_arc_credit_enabled", False) or trainer.fixed_latent_strategy:
            return stats
        records = list(self.host.rollout_strategy_arc_records)
        if not records:
            return stats
        device = trainer.device
        states = torch.stack(
            [r["global_state_0"].detach().float() for r in records], dim=0
        ).to(device)
        z = torch.as_tensor([int(r["z"]) for r in records], dtype=torch.long, device=device)
        old_log_prob = torch.as_tensor(
            [float(r["z_logprob_old"]) for r in records], dtype=torch.float32, device=device
        )
        arc_returns = torch.as_tensor(
            [float(r["arc_return"]) for r in records], dtype=torch.float32, device=device
        )
        selector_hidden = stack_selector_hidden_records(records, device=device)
        stats["latent_arc_count"] = float(arc_returns.numel())

        baseline_mode = str(
            getattr(trainer, "latent_arc_credit_baseline", "context_value")
            or "context_value"
        ).lower()
        coef = max(0.0, float(getattr(trainer, "latent_arc_credit_coef", 1.0) or 0.0))
        if coef <= 0.0:
            return stats
        n_epochs = max(1, int(getattr(trainer, "latent_arc_credit_n_epochs", 4) or 1))
        clip_eps = max(1e-6, float(getattr(trainer, "latent_arc_credit_clip_eps", 0.2) or 0.2))
        return_norm = bool(getattr(trainer, "latent_arc_credit_return_norm", True))
        value_coef = max(
            0.0, float(getattr(trainer, "latent_episode_strategy_value_coef", 0.5) or 0.0)
        )

        encoder_params = self.host._strategy_encoder_params()
        value_head_params = self.host._value_head_params()
        stats["latent_arc_credit_coef"] = coef

        with torch.no_grad():
            if baseline_mode == "running_mean":
                fixed_baseline = torch.full_like(
                    arc_returns, float(self.host.arc_return_running_mean)
                )
            elif trainer.model.episode_strategy_value_head is None:
                fixed_baseline = torch.zeros_like(arc_returns)
            else:
                fixed_baseline = trainer.model.episode_strategy_value(
                    states, z, selector_hidden=selector_hidden
                ).detach()
            raw_adv = (arc_returns - fixed_baseline).detach()
            # Raw (pre-normalization) advantage diagnostics: these expose
            # whether the ORIGINAL sparse signal has real spread before the
            # per-batch standardization scrubs the scale. Aggressive
            # normalization can make a nearly-flat signal look healthy.
            stats["latent_arc_baseline_mean"] = float(fixed_baseline.mean().item())
            stats["latent_arc_raw_advantage_mean"] = float(raw_adv.mean().item())
            stats["latent_arc_raw_advantage_std"] = (
                float(raw_adv.std(unbiased=False).item()) if raw_adv.numel() > 1 else 0.0
            )
            stats["latent_arc_positive_fraction"] = float((raw_adv > 0).float().mean().item())
            stats["latent_arc_running_mean_count"] = float(self.host.arc_return_running_count)
            stats["latent_arc_running_mean_value"] = float(self.host.arc_return_running_mean)
            _K = int(getattr(trainer, "latent_k", 4) or 4)
            _z_means: list[float] = []
            for _zi in range(_K):
                _m = z == _zi
                _zmean = float(raw_adv[_m].mean().item()) if _m.any() else float("nan")
                stats[f"latent_arc_raw_adv_mean_z{_zi}"] = _zmean
                stats[f"latent_arc_count_z{_zi}"] = float(_m.sum().item())
                if _m.any():
                    _z_means.append(_zmean)
            stats["latent_arc_raw_adv_z_spread"] = (
                float(max(_z_means) - min(_z_means)) if len(_z_means) >= 2 else 0.0
            )
            fixed_adv = raw_adv.clone()
            if return_norm and fixed_adv.numel() > 1:
                fixed_adv = (fixed_adv - fixed_adv.mean()) / (fixed_adv.std(unbiased=False) + 1e-8)
            fixed_adv = fixed_adv.detach()

        router_opt = getattr(trainer, "latent_router_optimizer", None)
        fallback_opt = None if router_opt is not None else trainer.optimizer
        registry = LatentOptimizerRegistry.from_trainer(trainer) if router_opt is not None else None
        engine = RouterPPOEngine(
            trainer=trainer, registry=registry, fallback_optimizer=fallback_opt
        )

        def value_fn(st, z_t, hidden):
            if baseline_mode == "running_mean":
                return fixed_baseline
            if trainer.model.episode_strategy_value_head is None:
                return torch.zeros_like(arc_returns)
            return trainer.model.episode_strategy_value(st, z_t, selector_hidden=hidden)

        effective_value_coef = (
            value_coef
            if baseline_mode == "context_value"
            and trainer.model.episode_strategy_value_head is not None
            else 0.0
        )
        batch = RouterPPOBatch(
            states=states,
            executed_z=z,
            old_behavior_log_prob=old_log_prob,
            fixed_advantages=fixed_adv,
            returns=arc_returns,
            selector_hidden=selector_hidden,
        )
        config = RouterPPOConfig(
            coef=coef,
            value_coef=effective_value_coef,
            clip_epsilon=clip_eps,
            epochs=n_epochs,
            normalize_advantages=return_norm,
        )
        ppo_stats, step_results = engine.run(
            batch,
            config=config,
            value_fn=value_fn,
            param_groups=[encoder_params, value_head_params],
            grad_split_groups={
                "encoder": encoder_params,
                "value_head": value_head_params,
            },
            collect_router_shape=True,
        )
        if ppo_stats:
            stats["latent_arc_advantage_mean"] = float(ppo_stats.get("advantage_mean", 0.0))
            stats["latent_arc_advantage_std"] = float(ppo_stats.get("advantage_std", 0.0))
            stats["latent_arc_policy_loss"] = float(ppo_stats.get("policy_loss", 0.0))
            stats["latent_arc_value_loss"] = float(ppo_stats.get("value_loss", 0.0))
            stats["latent_arc_clipfrac"] = float(ppo_stats.get("clipfrac", 0.0))
            stats["latent_arc_approx_kl"] = float(ppo_stats.get("approx_kl", 0.0))
            grad_norms = [float(s.grad_norm) for s in step_results if s.stepped]
            if grad_norms:
                grad_norm_mean = float(sum(grad_norms) / len(grad_norms))
                stats["latent_arc_grad_norm"] = grad_norm_mean
                stats["q_phi_grad_norm"] = grad_norm_mean
            enc_norms = [
                float(s.grad_splits["encoder"])
                for s in step_results
                if s.stepped and s.grad_splits and "encoder" in s.grad_splits
            ]
            vh_norms = [
                float(s.grad_splits["value_head"])
                for s in step_results
                if s.stepped and s.grad_splits and "value_head" in s.grad_splits
            ]
            if enc_norms:
                stats["q_phi_strategy_encoder_grad_norm"] = float(sum(enc_norms) / len(enc_norms))
            if vh_norms:
                stats["q_phi_value_head_grad_norm"] = float(sum(vh_norms) / len(vh_norms))
            if (
                grad_norms
                and enc_norms
                and vh_norms
                and getattr(trainer, "latent_router_optimizer", None) is not None
            ):
                other = [
                    max(
                        0.0,
                        float(g) ** 2 - float(e) ** 2 - float(v) ** 2,
                    )
                    ** 0.5
                    for g, e, v in zip(grad_norms, enc_norms, vh_norms)
                ]
                stats["q_phi_other_grad_norm"] = float(sum(other) / len(other))
            elif grad_norms and enc_norms and vh_norms:
                stats["q_phi_other_grad_norm"] = 0.0
            entropies = [float(s.q_phi_entropy) for s in step_results if s.stepped]
            max_probs = [float(s.q_phi_mean_max_prob) for s in step_results if s.stepped]
            if entropies:
                stats["q_phi_entropy"] = float(sum(entropies) / len(entropies))
            if max_probs:
                stats["q_phi_mean_max_prob"] = float(sum(max_probs) / len(max_probs))
        return stats


