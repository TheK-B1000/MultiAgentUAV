"""Macro-segment router credit (V6I1 Phase B/C)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from rl.custom_ppo.latent.optimization.router_ppo import RouterPPOEngine
from rl.custom_ppo.latent.optimization.router_registry import LatentOptimizerRegistry
from rl.custom_ppo.latent.records import stack_selector_hidden_records
from rl.custom_ppo.latent.types import RouterPPOBatch, RouterPPOConfig

if TYPE_CHECKING:
    from rl.custom_ppo.latent.state import LatentStrategyState


class MacroCreditManager:
    """Macro boundary accumulate / finalize / PPO via RouterPPOEngine."""

    def __init__(self, host: LatentStrategyState) -> None:
        self.host = host
        self._engine: RouterPPOEngine | None = None

    def _engine_for_host(self) -> RouterPPOEngine:
        if self._engine is None:
            registry = LatentOptimizerRegistry.from_trainer(self.host.trainer)
            self._engine = RouterPPOEngine(trainer=self.host.trainer, registry=registry)
        return self._engine

    def reset_macro_rollout_state(self) -> None:
        self.host.rollout_strategy_macro_records = []
        self.host.rollout_macro_finalized_count = 0
        self.host.rollout_macro_dropped_short_count = 0
        self.host.rollout_macro_length_sum = 0
        self.host.rollout_macro_return_sum = 0.0


    def macro_accumulate_step(self, rewards: torch.Tensor) -> None:
        if not self.host._v6i1_macro_enabled():
            return
        r = rewards.detach()
        if r.dim() > 1:
            r = r.mean(dim=tuple(range(1, r.dim())))
        active = self.host.macro_has_open
        if not bool(active.any().item()):
            return
        self.host.macro_return_accum = torch.where(
            active, self.host.macro_return_accum + r.to(self.host.macro_return_accum), self.host.macro_return_accum
        )
        self.host.macro_steps_accum = torch.where(
            active, self.host.macro_steps_accum + 1, self.host.macro_steps_accum
        )


    def macro_finalize(self, finalize_mask: torch.Tensor, *, reason: str = "boundary") -> int:
        if not self.host._v6i1_macro_enabled():
            return 0
        eligible = finalize_mask & self.host.macro_has_open
        if not bool(eligible.any().item()):
            return 0
        idx = torch.where(eligible)[0]
        pushed = 0
        for env_i in idx.detach().cpu().tolist():
            env_i = int(env_i)
            steps = int(self.host.macro_steps_accum[env_i].detach().cpu().item())
            macro_return = float(self.host.macro_return_accum[env_i].detach().cpu().item())
            self.host.rollout_macro_finalized_count += 1
            self.host.rollout_macro_length_sum += steps
            self.host.rollout_macro_return_sum += macro_return
            self.host.macro_return_running_count += 1
            alpha = 1.0 / float(self.host.macro_return_running_count)
            self.host.macro_return_running_mean += alpha * (macro_return - self.host.macro_return_running_mean)
            if steps < 1:
                self.host.rollout_macro_dropped_short_count += 1
                continue
            rec = {
                    "global_state_0": self.host.macro_open_ctx[env_i].detach().clone().cpu(),
                    "z": int(self.host.macro_open_z[env_i].detach().cpu().item()),
                    "z_logprob_old": float(self.host.macro_open_log_prob[env_i].detach().cpu().item()),
                    "macro_return": macro_return,
                    "macro_length": steps,
                    "reason": str(reason),
                }
            if self.host.macro_open_selector_hidden is not None:
                rec["selector_hidden_0"] = (
                    self.host.macro_open_selector_hidden[env_i].detach().clone().cpu()
                )
            self.host.rollout_strategy_macro_records.append(rec)
            pushed += 1
        self.host.macro_has_open[idx] = False
        self.host.macro_return_accum[idx] = 0.0
        self.host.macro_steps_accum[idx] = 0
        return pushed


    def macro_open(
        self,
        open_mask: torch.Tensor,
        *,
        global_state: torch.Tensor,
        z_idx: torch.Tensor,
        z_log_prob: torch.Tensor,
        selector_hidden: torch.Tensor | None = None,
    ) -> int:
        if not self.host._v6i1_macro_enabled():
            return 0
        if not bool(open_mask.any().item()):
            return 0
        idx = torch.where(open_mask)[0]
        gs = global_state.index_select(0, idx).detach()
        target_dim = int(self.host.macro_open_ctx.shape[1])
        if gs.shape[1] >= target_dim:
            gs = gs[:, :target_dim]
        else:
            pad = torch.zeros(
                (gs.shape[0], target_dim - gs.shape[1]),
                dtype=gs.dtype,
                device=gs.device,
            )
            gs = torch.cat([gs, pad], dim=1)
        self.host.macro_open_ctx[idx] = gs
        self.host.macro_open_z[idx] = z_idx.index_select(0, idx).detach().long()
        self.host.macro_open_log_prob[idx] = z_log_prob.index_select(0, idx).detach().float()
        if selector_hidden is not None and self.host.macro_open_selector_hidden is not None:
            self.host.macro_open_selector_hidden[idx] = selector_hidden.index_select(0, idx).detach()
        self.host.macro_has_open[idx] = True
        return int(idx.numel())


    @staticmethod
    def empty_macro_strategy_stats() -> dict[str, float]:
        return {
            "latent_macro_finalized_count": 0.0,
            "latent_macro_dropped_short_count": 0.0,
            "latent_macro_mean_length": 0.0,
            "latent_macro_mean_return": 0.0,
            "latent_macro_count": 0.0,
            "latent_macro_advantage_mean": 0.0,
            "latent_macro_advantage_std": 0.0,
            "latent_macro_policy_loss": 0.0,
            "latent_macro_value_loss": 0.0,
            "latent_macro_clipfrac": 0.0,
            "latent_macro_approx_kl": 0.0,
            "latent_macro_grad_norm": 0.0,
            "latent_macro_credit_coef": 0.0,
        }


    def apply_macro_strategy_ppo(self) -> dict[str, float]:
        """Per-macro-segment PPO update on q_phi for V6I1 Phase B/C."""
        trainer = self.host.trainer
        stats = MacroCreditManager.empty_macro_strategy_stats()
        stats["latent_macro_finalized_count"] = float(self.host.rollout_macro_finalized_count)
        stats["latent_macro_dropped_short_count"] = float(self.host.rollout_macro_dropped_short_count)
        if self.host.rollout_macro_finalized_count > 0:
            stats["latent_macro_mean_length"] = float(
                self.host.rollout_macro_length_sum / self.host.rollout_macro_finalized_count
            )
            stats["latent_macro_mean_return"] = float(
                self.host.rollout_macro_return_sum / self.host.rollout_macro_finalized_count
            )
        if not self.host._v6i1_macro_enabled() or trainer.fixed_latent_strategy:
            return stats
        records = list(self.host.rollout_strategy_macro_records)
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
        selector_hidden = stack_selector_hidden_records(records, device=device)
        macro_returns = torch.as_tensor(
            [float(r["macro_return"]) for r in records], dtype=torch.float32, device=device
        )
        stats["latent_macro_count"] = float(macro_returns.numel())
        coef = max(0.0, float(getattr(trainer.cfg, "v6i1_macro_strategy_ppo_coef", 1.0) or 0.0))
        stats["latent_macro_credit_coef"] = coef
        if coef <= 0.0:
            return stats
        n_epochs = max(1, int(getattr(trainer.cfg, "v6i1_macro_strategy_n_epochs", 4) or 1))
        clip_eps = max(1e-6, float(getattr(trainer.cfg, "v6i1_macro_strategy_clip_eps", 0.2) or 0.2))
        return_norm = bool(getattr(trainer.cfg, "v6i1_macro_strategy_return_norm", True))
        value_coef = max(
            0.0, float(getattr(trainer.cfg, "v6i1_macro_strategy_value_coef", 0.5) or 0.0)
        )
        encoder_params = self.host._strategy_encoder_params()
        value_head_params = self.host._value_head_params()
        with torch.no_grad():
            if trainer.model.episode_strategy_value_head is None:
                fixed_baseline = torch.zeros_like(macro_returns)
            else:
                fixed_baseline = trainer.model.episode_strategy_value(
                    states, z, selector_hidden=selector_hidden
                ).detach()
            fixed_adv = macro_returns - fixed_baseline
            if return_norm and fixed_adv.numel() > 1:
                fixed_adv = (fixed_adv - fixed_adv.mean()) / (fixed_adv.std(unbiased=False) + 1e-8)
            fixed_adv = fixed_adv.detach()

        router_opt = getattr(trainer, "router_optimizer", None) or getattr(
            trainer, "latent_router_optimizer", None
        )
        fallback_opt = None if router_opt is not None else trainer.optimizer
        registry = LatentOptimizerRegistry.from_trainer(trainer) if router_opt is not None else None
        engine = RouterPPOEngine(
            trainer=trainer, registry=registry, fallback_optimizer=fallback_opt
        )

        def value_fn(st, z_t, hidden):
            if trainer.model.episode_strategy_value_head is None:
                return torch.zeros_like(macro_returns)
            return trainer.model.episode_strategy_value(st, z_t, selector_hidden=hidden)

        batch = RouterPPOBatch(
            states=states,
            executed_z=z,
            old_behavior_log_prob=old_log_prob,
            fixed_advantages=fixed_adv,
            returns=macro_returns,
            selector_hidden=selector_hidden,
        )
        config = RouterPPOConfig(
            coef=coef,
            value_coef=value_coef,
            clip_epsilon=clip_eps,
            epochs=n_epochs,
            normalize_advantages=return_norm,
        )
        ppo_stats, _steps = engine.run(
            batch,
            config=config,
            value_fn=value_fn,
            param_groups=[encoder_params, value_head_params],
        )
        if ppo_stats:
            stats["latent_macro_advantage_mean"] = float(ppo_stats.get("advantage_mean", 0.0))
            stats["latent_macro_advantage_std"] = float(ppo_stats.get("advantage_std", 0.0))
            stats["latent_macro_policy_loss"] = float(ppo_stats.get("policy_loss", 0.0))
            stats["latent_macro_value_loss"] = float(ppo_stats.get("value_loss", 0.0))
            stats["latent_macro_clipfrac"] = float(ppo_stats.get("clipfrac", 0.0))
            stats["latent_macro_approx_kl"] = float(ppo_stats.get("approx_kl", 0.0))
            stats["latent_macro_grad_norm"] = float(ppo_stats.get("grad_norm", 0.0))
            self.host.router_optimizer_step_count += 1
        return stats


