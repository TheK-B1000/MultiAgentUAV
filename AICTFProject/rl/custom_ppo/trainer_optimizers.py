"""Optimizer bundle for :class:`~rl.custom_ppo.trainer.CustomPPOTrainer`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from rl.custom_ppo.curriculum_gates import is_staged_v6i1_curriculum


def _collect_params(model: torch.nn.Module, name_parts: tuple[str, ...]) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if any(part in name for part in name_parts):
            if "episode_strategy_value_head" in name and "critic" in name:
                continue
            params.append(param)
    return params


def _router_param_names() -> tuple[str, ...]:
    return (
        "strategy_encoder",
        "selector_gru",
        "episode_strategy_value_head",
        "strategy_aux_return_head",
        "phase_predictor",
    )


def _maybe_build_latent_router_optimizer(
    model: torch.nn.Module,
    *,
    hparams: Any,
) -> Optional[torch.optim.Optimizer]:
    if (
        hparams.latent_episode_strategy_lr is None
        or not hparams.use_latent_strategy
        or hparams.fixed_latent_strategy
    ):
        return None
    router_params = _collect_params(model, _router_param_names())
    if not router_params:
        return None
    return torch.optim.AdamW(
        router_params,
        lr=float(hparams.latent_episode_strategy_lr),
        eps=1e-5,
    )


@dataclass
class TrainerOptimizerBundle:
    """Grouped trainer optimizers with a single checkpoint surface."""

    primary: torch.optim.Optimizer
    actor: torch.optim.Optimizer
    critic: torch.optim.Optimizer
    router: Optional[torch.optim.Optimizer] = None
    v6i1_three_optimizer_mode: bool = False

    @property
    def latent_router_optimizer(self) -> Optional[torch.optim.Optimizer]:
        return self.router

    @classmethod
    def build(cls, *, model: torch.nn.Module, cfg: Any, hparams: Any) -> TrainerOptimizerBundle:
        if is_staged_v6i1_curriculum(cfg):
            return cls._build_v6i1(model=model, cfg=cfg, hparams=hparams)
        shared = torch.optim.Adam(model.parameters(), lr=float(hparams.learning_rate), eps=1e-5)
        return cls(
            primary=shared,
            actor=shared,
            critic=shared,
            router=_maybe_build_latent_router_optimizer(model, hparams=hparams),
            v6i1_three_optimizer_mode=False,
        )

    @classmethod
    def _build_v6i1(
        cls,
        *,
        model: torch.nn.Module,
        cfg: Any,
        hparams: Any,
        base_lr: float | None = None,
    ) -> TrainerOptimizerBundle:
        lr = float(base_lr if base_lr is not None else hparams.learning_rate)
        router_lr = float(
            getattr(cfg, "v6i1_router_lr", None)
            or getattr(hparams, "latent_episode_strategy_lr", None)
            or 5e-3
        )
        actor_params = _collect_params(model, ("actor_cnn", "latent_actor"))
        critic_params = _collect_params(model, ("critic",))
        router_params = _collect_params(model, _router_param_names())
        if not actor_params or not critic_params or not router_params:
            raise RuntimeError("V6I1 three-optimizer setup requires actor, critic, and router parameters.")
        actor_opt = torch.optim.Adam(actor_params, lr=lr, eps=1e-5)
        critic_opt = torch.optim.Adam(critic_params, lr=lr, eps=1e-5)
        router_opt = torch.optim.AdamW(router_params, lr=router_lr, eps=1e-5)
        return cls(
            primary=actor_opt,
            actor=actor_opt,
            critic=critic_opt,
            router=router_opt,
            v6i1_three_optimizer_mode=True,
        )

    def write_checkpoint(self, payload: dict[str, Any]) -> None:
        payload["optimizer_state_dict"] = self.primary.state_dict()
        if self.v6i1_three_optimizer_mode:
            payload["v6i1_three_optimizer_mode"] = True
            payload["actor_optimizer_state_dict"] = self.actor.state_dict()
            payload["critic_optimizer_state_dict"] = self.critic.state_dict()
            if self.router is not None:
                payload["router_optimizer_state_dict"] = self.router.state_dict()

    def load_checkpoint(self, payload: dict[str, Any]) -> None:
        self.primary.load_state_dict(payload["optimizer_state_dict"])
        if not bool(payload.get("v6i1_three_optimizer_mode", False)):
            return
        if "actor_optimizer_state_dict" in payload:
            self.actor.load_state_dict(payload["actor_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in payload:
            self.critic.load_state_dict(payload["critic_optimizer_state_dict"])
        if self.router is not None and "router_optimizer_state_dict" in payload:
            self.router.load_state_dict(payload["router_optimizer_state_dict"])


__all__ = ["TrainerOptimizerBundle"]
