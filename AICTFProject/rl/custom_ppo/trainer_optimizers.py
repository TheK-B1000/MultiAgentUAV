"""Optimizer bundle for :class:`~rl.custom_ppo.trainer.CustomPPOTrainer`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch

from rl.custom_ppo.curriculum_gates import is_staged_v6i1_curriculum


def collect_actor_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    """Explicit actor parameter list shared by the actor optimizer and CF diagnostics."""
    parts = ("actor_cnn", "latent_actor")
    if bool(getattr(model, "communication_enabled", False)):
        parts = ("actor_cnn", "latent_actor", "message_head")
    return _collect_params(model, parts)


def collect_actor_optimizer_parameters(
    optimizer: torch.optim.Optimizer,
) -> list[torch.nn.Parameter]:
    """Flatten every actor-optimizer parameter group (with duplicate guard)."""
    params = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    if len({id(p) for p in params}) != len(params):
        raise RuntimeError("Actor optimizer parameter groups contain duplicates.")
    return params


def freeze_actor_parameters(model: torch.nn.Module) -> int:
    params = collect_actor_parameters(model)
    for param in params:
        param.requires_grad_(False)
    return len(params)


# Names that identify z-specific parameters within latent_actor.
# These receive gradients during Stage 2 (repertoire) while the shared trunk is frozen.
_Z_SPECIFIC_SUBSTRINGS = (
    "latent_adapters",
    "latent_adapter_gates",
    "latent_action_biases",
    "latent_action_heads",
    "latent_branch_trunks",
    "strategy_embedding",
    "z_adapter",
)

# Names that identify the shared backbone (CNN + actor trunk) to freeze in Stage 2.
_SHARED_BACKBONE_PARTS = ("actor_cnn", "latent_actor")

# Router modules frozen during Stage 2 (repertoire) to guarantee grad is None.
# They are re-enabled implicitly at Stage 3 (router) via the model's default requires_grad=True.
_ROUTER_MODULE_NAMES = (
    "strategy_encoder",
    "selector_gru",
    "phase_predictor",
    "strategy_aux_return_head",
)


def is_z_specific_actor_param(name: str) -> bool:
    return any(sub in name for sub in _Z_SPECIFIC_SUBSTRINGS)


def is_shared_frozen_actor_param(name: str) -> bool:
    if not any(part in name for part in _SHARED_BACKBONE_PARTS):
        return False
    return not is_z_specific_actor_param(name)


def freeze_shared_trunk_train_z_only(model: torch.nn.Module) -> int:
    """Stage 2 (repertoire): freeze CNN + shared trunk + router; leave only z-specific + critic trainable."""
    frozen = 0
    for name, param in model.named_parameters():
        is_actor_part = any(part in name for part in _SHARED_BACKBONE_PARTS)
        is_router_part = any(part in name for part in _ROUTER_MODULE_NAMES)
        if not is_actor_part and not is_router_part:
            continue
        if is_actor_part:
            is_z_specific = any(sub in name for sub in _Z_SPECIFIC_SUBSTRINGS)
            if is_z_specific:
                continue  # leave z-specific trainable
        param.requires_grad_(False)
        frozen += 1
    return frozen


def freeze_z_specific_parameters(model: torch.nn.Module) -> int:
    """Stage 3 (router): freeze z-specific adapter/gate/bias/embedding params."""
    frozen = 0
    for name, param in model.named_parameters():
        if any(sub in name for sub in _Z_SPECIFIC_SUBSTRINGS):
            param.requires_grad_(False)
            frozen += 1
    return frozen


def _inactive_lro_branch_param(name: str, *, active_z: int, latent_k: int) -> bool:
    """True if ``name`` is a per-z specialist tensor for an inactive branch."""
    for sub in (
        "latent_adapters.",
        "latent_action_heads.",
        "latent_branch_trunks.",
        "latent_action_biases.",
    ):
        if sub in name:
            # e.g. latent_actor.latent_adapters.2.weight
            try:
                after = name.split(sub, 1)[1]
                z_idx = int(after.split(".", 1)[0])
            except (IndexError, ValueError):
                return False
            return int(z_idx) != int(active_z)
    # strategy_embedding is a single Parameter; freeze via embedding index is
    # not possible with requires_grad — inactive rows simply receive no grads
    # under forced-z. Treat the whole embedding as z-specific (active group).
    del latent_k
    return False


def freeze_inactive_lro_branch_parameters(
    model: torch.nn.Module,
    *,
    active_z: int,
    latent_k: int,
) -> int:
    """Freeze specialist modules for every z ≠ active_z (LRO active-branch-only)."""
    frozen = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _inactive_lro_branch_param(name, active_z=int(active_z), latent_k=int(latent_k)):
            param.requires_grad_(False)
            frozen += 1
    return frozen


def collect_repertoire_optimizer_groups(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    z_actor_lr_mult: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build disjoint Adam param groups for repertoire (z-actor vs critic).

    Assertions:
      - groups are disjoint
      - every trainable parameter appears exactly once
      - frozen/shared parameters appear in neither group
      - z-specific trainable tensors are in the actor group
    """
    z_params: list[torch.nn.Parameter] = []
    critic_params: list[torch.nn.Parameter] = []
    other_trainable: list[tuple[str, torch.nn.Parameter]] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if is_z_specific_actor_param(name):
            z_params.append(param)
        elif "critic" in name:
            critic_params.append(param)
        else:
            other_trainable.append((name, param))
    if other_trainable:
        names = ", ".join(n for n, _ in other_trainable[:8])
        raise RuntimeError(
            "Repertoire optimizer groups found unexpected trainable params "
            f"(expected only z-specific actor + critic): {names}"
        )
    z_ids = {id(p) for p in z_params}
    c_ids = {id(p) for p in critic_params}
    if z_ids & c_ids:
        raise RuntimeError("Repertoire optimizer groups are not disjoint.")
    trainable = [p for p in model.parameters() if p.requires_grad]
    covered = z_ids | c_ids
    if len(covered) != len(trainable) or any(id(p) not in covered for p in trainable):
        raise RuntimeError(
            "Repertoire optimizer groups must cover every trainable parameter exactly once."
        )
    if not z_params:
        raise RuntimeError("Repertoire z-actor param group is empty.")
    if not critic_params:
        raise RuntimeError("Repertoire critic param group is empty.")
    base_lr = float(learning_rate)
    mult = float(z_actor_lr_mult)
    if mult <= 0.0:
        raise ValueError(f"latent_lro_z_actor_lr_mult must be > 0, got {mult}")
    groups = [
        {"params": z_params, "lr": base_lr * mult, "name": "z_actor"},
        {"params": critic_params, "lr": base_lr, "name": "critic"},
    ]
    audit = {
        "n_z_actor_params": len(z_params),
        "n_critic_params": len(critic_params),
        "z_actor_lr": base_lr * mult,
        "critic_lr": base_lr,
        "z_actor_lr_mult": mult,
    }
    return groups, audit


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
    actor_cf: Optional[torch.optim.Optimizer] = None
    router: Optional[torch.optim.Optimizer] = None
    v6i1_three_optimizer_mode: bool = False

    @property
    def latent_router_optimizer(self) -> Optional[torch.optim.Optimizer]:
        return self.router

    @classmethod
    def build(cls, *, model: torch.nn.Module, cfg: Any, hparams: Any) -> TrainerOptimizerBundle:
        stage = str(getattr(cfg, "v6i9_training_stage", "") or "").lower().strip()
        if stage == "repertoire":
            n = freeze_shared_trunk_train_z_only(model)
            print(f"[V6I9 repertoire] Froze {n} shared-trunk params; z-specific modules remain trainable.")
            if bool(getattr(cfg, "latent_lro_active_branch_only", False)):
                active_z = int(getattr(cfg, "fixed_latent_strategy_id", 0) or 0)
                latent_k = int(getattr(model, "latent_k", getattr(cfg, "latent_k", 4)) or 4)
                n_inactive = freeze_inactive_lro_branch_parameters(
                    model, active_z=active_z, latent_k=latent_k
                )
                print(
                    f"[V6I26 LRO] Froze {n_inactive} inactive-branch params; "
                    f"active_z={active_z}."
                )
        elif stage == "router":
            freeze_actor_parameters(model)
            n = freeze_z_specific_parameters(model)
            print(f"[V6I9 router] Froze actor + {n} z-specific params; router+critic remain trainable.")
        elif bool(getattr(cfg, "router_freeze_actor", False)):
            freeze_actor_parameters(model)
        if is_staged_v6i1_curriculum(cfg):
            return cls._build_v6i1(model=model, cfg=cfg, hparams=hparams)

        separate_clip = bool(getattr(cfg, "latent_lro_separate_actor_critic_clip", False))
        z_lr_mult = float(getattr(cfg, "latent_lro_z_actor_lr_mult", 1.0) or 1.0)
        use_repertoire_groups = stage == "repertoire" and (
            separate_clip or abs(z_lr_mult - 1.0) > 1e-12
        )
        if use_repertoire_groups:
            groups, audit = collect_repertoire_optimizer_groups(
                model,
                learning_rate=float(hparams.learning_rate),
                z_actor_lr_mult=z_lr_mult,
            )
            shared = torch.optim.Adam(groups, eps=1e-5)
            print(
                f"[V6I26 actor-step] repertoire Adam groups: "
                f"z_actor_lr={audit['z_actor_lr']:.3e} "
                f"critic_lr={audit['critic_lr']:.3e} "
                f"n_z={audit['n_z_actor_params']} n_critic={audit['n_critic_params']} "
                f"separate_clip={separate_clip}"
            )
            return cls(
                primary=shared,
                actor=shared,
                critic=shared,
                router=_maybe_build_latent_router_optimizer(model, hparams=hparams),
                v6i1_three_optimizer_mode=False,
            )

        shared_params = [p for p in model.parameters() if p.requires_grad]
        shared = torch.optim.Adam(shared_params, lr=float(hparams.learning_rate), eps=1e-5)
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
        actor_params = collect_actor_parameters(model)
        critic_params = _collect_params(model, ("critic",))
        router_params = _collect_params(model, _router_param_names())
        if not actor_params or not critic_params or not router_params:
            raise RuntimeError("V6I1 three-optimizer setup requires actor, critic, and router parameters.")
        actor_opt = torch.optim.Adam(actor_params, lr=lr, eps=1e-5)
        actor_cf_opt = torch.optim.Adam(actor_params, lr=lr, eps=1e-5)
        critic_opt = torch.optim.Adam(critic_params, lr=lr, eps=1e-5)
        router_opt = torch.optim.AdamW(router_params, lr=router_lr, eps=1e-5)
        return cls(
            primary=actor_opt,
            actor=actor_opt,
            critic=critic_opt,
            actor_cf=actor_cf_opt,
            router=router_opt,
            v6i1_three_optimizer_mode=True,
        )

    def write_checkpoint(self, payload: dict[str, Any]) -> None:
        payload["optimizer_state_dict"] = self.primary.state_dict()
        if self.v6i1_three_optimizer_mode:
            payload["v6i1_three_optimizer_mode"] = True
            payload["actor_optimizer_state_dict"] = self.actor.state_dict()
            if self.actor_cf is not None:
                payload["actor_cf_optimizer_state_dict"] = self.actor_cf.state_dict()
            payload["critic_optimizer_state_dict"] = self.critic.state_dict()
            if self.router is not None:
                payload["router_optimizer_state_dict"] = self.router.state_dict()

    def load_checkpoint(self, payload: dict[str, Any], *, allow_architecture_migration: bool = False) -> None:
        try:
            self.primary.load_state_dict(payload["optimizer_state_dict"])
        except (ValueError, RuntimeError) as exc:
            if not allow_architecture_migration:
                raise RuntimeError(
                    f"Optimizer state mismatch on load — parameter groups do not match the current model. "
                    f"If this is an intentional architecture migration (e.g. adding a CNN channel), "
                    f"pass --allow-active-actor-module-migration to skip optimizer state and start fresh. "
                    f"Original error: {exc}"
                ) from exc
            print(
                f"[optimizer] Architecture migration allowed — skipping optimizer state (starting fresh). "
                f"Reason: {exc}"
            )
            return
        if not bool(payload.get("v6i1_three_optimizer_mode", False)):
            return
        if "actor_optimizer_state_dict" in payload:
            self.actor.load_state_dict(payload["actor_optimizer_state_dict"])
        if self.actor_cf is not None and "actor_cf_optimizer_state_dict" in payload:
            self.actor_cf.load_state_dict(payload["actor_cf_optimizer_state_dict"])
        if "critic_optimizer_state_dict" in payload:
            self.critic.load_state_dict(payload["critic_optimizer_state_dict"])
        if self.router is not None and "router_optimizer_state_dict" in payload:
            self.router.load_state_dict(payload["router_optimizer_state_dict"])


__all__ = [
    "TrainerOptimizerBundle",
    "collect_actor_optimizer_parameters",
    "collect_actor_parameters",
    "collect_repertoire_optimizer_groups",
    "freeze_actor_parameters",
    "freeze_inactive_lro_branch_parameters",
    "freeze_shared_trunk_train_z_only",
    "freeze_z_specific_parameters",
    "is_shared_frozen_actor_param",
    "is_z_specific_actor_param",
]
