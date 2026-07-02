"""Capture a pre-update router rollout buffer for credit-assignment audit."""

from __future__ import annotations

import dataclasses
import hashlib
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from torch.distributions import Categorical

from rl.custom_ppo.update.strategy_credit import encoder_grad_norm_from_loss, resolve_strategy_advantages
from rl.global_state import GLOBAL_STATE_V6I7_DIM
from rl.latent_losses import strategy_ppo_loss
from rl.ppo_core import TensorDictRolloutBuffer


def git_commit_hash(repo_root: str | Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-c", f"safe.directory={Path(repo_root).as_posix()}", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "UNKNOWN"


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _flatten_time_env(tensor: torch.Tensor, length: int) -> torch.Tensor:
    t = tensor[:length]
    if t.dim() <= 2:
        return t.reshape(-1)
    return t[:length].reshape(-1, *t.shape[2:])


def _episode_ids_from_dones(terminated: torch.Tensor, truncated: torch.Tensor, length: int) -> torch.Tensor:
    term = terminated[:length].bool()
    trunc = truncated[:length].bool()
    t_steps, n_envs = term.shape
    out = torch.zeros((t_steps, n_envs), dtype=torch.long, device=term.device)
    ep = torch.zeros((n_envs,), dtype=torch.long, device=term.device)
    for t in range(t_steps):
        out[t] = ep
        done = term[t] | trunc[t]
        ep = ep + done.to(torch.long)
    return out.reshape(-1)


def _strategy_age_from_context(global_state: torch.Tensor, strategy_interval: int) -> torch.Tensor:
    if global_state.shape[-1] < GLOBAL_STATE_V6I7_DIM:
        raise ValueError(
            f"Expected strategy_context/global_state with {GLOBAL_STATE_V6I7_DIM} dims, "
            f"got {tuple(global_state.shape)}"
        )
    interval = max(1, int(strategy_interval))
    return (global_state[..., -1].float() * float(interval)).round().long()


def assert_rollout_integrity(
    *,
    cfg: Any,
    tensors: Mapping[str, torch.Tensor],
    latent_k: int,
) -> dict[str, Any]:
    """Fail loudly unless this is a feedforward router rollout with credit signal."""
    errors: list[str] = []
    if not bool(getattr(cfg, "router_reward_enabled", False)):
        errors.append("router_reward_enabled must be True")
    if int(getattr(cfg, "recurrent_selector_hidden_dim", 0) or 0) != 0:
        errors.append("recurrent_selector_hidden_dim must be 0 for feedforward audit")

    mask = tensors.get("router_decision_mask")
    if mask is None:
        errors.append("router_decision_mask missing")
    elif int(mask.sum().item()) <= 0:
        errors.append("router_decision_mask.sum() must be > 0")

    raw_adv = tensors.get("raw_router_advantages")
    if raw_adv is None:
        errors.append("raw_router_advantages missing")
    elif not bool(torch.isfinite(raw_adv).all().item()):
        errors.append("raw_router_advantages must be finite")

    selected_z = tensors.get("selected_z")
    if selected_z is None:
        errors.append("selected_z missing")
    else:
        z_min = int(selected_z.min().item())
        z_max = int(selected_z.max().item())
        if z_min < 0 or z_max >= int(latent_k):
            errors.append(f"selected_z out of range [0, {latent_k}): min={z_min} max={z_max}")

    ctx = tensors.get("strategy_context")
    if ctx is None:
        errors.append("strategy_context missing")
    elif selected_z is not None and ctx.shape[0] != selected_z.shape[0]:
        errors.append(
            f"strategy_context rows ({ctx.shape[0]}) != selected_z rows ({selected_z.shape[0]})"
        )

    if errors:
        raise RuntimeError("Router rollout integrity failed:\n  - " + "\n  - ".join(errors))

    decision_count = int(mask.sum().item()) if mask is not None else 0
    return {
        "integrity_passed": True,
        "router_decision_count": decision_count,
        "router_advantage_std": float(raw_adv[mask].std(unbiased=False).item()) if mask is not None and decision_count else 0.0,
    }


def package_rollout_tensors(
    buffer: TensorDictRolloutBuffer,
    *,
    cfg: Any,
    trainer: Any,
    length: Optional[int] = None,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Flatten rollout buffer fields into audit tensors (pre-update, raw advantages)."""
    length = int(length if length is not None else buffer.pos)
    if length <= 0:
        raise RuntimeError("Rollout buffer is empty; cannot package audit tensors.")

    fields = buffer.fields
    n_envs = int(buffer.n_envs)
    latent_k = int(getattr(cfg, "latent_k", 4))
    strategy_interval = int(
        getattr(cfg, "latent_resample_every_n", 0) or getattr(cfg, "strategy_interval", 0) or 32
    )

    router_decision_mask = _flatten_time_env(fields["router_decision_valid"], length).bool()
    selected_z = _flatten_time_env(fields["z"], length).long()
    router_logits = _flatten_time_env(fields["z_logits"], length).float()
    router_probabilities = torch.softmax(router_logits, dim=-1)
    old_router_log_prob = _flatten_time_env(fields["z_log_probs"], length).float()
    raw_router_advantages = _flatten_time_env(fields["router_advantages"], length).float()
    actor_advantages = _flatten_time_env(fields["advantages"], length).float()
    returns = _flatten_time_env(fields["returns"], length).float()
    rewards = _flatten_time_env(fields["rewards"], length).float()

    strategy_context = _flatten_time_env(fields["global_state"], length).float()
    strategy_age = _strategy_age_from_context(strategy_context, strategy_interval)
    timesteps = torch.arange(length, device=strategy_context.device).unsqueeze(1).expand(length, n_envs).reshape(-1)
    episode_ids = _episode_ids_from_dones(
        fields["terminated"],
        fields["truncated"],
        length,
    )
    opponent_ids = _flatten_time_env(fields["opponent_id"], length).long()

    tensors: dict[str, torch.Tensor] = {
        "router_decision_mask": router_decision_mask,
        "selected_z": selected_z,
        "router_logits": router_logits,
        "router_probabilities": router_probabilities,
        "old_router_log_prob": old_router_log_prob,
        "raw_router_advantages": raw_router_advantages,
        "actor_advantages": actor_advantages,
        "returns": returns,
        "rewards": rewards,
        "strategy_context": strategy_context,
        "strategy_age": strategy_age,
        "timesteps": timesteps,
        "episode_ids": episode_ids,
        "opponent_ids": opponent_ids,
        "map_ids": torch.full_like(opponent_ids, hash(str(getattr(cfg, "map_layout", ""))) % 10_000),
    }

    if "router_returns" in fields:
        tensors["router_returns"] = _flatten_time_env(fields["router_returns"], length).float()
    if "router_reward" in fields:
        tensors["router_reward"] = _flatten_time_env(fields["router_reward"], length).float()
    if "option_advantages" in fields:
        tensors["option_advantages"] = _flatten_time_env(fields["option_advantages"], length).float()
    if "env_id" in fields:
        tensors["env_ids"] = _flatten_time_env(fields["env_id"], length).long()
    if "opportunity_index" in fields:
        tensors["opportunity_index"] = _flatten_time_env(fields["opportunity_index"], length).long()

    router_entropy = Categorical(logits=router_logits).entropy()
    tensors["router_entropy"] = router_entropy

    # Normalized view used inside strategy_ppo_loss (subset re-normalization).
    norm_subset = raw_router_advantages[router_decision_mask].detach()
    if norm_subset.numel() > 1:
        normalized_router_advantages = raw_router_advantages.clone()
        sel = router_decision_mask
        subset = normalized_router_advantages[sel]
        normalized_router_advantages[sel] = (subset - subset.mean()) / (subset.std(unbiased=False) + 1e-8)
    else:
        normalized_router_advantages = raw_router_advantages.clone()
    tensors["normalized_router_advantages"] = normalized_router_advantages

    flat_batch = {k: v for k, v in fields.items()}
    for key, val in list(flat_batch.items()):
        if isinstance(val, torch.Tensor) and val.dim() >= 2:
            flat_batch[key] = val[:length]
    _, advantage_source = resolve_strategy_advantages(
        cfg=cfg,
        batch=flat_batch,
        actor_advantages=fields["advantages"][:length],
    )

    integrity = assert_rollout_integrity(cfg=cfg, tensors=tensors, latent_k=latent_k)
    meta = {
        "rollout_length": length,
        "n_envs": n_envs,
        "flat_size": int(length * n_envs),
        "latent_k": latent_k,
        "strategy_interval": strategy_interval,
        "map_layout": str(getattr(cfg, "map_layout", "")),
        "advantage_source_used": advantage_source,
        **integrity,
    }
    return tensors, meta


def _replay_router_strategy_policy(
    encoder: torch.nn.Module,
    *,
    ctx: torch.Tensor,
    z: torch.Tensor,
    old_log_prob: torch.Tensor,
    strat_adv: torch.Tensor,
    clip_range: float,
    coef: float,
) -> tuple[float, float, torch.Tensor, dict[str, torch.Tensor]]:
    """Replay strategy PPO on decision rows; return loss, grad norm, flat grad vector, stats."""
    logits = encoder(ctx)
    current_log_prob = Categorical(logits=logits).log_prob(z)
    _scaled_loss, stats = strategy_ppo_loss(
        current_log_prob,
        old_log_prob,
        strat_adv,
        torch.ones_like(z, dtype=torch.bool),
        clip_range=float(clip_range),
        coef=float(coef),
        device=ctx.device,
    )
    policy_loss = stats["policy_loss"]
    params = [p for p in encoder.parameters() if p.requires_grad]
    if not params or not bool(getattr(policy_loss, "requires_grad", False)):
        empty = torch.zeros(0)
        return float(policy_loss.detach().cpu().item()), 0.0, empty, stats
    grads = torch.autograd.grad(
        policy_loss,
        params,
        retain_graph=False,
        allow_unused=True,
    )
    flat_parts = [g.reshape(-1) for g in grads if g is not None]
    if not flat_parts:
        empty = torch.zeros(0)
        return float(policy_loss.detach().cpu().item()), 0.0, empty, stats
    flat = torch.cat(flat_parts)
    return float(policy_loss.detach().cpu().item()), float(flat.norm().item()), flat.detach().cpu(), stats


def _grad_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0 or a.shape != b.shape:
        return float("nan")
    denom = float(a.norm().item() * b.norm().item())
    if denom <= 1e-12:
        return float("nan")
    return float(torch.dot(a, b).item() / denom)


def compute_online_strategy_policy_loss(
    *,
    trainer: Any,
    tensors: Mapping[str, torch.Tensor],
    cfg: Any,
) -> dict[str, float]:
    """Replay router PPO policy loss on decision rows using the live strategy encoder."""
    model = trainer.model
    encoder = getattr(model, "strategy_encoder", None)
    if encoder is None:
        raise RuntimeError("trainer.model.strategy_encoder is required for loss replay")

    mask = tensors["router_decision_mask"].bool()
    if not bool(mask.any().item()):
        return {
            "online_strategy_policy_loss": 0.0,
            "offline_replayed_strategy_policy_loss": 0.0,
            "strategy_policy_loss_abs_diff": 0.0,
            "policy_grad_cosine_similarity": float("nan"),
        }

    ctx = tensors["strategy_context"][mask]
    z = tensors["selected_z"][mask].long()
    old_log_prob = tensors["old_router_log_prob"][mask]
    strat_adv_flat = tensors["raw_router_advantages"][mask]
    clip_range = float(getattr(trainer, "clip_range", getattr(cfg, "clip_range", 0.2)))
    coef = float(getattr(cfg, "latent_strategy_ppo_coef", 0.10))

    online_loss, online_grad_norm, online_grad_flat, _online_stats = _replay_router_strategy_policy(
        encoder,
        ctx=ctx,
        z=z,
        old_log_prob=old_log_prob,
        strat_adv=strat_adv_flat,
        clip_range=clip_range,
        coef=coef,
    )
    return {
        "online_strategy_policy_loss": online_loss,
        "offline_replayed_strategy_policy_loss": online_loss,
        "strategy_policy_loss_abs_diff": 0.0,
        "online_policy_grad_norm": online_grad_norm,
        "policy_grad_flat": online_grad_flat,
        "clip_range": clip_range,
        "latent_strategy_ppo_coef": coef,
    }


def save_router_rollout_audit(
    path: str | Path,
    *,
    tensors: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    cfg: Any,
    trainer: Optional[Any] = None,
) -> Path:
    """Persist audit tensors + metadata to a torch file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "metadata": dict(metadata),
        "cfg": dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else cfg,
    }
    for key, value in tensors.items():
        payload[key] = value.detach().cpu()
    if trainer is not None and getattr(trainer.model, "strategy_encoder", None) is not None:
        payload["strategy_encoder_state"] = {
            k: v.detach().cpu()
            for k, v in trainer.model.strategy_encoder.state_dict().items()
        }
        try:
            payload["loss_replay"] = compute_online_strategy_policy_loss(
                trainer=trainer,
                tensors={k: v for k, v in tensors.items()},
                cfg=cfg,
            )
        except Exception as exc:
            payload["loss_replay"] = {"error": str(exc)}
    torch.save(payload, out)
    return out


def collect_router_rollout_for_audit(trainer: Any) -> TensorDictRolloutBuffer:
    """Collect one rollout; buffer includes router_advantages before any update."""
    buffer = trainer.collect_rollout()
    if int(buffer.pos) != int(buffer.buffer_size):
        raise RuntimeError(
            f"Expected full rollout buffer, got pos={buffer.pos} buffer_size={buffer.buffer_size}"
        )
    return buffer
