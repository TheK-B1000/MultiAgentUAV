"""Router credit-assignment audit helpers for feedforward v6i9 router training.

These utilities support the pre-scale diagnostic loop:
1. gradient-path telemetry on router opportunities
2. synthetic update-direction (sign) tests
3. router-advantage distribution summaries
4. PPO vs entropy gradient-norm ratios
5. offline feedforward predictability of hindsight best-z
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Categorical

from rl.latent_losses import strategy_entropy_loss, strategy_ppo_loss
from rl.ppo_core import ppo_policy_loss


@dataclass
class RouterOpportunityRecord:
    """One router decision point for gradient-path tracing."""

    context_features: list[float]
    router_logits: list[float]
    router_probabilities: list[float]
    selected_z: int
    old_log_prob: float
    current_log_prob: float
    router_advantage: float
    actor_advantage: float
    return_target: float
    importance_ratio: float
    clipped_ratio: float
    policy_loss_contrib: float
    entropy_loss_contrib: float
    total_router_loss_contrib: float
    opponent: str = ""
    map_name: str = ""
    opportunity_index: int = 0


@dataclass
class RouterAdvantageGroupStats:
    group_key: str
    count: int
    mean_advantage: float
    advantage_std: float
    fraction_positive: float
    mean_return: float
    return_correlation: float


@dataclass
class RouterGradientNormReport:
    router_ppo_grad_norm: float
    router_entropy_grad_norm: float
    router_critic_grad_norm: float
    ppo_to_entropy_ratio: float
    router_params_changed: bool
    param_delta_l2: float


@dataclass
class SyntheticSignTestResult:
    passed: bool
    p_z0_context_a_before: float
    p_z0_context_a_after: float
    p_z1_context_b_before: float
    p_z1_context_b_after: float
    reversed_passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class OfflinePredictabilityReport:
  n_samples: int
  n_classes: int
  accuracy: float
  top2_accuracy: float
  mean_regret_vs_best_z: float
  mean_return_predicted_z: float
  mean_return_best_fixed_z2: float
  beats_chance: bool
  beats_fixed_z2: bool


def resolve_strategy_advantage_source(
    batch: Mapping[str, Tensor],
    actor_advantages: Tensor,
    *,
    recurrent_selector_active: bool,
    latent_q_phi_option_advantage: bool = False,
) -> tuple[Tensor, str]:
    """Document which advantage tensor feeds strategy PPO for this update."""
    if recurrent_selector_active and "router_advantages" in batch:
        return batch["router_advantages"].float(), "router_advantages"
    if latent_q_phi_option_advantage and "option_advantages" in batch:
        return batch["option_advantages"].float(), "option_advantages"
    if "router_advantages" in batch:
        return batch["router_advantages"].float(), "router_advantages_unused_feedforward"
    return actor_advantages.float(), "actor_gae_advantages"


def trace_router_opportunities(
    *,
    context: Tensor,
    strategy_logits: Tensor,
    selected_z: Tensor,
    old_log_probs: Tensor,
    current_log_probs: Tensor,
    router_advantages: Tensor,
    actor_advantages: Tensor,
    return_targets: Tensor,
    resample_mask: Tensor,
    clip_range: float,
    lam_h: float,
    entropy_objective: str,
    opponents: Optional[Sequence[str]] = None,
    maps: Optional[Sequence[str]] = None,
) -> list[RouterOpportunityRecord]:
    """Record per-opportunity gradient-path scalars from one training minibatch."""
    records: list[RouterOpportunityRecord] = []
    if not bool(resample_mask.any()):
        return records

    idx = torch.where(resample_mask)[0]
    probs = torch.softmax(strategy_logits[idx], dim=-1)
    entropy = Categorical(logits=strategy_logits[idx]).entropy()
    ratio = torch.exp(current_log_probs[idx] - old_log_probs[idx])
    clipped = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    adv = router_advantages[idx].detach()
    pol_unscaled, _ = ppo_policy_loss(
        current_log_probs[idx],
        old_log_probs[idx],
        adv,
        float(clip_range),
    )
    ent_loss, _ = strategy_entropy_loss(
        entropy,
        torch.ones_like(entropy, dtype=torch.bool),
        objective=entropy_objective,
        lam_h=float(lam_h),
        device=strategy_logits.device,
    )

    for local_i, global_i in enumerate(idx.detach().cpu().tolist()):
        z_i = int(selected_z[global_i].item())
        records.append(
            RouterOpportunityRecord(
                context_features=context[global_i].detach().cpu().tolist(),
                router_logits=strategy_logits[global_i].detach().cpu().tolist(),
                router_probabilities=probs[local_i].detach().cpu().tolist(),
                selected_z=z_i,
                old_log_prob=float(old_log_probs[global_i].item()),
                current_log_prob=float(current_log_probs[global_i].item()),
                router_advantage=float(router_advantages[global_i].item()),
                actor_advantage=float(actor_advantages[global_i].item()),
                return_target=float(return_targets[global_i].item()),
                importance_ratio=float(ratio[local_i].item()),
                clipped_ratio=float(clipped[local_i].item()),
                policy_loss_contrib=float(pol_unscaled.item()) if local_i == 0 else 0.0,
                entropy_loss_contrib=float(ent_loss.item()) if local_i == 0 else 0.0,
                total_router_loss_contrib=float((pol_unscaled + ent_loss).item()) if local_i == 0 else 0.0,
                opponent=str(opponents[global_i]) if opponents is not None else "",
                map_name=str(maps[global_i]) if maps is not None else "",
                opportunity_index=int(global_i),
            )
        )
    return records


def verify_router_gradient_connectivity(
    current_log_prob: Tensor,
    router_loss: Tensor,
    router_params: Iterable[torch.nn.Parameter],
    advantages: Tensor,
) -> dict[str, Any]:
    """Fast checks that router loss can backprop into selector parameters."""
    selected_requires_grad = bool(current_log_prob.requires_grad)
    loss_requires_grad = bool(router_loss.requires_grad)
    adv_finite = bool(torch.isfinite(advantages).all().item())
    adv_nonzero_fraction = float((advantages.abs() > 1e-8).float().mean().item()) if advantages.numel() else 0.0

    before = [p.detach().clone() for p in router_params]
    if loss_requires_grad and selected_requires_grad:
        router_loss.backward(retain_graph=True)
        param_delta = 0.0
        changed = False
        for p, b in zip(router_params, before):
            if p.grad is not None and torch.isfinite(p.grad).all():
                delta = (p.data - b).norm().item()
                param_delta = max(param_delta, delta)
                changed = changed or delta > 0.0
    else:
        changed = False
        param_delta = 0.0

    return {
        "selected_log_prob_requires_grad": selected_requires_grad,
        "router_loss_requires_grad": loss_requires_grad,
        "advantages_finite": adv_finite,
        "advantages_nonzero_fraction": adv_nonzero_fraction,
        "router_params_changed_after_step": changed,
        "max_param_delta_l2": param_delta,
    }


def run_synthetic_router_sign_test(
    router_module: torch.nn.Module,
    *,
    context_dim: int,
    latent_k: int = 4,
    clip_range: float = 0.2,
    coef: float = 0.10,
    lr: float = 0.05,
    device: Optional[torch.device] = None,
) -> SyntheticSignTestResult:
    """Controlled batch: +advantage on z0, −advantage on z1; verify update direction."""
    dev = device or torch.device("cpu")
    router_module = router_module.to(dev)
    router_module.train()

    ctx_a = torch.randn(1, context_dim, device=dev)
    ctx_b = torch.randn(1, context_dim, device=dev)
    context = torch.cat([ctx_a, ctx_b], dim=0)

    def _forward(c: Tensor) -> Tensor:
        out = router_module(c)
        if isinstance(out, tuple):
            out = out[0]
        return out

    with torch.no_grad():
        logits_before = _forward(context)
        probs_before = torch.softmax(logits_before, dim=-1)

    z = torch.tensor([0, 1], device=dev, dtype=torch.long)
    initial_state = {k: v.detach().clone() for k, v in router_module.state_dict().items()}
    dist = Categorical(logits=_forward(context))
    old_log_prob = dist.log_prob(z).detach()
    current_log_prob = dist.log_prob(z)

    advantages = torch.tensor([1.0, -1.0], device=dev)
    mask = torch.ones(2, dtype=torch.bool, device=dev)
    loss, _ = strategy_ppo_loss(
        current_log_prob,
        old_log_prob,
        advantages,
        mask,
        clip_range=clip_range,
        coef=coef,
        device=dev,
    )

    opt = torch.optim.Adam(router_module.parameters(), lr=lr)
    opt.zero_grad()
    loss.backward()
    opt.step()

    with torch.no_grad():
        logits_after = _forward(context)
        probs_after = torch.softmax(logits_after, dim=-1)

    p0_up = float(probs_after[0, 0].item()) > float(probs_before[0, 0].item())
    p1_down = float(probs_after[1, 1].item()) < float(probs_before[1, 1].item())
    passed = p0_up and p1_down

    # Reverse advantages on a fresh copy from the pre-update snapshot.
    import copy

    encoder_rev = copy.deepcopy(router_module)
    encoder_rev.load_state_dict(initial_state)

    def _forward_rev(c: Tensor) -> Tensor:
        out = encoder_rev(c)
        if isinstance(out, tuple):
            out = out[0]
        return out

    with torch.no_grad():
        probs_mid = torch.softmax(_forward_rev(context), dim=-1)

    dist_rev = Categorical(logits=_forward_rev(context))
    old_log_prob_rev = dist_rev.log_prob(z).detach()
    current_log_prob_rev = dist_rev.log_prob(z)
    adv_rev = torch.tensor([-1.0, 1.0], device=dev)
    loss_rev, _ = strategy_ppo_loss(
        current_log_prob_rev,
        old_log_prob_rev,
        adv_rev,
        mask,
        clip_range=clip_range,
        coef=coef,
        device=dev,
    )
    opt_rev = torch.optim.Adam(encoder_rev.parameters(), lr=lr)
    opt_rev.zero_grad()
    loss_rev.backward()
    opt_rev.step()

    with torch.no_grad():
        probs_rev = torch.softmax(_forward_rev(context), dim=-1)

    reversed_passed = (
        float(probs_rev[0, 0].item()) < float(probs_mid[0, 0].item())
        and float(probs_rev[1, 1].item()) > float(probs_mid[1, 1].item())
    )

    return SyntheticSignTestResult(
        passed=passed,
        p_z0_context_a_before=float(probs_before[0, 0].item()),
        p_z0_context_a_after=float(probs_after[0, 0].item()),
        p_z1_context_b_before=float(probs_before[1, 1].item()),
        p_z1_context_b_after=float(probs_after[1, 1].item()),
        reversed_passed=reversed_passed,
        details={
            "loss_requires_grad": bool(loss.requires_grad),
            "selected_log_prob_requires_grad": bool(current_log_prob.requires_grad),
        },
    )


def summarize_router_advantages(
    *,
    router_advantages: Tensor,
    actor_advantages: Tensor,
    returns: Tensor,
    selected_z: Tensor,
    resample_mask: Tensor,
    opponents: Optional[Sequence[str]] = None,
    maps: Optional[Sequence[str]] = None,
) -> dict[str, Any]:
    """Group router advantages to detect z-dependent learning signal."""
    if not bool(resample_mask.any()):
        return {"groups": [], "global": {"count": 0}}

    idx = torch.where(resample_mask)[0]
    adv = router_advantages[idx].detach().cpu()
    act = actor_advantages[idx].detach().cpu()
    ret = returns[idx].detach().cpu()
    z = selected_z[idx].detach().cpu()

    def _corr(a: Tensor, b: Tensor) -> float:
        if a.numel() < 2:
            return float("nan")
        a0 = a - a.mean()
        b0 = b - b.mean()
        denom = (a0.std(unbiased=False) * b0.std(unbiased=False)).item()
        if denom <= 1e-8:
            return float("nan")
        return float((a0 * b0).mean().item() / denom)

    groups: list[RouterAdvantageGroupStats] = []
    for z_val in sorted(int(v) for v in torch.unique(z).tolist()):
        mask = z == z_val
        sub_adv = adv[mask]
        sub_ret = ret[mask]
        groups.append(
            RouterAdvantageGroupStats(
                group_key=f"z{z_val}",
                count=int(mask.sum().item()),
                mean_advantage=float(sub_adv.mean().item()),
                advantage_std=float(sub_adv.std(unbiased=False).item()) if sub_adv.numel() > 1 else 0.0,
                fraction_positive=float((sub_adv > 0).float().mean().item()),
                mean_return=float(sub_ret.mean().item()),
                return_correlation=_corr(sub_adv, sub_ret),
            )
        )

    by_opp: dict[str, list[float]] = defaultdict(list)
    if opponents is not None:
        for i in idx.detach().cpu().tolist():
            by_opp[str(opponents[i])].append(float(router_advantages[i].item()))

    return {
        "global": {
            "count": int(idx.numel()),
            "mean_router_advantage": float(adv.mean().item()),
            "mean_actor_advantage": float(act.mean().item()),
            "router_actor_corr": _corr(adv, act),
            "router_return_corr": _corr(adv, ret),
            "fraction_router_adv_positive": float((adv > 0).float().mean().item()),
            "fraction_router_adv_near_zero": float((adv.abs() < 1e-6).float().mean().item()),
        },
        "by_z": [asdict(g) for g in groups],
        "by_opponent_mean_adv": {k: float(sum(v) / len(v)) for k, v in by_opp.items()},
        "z_occupancy": dict(Counter(int(v) for v in z.tolist())),
    }


def measure_router_loss_gradient_norms(
    *,
    strategy_logits: Tensor,
    strategy_log_prob: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    resample_mask: Tensor,
    entropy: Tensor,
    clip_range: float,
    ppo_coef: float,
    lam_h: float,
    entropy_objective: str,
    router_params: Iterable[torch.nn.Parameter],
) -> RouterGradientNormReport:
    """Separate gradient norms for router PPO policy vs entropy terms."""
    params = [p for p in router_params if p.requires_grad]
    for p in params:
        if p.grad is not None:
            p.grad.zero_()

    pol_scaled, _ = strategy_ppo_loss(
        strategy_log_prob,
        old_log_probs,
        advantages,
        resample_mask,
        clip_range=clip_range,
        coef=ppo_coef,
        device=strategy_logits.device,
    )
    ent_loss, _ = strategy_entropy_loss(
        entropy,
        resample_mask,
        objective=entropy_objective,
        lam_h=lam_h,
        device=strategy_logits.device,
    )

    before = [p.detach().clone() for p in params]
    pol_grad_norm = 0.0
    ent_grad_norm = 0.0

    if pol_scaled.requires_grad:
        pol_scaled.backward(retain_graph=True)
        pol_grad_norm = float(
            torch.sqrt(
                sum((p.grad.detach() ** 2).sum() for p in params if p.grad is not None)
            ).item()
        ) if params else 0.0
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

    if ent_loss.requires_grad:
        ent_loss.backward()
        ent_grad_norm = float(
            torch.sqrt(
                sum((p.grad.detach() ** 2).sum() for p in params if p.grad is not None)
            ).item()
        ) if params else 0.0

    param_delta = 0.0
    changed = False
    for p, b in zip(params, before):
        delta = (p.data - b).norm().item()
        param_delta = max(param_delta, delta)
        changed = changed or delta > 0.0

    eps = 1e-12
    return RouterGradientNormReport(
        router_ppo_grad_norm=pol_grad_norm,
        router_entropy_grad_norm=ent_grad_norm,
        router_critic_grad_norm=0.0,
        ppo_to_entropy_ratio=float(pol_grad_norm / (ent_grad_norm + eps)),
        router_params_changed=changed,
        param_delta_l2=param_delta,
    )


def offline_feedforward_predictability(
    samples: Sequence[Mapping[str, Any]],
    *,
    feature_fn: Callable[[Mapping[str, Any]], Tensor],
    label_key: str = "best_z",
    return_key: str = "return",
    fixed_z2_return: float,
    latent_k: int = 4,
    train_fraction: float = 0.7,
    seed: int = 0,
) -> OfflinePredictabilityReport:
    """Train a tiny linear head on pre-decision features → hindsight best-z."""
    if len(samples) < 4:
        return OfflinePredictabilityReport(
            n_samples=len(samples),
            n_classes=latent_k,
            accuracy=0.0,
            top2_accuracy=0.0,
            mean_regret_vs_best_z=float("nan"),
            mean_return_predicted_z=0.0,
            mean_return_best_fixed_z2=fixed_z2_return,
            beats_chance=False,
            beats_fixed_z2=False,
        )

    xs = torch.stack([feature_fn(s) for s in samples]).float()
    ys = torch.tensor([int(s[label_key]) for s in samples], dtype=torch.long)
    rets = torch.tensor([float(s[return_key]) for s in samples], dtype=torch.float32)
    best_rets = torch.tensor([float(s.get("best_return", s[return_key])) for s in samples], dtype=torch.float32)

    n = xs.shape[0]
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_train = max(2, int(n * train_fraction))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]
    if test_idx.numel() == 0:
        test_idx = train_idx

    in_dim = xs.shape[1]
    head = torch.nn.Linear(in_dim, latent_k)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    for _ in range(200):
        logits = head(xs[train_idx])
        loss = F.cross_entropy(logits, ys[train_idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        test_logits = head(xs[test_idx])
        pred = test_logits.argmax(dim=-1)
        acc = float((pred == ys[test_idx]).float().mean().item())
        top2 = float(
            torch.topk(test_logits, k=min(2, latent_k), dim=-1).indices.eq(ys[test_idx, None]).any(dim=-1).float().mean().item()
        )
        pred_returns = rets[test_idx]
        regret = float((best_rets[test_idx] - pred_returns).mean().item())
        mean_pred_ret = float(pred_returns.mean().item())

    chance = 1.0 / float(latent_k)
    return OfflinePredictabilityReport(
        n_samples=n,
        n_classes=latent_k,
        accuracy=acc,
        top2_accuracy=top2,
        mean_regret_vs_best_z=regret,
        mean_return_predicted_z=mean_pred_ret,
        mean_return_best_fixed_z2=fixed_z2_return,
        beats_chance=acc > chance + 0.05,
        beats_fixed_z2=mean_pred_ret > fixed_z2_return,
    )


def audit_feedforward_router_credit_wiring(cfg: Any, batch: Mapping[str, Tensor]) -> dict[str, Any]:
    """Flag likely credit-assignment mis-wiring for feedforward router stage."""
    from rl.custom_ppo.update.strategy_credit import (
        is_feedforward_sparse_router,
        is_recurrent_router,
        resolve_strategy_advantages,
    )

    recurrent = is_recurrent_router(cfg)
    feedforward_router = is_feedforward_sparse_router(cfg)
    has_router_adv = "router_advantages" in batch
    has_router_reward = "router_reward" in batch
    actor_adv = batch.get("advantages", torch.zeros(1))
    _, source = resolve_strategy_advantages(
        cfg=cfg,
        batch=dict(batch),
        actor_advantages=actor_adv,
    ) if feedforward_router and has_router_adv else (actor_adv, "actor_gae")

    issue = None
    if feedforward_router and not has_router_adv:
        issue = "feedforward router stage enabled but rollout buffer lacks router_advantages"
    elif feedforward_router and source != "router":
        issue = f"expected router advantages for feedforward sparse router, got {source}"

    return {
        "recurrent_selector_active": recurrent,
        "feedforward_sparse_router": feedforward_router,
        "buffer_has_router_advantages": has_router_adv,
        "buffer_has_router_reward": has_router_reward,
        "strategy_ppo_advantage_source": source,
        "credit_wiring_issue": issue,
    }


def _corr(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2:
        return float("nan")
    a0 = a.float() - a.float().mean()
    b0 = b.float() - b.float().mean()
    denom = (a0.std(unbiased=False) * b0.std(unbiased=False)).item()
    if denom <= 1e-8:
        return float("nan")
    return float((a0 * b0).mean().item() / denom)


def audit_router_rollout_dump(
    payload: Mapping[str, Any],
    *,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Generate router credit audit artifacts from a rollout dump file."""
    import csv
    from dataclasses import asdict

    from rl.custom_ppo.diagnostics.router_rollout_dump import (
        _grad_cosine_similarity,
        _replay_router_strategy_policy,
    )
    from rl.custom_ppo.update.strategy_credit import encoder_grad_norm_from_loss, is_feedforward_sparse_router
    from rl.latent_losses import strategy_entropy_loss, strategy_ppo_loss
    from rl.latent_marl import StrategyEncoder

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = dict(payload.get("metadata") or {})
    cfg_dict = payload.get("cfg") or {}


    class _Cfg:
        pass

    cfg = _Cfg()
    for key, value in cfg_dict.items():
        setattr(cfg, key, value)

    def _t(name: str) -> torch.Tensor:
        val = payload.get(name)
        if val is None:
            raise KeyError(f"Dump missing tensor field: {name}")
        return val.detach().cpu()

    mask = payload["router_decision_mask"].bool()
    raw_adv = payload["raw_router_advantages"].float()
    norm_adv = payload.get("normalized_router_advantages", raw_adv).float()
    selected_z = payload["selected_z"].long()
    actor_adv = payload["actor_advantages"].float()
    returns = payload["returns"].float()
    router_reward = payload.get("router_reward")
    opponent_ids = payload.get("opponent_ids")
    map_ids = payload.get("map_ids")
    strategy_context = payload["strategy_context"].float()

    sel_raw = raw_adv[mask]
    sel_ret = returns[mask]
    z_vals = sorted(int(v) for v in torch.unique(selected_z[mask]).tolist())

    by_z_rows: list[dict[str, Any]] = []
    z_means: list[float] = []
    for z in z_vals:
        zmask = mask & (selected_z == z)
        sub = raw_adv[zmask]
        mean_adv = float(sub.mean().item()) if sub.numel() else float("nan")
        z_means.append(mean_adv)
        by_z_rows.append(
            {
                "z": z,
                "count": int(zmask.sum().item()),
                "mean_router_advantage": mean_adv,
                "std_router_advantage": float(sub.std(unbiased=False).item()) if sub.numel() > 1 else 0.0,
                "fraction_positive": float((sub > 0).float().mean().item()) if sub.numel() else 0.0,
                "mean_return": float(returns[zmask].mean().item()) if zmask.any() else float("nan"),
            }
        )

    between_z_mean_diff = float(max(z_means) - min(z_means)) if z_means else 0.0
    within_z_vars = [float(r["std_router_advantage"]) ** 2 for r in by_z_rows]
    within_z_variance = float(sum(within_z_vars) / len(within_z_vars)) if within_z_vars else 0.0

    summary = {
        "router_advantage_mean": float(sel_raw.mean().item()) if sel_raw.numel() else 0.0,
        "router_advantage_std": float(sel_raw.std(unbiased=False).item()) if sel_raw.numel() > 1 else 0.0,
        "fraction_positive": float((sel_raw > 0).float().mean().item()) if sel_raw.numel() else 0.0,
        "between_z_mean_difference": between_z_mean_diff,
        "within_z_variance": within_z_variance,
        "advantage_return_correlation": _corr(sel_raw, sel_ret),
        "advantage_router_reward_correlation": _corr(sel_raw, router_reward[mask].float())
        if router_reward is not None
        else float("nan"),
        "decision_opportunity_count": int(mask.sum().item()),
        "advantage_source_used": meta.get("advantage_source_used", payload.get("advantage_source_used")),
        "metadata": meta,
    }

    by_opp_rows: list[dict[str, Any]] = []
    if opponent_ids is not None:
        for opp in sorted(int(v) for v in torch.unique(opponent_ids[mask]).tolist()):
            om = mask & (opponent_ids == opp)
            sub = raw_adv[om]
            by_opp_rows.append(
                {
                    "opponent_id": opp,
                    "count": int(om.sum().item()),
                    "mean_router_advantage": float(sub.mean().item()),
                    "std_router_advantage": float(sub.std(unbiased=False).item()) if sub.numel() > 1 else 0.0,
                }
            )

    by_opp_z_rows: list[dict[str, Any]] = []
    if opponent_ids is not None:
        for opp in sorted(int(v) for v in torch.unique(opponent_ids[mask]).tolist()):
            for z in z_vals:
                om = mask & (opponent_ids == opp) & (selected_z == z)
                if not bool(om.any().item()):
                    continue
                sub = raw_adv[om]
                by_opp_z_rows.append(
                    {
                        "opponent_id": opp,
                        "z": z,
                        "count": int(om.sum().item()),
                        "mean_router_advantage": float(sub.mean().item()),
                    }
                )

    by_map_z_rows: list[dict[str, Any]] = []
    if map_ids is not None:
        for mid in sorted(int(v) for v in torch.unique(map_ids[mask]).tolist()):
            for z in z_vals:
                mm = mask & (map_ids == mid) & (selected_z == z)
                if not bool(mm.any().item()):
                    continue
                sub = raw_adv[mm]
                by_map_z_rows.append(
                    {
                        "map_id": mid,
                        "map_layout": meta.get("map_layout", ""),
                        "z": z,
                        "count": int(mm.sum().item()),
                        "mean_router_advantage": float(sub.mean().item()),
                    }
                )

    latent_k = int(meta.get("latent_k", getattr(cfg, "latent_k", 4)))
    context_dim = int(strategy_context.shape[-1])
    encoder = StrategyEncoder(state_dim=context_dim, latent_k=latent_k, hidden=64)
    if "strategy_encoder_state" in payload:
        encoder.load_state_dict(payload["strategy_encoder_state"])

    decision_logits = payload["router_logits"][mask]
    decision_z = selected_z[mask]
    decision_old = payload["old_router_log_prob"][mask]
    decision_adv = raw_adv[mask]
    clip_range = float(getattr(cfg, "clip_range", 0.2))
    ppo_coef = float(getattr(cfg, "latent_strategy_ppo_coef", 0.10))
    offline_loss, offline_grad_norm, offline_grad_flat, offline_stats = _replay_router_strategy_policy(
        encoder,
        ctx=strategy_context[mask],
        z=decision_z,
        old_log_prob=decision_old,
        strat_adv=decision_adv,
        clip_range=clip_range,
        coef=ppo_coef,
    )
    entropy = Categorical(logits=decision_logits).entropy()
    lam_h = float(getattr(cfg, "router_ent_coef", 0.005) or 0.0)
    ent_loss, _ = strategy_entropy_loss(
        entropy,
        torch.ones_like(entropy, dtype=torch.bool),
        objective=str(getattr(cfg, "latent_entropy_objective", "none") or "none"),
        lam_h=lam_h,
        device=entropy.device,
    )
    pol_grad = offline_grad_norm
    ent_grad = encoder_grad_norm_from_loss(ent_loss, encoder)

    loss_replay = dict(payload.get("loss_replay") or {})
    online_loss = float(loss_replay.get("online_strategy_policy_loss", offline_loss))
    online_grad_flat = loss_replay.get("policy_grad_flat")
    if isinstance(online_grad_flat, torch.Tensor):
        grad_cosine = _grad_cosine_similarity(online_grad_flat.float(), offline_grad_flat.float())
    else:
        grad_cosine = float("nan")
    gradient_report = {
        "online_strategy_policy_loss": online_loss,
        "offline_replayed_strategy_policy_loss": offline_loss,
        "strategy_policy_loss_abs_diff": abs(online_loss - offline_loss),
        "policy_grad_cosine_similarity": grad_cosine,
        "online_policy_grad_norm": float(loss_replay.get("online_policy_grad_norm", offline_grad_norm)),
        "offline_policy_grad_norm": offline_grad_norm,
        "router_ppo_grad_norm": pol_grad,
        "router_entropy_grad_norm": ent_grad,
        "policy_to_entropy_grad_ratio": float(pol_grad / (ent_grad + 1e-8)),
        "approx_kl": float(offline_stats["approx_kl"].item()),
        "scaled_strategy_policy_loss": float(offline_stats["policy_loss"].item() * ppo_coef),
    }

    # Context -> z predictability on decision rows only.
    decision_indices = torch.where(mask)[0]
    samples = []
    for j, idx in enumerate(decision_indices.tolist()):
        samples.append(
            {
                "best_z": int(selected_z[idx].item()),
                "return": float(returns[idx].item()),
                "best_return": float(returns[idx].item()),
                "_feature_index": j,
            }
        )

    def feature_fn(sample: Mapping[str, Any]) -> torch.Tensor:
        return strategy_context[decision_indices[int(sample["_feature_index"])]]

    predictability = asdict(
        offline_feedforward_predictability(
            samples,
            feature_fn=feature_fn,
            fixed_z2_return=float(raw_adv[(mask & (selected_z == 2))].mean().item())
            if bool((mask & (selected_z == 2)).any().item())
            else 0.0,
            latent_k=latent_k,
        )
    )

    def _write_csv(name: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        path = out / name
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    (out / "router_advantage_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    _write_csv("router_advantage_by_z.csv", by_z_rows)
    _write_csv("router_advantage_by_opponent.csv", by_opp_rows)
    _write_csv("router_advantage_by_opponent_z.csv", by_opp_z_rows)
    _write_csv("router_advantage_by_map_z.csv", by_map_z_rows)
    (out / "gradient_component_report.json").write_text(
        json.dumps(gradient_report, indent=2),
        encoding="utf-8",
    )
    (out / "context_predictability_report.json").write_text(
        json.dumps(predictability, indent=2),
        encoding="utf-8",
    )

    verdict = {
        "healthy_credit_signal": (
            summary["router_advantage_std"] > 1e-4
            and between_z_mean_diff > 1e-4
            and abs(summary["advantage_router_reward_correlation"]) > 0.05
            if router_reward is not None
            else summary["router_advantage_std"] > 1e-4
        ),
        "flat_credit": summary["router_advantage_std"] < 1e-5,
        "entropy_dominated": gradient_report["policy_to_entropy_grad_ratio"] < 0.25,
        "context_insufficient": not bool(predictability.get("beats_fixed_z2", False)),
        "feedforward_sparse_router": is_feedforward_sparse_router(cfg),
    }

    return {
        "summary": summary,
        "gradient_component_report": gradient_report,
        "context_predictability_report": predictability,
        "verdict": verdict,
        "output_dir": str(out),
    }
