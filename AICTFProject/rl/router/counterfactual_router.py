"""V6I25 counterfactual geometry→z router helpers (corrected protocol).

Primary objective (soft Q-targets from train-seed means):

    Q̂(c, z) = E_train[R | c, z]
    p*(z|c) = softmax(Q̂(c, z) / τ)
    L = - Σ_z p*(z|c) log q_φ(z|c)

Cross-fitted context oracle (not per-episode hindsight max):

    z*(c) = argmax_z Q̂_train(c, z)
    R_context-oracle = R_heldout(c, z*(c))

Centered-advantage loss is retained only as an ablation helper.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

Verdict = Literal["PASS", "PARTIAL", "FAIL_SIGNAL", "FAIL_ROUTER"]


# ---------------------------------------------------------------------------
# Pure tensor / array helpers
# ---------------------------------------------------------------------------


def advantages_from_returns(returns: torch.Tensor) -> torch.Tensor:
    """Center returns within each row: ``A = R - mean_z(R)``. Ablation helper."""
    r = returns.float()
    if r.dim() == 1:
        r = r.unsqueeze(0)
    if r.dim() != 2:
        raise ValueError(f"returns must be (B, K) or (K,), got {tuple(r.shape)}")
    return r - r.mean(dim=-1, keepdim=True)


def soft_target_from_q(
    q_values: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """``p*(z|c) = softmax(Q / τ)``. ``q_values`` shape ``(B, K)`` or ``(K,)``."""
    q = q_values.float()
    if q.dim() == 1:
        q = q.unsqueeze(0)
    tau = max(1e-6, float(temperature))
    return F.softmax(q / tau, dim=-1)


def soft_q_router_loss(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    spread_floor: float = 1e-6,
    q_for_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Cross-entropy ``-Σ p* log q_φ``; drop rows with negligible Q spread."""
    if logits.shape != target_probs.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} != targets shape {tuple(target_probs.shape)}"
        )
    log_q = F.log_softmax(logits.float(), dim=-1)
    p = target_probs.float()
    spread_src = q_for_mask.float() if q_for_mask is not None else p
    row_spread = spread_src.amax(dim=-1) - spread_src.amin(dim=-1)
    mask = row_spread >= float(spread_floor)
    if not bool(mask.any()):
        return logits.float().sum() * 0.0
    per_row = -(p * log_q).sum(dim=-1)
    return per_row[mask].mean()


def counterfactual_router_loss(
    logits: torch.Tensor,
    advantages: torch.Tensor,
    *,
    advantage_floor: float = 1e-6,
) -> torch.Tensor:
    """Ablation: ``L = -Σ_z A_z log q(z|c)`` with centered advantages."""
    if logits.shape != advantages.shape:
        raise ValueError(
            f"logits shape {tuple(logits.shape)} != advantages shape {tuple(advantages.shape)}"
        )
    log_q = F.log_softmax(logits.float(), dim=-1)
    adv = advantages.float()
    row_scale = adv.abs().amax(dim=-1)
    mask = row_scale >= float(advantage_floor)
    if not bool(mask.any()):
        return logits.float().sum() * 0.0
    per_row = -(adv * log_q).sum(dim=-1)
    return per_row[mask].mean()


def geometry_key(context: np.ndarray, *, decimals: int = 4) -> tuple[float, ...]:
    """Quantize continuous geometry so identical starts collide across opponents."""
    arr = np.asarray(context, dtype=np.float64).reshape(-1)
    return tuple(np.round(arr, decimals=int(decimals)).tolist())


def assert_valid_geometry_context(
    context: np.ndarray | torch.Tensor,
    *,
    name: str = "context",
) -> np.ndarray:
    """Fail loudly on missing / non-finite / all-zero geometry."""
    arr = np.asarray(context, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name}: empty geometry context")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name}: non-finite values in geometry context")
    if float(np.abs(arr).sum()) <= 0.0:
        raise ValueError(f"{name}: all-zero geometry context (global_state missing?)")
    return arr


def geometry_context_report(contexts: np.ndarray) -> dict[str, Any]:
    """Diagnostics for learnable variation in episode-start geometry."""
    x = np.asarray(contexts, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    keys = [geometry_key(row) for row in x]
    unique = set(keys)
    n = len(keys)
    n_unique = len(unique)
    return {
        "n_rows": n,
        "n_unique_contexts": n_unique,
        "duplicate_context_rate": float(1.0 - (n_unique / max(1, n))),
        "feature_variance_mean": float(np.var(x, axis=0).mean()) if n > 1 else 0.0,
        "feature_variance_max": float(np.var(x, axis=0).max()) if n > 1 else 0.0,
        "context_abs_sum_mean": float(np.abs(x).sum(axis=1).mean()),
    }


# ---------------------------------------------------------------------------
# Cross-fitted geometry Q / oracle
# ---------------------------------------------------------------------------


@dataclass
class GeometryQTable:
    """Train-seed mean returns keyed by quantized geometry (opponent-agnostic)."""

    q_by_key: dict[tuple[float, ...], np.ndarray]  # key -> (K,)
    counts_by_key: dict[tuple[float, ...], int]
    latent_k: int

    def q(self, key: tuple[float, ...]) -> np.ndarray | None:
        return self.q_by_key.get(key)

    def z_star(self, key: tuple[float, ...]) -> int | None:
        q = self.q(key)
        if q is None:
            return None
        return int(np.argmax(q))


def build_geometry_q_table(
    contexts: np.ndarray,
    returns: np.ndarray,
    *,
    decimals: int = 4,
) -> GeometryQTable:
    """Aggregate ``E[R|c,z]`` across all rows sharing a quantized geometry.

    Opponent identity is intentionally discarded: conflicting opponent rows
    under the same start geometry are averaged together.
    """
    ctx = np.asarray(contexts, dtype=np.float64)
    ret = np.asarray(returns, dtype=np.float64)
    if ctx.ndim != 2 or ret.ndim != 2 or ctx.shape[0] != ret.shape[0]:
        raise ValueError("contexts (N,C) and returns (N,K) row counts must match")
    k = int(ret.shape[1])
    buckets: dict[tuple[float, ...], list[np.ndarray]] = {}
    for i in range(ctx.shape[0]):
        key = geometry_key(ctx[i], decimals=decimals)
        buckets.setdefault(key, []).append(ret[i])
    q_by_key: dict[tuple[float, ...], np.ndarray] = {}
    counts: dict[tuple[float, ...], int] = {}
    for key, rows in buckets.items():
        stacked = np.stack(rows, axis=0)
        q_by_key[key] = stacked.mean(axis=0)
        counts[key] = int(stacked.shape[0])
    return GeometryQTable(q_by_key=q_by_key, counts_by_key=counts, latent_k=k)


def assign_cross_fitted_z(
    contexts: np.ndarray,
    q_table: GeometryQTable,
    *,
    decimals: int = 4,
    fallback_z: int = 0,
) -> np.ndarray:
    """``z*(c)`` from train Q-table for each held-out geometry row."""
    ctx = np.asarray(contexts, dtype=np.float64)
    out = np.empty(ctx.shape[0], dtype=np.int64)
    for i in range(ctx.shape[0]):
        key = geometry_key(ctx[i], decimals=decimals)
        z = q_table.z_star(key)
        out[i] = int(fallback_z if z is None else z)
    return out


def soft_targets_from_geometry_q(
    contexts: np.ndarray,
    q_table: GeometryQTable,
    *,
    temperature: float = 1.0,
    decimals: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ``(N, K)`` soft targets and Q rows aligned to ``contexts``."""
    ctx = np.asarray(contexts, dtype=np.float64)
    k = int(q_table.latent_k)
    q_rows = np.zeros((ctx.shape[0], k), dtype=np.float64)
    for i in range(ctx.shape[0]):
        key = geometry_key(ctx[i], decimals=decimals)
        q = q_table.q(key)
        if q is None:
            q_rows[i] = 0.0
        else:
            q_rows[i] = q
    targets = soft_target_from_q(torch.as_tensor(q_rows), temperature=temperature).numpy()
    return targets, q_rows


# ---------------------------------------------------------------------------
# Scoring / verdicts
# ---------------------------------------------------------------------------


@dataclass
class PairedScore:
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    ci_excludes_zero_positive: bool
    n: int


def paired_delta_ci(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> PairedScore:
    a = np.asarray(returns_a, dtype=np.float64).reshape(-1)
    b = np.asarray(returns_b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("paired returns must match shape")
    delta = a - b
    lo, hi = _bootstrap_mean_ci(delta, n_bootstrap=n_bootstrap, seed=seed)
    return PairedScore(
        mean_a=float(a.mean()) if a.size else float("nan"),
        mean_b=float(b.mean()) if b.size else float("nan"),
        delta=float(delta.mean()) if delta.size else float("nan"),
        ci_low=float(lo),
        ci_high=float(hi),
        ci_excludes_zero_positive=bool(lo > 0.0),
        n=int(a.size),
    )


@dataclass
class StageAResult:
    context_oracle_mean: float
    best_fixed_mean: float
    best_fixed_z: int
    delta: float
    ci_low: float
    ci_high: float
    signal_ok: bool
    n: int

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def stage_a_signal_validation(
    heldout_returns: np.ndarray,
    heldout_contexts: np.ndarray,
    train_q_table: GeometryQTable,
    *,
    train_returns_for_best_fixed: np.ndarray,
    decimals: int = 4,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> StageAResult:
    """Require cross-fitted geometry oracle > best-fixed on held-out seeds."""
    r = np.asarray(heldout_returns, dtype=np.float64)
    n, k = r.shape
    train_r = np.asarray(train_returns_for_best_fixed, dtype=np.float64)
    best_fixed_z = int(np.argmax(train_r.mean(axis=0)))
    best_fixed = r[:, best_fixed_z]
    z_star = assign_cross_fitted_z(
        heldout_contexts, train_q_table, decimals=decimals, fallback_z=best_fixed_z
    )
    ctx_oracle = r[np.arange(n), z_star]
    paired = paired_delta_ci(ctx_oracle, best_fixed, n_bootstrap=n_bootstrap, seed=seed)
    return StageAResult(
        context_oracle_mean=paired.mean_a,
        best_fixed_mean=paired.mean_b,
        best_fixed_z=best_fixed_z,
        delta=paired.delta,
        ci_low=paired.ci_low,
        ci_high=paired.ci_high,
        signal_ok=bool(paired.ci_excludes_zero_positive),
        n=n,
    )


@dataclass
class StageBResult:
    router_mean: float
    uniform_mean: float
    best_fixed_mean: float
    context_oracle_mean: float
    best_fixed_z: int
    delta_router_minus_best_fixed: float
    router_ci_low: float
    router_ci_high: float
    router_beats_best_fixed: bool
    gap_recovery: float
    n: int

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def stage_b_router_eval(
    heldout_returns: np.ndarray,
    router_z: np.ndarray,
    *,
    context_oracle_z: np.ndarray,
    best_fixed_z: int,
    gap_recovery_threshold: float = 0.5,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> StageBResult:
    r = np.asarray(heldout_returns, dtype=np.float64)
    n = r.shape[0]
    idx = np.arange(n)
    router = r[idx, np.asarray(router_z, dtype=np.int64)]
    uniform = r.mean(axis=1)
    best_fixed = r[:, int(best_fixed_z)]
    ctx_oracle = r[idx, np.asarray(context_oracle_z, dtype=np.int64)]
    paired = paired_delta_ci(router, best_fixed, n_bootstrap=n_bootstrap, seed=seed)
    denom = float(ctx_oracle.mean() - best_fixed.mean())
    if abs(denom) < 1e-12:
        gap_recovery = float("nan")
    else:
        gap_recovery = float(paired.delta / denom)
    return StageBResult(
        router_mean=float(router.mean()),
        uniform_mean=float(uniform.mean()),
        best_fixed_mean=float(best_fixed.mean()),
        context_oracle_mean=float(ctx_oracle.mean()),
        best_fixed_z=int(best_fixed_z),
        delta_router_minus_best_fixed=paired.delta,
        router_ci_low=paired.ci_low,
        router_ci_high=paired.ci_high,
        router_beats_best_fixed=bool(paired.ci_excludes_zero_positive),
        gap_recovery=gap_recovery,
        n=n,
    )


def decide_v6i25_verdict(
    stage_a: StageAResult,
    stage_b: StageBResult | None,
    *,
    gap_recovery_threshold: float = 0.5,
) -> Verdict:
    """Pre-registered V6I25 verdict (cross-fitted context oracle, not hindsight)."""
    if not stage_a.signal_ok:
        return "FAIL_SIGNAL"
    if stage_b is None:
        return "FAIL_ROUTER"
    recovery = stage_b.gap_recovery
    recovery_ok = recovery == recovery and recovery >= float(gap_recovery_threshold)
    if stage_b.router_beats_best_fixed and recovery_ok:
        return "PASS"
    if stage_b.router_beats_best_fixed and not recovery_ok:
        return "PARTIAL"
    # Signal exists but router does not beat best-fixed (CI).
    return "FAIL_ROUTER"


def _bootstrap_mean_ci(
    values: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        sample = rng.choice(v, size=v.size, replace=True)
        means[i] = float(sample.mean())
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def train_test_split_indices(
    n: int,
    *,
    test_frac: float = 0.25,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    if n <= 1:
        idx = np.arange(n)
        return idx, idx
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_test = max(1, int(round(float(test_frac) * n)))
    n_test = min(n_test, n - 1)
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    return train_idx, test_idx


# ---------------------------------------------------------------------------
# Model train / freeze helpers
# ---------------------------------------------------------------------------


@dataclass
class CFRouterTrainResult:
    n_steps: int
    loss_mean: float
    n_rows: int
    n_rows_used: float
    loss_mode: str


def freeze_non_router_parameters(model: torch.nn.Module) -> list[str]:
    trainable_prefixes = ("strategy_encoder.", "selector_gru.")
    frozen: list[str] = []
    for name, p in model.named_parameters():
        if any(name == pref[:-1] or name.startswith(pref) for pref in trainable_prefixes):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
            frozen.append(name)
    return frozen


def reinitialize_q_phi(model: torch.nn.Module) -> list[str]:
    """Fresh router init (plan: initialize router fresh). Returns module names reset."""
    reset: list[str] = []
    for mod_name in ("strategy_encoder", "selector_gru"):
        mod = getattr(model, mod_name, None)
        if mod is None:
            continue
        for module in mod.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
        reset.append(mod_name)
    return reset


def _iter_q_phi_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    params: list[torch.nn.Parameter] = []
    for mod_name in ("strategy_encoder", "selector_gru"):
        mod = getattr(model, mod_name, None)
        if mod is None:
            continue
        for p in mod.parameters():
            p.requires_grad_(True)
            params.append(p)
    seen: set[int] = set()
    unique: list[torch.nn.Parameter] = []
    for p in params:
        ptr = int(p.data_ptr())
        if ptr in seen:
            continue
        seen.add(ptr)
        unique.append(p)
    return unique


def _strategy_logits_batch(model: torch.nn.Module, contexts: torch.Tensor) -> torch.Tensor:
    if not bool(getattr(model, "uses_latent_strategy", False)):
        raise RuntimeError("counterfactual router requires uses_latent_strategy=True")
    B = int(contexts.shape[0])
    device = contexts.device
    if getattr(model, "selector_gru", None) is not None:
        h0 = torch.zeros(
            B,
            int(model.recurrent_selector_hidden_dim),
            device=device,
            dtype=contexts.dtype,
        )
        return model.strategy_logits(contexts, selector_hidden=h0)
    return model.strategy_logits(contexts)


@torch.no_grad()
def predict_router_z(model: torch.nn.Module, contexts: torch.Tensor) -> torch.Tensor:
    model.eval()
    logits = _strategy_logits_batch(model, contexts.float())
    return logits.argmax(dim=-1)


def train_counterfactual_router(
    model: torch.nn.Module,
    contexts: torch.Tensor,
    target_probs: torch.Tensor,
    *,
    q_values: torch.Tensor | None = None,
    n_steps: int = 200,
    batch_size: int = 64,
    lr: float = 1e-3,
    spread_floor: float = 1e-6,
    device: torch.device | str = "cpu",
    seed: int = 0,
    loss_mode: str = "soft_q",
) -> CFRouterTrainResult:
    """Train only q_phi. Primary ``loss_mode='soft_q'``; ``'advantage'`` is ablation."""
    device_t = torch.device(device)
    contexts = contexts.to(device_t).float()
    target_probs = target_probs.to(device_t).float()
    q_values_t = q_values.to(device_t).float() if q_values is not None else None

    router_params = list(_iter_q_phi_parameters(model))
    if not router_params:
        raise RuntimeError("No q_phi parameters found on model (strategy_encoder missing?)")

    opt = torch.optim.Adam(router_params, lr=float(lr))
    n = int(contexts.shape[0])
    if n == 0:
        return CFRouterTrainResult(
            n_steps=0, loss_mean=float("nan"), n_rows=0, n_rows_used=0.0, loss_mode=loss_mode
        )

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    losses: list[float] = []
    used_fracs: list[float] = []

    model.train()
    for _ in range(int(n_steps)):
        if n <= batch_size:
            idx = torch.arange(n)
        else:
            idx = torch.randint(0, n, (batch_size,), generator=g)
        ctx_b = contexts[idx]
        logits = _strategy_logits_batch(model, ctx_b)
        if loss_mode == "advantage":
            # Ablation only: ``target_probs`` slot carries raw returns when q_values is None.
            src = q_values_t[idx] if q_values_t is not None else target_probs[idx]
            adv_b = advantages_from_returns(src)
            row_scale = adv_b.abs().amax(dim=-1)
            used_fracs.append(float((row_scale >= float(spread_floor)).float().mean().item()))
            loss = counterfactual_router_loss(logits, adv_b, advantage_floor=spread_floor)
        else:
            p_b = target_probs[idx]
            q_b = q_values_t[idx] if q_values_t is not None else p_b
            row_spread = q_b.amax(dim=-1) - q_b.amin(dim=-1)
            used_fracs.append(float((row_spread >= float(spread_floor)).float().mean().item()))
            loss = soft_q_router_loss(
                logits, p_b, spread_floor=spread_floor, q_for_mask=q_b
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router_params, 1.0)
        opt.step()
        losses.append(float(loss.detach().item()))

    return CFRouterTrainResult(
        n_steps=int(n_steps),
        loss_mean=float(np.mean(losses)) if losses else float("nan"),
        n_rows=n,
        n_rows_used=float(np.mean(used_fracs)) if used_fracs else 0.0,
        loss_mode=str(loss_mode),
    )


def prepare_v6i7_episode_start_context(global_state: np.ndarray | torch.Tensor) -> np.ndarray:
    """Pad raw 34-d ``env.state()`` with strategy-phase 0 → 35-d V6I7 context."""
    from rl.global_state import GLOBAL_STATE_DIM, GLOBAL_STATE_V6I7_DIM

    gs = assert_valid_geometry_context(global_state, name="global_state")
    if gs.size == GLOBAL_STATE_V6I7_DIM:
        out = gs.copy()
        out[-1] = 0.0  # episode-start phase
        return assert_valid_geometry_context(out, name="v6i7_context")
    if gs.size != GLOBAL_STATE_DIM:
        raise ValueError(
            f"global_state dim {gs.size} not in {{{GLOBAL_STATE_DIM}, {GLOBAL_STATE_V6I7_DIM}}}"
        )
    out = np.concatenate([gs, np.zeros(1, dtype=np.float64)], axis=0)
    return assert_valid_geometry_context(out, name="v6i7_context")


__all__ = [
    "CFRouterTrainResult",
    "GeometryQTable",
    "PairedScore",
    "StageAResult",
    "StageBResult",
    "Verdict",
    "advantages_from_returns",
    "assert_valid_geometry_context",
    "assign_cross_fitted_z",
    "build_geometry_q_table",
    "counterfactual_router_loss",
    "decide_v6i25_verdict",
    "freeze_non_router_parameters",
    "geometry_context_report",
    "geometry_key",
    "paired_delta_ci",
    "prepare_v6i7_episode_start_context",
    "predict_router_z",
    "reinitialize_q_phi",
    "soft_q_router_loss",
    "soft_target_from_q",
    "soft_targets_from_geometry_q",
    "stage_a_signal_validation",
    "stage_b_router_eval",
    "train_counterfactual_router",
    "train_test_split_indices",
]
