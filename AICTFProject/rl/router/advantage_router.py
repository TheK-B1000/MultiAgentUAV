"""V6I12 paired-advantage router.

Splits the noisy raw-return Q signal into two orthogonal components:

  V(context)          — context baseline: E[normalized_return | context]
  A(context, z) = normalized_return - stopgrad(V(context))

The raw return variance (~2.6–3.9 std) is dominated by context variation
(who your opponent is, what map, how the episode played out).  V(context)
absorbs that component; A(context, z) isolates the latent-specific residual
which is the actual routing signal.

Double-centering:
  1. Global:   norm_ret = (episode_return - running_mean) / running_std
  2. Context:  a_target = norm_ret - stopgrad(V(context))

Route: argmax_z A(context, z)

Pass condition: ≥2 opponents with advantage gap CI excluding zero
(not spread alone — V(context) may compress advantages relative to raw Q).
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.global_state import GLOBAL_STATE_DIM
from rl.router.q_value_router import QRouterReplayBuffer, copy_arc_record  # noqa: F401 re-export

_DEFAULT_OPPONENT_ID_TO_IDX: dict[int, int] = {7: 0, 8: 1, 9: 2}


class ContextualVBaseline(nn.Module):
    """MLP: context → scalar V(context).

    Predicts normalized episode return given the episode-start context.
    Used as the baseline to subtract from the Q-router targets.
    """

    def __init__(
        self,
        state_dim: int = GLOBAL_STATE_DIM,
        n_opponents: int = 3,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.n_opponents = int(n_opponents)
        context_dim = self.state_dim + self.n_opponents
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    @property
    def context_dim(self) -> int:
        return self.state_dim + self.n_opponents

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Return V predictions shape [B]."""
        return self.net(context).squeeze(-1)


class AdvantageRouter(nn.Module):
    """MLP: context → A(context, z) for each z.

    Predicts the advantage of each latent given the episode-start context,
    after removing the context baseline V(context).
    """

    def __init__(
        self,
        state_dim: int = GLOBAL_STATE_DIM,
        n_opponents: int = 3,
        opponent_id_to_idx: dict[int, int] | None = None,
        latent_k: int = 4,
        hidden: int = 128,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.n_opponents = int(n_opponents)
        self.latent_k = int(latent_k)
        self.opponent_id_to_idx = (
            dict(opponent_id_to_idx) if opponent_id_to_idx is not None
            else dict(_DEFAULT_OPPONENT_ID_TO_IDX)
        )
        context_dim = self.state_dim + self.n_opponents
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.latent_k),
        )
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    @property
    def context_dim(self) -> int:
        return self.state_dim + self.n_opponents

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Return advantage values shape [B, K]."""
        return self.net(context)

    def build_context(
        self,
        global_state: torch.Tensor,
        opponent_ids_raw: torch.Tensor | list[int],
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = global_state.device
        device = torch.device(device)
        if isinstance(opponent_ids_raw, (list, np.ndarray)):
            opp_t = torch.tensor(opponent_ids_raw, dtype=torch.long, device=device)
        else:
            opp_t = opponent_ids_raw.to(device=device, dtype=torch.long)
        B = int(global_state.shape[0])
        gs = global_state[:, : self.state_dim].to(device=device, dtype=torch.float32)
        opp_onehot = torch.zeros(B, self.n_opponents, device=device, dtype=torch.float32)
        for raw_id, idx in self.opponent_id_to_idx.items():
            mask = opp_t == int(raw_id)
            if bool(mask.any()):
                opp_onehot[mask, idx] = 1.0
        return torch.cat([gs, opp_onehot], dim=-1)

    def build_context_from_record(self, rec: dict[str, Any]) -> torch.Tensor:
        gs = rec["global_state_0"].unsqueeze(0).float()
        opp_raw = int(rec.get("opponent_id", -1))
        return self.build_context(gs, [opp_raw]).squeeze(0)

    @torch.no_grad()
    def advantage_matrix(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return A-value matrix [N_opp, K] using zero-geometry, opp-only context."""
        device = torch.device(device)
        self.eval()
        rows = []
        for opp_raw in sorted(self.opponent_id_to_idx):
            gs_zero = torch.zeros(1, self.state_dim, device=device)
            ctx = self.build_context(gs_zero, [opp_raw], device=device)
            rows.append(self(ctx))
        return torch.cat(rows, dim=0)


def train_advantage_router(
    v_baseline: ContextualVBaseline,
    a_router: AdvantageRouter,
    replay: QRouterReplayBuffer,
    v_optimizer: torch.optim.Optimizer,
    a_optimizer: torch.optim.Optimizer,
    *,
    batch_size: int = 256,
    n_steps: int = 20,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    """Train V-baseline and A-router for ``n_steps`` gradient steps.

    Per-batch double-centering:
      1. Normalize returns globally within the minibatch.
      2. Train V to predict the normalized return.
      3. Advantage target = normalized_return - stopgrad(V(context)).
      4. Train A to predict the advantage target for the selected z.

    Returns telemetry including baseline_r2, advantage_target_std, losses.
    """
    if len(replay) < max(batch_size // 4, 16):
        return {"v_loss_mean": float("nan"), "a_loss_mean": float("nan"), "a_steps": 0}

    device = torch.device(device)
    v_baseline.to(device).train()
    a_router.to(device).train()

    v_losses: list[float] = []
    a_losses: list[float] = []
    v_grad_norms: list[float] = []
    a_grad_norms: list[float] = []
    baseline_r2_vals: list[float] = []
    adv_target_stds: list[float] = []

    for _ in range(n_steps):
        ctx, z, ret, _opp_idx = replay.sample(batch_size, device=device)

        # Global normalization within the minibatch.
        norm_ret = (ret - ret.mean()) / (ret.std() + 1e-8)

        # --- V update ---
        v_pred = v_baseline(ctx)
        v_loss = F.mse_loss(v_pred, norm_ret.detach())
        v_optimizer.zero_grad()
        v_loss.backward()
        vgn = float(torch.nn.utils.clip_grad_norm_(v_baseline.parameters(), 1.0).item())
        v_optimizer.step()

        # --- Baseline quality: R² ---
        with torch.no_grad():
            ss_res = float(((norm_ret - v_pred.detach()) ** 2).sum().item())
            ss_tot = float(((norm_ret - norm_ret.mean()) ** 2).sum().item())
            r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        baseline_r2_vals.append(r2)

        # --- A update: target = norm_ret - stopgrad(V(context)) ---
        with torch.no_grad():
            v_pred2 = v_baseline(ctx)
        a_target = norm_ret.detach() - v_pred2.detach()
        adv_target_stds.append(float(a_target.std().item()))

        a_vals = a_router(ctx)  # [B, K]
        a_pred = a_vals.gather(1, z.unsqueeze(1)).squeeze(1)  # [B]
        a_loss = F.huber_loss(a_pred, a_target)
        a_optimizer.zero_grad()
        a_loss.backward()
        agn = float(torch.nn.utils.clip_grad_norm_(a_router.parameters(), 1.0).item())
        a_optimizer.step()

        v_losses.append(float(v_loss.item()))
        a_losses.append(float(a_loss.item()))
        v_grad_norms.append(vgn)
        a_grad_norms.append(agn)

    v_baseline.eval()
    a_router.eval()
    return {
        "v_loss_mean": float(np.mean(v_losses)),
        "a_loss_mean": float(np.mean(a_losses)),
        "v_grad_norm": float(np.mean(v_grad_norms)),
        "a_grad_norm": float(np.mean(a_grad_norms)),
        "baseline_r2_mean": float(np.mean(baseline_r2_vals)),
        "advantage_target_std_mean": float(np.mean(adv_target_stds)),
        "a_steps": n_steps,
    }


def advantage_gap_ci(
    replay: QRouterReplayBuffer,
    v_baseline: ContextualVBaseline,
    a_router: AdvantageRouter,
    *,
    n_opponents: int,
    latent_k: int,
    opponent_id_to_idx: dict[int, int],
    device: str = "cpu",
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Bootstrap CI on best-vs-second advantage gap per opponent.

    Computes empirical advantage targets (norm_ret - stopgrad(V)) for each
    (opponent, z) cell and bootstraps the gap between the best-z and
    second-best-z mean advantage.
    """
    z_all, ret_all, opp_all = replay.raw_arrays()
    if ret_all.size == 0:
        return {}

    device_t = torch.device(device)
    v_baseline.to(device_t).eval()
    a_router.to(device_t).eval()

    from rl.custom_ppo.csv_writers import _opponent_tag_from_id
    opp_names = {idx: _opponent_tag_from_id(int(raw)) for raw, idx in opponent_id_to_idx.items()}

    # Compute advantage targets for all stored records.
    n = int(ret_all.size)
    ctx_tensor = replay._contexts[:n].to(device_t)
    ret_tensor = torch.tensor(ret_all, dtype=torch.float32, device=device_t)

    with torch.no_grad():
        # Global normalize over the whole replay (approximation; could be per-batch)
        norm_ret = (ret_tensor - ret_tensor.mean()) / (ret_tensor.std() + 1e-8)
        v_pred = v_baseline(ctx_tensor)
        a_targets = (norm_ret - v_pred).cpu().numpy()

    rng = np.random.default_rng(seed)
    alpha = (1.0 - ci) / 2.0
    out: dict[str, dict[str, float]] = {}

    for oi in range(n_opponents):
        name = opp_names.get(oi, str(oi))
        cell_adv: list[np.ndarray] = []
        cell_means: list[float] = []
        for zi in range(latent_k):
            mask = (opp_all == oi) & (z_all == zi)
            vals = a_targets[mask]
            cell_adv.append(vals)
            cell_means.append(float(np.mean(vals)) if vals.size else -np.inf)
        order = np.argsort(cell_means)[::-1]
        best_z, second_z = int(order[0]), int(order[1])
        best_v, second_v = cell_adv[best_z], cell_adv[second_z]
        if best_v.size < 2 or second_v.size < 2:
            out[name] = {
                "best_z": best_z, "second_z": second_z,
                "gap": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "ci_excludes_zero": False,
                "insufficient": True,
            }
            continue
        boot = np.empty(n_boot)
        for b in range(n_boot):
            bs = best_v[rng.integers(0, best_v.size, best_v.size)]
            ss = second_v[rng.integers(0, second_v.size, second_v.size)]
            boot[b] = bs.mean() - ss.mean()
        lo = float(np.quantile(boot, alpha))
        hi = float(np.quantile(boot, 1.0 - alpha))
        out[name] = {
            "best_z": best_z,
            "second_z": second_z,
            "gap": float(np.mean(best_v) - np.mean(second_v)),
            "ci_low": lo,
            "ci_high": hi,
            "ci_excludes_zero": bool(lo > 0.0),
            "insufficient": False,
        }
    return out


def advantage_matrix_from_replay(
    replay: QRouterReplayBuffer,
    v_baseline: ContextualVBaseline,
    *,
    n_opponents: int,
    latent_k: int,
    opponent_id_to_idx: dict[int, int],
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Return empirical mean advantage matrix [N_opp, K] and counts [N_opp, K].

    Advantage = normalized_return - stopgrad(V(context)).
    """
    z_all, ret_all, opp_all = replay.raw_arrays()
    if ret_all.size == 0:
        return np.full((n_opponents, latent_k), float("nan")), np.zeros((n_opponents, latent_k))

    device_t = torch.device(device)
    n = int(ret_all.size)
    ctx_tensor = replay._contexts[:n].to(device_t)
    ret_tensor = torch.tensor(ret_all, dtype=torch.float32, device=device_t)

    with torch.no_grad():
        norm_ret = (ret_tensor - ret_tensor.mean()) / (ret_tensor.std() + 1e-8)
        v_pred = v_baseline(ctx_tensor)
        a_targets = (norm_ret - v_pred).cpu().numpy()

    total = np.zeros((n_opponents, latent_k))
    count = np.zeros((n_opponents, latent_k))
    for i in range(ret_all.size):
        oi = int(opp_all[i])
        zi = int(z_all[i])
        if 0 <= oi < n_opponents and 0 <= zi < latent_k:
            total[oi, zi] += float(a_targets[i])
            count[oi, zi] += 1
    mean = np.full((n_opponents, latent_k), float("nan"))
    valid = count > 0
    mean[valid] = total[valid] / count[valid]
    return mean, count
