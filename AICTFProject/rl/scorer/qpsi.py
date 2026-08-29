"""Q_psi -- action-conditioned strategic payoff scorer.

Frozen form (PHASE0_ACTION_CONDITIONED_SCORER_PROTOCOL.json ::
AMENDMENT_BEFORE_ANY_DATA_2026_08_24.1_scorer_form):

    Q_psi(o,a1,a2,p) = b(o,p) + u(o,p)^T e(a1) + v(o,p)^T e(a2)
                                + e(a1)^T M(o,p) e(a2)

A purely additive decomposition Q = Q1(o,a1,p) + Q2(o,a2,p) is REJECTED by the
protocol: GUARD/BREACH value may arise from coordination between the two
robots, which an additive critic cannot represent. M is carried in low-rank
factored form M = P Q^T (d x r, r x d), so the interaction term evaluates as
(P^T e1) . (Q^T e2) without ever materialising a d x d matrix.

Per-agent action is a SINGLE categorical over 250 = 5 macros x 50 waypoints,
matching the frozen amendment. The environment's MultiDiscrete nvec is
[5,50,5,50] agent-major, so a_i = macro_i * 50 + waypoint_i.

Analytic expectation (amendment 2, replacing Monte-Carlo estimation):

    mu_i    = sum_{a_i} pi_i(a_i | o, z) e(a_i)
    V_hat   = b + u^T mu_1 + v^T mu_2 + mu_1^T M mu_2

This is EXACT rather than sampled because the bilinear form commutes with the
expectation when the policy factorises across the two agents -- which it does:
the four MultiDiscrete heads are independent categoricals, so
pi(a1,a2) = pi(m1)pi(w1)pi(m2)pi(w2).

Policy identity is NEVER an input. The protocol prohibits it explicitly: a
scorer given policy identity can cheat by learning "pi_A usually wins on A"
instead of which decisions are payoff-relevant. Only (o, a, pole) enter.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import torch
import torch.nn as nn

N_MACRO, N_WAYPOINT = 5, 50
N_ACTIONS = N_MACRO * N_WAYPOINT          # 250 per agent
N_AGENTS = 2
GRID_CHANNELS = 7
GRID_HW = 20
VEC_DIM = 20


def joint_action_index(action: torch.Tensor) -> torch.Tensor:
    """[macro1, wp1, macro2, wp2] -> (a1, a2) each in [0, 250).

    Agent-major MultiDiscrete layout, as produced by the collector.
    """
    if action.shape[-1] != 4:
        raise ValueError(f"expected 4 action components, got {action.shape[-1]}")
    a1 = action[..., 0] * N_WAYPOINT + action[..., 1]
    a2 = action[..., 2] * N_WAYPOINT + action[..., 3]
    return a1.long(), a2.long()


@dataclass(frozen=True)
class QPsiConfig:
    action_dim: int = 32          # d -- action embedding width
    rank: int = 8                 # r -- rank of the interaction term M
    hidden: int = 256
    conv_width: int = 64
    pole_embed: int = 8
    n_poles: int = 2
    n_regimes: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


class QPsi(nn.Module):
    """Low-rank joint critic over (observation, joint action, pole)."""

    def __init__(self, cfg: QPsiConfig | None = None):
        super().__init__()
        self.cfg = cfg or QPsiConfig()
        if self.cfg.n_regimes not in (1, 4):
            raise ValueError(f"n_regimes must be 1 or 4, got {self.cfg.n_regimes}")
        c, d, r = self.cfg, self.cfg.action_dim, self.cfg.rank

        # --- observation trunk -------------------------------------------------
        # both agents' grids are stacked channel-wise; the layout is fixed, so a
        # plain CNN sees the pair jointly rather than pooling them independently.
        self.conv = nn.Sequential(
            nn.Conv2d(N_AGENTS * GRID_CHANNELS, 32, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, c.conv_width, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c.conv_width, c.conv_width, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.pole_emb = nn.Embedding(c.n_poles, c.pole_embed)
        trunk_in = c.conv_width + N_AGENTS * VEC_DIM + N_AGENTS + c.pole_embed
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, c.hidden), nn.ReLU(),
            nn.Linear(c.hidden, c.hidden), nn.ReLU(),
        )

        # --- action-conditioned heads -----------------------------------------
        self.action_emb = nn.Embedding(N_ACTIONS, d)   # shared e(.) for both slots
        if c.n_regimes == 1:
            # Keep these exact attribute names and construction order: Phase-0
            # frozen checkpoints contain head_b/u/v/P/Q keys.
            self.head_b = nn.Linear(c.hidden, 1)
            self.head_u = nn.Linear(c.hidden, d)
            self.head_v = nn.Linear(c.hidden, d)
            self.head_P = nn.Linear(c.hidden, d * r)   # M = P Q^T, never formed
            self.head_Q = nn.Linear(c.hidden, d * r)
            projection_sets = ((self.head_b, self.head_u, self.head_v,
                                self.head_P, self.head_Q),)
        else:
            self.regime_heads = nn.ModuleList([
                nn.ModuleDict({
                    "b": nn.Linear(c.hidden, 1),
                    "u": nn.Linear(c.hidden, d),
                    "v": nn.Linear(c.hidden, d),
                    "P": nn.Linear(c.hidden, d * r),
                    "Q": nn.Linear(c.hidden, d * r),
                })
                for _ in range(c.n_regimes)
            ])
            projection_sets = tuple(
                (heads["b"], heads["u"], heads["v"], heads["P"], heads["Q"])
                for heads in self.regime_heads
            )

        # Start near a pure bias predictor. Apply the unchanged rule to every
        # regime-specific interaction projection.
        for projection_set in projection_sets:
            for h in projection_set[3:]:
                nn.init.zeros_(h.bias)
                nn.init.normal_(h.weight, std=1e-3)

    # ---------------------------------------------------------------- encoding
    def encode(self, grid: torch.Tensor, vec: torch.Tensor,
               agent_mask: torch.Tensor, pole: torch.Tensor) -> torch.Tensor:
        """(o, p) -> trunk features h. grid (N,2,7,20,20), vec (N,2,20)."""
        n = grid.shape[0]
        g = self.conv(grid.reshape(n, N_AGENTS * GRID_CHANNELS, GRID_HW, GRID_HW))
        x = torch.cat([g, vec.reshape(n, -1), agent_mask.reshape(n, -1),
                       self.pole_emb(pole.long())], dim=-1)
        return self.trunk(x)

    def regime_from_vec(self, vec: torch.Tensor) -> torch.Tensor:
        """Return the frozen D1 flag/carrying regime for ``(N,2,20)`` rows."""
        if vec.dim() != 3 or tuple(vec.shape[1:]) != (N_AGENTS, VEC_DIM):
            raise ValueError(
                f"regime reconstruction requires vec shape (N,2,20), got {tuple(vec.shape)}"
            )
        carrying = (vec[..., 10] > 0.5).any(dim=-1)

        x = vec[..., 0] * (GRID_HW - 1)
        y = vec[..., 1] * (GRID_HW - 1)
        flag_x = vec[..., 6] * GRID_HW + x
        flag_y = vec[..., 7] * GRID_HW + y
        if not (
            torch.allclose(flag_x[:, 0], flag_x[:, 1], atol=1e-3, rtol=0.0)
            and torch.allclose(flag_y[:, 0], flag_y[:, 1], atol=1e-3, rtol=0.0)
        ):
            raise ValueError("blue-flag reconstruction disagrees between agents")
        home = (
            torch.isclose(flag_x[:, 0], flag_x.new_tensor(2.0), atol=1e-3, rtol=0.0)
            & torch.isclose(flag_y[:, 0], flag_y.new_tensor(10.0), atol=1e-3, rtol=0.0)
        )
        stolen = ~home
        return 2 * stolen.long() + carrying.long()

    def coefficients(self, h: torch.Tensor, regime: torch.Tensor | None = None):
        """h -> (b, u, v, P, Q) with P,Q reshaped to (N, d, r)."""
        d, r = self.cfg.action_dim, self.cfg.rank
        n = h.shape[0]
        if self.cfg.n_regimes == 1:
            return (self.head_b(h).squeeze(-1),
                    self.head_u(h), self.head_v(h),
                    self.head_P(h).view(n, d, r), self.head_Q(h).view(n, d, r))
        if regime is None or regime.dim() != 1 or int(regime.shape[0]) != n:
            raise ValueError(f"regime must have shape ({n},) for four-head Q_psi")
        regime = regime.long()
        if bool(((regime < 0) | (regime >= self.cfg.n_regimes)).any().item()):
            raise ValueError("regime index is outside [0, n_regimes)")

        projected = [
            torch.stack([heads[name](h) for heads in self.regime_heads], dim=1)
            for name in ("b", "u", "v", "P", "Q")
        ]
        row = torch.arange(n, device=h.device)
        b, u, v, P, Q = (value[row, regime] for value in projected)
        return b.squeeze(-1), u, v, P.view(n, d, r), Q.view(n, d, r)

    # ------------------------------------------------------------------ scoring
    def _combine(self, b, u, v, P, Q, e1: torch.Tensor, e2: torch.Tensor):
        """b + u.e1 + v.e2 + (P^T e1).(Q^T e2).

        e1/e2 are either action embeddings e(a_i) or expected embeddings mu_i;
        the algebra is identical, which is exactly why the analytic expectation
        is available in closed form.
        """
        lin = (u * e1).sum(-1) + (v * e2).sum(-1)
        inter = (torch.einsum("ndr,nd->nr", P, e1) *
                 torch.einsum("ndr,nd->nr", Q, e2)).sum(-1)
        return b + lin + inter

    def forward(self, grid, vec, agent_mask, pole, a1, a2) -> torch.Tensor:
        """Q_psi(o, a1, a2, p) for concrete joint actions."""
        regime = self.regime_from_vec(vec) if self.cfg.n_regimes > 1 else None
        b, u, v, P, Q = self.coefficients(
            self.encode(grid, vec, agent_mask, pole), regime
        )
        return self._combine(b, u, v, P, Q,
                             self.action_emb(a1), self.action_emb(a2))

    def expected_value(self, grid, vec, agent_mask, pole,
                       p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """V_hat(o, pi, p) = E_{a ~ pi} Q_psi(o, a, p), computed ANALYTICALLY.

        p1, p2 are (N, 250) per-agent action distributions. No sampling, no
        enumeration of the 62,500 joint actions.
        """
        regime = self.regime_from_vec(vec) if self.cfg.n_regimes > 1 else None
        b, u, v, P, Q = self.coefficients(
            self.encode(grid, vec, agent_mask, pole), regime
        )
        mu1 = p1 @ self.action_emb.weight        # (N, d)
        mu2 = p2 @ self.action_emb.weight
        return self._combine(b, u, v, P, Q, mu1, mu2)
