"""V6I11 contextual Q-value router.

Replaces BPTT PPO policy-gradient routing with direct return regression.

Design
------
* Router is a 2-hidden-layer MLP whose outputs are expected returns Q(z|context).
* Context = [global_state_geometry (35d)] + [opponent_onehot (N_opp)]
  (map identity can be added later; opponent alone breaks 3 distinct cells).
* Training: Huber loss on Q_pred[selected_z] vs normalised arc_return,
  sampled from a ring replay buffer.
* Exploration: epsilon-greedy — epsilon fraction of episodes get uniform z,
  the rest get argmax(Q).  The epsilon is set via the existing PPO config
  field ``router_uniform_exploration_prob`` (0.5 for the initial diagnostic).
* Actor remains fully frozen; Q-router is trained as an independent bandit.

Usage in experiments/run_v6i11_q_router.py
-------------------------------------------
    q_router = ContextualQRouter(n_opponents=3)
    replay   = QRouterReplayBuffer(capacity=10_000)
    opt      = torch.optim.Adam(q_router.parameters(), lr=3e-4)

    # After each rollout:
    for rec in trainer.latent_state.rollout_strategy_arc_records:
        ctx = q_router.build_context_from_record(rec)
        replay.push(ctx, rec["z"], rec["arc_return"], rec["opponent_id"])

    loss = train_q_router(q_router, replay, opt)
    print(q_router.q_matrix_summary(replay))  # (N_opp x K) return estimates
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.global_state import GLOBAL_STATE_DIM

# Map from integer opponent ID stored in arc records to a stable row index.
# The arc credit manager stamps the canonical ``_opponent_id_int_from_info``
# value (see ``rl/custom_ppo/csv_writers.py::_OPPONENT_TAG_TO_ID``). That table
# follows the OP_N -> N-1 scheme, so OP8->7, OP9->8, OP10->9 (NOT 8/9/10).
_DEFAULT_OPPONENT_ID_TO_IDX: dict[int, int] = {7: 0, 8: 1, 9: 2}


class ContextualQRouter(nn.Module):
    """MLP regressor: context -> Q-values per latent z.

    Parameters
    ----------
    state_dim:
        Number of geometry features taken from global_state[:, :state_dim].
    n_opponents:
        Number of opponent cells; controls one-hot encoding width.
    opponent_id_to_idx:
        Maps raw opponent integer IDs (canonical scheme: OP8->7, OP9->8, OP10->9)
        to row indices [0..n_opponents). Defaults to {7:0, 8:1, 9:2}.
    latent_k:
        Number of latent strategies; output dimension.
    hidden:
        Width of the two hidden layers.
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
        # Initialise output layer near zero so early Q-values don't bias exploration.
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    @property
    def context_dim(self) -> int:
        return self.state_dim + self.n_opponents

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Return Q-values for shape [B, K]."""
        return self.net(context)

    def build_context(
        self,
        global_state: torch.Tensor,
        opponent_ids_raw: torch.Tensor | list[int],
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Build [B, context_dim] tensor from geometry + opponent one-hot.

        Parameters
        ----------
        global_state:
            Shape [B, >=state_dim]; only the first ``state_dim`` columns are used.
        opponent_ids_raw:
            Raw opponent integer IDs (e.g. 8, 9, 10) per row.
        """
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
        """Build a single-row [1, context_dim] context from an arc-credit record dict."""
        gs = rec["global_state_0"].unsqueeze(0).float()  # [1, D]
        opp_raw = int(rec.get("opponent_id", -1))
        return self.build_context(gs, [opp_raw]).squeeze(0)  # [context_dim]

    @torch.no_grad()
    def q_matrix(
        self,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        """Return Q-value matrix [N_opp, K] using the canonical context per opponent.

        Uses a zero geometry vector so the matrix reflects only the opponent
        one-hot component, isolating the opponent-conditional signal.
        """
        device = torch.device(device)
        self.eval()
        rows = []
        for opp_raw in sorted(self.opponent_id_to_idx):
            gs_zero = torch.zeros(1, self.state_dim, device=device)
            ctx = self.build_context(gs_zero, [opp_raw], device=device)
            rows.append(self(ctx))
        return torch.cat(rows, dim=0)  # [N_opp, K]

    def q_matrix_summary(self, replay: "QRouterReplayBuffer") -> str:
        """Human-readable Q-value matrix summary for logging."""
        mat, counts = replay.mean_return_matrix(
            n_opponents=self.n_opponents,
            latent_k=self.latent_k,
            opponent_id_to_idx=self.opponent_id_to_idx,
        )
        from rl.custom_ppo.csv_writers import _opponent_tag_from_id
        opp_raw_sorted = sorted(self.opponent_id_to_idx)
        opp_tags = [_opponent_tag_from_id(int(r)) for r in opp_raw_sorted]
        lines = ["Q-value matrix (empirical mean arc return per opp × z):"]
        header = "       " + "  ".join(f"  z{k}" for k in range(self.latent_k))
        lines.append(header)
        for i, tag in enumerate(opp_tags):
            cells = []
            for k in range(self.latent_k):
                v = mat[i, k]
                n = int(counts[i, k])
                cells.append(f"{v:+.3f}({n:3d})" if not math.isnan(v) else "   nan(  0)")
            lines.append(f"  {tag}: " + "  ".join(cells))
        # Separation: max minus min Q within each opponent row.
        lines.append("  row-spread (max-min):")
        for i, tag in enumerate(opp_tags):
            row = mat[i]
            if np.all(np.isnan(row)):
                spread = float("nan")
            else:
                spread = float(np.nanmax(row) - np.nanmin(row))
            lines.append(f"    {tag}: {spread:+.4f}")
        return "\n".join(lines)


class QRouterReplayBuffer:
    """Fixed-capacity circular replay buffer for (context, z, arc_return, opponent_idx)."""

    def __init__(
        self,
        capacity: int = 10_000,
        context_dim: int = GLOBAL_STATE_DIM + 3,
        latent_k: int = 4,
    ) -> None:
        self.capacity = int(capacity)
        self.context_dim = int(context_dim)
        self.latent_k = int(latent_k)
        self._contexts = torch.zeros(self.capacity, self.context_dim)
        self._z = torch.zeros(self.capacity, dtype=torch.long)
        self._returns = torch.zeros(self.capacity)
        self._opponent_idx = torch.full((self.capacity,), -1, dtype=torch.long)
        self._arc_length = torch.zeros(self.capacity, dtype=torch.long)
        self._is_terminal = torch.zeros(self.capacity, dtype=torch.bool)
        self._pos = 0
        self._size = 0
        # Duplicate-insertion guard.  We reject by *identity* (stable record_id)
        # rather than content hash: two legitimate episodes can occasionally
        # produce identical (context, z, return) triples.  A record_id is
        # ``(rollout_index, env_index, arc_uid)`` where ``arc_uid`` is monotonic
        # for the trainer's lifetime, so identity collisions ⇒ a real re-insertion
        # regression (e.g. reading the arc buffer twice around the update drain).
        self.total_offered = 0
        self.inserted_total = 0
        self.duplicates_rejected = 0
        self._seen_ids: set[tuple] = set()

    def push(
        self,
        context: torch.Tensor,
        z: int,
        arc_return: float,
        opponent_idx: int,
        *,
        record_id: tuple | None = None,
        arc_length: int = 0,
        is_terminal: bool = True,
    ) -> bool:
        """Push a single transition; returns True if inserted, False if rejected.

        A record whose ``record_id`` was already seen is REJECTED (not inserted)
        so replay contamination cannot silently inflate sample counts.  When
        ``record_id`` is None a content signature is used as a weaker fallback.
        """
        self.total_offered += 1
        if record_id is None:
            c = context.detach().cpu().float()
            record_id = ("sig", int(z), round(float(arc_return), 6),
                         round(float(c.sum().item()), 4))
        if record_id in self._seen_ids:
            self.duplicates_rejected += 1
            return False
        self._seen_ids.add(record_id)
        self._contexts[self._pos] = context.detach().cpu().float()
        self._z[self._pos] = int(z)
        self._returns[self._pos] = float(arc_return)
        self._opponent_idx[self._pos] = int(opponent_idx)
        self._arc_length[self._pos] = int(arc_length)
        self._is_terminal[self._pos] = bool(is_terminal)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self.inserted_total += 1
        return True

    def push_many(
        self,
        arc_records: list[dict],
        *,
        rollout_index: int,
        opponent_id_to_idx: dict[int, int],
        build_context,
    ) -> dict[str, int]:
        """Push a batch of arc-record dicts, deduplicating by stable record_id.

        ``build_context(rec) -> 1-D tensor`` maps an arc record to a context
        vector (typically ``q_router.build_context_from_record``).  Returns
        {inserted, duplicates_rejected, size_before, size_after}.
        """
        size_before = self._size
        inserted = 0
        rejected = 0
        for rec in arc_records:
            ctx = build_context(rec)
            opp_raw = int(rec.get("opponent_id", -1))
            opp_idx = opponent_id_to_idx.get(opp_raw, -1)
            arc_uid = rec.get("arc_uid")
            env_index = rec.get("env_index", -1)
            record_id = (
                None if arc_uid is None
                else (int(rollout_index), int(env_index), int(arc_uid))
            )
            ok = self.push(
                ctx,
                int(rec["z"]),
                float(rec["arc_return"]),
                opp_idx,
                record_id=record_id,
                arc_length=int(rec.get("arc_length", 0)),
                is_terminal=(str(rec.get("reason", "")) == "episode_end"),
            )
            inserted += int(ok)
            rejected += int(not ok)
        return {
            "inserted": inserted,
            "duplicates_rejected": rejected,
            "size_before": size_before,
            "size_after": self._size,
        }

    def __len__(self) -> int:
        return self._size

    def sample(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (contexts, z, returns, opponent_idx) for a random minibatch."""
        device = torch.device(device)
        n = self._size
        batch_size = min(batch_size, n)
        idx = torch.randint(0, n, (batch_size,))
        return (
            self._contexts[idx].to(device),
            self._z[idx].to(device),
            self._returns[idx].to(device),
            self._opponent_idx[idx].to(device),
        )

    def mean_return_matrix(
        self,
        *,
        n_opponents: int,
        latent_k: int,
        opponent_id_to_idx: dict[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean_return [N_opp, K], count [N_opp, K]) from all stored transitions."""
        n = self._size
        total = np.zeros((n_opponents, latent_k))
        count = np.zeros((n_opponents, latent_k))
        opp_idx_all = self._opponent_idx[:n].numpy()
        z_all = self._z[:n].numpy()
        ret_all = self._returns[:n].numpy()
        for i in range(n):
            oi = int(opp_idx_all[i])
            zi = int(z_all[i])
            ri = float(ret_all[i])
            if 0 <= oi < n_opponents and 0 <= zi < latent_k:
                total[oi, zi] += ri
                count[oi, zi] += 1
        mean = np.full((n_opponents, latent_k), float("nan"))
        valid = count > 0
        mean[valid] = total[valid] / count[valid]
        return mean, count

    def raw_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (z [N], returns [N], opponent_idx [N]) numpy views over stored rows."""
        n = self._size
        return (
            self._z[:n].numpy().copy(),
            self._returns[:n].numpy().copy(),
            self._opponent_idx[:n].numpy().copy(),
        )

    def count_by_z(self) -> dict[int, int]:
        """Marginal z occupancy across all stored transitions."""
        z_all = self._z[: self._size].numpy()
        return {int(zi): int((z_all == zi).sum()) for zi in range(self.latent_k)}

    def validity_report(
        self,
        *,
        n_opponents: int,
        latent_k: int,
        opponent_id_to_idx: dict[int, int],
    ) -> dict[str, Any]:
        """Replay-buffer validity checks required before trusting Q separations.

        Returns count-by-z, per-(opponent, z) count/mean/std/sem, overall return
        variance, and the duplicate-insertion guard.  Map-conditioned coverage
        (``count by map x z``) is intentionally reported as NOT_INSTRUMENTED:
        the arc-credit record does not carry a map id, so map balance cannot be
        verified from the replay buffer alone.  Adding a ``map_id`` field to the
        arc record is the prerequisite follow-up for a map-aware held-out grid.
        """
        z_all, ret_all, opp_all = self.raw_arrays()
        from rl.custom_ppo.csv_writers import _opponent_tag_from_id
        opp_names = {idx: _opponent_tag_from_id(int(raw)) for raw, idx in opponent_id_to_idx.items()}
        per_cell: dict[str, dict[str, float]] = {}
        for oi in range(n_opponents):
            for zi in range(latent_k):
                mask = (opp_all == oi) & (z_all == zi)
                vals = ret_all[mask]
                n = int(vals.size)
                key = f"{opp_names.get(oi, oi)}_z{zi}"
                if n == 0:
                    per_cell[key] = {"n": 0, "mean": float("nan"),
                                     "std": float("nan"), "sem": float("nan")}
                else:
                    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
                    per_cell[key] = {
                        "n": n,
                        "mean": float(np.mean(vals)),
                        "std": std,
                        "sem": float(std / math.sqrt(n)) if n > 0 else float("nan"),
                    }
        cnt_by_z = self.count_by_z()
        opp_present = sorted({int(o) for o in opp_all.tolist() if o >= 0})
        z_present = sorted({int(z) for z in z_all.tolist()})
        count_by_opponent = {
            opp_names.get(oi, str(oi)): int((opp_all == oi).sum())
            for oi in range(n_opponents)
        }
        n = self._size
        arc_len = self._arc_length[:n].numpy()
        is_term = self._is_terminal[:n].numpy()
        terminal_fraction = float(is_term.mean()) if n else float("nan")
        return {
            "replay_size": int(self._size),
            "total_offered": int(self.total_offered),
            "inserted_total": int(self.inserted_total),
            "duplicates_rejected": int(self.duplicates_rejected),
            "no_duplicate_arcs": bool(self.duplicates_rejected == 0),
            "count_by_z": cnt_by_z,
            "all_z_represented": bool(all(cnt_by_z.get(zi, 0) > 0 for zi in range(latent_k))),
            "count_by_opponent": count_by_opponent,
            "opponents_present": opp_present,
            "all_opponents_represented": bool(len(opp_present) >= n_opponents),
            "z_present": z_present,
            "return_variance": float(np.var(ret_all)) if ret_all.size else float("nan"),
            "return_variance_nonzero": bool(ret_all.size and np.var(ret_all) > 1e-9),
            "mean_arc_length": float(arc_len.mean()) if n else float("nan"),
            "terminal_finalized_fraction": terminal_fraction,
            "per_cell": per_cell,
            "map_coverage": "NOT_INSTRUMENTED (arc record carries no map_id; "
                            "count_by_opponent x z is available instead)",
        }

    def best_second_gap_ci(
        self,
        *,
        n_opponents: int,
        latent_k: int,
        opponent_id_to_idx: dict[int, int],
        n_boot: int = 2000,
        ci: float = 0.95,
        seed: int = 0,
    ) -> dict[str, dict[str, float]]:
        """Bootstrap CI on (best-z mean minus second-best-z mean) per opponent.

        This is the *reliability* half of the separation gate: a Q-spread is only
        credible if the best-vs-second-best mean-return gap has a bootstrap CI
        that excludes zero.  Returns per-opponent dicts with best_z, second_z,
        gap, ci_low, ci_high, ci_excludes_zero.
        """
        z_all, ret_all, opp_all = self.raw_arrays()
        rng = np.random.default_rng(seed)
        from rl.custom_ppo.csv_writers import _opponent_tag_from_id
        opp_names = {idx: _opponent_tag_from_id(int(raw)) for raw, idx in opponent_id_to_idx.items()}
        alpha = (1.0 - ci) / 2.0
        out: dict[str, dict[str, float]] = {}
        for oi in range(n_opponents):
            name = opp_names.get(oi, str(oi))
            cell_vals = []
            cell_means = []
            for zi in range(latent_k):
                v = ret_all[(opp_all == oi) & (z_all == zi)]
                cell_vals.append(v)
                cell_means.append(np.mean(v) if v.size else -np.inf)
            order = np.argsort(cell_means)[::-1]
            best_z, second_z = int(order[0]), int(order[1])
            best_v, second_v = cell_vals[best_z], cell_vals[second_z]
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

    def per_cell_normalize_returns(
        self,
        returns: torch.Tensor,
        z: torch.Tensor,
        opponent_idx: torch.Tensor,
        *,
        n_opponents: int,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """Normalize returns within each (opponent, z) cell to zero-mean unit-std."""
        out = returns.clone()
        for oi in range(n_opponents):
            for zi in range(self.latent_k):
                mask = (opponent_idx == oi) & (z == zi)
                if bool(mask.any()):
                    vals = returns[mask]
                    out[mask] = (vals - vals.mean()) / (vals.std() + eps)
        return out


def train_q_router(
    q_router: ContextualQRouter,
    replay: QRouterReplayBuffer,
    optimizer: torch.optim.Optimizer,
    *,
    batch_size: int = 256,
    n_steps: int = 20,
    device: torch.device | str = "cpu",
    normalize_per_batch: bool = True,
) -> dict[str, float]:
    """Train the Q-router for ``n_steps`` gradient steps from replay.

    Returns telemetry dict with ``q_loss_mean``, ``q_loss_min``, ``q_loss_max``,
    ``q_grad_norm``.
    """
    if len(replay) < max(batch_size // 4, 16):
        return {"q_loss_mean": float("nan"), "q_steps": 0}
    device = torch.device(device)
    q_router.to(device).train()
    losses: list[float] = []
    grad_norms: list[float] = []
    for _ in range(n_steps):
        ctx, z, ret, opp_idx = replay.sample(batch_size, device=device)
        if normalize_per_batch:
            ret = (ret - ret.mean()) / (ret.std() + 1e-8)
        q_vals = q_router(ctx)  # [B, K]
        q_pred = q_vals.gather(1, z.unsqueeze(1)).squeeze(1)  # [B]
        loss = F.huber_loss(q_pred, ret.detach())
        optimizer.zero_grad()
        loss.backward()
        gn = float(
            torch.nn.utils.clip_grad_norm_(q_router.parameters(), max_norm=1.0).item()
        )
        optimizer.step()
        losses.append(float(loss.item()))
        grad_norms.append(gn)
    q_router.eval()
    return {
        "q_loss_mean": float(np.mean(losses)),
        "q_loss_min": float(np.min(losses)),
        "q_loss_max": float(np.max(losses)),
        "q_grad_norm": float(np.mean(grad_norms)),
        "q_steps": n_steps,
    }


class ArcIntegrityError(RuntimeError):
    """Raised when the arc-collection pipeline is broken (zero data / no insert).

    The V6I11 experiment must ABORT on this rather than emit a ``FLAT`` verdict:
    a flat Q-router trained on zero arcs is a tooling failure, not evidence
    about the repertoire.
    """


def copy_arc_record(rec: dict[str, Any]) -> dict[str, Any]:
    """Deep-enough copy of an arc record so a later rollout-state reset cannot
    mutate or clear the copied object (tensors are detached + cloned)."""
    out: dict[str, Any] = {}
    for k, v in rec.items():
        out[k] = v.detach().clone() if torch.is_tensor(v) else v
    return out


def check_arc_guards(
    *,
    records_before_update: int,
    inserted: int,
    size_before: int,
    size_after: int,
) -> None:
    """Hard integrity guards run every update BEFORE Q-router training.

    Raises ArcIntegrityError if any invariant fails:
      * records_before_update > 0   (the rollout actually produced arcs)
      * inserted > 0                (replay retained new unique records)
      * size_after > size_before    (replay grew)
    """
    problems: list[str] = []
    if records_before_update <= 0:
        problems.append(
            f"records_before_update={records_before_update} (rollout produced no "
            "arcs — check latent_arc_credit_enabled and extraction ordering)"
        )
    if inserted <= 0:
        problems.append(
            f"new_unique_arc_count={inserted} (nothing inserted into replay — "
            "possible duplicate contamination or empty extraction)"
        )
    if size_after <= size_before:
        problems.append(
            f"replay did not grow (before={size_before}, after={size_after})"
        )
    if problems:
        raise ArcIntegrityError("; ".join(problems))


def decide_verdict(
    *,
    validity: dict[str, Any],
    gap_ci: dict[str, dict[str, float]],
    spread: dict[str, float],
    spread_threshold: float,
    min_cell_arcs: float,
    n_opponents: int,
    opp_names: dict[int, str],
    min_arcs_per_cell: int = 20,
    min_terminal_fraction: float = 0.5,
) -> tuple[str, int]:
    """Map replay stats + reliability CIs onto the 5-state verdict.

    States (see run_v6i11_q_router docstring):
      INVALID           zero arcs / duplicate contamination / horizon mismatch.
      INSUFFICIENT_DATA coverage or sample count too weak to judge.
      FLAT              coverage OK but no reliable separation learned.
      WEAK_SEPARATION   exactly one opponent reliably separates.
      SEPARATING        >=2 opponents reliably separate.
    Returns (verdict, reliably_separating_count).
    """
    # INVALID: the data itself is untrustworthy.
    if validity.get("replay_size", 0) <= 0:
        return "INVALID", 0
    if not validity.get("no_duplicate_arcs", True):
        return "INVALID", 0
    tf = validity.get("terminal_finalized_fraction", float("nan"))
    if not (isinstance(tf, float) and math.isnan(tf)) and tf < min_terminal_fraction:
        # Arcs are not episode-terminal ⇒ the horizon is not episode-level.
        return "INVALID", 0

    # INSUFFICIENT_DATA: coverage / sample count too weak to judge.
    if (
        not validity.get("all_z_represented", False)
        or not validity.get("all_opponents_represented", False)
        or not validity.get("return_variance_nonzero", False)
        or min_cell_arcs < min_arcs_per_cell
    ):
        return "INSUFFICIENT_DATA", 0

    # Reliable separation requires BOTH magnitude and CI-excludes-zero.
    reliably_separating = 0
    for oi in range(n_opponents):
        name = opp_names.get(oi, str(oi))
        sp = spread.get(f"spread_{name}", float("nan"))
        spread_ok = (not math.isnan(sp)) and sp >= spread_threshold
        ci_ok = bool(gap_ci.get(name, {}).get("ci_excludes_zero", False))
        if spread_ok and ci_ok:
            reliably_separating += 1
    if reliably_separating >= 2:
        return "SEPARATING", reliably_separating
    if reliably_separating == 1:
        return "WEAK_SEPARATION", reliably_separating
    return "FLAT", reliably_separating
