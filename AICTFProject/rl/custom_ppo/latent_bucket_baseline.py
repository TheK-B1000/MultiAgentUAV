"""Context-bucketed empirical baseline for the q_phi advantage (v3d).

This is the "smart coach router" baseline: instead of subtracting the
V-marginal ``mean_k V(s, z_k)`` (which depends on a noisily-trained value head
for off-policy z slots), the q_phi advantage subtracts the *empirical* mean
episode return WITHIN the episode's context bucket.

Formally::

    v3c:  adv_i = R_i - mean_k V(s_i, z_k)
    v3d:  adv_i = R_i - mean(R | bucket(s_i))

Where ``bucket(s)`` is a discrete categorical id derived from reward-relevant
context (opponent identity, flag state, score state, etc.). This is variance
reduction by stratification -- standard Monte Carlo / PPO technique. q_phi
learns "is this z better than the average z WITHIN this bucket?" rather than
"better than overall average?", which is the gradient signal needed for
context-conditioned strategy selection.

Plan-faithfulness contract
--------------------------
Bucket ids are gradient-shaping inputs to the *baseline*, never inputs to
the policy. The q_phi network sees only ``s`` and learns ``pi(z|s)``. The
bucket label affects only the variance of the gradient estimator. This is
mathematically identical to using a state-dependent baseline (which PPO/A2C
do trivially) -- we're just discretizing the baseline's input.

In particular, opponent-id bucketing does NOT leak "OP5 -> z2" supervision
to the policy. Two episodes against OP5, one where z=2 won and one where
z=2 lost, contribute oppositely-signed advantages to the gradient; q_phi
still has to *discover from s alone* which z to pick under each context.

Variance / stability
--------------------
Bucket means are noisy with few episodes per bucket. Two guardrails:

1. ``ema``: cross-rollout exponential moving average of bucket means.
   Default 0.9 retains 90% prior + 10% current rollout. Higher = smoother
   but slower to adapt; lower = noisier but more responsive.

2. ``min_count``: when fewer than this many episodes hit a bucket in the
   current rollout, those episodes use the rollout's *global* mean instead.
   Avoids huge advantages from singleton-bucket episodes during early
   "EMA priming" rollouts. Default 8 is conservative; the typical
   3-bucket opponent split with ~3000 episodes/rollout never trips this.

Usage
-----
::

    baseline = BucketBaseline(ema=0.9, min_count=8)
    # ...one call per ``apply_episode_strategy_ppo`` invocation:
    per_episode_baseline = baseline.update_and_compute(
        episode_returns,           # (N_eps,) float32
        bucket_ids,                # (N_eps,) long
    )
    adv = episode_returns - per_episode_baseline
"""
from __future__ import annotations

from typing import Any

import torch


class BucketBaseline:
    """Per-bucket EMA mean of episode returns for q_phi advantage stratification.

    Maintains a running ``dict[int, float]`` of bucket -> EMA mean across
    rollouts. Each ``update_and_compute`` call:

      1. Computes per-bucket means from the current rollout.
      2. EMA-updates each bucket's stored mean.
      3. Updates the global mean (also EMA).
      4. Returns a tensor of per-episode baselines, with low-count buckets
         falling back to the EMA-updated global mean.

    Statistics are accumulated in float64 internally to avoid drift from
    repeated EMA application; the returned tensor matches the input dtype/device.

    Telemetry from the last call is exposed via :attr:`last_stats`. Use those
    for the ``[bucket-baseline]`` print line.
    """

    def __init__(self, *, ema: float, min_count: int) -> None:
        self.ema = float(ema)
        if not 0.0 <= self.ema <= 1.0:
            raise ValueError(f"ema must be in [0, 1], got {ema!r}")
        self.min_count = max(1, int(min_count))
        self._bucket_means: dict[int, float] = {}
        self._global_mean: float = 0.0
        self._global_initialized: bool = False
        # Reset by reset_stats(); see below.
        self.last_stats: dict[str, Any] = self._empty_stats()

    @staticmethod
    def _empty_stats() -> dict[str, Any]:
        return {
            "bucket_count": 0,
            "fallback_fraction": 0.0,
            "global_mean": 0.0,
            "per_bucket_count": {},
            "per_bucket_mean": {},
            "raw_return_std": 0.0,
            "adv_std": 0.0,
            "variance_reduction_ratio": 1.0,
        }

    def reset_state(self) -> None:
        """Clear all EMA state. Use between distinct runs / resumes."""
        self._bucket_means.clear()
        self._global_mean = 0.0
        self._global_initialized = False
        self.last_stats = self._empty_stats()

    def update_and_compute(
        self,
        episode_returns: torch.Tensor,
        bucket_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Update EMA stats from this rollout and return per-episode baselines.

        Parameters
        ----------
        episode_returns : Tensor of shape ``(N,)``, float.
        bucket_ids : Tensor of shape ``(N,)``, integer. ``-1`` is allowed and
            is grouped into a single "unknown" bucket; downstream behavior is
            identical to any other bucket id (subject to the min_count fallback).

        Returns
        -------
        baselines : Tensor of shape ``(N,)``, same dtype/device as
            ``episode_returns``. Per-episode baseline to subtract from the
            episode return when forming the q_phi advantage. Detached (no
            autograd) -- this is a value-target baseline, not a learned
            function.
        """
        if episode_returns.dim() != 1:
            raise ValueError(
                f"episode_returns must be 1-D, got shape {tuple(episode_returns.shape)}"
            )
        if bucket_ids.dim() != 1 or bucket_ids.numel() != episode_returns.numel():
            raise ValueError(
                f"bucket_ids must be 1-D matching episode_returns ({episode_returns.numel()}),"
                f" got shape {tuple(bucket_ids.shape)}"
            )

        device = episode_returns.device
        dtype = episode_returns.dtype
        n = episode_returns.numel()

        # ---- Per-rollout bucket means (this rollout only) --------------------
        returns_np = episode_returns.detach().to(torch.float64).cpu().numpy()
        ids_np = bucket_ids.detach().to(torch.long).cpu().numpy()
        rollout_global = float(returns_np.mean()) if n > 0 else 0.0

        rollout_counts: dict[int, int] = {}
        rollout_sums: dict[int, float] = {}
        for r, b in zip(returns_np.tolist(), ids_np.tolist()):
            b = int(b)
            rollout_counts[b] = rollout_counts.get(b, 0) + 1
            rollout_sums[b] = rollout_sums.get(b, 0.0) + float(r)
        rollout_means: dict[int, float] = {
            b: rollout_sums[b] / max(1, rollout_counts[b]) for b in rollout_counts
        }

        # ---- EMA update of stored bucket means and global mean ---------------
        # First sighting of a bucket primes it with the current rollout mean.
        # Subsequent rollouts apply EMA: new = ema * old + (1 - ema) * rollout.
        ema = self.ema
        for b, rmean in rollout_means.items():
            if b in self._bucket_means:
                self._bucket_means[b] = ema * self._bucket_means[b] + (1.0 - ema) * rmean
            else:
                self._bucket_means[b] = rmean

        if self._global_initialized:
            self._global_mean = ema * self._global_mean + (1.0 - ema) * rollout_global
        else:
            self._global_mean = rollout_global
            self._global_initialized = True

        # ---- Per-episode baseline lookup with min_count fallback -------------
        # Buckets with fewer than ``min_count`` episodes in this rollout get
        # the (EMA-updated) global mean -- their per-bucket EMA may have been
        # primed by an older, very-different-policy rollout and would otherwise
        # produce wildly misleading advantages.
        baselines = [0.0] * n
        fallback_used = 0
        for i, b in enumerate(ids_np.tolist()):
            b = int(b)
            if rollout_counts.get(b, 0) < self.min_count:
                baselines[i] = self._global_mean
                fallback_used += 1
            else:
                baselines[i] = self._bucket_means[b]

        result = torch.as_tensor(baselines, dtype=dtype, device=device)

        # ---- Telemetry -------------------------------------------------------
        with torch.no_grad():
            raw_std = float(episode_returns.detach().std(unbiased=False).cpu().item()) if n > 1 else 0.0
            adv = episode_returns - result
            adv_std = float(adv.detach().std(unbiased=False).cpu().item()) if n > 1 else 0.0
        var_reduction = (adv_std / raw_std) if raw_std > 1e-12 else 1.0

        self.last_stats = {
            "bucket_count": int(len(rollout_counts)),
            "fallback_fraction": float(fallback_used) / max(1, n),
            "global_mean": float(self._global_mean),
            "per_bucket_count": dict(rollout_counts),
            "per_bucket_mean": {int(k): float(v) for k, v in self._bucket_means.items()},
            "raw_return_std": raw_std,
            "adv_std": adv_std,
            "variance_reduction_ratio": var_reduction,
        }
        return result


def resolve_bucket_ids(
    *,
    mode: str,
    opponent_ids: torch.Tensor,
    bucket_ids: torch.Tensor,
) -> torch.Tensor:
    """Map a bucket-baseline mode string to a per-episode bucket-key tensor.

    ``mode`` selects which captured per-episode key (or composite) to use:

      - ``"opponent"``          -- opponent id only (3 buckets for OP3/5/6)
      - ``"bucket_id"``         -- 216-bucket flag/score/spread composite captured at z-commit
      - ``"opponent_x_bucket"`` -- cross product (opponent_id * 256 + bucket_id); ~648 buckets

    The ``"opponent_x_bucket"`` shift of 256 (not 216) leaves a safety margin
    so we never accidentally collide ``opponent_x_bucket`` ids with the raw
    bucket id space, in case the composite definition expands later.

    Returns a 1-D long tensor matching the length of the inputs.
    """
    mode = str(mode).strip().lower()
    if opponent_ids.shape != bucket_ids.shape:
        raise ValueError(
            f"opponent_ids and bucket_ids must match shape; got {tuple(opponent_ids.shape)} vs "
            f"{tuple(bucket_ids.shape)}"
        )
    if mode == "opponent":
        return opponent_ids.long()
    if mode == "bucket_id":
        return bucket_ids.long()
    if mode == "opponent_x_bucket":
        return (opponent_ids.long() * 256 + bucket_ids.long()).long()
    raise ValueError(
        f"Unknown latent_q_phi_bucket_baseline mode {mode!r}; expected one of "
        "'opponent' | 'bucket_id' | 'opponent_x_bucket'"
    )
