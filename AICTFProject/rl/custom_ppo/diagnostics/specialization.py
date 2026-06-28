"""Per-phase and behavior specialization diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from rl.behavior_telemetry import BEHAVIOR_TELEMETRY_NAMES, N_TELEMETRY
from rl.latent_phase_labels import TEAM_PHASES


def _q_phi_probs_and_entropy(
    buffer: Any, length: int, K: int
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Softmax probs and per-step entropy of stored q_phi logits, or (None, None)."""
    if "z_logits" not in buffer.fields:
        return None, None
    logits = buffer.fields["z_logits"][:length].reshape(-1, K).float()
    probs = torch.softmax(logits, dim=-1)
    entropy = -(probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()).sum(dim=-1)
    return probs.detach().cpu().numpy(), entropy.detach().cpu().numpy()


def _flag_state_per_step(buffer: Any, length: int) -> np.ndarray:
    """Encode (blue_has_red_flag, red_has_blue_flag) as a 4-valued bucket per step."""
    gs_all = buffer.fields["global_state"][:length].cpu().numpy()
    gs_flat = gs_all.reshape(-1, gs_all.shape[-1])
    blue_cap = (gs_flat[:, 10] > 0.5).astype(np.int64)
    red_cap = (gs_flat[:, 11] > 0.5).astype(np.int64)
    return blue_cap + 2 * red_cap


def _phase_block(
    out: dict[str, float],
    z: np.ndarray,
    K: int,
    sw: np.ndarray,
    ba: np.ndarray | None,
    rsp_bin: np.ndarray | None,
    pid: np.ndarray,
    q_probs: np.ndarray | None,
    q_entropy: np.ndarray | None,
) -> None:
    """Per-phase z fractions, switch / blue-ahead / capture means, and q_phi means."""
    for p in range(len(TEAM_PHASES)):
        mask = pid == p
        if not bool(mask.any()):
            for k in range(K):
                out[f"latent_phase{p}_z{k}_frac"] = 0.0
            out[f"latent_phase{p}_switch_mean"] = 0.0
            out[f"latent_phase{p}_blue_ahead_mean"] = 0.0
            out[f"latent_phase{p}_capture_step_mean"] = 0.0
            out[f"q_phi_phase{p}_entropy_mean"] = 0.0
            for k in range(K):
                out[f"q_phi_phase{p}_z{k}_prob_mean"] = 0.0
            continue

        z_sub = np.clip(z[mask], 0, K - 1)
        for k in range(K):
            out[f"latent_phase{p}_z{k}_frac"] = float((z_sub == k).mean())
        out[f"latent_phase{p}_switch_mean"] = float(sw[mask].mean())
        out[f"latent_phase{p}_blue_ahead_mean"] = float(ba[mask].mean()) if ba is not None else 0.0
        out[f"latent_phase{p}_capture_step_mean"] = (
            float(rsp_bin[mask].mean()) if rsp_bin is not None else 0.0
        )

        if q_probs is None or q_entropy is None:
            out[f"q_phi_phase{p}_entropy_mean"] = 0.0
            for k in range(K):
                out[f"q_phi_phase{p}_z{k}_prob_mean"] = 0.0
        else:
            out[f"q_phi_phase{p}_entropy_mean"] = float(q_entropy[mask].mean())
            q_phase = q_probs[mask]
            for k in range(K):
                out[f"q_phi_phase{p}_z{k}_prob_mean"] = float(q_phase[:, k].mean())


def _behavior_diversity_stats(trainer: Any, buffer: Any) -> dict[str, float]:
    """Post-hoc behavior spread by sampled z; diagnostics only, no labels or losses."""
    if not trainer.use_latent_strategy or "z" not in buffer.fields or "behavior_telemetry" not in buffer.fields:
        return {}
    length = int(buffer.pos)
    if length <= 0:
        return {}
    z = buffer.fields["z"][:length].reshape(-1).long()
    beh = buffer.fields["behavior_telemetry"][:length].reshape(-1, N_TELEMETRY).float()
    out: dict[str, float] = {}
    means: list[torch.Tensor] = []
    for k in range(int(trainer.latent_k)):
        mask = z == k
        if bool(mask.any().item()):
            mean_k = beh[mask].mean(dim=0)
            means.append(mean_k)
        else:
            mean_k = torch.zeros((N_TELEMETRY,), dtype=torch.float32, device=beh.device)
        for j, name in enumerate(BEHAVIOR_TELEMETRY_NAMES):
            out[f"latent_z{k}_behavior_{name}_mean"] = float(mean_k[j].detach().cpu().item())

    if len(means) >= 2:
        pairwise: list[float] = []
        for i in range(len(means)):
            for j in range(i + 1, len(means)):
                pairwise.append(float(torch.linalg.vector_norm(means[i] - means[j]).detach().cpu().item()))
        out["latent_behavior_diversity_l2_mean"] = float(np.mean(pairwise)) if pairwise else 0.0
    else:
        out["latent_behavior_diversity_l2_mean"] = 0.0
    return out


q_phi_probs_and_entropy = _q_phi_probs_and_entropy
flag_state_per_step = _flag_state_per_step
phase_block = _phase_block
behavior_diversity_stats = _behavior_diversity_stats

__all__ = [
    "_q_phi_probs_and_entropy",
    "_flag_state_per_step",
    "_phase_block",
    "_behavior_diversity_stats",
    "q_phi_probs_and_entropy",
    "flag_state_per_step",
    "phase_block",
    "behavior_diversity_stats",
]
