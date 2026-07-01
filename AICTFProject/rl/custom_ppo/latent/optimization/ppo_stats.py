"""Multi-epoch PPO stats accumulator (shared by router engine paths)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EpisodeStatsAccumulator:
    """Aggregate multi-epoch episode-router metrics with explicit semantics."""

    pg_losses: list[float] = field(default_factory=list)
    value_losses: list[float] = field(default_factory=list)
    approx_kls: list[float] = field(default_factory=list)
    clip_fractions: list[float] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    aux_losses: list[float] = field(default_factory=list)
    ratio_last: float = 1.0
    epochs_completed: int = 0
    optimizer_steps: int = 0
    early_stop: int = 0
    early_stop_kl: float = 0.0
    stop_reason: str = ""

    def record_epoch(
        self,
        *,
        pg_loss: float,
        value_loss: float,
        approx_kl: float,
        clip_fraction: float,
        grad_norm: float,
        aux_loss: float,
        ratio_mean: float,
        stepped: bool,
    ) -> None:
        self.pg_losses.append(float(pg_loss))
        self.value_losses.append(float(value_loss))
        self.approx_kls.append(float(approx_kl))
        self.clip_fractions.append(float(clip_fraction))
        self.grad_norms.append(float(grad_norm))
        self.aux_losses.append(float(aux_loss))
        self.ratio_last = float(ratio_mean)
        self.epochs_completed += 1
        if stepped:
            self.optimizer_steps += 1

    def finalize_base(self) -> dict[str, float]:
        pg_mean = sum(self.pg_losses) / max(1, len(self.pg_losses))
        v_mean = sum(self.value_losses) / max(1, len(self.value_losses))
        clip_mean = sum(self.clip_fractions) / max(1, len(self.clip_fractions))
        aux_mean = sum(self.aux_losses) / max(1, len(self.aux_losses))
        grad_mean = sum(self.grad_norms) / max(1, len(self.grad_norms))
        kl_last = self.approx_kls[-1] if self.approx_kls else 0.0
        kl_max = max(self.approx_kls) if self.approx_kls else 0.0
        grad_max = max(self.grad_norms) if self.grad_norms else 0.0
        return {
            "latent_episode_pg_loss": pg_mean,
            "latent_episode_pg_loss_mean": pg_mean,
            "latent_episode_v_loss": v_mean,
            "latent_episode_v_loss_mean": v_mean,
            "latent_episode_approx_kl": kl_last,
            "latent_episode_approx_kl_last": kl_last,
            "latent_episode_approx_kl_max": kl_max,
            "latent_episode_clip_fraction": clip_mean,
            "latent_episode_clip_fraction_mean": clip_mean,
            "latent_episode_grad_norm_mean": grad_mean,
            "latent_episode_grad_norm_max": grad_max,
            "latent_episode_aux_loss_mean": aux_mean,
            "latent_episode_ratio_mean": self.ratio_last,
            "latent_episode_epochs_completed": float(self.epochs_completed),
            "latent_episode_early_stop": float(self.early_stop),
            "latent_episode_early_stop_kl": float(self.early_stop_kl),
            "latent_episode_optimizer_steps": float(self.optimizer_steps),
        }
