"""Reusable PPO/latent configs aligned with the Summer Implementation Plan and paper ablations.

**Main result (clean default):** sample ``z`` once per episode; no event-based or KL extras
(``latent_resample_every_n=0``, ``latent_resample_on_flag=False``, ``latent_kl_consecutive=0``).

**E3 baseline (fair comparison):** :func:`paper_default_no_latent_config` is
``replace(paper_default_latent_config(), use_latent_strategy=False)`` so every PPO / env
hyperparameter matches; only the latent path is off.

Must-have baselines are also exposed below: flat/no-latent PPO-MARL, latent PPO
without persistence, and fixed-latent PPO.

Use a separate run / config when ablating flag-triggered resampling (plan §12).
"""

from __future__ import annotations

from dataclasses import replace

from rl.train_ppo import PPOConfig


def paper_default_latent_config() -> PPOConfig:
    """Primary paper table: latent on; episode-start ``z`` only; no optional §12 resampling in the run."""
    return PPOConfig(
        use_latent_strategy=True,
        latent_resample_every_n=0,
        latent_resample_on_flag=False,
        latent_kl_consecutive=0.0,
    )


def paper_default_no_latent_config() -> PPOConfig:
    """E3 / baseline: **single** field flipped vs :func:`paper_default_latent_config` — ``use_latent_strategy=False``."""
    return replace(paper_default_latent_config(), use_latent_strategy=False)


def flat_ppo_marl_baseline_config() -> PPOConfig:
    """Must-have baseline: flat/no-latent PPO-MARL; proves whether latent strategy helps."""
    return paper_default_no_latent_config()


def latent_no_persistence_baseline_config(*, resample_every: int = 20) -> PPOConfig:
    """Must-have baseline: latent PPO with refreshes but no persistence penalty."""
    return replace(
        paper_default_latent_config(),
        latent_resample_every_n=max(2, int(resample_every)),
        latent_lam_p=0.0,
    )


def fixed_latent_baseline_config(*, strategy_id: int = 0) -> PPOConfig:
    """Must-have baseline: latent actor/critic receive one fixed z ID for the whole run."""
    return replace(
        paper_default_latent_config(),
        fixed_latent_strategy=True,
        fixed_latent_strategy_id=max(0, int(strategy_id)),
    )


def ablation_flag_resample_config(*, base: PPOConfig | None = None) -> PPOConfig:
    """Ablate optional plan §12: resample when global flag/territory slice changes (keep all else from ``base``)."""
    c = base if base is not None else PPOConfig()
    return replace(c, latent_resample_on_flag=True)
