"""Reusable PPO/latent configs aligned with the Summer Implementation Plan and paper ablations.

**Main result (clean default):** sample ``z`` once per episode; no event-based or KL extras
(``latent_resample_every_n=0``, ``latent_resample_on_flag=False``, ``latent_kl_consecutive=0``).

**E3 baseline (fair comparison):** :func:`paper_default_no_latent_config` is
``replace(paper_default_latent_config(), use_latent_strategy=False)`` so the
Summer default opponent/training setup is held fixed while the latent path is off.

Professor-requested baselines are exposed below: curriculum and no-latent PPO.
The no-persistence ablation is retained as an optional new-method ablation.

Use a separate run / config when ablating flag-triggered resampling (plan §12).
"""

from __future__ import annotations

from dataclasses import replace

from rl.train_ppo import PPOConfig, TrainMode


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
    """Backward-compatible alias for the fixed-OP3 no-latent baseline."""
    return paper_default_no_latent_config()


def curriculum_baseline_config() -> PPOConfig:
    """Professor-requested baseline: Jacob-style OP1->OP2->OP3 curriculum with latent strategy off."""
    return replace(
        paper_default_latent_config(),
        mode=TrainMode.CURRICULUM.value,
        use_latent_strategy=False,
    )


def no_latent_baseline_config() -> PPOConfig:
    """Professor-requested baseline: no-latent PPO under the Summer default fixed-OP3 setting."""
    return paper_default_no_latent_config()


def fixed_opponent_no_latent_config() -> PPOConfig:
    """Backward-compatible alias for :func:`no_latent_baseline_config`."""
    return no_latent_baseline_config()


def jacob_original_baseline_config() -> PPOConfig:
    """Legacy Jacob-style control: OP1->OP2->OP3 curriculum with latent strategy disabled."""
    return curriculum_baseline_config()


def latent_no_persistence_baseline_config(*, resample_every: int = 20) -> PPOConfig:
    """Optional new-method ablation: sparse strategy refresh without persistence penalty."""
    return replace(
        paper_default_latent_config(),
        latent_resample_every_n=max(2, int(resample_every)),
        latent_lam_p=0.0,
    )


def fixed_latent_baseline_config(*, strategy_id: int = 0) -> PPOConfig:
    """Older optional ablation: latent actor/critic receive one fixed z ID for the whole run."""
    return replace(
        paper_default_latent_config(),
        fixed_latent_strategy=True,
        fixed_latent_strategy_id=max(0, int(strategy_id)),
    )


def ablation_flag_resample_config(*, base: PPOConfig | None = None) -> PPOConfig:
    """Ablate optional plan §12: resample when global flag/territory slice changes (keep all else from ``base``)."""
    c = base if base is not None else PPOConfig()
    return replace(c, latent_resample_on_flag=True)
