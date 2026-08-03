"""PBRS terminal handling: auto-reset state must never enter F(s, s').

Potential-based shaping is only policy-invariant if F(s,s') = gamma*Phi(s') -
Phi(s) uses the TRUE successor state. On a terminal step the vec env auto-resets,
so ``self.blue_x`` and friends already describe the NEXT episode by the time the
shaping term is computed. If that leaks in, F becomes a random jump between two
unrelated states and the invariance guarantee is void.

Terminal/reset boundaries have already produced several measurement bugs in this
project (post-reset scores read as 0-0; step counters zeroed on reset markers),
so this boundary gets an explicit test rather than an assumption.

These tests observe only. If they find a defect, the fix is a separate decision.
"""
from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from gpu_env import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402

V2 = dict(
    taggers_required=1,
    tag_min_interval_seconds=10.0,
    tag_nearest_only=True,
    tag_channel_seconds=0.0,
    suppression_attackers_required=2,
)


def _env(n_envs: int = 4, max_steps: int = 24, seed: int = 31337):
    cfg = GPUFieldConfig(
        n_envs=n_envs, max_blue_agents=2, max_red_agents=2,
        map_set="train", map_layout="map_a", max_decision_steps=max_steps,
        aquaticus_profile=True, rules_profile="OURS", device="cpu",
        seed=seed, obstacle_obs_channel=True, tag_telemetry_enabled=True, **V2,
    )
    return GPUCTFVecEnv(cfg)


def test_pbrs_gamma_matches_ppo_gamma():
    """Shaping discount must equal the discount PPO actually trains with."""
    from rl.config.ppo_config import PPOConfig

    env = _env()
    try:
        assert float(env.core.cfg.pbrs_gamma) == pytest.approx(float(PPOConfig().gamma)), (
            "PBRS gamma differs from PPO gamma; F(s,s') is then not the shaping "
            "term the learner is discounting"
        )
    finally:
        env.close()


def test_pbrs_is_finite_across_a_full_episode_including_terminal():
    env = _env(max_steps=24)
    env.reset()
    core = env.core
    n = core.B * core.Nb * 2
    seen_terminal = False
    try:
        for i in range(30):
            _, _, done, _ = env.step(np.full(n, i % 5, dtype=np.int64))
            pbrs = core._last_dense_progress
            assert torch.isfinite(torch.as_tensor(pbrs)).all(), (
                f"non-finite PBRS at step {i}"
            )
            if np.asarray(done).any():
                seen_terminal = True
        assert seen_terminal, "no terminal step occurred; test proves nothing"
    finally:
        env.close()


def test_pbrs_magnitude_does_not_spike_on_the_terminal_step():
    """A leak shows up as an outsized F on exactly the reset boundary.

    Phi is a bounded closeness in [0, 1], so a legitimate single-step change is
    small. A jump between two unrelated episodes' states would be large.
    """
    env = _env(max_steps=24)
    env.reset()
    core = env.core
    n = core.B * core.Nb * 2
    terminal_mags, normal_mags = [], []
    try:
        for i in range(60):
            _, _, done, _ = env.step(np.full(n, i % 5, dtype=np.int64))
            mag = torch.as_tensor(core._last_dense_progress).abs()
            d = np.asarray(done).reshape(-1)
            for b in range(core.B):
                (terminal_mags if (b < d.size and d[b]) else normal_mags).append(
                    float(mag.reshape(-1)[b])
                )
        assert terminal_mags, "no terminal transitions observed"
        assert normal_mags
        worst_terminal = max(terminal_mags)
        typical = float(np.percentile(normal_mags, 99)) if normal_mags else 0.0
        # Generous bound: this is a leak detector, not a tightness check.
        assert worst_terminal <= max(1.0, 10.0 * typical), (
            f"PBRS on terminal steps ({worst_terminal:.4f}) dwarfs normal steps "
            f"(p99 {typical:.4f}); auto-reset state may be entering F(s,s')"
        )
    finally:
        env.close()


def test_pbrs_uses_previous_state_not_post_reset_state():
    """Directly: after a reset the potential must be recomputed, not carried.

    Drives every env to termination, then verifies the first post-reset step's
    shaping is of ordinary magnitude rather than reflecting a discontinuity
    between the old episode's geometry and the new one's.
    """
    env = _env(n_envs=2, max_steps=12)
    env.reset()
    core = env.core
    n = core.B * core.Nb * 2
    post_reset_mags = []
    try:
        prev_ep = core.episode_id.clone()
        for i in range(40):
            env.step(np.full(n, i % 5, dtype=np.int64))
            advanced = (core.episode_id > prev_ep)
            if advanced.any():
                mag = torch.as_tensor(core._last_dense_progress).abs().reshape(-1)
                for b in advanced.nonzero(as_tuple=False).flatten().tolist():
                    post_reset_mags.append(float(mag[int(b)]))
            prev_ep = core.episode_id.clone()
        assert post_reset_mags, "no episode boundary observed"
        assert max(post_reset_mags) <= 1.0, (
            f"shaping magnitude {max(post_reset_mags):.4f} at an episode boundary "
            "exceeds the bounded-potential range; state is leaking across the reset"
        )
    finally:
        env.close()


def test_potentials_are_bounded_closeness():
    """Phi in [0, 1] is the premise every bound above relies on."""
    env = _env()
    env.reset()
    core = env.core
    try:
        a, r, d = core._compute_potentials(
            core.blue_x, core.blue_y, core.blue_carrying, core.red_carrying
        )
        for name, phi in (("attack", a), ("return", r), ("defend", d)):
            assert torch.isfinite(phi).all(), f"{name} potential non-finite"
            assert float(phi.min()) >= -1e-6, f"{name} potential below 0"
            assert float(phi.max()) <= 1.0 + 1e-6, f"{name} potential above 1"
    finally:
        env.close()
