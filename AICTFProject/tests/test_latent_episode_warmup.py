"""Regression tests for ``latent_episode_strategy_warmup_decision_steps``.

The motivating bug: under episode-credit mode (``latent_resample_every_n=0``,
``latent_episode_strategy_ppo=True``) q_phi snapshots its training (context, z,
log_prob) pair on the very first decision step of the episode. At that moment
the ctx170 EMAs have just been reset and carry zero opponent-behavior signal,
so q_phi structurally cannot learn ``MI(z; opponent) > 0`` regardless of how
clean the per-episode credit is.

The fix introduces a ``warmup_decision_steps`` guard: a provisional z drives
actions during the warmup window, then a forced resample at step W snapshots
the *committed* (context, z) pair for q_phi credit. These tests pin that
behavior end-to-end through ``LatentStrategyState.strategy_for_step``.
"""

import importlib.util
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np
import torch


def _stub_strategy_experience_bucket_ids(state: torch.Tensor) -> torch.Tensor:
    return torch.zeros(state.shape[0], dtype=torch.long, device=state.device)


# Load ``rl.custom_ppo.latent_strategy_state`` in isolation. Importing it via
# the normal ``rl.custom_ppo`` package path triggers the trainer + diagnostics
# graph; we only need the state-machine class for these unit tests.
def _load_latent_strategy_state():
    # Provide a minimal ppo_core stub so ``from rl.ppo_core import ppo_policy_loss``
    # in latent_strategy_state succeeds without dragging in the full module graph.
    ppo_core_mod = types.ModuleType("rl.ppo_core")

    def _noop_policy_loss(*args, **kwargs):
        raise NotImplementedError("stubbed for unit test; not called by these tests")

    ppo_core_mod.ppo_policy_loss = _noop_policy_loss
    sys.modules.setdefault("rl.ppo_core", ppo_core_mod)

    target = (
        pathlib.Path(__file__).resolve().parent.parent
        / "rl"
        / "custom_ppo"
        / "latent_strategy_state.py"
    )
    spec = importlib.util.spec_from_file_location(
        "rl.custom_ppo.latent_strategy_state_isolated", str(target)
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module._strategy_experience_bucket_ids = _stub_strategy_experience_bucket_ids
    if "rl.ppo_core" in sys.modules:
        del sys.modules["rl.ppo_core"]
    return module.LatentStrategyState


LatentStrategyState = _load_latent_strategy_state()


class _FakeStrategyHead(torch.nn.Module):
    """Deterministic strategy_logits head: state column 0 picks the argmax z."""

    def __init__(self, latent_k: int, global_state_dim: int) -> None:
        super().__init__()
        self.latent_k = int(latent_k)
        self.global_state_dim = int(global_state_dim)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        self._sampling_gen_strategy = gen

    def strategy_logits(self, state: torch.Tensor) -> torch.Tensor:
        # Read the FIRST column of state and route every env's mass onto z_idx=int(col0).
        logits = torch.full(
            (state.shape[0], self.latent_k), -1.0e8, dtype=torch.float32, device=state.device
        )
        idx = state[:, 0].long().clamp(min=0, max=self.latent_k - 1)
        logits.scatter_(1, idx.unsqueeze(-1), 0.0)
        return logits

    @staticmethod
    def _categorical_argmax_or_sample(dist, *, deterministic: bool, generator):
        # Logits are one-hot, so the sample equals the argmax. Honor the API the
        # production model exposes by returning a 1D long tensor of indices.
        return torch.argmax(dist.logits, dim=-1)


def _make_trainer(
    n_envs: int,
    *,
    warmup: int,
    episode_credit: bool = True,
    resample_every_n: int = 0,
    latent_k: int = 4,
    gs_dim: int = 4,
) -> SimpleNamespace:
    device = torch.device("cpu")
    model = _FakeStrategyHead(latent_k=latent_k, global_state_dim=gs_dim)
    env = SimpleNamespace(num_envs=n_envs)
    trainer = SimpleNamespace(
        env=env,
        device=device,
        model=model,
        use_latent_strategy=True,
        fixed_latent_strategy=False,
        fixed_latent_strategy_id=0,
        latent_k=latent_k,
        latent_kl_consecutive=0.0,
        latent_resample_every_n=resample_every_n,
        latent_episode_strategy_ppo=episode_credit,
        latent_episode_strategy_warmup_decision_steps=warmup,
        temporal_tracker=None,
        _last_context_state=None,
    )
    return trainer


class LatentEpisodeWarmupTests(unittest.TestCase):
    def _state_with_z_signal(self, env_z: list[int], gs_dim: int = 4) -> torch.Tensor:
        # Column 0 controls which z the fake head will sample (one-hot logits).
        state = torch.zeros((len(env_z), gs_dim), dtype=torch.float32)
        state[:, 0] = torch.tensor(env_z, dtype=torch.float32)
        return state

    def test_warmup_defers_snapshot_to_commit_step(self):
        """With warmup=5, the (context, z) snapshot fires at decision step 5, not step 0."""
        n_envs = 2
        warmup = 5
        trainer = _make_trainer(n_envs, warmup=warmup)
        ls = LatentStrategyState(trainer)
        ls.reset()

        # Provisional z at step 0 routes to z=1 (state col0=1).
        provisional_state = self._state_with_z_signal([1, 1])
        z0, _, _ = ls.strategy_for_step(provisional_state)
        self.assertTrue(torch.equal(z0, torch.tensor([1, 1])))
        # No snapshot yet: episode_strategy_has_start must stay False.
        self.assertFalse(bool(ls.episode_strategy_has_start.any().item()))
        self.assertFalse(bool(ls.episode_strategy_committed.any().item()))
        self.assertTrue(torch.equal(ls.first_z_sample_step, torch.tensor([-1, -1])))

        # Steps 1..warmup-1: same provisional z, no snapshot.
        for step in range(1, warmup):
            ls.mark_strategy_step_done(np.zeros((n_envs,), dtype=bool))
            _, _, _ = ls.strategy_for_step(provisional_state)
            self.assertFalse(
                bool(ls.episode_strategy_has_start.any().item()),
                msg=f"snapshot fired prematurely at step {step}",
            )

        # Step W: commit step. State now has a richer signal (route to z=3) -- this
        # mimics ctx170 EMAs having absorbed opponent dynamics during the warmup.
        ls.mark_strategy_step_done(np.zeros((n_envs,), dtype=bool))
        committed_state = self._state_with_z_signal([3, 2])
        z_committed, _, _ = ls.strategy_for_step(committed_state)
        self.assertTrue(torch.equal(z_committed, torch.tensor([3, 2])))
        # Snapshot fires now, with the COMMITTED z and the post-warmup context.
        self.assertTrue(bool(ls.episode_strategy_has_start.all().item()))
        self.assertTrue(bool(ls.episode_strategy_committed.all().item()))
        self.assertTrue(torch.equal(ls.episode_strategy_z, torch.tensor([3, 2])))
        # The snapshotted state must be the post-warmup context, not the step-0 one.
        snapshotted_col0 = ls.episode_strategy_state[:, 0].tolist()
        self.assertEqual(snapshotted_col0, [3.0, 2.0])
        # And first_z_sample_step records the actual commit decision step.
        self.assertTrue(torch.equal(ls.first_z_sample_step, torch.tensor([warmup, warmup])))

        # Step W+1 onwards: no further snapshot, z stays locked.
        ls.mark_strategy_step_done(np.zeros((n_envs,), dtype=bool))
        followup_state = self._state_with_z_signal([0, 0])
        z_after, _, _ = ls.strategy_for_step(followup_state)
        self.assertTrue(torch.equal(z_after, torch.tensor([3, 2])))
        # Snapshot must not get overwritten by post-commit calls.
        self.assertEqual(ls.episode_strategy_state[:, 0].tolist(), [3.0, 2.0])

    def test_warmup_zero_preserves_legacy_step_zero_snapshot(self):
        """Back-compat: warmup=0 means snapshot at step 0 (the bug-exhibiting path)."""
        n_envs = 1
        trainer = _make_trainer(n_envs, warmup=0)
        ls = LatentStrategyState(trainer)
        ls.reset()

        state = self._state_with_z_signal([2])
        ls.strategy_for_step(state)
        # Legacy behavior: snapshot at step 0 with the provided context.
        self.assertTrue(bool(ls.episode_strategy_has_start.item()))
        self.assertEqual(int(ls.episode_strategy_z.item()), 2)
        self.assertEqual(int(ls.first_z_sample_step.item()), 0)

    def test_warmup_state_resets_on_done(self):
        """Per-env counters and committed flag clear when an episode ends."""
        n_envs = 2
        warmup = 3
        trainer = _make_trainer(n_envs, warmup=warmup)
        ls = LatentStrategyState(trainer)
        ls.reset()

        state = self._state_with_z_signal([1, 1])
        ls.strategy_for_step(state)
        # Drive 5 steps so env 0 commits.
        for _ in range(warmup + 1):
            ls.mark_strategy_step_done(np.zeros((n_envs,), dtype=bool))
            ls.strategy_for_step(state)
        self.assertTrue(bool(ls.episode_strategy_committed[0].item()))

        # Env 0 dones -> warmup state and committed flag must reset for that env only.
        ls.mark_strategy_step_done(np.array([True, False]))
        self.assertEqual(int(ls.steps_since_ep_start[0].item()), 0)
        self.assertEqual(int(ls.first_z_sample_step[0].item()), -1)
        self.assertFalse(bool(ls.episode_strategy_committed[0].item()))
        # Env 1 untouched: still mid-episode, still committed.
        self.assertTrue(bool(ls.episode_strategy_committed[1].item()))

    def test_warmup_active_when_episode_credit_off_and_z_resampled_masking(self):
        """Warmup is active when episode_credit=False, but z_resampled (eligible for training) is False at step 0 and True at commit step."""
        n_envs = 1
        warmup = 5
        trainer = _make_trainer(n_envs, warmup=warmup, episode_credit=False)
        ls = LatentStrategyState(trainer)
        ls.reset()

        state = self._state_with_z_signal([1])
        # Step 0: provisional sample
        z0, _, aux0 = ls.strategy_for_step(state)
        # Warmup is active, so commit has not happened yet
        self.assertFalse(bool(ls.episode_strategy_committed.any().item()))
        # z_resampled (eligible for PPO training) must be False at step 0
        self.assertFalse(bool(aux0["z_resampled"].any().item()))
        # but actual resample did occur provisionally
        self.assertTrue(bool(aux0["z_resampled_actual"].any().item()))

        # Steps 1..warmup-1
        for step in range(1, warmup):
            ls.mark_strategy_step_done(np.zeros((n_envs,), dtype=bool))
            _, _, aux_step = ls.strategy_for_step(state)
            self.assertFalse(bool(ls.episode_strategy_committed.any().item()))
            self.assertFalse(bool(aux_step["z_resampled"].any().item()))
            self.assertFalse(bool(aux_step["z_resampled_actual"].any().item()))

        # Step W: commit step
        ls.mark_strategy_step_done(np.zeros((n_envs,), dtype=bool))
        _, _, aux_w = ls.strategy_for_step(state)
        # Warmup commit has now occurred
        self.assertTrue(bool(ls.episode_strategy_committed.all().item()))
        # z_resampled (eligible for PPO training) must be True at the commit step
        self.assertTrue(bool(aux_w["z_resampled"].all().item()))
        self.assertTrue(bool(aux_w["z_resampled_actual"].all().item()))


if __name__ == "__main__":
    unittest.main()
