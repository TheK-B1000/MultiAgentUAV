"""Unit tests for the v3i19 per-arc consequence credit machinery.

The arc-credit channel lives in ``LatentStrategyState`` and is composed of
three lifecycle hooks (``arc_open``, ``arc_accumulate_step``,
``arc_finalize``) plus a PPO update method (``apply_arc_strategy_ppo``).
These tests pin the state-machine semantics that ``rollout_collector`` and
``ppo_updater`` rely on:

* Open / close lifecycle: ``arc_has_open`` only True between open and finalize.
* Accumulation: only open arcs grow ``arc_return_accum`` and ``arc_steps_accum``;
  closed arcs are ignored.
* Finalize: above-``min_len`` arcs push records into the buffer; below-``min_len``
  arcs are dropped but counted in the telemetry counters.
* Buffer drain: ``reset_arc_credit_rollout_state`` clears records + counters
  WITHOUT touching the per-env open-arc state.
* Disabled mode: every hook is a no-op when ``latent_arc_credit_enabled`` is
  False, so legacy presets pay zero overhead.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
import unittest
from types import SimpleNamespace

import torch


def _stub_strategy_experience_bucket_ids(state: torch.Tensor) -> torch.Tensor:
    return torch.zeros(state.shape[0], dtype=torch.long, device=state.device)


def _load_latent_strategy_state():
    import rl.ppo_core
    ppo_core_mod = types.ModuleType("rl.ppo_core")
    ppo_core_mod.TensorDictRolloutBuffer = rl.ppo_core.TensorDictRolloutBuffer
    ppo_core_mod.ppo_policy_loss = rl.ppo_core.ppo_policy_loss
    sys.modules.setdefault("rl.ppo_core", ppo_core_mod)

    target = (
        pathlib.Path(__file__).resolve().parent.parent
        / "rl"
        / "custom_ppo"
        / "latent_strategy_state.py"
    )
    spec = importlib.util.spec_from_file_location(
        "rl.custom_ppo.latent_strategy_state_isolated_arc", str(target)
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module._strategy_experience_bucket_ids = _stub_strategy_experience_bucket_ids
    if "rl.ppo_core" in sys.modules:
        del sys.modules["rl.ppo_core"]
    return module.LatentStrategyState


LatentStrategyState = _load_latent_strategy_state()


class _FakeStrategyValueHead(torch.nn.Module):
    """Tiny linear ``V_phi(s, z)`` head: returns (state[:,0] - z) so different
    z's see different baselines and tests can verify the gradient flows."""

    def __init__(self, latent_k: int, global_state_dim: int) -> None:
        super().__init__()
        self.latent_k = int(latent_k)
        self.global_state_dim = int(global_state_dim)
        # Single learnable weight makes ``apply_arc_strategy_ppo``'s backward
        # call observable in the optimizer.step() pre/post check.
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, state: torch.Tensor, z_onehot: torch.Tensor) -> torch.Tensor:
        return self.scale * (state[:, 0:1] - z_onehot.argmax(dim=-1, keepdim=True).float())


class _FakeStrategyHead(torch.nn.Module):
    """Deterministic strategy_logits head: state column 0 picks the argmax z."""

    def __init__(self, latent_k: int, global_state_dim: int) -> None:
        super().__init__()
        self.latent_k = int(latent_k)
        self.global_state_dim = int(global_state_dim)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        self._sampling_gen_strategy = gen
        # Parameter so the apply method can backward + optimizer.step.
        self.logit_bias = torch.nn.Parameter(torch.zeros(self.latent_k))
        self.episode_strategy_value_head = _FakeStrategyValueHead(
            latent_k=latent_k, global_state_dim=global_state_dim
        )

    def strategy_logits(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.logit_bias.unsqueeze(0).expand(state.shape[0], -1).clone()
        idx = state[:, 0].long().clamp(min=0, max=self.latent_k - 1)
        logits = logits.scatter(1, idx.unsqueeze(-1), logits.gather(1, idx.unsqueeze(-1)) + 5.0)
        return logits

    def episode_strategy_value(self, state: torch.Tensor, z_idx: torch.Tensor) -> torch.Tensor:
        z_onehot = torch.zeros(
            (state.shape[0], self.latent_k), dtype=torch.float32, device=state.device
        )
        z_onehot.scatter_(1, z_idx.unsqueeze(-1), 1.0)
        return self.episode_strategy_value_head(state, z_onehot).squeeze(-1)

    @staticmethod
    def _categorical_argmax_or_sample(dist, *, deterministic: bool, generator):
        return torch.argmax(dist.logits, dim=-1)


def _make_trainer(
    n_envs: int = 2,
    *,
    arc_enabled: bool = True,
    arc_coef: float = 1.0,
    arc_baseline: str = "context_value",
    arc_min_len: int = 32,
    arc_n_epochs: int = 2,
    arc_clip_eps: float = 0.2,
    arc_return_norm: bool = True,
    latent_k: int = 4,
    gs_dim: int = 4,
) -> SimpleNamespace:
    device = torch.device("cpu")
    model = _FakeStrategyHead(latent_k=latent_k, global_state_dim=gs_dim)
    env = SimpleNamespace(num_envs=n_envs)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    trainer = SimpleNamespace(
        env=env,
        device=device,
        model=model,
        optimizer=optimizer,
        latent_router_optimizer=None,
        use_latent_strategy=True,
        fixed_latent_strategy=False,
        fixed_latent_strategy_id=0,
        latent_k=latent_k,
        latent_kl_consecutive=0.0,
        latent_resample_every_n=64,
        latent_episode_strategy_ppo=False,
        latent_episode_strategy_warmup_decision_steps=0,
        latent_episode_strategy_value_coef=0.5,
        latent_arc_credit_enabled=arc_enabled,
        latent_arc_credit_coef=arc_coef,
        latent_arc_credit_baseline=arc_baseline,
        latent_arc_credit_min_len=arc_min_len,
        latent_arc_credit_n_epochs=arc_n_epochs,
        latent_arc_credit_clip_eps=arc_clip_eps,
        latent_arc_credit_return_norm=arc_return_norm,
        temporal_tracker=None,
        _last_context_state=None,
    )
    return trainer


class ArcCreditStateMachineTests(unittest.TestCase):
    def test_arc_open_snapshots_ctx_z_log_prob_and_marks_arc_open(self) -> None:
        trainer = _make_trainer(n_envs=3, gs_dim=4)
        ls = LatentStrategyState(trainer)
        ls.reset()

        global_state = torch.tensor(
            [[2.0, 0.1, 0.2, 0.3], [1.0, 0.4, 0.5, 0.6], [3.0, 0.7, 0.8, 0.9]],
            dtype=torch.float32,
        )
        z_idx = torch.tensor([2, 1, 3], dtype=torch.long)
        z_log_prob = torch.tensor([-0.1, -0.2, -0.3], dtype=torch.float32)
        open_mask = torch.tensor([True, False, True])

        pushed = ls.arc_open(
            open_mask,
            global_state=global_state,
            z_idx=z_idx,
            z_log_prob=z_log_prob,
        )
        self.assertEqual(pushed, 2)
        # Only envs in open_mask should have arc_has_open=True.
        self.assertTrue(bool(ls.arc_has_open[0].item()))
        self.assertFalse(bool(ls.arc_has_open[1].item()))
        self.assertTrue(bool(ls.arc_has_open[2].item()))
        # ctx + z + log_prob exactly mirror the snapshot.
        self.assertTrue(torch.allclose(ls.arc_open_ctx[0, :4], global_state[0]))
        self.assertEqual(int(ls.arc_open_z[0].item()), 2)
        self.assertAlmostEqual(float(ls.arc_open_log_prob[0].item()), -0.1, places=6)
        self.assertEqual(int(ls.arc_open_z[2].item()), 3)
        self.assertAlmostEqual(float(ls.arc_open_log_prob[2].item()), -0.3, places=6)

    def test_arc_accumulate_step_grows_only_open_arcs(self) -> None:
        trainer = _make_trainer(n_envs=3, arc_min_len=1)
        ls = LatentStrategyState(trainer)
        ls.reset()

        global_state = torch.zeros((3, 4), dtype=torch.float32)
        z_idx = torch.zeros((3,), dtype=torch.long)
        log_prob = torch.zeros((3,), dtype=torch.float32)
        open_mask = torch.tensor([True, False, True])
        ls.arc_open(
            open_mask, global_state=global_state, z_idx=z_idx, z_log_prob=log_prob
        )

        rewards = torch.tensor([1.0, 0.5, 2.0], dtype=torch.float32)
        for _ in range(5):
            ls.arc_accumulate_step(rewards)
        # Env 0: 5 * 1.0, Env 2: 5 * 2.0. Env 1 has no open arc.
        self.assertAlmostEqual(float(ls.arc_return_accum[0].item()), 5.0)
        self.assertAlmostEqual(float(ls.arc_return_accum[1].item()), 0.0)
        self.assertAlmostEqual(float(ls.arc_return_accum[2].item()), 10.0)
        self.assertEqual(int(ls.arc_steps_accum[0].item()), 5)
        self.assertEqual(int(ls.arc_steps_accum[1].item()), 0)
        self.assertEqual(int(ls.arc_steps_accum[2].item()), 5)

    def test_arc_finalize_pushes_records_above_min_len_and_drops_short(self) -> None:
        trainer = _make_trainer(n_envs=3, arc_min_len=4)
        ls = LatentStrategyState(trainer)
        ls.reset()

        global_state = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        ls.arc_open(
            torch.tensor([True, True, True]),
            global_state=global_state,
            z_idx=torch.tensor([1, 2, 3], dtype=torch.long),
            z_log_prob=torch.tensor([-0.1, -0.2, -0.3], dtype=torch.float32),
        )
        rewards = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
        # Env 0: 5 steps (above min_len=4). Env 1: 3 steps (below). Env 2: 4
        # steps (exactly min_len -- still kept). We achieve this by finalizing
        # envs at different points.
        for _ in range(3):
            ls.arc_accumulate_step(rewards)
        # Finalize env 1 with only 3 steps -> dropped from records, counted as
        # dropped-short.
        pushed = ls.arc_finalize(torch.tensor([False, True, False]))
        self.assertEqual(pushed, 0)
        self.assertEqual(ls.rollout_arc_finalized_count, 1)
        self.assertEqual(ls.rollout_arc_dropped_short_count, 1)
        self.assertEqual(len(ls.rollout_strategy_arc_records), 0)

        for _ in range(2):
            ls.arc_accumulate_step(rewards)
        # Env 0 has 5 steps, env 2 has 5 steps (env 1 was already closed so the
        # accumulate steps above had no effect on it).
        self.assertEqual(int(ls.arc_steps_accum[0].item()), 5)
        self.assertEqual(int(ls.arc_steps_accum[2].item()), 5)
        pushed = ls.arc_finalize(torch.tensor([True, False, True]))
        self.assertEqual(pushed, 2)
        self.assertEqual(ls.rollout_arc_finalized_count, 3)
        self.assertEqual(ls.rollout_arc_dropped_short_count, 1)
        self.assertEqual(len(ls.rollout_strategy_arc_records), 2)

        rec_by_env_z = {int(r["z"]): r for r in ls.rollout_strategy_arc_records}
        self.assertIn(1, rec_by_env_z)
        self.assertIn(3, rec_by_env_z)
        self.assertAlmostEqual(rec_by_env_z[1]["arc_return"], 5.0)
        self.assertEqual(rec_by_env_z[1]["arc_length"], 5)
        self.assertAlmostEqual(rec_by_env_z[1]["z_logprob_old"], -0.1, places=6)
        self.assertAlmostEqual(rec_by_env_z[3]["arc_return"], 5.0)
        self.assertEqual(rec_by_env_z[3]["arc_length"], 5)

    def test_arc_finalize_clears_open_state_so_subsequent_steps_dont_accumulate(self) -> None:
        trainer = _make_trainer(n_envs=2, arc_min_len=1)
        ls = LatentStrategyState(trainer)
        ls.reset()
        ls.arc_open(
            torch.tensor([True, True]),
            global_state=torch.zeros((2, 4), dtype=torch.float32),
            z_idx=torch.tensor([0, 1], dtype=torch.long),
            z_log_prob=torch.zeros((2,), dtype=torch.float32),
        )
        ls.arc_accumulate_step(torch.tensor([1.0, 1.0]))
        ls.arc_finalize(torch.tensor([True, True]))
        self.assertEqual(int(ls.arc_has_open.sum().item()), 0)
        # Subsequent accumulate steps must be no-ops.
        ls.arc_accumulate_step(torch.tensor([99.0, 99.0]))
        self.assertAlmostEqual(float(ls.arc_return_accum[0].item()), 0.0)
        self.assertAlmostEqual(float(ls.arc_return_accum[1].item()), 0.0)

    def test_reset_arc_credit_rollout_state_drains_buffer_but_keeps_open_arc(self) -> None:
        trainer = _make_trainer(n_envs=2, arc_min_len=1)
        ls = LatentStrategyState(trainer)
        ls.reset()
        # Open an arc on env 0, finalize, and start a fresh one on env 1.
        ls.arc_open(
            torch.tensor([True, False]),
            global_state=torch.zeros((2, 4), dtype=torch.float32),
            z_idx=torch.tensor([0, 0], dtype=torch.long),
            z_log_prob=torch.zeros((2,), dtype=torch.float32),
        )
        ls.arc_accumulate_step(torch.tensor([3.0, 0.0]))
        ls.arc_finalize(torch.tensor([True, False]))
        self.assertEqual(len(ls.rollout_strategy_arc_records), 1)

        ls.arc_open(
            torch.tensor([False, True]),
            global_state=torch.zeros((2, 4), dtype=torch.float32),
            z_idx=torch.tensor([0, 2], dtype=torch.long),
            z_log_prob=torch.zeros((2,), dtype=torch.float32),
        )
        ls.arc_accumulate_step(torch.tensor([0.0, 7.0]))
        # Env 1 has an open arc with 7.0 accumulated; reset should NOT touch it.
        self.assertTrue(bool(ls.arc_has_open[1].item()))
        self.assertAlmostEqual(float(ls.arc_return_accum[1].item()), 7.0)

        ls.reset_arc_credit_rollout_state()
        # Buffer and counters drained.
        self.assertEqual(len(ls.rollout_strategy_arc_records), 0)
        self.assertEqual(ls.rollout_arc_finalized_count, 0)
        self.assertEqual(ls.rollout_arc_dropped_short_count, 0)
        # Open-arc state preserved.
        self.assertTrue(bool(ls.arc_has_open[1].item()))
        self.assertAlmostEqual(float(ls.arc_return_accum[1].item()), 7.0)
        self.assertEqual(int(ls.arc_steps_accum[1].item()), 1)

    def test_disabled_mode_is_a_total_no_op(self) -> None:
        trainer = _make_trainer(n_envs=2, arc_enabled=False)
        ls = LatentStrategyState(trainer)
        ls.reset()
        # All three hooks must be no-ops.
        n = ls.arc_open(
            torch.tensor([True, True]),
            global_state=torch.zeros((2, 4), dtype=torch.float32),
            z_idx=torch.zeros((2,), dtype=torch.long),
            z_log_prob=torch.zeros((2,), dtype=torch.float32),
        )
        self.assertEqual(n, 0)
        ls.arc_accumulate_step(torch.tensor([1.0, 1.0]))
        self.assertAlmostEqual(float(ls.arc_return_accum[0].item()), 0.0)
        n = ls.arc_finalize(torch.tensor([True, True]))
        self.assertEqual(n, 0)
        self.assertEqual(len(ls.rollout_strategy_arc_records), 0)


class ArcCreditPPOUpdateTests(unittest.TestCase):
    def test_apply_arc_strategy_ppo_emits_zeroed_stats_when_disabled(self) -> None:
        trainer = _make_trainer(arc_enabled=False)
        ls = LatentStrategyState(trainer)
        ls.reset()
        stats = ls.apply_arc_strategy_ppo()
        self.assertEqual(stats["latent_arc_count"], 0.0)
        self.assertEqual(stats["latent_arc_finalized_count"], 0.0)
        self.assertEqual(stats["latent_arc_policy_loss"], 0.0)

    def test_apply_arc_strategy_ppo_with_no_records_is_safe_no_op(self) -> None:
        trainer = _make_trainer(arc_enabled=True)
        ls = LatentStrategyState(trainer)
        ls.reset()
        stats = ls.apply_arc_strategy_ppo()
        self.assertEqual(stats["latent_arc_count"], 0.0)
        self.assertEqual(stats["latent_arc_policy_loss"], 0.0)

    def test_apply_arc_strategy_ppo_steps_optimizer_when_records_exist(self) -> None:
        trainer = _make_trainer(arc_enabled=True, arc_min_len=1, arc_n_epochs=2)
        ls = LatentStrategyState(trainer)
        ls.reset()
        # Two complete arcs with different returns -> nonzero advantage.
        ls.arc_open(
            torch.tensor([True, True]),
            global_state=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]], dtype=torch.float32
            ),
            z_idx=torch.tensor([0, 1], dtype=torch.long),
            z_log_prob=torch.tensor([-0.5, -0.7], dtype=torch.float32),
        )
        for _ in range(3):
            ls.arc_accumulate_step(torch.tensor([1.0, 5.0]))
        ls.arc_finalize(torch.tensor([True, True]))
        self.assertEqual(len(ls.rollout_strategy_arc_records), 2)

        # Snapshot params; expect at least one parameter to change after PPO.
        before = {
            name: p.detach().clone()
            for name, p in trainer.model.named_parameters()
            if p.requires_grad
        }
        stats = ls.apply_arc_strategy_ppo()
        any_changed = any(
            not torch.equal(before[name], p.detach())
            for name, p in trainer.model.named_parameters()
            if p.requires_grad
        )
        self.assertTrue(
            any_changed,
            "apply_arc_strategy_ppo must step the optimizer when records exist",
        )
        self.assertEqual(stats["latent_arc_count"], 2.0)
        self.assertEqual(stats["latent_arc_finalized_count"], 2.0)
        self.assertEqual(stats["latent_arc_mean_length"], 3.0)
        self.assertAlmostEqual(stats["latent_arc_mean_return"], 9.0, places=5)

    def test_apply_arc_strategy_ppo_emits_smoke_alarm_telemetry(self) -> None:
        """The v3i19 smoke alarm fields must populate when arcs exist.

        Confirms ``q_phi_grad_norm`` is nonzero (proving the arc-credit loss
        actually flows gradient onto q_phi parameters), and that the q_phi
        posterior diagnostics (``q_phi_entropy``, ``q_phi_mean_max_prob``)
        reflect the strategy_logits batch.
        """
        trainer = _make_trainer(arc_enabled=True, arc_min_len=1, arc_n_epochs=2)
        ls = LatentStrategyState(trainer)
        ls.reset()
        ls.arc_open(
            torch.tensor([True, True]),
            global_state=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]], dtype=torch.float32
            ),
            z_idx=torch.tensor([0, 1], dtype=torch.long),
            z_log_prob=torch.tensor([-0.5, -0.7], dtype=torch.float32),
        )
        for _ in range(3):
            ls.arc_accumulate_step(torch.tensor([1.0, 5.0]))
        ls.arc_finalize(torch.tensor([True, True]))
        stats = ls.apply_arc_strategy_ppo()

        # Coef is always emitted (smoke alarm includes config audit).
        self.assertAlmostEqual(stats["latent_arc_credit_coef"], 1.0)
        # Grad norm must be positive: differing arc returns -> nonzero
        # advantage -> nonzero gradient onto strategy_encoder logit_bias.
        self.assertGreater(stats["q_phi_grad_norm"], 0.0)
        self.assertGreater(stats["latent_arc_grad_norm"], 0.0)
        self.assertAlmostEqual(
            stats["q_phi_grad_norm"], stats["latent_arc_grad_norm"], places=5
        )
        # q_phi posterior shape is well-defined: entropy in [0, ln K], max_prob in (1/K, 1].
        import math
        self.assertGreaterEqual(stats["q_phi_entropy"], 0.0)
        self.assertLessEqual(stats["q_phi_entropy"], math.log(trainer.latent_k) + 1e-5)
        self.assertGreater(stats["q_phi_mean_max_prob"], 1.0 / trainer.latent_k - 1e-5)
        self.assertLessEqual(stats["q_phi_mean_max_prob"], 1.0 + 1e-5)

    def test_smoke_alarm_zeroed_when_disabled(self) -> None:
        trainer = _make_trainer(arc_enabled=False)
        ls = LatentStrategyState(trainer)
        ls.reset()
        stats = ls.apply_arc_strategy_ppo()
        # All smoke alarm fields default to 0 when the channel is off.
        for k in (
            "q_phi_grad_norm",
            "latent_arc_grad_norm",
            "q_phi_entropy",
            "q_phi_mean_max_prob",
            "latent_arc_credit_coef",
        ):
            self.assertEqual(stats[k], 0.0)

    def test_apply_arc_strategy_ppo_running_mean_baseline_avoids_value_head(self) -> None:
        """``running_mean`` baseline path bypasses ``episode_strategy_value_head``."""
        trainer = _make_trainer(arc_enabled=True, arc_min_len=1, arc_baseline="running_mean")
        ls = LatentStrategyState(trainer)
        ls.reset()
        ls.arc_open(
            torch.tensor([True, True]),
            global_state=torch.zeros((2, 4), dtype=torch.float32),
            z_idx=torch.tensor([0, 1], dtype=torch.long),
            z_log_prob=torch.zeros((2,), dtype=torch.float32),
        )
        for _ in range(2):
            ls.arc_accumulate_step(torch.tensor([2.0, 4.0]))
        ls.arc_finalize(torch.tensor([True, True]))
        stats = ls.apply_arc_strategy_ppo()
        # Running mean should be (4 + 8)/2 = 6.0 -> EMA after both finalizes.
        # Advantages = arc_returns - running_mean: [4-6, 8-6] = [-2, 2].
        # After normalization (mean 0, std 1) the magnitudes are nonzero so a
        # PPO update step runs.
        self.assertEqual(stats["latent_arc_count"], 2.0)
        self.assertNotEqual(stats["latent_arc_policy_loss"], 0.0)
        self.assertAlmostEqual(ls.arc_return_running_mean, 6.0, places=5)


if __name__ == "__main__":
    unittest.main()
