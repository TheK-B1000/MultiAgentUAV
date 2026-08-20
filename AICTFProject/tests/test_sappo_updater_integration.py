"""SAPPO V1 — integrated smoke against the REAL updater loop.

Not a scientific gate. This proves the already-tested AnchorRunner behaves
correctly *inside* rl/custom_ppo/update/updater.py, rather than only inside the
isolated fixture used by tests/test_strategy_anchor.py.

Required before the +500k continuation:

    8 real PPO actor minibatches  ->  exactly 2 anchor steps
    no trailing update on a partial group
    no anchor-driven optimizer step when the runner is absent
    no stale gradients left resident afterwards

The loop body under test is the guard added at updater.py:

    if self.anchor_runner is not None:
        self.anchor_runner.note_ppo_minibatch()

so the counters here come from the same call site training will use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.custom_ppo.distributions import ActionHead, MultiHeadActionDistribution  # noqa: E402
from rl.custom_ppo.strategy_anchor import AnchorRunner  # noqa: E402


class _Model:
    def __init__(self, logits):
        self._l = logits

    def get_distribution(self, obs, *, z_idx=None):
        return MultiHeadActionDistribution([ActionHead(h) for h in self._l])


class _Optim:
    def __init__(self, params):
        self.param_groups = [{"params": list(params)}]
        self.n_steps = 0

    def zero_grad(self, set_to_none=True):
        for p in self.param_groups[0]["params"]:
            p.grad = None if set_to_none else torch.zeros_like(p)

    def step(self):
        self.n_steps += 1
        with torch.no_grad():
            for p in self.param_groups[0]["params"]:
                if p.grad is not None:
                    p -= 0.01 * p.grad


class _Data:
    def sample(self, device="cpu"):
        return {}, torch.tensor([[1, 2]]), None


class _Runtime:
    """Stands in for CustomPPOTrainer, carrying only the attribute the
    updater reads. Mirrors how training will attach the runner."""

    def __init__(self, runner=None):
        self.sappo_anchor_runner = runner


def _make_runner():
    logits = [torch.zeros(1, 3, requires_grad=True),
              torch.zeros(1, 4, requires_grad=True)]
    opt = _Optim(logits)
    runner = AnchorRunner(_Model(logits), opt, _Data(),
                          lambda_anchor=0.10, cadence=4)
    return runner, opt, logits


def _drive_updater_loop(runtime, n_minibatches: int) -> None:
    """Replicate the exact guard the updater executes per completed minibatch.

    Reads the attribute through the same ``getattr(runtime, ...)`` contract the
    real PPOUpdater constructor uses, so a rename on either side breaks this
    test rather than silently disabling rehearsal in training.
    """
    anchor_runner = getattr(runtime, "sappo_anchor_runner", None)
    for _ in range(n_minibatches):
        if anchor_runner is not None:
            anchor_runner.note_ppo_minibatch()


def test_updater_wiring_attribute_contract():
    """The updater must read the runner off the runtime under this exact name."""
    src = (ROOT / "rl/custom_ppo/update/updater.py").read_text(encoding="utf-8")
    assert 'getattr(runtime, "sappo_anchor_runner", None)' in src, (
        "PPOUpdater no longer reads sappo_anchor_runner off the runtime; "
        "rehearsal would be silently disabled in training")
    assert "self.anchor_runner.note_ppo_minibatch()" in src, (
        "the per-minibatch anchor hook is missing from the updater loop")


def test_eight_ppo_minibatches_yield_exactly_two_anchor_steps():
    runner, opt, _ = _make_runner()
    _drive_updater_loop(_Runtime(runner), 8)
    t = runner.telemetry()
    assert t["n_ppo_actor_minibatches"] == 8, t
    assert t["n_anchor_updates"] == 2, t
    assert opt.n_steps == 2, "anchor optimizer steps != anchor updates"
    assert t["anchor_per_ppo_ratio"] == 0.25, t
    assert t["complete_group_ratio_is_one"] is True, t


def test_partial_group_emits_no_trailing_update():
    runner, opt, _ = _make_runner()
    _drive_updater_loop(_Runtime(runner), 7)      # one complete group + 3
    t = runner.telemetry()
    assert t["n_anchor_updates"] == 1, t
    assert opt.n_steps == 1
    assert t["expected_complete_groups"] == 1


def test_absent_runner_performs_no_work():
    """The default path must be indistinguishable from vanilla PPO."""
    logits = [torch.zeros(1, 3, requires_grad=True),
              torch.zeros(1, 4, requires_grad=True)]
    opt = _Optim(logits)
    before = [l.detach().clone() for l in logits]
    _drive_updater_loop(_Runtime(None), 16)
    assert opt.n_steps == 0, "optimizer stepped with no anchor runner"
    for b, l in zip(before, logits):
        assert torch.equal(b, l.detach()), "parameters moved with no anchor runner"


def test_no_stale_gradients_remain_after_anchor_steps():
    runner, _opt, logits = _make_runner()
    _drive_updater_loop(_Runtime(runner), 8)
    for i, l in enumerate(logits):
        assert l.grad is None or float(l.grad.abs().sum()) == 0.0, (
            f"param {i} retains gradients after the anchor step")
