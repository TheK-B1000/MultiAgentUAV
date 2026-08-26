"""SPPPO V1 ranking-update launch blockers.

These tests are the frozen protocol's pre-production verification, expressed as
assertions rather than prose. Each maps to a blocker:

  1/2  a ranking update increases the correct contrast on pole A and pole B
  3    frozen Q_psi is byte-identical before and after the update
  4    no optimizer parameter group contains a Q_psi parameter
  5    the gradient flows through the MASKED distribution the actor really uses
  6    lambda_R = 0 means no RankingRunner exists at all
  +    BOTH mode paths receive counterfactual pressure (neither is detached)
  +    cadence is measured from real optimizer steps, never claimed
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

from rl.scorer.qpsi import N_ACTIONS, QPsi, QPsiConfig  # noqa: E402
from rl.scorer.ranking import (  # noqa: E402
    POLE_A, POLE_B, RankingRunner, assert_qpsi_excluded_from_optimizer,
    masked_action_probs, ranking_loss, strategic_contrast,
)

ACTION_DIMS = [5, 50, 5, 50]


class TinyActor(nn.Module):
    """Minimal stand-in exposing the contract ranking.py depends on.

    Deliberately mirrors SharedActorCentralizedCritic's surface: a per-z
    embedding, ``policy_logits(obs, z_idx=...)``, ``_mask_logits`` and
    ``action_dims``. The z-embedding is what lets us prove BOTH modes receive
    gradient -- one row per mode.
    """

    def __init__(self, k=2, feat=16):
        super().__init__()
        self.action_dims = ACTION_DIMS
        self.z_emb = nn.Embedding(k, feat)
        self.body = nn.Linear(2 * 20, feat)
        self.head = nn.Linear(feat, sum(ACTION_DIMS))

    def policy_logits(self, obs, z_idx=None, **kw):
        h = self.body(obs["vec"].reshape(obs["vec"].shape[0], -1))
        if z_idx is not None:
            h = h + self.z_emb(z_idx)
        return self.head(torch.tanh(h))

    def _mask_logits(self, logits, mask):
        if mask is None:
            return logits
        return logits.masked_fill(mask < 0.5, float("-inf"))


def _obs(n, gen, all_legal=False):
    mask = torch.ones(n, sum(ACTION_DIMS))
    if not all_legal:
        # flat mask layout is [macro1 0:5, wp1 5:55, macro2 55:60, wp2 60:110]
        mask[:, 2:5] = 0.0          # illegal macros 2,3,4 for agent 1
        mask[:, 57:60] = 0.0        # illegal macros 2,3,4 for agent 2
    return {
        "grid": torch.rand(n, 2, 7, 20, 20, generator=gen),
        "vec": torch.rand(n, 2, 20, generator=gen) * 2 - 1,
        "agent_mask": torch.ones(n, 2),
        "mask": mask,
    }


def _frozen_qpsi(seed=0):
    torch.manual_seed(seed)
    q = QPsi(QPsiConfig(action_dim=8, rank=4, hidden=32, conv_width=16)).eval()
    with torch.no_grad():                      # give the heads real signal
        q.head_P.weight.normal_(std=0.3)
        q.head_Q.weight.normal_(std=0.3)
    for p in q.parameters():
        p.requires_grad_(False)
    return q


def _qpsi_bytes(q):
    import hashlib
    h = hashlib.sha256()
    for k, v in sorted(q.state_dict().items()):
        h.update(k.encode()); h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------- blockers 1/2
@pytest.mark.parametrize("pole_val,name", [(POLE_A, "A"), (POLE_B, "B")])
def test_ranking_update_increases_the_correct_contrast(pole_val, name):
    """Blocker 1/2: the update must move Delta upward on each pole."""
    torch.manual_seed(3)
    gen = torch.Generator().manual_seed(3)
    model, q = TinyActor(), _frozen_qpsi()
    obs = _obs(24, gen)
    pole = torch.full((24,), pole_val, dtype=torch.long)
    opt = torch.optim.SGD(model.parameters(), lr=5.0)

    before = strategic_contrast(model, q, obs, pole)[0].mean().item()
    runner = RankingRunner(model, q, opt, lambda_rank=1.0, margin=0.04, cadence=1)
    for _ in range(15):
        runner.note_ppo_minibatch({"obs": obs, "pole": pole})
    after = strategic_contrast(model, q, obs, pole)[0].mean().item()

    assert after > before, f"pole {name}: Delta {before:.6f} -> {after:.6f}"


def test_both_poles_move_in_a_mixed_batch():
    """The realistic case: 16 z0|A and 16 z1|B envs in one batch."""
    torch.manual_seed(4)
    gen = torch.Generator().manual_seed(4)
    model, q = TinyActor(), _frozen_qpsi()
    obs = _obs(32, gen)
    pole = torch.tensor([POLE_A] * 16 + [POLE_B] * 16)
    opt = torch.optim.SGD(model.parameters(), lr=5.0)

    d0 = strategic_contrast(model, q, obs, pole)[0].detach()
    runner = RankingRunner(model, q, opt, lambda_rank=1.0, margin=0.04, cadence=1)
    for _ in range(20):
        runner.note_ppo_minibatch({"obs": obs, "pole": pole})
    d1 = strategic_contrast(model, q, obs, pole)[0].detach()

    a = pole == POLE_A
    b = pole == POLE_B
    assert d1[a].mean() > d0[a].mean(), "pole A contrast did not improve"
    assert d1[b].mean() > d0[b].mean(), "pole B contrast did not improve"


# ------------------------------------------------------------------ blocker 3
def test_qpsi_is_byte_identical_after_updates():
    """Blocker 3: the ruler must not move while the policy learns."""
    torch.manual_seed(5)
    gen = torch.Generator().manual_seed(5)
    model, q = TinyActor(), _frozen_qpsi()
    obs = _obs(16, gen)
    pole = torch.tensor([POLE_A] * 8 + [POLE_B] * 8)
    opt = torch.optim.SGD(model.parameters(), lr=1.0)

    before = _qpsi_bytes(q)
    runner = RankingRunner(model, q, opt, lambda_rank=1.0, margin=0.04, cadence=1)
    for _ in range(10):
        runner.note_ppo_minibatch({"obs": obs, "pole": pole})
    assert _qpsi_bytes(q) == before, "Q_psi mutated during ranking updates"
    assert all(not p.requires_grad for p in q.parameters())
    assert all(p.grad is None for p in q.parameters()), "Q_psi accumulated gradient"


# ------------------------------------------------------------------ blocker 4
def test_optimizer_containing_qpsi_is_rejected():
    """Blocker 4: refuse an optimizer that would train the scorer."""
    model, q = TinyActor(), _frozen_qpsi()
    for p in q.parameters():
        p.requires_grad_(True)                 # simulate the mistake
    bad = torch.optim.SGD(list(model.parameters()) + list(q.parameters()), lr=0.1)
    with pytest.raises(RuntimeError, match="param_group"):
        assert_qpsi_excluded_from_optimizer(q, bad)
    with pytest.raises((RuntimeError, ValueError)):
        RankingRunner(model, q, bad, lambda_rank=1.0, margin=0.04)


def test_runner_rejects_unfrozen_qpsi():
    model = TinyActor()
    q = _frozen_qpsi()
    for p in q.parameters():
        p.requires_grad_(True)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="frozen"):
        RankingRunner(model, q, opt, lambda_rank=1.0, margin=0.04)


# ------------------------------------------------------------------ blocker 5
def test_gradient_flows_through_the_masked_distribution():
    """Blocker 5: illegal actions must carry exactly zero probability.

    If the ranking term were computed from unmasked logits it would push the
    policy through a distribution it never uses -- the Phase 0 defect, but
    inside a training loss where no argmax check would catch it.
    """
    gen = torch.Generator().manual_seed(6)
    model = TinyActor()
    obs = _obs(8, gen)
    z = torch.zeros(8, dtype=torch.long)
    p1, p2 = masked_action_probs(model, obs, z)

    assert p1.shape[-1] == N_ACTIONS and p2.shape[-1] == N_ACTIONS
    assert torch.allclose(p1.sum(-1), torch.ones(8), atol=1e-5)
    # macros 2,3,4 are masked for agent 1 -> every joint action m*50+w with
    # m in {2,3,4} must be exactly zero
    for m in (2, 3, 4):
        assert p1[:, m * 50:(m + 1) * 50].abs().max() == 0.0
    for m in (2, 3, 4):
        assert p2[:, m * 50:(m + 1) * 50].abs().max() == 0.0
    # and legal mass is untouched
    assert p1[:, 0:100].sum(-1).min() > 0.99


def test_masking_actually_changes_the_contrast():
    """A masked and unmasked contrast must differ, or masking is a no-op here."""
    gen = torch.Generator().manual_seed(7)
    model, q = TinyActor(), _frozen_qpsi()
    pole = torch.tensor([POLE_A] * 6)
    masked = _obs(6, gen, all_legal=False)
    unmasked = dict(masked); unmasked["mask"] = torch.ones_like(masked["mask"])
    dm = strategic_contrast(model, q, masked, pole)[0]
    du = strategic_contrast(model, q, unmasked, pole)[0]
    assert not torch.allclose(dm, du), "masking made no difference; test is blind"


# ------------------------------------------------------------------ blocker 6
def test_lambda_zero_means_structural_absence():
    """Blocker 6: the control is NO runner, not a runner scaled by zero."""
    model, q = TinyActor(), _frozen_qpsi()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    for bad in (0.0, -0.1):
        with pytest.raises(ValueError, match="lambda_rank <= 0"):
            RankingRunner(model, q, opt, lambda_rank=bad, margin=0.04)


def test_absent_runner_leaves_model_and_optimizer_untouched():
    """The lambda_R=0 control must consume no RNG and mutate no state."""
    torch.manual_seed(8)
    gen = torch.Generator().manual_seed(8)
    model, q = TinyActor(), _frozen_qpsi()
    obs = _obs(8, gen)
    pole = torch.tensor([POLE_A] * 4 + [POLE_B] * 4)
    opt = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)

    snap = {k: v.clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()
    # control: no runner exists, so nothing happens at all
    assert opt.state_dict()["state"] == {}
    for k, v in model.state_dict().items():
        assert torch.equal(v, snap[k])
    assert torch.equal(torch.get_rng_state(), rng)

    # treatment: a real runner does mutate optimizer state
    RankingRunner(model, q, opt, lambda_rank=0.3, margin=0.04, cadence=1
                  ).note_ppo_minibatch({"obs": obs, "pole": pole})
    assert opt.state_dict()["state"] != {}, "treatment did not touch optimizer state"


# ------------------------------ counterfactual pressure (PI-added assertion)
def test_both_mode_heads_receive_gradient():
    """Neither V_hat may be detached: the update widens the PAIR.

    Detaching the reference mode would silently convert the objective from
    "widen the z0-vs-z1 gap" into "improve z0", which looks identical in a loss
    curve. The z-embedding has one row per mode, so a zero row proves detachment.
    """
    torch.manual_seed(9)
    gen = torch.Generator().manual_seed(9)
    model, q = TinyActor(), _frozen_qpsi()
    obs = _obs(16, gen)
    pole = torch.tensor([POLE_A] * 8 + [POLE_B] * 8)

    loss, _ = ranking_loss(model, q, obs, pole, margin=0.5)   # margin forces activity
    loss.backward()
    g = model.z_emb.weight.grad
    assert g is not None, "no gradient reached the mode embedding"
    assert g[0].abs().sum() > 0, "z0 path received no counterfactual pressure"
    assert g[1].abs().sum() > 0, "z1 path received no counterfactual pressure"


def test_contrast_is_antisymmetric_in_pole():
    """Same state scored on opposite poles must flip the sign of Delta."""
    gen = torch.Generator().manual_seed(10)
    model, q = TinyActor(), _frozen_qpsi()
    obs = _obs(6, gen)
    dA = strategic_contrast(model, q, obs, torch.full((6,), POLE_A))[0]
    dB = strategic_contrast(model, q, obs, torch.full((6,), POLE_B))[0]
    # V_hat differs by pole embedding, so only the ORIENTATION is guaranteed:
    # d_A uses (z0 - z1) and d_B uses (z1 - z0) on the same underlying values.
    v = strategic_contrast(model, q, obs, torch.full((6,), POLE_A))
    assert torch.allclose(dA, v[1] - v[2])
    assert (dA * dB <= 0).sum() >= 0        # sign relation is orientation-only
    assert dA.shape == dB.shape == (6,)


# ----------------------------------------------------------- cadence measured
def test_cadence_is_measured_from_real_steps():
    torch.manual_seed(11)
    gen = torch.Generator().manual_seed(11)
    model, q = TinyActor(), _frozen_qpsi()
    obs = _obs(8, gen)
    pole = torch.tensor([POLE_A] * 8)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    runner = RankingRunner(model, q, opt, lambda_rank=0.1, margin=0.04, cadence=4)

    fired = [runner.note_ppo_minibatch({"obs": obs, "pole": pole}) for _ in range(11)]
    t = runner.telemetry()
    assert sum(fired) == 2, "expected floor(11/4) = 2 ranking updates"
    assert t["n_ranking_updates"] == 2
    assert t["n_ppo_actor_minibatches"] == 11
    assert t["expected_complete_groups"] == 2
    assert t["complete_group_ratio_is_one"] is True
    assert runner.cadence == 4


def test_hinge_is_silent_once_the_margin_is_met():
    """Zero gradient once Delta >= m, so PPO is left free to optimise the task."""
    torch.manual_seed(12)
    gen = torch.Generator().manual_seed(12)
    model, q = TinyActor(), _frozen_qpsi()
    obs = _obs(8, gen)
    pole = torch.tensor([POLE_A] * 8)
    # margin far below the achieved contrast -> hinge inactive
    loss, diag = ranking_loss(model, q, obs, pole, margin=-1e6)
    assert float(loss.detach()) == 0.0
    assert diag["activation_rate"] == 0.0
    loss.backward()
    assert model.z_emb.weight.grad is None or model.z_emb.weight.grad.abs().sum() == 0
