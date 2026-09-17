"""Gated GET_FLAG macro preservation: surgical, not global JSD."""
from __future__ import annotations

import torch

from macro_actions import MacroAction
from rl.custom_ppo.getflag_preservation import (
    GET_FLAG,
    GO_TO,
    VEC_OWN_CARRYING_IDX,
    getflag_preserve_loss,
)

N_AGENTS = 2
MACRO = 5
WP = 50
DIMS = (MACRO, WP, MACRO, WP)
FLAT = sum(DIMS)
B = 4


class _Stub:
    n_agents = N_AGENTS
    action_dims = DIMS

    def __init__(self, macro: torch.Tensor):
        # macro: (B, N, 5)
        self.macro = macro

    def policy_logits(self, obs, z_idx=None, **kwargs):
        b = int(self.macro.shape[0])
        parts = []
        zeros_wp = self.macro.new_zeros((b, WP))
        for i in range(N_AGENTS):
            parts.append(self.macro[:, i])
            parts.append(zeros_wp)
        return torch.cat(parts, dim=-1)

    def _mask_logits(self, logits, mask):
        return logits.masked_fill(mask <= 0, -1.0e4)


def _obs(*, carrying, legal=True):
    vec = torch.zeros(B, N_AGENTS, 20)
    vec[..., VEC_OWN_CARRYING_IDX] = carrying.float()
    mask = torch.ones(B, FLAT)
    if not legal:
        # one-hot GO_TO on every macro head -> not decision-eligible
        mask[:, :] = 0
        mask[:, 0] = 1
        mask[:, MACRO + WP] = 1
    grid = torch.zeros(B, N_AGENTS, 7, 4, 4)
    return {
        "grid": grid,
        "vec": vec,
        "agent_mask": torch.ones(B, N_AGENTS),
        "mask": mask,
    }


def _one_hot_macro(idx: int, peak: float = 5.0) -> torch.Tensor:
    m = torch.zeros(B, N_AGENTS, MACRO)
    m[..., idx] = peak
    return m


def test_loss_zero_when_student_already_prefers_getflag():
    macros = _one_hot_macro(GET_FLAG)
    student, anchor = _Stub(macros.clone().requires_grad_(True)), _Stub(macros)
    carrying = torch.zeros(B, N_AGENTS, dtype=torch.bool)
    loss, tel = getflag_preserve_loss(student, anchor, _obs(carrying=carrying))
    assert tel["n_gated"] == B * N_AGENTS
    assert float(loss) < 0.05


def test_loss_positive_when_student_prefers_goto_on_gated_states():
    student = _Stub(_one_hot_macro(GO_TO).requires_grad_(True))
    anchor = _Stub(_one_hot_macro(GET_FLAG))
    carrying = torch.zeros(B, N_AGENTS, dtype=torch.bool)
    loss, tel = getflag_preserve_loss(student, anchor, _obs(carrying=carrying))
    assert tel["n_gated"] == B * N_AGENTS
    assert float(loss) > 1.0
    loss.backward()
    assert student.macro.grad is not None
    # GET_FLAG logit should be pushed up (negative grad on the NLL of GET_FLAG
    # flows through log_softmax; the GET_FLAG column of the student macro
    # receives a negative gradient).
    gf_grad = student.macro.grad[..., GET_FLAG].mean()
    goto_grad = student.macro.grad[..., GO_TO].mean()
    assert float(gf_grad) < 0
    assert float(goto_grad) > 0


def test_carrying_ungates_even_when_anchor_wants_getflag():
    student = _Stub(_one_hot_macro(GO_TO).requires_grad_(True))
    anchor = _Stub(_one_hot_macro(GET_FLAG))
    carrying = torch.ones(B, N_AGENTS, dtype=torch.bool)
    loss, tel = getflag_preserve_loss(student, anchor, _obs(carrying=carrying))
    assert tel["n_gated"] == 0
    assert float(loss) == 0.0


def test_anchor_goto_does_not_gate():
    student = _Stub(_one_hot_macro(GO_TO).requires_grad_(True))
    anchor = _Stub(_one_hot_macro(GO_TO))
    carrying = torch.zeros(B, N_AGENTS, dtype=torch.bool)
    loss, tel = getflag_preserve_loss(student, anchor, _obs(carrying=carrying))
    assert tel["n_gated"] == 0
    assert float(loss) == 0.0


def test_locked_commit_does_not_gate():
    student = _Stub(_one_hot_macro(GO_TO).requires_grad_(True))
    anchor = _Stub(_one_hot_macro(GET_FLAG))
    carrying = torch.zeros(B, N_AGENTS, dtype=torch.bool)
    loss, tel = getflag_preserve_loss(
        student, anchor, _obs(carrying=carrying, legal=False)
    )
    assert tel["n_gated"] == 0
    assert float(loss) == 0.0


def test_waypoint_logits_do_not_enter_loss():
    macros = _one_hot_macro(GO_TO)
    student = _Stub(macros.clone().requires_grad_(True))
    anchor = _Stub(_one_hot_macro(GET_FLAG))
    carrying = torch.zeros(B, N_AGENTS, dtype=torch.bool)
    loss, _ = getflag_preserve_loss(student, anchor, _obs(carrying=carrying))
    # Reconstruct flat logits and confirm the loss equals NLL on GET_FLAG
    # of the MACRO head only, independent of waypoint columns (which are 0).
    assert GET_FLAG == int(MacroAction.GET_FLAG)
    logp = student.macro.log_softmax(dim=-1)[..., GET_FLAG]
    expected = -logp.mean()
    assert torch.allclose(loss, expected)


def test_runner_refuses_zero_lambda():
    from rl.custom_ppo.getflag_preservation import GetflagPreserveRunner

    try:
        GetflagPreserveRunner(_Stub(_one_hot_macro(GET_FLAG)), None, _Stub(_one_hot_macro(GET_FLAG)),
                              lambda_preserve=0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for lambda<=0")
