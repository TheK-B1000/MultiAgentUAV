"""Action-conditioned strategic payoff scorer (Phase 0 / SPPPO).

Q_psi(o, a1, a2, p) -- centralised over the joint action, with a low-rank
interaction term so coordination between the two robots is representable.
"""
from rl.scorer.qpsi import QPsi, QPsiConfig, joint_action_index

__all__ = ["QPsi", "QPsiConfig", "joint_action_index"]
