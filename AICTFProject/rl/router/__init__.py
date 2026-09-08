"""Contextual Q-value and advantage routers for V6I11/V6I12 bandit-style latent selection.

V6I25 counterfactual geometry→z helpers live in
``rl.router.counterfactual_router`` (soft Q-targets + cross-fitted oracle).
"""
from rl.router.q_value_router import ContextualQRouter, QRouterReplayBuffer, train_q_router
from rl.router.advantage_router import (
    ContextualVBaseline,
    AdvantageRouter,
    train_advantage_router,
)

__all__ = [
    "ContextualQRouter",
    "QRouterReplayBuffer",
    "train_q_router",
    "ContextualVBaseline",
    "AdvantageRouter",
    "train_advantage_router",
]
