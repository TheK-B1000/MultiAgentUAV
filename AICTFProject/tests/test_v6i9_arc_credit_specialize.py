"""Pinning tests for the v6i9 arc-credit *specialize* preset.

Guards the entropy-mode wiring bug: the preset must engage the rollout-level
MARGINAL coverage path (``latent_entropy_mode='marginal'`` +
``latent_entropy_objective='maximize'``), not merely set the legacy ``h_mode``
alias (which has no runtime consumer in the entropy path). Setting only
``h_mode`` silently leaves ``latent_lam_h`` as a CONDITIONAL entropy-maximization
term — the exact opposite of the preset's "decisive within each context" intent.
"""
from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rl.config.ppo_config import PPOConfig
from rl.presets import apply_preset

_PARENT = "v6i9_arc_credit_running_mean_hardpool"
_SPECIALIZE = "v6i9_arc_credit_specialize_hardpool"
_ALIASES = [
    "v6i9_arc_credit_specialize",
    "v6i9_arc_credit_specialize_hardpool",
    "plan_faithful_latent_v6i9_arc_credit_specialize_hardpool",
]


def _resolved(name: str) -> dict:
    return dataclasses.asdict(apply_preset(PPOConfig(), name))


class TestSpecializeEntropyWiring(unittest.TestCase):
    def test_marginal_runtime_field_is_set(self) -> None:
        cfg = _resolved(_SPECIALIZE)
        # The field the runtime + banner actually read.
        self.assertEqual(cfg["latent_entropy_mode"], "marginal")
        # Marginal path also requires an active objective (h_goal != "none").
        self.assertEqual(cfg["latent_entropy_objective"], "maximize")
        # Legacy alias kept consistent (harmless, but should not contradict).
        self.assertEqual(cfg["h_mode"], "marginal")

    def test_entropy_balance_values(self) -> None:
        cfg = _resolved(_SPECIALIZE)
        self.assertEqual(cfg["router_ent_coef"], 0.001)
        self.assertEqual(cfg["latent_lam_h"], 0.01)

    def test_marginal_coverage_path_would_engage(self) -> None:
        """Replicates the runtime gate in entropy_objectives.RolloutMarginalPrep."""
        cfg = _resolved(_SPECIALIZE)
        mode = str(cfg.get("latent_entropy_mode") or "conditional").lower()
        goal = str(cfg.get("latent_entropy_objective") or "maximize").lower()
        lam_h = float(cfg.get("latent_lam_h") or 0.0)
        has_dedicated_router_opt = float(cfg.get("latent_episode_strategy_lr") or 0.0) > 0.0
        would_apply = (
            bool(cfg.get("use_latent_strategy"))
            and mode == "marginal"
            and not has_dedicated_router_opt
            and lam_h > 0.0
            and goal != "none"
            and not bool(cfg.get("fixed_latent_strategy"))
        )
        self.assertTrue(
            would_apply,
            msg="Marginal coverage path would NOT engage; latent_lam_h would act "
            "as conditional entropy instead of marginal coverage.",
        )

    def test_arc_credit_channel_unchanged_vs_parent(self) -> None:
        parent = _resolved(_PARENT)
        cfg = _resolved(_SPECIALIZE)
        for key in (
            "latent_arc_credit_enabled",
            "latent_arc_credit_baseline",
            "latent_arc_credit_coef",
            "latent_arc_credit_min_len",
            "latent_strategy_ppo_coef",
        ):
            self.assertEqual(cfg[key], parent[key], msg=f"{key} drifted from parent")
        self.assertTrue(cfg["latent_arc_credit_enabled"])
        self.assertEqual(cfg["latent_arc_credit_baseline"], "running_mean")
        self.assertEqual(cfg["latent_strategy_ppo_coef"], 0.0)

    def test_single_purpose_diff_vs_parent(self) -> None:
        parent = _resolved(_PARENT)
        cfg = _resolved(_SPECIALIZE)
        diff = {k for k in set(parent) | set(cfg) if parent.get(k) != cfg.get(k)}
        expected = {
            "h_mode",
            "latent_entropy_mode",
            "latent_entropy_objective",
            "latent_lam_h",
            "router_ent_coef",
            "run_tag",
        }
        self.assertEqual(
            diff,
            expected,
            msg=f"Unexpected resolved-config delta vs {_PARENT}: {diff ^ expected}",
        )

    def test_aliases_resolve_identically(self) -> None:
        base = _resolved(_ALIASES[0])
        base.pop("run_tag", None)
        for alias in _ALIASES[1:]:
            other = _resolved(alias)
            other.pop("run_tag", None)
            self.assertEqual(base, other, msg=f"alias {alias} diverged")


if __name__ == "__main__":
    unittest.main()
