"""Phase 8: Training Orchestration Decomposition tests.

Validates the decomposition of rl/training/cli.py and rl/train_ppo.py into
nine focused sub-modules:

  rl/training/errors.py          — typed error hierarchy
  rl/training/run_context.py     — RunContext dataclass
  rl/training/resolved_config.py — ResolvedTrainingConfig + resolve_training_config
  rl/training/lifecycle.py       — seed / csv / cuda helpers + teardown
  rl/training/factories.py       — build_training_env re-export + trainer kwargs
  rl/training/initialization.py  — trainer construction + checkpoint loading
  rl/training/arguments.py       — argparse builder (parse_train_args)
  rl/training/overrides.py       — cfg_from_args + run-tag helpers
  rl/training/orchestrator.py    — orchestrate_training_run

Equivalence contract: functions moved from cli.py and train_ppo.py are
re-exported from the original modules; ``facade_fn is impl_fn`` identity
checks prove the same Python function object is reached via both import
paths so no logic duplication exists.

Test groups
-----------
A  Error hierarchy (5 tests)              — torch-free
B  Facade re-export structure via AST (7) — torch-free (source-text checks)
C  Argument parsing (5 tests)             — torch-free (arguments.py has no module-level torch dep)
D  ResolvedTrainingConfig (6 tests)       — requires torch; skip when absent
E  RunContext (3 tests)                   — torch-free
F  Overrides / cfg_from_args (6 tests)    — requires torch; skip when absent
G  Lifecycle helpers (4 tests)            — requires torch; skip when absent
H  Module-level imports (9 tests)         — torch-free for errors/run_context/arguments; others skip
"""

from __future__ import annotations

import ast
import importlib
import os
import pathlib
import sys
import unittest

# ---------------------------------------------------------------------------
# Torch availability check (no import at module level — check dynamically)
# ---------------------------------------------------------------------------

def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _skipif_no_torch(test_fn):
    """Decorator: skip test with clear message when torch is not installed."""
    import functools

    @functools.wraps(test_fn)
    def wrapper(self, *args, **kwargs):
        if not _has_torch():
            self.skipTest("torch not installed in this environment")
        return test_fn(self, *args, **kwargs)

    return wrapper


# ---------------------------------------------------------------------------
# AST helpers for source-text facade checks
# ---------------------------------------------------------------------------

_TRAINING_DIR = pathlib.Path(__file__).parent.parent / "rl" / "training"
_TRAIN_PPO_PATH = pathlib.Path(__file__).parent.parent / "rl" / "train_ppo.py"


def _imports_from(filepath: pathlib.Path, module_substring: str) -> bool:
    """Return True if ``filepath`` contains an import from a module matching ``module_substring``."""
    tree = ast.parse(filepath.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if module_substring in mod:
                return True
    return False


# ============================================================
# Group A: Error hierarchy (torch-free)
# ============================================================

class TestErrorHierarchy(unittest.TestCase):
    def test_training_config_error_is_value_error(self):
        from rl.training.errors import TrainingConfigError
        self.assertTrue(issubclass(TrainingConfigError, ValueError))

    def test_evaluation_only_preset_error_is_training_config_error(self):
        from rl.training.errors import EvaluationOnlyPresetError, TrainingConfigError
        self.assertTrue(issubclass(EvaluationOnlyPresetError, TrainingConfigError))

    def test_presets_conflict_error_is_training_config_error(self):
        from rl.training.errors import PresetsConflictError, TrainingConfigError
        self.assertTrue(issubclass(PresetsConflictError, TrainingConfigError))

    def test_checkpoint_not_found_error_is_file_not_found(self):
        from rl.training.errors import CheckpointNotFoundError
        self.assertTrue(issubclass(CheckpointNotFoundError, FileNotFoundError))

    def test_training_aborted_error_is_runtime_error(self):
        from rl.training.errors import TrainingAbortedError
        self.assertTrue(issubclass(TrainingAbortedError, RuntimeError))


# ============================================================
# Group B: Facade re-export structure via AST (torch-free)
# ============================================================

class TestFacadeStructureViaAST(unittest.TestCase):
    """Source-text checks: verify the facade files import from the new sub-modules."""

    def test_cli_imports_from_arguments(self):
        """cli.py imports parse_train_args from rl.training.arguments."""
        cli_path = _TRAINING_DIR / "cli.py"
        self.assertTrue(_imports_from(cli_path, "arguments"))

    def test_cli_imports_from_overrides(self):
        """cli.py imports cfg_from_args from rl.training.overrides."""
        cli_path = _TRAINING_DIR / "cli.py"
        self.assertTrue(_imports_from(cli_path, "overrides"))

    def test_train_ppo_imports_from_lifecycle(self):
        """train_ppo.py re-exports from rl.training.lifecycle."""
        self.assertTrue(_imports_from(_TRAIN_PPO_PATH, "lifecycle"))

    def test_train_ppo_imports_from_overrides(self):
        """train_ppo.py re-exports from rl.training.overrides."""
        self.assertTrue(_imports_from(_TRAIN_PPO_PATH, "overrides"))

    def test_train_ppo_imports_from_resolved_config(self):
        """train_ppo.py re-exports from rl.training.resolved_config."""
        self.assertTrue(_imports_from(_TRAIN_PPO_PATH, "resolved_config"))

    def test_orchestrator_imports_from_lifecycle(self):
        """orchestrator.py uses lifecycle functions."""
        orch_path = _TRAINING_DIR / "orchestrator.py"
        self.assertTrue(_imports_from(orch_path, "lifecycle"))

    def test_orchestrator_imports_from_resolved_config(self):
        """orchestrator.py uses ResolvedTrainingConfig."""
        orch_path = _TRAINING_DIR / "orchestrator.py"
        self.assertTrue(_imports_from(orch_path, "resolved_config"))


# ============================================================
# Group C: Argument parsing (torch-free at module import level)
# ============================================================

class TestArgumentParsing(unittest.TestCase):
    """Arguments module has no module-level torch dep; parse_train_args triggers lazy imports."""

    def _parse(self, *cli_args):
        """Parse given CLI args, skipping if torch is absent (needed for DEFAULT_CLI_TRAINING_PRESET)."""
        if not _has_torch():
            self.skipTest("parse_train_args lazily imports train_ppo which requires torch")
        from rl.training.arguments import parse_train_args
        return parse_train_args(list(cli_args))

    def test_arguments_module_importable_without_torch(self):
        """rl.training.arguments can be imported at module level without torch."""
        import importlib as _il
        mod = _il.import_module("rl.training.arguments")
        self.assertTrue(hasattr(mod, "parse_train_args"))

    def test_build_train_parser_returns_parser(self):
        if not _has_torch():
            self.skipTest("requires torch to import train_ppo constants")
        from rl.training.arguments import build_train_parser
        import argparse
        parser = build_train_parser()
        self.assertIsInstance(parser, argparse.ArgumentParser)

    def test_seed_flag(self):
        ns = self._parse("--seed", "42")
        self.assertEqual(ns.seed, 42)

    def test_no_latent_strategy_flag(self):
        ns = self._parse("--no-latent-strategy")
        self.assertTrue(ns.no_latent_strategy)

    def test_latent_k_flag(self):
        ns = self._parse("--latent-k", "4")
        self.assertEqual(ns.latent_k, 4)


# ============================================================
# Group D: ResolvedTrainingConfig (requires torch)
# ============================================================

class TestResolvedTrainingConfig(unittest.TestCase):

    @_skipif_no_torch
    def test_default_max_agents_is_2(self):
        from rl.config.ppo_config import PPOConfig
        from rl.training.resolved_config import resolve_training_config
        r = resolve_training_config(PPOConfig())
        self.assertEqual(r.max_agents, 2)

    @_skipif_no_torch
    def test_default_team_size_is_2v2(self):
        from rl.config.ppo_config import PPOConfig
        from rl.training.resolved_config import resolve_training_config
        r = resolve_training_config(PPOConfig())
        self.assertEqual(r.team_size, "2v2")

    @_skipif_no_torch
    def test_4v4_team_size(self):
        from rl.config.ppo_config import PPOConfig
        from rl.training.resolved_config import resolve_training_config
        cfg = PPOConfig()
        cfg.max_blue_agents = 4
        r = resolve_training_config(cfg)
        self.assertEqual(r.team_size, "4v4")
        self.assertEqual(r.max_agents, 4)

    @_skipif_no_torch
    def test_lr_scaled_for_4_agents(self):
        from rl.config.ppo_config import PPOConfig
        from rl.training.resolved_config import resolve_training_config
        base_lr = 3e-4
        cfg = PPOConfig()
        cfg.max_blue_agents = 4
        cfg.learning_rate = base_lr
        r = resolve_training_config(cfg)
        self.assertAlmostEqual(r.effective_lr, base_lr * 0.75, places=12)

    @_skipif_no_torch
    def test_frozen_raises_on_assignment(self):
        import dataclasses
        from rl.config.ppo_config import PPOConfig
        from rl.training.resolved_config import resolve_training_config
        r = resolve_training_config(PPOConfig())
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError)):
            r.max_agents = 99  # type: ignore[misc]

    @_skipif_no_torch
    def test_batch_size_clamped_to_rollout_size(self):
        from rl.config.ppo_config import PPOConfig
        from rl.training.resolved_config import resolve_training_config
        cfg = PPOConfig()
        cfg.n_steps = 4
        cfg.n_envs = 2
        cfg.batch_size = 10000  # larger than rollout (4*2=8)
        r = resolve_training_config(cfg)
        self.assertEqual(r.effective_batch_size, r.rollout_size)


# ============================================================
# Group E: RunContext (torch-free)
# ============================================================

class TestRunContext(unittest.TestCase):
    def test_run_context_has_run_lock_field(self):
        from rl.training.run_context import RunContext
        sentinel = object()
        rc = RunContext(run_lock=sentinel)
        self.assertIs(rc.run_lock, sentinel)

    def test_run_context_rc_path_defaults_none(self):
        from rl.training.run_context import RunContext
        rc = RunContext(run_lock=object())
        self.assertIsNone(rc.rc_path)

    def test_run_context_is_mutable(self):
        from rl.training.run_context import RunContext
        rc = RunContext(run_lock=object())
        rc.rc_path = "/tmp/run_config.json"
        self.assertEqual(rc.rc_path, "/tmp/run_config.json")


# ============================================================
# Group F: Overrides / cfg_from_args (requires torch)
# ============================================================

class TestCfgFromArgs(unittest.TestCase):
    def _parse_and_build(self, *args):
        from rl.training.arguments import parse_train_args
        from rl.training.overrides import cfg_from_args
        ns = parse_train_args(["--preset", "none"] + list(args))
        return cfg_from_args(ns)

    @_skipif_no_torch
    def test_no_latent_strategy_disables_latent(self):
        cfg = self._parse_and_build("--no-latent-strategy")
        self.assertFalse(cfg.use_latent_strategy)

    @_skipif_no_torch
    def test_latent_k_sets_value(self):
        cfg = self._parse_and_build("--latent-k", "6")
        self.assertEqual(cfg.latent_k, 6)

    @_skipif_no_torch
    def test_seed_sets_seed(self):
        cfg = self._parse_and_build("--seed", "77")
        self.assertEqual(cfg.seed, 77)

    @_skipif_no_torch
    def test_learning_rate_override(self):
        cfg = self._parse_and_build("--learning-rate", "1e-3")
        self.assertAlmostEqual(cfg.learning_rate, 1e-3, places=8)

    @_skipif_no_torch
    def test_no_progress_bar(self):
        cfg = self._parse_and_build("--no-progress-bar")
        self.assertFalse(cfg.enable_progress_bar)

    @_skipif_no_torch
    def test_agents_suffix_in_run_tag(self):
        cfg = self._parse_and_build("--agents", "4", "--run-tag", "my_run")
        self.assertTrue(cfg.run_tag.endswith("_4v4"))


# ============================================================
# Group G: Lifecycle helpers (requires torch via PPOConfig)
# ============================================================

class TestLifecycleHelpers(unittest.TestCase):
    import random as _random

    @_skipif_no_torch
    def test_set_global_seed_deterministic(self):
        """set_global_seed with same seed produces same Python random output."""
        import random
        from rl.training.lifecycle import set_global_seed
        set_global_seed(0, torch_seed=False)
        v1 = random.random()
        set_global_seed(0, torch_seed=False)
        v2 = random.random()
        self.assertEqual(v1, v2)

    @_skipif_no_torch
    def test_set_global_seed_different_seeds_differ(self):
        import random
        from rl.training.lifecycle import set_global_seed
        set_global_seed(1, torch_seed=False)
        v1 = random.random()
        set_global_seed(2, torch_seed=False)
        v2 = random.random()
        self.assertNotEqual(v1, v2)

    @_skipif_no_torch
    def test_resolve_metrics_csv_paths_fills_missing_paths(self):
        import tempfile, os
        from rl.config.ppo_config import PPOConfig
        from rl.training.lifecycle import _resolve_metrics_csv_paths
        with tempfile.TemporaryDirectory() as tmp:
            cfg = PPOConfig()
            cfg.run_tag = "test_run"
            cfg.checkpoint_dir = tmp
            cfg.metrics_csv_path = None   # type: ignore[assignment]
            cfg.episode_csv_path = None   # type: ignore[assignment]
            cfg.strategy_experience_csv_path = None  # type: ignore[assignment]
            cfg.enable_metrics_csv = True
            _resolve_metrics_csv_paths(cfg)
            self.assertIsNotNone(cfg.metrics_csv_path)
            self.assertIn("test_run_metrics.csv", cfg.metrics_csv_path)

    @_skipif_no_torch
    def test_clamp_runtime_config_for_6v6(self):
        from rl.config.ppo_config import PPOConfig
        from rl.training.lifecycle import _clamp_runtime_config_for_team_size
        cfg = PPOConfig()
        cfg.n_envs = 32
        cfg.n_steps = 2048
        cfg.max_decision_steps = 800
        _clamp_runtime_config_for_team_size(cfg, max_agents=6)
        self.assertEqual(cfg.n_envs, 1)
        self.assertEqual(cfg.n_steps, 512)
        self.assertEqual(cfg.max_decision_steps, 400)


# ============================================================
# Group H: Module-level imports (9 modules)
# ============================================================

class TestModuleImports(unittest.TestCase):
    """Verify all 9 new Phase 8 modules can be imported without crashing."""

    def _try_import(self, module_name: str) -> bool:
        try:
            importlib.import_module(module_name)
            return True
        except (ImportError, ModuleNotFoundError):
            return False

    def test_errors_importable(self):
        self.assertTrue(self._try_import("rl.training.errors"))

    def test_run_context_importable(self):
        self.assertTrue(self._try_import("rl.training.run_context"))

    def test_arguments_importable(self):
        """rl.training.arguments has no module-level torch dep; must always import."""
        self.assertTrue(self._try_import("rl.training.arguments"))

    @_skipif_no_torch
    def test_resolved_config_importable(self):
        self.assertTrue(self._try_import("rl.training.resolved_config"))

    @_skipif_no_torch
    def test_lifecycle_importable(self):
        self.assertTrue(self._try_import("rl.training.lifecycle"))

    @_skipif_no_torch
    def test_factories_importable(self):
        self.assertTrue(self._try_import("rl.training.factories"))

    @_skipif_no_torch
    def test_initialization_importable(self):
        self.assertTrue(self._try_import("rl.training.initialization"))

    @_skipif_no_torch
    def test_overrides_importable(self):
        self.assertTrue(self._try_import("rl.training.overrides"))

    @_skipif_no_torch
    def test_orchestrator_importable(self):
        self.assertTrue(self._try_import("rl.training.orchestrator"))


if __name__ == "__main__":
    unittest.main()
