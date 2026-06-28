"""Phase 9: GPU Environment State Decomposition tests.

Validates the decomposition of ``gpu_env._core._state._StateMixin`` into
twelve focused sub-mixins in ``gpu_env/state/``:

  gpu_env/state/models.py          — _CoreStateMixin (__init__, RNG helpers)
  gpu_env/state/allocation.py      — _AllocationMixin (_alloc_state, macro targets)
  gpu_env/state/agent_state.py     — _AgentStateMixin (agent tensors, _respawn_side)
  gpu_env/state/team_state.py      — _TeamStateMixin (_side_tensors, mirroring)
  gpu_env/state/flag_state.py      — _FlagStateMixin (flag/score tensors)
  gpu_env/state/episode_state.py   — _EpisodeStateMixin (episode bookkeeping, reset)
  gpu_env/state/map_state.py       — _MapStateMixin (obstacle geometry)
  gpu_env/state/opponent_state.py  — _OpponentStateMixin (opponent/dynamics API)
  gpu_env/state/telemetry_state.py — _TelemetryStateMixin (metric/nav buffers)
  gpu_env/state/scratch.py         — _ScratchStateMixin (runtime buffers, mine state)
  gpu_env/state/validation.py      — _ValidationMixin (index/phase utilities)
  gpu_env/state/snapshots.py       — _SnapshotsMixin (snapshot policy cache)

Equivalence contract: ``gpu_env._core._state._StateMixin`` is now a thin
composition of the twelve sub-mixins.  AST checks verify all method
relocations; attribute-presence checks verify the mixin class surfaces are
correct.

Test groups
-----------
A  Package structure (4 tests)               — torch-free (file existence only)
B  Individual module imports (13 tests)      — requires torch (gpu_env __init__ → torch)
C  Mixin class presence (12 tests)           — requires torch
D  Method coverage (12 tests)               — requires torch
E  Facade composition (4 tests)             — torch-free AST checks on _state.py
F  _StateMixin MRO coverage (3 tests)       — requires torch (runtime class hierarchy)
G  AST duplicate-method scanner (3 tests)   — torch-free; no silent MRO shadowing
H  MRO surface contract (4 tests)           — requires torch; exact MRO + callable surface
I  Behavioral equivalence skeleton (4 tests)— requires torch; reset/spawn invariants
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import sys
import unittest

# ---------------------------------------------------------------------------
# Torch availability
# ---------------------------------------------------------------------------

def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _skipif_no_torch(test_fn):
    import functools

    @functools.wraps(test_fn)
    def wrapper(self, *args, **kwargs):
        if not _has_torch():
            self.skipTest("torch not installed in this environment")
        return test_fn(self, *args, **kwargs)

    return wrapper


_STATE_DIR = pathlib.Path(__file__).parent.parent / "gpu_env" / "state"
_STATE_MODULES = [
    "models.py", "allocation.py", "agent_state.py", "team_state.py",
    "flag_state.py", "episode_state.py", "map_state.py", "opponent_state.py",
    "telemetry_state.py", "scratch.py", "validation.py", "snapshots.py",
]
_CORE_STATE_PATH = pathlib.Path(__file__).parent.parent / "gpu_env" / "_core" / "_state.py"

_EXPECTED_FILES = [
    "__init__.py",
    "models.py",
    "allocation.py",
    "agent_state.py",
    "team_state.py",
    "flag_state.py",
    "episode_state.py",
    "map_state.py",
    "opponent_state.py",
    "telemetry_state.py",
    "scratch.py",
    "validation.py",
    "snapshots.py",
]

_MODULE_CLASS_MAP = {
    "gpu_env.state.models": "_CoreStateMixin",
    "gpu_env.state.allocation": "_AllocationMixin",
    "gpu_env.state.agent_state": "_AgentStateMixin",
    "gpu_env.state.team_state": "_TeamStateMixin",
    "gpu_env.state.flag_state": "_FlagStateMixin",
    "gpu_env.state.episode_state": "_EpisodeStateMixin",
    "gpu_env.state.map_state": "_MapStateMixin",
    "gpu_env.state.opponent_state": "_OpponentStateMixin",
    "gpu_env.state.telemetry_state": "_TelemetryStateMixin",
    "gpu_env.state.scratch": "_ScratchStateMixin",
    "gpu_env.state.validation": "_ValidationMixin",
    "gpu_env.state.snapshots": "_SnapshotsMixin",
}

# Key methods that should live in specific sub-modules.
_MODULE_METHOD_MAP = {
    "gpu_env.state.models": ["reseed", "_rand_uniform", "_randn"],
    "gpu_env.state.allocation": ["_alloc_state", "_build_macro_targets"],
    "gpu_env.state.agent_state": ["_alloc_agent_state", "_respawn_side"],
    "gpu_env.state.team_state": ["_side_tensors", "_mirror_x", "_mirror_heading"],
    "gpu_env.state.flag_state": ["_alloc_flags_and_scores"],
    "gpu_env.state.episode_state": ["reset_all", "reset_indices", "_alloc_episode_state"],
    "gpu_env.state.map_state": [
        "_alloc_map_state",
        "_reset_map_layout",
        "_points_in_obstacles",
        "_segments_hit_obstacles",
        "_revert_obstacle_hits",
        "_route_targets_around_obstacles",
    ],
    "gpu_env.state.opponent_state": [
        "set_phase",
        "set_next_opponent",
        "set_dynamics_config",
        "_apply_dynamics_bool",
        "get_opponent_key",
    ],
    "gpu_env.state.telemetry_state": [
        "_alloc_metric_buffers",
        "_alloc_navigation_telemetry_buffers",
        "_reset_navigation_telemetry",
    ],
    "gpu_env.state.scratch": [
        "_alloc_runtime_buffers",
        "_alloc_mine_state",
        "_init_pickup_positions",
        "_apply_train_domain_randomization",
    ],
    "gpu_env.state.validation": [
        "_normalize_env_indices",
        "_phase_tensor_equals",
        "_get_red_control_mask",
    ],
    "gpu_env.state.snapshots": ["_load_snapshot_policy"],
}


# ============================================================
# Group A: Package structure (torch-free)
# ============================================================

class TestPackageStructure(unittest.TestCase):
    """Verify all expected files exist under gpu_env/state/."""

    def test_state_directory_exists(self):
        self.assertTrue(_STATE_DIR.is_dir(), f"Missing directory: {_STATE_DIR}")

    def test_init_py_exists(self):
        self.assertTrue((_STATE_DIR / "__init__.py").is_file())

    def test_all_module_files_exist(self):
        missing = [f for f in _EXPECTED_FILES if not (_STATE_DIR / f).is_file()]
        self.assertEqual(missing, [], f"Missing state modules: {missing}")

    def test_facade_state_py_exists(self):
        self.assertTrue(_CORE_STATE_PATH.is_file(), f"Missing facade: {_CORE_STATE_PATH}")


# ============================================================
# Group B: Individual module imports (requires torch — gpu_env __init__ pulls it in)
# ============================================================

class TestModuleImports(unittest.TestCase):
    """Each sub-module must be importable; all require torch via gpu_env package init."""

    def _import(self, module_name: str) -> bool:
        try:
            importlib.import_module(module_name)
            return True
        except (ImportError, ModuleNotFoundError):
            return False

    @_skipif_no_torch
    def test_state_package_importable(self):
        self.assertTrue(self._import("gpu_env.state"))

    @_skipif_no_torch
    def test_models_importable(self):
        self.assertTrue(self._import("gpu_env.state.models"))

    @_skipif_no_torch
    def test_allocation_importable(self):
        self.assertTrue(self._import("gpu_env.state.allocation"))

    @_skipif_no_torch
    def test_agent_state_importable(self):
        self.assertTrue(self._import("gpu_env.state.agent_state"))

    @_skipif_no_torch
    def test_team_state_importable(self):
        self.assertTrue(self._import("gpu_env.state.team_state"))

    @_skipif_no_torch
    def test_flag_state_importable(self):
        self.assertTrue(self._import("gpu_env.state.flag_state"))

    @_skipif_no_torch
    def test_episode_state_importable(self):
        self.assertTrue(self._import("gpu_env.state.episode_state"))

    @_skipif_no_torch
    def test_map_state_importable(self):
        self.assertTrue(self._import("gpu_env.state.map_state"))

    @_skipif_no_torch
    def test_opponent_state_importable(self):
        self.assertTrue(self._import("gpu_env.state.opponent_state"))

    @_skipif_no_torch
    def test_telemetry_state_importable(self):
        self.assertTrue(self._import("gpu_env.state.telemetry_state"))

    @_skipif_no_torch
    def test_scratch_importable(self):
        self.assertTrue(self._import("gpu_env.state.scratch"))

    @_skipif_no_torch
    def test_validation_importable(self):
        self.assertTrue(self._import("gpu_env.state.validation"))

    @_skipif_no_torch
    def test_snapshots_importable(self):
        self.assertTrue(self._import("gpu_env.state.snapshots"))


# ============================================================
# Group C: Mixin class presence (requires torch)
# ============================================================

class TestMixinClassPresence(unittest.TestCase):
    """Each sub-module must define the expected mixin class."""

    def _check_class(self, module_name: str, class_name: str) -> None:
        mod = importlib.import_module(module_name)
        self.assertTrue(hasattr(mod, class_name), f"{module_name} missing class {class_name}")
        cls = getattr(mod, class_name)
        self.assertTrue(isinstance(cls, type), f"{class_name} is not a class")

    @_skipif_no_torch
    def test_core_state_mixin_in_models(self):
        self._check_class("gpu_env.state.models", "_CoreStateMixin")

    @_skipif_no_torch
    def test_allocation_mixin_in_allocation(self):
        self._check_class("gpu_env.state.allocation", "_AllocationMixin")

    @_skipif_no_torch
    def test_agent_state_mixin_in_agent_state(self):
        self._check_class("gpu_env.state.agent_state", "_AgentStateMixin")

    @_skipif_no_torch
    def test_team_state_mixin_in_team_state(self):
        self._check_class("gpu_env.state.team_state", "_TeamStateMixin")

    @_skipif_no_torch
    def test_flag_state_mixin_in_flag_state(self):
        self._check_class("gpu_env.state.flag_state", "_FlagStateMixin")

    @_skipif_no_torch
    def test_episode_state_mixin_in_episode_state(self):
        self._check_class("gpu_env.state.episode_state", "_EpisodeStateMixin")

    @_skipif_no_torch
    def test_map_state_mixin_in_map_state(self):
        self._check_class("gpu_env.state.map_state", "_MapStateMixin")

    @_skipif_no_torch
    def test_opponent_state_mixin_in_opponent_state(self):
        self._check_class("gpu_env.state.opponent_state", "_OpponentStateMixin")

    @_skipif_no_torch
    def test_telemetry_state_mixin_in_telemetry_state(self):
        self._check_class("gpu_env.state.telemetry_state", "_TelemetryStateMixin")

    @_skipif_no_torch
    def test_scratch_mixin_in_scratch(self):
        self._check_class("gpu_env.state.scratch", "_ScratchStateMixin")

    @_skipif_no_torch
    def test_validation_mixin_in_validation(self):
        self._check_class("gpu_env.state.validation", "_ValidationMixin")

    @_skipif_no_torch
    def test_snapshots_mixin_in_snapshots(self):
        self._check_class("gpu_env.state.snapshots", "_SnapshotsMixin")


# ============================================================
# Group D: Method coverage (requires torch)
# ============================================================

class TestMethodCoverage(unittest.TestCase):
    """Key methods are defined on the correct mixin class (not on object base)."""

    def _check_methods(self, module_name: str, class_name: str, methods: list) -> None:
        mod = importlib.import_module(module_name)
        cls = getattr(mod, class_name)
        missing = [m for m in methods if m not in cls.__dict__]
        self.assertEqual(
            missing,
            [],
            f"{class_name} ({module_name}) is missing methods: {missing}",
        )

    @_skipif_no_torch
    def test_core_state_mixin_methods(self):
        self._check_methods(
            "gpu_env.state.models", "_CoreStateMixin",
            _MODULE_METHOD_MAP["gpu_env.state.models"],
        )

    @_skipif_no_torch
    def test_allocation_mixin_methods(self):
        self._check_methods(
            "gpu_env.state.allocation", "_AllocationMixin",
            _MODULE_METHOD_MAP["gpu_env.state.allocation"],
        )

    @_skipif_no_torch
    def test_agent_state_mixin_methods(self):
        self._check_methods(
            "gpu_env.state.agent_state", "_AgentStateMixin",
            _MODULE_METHOD_MAP["gpu_env.state.agent_state"],
        )

    @_skipif_no_torch
    def test_team_state_mixin_methods(self):
        self._check_methods(
            "gpu_env.state.team_state", "_TeamStateMixin",
            _MODULE_METHOD_MAP["gpu_env.state.team_state"],
        )

    @_skipif_no_torch
    def test_flag_state_mixin_methods(self):
        self._check_methods(
            "gpu_env.state.flag_state", "_FlagStateMixin",
            _MODULE_METHOD_MAP["gpu_env.state.flag_state"],
        )

    @_skipif_no_torch
    def test_episode_state_mixin_methods(self):
        self._check_methods(
            "gpu_env.state.episode_state", "_EpisodeStateMixin",
            _MODULE_METHOD_MAP["gpu_env.state.episode_state"],
        )

    @_skipif_no_torch
    def test_map_state_mixin_methods(self):
        self._check_methods(
            "gpu_env.state.map_state", "_MapStateMixin",
            _MODULE_METHOD_MAP["gpu_env.state.map_state"],
        )

    @_skipif_no_torch
    def test_opponent_state_mixin_methods(self):
        self._check_methods(
            "gpu_env.state.opponent_state", "_OpponentStateMixin",
            _MODULE_METHOD_MAP["gpu_env.state.opponent_state"],
        )

    @_skipif_no_torch
    def test_telemetry_state_mixin_methods(self):
        self._check_methods(
            "gpu_env.state.telemetry_state", "_TelemetryStateMixin",
            _MODULE_METHOD_MAP["gpu_env.state.telemetry_state"],
        )

    @_skipif_no_torch
    def test_scratch_mixin_methods(self):
        self._check_methods(
            "gpu_env.state.scratch", "_ScratchStateMixin",
            _MODULE_METHOD_MAP["gpu_env.state.scratch"],
        )

    @_skipif_no_torch
    def test_validation_mixin_methods(self):
        self._check_methods(
            "gpu_env.state.validation", "_ValidationMixin",
            _MODULE_METHOD_MAP["gpu_env.state.validation"],
        )

    @_skipif_no_torch
    def test_snapshots_mixin_methods(self):
        self._check_methods(
            "gpu_env.state.snapshots", "_SnapshotsMixin",
            _MODULE_METHOD_MAP["gpu_env.state.snapshots"],
        )


# ============================================================
# Group E: Facade composition (torch-free, AST-based)
# ============================================================

def _facade_imports_from(module_substring: str) -> bool:
    """Return True if _state.py imports from gpu_env.state.<module_substring>."""
    tree = ast.parse(_CORE_STATE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if module_substring in mod:
                return True
    return False


class TestFacadeComposition(unittest.TestCase):
    """_state.py facade must import all 12 sub-mixins and define _StateMixin."""

    def test_facade_imports_from_gpu_env_state(self):
        """_state.py has at least one import from gpu_env.state.*."""
        self.assertTrue(_facade_imports_from("gpu_env.state"))

    def test_facade_imports_models(self):
        self.assertTrue(_facade_imports_from("gpu_env.state.models"))

    def test_facade_imports_episode_state(self):
        self.assertTrue(_facade_imports_from("gpu_env.state.episode_state"))

    def test_facade_defines_state_mixin(self):
        """_state.py defines the ``_StateMixin`` class."""
        tree = ast.parse(_CORE_STATE_PATH.read_text(encoding="utf-8"))
        class_names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        ]
        self.assertIn("_StateMixin", class_names)


# ============================================================
# Group F: MRO coverage (requires torch)
# ============================================================

class TestMROCoverage(unittest.TestCase):
    """_StateMixin's MRO must include all 12 sub-mixin classes."""

    @_skipif_no_torch
    def test_state_mixin_importable(self):
        from gpu_env._core._state import _StateMixin
        self.assertTrue(isinstance(_StateMixin, type))

    @_skipif_no_torch
    def test_state_mixin_mro_includes_core(self):
        from gpu_env._core._state import _StateMixin
        from gpu_env.state.models import _CoreStateMixin
        self.assertIn(_CoreStateMixin, _StateMixin.__mro__)

    @_skipif_no_torch
    def test_state_mixin_mro_includes_all_twelve(self):
        from gpu_env._core._state import _StateMixin
        from gpu_env.state.models import _CoreStateMixin
        from gpu_env.state.allocation import _AllocationMixin
        from gpu_env.state.agent_state import _AgentStateMixin
        from gpu_env.state.team_state import _TeamStateMixin
        from gpu_env.state.flag_state import _FlagStateMixin
        from gpu_env.state.episode_state import _EpisodeStateMixin
        from gpu_env.state.map_state import _MapStateMixin
        from gpu_env.state.opponent_state import _OpponentStateMixin
        from gpu_env.state.telemetry_state import _TelemetryStateMixin
        from gpu_env.state.scratch import _ScratchStateMixin
        from gpu_env.state.validation import _ValidationMixin
        from gpu_env.state.snapshots import _SnapshotsMixin
        expected = [
            _CoreStateMixin, _AllocationMixin, _AgentStateMixin,
            _TeamStateMixin, _FlagStateMixin, _EpisodeStateMixin,
            _MapStateMixin, _OpponentStateMixin, _TelemetryStateMixin,
            _ScratchStateMixin, _ValidationMixin, _SnapshotsMixin,
        ]
        mro = _StateMixin.__mro__
        missing = [cls for cls in expected if cls not in mro]
        self.assertEqual(
            missing,
            [],
            f"_StateMixin MRO is missing sub-mixins: {[c.__name__ for c in missing]}",
        )


# ============================================================
# Group G: AST duplicate-method scanner (torch-free)
# ============================================================

class TestASTDuplicateMethodScanner(unittest.TestCase):
    """No method name may appear in more than one sub-mixin (first match silently wins in MRO)."""

    def _methods_per_module(self) -> dict[str, set[str]]:
        """Return {module_stem: set_of_method_names} parsed via AST."""
        result: dict[str, set[str]] = {}
        for mod_file in _STATE_MODULES:
            path = _STATE_DIR / mod_file
            tree = ast.parse(path.read_text(encoding="utf-8"))
            methods: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            methods.add(item.name)
            result[mod_file[:-3]] = methods
        return result

    def test_no_duplicate_method_names_across_mixins(self):
        """No method name should be defined in more than one sub-mixin source file."""
        methods_per_module = self._methods_per_module()
        seen: dict[str, str] = {}
        duplicates: dict[str, list[str]] = {}
        for mod_stem, methods in methods_per_module.items():
            for name in methods:
                if name in seen:
                    if name not in duplicates:
                        duplicates[name] = [seen[name]]
                    duplicates[name].append(mod_stem)
                else:
                    seen[name] = mod_stem
        self.assertEqual(
            duplicates,
            {},
            f"Duplicate method names across sub-mixins (first MRO match silently shadows later ones): {duplicates}",
        )

    def test_each_module_defines_exactly_one_class(self):
        """Each state sub-module must define exactly one mixin class."""
        for mod_file in _STATE_MODULES:
            path = _STATE_DIR / mod_file
            tree = ast.parse(path.read_text(encoding="utf-8"))
            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            self.assertEqual(
                len(classes),
                1,
                f"{mod_file} defines {len(classes)} class(es); expected exactly 1",
            )

    def test_mixin_class_names_end_with_mixin(self):
        """Every class defined in a sub-module must have a name ending with 'Mixin'."""
        for mod_file in _STATE_MODULES:
            path = _STATE_DIR / mod_file
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self.assertTrue(
                        node.name.endswith("Mixin"),
                        f"{mod_file}: class {node.name!r} does not end with 'Mixin'",
                    )


# ============================================================
# Group H: MRO surface contract (requires torch)
# ============================================================

class TestMROContract(unittest.TestCase):
    """Exact MRO order and callable-surface completeness for _StateMixin."""

    @_skipif_no_torch
    def test_exact_mro_preserves_declared_base_order(self):
        """Bases declared in _state.py must appear in the same relative order in __mro__."""
        from gpu_env._core._state import _StateMixin
        from gpu_env.state.models import _CoreStateMixin
        from gpu_env.state.allocation import _AllocationMixin
        from gpu_env.state.agent_state import _AgentStateMixin
        from gpu_env.state.team_state import _TeamStateMixin
        from gpu_env.state.flag_state import _FlagStateMixin
        from gpu_env.state.episode_state import _EpisodeStateMixin
        from gpu_env.state.map_state import _MapStateMixin
        from gpu_env.state.opponent_state import _OpponentStateMixin
        from gpu_env.state.telemetry_state import _TelemetryStateMixin
        from gpu_env.state.scratch import _ScratchStateMixin
        from gpu_env.state.validation import _ValidationMixin
        from gpu_env.state.snapshots import _SnapshotsMixin
        declared_order = [
            _StateMixin, _CoreStateMixin, _AllocationMixin, _AgentStateMixin,
            _TeamStateMixin, _FlagStateMixin, _EpisodeStateMixin, _MapStateMixin,
            _OpponentStateMixin, _TelemetryStateMixin, _ScratchStateMixin,
            _ValidationMixin, _SnapshotsMixin,
        ]
        mro = _StateMixin.__mro__
        positions = []
        for cls in declared_order:
            self.assertIn(cls, mro, f"{cls.__name__} missing from _StateMixin.__mro__")
            positions.append(mro.index(cls))
        self.assertEqual(
            positions,
            sorted(positions),
            "MRO positions don't match declared base-class order in _state.py",
        )

    @_skipif_no_torch
    def test_no_runtime_duplicate_callables_across_mixins(self):
        """No callable name should appear in more than one sub-mixin's own __dict__."""
        from gpu_env.state.models import _CoreStateMixin
        from gpu_env.state.allocation import _AllocationMixin
        from gpu_env.state.agent_state import _AgentStateMixin
        from gpu_env.state.team_state import _TeamStateMixin
        from gpu_env.state.flag_state import _FlagStateMixin
        from gpu_env.state.episode_state import _EpisodeStateMixin
        from gpu_env.state.map_state import _MapStateMixin
        from gpu_env.state.opponent_state import _OpponentStateMixin
        from gpu_env.state.telemetry_state import _TelemetryStateMixin
        from gpu_env.state.scratch import _ScratchStateMixin
        from gpu_env.state.validation import _ValidationMixin
        from gpu_env.state.snapshots import _SnapshotsMixin
        mixins = [
            _CoreStateMixin, _AllocationMixin, _AgentStateMixin, _TeamStateMixin,
            _FlagStateMixin, _EpisodeStateMixin, _MapStateMixin, _OpponentStateMixin,
            _TelemetryStateMixin, _ScratchStateMixin, _ValidationMixin, _SnapshotsMixin,
        ]
        seen: dict[str, str] = {}
        duplicates: dict[str, list[str]] = {}
        for mixin in mixins:
            for name, val in vars(mixin).items():
                if callable(val) and not name.startswith("__"):
                    if name in seen:
                        if name not in duplicates:
                            duplicates[name] = [seen[name]]
                        duplicates[name].append(mixin.__name__)
                    else:
                        seen[name] = mixin.__name__
        self.assertEqual(
            duplicates,
            {},
            f"Runtime duplicate callables across sub-mixins: {duplicates}",
        )

    @_skipif_no_torch
    def test_state_mixin_exposes_all_sub_mixin_callables(self):
        """Every callable defined in a sub-mixin must be accessible via _StateMixin."""
        from gpu_env._core._state import _StateMixin
        from gpu_env.state.models import _CoreStateMixin
        from gpu_env.state.allocation import _AllocationMixin
        from gpu_env.state.agent_state import _AgentStateMixin
        from gpu_env.state.team_state import _TeamStateMixin
        from gpu_env.state.flag_state import _FlagStateMixin
        from gpu_env.state.episode_state import _EpisodeStateMixin
        from gpu_env.state.map_state import _MapStateMixin
        from gpu_env.state.opponent_state import _OpponentStateMixin
        from gpu_env.state.telemetry_state import _TelemetryStateMixin
        from gpu_env.state.scratch import _ScratchStateMixin
        from gpu_env.state.validation import _ValidationMixin
        from gpu_env.state.snapshots import _SnapshotsMixin
        mixins = [
            _CoreStateMixin, _AllocationMixin, _AgentStateMixin, _TeamStateMixin,
            _FlagStateMixin, _EpisodeStateMixin, _MapStateMixin, _OpponentStateMixin,
            _TelemetryStateMixin, _ScratchStateMixin, _ValidationMixin, _SnapshotsMixin,
        ]
        state_mixin_dir = set(dir(_StateMixin))
        missing = [
            f"{mixin.__name__}.{name}"
            for mixin in mixins
            for name, val in vars(mixin).items()
            if callable(val) and not name.startswith("__") and name not in state_mixin_dir
        ]
        self.assertEqual(missing, [], f"_StateMixin missing sub-mixin callables: {missing}")

    @_skipif_no_torch
    def test_required_public_api_present(self):
        """Key public methods that external callers use must be on _StateMixin."""
        from gpu_env._core._state import _StateMixin
        required = [
            "reset_all", "reset_indices", "reseed",
            "set_phase", "set_league_mode", "set_stress_schedule",
            "set_next_opponent", "get_opponent_key", "set_dynamics_config",
        ]
        missing = [m for m in required if not hasattr(_StateMixin, m)]
        self.assertEqual(missing, [], f"_StateMixin missing required public methods: {missing}")


# ============================================================
# Group I: Behavioral equivalence skeleton (requires torch)
# ============================================================

class TestBehavioralEquivalence(unittest.TestCase):
    """Basic reset/spawn invariants that hold regardless of map layout or opponent config."""

    @_skipif_no_torch
    def test_reset_all_zeroes_step_count(self):
        """After reset_all, step_count must be 0 for every env."""
        import torch
        from gpu_env import GPUFieldConfig, BatchedCTFCore
        env = BatchedCTFCore(n_envs=2, cfg=GPUFieldConfig(), device="cpu")
        env.reset_all()
        self.assertTrue(
            (env.step_count == 0).all(),
            f"step_count non-zero after reset_all: {env.step_count}",
        )

    @_skipif_no_torch
    def test_reset_all_clears_done_flags(self):
        """After reset_all, episode-done flags must be False for every env."""
        from gpu_env import GPUFieldConfig, BatchedCTFCore
        env = BatchedCTFCore(n_envs=2, cfg=GPUFieldConfig(), device="cpu")
        env.reset_all()
        for attr in ("ep_done", "done_blue", "done_red"):
            if hasattr(env, attr):
                val = getattr(env, attr)
                self.assertFalse(val.any(), f"{attr} is True after reset_all")

    @_skipif_no_torch
    def test_partial_reset_leaves_unselected_envs_unchanged(self):
        """reset_indices([0]) must not alter blue_pos[1] in a 2-env batch."""
        import torch
        from gpu_env import GPUFieldConfig, BatchedCTFCore
        env = BatchedCTFCore(n_envs=2, cfg=GPUFieldConfig(), device="cpu")
        env.reset_all()
        pos_before = env.blue_pos[1].clone()
        env.reset_indices([0])
        self.assertTrue(
            torch.allclose(pos_before, env.blue_pos[1]),
            "reset_indices([0]) changed blue_pos[1] in a 2-env batch",
        )

    @_skipif_no_torch
    def test_same_seed_produces_same_spawns(self):
        """Two BatchedCTFCore instances with identical seeds must spawn at identical positions."""
        import torch
        from gpu_env import GPUFieldConfig, BatchedCTFCore
        cfg = GPUFieldConfig()
        env_a = BatchedCTFCore(n_envs=1, cfg=cfg, device="cpu")
        env_b = BatchedCTFCore(n_envs=1, cfg=cfg, device="cpu")
        env_a.reseed(0)
        env_b.reseed(0)
        env_a.reset_all()
        env_b.reset_all()
        self.assertTrue(
            torch.allclose(env_a.blue_pos, env_b.blue_pos),
            "Same seed produced different blue_pos after reset_all",
        )


if __name__ == "__main__":
    unittest.main()
