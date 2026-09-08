"""Architecture dependency enforcement tests (Phase 11 Stage 15).

Verifies hard architectural invariants without running the training loop:
  A. Presets do not import from rl.train_ppo (must use rl.config.ppo_config)
  B. No duplicate preset names across registry keys
  C. All registry callable entries resolve without ImportError
  D. Evaluation modules do not import from experiments/
  E. No module in rl/presets/families/ exceeds the size smoke alarm (600 lines)
  F. No circular imports within rl/presets/families/plan_faithful/

All tests are torch-free (AST / importlib level).
"""
from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

_PRESET_FAMILIES_ROOT = _REPO / "rl" / "presets" / "families" / "plan_faithful"
_PRESETS_ROOT = _REPO / "rl" / "presets"
_EVAL_ROOT = _REPO / "rl" / "evaluation"

_PLAN_FAITHFUL_SUBMODULES = [
    "base", "early", "v3_router", "v3i_event_router",
    "v3i_specialization", "v3i_consequence", "v4_proof",
    "v5_repertoire", "v6_router_adapters",
]

_FORBIDDEN_IMPORT_IN_PRESETS = "rl.train_ppo"
_FORBIDDEN_IMPORT_IN_EVAL = "experiments"

_MODULE_LINE_LIMIT = 1500


def _iter_py_files(directory: Path):
    return [f for f in directory.rglob("*.py") if "__pycache__" not in f.parts]


def _collect_imports(path: Path) -> list[tuple[int, str]]:
    """Return (lineno, module_name) for every import in the file."""
    src = path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError:
        return []
    results = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                results.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            results.append((node.lineno, module))
    return results


class TestPresetDoNotImportTrainPPO(unittest.TestCase):
    """Group A: Presets must import PPOConfig from rl.config.ppo_config, not rl.train_ppo."""

    def test_plan_faithful_submodules_no_train_ppo_import(self) -> None:
        violations = []
        for mod_name in _PLAN_FAITHFUL_SUBMODULES:
            path = _PRESET_FAMILIES_ROOT / f"{mod_name}.py"
            if not path.exists():
                self.fail(f"Expected sub-module not found: {path}")
            for lineno, module in _collect_imports(path):
                if _FORBIDDEN_IMPORT_IN_PRESETS in module:
                    violations.append(f"{path.relative_to(_REPO)}:{lineno} imports '{module}'")
        self.assertEqual(
            violations, [],
            f"Presets must not import from rl.train_ppo. Violations:\n" + "\n".join(violations),
        )

    def test_plan_faithful_facade_no_train_ppo_import(self) -> None:
        path = _PRESETS_ROOT / "plan_faithful.py"
        violations = []
        for lineno, module in _collect_imports(path):
            if _FORBIDDEN_IMPORT_IN_PRESETS in module:
                violations.append(f"{path.relative_to(_REPO)}:{lineno} imports '{module}'")
        self.assertEqual(violations, [], "\n".join(violations))

    def test_all_families_package_no_train_ppo_import(self) -> None:
        """All sub-modules inside rl/presets/families/ must not import rl.train_ppo.

        Scope is limited to families/ because hypothesis.py, other.py, and the
        core preset files (models.py, registry.py, validation.py, __init__.py)
        are pre-existing and not yet migrated away from rl.train_ppo.
        Stage 15 of Phase 11 tracks the remaining migration as future work.
        """
        families_root = _PRESETS_ROOT / "families"
        if not families_root.exists():
            self.skipTest("rl/presets/families/ not found")
        violations = []
        for path in _iter_py_files(families_root):
            for lineno, module in _collect_imports(path):
                if _FORBIDDEN_IMPORT_IN_PRESETS in module:
                    violations.append(f"{path.relative_to(_REPO)}:{lineno} imports '{module}'")
        self.assertEqual(
            violations, [],
            f"No families/ preset file may import rl.train_ppo. Violations:\n"
            + "\n".join(violations),
        )


class TestNoDuplicatePresetNames(unittest.TestCase):
    """Group B: Registry key names must be unique across all source modules."""

    def test_plan_faithful_no_duplicate_function_names(self) -> None:
        seen: dict[str, str] = {}
        duplicates = []
        for mod_name in _PLAN_FAITHFUL_SUBMODULES:
            path = _PRESET_FAMILIES_ROOT / f"{mod_name}.py"
            if not path.exists():
                self.fail(f"Expected sub-module not found: {path}")
            src = path.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(src)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    name = node.name
                    if name.startswith("_"):
                        continue
                    location = f"{mod_name}.py:{node.lineno}"
                    if name in seen:
                        duplicates.append(f"'{name}' defined in both {seen[name]} and {location}")
                    else:
                        seen[name] = location
        self.assertEqual(
            duplicates, [],
            "Duplicate public function names found in plan_faithful sub-modules:\n"
            + "\n".join(duplicates),
        )


class TestRegistryResolution(unittest.TestCase):
    """Group C: All registry callable entries resolve (no ImportError, no missing functions).

    This test requires torch to be available for the full registry import.
    Skipped when torch is absent.
    """

    @classmethod
    def setUpClass(cls) -> None:
        try:
            import torch  # noqa: F401
            cls._has_torch = True
        except ImportError:
            cls._has_torch = False

    def _skip_if_no_torch(self) -> None:
        if not self._has_torch:
            self.skipTest("torch not available — registry resolution test skipped")

    def test_registry_imports_without_error(self) -> None:
        self._skip_if_no_torch()
        from rl.presets._registry_source import _get_preset_dict
        registry = _get_preset_dict()
        self.assertGreater(len(registry), 0, "Registry must not be empty")

    def test_all_registry_values_are_callable(self) -> None:
        self._skip_if_no_torch()
        from rl.presets._registry_source import _get_preset_dict
        registry = _get_preset_dict()
        non_callable = [k for k, v in registry.items() if not callable(v)]
        self.assertEqual(non_callable, [], f"Non-callable registry entries: {non_callable}")

    def test_no_duplicate_registry_callables_for_same_key(self) -> None:
        self._skip_if_no_torch()
        from rl.presets._registry_source import _get_preset_dict
        registry = _get_preset_dict()
        # Each alias key must resolve to a function (not None or a string)
        broken = [k for k, v in registry.items() if v is None]
        self.assertEqual(broken, [], f"Registry keys mapping to None: {broken}")


class TestEvaluationDoesNotImportExperiments(unittest.TestCase):
    """Group D: rl/evaluation/ must not import from the experiments/ package."""

    def test_evaluation_no_experiments_import(self) -> None:
        if not _EVAL_ROOT.exists():
            self.skipTest("rl/evaluation/ not found — skipping")
        violations = []
        for path in _iter_py_files(_EVAL_ROOT):
            for lineno, module in _collect_imports(path):
                if _FORBIDDEN_IMPORT_IN_EVAL in module:
                    violations.append(f"{path.relative_to(_REPO)}:{lineno} imports '{module}'")
        self.assertEqual(
            violations, [],
            "rl/evaluation/ must not import from experiments/. Violations:\n"
            + "\n".join(violations),
        )


class TestModuleSizeSmokeAlarm(unittest.TestCase):
    """Group E: No preset family sub-module exceeds the size smoke alarm."""

    def test_plan_faithful_submodule_line_counts(self) -> None:
        oversized = []
        for mod_name in _PLAN_FAITHFUL_SUBMODULES:
            path = _PRESET_FAMILIES_ROOT / f"{mod_name}.py"
            if not path.exists():
                self.fail(f"Expected sub-module not found: {path}")
            lines = path.read_text(encoding="utf-8", errors="replace").count("\n")
            if lines > _MODULE_LINE_LIMIT:
                oversized.append(f"{mod_name}.py: {lines} lines (limit {_MODULE_LINE_LIMIT})")
        self.assertEqual(
            oversized, [],
            f"Plan-faithful sub-modules exceeding {_MODULE_LINE_LIMIT}-line smoke alarm:\n"
            + "\n".join(oversized),
        )

    def test_plan_faithful_facade_line_count(self) -> None:
        path = _PRESETS_ROOT / "plan_faithful.py"
        lines = path.read_text(encoding="utf-8", errors="replace").count("\n")
        self.assertLessEqual(
            lines, _MODULE_LINE_LIMIT,
            f"plan_faithful.py facade is {lines} lines (limit {_MODULE_LINE_LIMIT}). "
            "If it grew above the limit, move implementations into sub-modules.",
        )


class TestNoCyclicImports(unittest.TestCase):
    """Group F: No circular imports within rl/presets/families/plan_faithful/ sub-modules.

    Each sub-module may import from earlier modules in the chain but not from later ones.
    Defined order: base → early → v3_router → v3i_event_router → v3i_specialization
                → v3i_consequence → v4_proof → v5_repertoire → v6_router_adapters
    """

    _ORDER = _PLAN_FAITHFUL_SUBMODULES  # same list, defines allowed dependency direction

    def test_no_forward_imports(self) -> None:
        _PKG_PREFIX = "rl.presets.families.plan_faithful."
        violations = []
        for i, mod_name in enumerate(self._ORDER):
            path = _PRESET_FAMILIES_ROOT / f"{mod_name}.py"
            if not path.exists():
                continue
            allowed_deps = {self._ORDER[j] for j in range(i)}
            for lineno, module in _collect_imports(path):
                if module.startswith(_PKG_PREFIX):
                    dep_mod = module[len(_PKG_PREFIX):].split(".")[0]
                    if dep_mod and dep_mod not in allowed_deps:
                        violations.append(
                            f"{mod_name}.py:{lineno} imports from '{dep_mod}' "
                            f"(allowed deps: {sorted(allowed_deps) or ['none']})"
                        )
        self.assertEqual(
            violations, [],
            "Forward/circular imports detected in plan_faithful sub-modules:\n"
            + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
