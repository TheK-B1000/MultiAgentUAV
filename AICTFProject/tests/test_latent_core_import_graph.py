"""Import-level guard: latent training core must not depend on opponent-pool / scripted-opponent modules."""

from __future__ import annotations

import ast
import os
import unittest

# First segment of a forbidden root package (opponent-tagged training plumbing).
_FORBIDDEN_ROOTS = (
    "opponent_params",
    "opponent",
    "league",
    "curriculum",
    "species",
)


def _root_name(module: str | None) -> str | None:
    if not module:
        return None
    return module.split(".", 1)[0].strip()


def _visit_imports(path: str) -> set[str]:
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                r = _root_name(alias.name)
                if r:
                    roots.add(r)
        elif isinstance(node, ast.ImportFrom):
            r = _root_name(node.module)
            if r:
                roots.add(r)
    return roots


def _is_forbidden(r: str) -> bool:
    if r in _FORBIDDEN_ROOTS:
        return True
    for prefix in ("opponent", "league"):
        if r.startswith(prefix):
            return True
    return False


class LatentCoreImportGraphTests(unittest.TestCase):
    def test_rl_modules_for_latent_path_have_no_forbidden_imports(self) -> None:
        """`custom_ppo`, `latent_marl`, `ppo_core`, `networks` must not import opponent/league/skill stacks."""
        here = os.path.join(os.path.dirname(__file__), "..", "rl")
        for name in (
            "custom_ppo/inference.py",
            "custom_ppo/policy.py",
            "custom_ppo/csv_writers.py",
            "custom_ppo/latent_diagnostics.py",
            "custom_ppo/curriculum_runtime.py",
            "custom_ppo/return_normalization.py",
            "custom_ppo/trainer.py",
            "custom_ppo/__init__.py",
            "latent_marl.py",
            "ppo_core.py",
            "networks.py",
        ):
            path = os.path.join(here, name)
            roots = _visit_imports(path)
            bad = sorted(r for r in roots if _is_forbidden(r))
            self.assertEqual(
                bad,
                [],
                f"{name} must not import opponent-pool or scripted-curriculum roots; got {bad}",
            )


if __name__ == "__main__":
    unittest.main()
