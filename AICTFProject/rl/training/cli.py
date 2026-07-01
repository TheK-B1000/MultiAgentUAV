"""CLI parser and entry point for ``python rl/train_ppo.py``.

Backward-compatibility facade — all implementations live in the
:mod:`rl.training` sub-modules:

* :mod:`rl.training.arguments` — :func:`parse_train_args` (argparse builder)
* :mod:`rl.training.overrides`  — :func:`cfg_from_args` (namespace → PPOConfig)

This module re-exports both under their original names so that existing
``from rl.training.cli import parse_train_args`` import paths continue to
work without changes.

Reproducibility contract: CLI flag names, defaults, and the deprecation
warning text are part of the user-facing contract. Treat any rename as a
breaking change for existing run scripts (``experiments/*.ps1``).
"""

from __future__ import annotations

import sys
from typing import Optional

from rl.training.arguments import parse_train_args  # noqa: F401 — re-export
from rl.training.overrides import cfg_from_args  # noqa: F401 — re-export

__all__ = ["parse_train_args", "cfg_from_args", "main"]


def main(argv: Optional[list[str]] = None) -> None:
    """``python rl/train_ppo.py`` entry point.

    Honors the two read-only diagnostic flags first (``--verify-4v4`` and
    ``--test-vec-schema``) so they keep working without parsing the full
    training argparse surface; otherwise builds ``PPOConfig`` from CLI and
    runs :func:`rl.train_ppo.train_ppo`.

    ``argv`` follows the same convention as :func:`argparse.parse_args`: pass
    ``None`` (default) to inherit ``sys.argv``; pass an explicit list (without
    the program name) to drive the CLI programmatically.
    """
    flag_source = sys.argv[1:] if argv is None else list(argv)
    if "--verify-4v4" in flag_source:
        from rl.train_ppo import run_verify_4v4

        run_verify_4v4(num_episodes=10)
        return
    if "--test-vec-schema" in flag_source:
        from rl.train_ppo import run_test_vec_schema

        run_test_vec_schema()
        return

    from rl.train_ppo import train_ppo

    args = parse_train_args(argv)
    cfg = cfg_from_args(args)
    train_ppo(cfg)
