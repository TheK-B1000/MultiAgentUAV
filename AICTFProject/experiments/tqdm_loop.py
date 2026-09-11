"""Shared tqdm wrapper for long experiment / eval loops.

Always iterate long seed/episode grids through :func:`tqdm_iter` so ETA is
visible on stderr. PPO training already uses the SB3-style bar in
``rl.custom_ppo.trainer``.
"""
from __future__ import annotations

import sys
from typing import Any, Iterable, Optional, TypeVar

T = TypeVar("T")


def tqdm_iter(
    iterable: Iterable[T],
    *,
    desc: str,
    total: Optional[int] = None,
    unit: str = "ep",
    leave: bool = True,
) -> Any:
    """Return a tqdm-wrapped iterable on stderr (or bare iterable if tqdm missing)."""
    n = total
    if n is None:
        try:
            n = len(iterable)  # type: ignore[arg-type]
        except TypeError:
            n = None

    try:
        from tqdm import tqdm
    except ImportError:
        print(
            f"[PROGRESS] tqdm not installed; continuing without bar ({desc}). "
            "pip install tqdm",
            file=sys.stderr,
            flush=True,
        )
        return iterable

    return tqdm(
        iterable,
        total=n,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        file=sys.stderr,
        mininterval=0.25,
        leave=leave,
    )


def set_postfix(bar: Any, text: str) -> None:
    """Best-effort postfix update (no-op if not a tqdm bar)."""
    setter = getattr(bar, "set_postfix_str", None)
    if callable(setter):
        setter(text, refresh=False)
