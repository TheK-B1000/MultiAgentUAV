"""Shared tqdm wrapper for long experiment / eval loops.

Always iterate long seed/episode grids through :func:`tqdm_iter` so ETA is
visible when watching redirected logs. PPO training uses the SB3-style bar in
``rl.custom_ppo.trainer`` (with the same non-TTY / log-watcher contract).
"""
from __future__ import annotations

import sys
from typing import Any, Iterable, Optional, TypeVar

T = TypeVar("T")


def _stderr_is_interactive() -> bool:
    try:
        return bool(sys.stderr.isatty())
    except Exception:
        return False


def tqdm_iter(
    iterable: Iterable[T],
    *,
    desc: str,
    total: Optional[int] = None,
    unit: str = "ep",
    leave: bool = True,
) -> Any:
    """Return a tqdm-wrapped iterable (or bare iterable if tqdm missing).

    Interactive TTY: classic dynamic bar on stderr.
    Redirected stderr (``*.log.err``): ASCII bar + flushed writes so
    ``Get-Content -Wait`` / log tails always show progress.
    """
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

    interactive = _stderr_is_interactive()
    if interactive:
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

    try:
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    return tqdm(
        iterable,
        total=n,
        desc=desc,
        unit=unit,
        dynamic_ncols=False,
        ncols=100,
        ascii=True,
        file=sys.stderr,
        mininterval=1.0,
        leave=leave,
    )


def set_postfix(bar: Any, text: str) -> None:
    """Best-effort postfix update (no-op if not a tqdm bar)."""
    setter = getattr(bar, "set_postfix_str", None)
    if callable(setter):
        setter(text, refresh=False)
