"""Internal: hand the raw preset dict to ``registry.py``.

This module exists solely to break the circular import that would occur if
``registry.py`` imported ``rl.presets`` at module scope (``rl.presets``
imports ``registry``). The import below is deferred into the function body,
so it runs only once ``get_registry()`` is first called — by which point
``rl.presets`` has finished importing.

``PRESET_REGISTRY`` is returned directly rather than mirrored here: an
earlier hand-copied duplicate silently fell dozens of presets behind the
real mapping, so ``PresetRegistry`` could not resolve them.

Do NOT import this module from outside ``rl.presets``.
"""
from __future__ import annotations

from typing import Callable


def _get_preset_dict() -> dict[str, Callable]:
    """Return the authoritative preset mapping."""
    from rl.presets import PRESET_REGISTRY

    return dict(PRESET_REGISTRY)
