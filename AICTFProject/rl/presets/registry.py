"""Authoritative preset registry.

``PresetRegistry`` provides:

* Typed registration of preset definitions.
* Canonical-name lookup with alias resolution.
* Typed errors for unknown names and duplicate registrations.
* A module-level ``REGISTRY`` singleton populated from the existing
  ``PRESET_REGISTRY`` dict at import time.

The singleton is read-only after module load.  No training code may
mutate the registry after startup.

Canonical-name convention
-------------------------
The canonical name for a function ``apply_X`` is ``X`` (strip the
``apply_`` prefix).  If ``X`` is not present in the legacy dict, the
function's ``__name__`` minus ``apply_`` is used as a fallback.  All
other dict keys that map to the same function become aliases.
"""
from __future__ import annotations

from typing import Callable, Iterator

from rl.presets.models import (
    DuplicatePresetAliasError,
    DuplicatePresetError,
    PresetDefinition,
    PresetIdentity,
    PresetNotFoundError,
    PresetStatus,
)


def _infer_family(name: str) -> str:
    """Derive a family label from a canonical preset name."""
    for prefix in (
        "latent_v6i9", "plan_faithful_latent_v6i9",
        "v6i9",
    ):
        if name.startswith(prefix):
            return "v6i9"
    for prefix in (
        "latent_v6i8", "plan_faithful_latent_v6i8",
        "v6i8",
    ):
        if name.startswith(prefix):
            return "v6i8"
    for prefix in (
        "latent_v6i7", "plan_faithful_latent_v6i7",
        "v6i7",
    ):
        if name.startswith(prefix):
            return "v6i7"
    for prefix in ("latent_v6i", "plan_faithful_latent_v6i", "v6i"):
        if name.startswith(prefix):
            return "v6"
    for prefix in ("latent_v5", "plan_faithful_latent_v5", "v5"):
        if name.startswith(prefix):
            return "v5"
    for prefix in ("latent_v4", "plan_faithful_latent_v4", "v4"):
        if name.startswith(prefix):
            return "v4"
    for prefix in ("latent_v3", "plan_faithful_latent_v3"):
        if name.startswith(prefix):
            return "v3"
    if name.startswith("hypothesis"):
        return "hypothesis"
    if "v6i9" in name:
        return "v6i9"
    return "plan_faithful"


def _infer_status(name: str) -> PresetStatus:
    """Assign a best-effort status from the canonical preset name."""
    active_families = frozenset({"v6i8", "v6i9"})
    if _infer_family(name) in active_families:
        return PresetStatus.ACTIVE
    if any(x in name for x in ("v6i7", "v6i6", "v6i5", "v6i4", "v6i3", "v6i2", "v6i1")):
        return PresetStatus.HISTORICAL_REPRODUCTION
    if any(x in name for x in ("v5", "v4i3", "v4i4")):
        return PresetStatus.HISTORICAL_REPRODUCTION
    if any(x in name for x in ("v3i", "v3b", "v3c", "v3d", "v3e", "v3f", "v3g", "v3h")):
        return PresetStatus.HISTORICAL_REPRODUCTION
    if name.startswith("hypothesis"):
        return PresetStatus.EXPERIMENTAL
    return PresetStatus.UNKNOWN


class PresetRegistry:
    """Authoritative, immutable-after-build registry of preset definitions.

    ``register()`` must be called before any ``resolve()`` / ``canonical_name()``
    call.  Once the module-level ``REGISTRY`` singleton is built, the registry
    should be treated as read-only.
    """

    def __init__(self) -> None:
        self._canonical: dict[str, PresetDefinition] = {}
        self._alias_map: dict[str, str] = {}  # any name → canonical name

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, definition: PresetDefinition) -> None:
        """Register a preset definition.

        Raises
        ------
        DuplicatePresetError
            If ``definition.identity.name`` is already registered.
        DuplicatePresetAliasError
            If any alias in ``definition.identity.aliases`` conflicts with an
            existing canonical name or alias.
        """
        name = definition.identity.name
        if name in self._canonical:
            raise DuplicatePresetError(
                f"Duplicate canonical preset name: {name!r}"
            )
        if name in self._alias_map and self._alias_map[name] != name:
            raise DuplicatePresetAliasError(
                f"Canonical name {name!r} conflicts with an existing alias"
            )

        for alias in definition.identity.aliases:
            if alias in self._canonical:
                raise DuplicatePresetAliasError(
                    f"Alias {alias!r} conflicts with existing canonical name"
                )
            if alias in self._alias_map and self._alias_map[alias] != name:
                raise DuplicatePresetAliasError(
                    f"Alias {alias!r} is already registered for a different preset"
                )

        self._canonical[name] = definition
        self._alias_map[name] = name
        for alias in definition.identity.aliases:
            self._alias_map[alias] = name

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def canonical_name(self, name: str) -> str:
        """Return the canonical name for ``name`` (which may be an alias).

        Raises
        ------
        PresetNotFoundError
            If ``name`` is not registered.
        """
        if name not in self._alias_map:
            raise PresetNotFoundError(
                f"Unknown preset {name!r}. "
                f"Use list_presets() to see available names."
            )
        return self._alias_map[name]

    def get_definition(self, name: str) -> PresetDefinition:
        """Return the ``PresetDefinition`` for ``name`` (may be an alias)."""
        return self._canonical[self.canonical_name(name)]

    def resolve(self, name: str) -> object:
        """Apply the preset to a fresh ``PPOConfig`` and return the result.

        Returns a fully resolved ``PPOConfig``; the caller should treat it
        as immutable (though the dataclass is mutable by design).
        """
        from rl.train_ppo import PPOConfig  # local import to avoid circular deps

        defn = self.get_definition(name)
        cfg = PPOConfig()
        defn.apply_fn(cfg)
        return cfg

    def list_presets(self) -> tuple[PresetIdentity, ...]:
        """Return all registered ``PresetIdentity`` objects, sorted by name."""
        return tuple(
            d.identity for d in sorted(self._canonical.values(), key=lambda d: d.identity.name)
        )

    def list_aliases(self, canonical: str) -> tuple[str, ...]:
        """Return all aliases for a canonical preset name."""
        defn = self.get_definition(canonical)
        return defn.identity.aliases

    # ------------------------------------------------------------------
    # Membership
    # ------------------------------------------------------------------

    def __contains__(self, name: str) -> bool:
        return name in self._alias_map

    def __len__(self) -> int:
        return len(self._canonical)

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._canonical.keys()))


# ---------------------------------------------------------------------------
# Build the module-level singleton from the existing PRESET_REGISTRY dict
# ---------------------------------------------------------------------------

def build_registry_from_dict(
    preset_dict: dict[str, Callable],
) -> PresetRegistry:
    """Construct a ``PresetRegistry`` from the legacy ``PRESET_REGISTRY`` dict.

    Canonical name assignment
    -------------------------
    For each unique function ``apply_X`` in the dict, the canonical name is
    ``X`` (``apply_`` prefix stripped).  If the computed canonical name is not
    present as a key in the dict, the first dict key that maps to the function
    is used as a fallback.

    All other dict keys mapping to the same function become aliases.
    """
    # Group dict keys by function identity (object id).
    fn_to_keys: dict[int, tuple[Callable, list[str]]] = {}
    for key, fn in preset_dict.items():
        fn_id = id(fn)
        if fn_id not in fn_to_keys:
            fn_to_keys[fn_id] = (fn, [])
        fn_to_keys[fn_id][1].append(key)

    registry = PresetRegistry()

    for fn, keys in fn_to_keys.values():
        # Derive candidate canonical name from the function name.
        fn_name = getattr(fn, "__name__", "")
        candidate = fn_name[len("apply_"):] if fn_name.startswith("apply_") else fn_name

        # Use the candidate if it exists in the dict; otherwise fall back to
        # the first registered key for this function.
        if candidate in keys:
            canonical = candidate
        else:
            canonical = keys[0]

        aliases = tuple(k for k in keys if k != canonical)

        definition = PresetDefinition(
            identity=PresetIdentity(
                name=canonical,
                family=_infer_family(canonical),
                version=1,
                description=getattr(fn, "__doc__", "").split("\n")[0].strip() if fn.__doc__ else "",
                aliases=aliases,
                predecessor=None,
                status=_infer_status(canonical),
            ),
            apply_fn=fn,
        )
        registry.register(definition)

    return registry


def _build_singleton() -> PresetRegistry:
    # Deferred import to avoid circular dependency: registry.py does not
    # import from rl.presets.__init__; it imports only from plan_faithful etc.
    from rl.presets._registry_source import _get_preset_dict
    return build_registry_from_dict(_get_preset_dict())


# The singleton is populated lazily on first access to avoid import-time
# side-effects from torch/cuda initialization in PPOConfig defaults.
_registry_instance: PresetRegistry | None = None


def get_registry() -> PresetRegistry:
    """Return the module-level ``PresetRegistry`` singleton (built on first call)."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = _build_singleton()
    return _registry_instance


__all__ = [
    "PresetRegistry",
    "build_registry_from_dict",
    "get_registry",
]
