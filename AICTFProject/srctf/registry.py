"""SRCTF canonical opponent registry and the six-condition admission guard.

Names are frozen by SRCTF_V1_ERRATUM_01.json: opponents are named for WHAT THEY
DO, with no OP prefix, no numbering, and no alias or synonym layer. That is a
correctness fix rather than a style choice -- in the historical roster
OP9_FORTRESS canonicalizes to OP9_SPLIT_LANE_FEINT and OP6_TURTLE to
OP6_IMMEDIATE_DUAL_RUSH, so reading a name told you the opposite of the
behaviour.

The admission guard implements the conjunction frozen in ERRATUM_02. All six
must hold; C6 proved the first four can pass while an opponent never reaches the
simulator at all:

    1. name in SRCTF_CANONICAL_OPPONENTS      explicit registry membership
    2. canonicalize(name) == name             canonical identity
    3. resolve(name).profile_name == name     the profile agrees
    4. profile_hash == frozen expected        the knobs are the frozen knobs
    5. dispatch(name) reaches the BT brain    it is actually driven
    6. trajectory_hash == frozen expected     it actually behaves as recorded

CONTENT IS DEFERRED. The names and the guard are frozen and C7-independent;
the BTProfile knob values are not, because C7's result determines how strong the
opportunity-cost mechanics need to be. register() is what fills that in later.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Callable, Optional

# Frozen by SRCTF_V1_ERRATUM_01.json. TURTLE is deliberately absent: it is a
# reserved historical confusable (OP6_TURTLE already denotes a dual rush).
SRCTF_CANONICAL_OPPONENTS: frozenset[str] = frozenset({
    "RAIDER",       # commits strongly forward, exposes home
    "FORTRESS",     # strong home allocation, weak offensive throughput
    "ESCORT",       # protects possession, advances deliberately
    "SPLIT",        # attacks multiple lanes
    "INTERCEPTOR",  # hunts carriers away from home
    "ADAPTIVE",     # changes allocation according to score and game state
})

MECHANICS = {
    "RAIDER": "commits strongly forward, exposes home",
    "FORTRESS": "strong home allocation, weak offensive throughput",
    "ESCORT": "protects possession, advances deliberately",
    "SPLIT": "attacks multiple lanes",
    "INTERCEPTOR": "hunts carriers away from home",
    "ADAPTIVE": "changes allocation according to score and game state",
}


class SRCTFRegistryError(RuntimeError):
    """Raised on any admission-guard violation. The guard fails closed."""


@dataclasses.dataclass(frozen=True)
class SRCTFOpponent:
    """A registered SRCTF opponent. Hashes are frozen at registration."""

    name: str
    profile: object                    # BTProfile, supplied when content lands
    expected_profile_sha256: str
    expected_trajectory_sha256: str
    mechanics: str

    def __post_init__(self) -> None:
        if self.name not in SRCTF_CANONICAL_OPPONENTS:
            raise SRCTFRegistryError(
                f"{self.name!r} is not in SRCTF_CANONICAL_OPPONENTS. The registry is "
                f"explicit precisely so a typo cannot canonicalize to itself and look valid."
            )


_REGISTRY: dict[str, SRCTFOpponent] = {}


def profile_sha256(profile: object) -> str:
    """Stable hash of a profile's fields, independent of dataclass identity."""
    d = dataclasses.asdict(profile) if dataclasses.is_dataclass(profile) else dict(profile)
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()


def register(name: str, profile: object, *, expected_profile_sha256: str,
             expected_trajectory_sha256: str) -> SRCTFOpponent:
    """Register one SRCTF opponent. Content step -- deferred until after C7."""
    if name in _REGISTRY:
        raise SRCTFRegistryError(f"{name!r} already registered; re-registration is not allowed")
    got = profile_sha256(profile)
    if got != expected_profile_sha256:
        raise SRCTFRegistryError(
            f"{name}: profile hash {got[:16]} != frozen {expected_profile_sha256[:16]}")
    entry = SRCTFOpponent(name=name, profile=profile,
                          expected_profile_sha256=expected_profile_sha256,
                          expected_trajectory_sha256=expected_trajectory_sha256,
                          mechanics=MECHANICS[name])
    _REGISTRY[name] = entry
    return entry


def registered() -> dict[str, SRCTFOpponent]:
    return dict(_REGISTRY)


def _clear_for_tests() -> None:
    _REGISTRY.clear()


def admit(
    name: str,
    *,
    canonicalize: Callable[[str], str],
    resolve_profile: Callable[[str], object],
    dispatch_level: Callable[[str], Optional[int]],
    trajectory_hash: Callable[[str], str],
    synonym_tables: tuple[dict, ...] = (),
) -> None:
    """Run the frozen six-condition conjunction. Raises on any failure.

    Every check is separate and named so a failure says which property broke.
    """
    # 1. explicit registry membership
    if name not in SRCTF_CANONICAL_OPPONENTS:
        raise SRCTFRegistryError(f"{name!r}: not in SRCTF_CANONICAL_OPPONENTS")
    entry = _REGISTRY.get(name)
    if entry is None:
        raise SRCTFRegistryError(f"{name!r}: declared but not registered (no profile bound)")

    # 2. canonical identity -- necessary but NOT sufficient on its own, since a
    #    canonicalizer may return unknown strings unchanged.
    if canonicalize(name) != name:
        raise SRCTFRegistryError(
            f"{name!r}: canonicalizes to {canonicalize(name)!r}; SRCTF names are canonical-only")

    # no alias/synonym layer may reference an SRCTF name
    for table in synonym_tables:
        for k, v in table.items():
            if k == name or v == name:
                raise SRCTFRegistryError(
                    f"{name!r}: appears in an alias/synonym table ({k!r} -> {v!r}). "
                    f"The historical alias layer is exactly what made names lie.")

    # 3/4. the profile agrees, and its knobs are the frozen knobs
    prof = resolve_profile(name)
    prof_name = getattr(prof, "name", None)
    if prof_name != name:
        raise SRCTFRegistryError(f"{name!r}: resolved profile is named {prof_name!r}")
    got = profile_sha256(prof)
    if got != entry.expected_profile_sha256:
        raise SRCTFRegistryError(
            f"{name}: profile hash {got[:16]} != frozen {entry.expected_profile_sha256[:16]}")

    # 5. dispatch reachability -- config correctness is not reachability
    if dispatch_level(name) is None:
        raise SRCTFRegistryError(
            f"{name!r}: resolves correctly but does NOT reach the BT brain. C6 produced two "
            f"results that looked like findings from exactly this state.")

    # 6. behavioural identity
    got_traj = trajectory_hash(name)
    if got_traj != entry.expected_trajectory_sha256:
        raise SRCTFRegistryError(
            f"{name}: trajectory hash {got_traj[:16]} != frozen "
            f"{entry.expected_trajectory_sha256[:16]}; it does not behave as recorded")


def admit_all(**kw) -> None:
    """Admit every declared SRCTF opponent, or raise. Fails closed on a partial set."""
    missing = sorted(SRCTF_CANONICAL_OPPONENTS - set(_REGISTRY))
    if missing:
        raise SRCTFRegistryError(f"not registered: {missing}")
    for name in sorted(SRCTF_CANONICAL_OPPONENTS):
        admit(name, **kw)
