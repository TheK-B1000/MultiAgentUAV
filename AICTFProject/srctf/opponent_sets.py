"""Injected opponent registries for the reversal scanner.

The scanner previously imported a hardcoded OPPONENTS tuple. This replaces that
with an injected canonical registry and changes NOTHING scientific. The
acceptance rule for the refactor is an equality:

    old scanner + historical roster  ==  generalized scanner + historical registry

Ordering is FROZEN as an explicit tuple, never derived from set iteration or
registry insertion order, because cell order defines the parallel merge order
and therefore the bytes of the merged states file.

The SRCTF registry is declared but GATED: it cannot be handed to the scanner
until the registry, dispatch-reachability, profile-hash and trajectory guards
have all passed. C6 produced two results that looked like findings from
opponents that resolved perfectly and never reached the simulator.
"""
from __future__ import annotations

from typing import Callable, Optional

from srctf import registry as _reg

# Frozen literal. run_g0_v2_seed.OPPONENTS is the historical source of truth and
# is asserted against this below, so silent drift in either is caught rather
# than quietly repartitioning every historical comparison.
HISTORICAL: tuple[str, ...] = ("OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12")

# Frozen order for SRCTF. Alphabetical would couple cell order to naming, so the
# order is stated once, explicitly, and never recomputed.
SRCTF: tuple[str, ...] = ("RAIDER", "FORTRESS", "ESCORT", "SPLIT", "INTERCEPTOR", "ADAPTIVE")

_SETS = {"historical": HISTORICAL, "srctf": SRCTF}


class OpponentSetError(RuntimeError):
    """Raised when an opponent set is unavailable or has drifted. Fails closed."""


def verify_historical_matches_source() -> None:
    """The frozen literal must equal the historical source of truth, in order."""
    from experiments.run_g0_v2_seed import OPPONENTS
    if tuple(OPPONENTS) != HISTORICAL:
        raise OpponentSetError(
            f"historical opponent set drifted: run_g0_v2_seed.OPPONENTS={tuple(OPPONENTS)} "
            f"but the frozen literal is {HISTORICAL}. Every historical comparison depends "
            f"on this order.")


def get(set_name: str, *, admit_guard: Optional[Callable[[], None]] = None) -> tuple[str, ...]:
    """Return a frozen, ordered opponent tuple.

    `historical` is checked against its source. `srctf` additionally requires the
    six-condition admission guard to have passed for every declared opponent.
    """
    if set_name not in _SETS:
        raise OpponentSetError(f"unknown opponent set {set_name!r}; expected one of "
                               f"{sorted(_SETS)}")
    if set_name == "historical":
        verify_historical_matches_source()
        return HISTORICAL

    # srctf -- gated
    declared = set(SRCTF)
    if declared != set(_reg.SRCTF_CANONICAL_OPPONENTS):
        raise OpponentSetError(
            f"SRCTF order tuple {SRCTF} disagrees with the canonical registry "
            f"{sorted(_reg.SRCTF_CANONICAL_OPPONENTS)}")
    missing = sorted(declared - set(_reg.registered()))
    if missing:
        raise OpponentSetError(
            f"SRCTF opponents not registered: {missing}. Content is deferred until C7 "
            f"answers; the scanner may not touch an SRCTF block before then.")
    if admit_guard is None:
        raise OpponentSetError(
            "the SRCTF set requires an admission guard. Registry, canonical identity, "
            "profile hash, dispatch reachability and trajectory identity must all pass "
            "before any SRCTF qualification datum is produced.")
    admit_guard()          # raises on any of the six conditions
    return SRCTF
