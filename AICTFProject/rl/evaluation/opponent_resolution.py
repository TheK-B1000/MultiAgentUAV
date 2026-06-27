"""Opponent selection and verification utilities.

Encapsulates: opponent name normalization, validation, environment-level
opponent setting, and pre-flight verification.  Never touches policy state.
"""
from __future__ import annotations

from typing import Any, Sequence


SUPPORTED_OPPONENTS: frozenset[str] = frozenset({"OP8", "OP9", "OP10"})


def normalize_opponent(value: Any) -> str:
    """Strip whitespace and the SCRIPTED: prefix, then upper-case."""
    opponent = str(value).strip().upper()
    if opponent.startswith("SCRIPTED:"):
        opponent = opponent.split(":", 1)[1]
    return opponent


def validate_opponent_name(opponent: str) -> str:
    """Return the canonical name or raise ValueError for unsupported opponents."""
    canonical = normalize_opponent(opponent)
    if canonical not in SUPPORTED_OPPONENTS:
        raise ValueError(
            f"Unsupported opponent {opponent!r}. "
            f"Expected one of {sorted(SUPPORTED_OPPONENTS)}."
        )
    return canonical


def _unwrap_env_method_result(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    import numpy as np
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0] if value.size else None
    return value


def get_opponent_key(env: Any) -> str:
    """Read the current opponent key from the environment."""
    errors: list[str] = []

    env_method = getattr(env, "env_method", None)
    if callable(env_method):
        try:
            result = env_method("get_opponent_key")
            result = _unwrap_env_method_result(result)
            if result is not None:
                return normalize_opponent(result)
        except Exception as exc:
            errors.append(f"env_method: {exc}")

    core_method = getattr(getattr(env, "core", None), "get_opponent_key", None)
    if callable(core_method):
        try:
            return normalize_opponent(core_method())
        except Exception as exc:
            errors.append(f"core: {exc}")

    raise RuntimeError(
        f"Could not read the environment opponent key. Errors: {errors}"
    )


def set_opponent(env: Any, opponent: str) -> str:
    """Set and verify the opponent, returning the resolved name."""
    requested = validate_opponent_name(opponent)
    errors: list[str] = []

    env_method = getattr(env, "env_method", None)
    if callable(env_method):
        try:
            env_method("set_next_opponent", "SCRIPTED", requested)
        except Exception as exc:
            errors.append(f"env_method: {exc}")
        else:
            resolved = get_opponent_key(env)
            if resolved != requested:
                raise RuntimeError(
                    f"Requested opponent {requested}, but environment "
                    f"reported {resolved}."
                )
            return resolved

    core_method = getattr(getattr(env, "core", None), "set_next_opponent", None)
    if callable(core_method):
        try:
            core_method("SCRIPTED", requested)
        except Exception as exc:
            errors.append(f"core: {exc}")
        else:
            resolved = get_opponent_key(env)
            if resolved != requested:
                raise RuntimeError(
                    f"Requested opponent {requested}, but environment "
                    f"reported {resolved}."
                )
            return resolved

    raise RuntimeError(f"Could not select opponent {requested}. Errors: {errors}")


def preflight_opponents(
    *,
    opponents: Sequence[str],
    make_env: Any,  # callable(seed) -> env
) -> None:
    """Verify each requested opponent can be set and survives a reset.

    ``make_env`` must accept a keyword ``seed`` and return a fresh environment
    that is automatically closed after verification.
    """
    print("[preflight] Validating scripted opponents...")

    for index, opponent in enumerate(opponents):
        requested = validate_opponent_name(opponent)
        env = make_env(seed=9100 + index)

        try:
            before_reset = set_opponent(env, requested)
            env.reset()
            after_reset = get_opponent_key(env)

            if before_reset != requested or after_reset != requested:
                raise RuntimeError(
                    f"Opponent verification failed for {requested}: "
                    f"before_reset={before_reset}, after_reset={after_reset}."
                )

            print(
                f"[preflight] requested={requested} "
                f"resolved={after_reset} PASS"
            )
        finally:
            env.close()
