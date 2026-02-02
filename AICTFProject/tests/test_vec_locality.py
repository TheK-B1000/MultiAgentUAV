"""
Vec locality: build_continuous_features(agent) must not depend on hidden global state.

Randomize hidden red state (positions, etc.); ensure blue agent's vec does not change
unless the agent could observe the change (e.g. flag moved into view).
"""
from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _root not in sys.path:
        sys.path.insert(0, _root)
    _parent = os.path.dirname(_root)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)

import numpy as np

from game_field import make_game_field, GameField
from config import MAP_NAME, MAP_PATH


def test_vec_unchanged_when_hidden_red_state_randomized() -> None:
    """
    Randomize hidden red state (red agent positions). Blue agent's vec must not change,
    since build_continuous_features uses only local state and flag positions, not red positions.
    """
    gf = make_game_field(map_name=MAP_NAME or None, map_path=MAP_PATH or None)
    gf.reset_default()

    blue_agents = getattr(gf, "blue_agents", []) or []
    red_agents = getattr(gf, "red_agents", []) or []
    assert blue_agents and blue_agents[0] is not None, "No blue agent after reset_default"
    assert red_agents and red_agents[0] is not None, "No red agent after reset_default"

    blue = blue_agents[0]
    cols = max(1, getattr(gf, "col_count", 20))
    rows = max(1, getattr(gf, "row_count", 20))

    vec0 = np.asarray(gf.build_continuous_features(blue), dtype=np.float32).reshape(-1)

    # Randomize hidden red state: move each red agent to a different cell
    rng = np.random.default_rng(42)
    for red in red_agents:
        if red is None:
            continue
        red.x = int(rng.integers(0, cols))
        red.y = int(rng.integers(0, rows))

    vec1 = np.asarray(gf.build_continuous_features(blue), dtype=np.float32).reshape(-1)

    np.testing.assert_allclose(
        vec0, vec1,
        atol=1e-5, rtol=1e-5,
        err_msg="vec changed when only hidden red state (red positions) changed — info leak",
    )


def test_vec_unchanged_multiple_red_perturbations() -> None:
    """Apply several red-position perturbations; vec must stay identical for blue."""
    gf = make_game_field(map_name=MAP_NAME or None, map_path=MAP_PATH or None)
    gf.reset_default()
    blue_agents = getattr(gf, "blue_agents", []) or []
    red_agents = getattr(gf, "red_agents", []) or []
    assert blue_agents and blue_agents[0] is not None
    blue = blue_agents[0]
    cols = max(1, getattr(gf, "col_count", 20))
    rows = max(1, getattr(gf, "row_count", 20))

    vec_ref = np.asarray(gf.build_continuous_features(blue), dtype=np.float32).reshape(-1)
    rng = np.random.default_rng(123)
    for _ in range(5):
        for red in red_agents:
            if red is None:
                continue
            red.x = int(rng.integers(0, cols))
            red.y = int(rng.integers(0, rows))
        vec = np.asarray(gf.build_continuous_features(blue), dtype=np.float32).reshape(-1)
        np.testing.assert_allclose(vec_ref, vec, atol=1e-5, rtol=1e-5)


def test_vec_shape_and_finite() -> None:
    """Sanity: vec shape (12,) and all finite."""
    gf = make_game_field(map_name=MAP_NAME or None, map_path=MAP_PATH or None)
    gf.reset_default()
    blue_agents = getattr(gf, "blue_agents", []) or []
    assert blue_agents and blue_agents[0] is not None
    vec = np.asarray(gf.build_continuous_features(blue_agents[0]), dtype=np.float32)
    assert vec.shape == (12,), f"expected (12,), got {vec.shape}"
    assert np.all(np.isfinite(vec)), "vec contained non-finite values"


if __name__ == "__main__":
    test_vec_shape_and_finite()
    test_vec_unchanged_when_hidden_red_state_randomized()
    test_vec_unchanged_multiple_red_perturbations()
    print("test_vec_locality: OK")
