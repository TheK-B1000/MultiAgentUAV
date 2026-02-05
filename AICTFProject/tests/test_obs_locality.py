"""
Step 5.2: Observation locality unit test.

Ensures build_continuous_features(agent) does not depend on global state that
is not observable by that agent. Two game states that are identical locally
for one agent but different globally (e.g. far-away red agent moved) must
produce the same vec for that agent.
"""
from __future__ import annotations

import sys
import os

# Allow importing from project root (AICTFProject or parent)
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


def test_vec_unchanged_when_global_only_changes() -> None:
    """
    Build two game states identical locally for one blue agent but different globally
    (move a red agent far away). Assert that blue agent's vec does not change.
    """
    gf = make_game_field(map_name=MAP_NAME or None, map_path=MAP_PATH or None)
    gf.reset_default()

    blue_agents = getattr(gf, "blue_agents", []) or []
    red_agents = getattr(gf, "red_agents", []) or []
    if not blue_agents or blue_agents[0] is None:
        raise AssertionError("No blue agent after reset_default")
    if not red_agents or red_agents[0] is None:
        raise AssertionError("No red agent after reset_default")

    blue = blue_agents[0]
    red = red_agents[0]
    cols = getattr(gf, "col_count", 20)
    rows = getattr(gf, "row_count", 20)

    vec1 = gf.build_continuous_features(blue)
    vec1 = np.asarray(vec1, dtype=np.float32).reshape(-1)

    # Change only global state: move red agent to a different (far) position.
    # Blue agent's local view (self, flags, time) is unchanged; vec must not change.
    old_x = int(getattr(red, "x", 0))
    old_y = int(getattr(red, "y", 0))
    new_x = (old_x + 10) % max(1, cols)
    new_y = (old_y + 5) % max(1, rows)
    if (new_x, new_y) == (old_x, old_y):
        new_x = min(cols - 1, old_x + 1)
        new_y = min(rows - 1, old_y + 1)
    red.x = new_x
    red.y = new_y
    # float_pos is read-only on Agent; build_continuous_features(blue) does not read red position

    vec2 = gf.build_continuous_features(blue)
    vec2 = np.asarray(vec2, dtype=np.float32).reshape(-1)

    np.testing.assert_allclose(
        vec1, vec2,
        atol=1e-5, rtol=1e-5,
        err_msg="vec changed when only global state (red agent position) changed — possible info leak",
    )


def test_vec_shape_and_finite() -> None:
    """Sanity: vec has expected shape and is finite."""
    gf = make_game_field(map_name=MAP_NAME or None, map_path=MAP_PATH or None)
    gf.reset_default()
    blue_agents = getattr(gf, "blue_agents", []) or []
    if not blue_agents or blue_agents[0] is None:
        raise AssertionError("No blue agent after reset_default")
    blue = blue_agents[0]
    vec = gf.build_continuous_features(blue)
    vec = np.asarray(vec, dtype=np.float32)
    assert vec.shape == (12,), f"expected shape (12,), got {vec.shape}"
    assert np.all(np.isfinite(vec)), "vec contained non-finite values"


if __name__ == "__main__":
    test_vec_shape_and_finite()
    test_vec_unchanged_when_global_only_changes()
    print("test_obs_locality: OK")
