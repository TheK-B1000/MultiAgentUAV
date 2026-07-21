"""Unit tests for reward ablation presets and run-tag helpers."""

from __future__ import annotations

import os
import sys

import pytest

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from rl.train_ppo import (  # noqa: E402
    REWARD_ABLATION_PRESETS,
    _append_reward_ablation_to_run_tag,
    _default_run_tag_for_mode,
    _normalize_train_mode,
    normalize_reward_ablation,
    reward_overrides_for_ablation,
)
from rl.run_ablations import DEFAULT_ABLATIONS, build_command, build_run_tag, resolve_python, select_ablations  # noqa: E402
from game_field_gpu import GPUFieldConfig  # noqa: E402


def test_normalize_reward_aliases():
    assert normalize_reward_ablation("FULL") == "full"
    assert normalize_reward_ablation("sparse") == "no_shaping"
    assert normalize_reward_ablation("terminal-only") == "terminal"


def test_reward_overrides_presets():
    assert reward_overrides_for_ablation("full") == {}
    no_shape = reward_overrides_for_ablation("no_shaping")
    assert no_shape["pbrs_attack_coef"] == 0.0
    assert no_shape["team_escort_reward"] == 0.0
    assert "flag_pickup_reward" not in no_shape

    terminal = reward_overrides_for_ablation("terminal")
    assert terminal["flag_pickup_reward"] == 0.0
    assert terminal["sparse_weight"] == 0.0


def test_reward_overrides_apply_to_gpu_config():
    overrides = reward_overrides_for_ablation("no_shaping")
    cfg = GPUFieldConfig(n_envs=1, **overrides)
    assert cfg.pbrs_attack_coef == 0.0
    assert cfg.flag_pickup_reward != 0.0  # offense events kept


def test_unknown_reward_raises():
    with pytest.raises(ValueError):
        reward_overrides_for_ablation("not_a_real_preset")


def test_mode_aliases_include_no_curriculum():
    assert _normalize_train_mode("NO_CURRICULUM") == "FIXED_OPPONENT"
    assert _normalize_train_mode("PAPER") == "CURRICULUM_NO_LEAGUE"


def test_run_tag_includes_reward_ablation():
    tag = _default_run_tag_for_mode("LEAGUE", n_agents=2, reward_ablation="no_shaping")
    assert tag == "ppo_league_rew_no_shaping_2v2"
    assert _append_reward_ablation_to_run_tag("ppo_league_2v2", "full", 2) == "ppo_league_2v2"


def test_ablation_matrix_covers_reviewer_axes():
    names = {a.name for a in DEFAULT_ABLATIONS}
    assert names == {"ours", "no_league", "no_curriculum", "no_shaping"}
    by_name = {a.name: a for a in DEFAULT_ABLATIONS}
    assert by_name["no_league"].mode == "CURRICULUM_NO_LEAGUE"
    assert by_name["no_curriculum"].mode == "FIXED_OPPONENT"
    assert by_name["no_shaping"].reward_ablation == "no_shaping"


def test_build_command_and_select():
    specs = select_ablations("ours,no_shaping")
    assert [s.name for s in specs] == ["ours", "no_shaping"]
    cmd = build_command(
        specs[1],
        n_agents=2,
        total_steps=1000,
        seed=7,
        seed_count=1,
        device="cpu",
        checkpoint_dir=None,
        python_exe=sys.executable,
    )
    assert "--reward-ablation" in cmd
    assert "no_shaping" in cmd
    assert build_run_tag(specs[0], 2, 42, 1) == "ppo_ablate_ours_2v2"
    assert build_run_tag(specs[0], 2, 43, 2) == "ppo_ablate_ours_seed43_2v2"


def test_resolve_python_prefers_project_venv():
    py = resolve_python()
    assert os.path.isfile(py)
    # On this machine the project .venv should win over hermes/sys.executable.
    assert "AICTFProject" in py.replace("\\", "/") or os.path.abspath(py) == os.path.abspath(sys.executable)


def test_all_presets_are_valid_gpu_fields():
    field_names = set(GPUFieldConfig.__dataclass_fields__)
    for name, overrides in REWARD_ABLATION_PRESETS.items():
        unknown = set(overrides) - field_names
        assert not unknown, f"{name} has unknown fields: {unknown}"
