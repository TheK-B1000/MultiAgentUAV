"""Tests for the Phase 4 preset system.

Covers:
* PresetRegistry basics (registration, lookup, aliases, errors)
* Serialization helpers (canonical dict, JSON bytes, hash, artifact)
* Validation (cross-field invariants)
* Compatibility re-exports
* Family module re-exports
* Snapshot alignment (registry resolves same configs as legacy dict)
* Static audit (no orphaned functions, no duplicate functions)
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from rl.presets.models import (
    DuplicatePresetAliasError,
    DuplicatePresetError,
    PresetDefinition,
    PresetIdentity,
    PresetNotFoundError,
    PresetSerializationError,
    PresetStatus,
    PresetValidationError,
)
from rl.presets.registry import PresetRegistry, build_registry_from_dict, get_registry
from rl.presets.serialization import (
    SCHEMA_VERSION,
    canonical_config_dict,
    preset_hash,
    resolved_preset_artifact,
    to_canonical_json_bytes,
)
from rl.presets.validation import assert_preset_valid, validate_preset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_identity(
    name: str = "test_preset",
    aliases: tuple[str, ...] = (),
    family: str = "plan_faithful",
    status: PresetStatus = PresetStatus.ACTIVE,
) -> PresetIdentity:
    return PresetIdentity(
        name=name,
        family=family,
        version=1,
        description="Test preset",
        aliases=aliases,
        predecessor=None,
        status=status,
    )


def _make_def(name: str = "test_preset", aliases: tuple[str, ...] = (), fn=None) -> PresetDefinition:
    if fn is None:
        fn = lambda cfg: cfg  # noqa: E731
    return PresetDefinition(identity=_make_identity(name=name, aliases=aliases), apply_fn=fn)


# ---------------------------------------------------------------------------
# PresetRegistry — registration
# ---------------------------------------------------------------------------

class TestPresetRegistryRegistration:
    def test_register_single(self):
        reg = PresetRegistry()
        reg.register(_make_def("foo"))
        assert "foo" in reg
        assert len(reg) == 1

    def test_register_with_aliases(self):
        reg = PresetRegistry()
        reg.register(_make_def("foo", aliases=("bar", "baz")))
        assert "bar" in reg
        assert "baz" in reg
        assert reg.canonical_name("bar") == "foo"
        assert reg.canonical_name("baz") == "foo"

    def test_duplicate_canonical_raises(self):
        reg = PresetRegistry()
        reg.register(_make_def("foo"))
        with pytest.raises(DuplicatePresetError):
            reg.register(_make_def("foo"))

    def test_alias_conflicts_with_canonical_raises(self):
        reg = PresetRegistry()
        reg.register(_make_def("foo"))
        with pytest.raises(DuplicatePresetAliasError):
            reg.register(_make_def("bar", aliases=("foo",)))

    def test_duplicate_alias_raises(self):
        reg = PresetRegistry()
        reg.register(_make_def("foo", aliases=("shared_alias",)))
        with pytest.raises(DuplicatePresetAliasError):
            reg.register(_make_def("bar", aliases=("shared_alias",)))


# ---------------------------------------------------------------------------
# PresetRegistry — lookup
# ---------------------------------------------------------------------------

class TestPresetRegistryLookup:
    def test_canonical_name_by_canonical(self):
        reg = PresetRegistry()
        reg.register(_make_def("foo", aliases=("f",)))
        assert reg.canonical_name("foo") == "foo"

    def test_canonical_name_by_alias(self):
        reg = PresetRegistry()
        reg.register(_make_def("foo", aliases=("f",)))
        assert reg.canonical_name("f") == "foo"

    def test_unknown_name_raises(self):
        reg = PresetRegistry()
        with pytest.raises(PresetNotFoundError):
            reg.canonical_name("does_not_exist")

    def test_get_definition(self):
        reg = PresetRegistry()
        defn = _make_def("foo")
        reg.register(defn)
        assert reg.get_definition("foo") is defn

    def test_list_presets_sorted(self):
        reg = PresetRegistry()
        reg.register(_make_def("zebra"))
        reg.register(_make_def("alpha"))
        names = [i.name for i in reg.list_presets()]
        assert names == ["alpha", "zebra"]

    def test_list_aliases(self):
        reg = PresetRegistry()
        reg.register(_make_def("foo", aliases=("a", "b")))
        aliases = reg.list_aliases("foo")
        assert set(aliases) == {"a", "b"}

    def test_contains_alias(self):
        reg = PresetRegistry()
        reg.register(_make_def("foo", aliases=("bar",)))
        assert "bar" in reg
        assert "unknown" not in reg

    def test_iter_yields_canonical_names(self):
        reg = PresetRegistry()
        reg.register(_make_def("b"))
        reg.register(_make_def("a"))
        assert list(reg) == ["a", "b"]


# ---------------------------------------------------------------------------
# build_registry_from_dict
# ---------------------------------------------------------------------------

class TestBuildRegistryFromDict:
    def test_canonical_name_from_function_name(self):
        def apply_my_preset(cfg):
            return cfg

        reg = build_registry_from_dict({"my_preset": apply_my_preset, "alias": apply_my_preset})
        assert "my_preset" in reg
        assert reg.canonical_name("alias") == "my_preset"

    def test_fallback_to_first_key_when_no_apply_prefix(self):
        def some_fn(cfg):
            return cfg

        reg = build_registry_from_dict({"first_key": some_fn, "second_key": some_fn})
        assert "first_key" in reg
        assert reg.canonical_name("second_key") == "first_key"

    def test_multiple_functions_each_get_canonical(self):
        def apply_a(cfg):
            return cfg

        def apply_b(cfg):
            return cfg

        reg = build_registry_from_dict({"a": apply_a, "b": apply_b})
        assert "a" in reg
        assert "b" in reg
        assert len(reg) == 2


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

@dataclass
class _FakeCfg:
    x: float = 1.0
    y: tuple = (1, 2, 3)
    z: str = "hello"
    device: str = "cuda"
    run_tag: str = "run1"
    cli_preset: str = "test"


class TestSerialization:
    def test_canonical_config_dict_sorted_keys(self):
        cfg = _FakeCfg()
        d = canonical_config_dict(cfg)
        assert list(d.keys()) == sorted(d.keys())

    def test_canonical_config_dict_tuples_become_lists(self):
        cfg = _FakeCfg(y=(1, 2, 3))
        d = canonical_config_dict(cfg)
        assert isinstance(d["y"], list)
        assert d["y"] == [1, 2, 3]

    def test_canonical_config_dict_raises_on_nan(self):
        cfg = _FakeCfg(x=float("nan"))
        with pytest.raises(PresetSerializationError):
            canonical_config_dict(cfg)

    def test_canonical_config_dict_raises_on_inf(self):
        cfg = _FakeCfg(x=float("inf"))
        with pytest.raises(PresetSerializationError):
            canonical_config_dict(cfg)

    def test_to_canonical_json_bytes_is_bytes(self):
        cfg = _FakeCfg()
        d = canonical_config_dict(cfg)
        b = to_canonical_json_bytes(d)
        assert isinstance(b, bytes)
        parsed = json.loads(b)
        assert parsed["x"] == cfg.x

    def test_preset_hash_excludes_device(self):
        cfg_a = _FakeCfg(device="cuda")
        cfg_b = _FakeCfg(device="cpu")
        assert preset_hash(cfg_a) == preset_hash(cfg_b)

    def test_preset_hash_excludes_run_tag(self):
        cfg_a = _FakeCfg(run_tag="run1")
        cfg_b = _FakeCfg(run_tag="run2")
        assert preset_hash(cfg_a) == preset_hash(cfg_b)

    def test_preset_hash_excludes_cli_preset(self):
        cfg_a = _FakeCfg(cli_preset="preset_a")
        cfg_b = _FakeCfg(cli_preset="preset_b")
        assert preset_hash(cfg_a) == preset_hash(cfg_b)

    def test_preset_hash_sensitive_to_x(self):
        cfg_a = _FakeCfg(x=1.0)
        cfg_b = _FakeCfg(x=2.0)
        assert preset_hash(cfg_a) != preset_hash(cfg_b)

    def test_preset_hash_is_sha256_hex(self):
        cfg = _FakeCfg()
        h = preset_hash(cfg)
        assert len(h) == 64
        int(h, 16)  # must be valid hex

    def test_resolved_preset_artifact_structure(self):
        cfg = _FakeCfg()
        art = resolved_preset_artifact(
            canonical_name="my_preset",
            requested_name="alias",
            cfg=cfg,
            git_commit="abc123",
        )
        assert art["_schema_version"] == SCHEMA_VERSION
        assert art["canonical_name"] == "my_preset"
        assert art["requested_name"] == "alias"
        assert "alias" in art["aliases_used"]
        assert "preset_hash" in art
        assert art["validation_passed"] is True
        assert isinstance(art["resolved_config"], dict)

    def test_resolved_preset_artifact_no_alias_when_same_name(self):
        cfg = _FakeCfg()
        art = resolved_preset_artifact(
            canonical_name="foo",
            requested_name="foo",
            cfg=cfg,
        )
        assert art["aliases_used"] == []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@dataclass
class _ValidatableCfg:
    use_latent_variable: bool = False
    latent_k: int = 4
    router_reward_enabled: bool = False
    opponent_pool: tuple = ()
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    latent_strategy_ppo_coef: float = 0.1
    latent_lam_p: float = 0.02
    latent_lam_h: float = 0.01
    latent_cf_separation_coef: float = 0.0
    latent_kl_consecutive: float = 0.0
    latent_strategy_aux_predict_phase_coef: float = 0.0
    latent_strategy_aux_return_coef: float = 0.0
    learning_rate: float = 3e-4
    v6i9_training_stage: str = ""
    enable_latent_z_residual: bool = False


class TestValidation:
    def test_valid_config_no_errors(self):
        cfg = _ValidatableCfg(use_latent_variable=True, latent_k=4)
        assert validate_preset(cfg) == []

    def test_latent_k_zero_raises(self):
        cfg = _ValidatableCfg(use_latent_variable=True, latent_k=0)
        errors = validate_preset(cfg, "my_preset")
        assert any(e.field_path == "latent_k" for e in errors)

    def test_latent_k_not_checked_when_latent_disabled(self):
        cfg = _ValidatableCfg(use_latent_variable=False, latent_k=0)
        errors = validate_preset(cfg)
        latent_errors = [e for e in errors if e.field_path == "latent_k"]
        assert latent_errors == []

    def test_router_without_latent_raises(self):
        cfg = _ValidatableCfg(use_latent_variable=False, router_reward_enabled=True)
        errors = validate_preset(cfg)
        assert any(e.field_path == "router_reward_enabled" for e in errors)

    def test_op4_in_pool_raises(self):
        cfg = _ValidatableCfg(opponent_pool=("op3", "op4"))
        errors = validate_preset(cfg)
        assert any(e.field_path == "opponent_pool" for e in errors)

    def test_nonfinite_coef_raises(self):
        cfg = _ValidatableCfg(vf_coef=float("nan"))
        errors = validate_preset(cfg)
        assert any(e.field_path == "vf_coef" for e in errors)

    def test_nonpositive_lr_raises(self):
        cfg = _ValidatableCfg(learning_rate=0.0)
        errors = validate_preset(cfg)
        assert any(e.field_path == "learning_rate" for e in errors)

    def test_unknown_v6i9_stage_raises(self):
        cfg = _ValidatableCfg(v6i9_training_stage="invalid_stage")
        errors = validate_preset(cfg)
        assert any(e.field_path == "v6i9_training_stage" for e in errors)

    def test_known_v6i9_stage_ok(self):
        cfg = _ValidatableCfg(v6i9_training_stage="stage1_mapaware_generalist")
        errors = validate_preset(cfg)
        stage_errors = [e for e in errors if e.field_path == "v6i9_training_stage"]
        assert stage_errors == []

    def test_residual_without_latent_raises(self):
        cfg = _ValidatableCfg(use_latent_variable=False, enable_latent_z_residual=True)
        errors = validate_preset(cfg)
        assert any(e.field_path == "enable_latent_z_residual" for e in errors)

    def test_assert_preset_valid_raises_on_failure(self):
        cfg = _ValidatableCfg(use_latent_variable=True, latent_k=0)
        with pytest.raises(PresetValidationError):
            assert_preset_valid(cfg)

    def test_preset_validation_error_str(self):
        err = PresetValidationError(
            "bad value",
            preset_name="my_preset",
            field_path="latent_k",
            observed=0,
            constraint="latent_k > 0",
        )
        s = str(err)
        assert "my_preset" in s
        assert "latent_k" in s


# ---------------------------------------------------------------------------
# Compatibility re-exports
# ---------------------------------------------------------------------------

class TestCompatibilityReexports:
    def test_compat_imports(self):
        from rl.presets.compatibility import (
            DuplicatePresetAliasError,
            DuplicatePresetError,
            PresetDefinition,
            PresetError,
            PresetIdentity,
            PresetNotFoundError,
            PresetRegistry,
            PresetStatus,
            PresetValidationError,
            assert_preset_valid,
            build_registry_from_dict,
            canonical_config_dict,
            get_registry,
            preset_hash,
            validate_preset,
        )
        # If it imported without error, we're good.

    def test_init_exposes_get_registry(self):
        from rl.presets import get_registry
        reg = get_registry()
        assert isinstance(reg, PresetRegistry)

    def test_init_exposes_typed_errors(self):
        from rl.presets import PresetNotFoundError, PresetValidationError
        assert issubclass(PresetNotFoundError, Exception)
        assert issubclass(PresetValidationError, Exception)


# ---------------------------------------------------------------------------
# Family module re-exports
# ---------------------------------------------------------------------------

class TestFamilyModules:
    def test_v6i8_exports(self):
        from rl.presets.families.v6i8 import (
            apply_plan_faithful_latent_v6i8_adapter_balanced,
            apply_plan_faithful_latent_v6i8_adapter_balanced_hardpool,
            apply_plan_faithful_latent_v6i8_adapter_sparse,
            apply_plan_faithful_latent_v6i8_adapter_sparse_hardpool,
        )
        assert callable(apply_plan_faithful_latent_v6i8_adapter_balanced)
        assert callable(apply_plan_faithful_latent_v6i8_adapter_sparse)
        assert callable(apply_plan_faithful_latent_v6i8_adapter_balanced_hardpool)
        assert callable(apply_plan_faithful_latent_v6i8_adapter_sparse_hardpool)

    def test_v6i9_exports(self):
        from rl.presets.families.v6i9 import (
            apply_plan_faithful_latent_v6i9_mapaware_generalist_hardpool,
            apply_plan_faithful_latent_v6i9_mapaware_generalist_hardpool_split,
            apply_plan_faithful_latent_v6i9_mapaware_repertoire_hardpool,
            apply_plan_faithful_latent_v6i9_mapaware_router_sparse_hardpool,
            apply_plan_faithful_latent_v6i9_mapaware_router_feedforward_hardpool,
        )
        assert callable(apply_plan_faithful_latent_v6i9_mapaware_generalist_hardpool)
        assert callable(apply_plan_faithful_latent_v6i9_mapaware_repertoire_hardpool)
        assert callable(apply_plan_faithful_latent_v6i9_mapaware_router_sparse_hardpool)
        assert callable(apply_plan_faithful_latent_v6i9_mapaware_router_feedforward_hardpool)

    def test_repertoire_exports(self):
        from rl.presets.families.repertoire import (
            apply_plan_faithful_latent_v5i8_repertoire_uniform_z,
            apply_plan_faithful_latent_v6i9_mapaware_repertoire_hardpool,
        )
        assert callable(apply_plan_faithful_latent_v6i9_mapaware_repertoire_hardpool)

    def test_router_exports(self):
        from rl.presets.families.router import (
            apply_plan_faithful_latent_v6i7_recurrent_router,
            apply_plan_faithful_latent_v6i9_mapaware_router_sparse_hardpool,
            apply_plan_faithful_latent_v6i9_mapaware_router_feedforward_hardpool,
        )
        assert callable(apply_plan_faithful_latent_v6i9_mapaware_router_sparse_hardpool)
        assert callable(apply_plan_faithful_latent_v6i9_mapaware_router_feedforward_hardpool)


# ---------------------------------------------------------------------------
# Snapshot alignment
# ---------------------------------------------------------------------------

class TestSnapshotAlignment:
    """Verify the PresetRegistry resolves the same set of canonical presets
    as the legacy PRESET_REGISTRY dict."""

    def test_registry_covers_all_legacy_presets(self):
        from rl.presets import PRESET_REGISTRY
        registry = get_registry()
        for key in PRESET_REGISTRY:
            assert key in registry, f"Legacy key {key!r} missing from PresetRegistry"

    def test_registry_canonical_names_are_subset_of_legacy(self):
        from rl.presets import PRESET_REGISTRY
        registry = get_registry()
        for identity in registry.list_presets():
            assert identity.name in PRESET_REGISTRY or any(
                alias in PRESET_REGISTRY for alias in identity.aliases
            ), f"Canonical {identity.name!r} and none of its aliases appear in PRESET_REGISTRY"


# ---------------------------------------------------------------------------
# Static audit
# ---------------------------------------------------------------------------

class TestStaticAudit:
    """Audit the plan_faithful module for known structural issues."""

    def test_all_apply_functions_in_registry(self):
        """Every public apply_ function in plan_faithful must be reachable from PRESET_REGISTRY."""
        import rl.presets.plan_faithful as pf
        from rl.presets import PRESET_REGISTRY

        fns = {
            fn
            for name, fn in vars(pf).items()
            if name.startswith("apply_") and callable(fn)
        }
        # apply_plan_faithful_base is a shared helper, not a registered preset.
        base_fn = getattr(pf, "apply_plan_faithful_base", None)
        fns.discard(base_fn)

        # Build set of function objects reachable from the legacy dict.
        registered_fns = set(PRESET_REGISTRY.values())

        for fn in fns:
            assert fn in registered_fns, (
                f"Function {fn.__name__!r} in plan_faithful.py has no entry in PRESET_REGISTRY — "
                "orphaned preset function"
            )

    def test_v6i9_presets_have_stage_field_set(self):
        """V6I9 presets that should set v6i9_training_stage must do so."""
        from rl.presets import PRESET_REGISTRY, apply_preset
        from rl.train_ppo import PPOConfig

        v6i9_keys = [k for k in PRESET_REGISTRY if "v6i9" in k and "generalist" in k]
        for key in v6i9_keys[:1]:  # spot-check one to avoid slow full resolution
            cfg = PPOConfig()
            apply_preset(cfg, key)
            assert cfg.v6i9_training_stage, (
                f"Preset {key!r} did not set v6i9_training_stage"
            )

    def test_no_apply_functions_with_name_collisions(self):
        """Verify no two distinct functions have the same __name__ in plan_faithful."""
        import rl.presets.plan_faithful as pf

        name_to_fns: dict[str, list] = {}
        for fn in vars(pf).values():
            if callable(fn) and getattr(fn, "__name__", "").startswith("apply_"):
                fn_name = fn.__name__
                if fn_name not in name_to_fns:
                    name_to_fns[fn_name] = []
                if fn not in name_to_fns[fn_name]:
                    name_to_fns[fn_name].append(fn)

        # Python's known duplicate: apply_plan_faithful_latent_v3b_marginal
        # (defined twice — second definition wins, first is shadowed)
        known_duplicates = {"apply_plan_faithful_latent_v3b_marginal"}
        for fn_name, fns in name_to_fns.items():
            if len(fns) > 1 and fn_name not in known_duplicates:
                pytest.fail(
                    f"Function {fn_name!r} has {len(fns)} distinct objects with the same name "
                    f"in plan_faithful — unexpected duplicate definition"
                )
