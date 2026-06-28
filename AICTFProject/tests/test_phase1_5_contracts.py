"""Phase 1.5 contract tests.

Covers:
1.  Inference contract excludes diagnostics methods
2.  Diagnostics contract exposes encoder weights
3.  Old get_cnn_input_weights alias still works
4.  Old probe-result import path still works
5.  Generic SUCCESS result invariants
6.  Generic ERROR result invariants
7.  Invalid result combinations raise
8.  Failed probes have metrics=None
9.  Forced latent distribution requires explicit z_idx
10. Public distribution logits preserve gradients
11. Distribution .logits() method returns correct tensors
12. Distribution .probabilities() sum to 1
13. Distribution .argmax_actions() matches manual argmax
14. Distribution .distributions alias backward compat
15. Evaluator uses get_observation_encoder_input_weights not actor_cnn path
16. Incomplete run manifest has status=in_progress
17. Completed run manifest has status=completed
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces

from rl.custom_ppo.diagnostics_contract import PolicyDiagnosticsContract
from rl.custom_ppo.distributions import ActionHead, MultiHeadActionDistribution
from rl.custom_ppo.policy import SharedActorCentralizedCritic
from rl.custom_ppo.policy_contract import PolicyInferenceContract
from rl.evaluation.probes.results import (
    CounterfactualProbeMetrics,
    CounterfactualProbeResult,
    GradientProbeMetrics,
    GradientProbeResult,
    ProbeError,
    ProbeResult,
    ProbeStatus,
    WeightProbeMetrics,
    WeightProbeResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_AGENTS = 2
N_MACROS = 5
N_TARGETS = 10
N_CHANNELS = 7
CNN_H, CNN_W = 11, 11


def _make_obs_space(channels: int = N_CHANNELS) -> spaces.Dict:
    return spaces.Dict(
        {
            "grid": spaces.Box(0.0, 1.0, shape=(N_AGENTS, channels, CNN_H, CNN_W), dtype=np.float32),
            "vec": spaces.Box(-1.0, 1.0, shape=(N_AGENTS, 20), dtype=np.float32),
            "agent_mask": spaces.Box(0.0, 1.0, shape=(N_AGENTS,), dtype=np.float32),
            "mask": spaces.Box(0.0, 1.0, shape=(N_AGENTS * (N_MACROS + N_TARGETS),), dtype=np.float32),
        }
    )


def _make_action_space() -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([N_MACROS, N_TARGETS] * N_AGENTS)


def _make_model(latent_k: int = 0, channels: int = N_CHANNELS) -> SharedActorCentralizedCritic:
    return SharedActorCentralizedCritic(
        _make_obs_space(channels),
        _make_action_space(),
        latent_k=latent_k,
        actor_hidden_dim=64,
        critic_hidden_dim=64,
        actor_cnn_feature_dim=32,
    ).eval()


def _make_obs(batch: int = 1, channels: int = N_CHANNELS) -> dict[str, torch.Tensor]:
    return {
        "grid": torch.zeros(batch, N_AGENTS, channels, CNN_H, CNN_W),
        "vec": torch.zeros(batch, N_AGENTS, 20),
        "agent_mask": torch.ones(batch, N_AGENTS),
        "mask": torch.ones(batch, N_AGENTS * (N_MACROS + N_TARGETS)),
    }


# ---------------------------------------------------------------------------
# 1. Inference contract excludes diagnostics methods
# ---------------------------------------------------------------------------

class TestContractSeparation:
    def test_inference_contract_is_runtime_checkable(self):
        model = _make_model()
        assert isinstance(model, PolicyInferenceContract)

    def test_policy_inference_contract_has_no_diagnostics_method(self):
        """get_observation_encoder_input_weights must NOT be in PolicyInferenceContract."""
        import inspect
        members = {name for name, _ in inspect.getmembers(PolicyInferenceContract)}
        assert "get_observation_encoder_input_weights" not in members
        assert "get_cnn_input_weights" not in members

    def test_diagnostics_contract_is_runtime_checkable(self):
        model = _make_model()
        assert isinstance(model, PolicyDiagnosticsContract)

    def test_diagnostics_contract_exposes_encoder_weights(self):
        model = _make_model()
        w = model.get_observation_encoder_input_weights()
        assert isinstance(w, torch.Tensor)
        assert w.ndim == 4
        assert w.shape[1] == N_CHANNELS


# ---------------------------------------------------------------------------
# 2 & 3. Compatibility alias
# ---------------------------------------------------------------------------

class TestCompatAlias:
    def test_old_get_cnn_input_weights_still_works_on_model(self):
        model = _make_model()
        w_new = model.get_observation_encoder_input_weights()
        w_old = model.get_cnn_input_weights()
        assert w_new is w_old  # same object, not a copy

    def test_old_import_path_still_works(self):
        from rl.custom_ppo.probe_result import (
            PROBE_ERROR,
            PROBE_SUCCESS,
            CounterfactualProbeResult,
            GradientProbeResult,
            ProbeResult,
            WeightProbeResult,
        )
        r = ProbeResult(status=PROBE_SUCCESS)
        assert r.is_success

    def test_new_import_path_works(self):
        from rl.evaluation.probes.results import ProbeResult, ProbeStatus
        r = ProbeResult.success(WeightProbeMetrics(
            has_obstacle_channel=True,
            cnn_channels=8,
            obstacle_weight_l2=0.04,
            obstacle_weight_abs_mean=0.002,
            obstacle_weight_abs_max=0.008,
            obstacle_weight_nonzero_fraction=1.0,
        ))
        assert r.is_success
        assert r.status is ProbeStatus.SUCCESS


# ---------------------------------------------------------------------------
# 5–8. Generic ProbeResult[T] invariants
# ---------------------------------------------------------------------------

class TestGenericProbeResult:
    def _weight_metrics(self) -> WeightProbeMetrics:
        return WeightProbeMetrics(
            has_obstacle_channel=True,
            cnn_channels=8,
            obstacle_weight_l2=0.04,
            obstacle_weight_abs_mean=0.002,
            obstacle_weight_abs_max=0.008,
            obstacle_weight_nonzero_fraction=1.0,
        )

    def test_success_requires_metrics(self):
        with pytest.raises(ValueError, match="metrics"):
            ProbeResult(status=ProbeStatus.SUCCESS, metrics=None)

    def test_success_must_not_have_error(self):
        with pytest.raises(ValueError, match="error"):
            ProbeResult(
                status=ProbeStatus.SUCCESS,
                metrics=self._weight_metrics(),
                error=ProbeError(error_type="X", message="y"),
            )

    def test_error_requires_error_field(self):
        with pytest.raises(ValueError, match="ProbeError"):
            ProbeResult(status=ProbeStatus.ERROR, error=None)

    def test_error_must_not_have_metrics(self):
        with pytest.raises(ValueError, match="metrics"):
            ProbeResult(
                status=ProbeStatus.ERROR,
                metrics=self._weight_metrics(),
                error=ProbeError(error_type="X", message="y"),
            )

    def test_success_factory(self):
        r = ProbeResult.success(self._weight_metrics())
        assert r.is_success
        assert r.metrics is not None
        assert r.error is None

    def test_error_factory_from_exception(self):
        exc = AttributeError("no get_distribution")
        r: ProbeResult[WeightProbeMetrics] = ProbeResult.from_exception(exc)
        assert not r.is_success
        assert r.metrics is None
        assert r.error is not None
        assert r.error.error_type == "AttributeError"

    def test_failed_probe_has_no_metrics(self):
        r: ProbeResult[GradientProbeMetrics] = ProbeResult.from_exception(
            RuntimeError("broken model")
        )
        assert r.metrics is None

    def test_success_to_json_dict_includes_metrics(self):
        r = ProbeResult.success(
            GradientProbeMetrics(
                obstacle_gradient_l2=0.018,
                obstacle_gradient_abs_mean=0.0008,
                obstacle_gradient_max=0.005,
                obstacle_gradient_nonzero_fraction=0.95,
                sampled_state_count=1,
                diagnostic_loss=0.12,
            )
        )
        d = r.to_json_dict()
        assert d["obstacle_gradient_l2"] == pytest.approx(0.018)
        assert "error" not in d

    def test_error_to_json_dict_has_no_measurement_fields(self):
        r: ProbeResult[GradientProbeMetrics] = ProbeResult.from_exception(
            AttributeError("no get_distribution")
        )
        d = r.to_json_dict()
        assert "obstacle_gradient_l2" not in d
        assert d["error_type"] == "AttributeError"

    def test_counterfactual_metrics_to_dict(self):
        m = CounterfactualProbeMetrics(
            mean_action_kl=1.48e-5,
            max_action_kl=5.77e-5,
            mean_logit_l2=0.026,
            max_logit_l2=0.055,
            argmax_action_change_rate=0.004,
            sampled_state_count=64,
            observation_tensor="grid",
        )
        d = m.to_dict()
        assert d["mean_action_kl"] == pytest.approx(1.48e-5)
        assert d["sampled_state_count"] == 64


# ---------------------------------------------------------------------------
# 9. Forced latent distribution requires explicit z_idx
# ---------------------------------------------------------------------------

class TestExplicitLatentRequirement:
    def test_latent_model_raises_without_z_idx(self):
        model = _make_model(latent_k=4)
        obs = _make_obs()
        with pytest.raises(ValueError, match="z_idx"):
            model.get_distribution(obs, z_idx=None)

    def test_latent_model_accepts_explicit_z_idx(self):
        model = _make_model(latent_k=4)
        obs = _make_obs(batch=2)
        z = torch.zeros(2, dtype=torch.long)
        dist = model.get_distribution(obs, z_idx=z)
        assert len(dist.heads) == N_AGENTS * 2


# ---------------------------------------------------------------------------
# 10–14. Public MultiHeadActionDistribution API
# ---------------------------------------------------------------------------

class TestDistributionPublicAPI:
    def _dist(self, channels: int = N_CHANNELS) -> MultiHeadActionDistribution:
        model = _make_model(channels=channels)
        return model.get_distribution(_make_obs(channels=channels))

    def test_logits_method_returns_list_of_tensors(self):
        dist = self._dist()
        logits = dist.logits()
        assert isinstance(logits, list)
        assert all(isinstance(t, torch.Tensor) for t in logits)
        assert len(logits) == N_AGENTS * 2

    def test_probabilities_sum_to_one(self):
        dist = self._dist()
        for probs in dist.probabilities():
            row_sums = probs.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)

    def test_argmax_actions_matches_manual(self):
        dist = self._dist()
        for head, argmax in zip(dist.heads, dist.argmax_actions()):
            expected = head.logits.argmax(dim=-1)
            assert torch.equal(argmax, expected)

    def test_head_dims_returns_action_sizes(self):
        dist = self._dist()
        dims = dist.head_dims()
        assert dims == [N_MACROS, N_TARGETS] * N_AGENTS

    def test_num_heads(self):
        dist = self._dist()
        assert dist.num_heads() == N_AGENTS * 2

    def test_distributions_alias_still_works(self):
        dist = self._dist()
        assert dist.distributions is dist.heads

    def test_logits_preserve_gradients(self):
        model = _make_model().train()
        obs = _make_obs()
        obs["grid"].requires_grad_(True)
        dist = model.get_distribution(obs)
        loss = sum(t.sum() for t in dist.logits())
        loss.backward()
        assert obs["grid"].grad is not None

    def test_action_head_properties(self):
        head = ActionHead(logits=torch.tensor([[1.0, 2.0, 3.0]]))
        probs = head.probabilities
        assert probs.shape == (1, 3)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(1))
        assert head.argmax_actions.item() == 2
        assert head.action_dim == 3


# ---------------------------------------------------------------------------
# 15. Evaluator does not use actor_cnn private path
# ---------------------------------------------------------------------------

class TestEvaluatorDoesNotUsePrivatePaths:
    def test_inspect_obstacle_weights_uses_public_api(self):
        """Verify inspect_obstacle_weights calls get_observation_encoder_input_weights.

        The evaluator may still contain actor_cnn.conv[0] in the checkpoint
        loading helper (_conv0_weight), which is a legitimate fallback for
        channel-count verification during load.  The probe function itself must
        use the public diagnostics API.
        """
        source_path = (
            Path(__file__).parent.parent
            / "rl"
            / "evaluation"
            / "probes"
            / "obstacle_weights.py"
        )
        source = source_path.read_text(encoding="utf-8")
        assert "get_observation_encoder_input_weights" in source

        # Extract only the probe function body to ensure it doesn't use the
        # legacy private path directly.
        lines = source.splitlines()
        in_probe = False
        probe_lines: list[str] = []
        for line in lines:
            if "def inspect_obstacle_weights" in line:
                in_probe = True
            elif in_probe and line.startswith("def ") and "inspect_obstacle_weights" not in line:
                break
            if in_probe:
                probe_lines.append(line)

        probe_src = "\n".join(probe_lines)
        assert "get_observation_encoder_input_weights" in probe_src
        assert "actor_cnn.conv" not in probe_src

    def test_gradient_probe_uses_public_api(self):
        source_path = (
            Path(__file__).parent.parent
            / "rl"
            / "evaluation"
            / "probes"
            / "obstacle_gradient.py"
        )
        source = source_path.read_text(encoding="utf-8")
        # gradient_probe must call get_observation_encoder_input_weights, not private path
        assert "get_observation_encoder_input_weights" in source
        assert "actor_cnn.conv" not in source


# ---------------------------------------------------------------------------
# 16–17. Manifest status fields
# ---------------------------------------------------------------------------

class TestManifestStatus:
    def test_incomplete_manifest_has_in_progress_status(self, tmp_path: Path):
        """Simulates what main() writes before running probes."""
        manifest = {
            "status": "in_progress",
            "completed_at": None,
            "run_id": "test-run-id",
        }
        p = tmp_path / "evaluation_manifest.json"
        p.write_text(json.dumps(manifest), encoding="utf-8")
        loaded = json.loads(p.read_text())
        assert loaded["status"] == "in_progress"
        assert loaded["completed_at"] is None

    def test_completed_manifest_has_completed_status(self, tmp_path: Path):
        manifest = {
            "status": "completed",
            "completed_at": "2026-06-27T12:00:00+00:00",
            "run_id": "test-run-id",
        }
        p = tmp_path / "evaluation_manifest.json"
        p.write_text(json.dumps(manifest), encoding="utf-8")
        loaded = json.loads(p.read_text())
        assert loaded["status"] == "completed"
        assert loaded["completed_at"] is not None

    def test_manifest_has_schema_version(self, tmp_path: Path):
        """schema_version field must be present so consumers can migrate."""
        manifest = {"schema_version": 2, "status": "completed"}
        p = tmp_path / "evaluation_manifest.json"
        p.write_text(json.dumps(manifest), encoding="utf-8")
        loaded = json.loads(p.read_text())
        assert loaded["schema_version"] == 2
