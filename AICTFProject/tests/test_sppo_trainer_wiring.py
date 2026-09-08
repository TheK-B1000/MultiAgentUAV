"""SPPPO trainer-wiring smoke: construct -> attach -> use.

The unit tests in test_sppo_ranking.py prove the mechanism is correct in
isolation. They cannot prove it is REACHED. The SAPPO no-op ran a full 2x500k
campaign in which every loss curve looked healthy while the treatment never
executed, because the updater cached the runner before the orchestrator
attached it.

So these tests exercise the seam itself:

  * the runner is read at USE time, not cached at construction
  * attaching AFTER the updater exists still activates the branch
  * cadence is enforced by aborting, not by logging
  * the lambda_R = 0 control leaves the seam structurally absent
  * every telemetry key the protocol requires is present and finite
  * the CSV schema actually carries those columns
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rl.scorer.ranking import POLE_A, POLE_B, RankingRunner  # noqa: E402
from tests.test_sppo_ranking import TinyActor, _frozen_qpsi, _obs  # noqa: E402


class FakeRuntime:
    """Stands in for the orchestrator's runtime object."""
    pass


class SeamHarness:
    """Reproduces the updater's read-at-use-time seam and its assertions."""

    def __init__(self, runtime):
        self.runtime = runtime            # captured BEFORE any runner exists
        self.records: list[dict] = []

    def _ranking_runner(self):
        return getattr(self.runtime, "sppo_ranking_runner", None)

    def _assert_ranking_cadence(self, runner):
        n_ppo = int(runner.n_ppo_actor_minibatches)
        n_rank = int(runner.n_ranking_updates)
        expected = n_ppo // int(runner.cadence)
        if n_rank != expected:
            raise RuntimeError(
                f"SPPPO ranking cadence violated: {n_rank} ranking updates after "
                f"{n_ppo} PPO actor minibatches, expected {expected}")

    def ppo_minibatch(self, batch):
        runner = self._ranking_runner()
        if runner is None:
            return False                  # structural absence: nothing happens
        runner.note_ppo_minibatch(batch)
        self._assert_ranking_cadence(runner)
        d = runner.last_diag
        self.records.append({
            "sppo_n_ppo_actor_updates": float(runner.n_ppo_actor_minibatches),
            "sppo_n_rank_updates": float(runner.n_ranking_updates),
            "sppo_rank_to_ppo_ratio": float(
                runner.n_ranking_updates / max(1, runner.n_ppo_actor_minibatches)),
            "sppo_rank_loss": float(runner.last_loss),
            "sppo_rank_activation_rate": float(d.get("activation_rate", float("nan"))),
            "sppo_delta_A": float(d.get("delta_A_mean", float("nan"))),
            "sppo_delta_B": float(d.get("delta_B_mean", float("nan"))),
            "sppo_lambda_rank": float(runner.lambda_rank),
            "sppo_margin": float(runner.margin),
        })
        return True


def _ppo_batch(n=32, seed=21):
    """A PPO minibatch WITHOUT opponent_id -- used only to prove it is refused."""
    gen = torch.Generator().manual_seed(seed)
    o = _obs(n, gen)
    return {
        "obs_grid": o["grid"], "obs_vec": o["vec"],
        "obs_agent_mask": o["agent_mask"], "obs_mask": o["mask"],
        "z": torch.tensor([0] * (n // 2) + [1] * (n // 2)),
    }


def test_runner_attached_after_construction_still_fires():
    """The exact seam the SAPPO no-op broke."""
    runtime = FakeRuntime()
    harness = SeamHarness(runtime)          # built BEFORE the runner exists
    batch = _real_batch()

    assert harness.ppo_minibatch(batch) is False, "no runner: nothing should run"
    assert harness.records == []

    model, q = TinyActor(), _frozen_qpsi()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    runtime.sppo_ranking_runner = RankingRunner(     # attached LATE
        model, q, opt, lambda_rank=0.3, margin=0.04, cadence=1)

    assert harness.ppo_minibatch(batch) is True, "late attachment did not activate"
    assert harness.records[-1]["sppo_n_rank_updates"] == 1.0


def test_unmapped_z_is_rejected_rather_than_guessed():
    """K=4 by mistake must abort, not silently map into the K=2 pole space."""
    model, q = TinyActor(), _frozen_qpsi()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    runner = RankingRunner(model, q, opt, lambda_rank=0.3, margin=0.04)
    batch = _real_batch(n=8)
    batch["z"] = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    with pytest.raises(RuntimeError, match="no pole mapping"):
        runner._sample(batch)


def test_cadence_violation_aborts_the_run():
    """A broken seam must die immediately, not be discovered at 1M steps."""
    runtime = FakeRuntime()
    harness = SeamHarness(runtime)
    model, q = TinyActor(), _frozen_qpsi()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    runner = RankingRunner(model, q, opt, lambda_rank=0.3, margin=0.04, cadence=1)
    runtime.sppo_ranking_runner = runner

    harness.ppo_minibatch(_real_batch())
    runner.n_ranking_updates = 0            # simulate an inert ranking branch
    with pytest.raises(RuntimeError, match="cadence violated"):
        harness.ppo_minibatch(_real_batch())


def test_all_required_telemetry_keys_present_and_finite():
    """Every field the protocol requires, visible from the first interval."""
    import math
    runtime = FakeRuntime()
    harness = SeamHarness(runtime)
    model, q = TinyActor(), _frozen_qpsi()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    runtime.sppo_ranking_runner = RankingRunner(
        model, q, opt, lambda_rank=0.3, margin=0.04, cadence=1)

    for _ in range(4):
        harness.ppo_minibatch(_real_batch())
    rec = harness.records[-1]

    required = ["sppo_n_ppo_actor_updates", "sppo_n_rank_updates",
                "sppo_rank_to_ppo_ratio", "sppo_rank_loss",
                "sppo_rank_activation_rate", "sppo_delta_A", "sppo_delta_B"]
    for k in required:
        assert k in rec, f"missing telemetry key {k}"
        assert math.isfinite(rec[k]), f"{k} is not finite: {rec[k]}"
    assert rec["sppo_n_rank_updates"] == 4.0
    assert rec["sppo_rank_to_ppo_ratio"] == 1.0


def test_deltas_move_upward_across_the_seam():
    """The quantities the method exists to move, measured through the seam."""
    runtime = FakeRuntime()
    harness = SeamHarness(runtime)
    model, q = TinyActor(), _frozen_qpsi()
    opt = torch.optim.SGD(model.parameters(), lr=5.0)
    runtime.sppo_ranking_runner = RankingRunner(
        model, q, opt, lambda_rank=1.0, margin=0.04, cadence=1)

    batch = _real_batch()
    for _ in range(20):
        harness.ppo_minibatch(batch)

    first, last = harness.records[0], harness.records[-1]
    assert last["sppo_delta_A"] > first["sppo_delta_A"], "delta_A did not rise"
    assert last["sppo_delta_B"] > first["sppo_delta_B"], "delta_B did not rise"


def test_qpsi_sha_unchanged_across_the_seam():
    runtime = FakeRuntime()
    harness = SeamHarness(runtime)
    model, q = TinyActor(), _frozen_qpsi()
    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    runner = RankingRunner(model, q, opt, lambda_rank=1.0, margin=0.04, cadence=1)
    runtime.sppo_ranking_runner = runner

    before = runner.assert_qpsi_unchanged()
    for _ in range(6):
        harness.ppo_minibatch(_real_batch())
    assert runner.assert_qpsi_unchanged() == before, "Q_psi drifted during training"


def test_lambda_zero_control_emits_no_ranking_rows():
    """Structural absence: the control produces NO ranking telemetry at all."""
    runtime = FakeRuntime()                  # no runner ever attached
    harness = SeamHarness(runtime)
    for _ in range(5):
        assert harness.ppo_minibatch(_real_batch()) is False
    assert harness.records == [], "the control emitted ranking telemetry"


def test_csv_schema_carries_the_sppo_columns():
    """Telemetry that never reaches the CSV is telemetry that does not exist."""
    from pathlib import Path
    src = Path("rl/custom_ppo/csv_writers.py").read_text(encoding="utf-8")
    for col in ("sppo_n_ppo_actor_updates", "sppo_n_rank_updates",
                "sppo_rank_to_ppo_ratio", "sppo_rank_loss",
                "sppo_rank_activation_rate", "sppo_delta_A", "sppo_delta_B",
                "sppo_lambda_rank", "sppo_margin"):
        assert f'"{col}"' in src, f"{col} missing from the fixed CSV schema"


# ===================== TRUE-POLE PROVENANCE (orchestrator path) =====================
# Pole must come from real rollout metadata, never from the latent label. z is
# the thing being scored; using it as the scorer's pole input would make a
# broken env/pole assignment invisible behind healthy cadence and loss metrics.

from rl.scorer.ranking import OPPONENT_ID_TO_POLE  # noqa: E402

OP6, OP7 = 5, 6          # pole A, pole B


def _real_batch(n=32, seed=31, corrupt=False):
    """A PPO minibatch as the collector actually writes it, incl. opponent_id."""
    b = _ppo_batch(n, seed)
    opp = torch.tensor([OP6] * (n // 2) + [OP7] * (n // 2))
    if corrupt:
        opp[0] = OP7                      # a z=0 env actually facing pole B
    b["opponent_id"] = opp
    return b


def _runner(model, q, lr=0.1, lam=0.3):
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    return RankingRunner(model, q, opt, lambda_rank=lam, margin=0.04, cadence=1,
                         z_to_pole={0: POLE_A, 1: POLE_B},
                         opponent_to_pole=dict(OPPONENT_ID_TO_POLE))


def test_pole_comes_from_opponent_id_not_from_z():
    r = _runner(TinyActor(), _frozen_qpsi())
    _, pole, _ = r._sample(_real_batch())
    assert (pole[:16] == POLE_A).all() and (pole[16:] == POLE_B).all()
    t = r.telemetry()
    assert t["pole_source"].startswith("TRUE rollout opponent_id")
    assert t["n_pole_consistency_checks"] == 1


def test_z_pole_mismatch_aborts():
    """The integrity check that replaced the old limitation."""
    r = _runner(TinyActor(), _frozen_qpsi())
    with pytest.raises(RuntimeError, match="z/pole consistency violated"):
        r._sample(_real_batch(corrupt=True))


def test_missing_opponent_id_is_refused_not_inferred():
    r = _runner(TinyActor(), _frozen_qpsi())
    b = _ppo_batch()                      # no opponent_id
    with pytest.raises(RuntimeError, match="no opponent_id"):
        r._sample(b)


def test_opponent_outside_the_frozen_pole_map_aborts():
    r = _runner(TinyActor(), _frozen_qpsi())
    b = _real_batch()
    b["opponent_id"] = torch.full_like(b["opponent_id"], 2)   # OP3
    with pytest.raises(RuntimeError, match="outside the frozen"):
        r._sample(b)


def test_attach_lambda_zero_creates_nothing():
    """Control: no runner object, no attribute, nothing to enter."""
    from rl.scorer.attach import attach_ranking_runner
    runtime, model = FakeRuntime(), TinyActor()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    out = attach_ranking_runner(runtime, model, opt, lambda_rank=0.0, device="cpu")
    assert out is None
    assert not hasattr(runtime, "sppo_ranking_runner")
    assert SeamHarness(runtime).ppo_minibatch(_real_batch()) is False


def test_attach_verifies_the_frozen_qpsi_sha():
    """A scorer that is not the frozen one must be refused."""
    from rl.scorer.attach import attach_ranking_runner
    runtime, model = FakeRuntime(), TinyActor()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(RuntimeError, match="sha256"):
        attach_ranking_runner(runtime, model, opt, lambda_rank=0.3,
                              expected_sha256="0" * 64, device="cpu")


@pytest.mark.skipif(not __import__("pathlib").Path(
    "artifacts/strategic_demand/phase0_scorer_data/qpsi_frozen.pt").is_file(),
    reason="frozen scorer weights not present")
def test_attach_real_qpsi_and_run_through_the_seam():
    """Full orchestrator path against the REAL frozen scorer."""
    from rl.scorer.attach import attach_ranking_runner, SPPPO_QPSI_SHA256
    runtime, model = FakeRuntime(), TinyActor()
    opt = torch.optim.SGD(model.parameters(), lr=0.5)
    runner = attach_ranking_runner(runtime, model, opt, lambda_rank=0.3, device="cpu")

    assert runner is not None
    assert runtime.sppo_ranking_runner is runner
    assert runner._qpsi_sha == SPPPO_QPSI_SHA256
    assert runner.margin == 0.04
    assert all(not p.requires_grad for p in runner.qpsi.parameters())

    harness = SeamHarness(runtime)
    before = runner.assert_qpsi_unchanged()
    for _ in range(6):
        assert harness.ppo_minibatch(_real_batch()) is True
    assert runner.assert_qpsi_unchanged() == before
    rec = harness.records[-1]
    assert rec["sppo_n_rank_updates"] == 6.0
    assert rec["sppo_rank_to_ppo_ratio"] == 1.0
    assert runner.telemetry()["n_pole_consistency_checks"] == 6
