"""Live diagnostic printing for CausalSequenceRunner, added AFTER attempt 1's restart.

Attempt 1 (CCP_SUCCESSOR_ATTEMPT1_ABORTED_FOR_OBSERVABILITY.json) was stopped purely because
the causal mechanism's liveness during a 1M-step run could only be confirmed by reading the
runner object at the end -- fine for a terminal record, useless for watching training happen.

This module changes NOTHING about training: no new tensor is created, no RNG is consumed, no
gradient path is touched. It wraps ``note_ppo_minibatch`` on the LIVE INSTANCE only -- never
the class -- calls the original method exactly once, and prints a line if (and only if) that
call actually fired an update. Same pattern experiments/run_hog_psp_v3_production.py's
TrajectoryChannel already uses for wrapping the collector's collect().

tests/test_causal_sequence_diagnostics_neutral.py proves this claim rather than asserting it:
two identically-seeded runners, one wrapped and one not, produce bitwise-identical model
parameters after the same number of updates.
"""
from __future__ import annotations


def install_diagnostics_reporter(seq_runner, *, every: int = 1):
    """Wrap seq_runner.note_ppo_minibatch to print after every ``every``-th firing update.

    Returns the ORIGINAL bound method, so the caller can restore it -- matching the
    install/restore pattern used throughout this program (critic auditors, legacy tripwires).
    """
    original = seq_runner.note_ppo_minibatch
    state = {"fires": 0}

    def wrapped():
        fired = original()
        if fired:
            state["fires"] += 1
            if state["fires"] % max(1, int(every)) == 0:
                tel = seq_runner.telemetry()
                print(
                    f"[causal] update={tel['updates']:5d}  "
                    f"minibatches={tel['n_ppo_minibatches']:6d}  "
                    f"z0={tel['z0_exposures']:5d}  z1={tel['z1_exposures']:5d}  "
                    f"pos={tel['positive_routes']:4d}  neg={tel['negative_routes']:4d}  "
                    f"loss={tel['last_loss']:+.4f}", flush=True)
        return fired

    seq_runner.note_ppo_minibatch = wrapped
    return original


def restore(seq_runner, original) -> None:
    seq_runner.note_ppo_minibatch = original
