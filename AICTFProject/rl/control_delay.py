"""Eval-only control (action) delay for deployment robustness testing.

Implements DEPLOYMENT_ROBUSTNESS_SPEC.json#control_delay. The policy's selected action is
delayed by N ticks before the environment executes it -- modeling actuator/communication
latency between decision and execution, distinct from sensing latency (localization noise)
or execution fidelity (motion error).

This module contains NO training-loop code, no reference to any trainer/updater, and no
import of anything training-related. It is a pure FIFO buffer sitting between "policy picked
an action" and "environment executes an action," meant to be inserted into an EVAL rollout
loop only. Structural deployment-only guarantee: grep-verified in
verify_no_perturbation_in_training.py that no training entrypoint in experiments/ imports
this module.

Semantics: at delay=0 ticks, push() returns the action unchanged (a true no-op -- the nominal,
undisturbed condition). At delay=N>0, the first N calls return a HOLD action (the environment's
existing macro-commitment semantics already treat a repeated/no-new-command tick correctly, so
the hold action is simply "whatever the agent's current commitment already is" -- concretely,
we return the PREVIOUS realized action, matching the real-world meaning of "the actuator is
still executing the last command it received" rather than inventing a new no-op action type),
then returns the action from N ticks ago on every subsequent call.
"""
from __future__ import annotations

from collections import deque
from typing import Any


class ControlDelayError(RuntimeError):
    pass


class DelayBuffer:
    """FIFO action delay of exactly `ticks` steps. ticks=0 is a verified no-op."""

    def __init__(self, ticks: int):
        if ticks < 0:
            raise ControlDelayError(f"ticks must be >= 0, got {ticks}")
        self.ticks = int(ticks)
        self._q: deque = deque(maxlen=max(1, self.ticks))
        self._last_real_action: Any = None

    def push(self, action: Any) -> Any:
        """Submit the policy's freshly-selected action; returns the action to EXECUTE."""
        if self.ticks == 0:
            return action
        if self._last_real_action is None:
            self._last_real_action = action
        if len(self._q) < self.ticks:
            # buffer not yet full: actuator is still "warming up" under the delay --
            # execute the most recent REAL action received so far (matches "the actuator
            # is still executing the last command it received", not a fabricated no-op)
            out = self._last_real_action
            self._q.append(action)
            self._last_real_action = action
            return out
        out = self._q[0]
        self._q.append(action)
        self._last_real_action = action
        return out

    def reset(self) -> None:
        """Call at episode boundaries -- delay state must not leak across episodes."""
        self._q.clear()
        self._last_real_action = None


def self_test() -> dict:
    """Proves the three properties the frozen spec and any preflight will check for,
    using only plain Python objects -- no env, no GPU, no policy. Returns a report dict;
    raises ControlDelayError on any failure rather than returning a silent False.
    """
    report = {}

    # 1. ticks=0 is a TRUE no-op: every pushed action comes back unchanged, immediately
    buf = DelayBuffer(0)
    seq = [f"a{i}" for i in range(10)]
    out0 = [buf.push(a) for a in seq]
    if out0 != seq:
        raise ControlDelayError(f"ticks=0 is not a no-op: {out0} != {seq}")
    report["zero_ticks_is_noop"] = True

    # 2. ticks=N: output at step t (for t>=N) equals the REAL action submitted at step t-N
    for n in (1, 2, 4):
        buf = DelayBuffer(n)
        seq = [f"a{i}" for i in range(20)]
        out = [buf.push(a) for a in seq]
        mismatches = [(t, out[t], seq[t - n]) for t in range(n, len(seq)) if out[t] != seq[t - n]]
        if mismatches:
            raise ControlDelayError(f"ticks={n}: delayed output does not match "
                                    f"seq[t-{n}] at {mismatches[:3]}")
        report[f"ticks_{n}_correctly_delayed"] = True

    # 3. reset() clears state -- a fresh episode does not see the previous episode's tail
    buf = DelayBuffer(2)
    for a in ["x0", "x1", "x2", "x3"]:
        buf.push(a)
    buf.reset()
    out_after_reset = [buf.push(a) for a in ["y0", "y1", "y2"]]
    # immediately after reset, ticks=2 means the first 2 calls echo the most recent real
    # action received so far (warm-up), matching case 2's own semantics from a cold start
    expected = ["y0", "y0", "y0"]
    if out_after_reset != expected:
        raise ControlDelayError(f"reset() did not clear state: got {out_after_reset}, "
                                f"expected {expected} (identical to a cold-start buffer)")
    report["reset_clears_state"] = True

    return report


if __name__ == "__main__":
    r = self_test()
    print("DelayBuffer self-test:")
    for k, v in r.items():
        print(f"  [PASS] {k}: {v}")
    print("\nALL PASS")
