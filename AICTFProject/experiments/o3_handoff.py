"""O3 pre-action handoff at the collector's action-selection seam.

Frozen by: artifacts/o3_preregistration/O3_OPTIMIZATION_BOUNDARY_FROZEN.json (0b18ed3)

    G0 chooses the natural prefix; at the first C_fork, O3 owns the rest of the
    episode; only that suffix teaches O3.

THE ORDERING THAT MATTERS
-------------------------
The detector is evaluated on env.core BEFORE actions are chosen, so an
environment firing at step h has O3 supply the action for step h itself. The
alternative ordering -- choose G0's action, then evaluate -- starts O3 at h+1
and silently shifts the context under study by one decision. That difference is
tiny in code and invisible in loss curves, so it is asserted rather than
trusted: trigger_step[i] == first_o3_action_step[i].

PER-ENV INDEPENDENCE
--------------------
o3_active is a (B,) vector, not a scalar. Sixteen environments advance and
terminate independently, so at any step some are in G0 prefix and others in O3
suffix. The latch is one-way WITHIN an episode and is cleared only for the
environments that actually terminated.

SUB-BATCHED EXECUTION
---------------------
O3 is invoked only on active rows and G0 only on inactive rows. Running both
models on all rows and discarding the unused half would be weaker: excluding
inactive rows means O3 never observes prefix states at all, which protects
against RNG consumption and any future batch-dependent layer.

SCOPE: this module performs action handoff and credit bookkeeping only. It does
not restrict PPO optimization -- that is the separate credit-boundary step.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from rl.analysis.c_fork_detector import BatchedCForkDetector  # noqa: E402


@dataclass
class HandoffState:
    """Per-env latch, credit vector and throughput counters."""

    n_envs: int
    o3_active: np.ndarray = field(init=False)
    last_credit: np.ndarray = field(init=False)
    trigger_step: np.ndarray = field(init=False)
    first_o3_action_step: np.ndarray = field(init=False)
    step_in_episode: np.ndarray = field(init=False)

    episodes_seen: int = 0
    episodes_with_handoff: int = 0
    environment_steps: int = 0
    credited_o3_steps: int = 0
    post_handoff_lengths: list = field(default_factory=list)
    _len_accum: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        n = int(self.n_envs)
        self.o3_active = np.zeros(n, dtype=bool)
        self.last_credit = np.zeros(n, dtype=bool)
        self.trigger_step = np.full(n, -1, dtype=np.int64)
        self.first_o3_action_step = np.full(n, -1, dtype=np.int64)
        self.step_in_episode = np.zeros(n, dtype=np.int64)
        self._len_accum = np.zeros(n, dtype=np.int64)

    def on_episode_end(self, done_mask: np.ndarray) -> None:
        """Reset ONLY terminated envs; the others keep their latch."""
        idx = np.asarray(done_mask, dtype=bool)
        if not idx.any():
            return
        self.episodes_seen += int(idx.sum())
        self.episodes_with_handoff += int((idx & self.o3_active).sum())
        for i in np.flatnonzero(idx & self.o3_active):
            self.post_handoff_lengths.append(int(self._len_accum[i]))
        self.o3_active[idx] = False
        self.trigger_step[idx] = -1
        self.first_o3_action_step[idx] = -1
        self.step_in_episode[idx] = 0
        self._len_accum[idx] = 0

    def throughput(self) -> dict:
        env_steps = max(int(self.environment_steps), 1)
        eps = max(int(self.episodes_seen), 1)
        return {
            "episodes_seen": int(self.episodes_seen),
            "episodes_with_handoff": int(self.episodes_with_handoff),
            "handoff_rate": round(self.episodes_with_handoff / eps, 4),
            "environment_steps": int(self.environment_steps),
            "credited_o3_steps": int(self.credited_o3_steps),
            "credited_fraction": round(self.credited_o3_steps / env_steps, 4),
            "mean_post_handoff_length": (
                round(float(np.mean(self.post_handoff_lengths)), 2)
                if self.post_handoff_lengths else None
            ),
            "effective_o3_steps_per_1k_env_steps": round(
                1000.0 * self.credited_o3_steps / env_steps, 1
            ),
        }


def install_o3_handoff(trainer, g0_policy, *, strict: bool = True):
    """Wrap the O3 model's ``act`` so G0 supplies actions before the handoff.

    Returns (HandoffState, uninstall). ``strict`` asserts the frozen ordering
    invariant trigger_step == first_o3_action_step on every firing.
    """
    model = trainer.model
    env = trainer.env
    core = env.core
    n_envs = int(env.num_envs)

    detector = BatchedCForkDetector(n_envs=n_envs)
    state = HandoffState(n_envs=n_envs)
    real_act = model.act

    def act(obs_t, context_state=None, z_idx=None, **kw):
        # 1. Detector BEFORE action selection, so a firing env acts with O3 now.
        fired = detector.step(core)
        newly = fired & ~state.o3_active
        if newly.any():
            state.trigger_step[newly] = state.step_in_episode[newly]
        state.o3_active |= fired

        active = state.o3_active.copy()
        state.last_credit = active.copy()

        # 2. Sub-batched execution: O3 never sees prefix rows.
        actions, values, log_probs, extra = real_act(obs_t, context_state, z_idx=z_idx, **kw)
        if not active.all():
            g0_actions = _g0_actions(g0_policy, obs_t, n_envs)
            inactive = ~active
            actions = actions.clone()
            actions[inactive] = g0_actions[inactive].to(actions.device, actions.dtype)

        # 3. Record the first step O3 actually supplied an action.
        first = active & (state.first_o3_action_step < 0)
        if first.any():
            state.first_o3_action_step[first] = state.step_in_episode[first]
        if strict and newly.any():
            bad = np.flatnonzero(
                newly & (state.trigger_step != state.first_o3_action_step)
            )
            if bad.size:
                raise AssertionError(
                    "O3 handoff ordering violated: trigger_step != "
                    f"first_o3_action_step for envs {bad.tolist()} "
                    f"({state.trigger_step[bad].tolist()} vs "
                    f"{state.first_o3_action_step[bad].tolist()})"
                )

        state.environment_steps += n_envs
        state.credited_o3_steps += int(active.sum())
        state._len_accum[active] += 1
        state.step_in_episode += 1
        return actions, values, log_probs, extra

    model.act = act

    def uninstall():
        model.act = real_act

    return state, detector, uninstall


def _g0_actions(g0_policy, obs_t, n_envs: int) -> torch.Tensor:
    """Frozen-G0 actions for the whole batch, no gradient."""
    with torch.no_grad():
        out = g0_policy.act(obs_t, None, z_idx=None)
    actions = out[0] if isinstance(out, tuple) else out
    return actions.detach()


__all__ = ["HandoffState", "install_o3_handoff"]
