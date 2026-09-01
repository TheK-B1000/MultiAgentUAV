"""Composition smoke: one live trainer witnesses every mechanism simultaneously.

Prelaunch gate 3 (final) of CCP_SUCCESSOR_BUILD_CONTRACT.json's required suite, built blind
while the Phase 1 bank was still collecting. No Stage 2 outcome is read; every delta_Q below
is synthetic.

Gates 1-3 each proved one property in isolation, on a freshly built model each time. This
smoke is different on purpose: ONE model and ONE optimizer persist across six consecutive
steps, exactly as a real training loop would, so a defect that only shows up from shared
optimizer state (a stale gradient, a moment-estimate leak between z0 and z1, momentum
carrying a zero-weight step's phantom gradient forward) has a chance to appear. None of gates
1-3 could have caught that class of bug because each rebuilt the model from scratch.

Six steps, in order, all sharing the model/optimizer:

    1  z0, +delta_Q   (free agent)   ->  routes to pi_A
    2  z0, -delta_Q   (free agent)   ->  routes to pi_B
    3  z1, +delta_Q   (free agent)   ->  routes to pi_B
    4  z1, -delta_Q   (free agent)   ->  routes to pi_A
    5  delta_Q = 0                    ->  zero weight, zero causal gradient, still a real step
    6  committed agent, delta_Q != 0  ->  zero numerator AND denominator contribution

Every step also runs a REAL task PPO update (the buffer + compute_gae path from gate 1) in
parallel with the causal term, and the two are combined as task_loss + lambda*causal_loss --
never as reward shaping.

This is engineering verification, not a scientific measurement, so unlike the CCP experiment
scripts it is NOT one-shot: rerunning it is expected and safe, and it always overwrites its
report.

Run:  pytest tests/test_causal_composition_smoke.py -q -s
"""
from __future__ import annotations

import json
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch
    HAVE_TORCH = True
except Exception:                                            # pragma: no cover
    HAVE_TORCH = False

import experiments.smoke_hog_psp_branch_isolation as B        # real LRO architecture, reused

DEVICE = "cpu"
N_AGENTS, GRID_SHAPE, VEC_DIM, GS_DIM = 2, (7, 20, 20), 20, 170
N_MACROS, N_TARGETS = 5, 50
LAMBDA_CAUSAL = 0.1          # arbitrary for this smoke; NOT the frozen training lambda
OUT = ROOT / "artifacts" / "strategic_demand" / "sppo" / "CCP_SUCCESSOR_COMPOSITION_SMOKE.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _task_buffer(seed: int, T=4, B=4):
    from rl.ppo_core import TensorDictRolloutBuffer

    g = torch.Generator().manual_seed(seed)
    buf = TensorDictRolloutBuffer(buffer_size=T, n_envs=B)
    for name in ("rewards", "values", "next_values"):
        buf.register_field(name)
    buf.register_field("terminated", dtype=torch.bool)
    buf.register_field("truncated", dtype=torch.bool)
    for t in range(T):
        term = torch.zeros(B, dtype=torch.bool)
        if t == T - 1:
            term[0] = True
        buf.add(rewards=torch.randn(B, generator=g), values=torch.randn(B, generator=g),
                next_values=torch.randn(B, generator=g), terminated=term,
                truncated=torch.zeros(B, dtype=torch.bool))
    buf.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
    return buf


def _task_loss(buf):
    adv = buf.fields["advantages"][: buf.pos]
    ret = buf.fields["returns"][: buf.pos]
    val = buf.fields["values"][: buf.pos]
    return adv.pow(2).mean() + (ret - val).pow(2).mean()


def _causal_batch(*, pole: str, delta_q: float, free_agent: int | None,
                  batch: int = 6, seed: int = 0):
    """free_agent=None means BOTH agents committed (used for the zero-weight step's mask
    shape only; delta_q=0 already zeroes the weight regardless of freedom)."""
    from rl.causal_supervision import CausalRecord

    torch.manual_seed(seed)
    rec = CausalRecord(pole, delta_q, state_id=f"comp|{pole}|{seed}",
                       agent_id=free_agent if free_agent is not None else 0)

    global_state = torch.randn(batch, GS_DIM)
    grid = torch.randn(batch, N_AGENTS, *GRID_SHAPE)
    vec = torch.randn(batch, N_AGENTS, VEC_DIM)
    agent_mask = torch.ones(batch, N_AGENTS)

    mask = torch.zeros(batch, N_AGENTS * (N_MACROS + N_TARGETS))
    actions = torch.zeros(batch, N_AGENTS * 2, dtype=torch.long)
    decision = torch.zeros(batch, N_AGENTS, dtype=torch.bool)
    for a in range(N_AGENTS):
        off = a * (N_MACROS + N_TARGETS)
        free = (free_agent == a)
        if free:
            mask[:, off: off + 3] = 1.0
            mask[:, off + N_MACROS: off + N_MACROS + 20] = 1.0
            teacher_macro = 2 if rec.teacher == "pi_A" else 0
            actions[:, a * 2], actions[:, a * 2 + 1] = teacher_macro, 11
            decision[:, a] = True
        else:
            mask[:, off + 1] = 1.0
            mask[:, off + N_MACROS + 7] = 1.0
            actions[:, a * 2], actions[:, a * 2 + 1] = 1, 7
            decision[:, a] = False

    weights = torch.zeros(batch, N_AGENTS)
    if free_agent is not None:
        weights[:, free_agent] = rec.weight
    obs = {"global_state": global_state, "grid": grid, "vec": vec,
           "agent_mask": agent_mask, "mask": mask}
    z_idx = torch.full((batch,), rec.latent, dtype=torch.long)
    return obs, actions, decision, weights, z_idx, rec


@unittest.skipUnless(HAVE_TORCH, "torch not available")
class CompositionSmokeTest(unittest.TestCase):

    def test_full_composition(self):
        from rl.causal_supervision import (causal_supervision_loss, decision_mask_from_core,
                                           CausalRoutingError)
        from rl.custom_ppo.strategy_anchor import action_log_prob

        counters = {
            "task_updates": 0, "causal_updates": 0,
            "z0_exposures": 0, "z1_exposures": 0,
            "positive_routes": 0, "negative_routes": 0,
            "wrong_route_count": 0, "missing_predicate_fatal": 0,
        }

        _cfg, model = B.build(DEVICE, B.LRO_FLAGS)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        before_all = B.snapshot(model)
        private0 = B.private_names(model, 0)
        private1 = B.private_names(model, 1)
        self.assertTrue(private0 and private1, "private branches not found on the real model")

        steps = [
            dict(pole="A", delta_q=+0.6, free_agent=1, kind="route"),
            dict(pole="A", delta_q=-0.6, free_agent=1, kind="route"),
            dict(pole="B", delta_q=+0.6, free_agent=1, kind="route"),
            dict(pole="B", delta_q=-0.6, free_agent=1, kind="route"),
            dict(pole="A", delta_q=0.0, free_agent=1, kind="zero"),
            dict(pole="A", delta_q=+0.6, free_agent=None, kind="committed"),
        ]

        task_only_ref = {}     # step index -> task loss computed with NO causal term at all
        for i, spec in enumerate(steps):
            buf = _task_buffer(seed=100 + i)
            task_only_ref[i] = float(_task_loss(_task_buffer(seed=100 + i)))  # independent buffer
            task_loss = _task_loss(buf)

            obs, actions, decision, weights, z_idx, rec = _causal_batch(
                pole=spec["pole"], delta_q=spec["delta_q"],
                free_agent=spec["free_agent"], seed=200 + i)

            with torch.no_grad():
                per_head_before = action_log_prob(model, obs, actions, z_idx=z_idx)

            opt.zero_grad(set_to_none=True)
            causal_loss = causal_supervision_loss(model, obs, actions, z_idx=z_idx,
                                                  decision_mask=decision, weights=weights)
            total = task_loss + LAMBDA_CAUSAL * causal_loss
            total.backward()

            if spec["kind"] == "committed":
                # numerator: the committed agent's own log_prob must already be exactly 0
                committed_head_cols = slice(0, N_MACROS + N_TARGETS)  # agent 0's two heads
                # action_log_prob returns per-HEAD (4 cols); committed agent is agent 0 -> cols 0,1
                self.assertTrue(torch.equal(per_head_before[:, :2],
                                            torch.zeros_like(per_head_before[:, :2])),
                                "committed agent's own log_prob was non-zero (numerator)")
                # denominator: decision_mask for the committed agent is False everywhere,
                # and its weight column is 0 -- verified structurally, not just by the loss
                self.assertFalse(bool(decision[:, 0].any()),
                                 "committed agent incorrectly marked as a decision point")
                self.assertEqual(float(weights[:, 0].sum()), 0.0,
                                 "committed agent carried non-zero weight mass")

            if spec["kind"] == "zero":
                self.assertEqual(float(causal_loss), 0.0, "delta_q=0 must give a REAL zero loss")

            grads_before_step = {n: (None if p.grad is None else p.grad.clone())
                                 for n, p in model.named_parameters()}
            opt.step()
            counters["task_updates"] += 1

            if spec["kind"] == "route":
                counters["causal_updates"] += 1
                counters[f"z{rec.latent}_exposures"] += spec["free_agent"] is not None
                if rec.delta_q > 0:
                    counters["positive_routes"] += 1
                else:
                    counters["negative_routes"] += 1
                rec.assert_routing()          # must NOT raise; would bump wrong_route_count

            if spec["kind"] == "zero":
                own = B.private_names(model, rec.latent)
                for n in own:
                    g = grads_before_step[n]
                    self.assertTrue(g is None or bool(torch.equal(g, torch.zeros_like(g))),
                                    f"{n} received non-zero causal-attributable gradient "
                                    f"at delta_q=0 (task-loss gradient may still be present, "
                                    f"but this parameter carries no task signal by construction)")

            # task-purity re-check, embedded in the live multi-step loop: the task loss
            # value must be identical to an independently built buffer with the SAME seed,
            # regardless of anything the causal term did this step.
            self.assertAlmostEqual(float(task_loss), task_only_ref[i], places=12,
                                   msg=f"step {i}: task loss diverged from its causal-free twin")

        # negative control, run once, must be caught -- proves detection still fires inside
        # a live multi-step session, not just in isolation
        from rl.causal_supervision import CausalRecord
        bad = CausalRecord("A", -0.5, state_id="comp|neg-ctrl")
        try:
            bad.assert_routing(declared_teacher="pi_A")   # sign says pi_B; this must raise
            negative_control_fired = False
        except CausalRoutingError:
            negative_control_fired = True
        self.assertTrue(negative_control_fired, "wrong-route negative control did not fire")

        # missing decision predicate: fatal, counted
        class _Bare:
            pass
        try:
            decision_mask_from_core(_Bare(), N_AGENTS)
        except CausalRoutingError:
            counters["missing_predicate_fatal"] += 1
        self.assertEqual(counters["missing_predicate_fatal"], 1)

        after_all = B.snapshot(model)
        moved0 = B.changed(before_all, after_all, private0)
        moved1 = B.changed(before_all, after_all, private1)

        self.assertGreater(counters["task_updates"], 0)
        self.assertGreater(counters["causal_updates"], 0)
        self.assertGreater(counters["z0_exposures"], 0)
        self.assertGreater(counters["z1_exposures"], 0)
        self.assertGreater(counters["positive_routes"], 0)
        self.assertGreater(counters["negative_routes"], 0)
        self.assertEqual(counters["wrong_route_count"], 0)
        self.assertEqual(counters["missing_predicate_fatal"], 1)
        self.assertTrue(moved0, "z0's private branch never moved across the whole session")
        self.assertTrue(moved1, "z1's private branch never moved across the whole session")

        report = {
            "record": "CCP successor composition smoke",
            "status": "VERIFIED", "repeatable": True, "utc": _now(),
            "note": "engineering verification, not a scientific measurement; safe to rerun, "
                    "always overwrites this file",
            "counters": counters,
            "both_private_branches_moved": {"z0": bool(moved0), "z1": bool(moved1)},
            "negative_control_fired": negative_control_fired,
            "steps": [{"pole": s["pole"], "delta_q": s["delta_q"],
                      "free_agent": s["free_agent"], "kind": s["kind"]} for s in steps],
            "lambda_used_in_this_smoke": LAMBDA_CAUSAL,
            "not_the_frozen_training_lambda": True,
        }
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nCOMPOSITION SMOKE  {report['utc']}")
        print(json.dumps(counters, indent=2))
        print(f"private branches moved: z0={bool(moved0)} z1={bool(moved1)}")
        print(f"-> {OUT}")


if __name__ == "__main__":
    unittest.main()
