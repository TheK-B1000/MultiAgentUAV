"""RSCFT preflight addendum: verify the EMA update ordering mechanically.

The frozen preflight (RSCFT_PREFLIGHT.json) covered 12 checks. This addendum records the
13th, raised by the PI after that record was frozen:

    theta_k -> actor optimizer step -> theta_{k+1} -> EMA update -> theta_bar_{k+1}

The EMA teacher must consume the POST-update student parameters. If it consumed the
pre-update ones, the teacher would lag by exactly one step -- which would still look healthy
in every counter (86 EMA updates for 86 retention updates) while quietly regularising toward
a stale target.

Verified two independent ways rather than asserted in prose:
  13a. ARITHMETIC: after a simulated actor step, check theta_bar_1 equals
       decay*theta_bar_0 + (1-decay)*theta_POST exactly, and does NOT equal the same
       expression formed with theta_PRE.
  13b. STRUCTURAL: inspect RetentionRunner.note_ppo_minibatch's own source and confirm
       opt.step() precedes teacher.update(), and that both are present.

Runs on CPU by default so it cannot contend with a training run for the GPU.

Run:  python experiments/rscft_preflight_addendum_ema_ordering.py
"""
from __future__ import annotations

import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
OUT = SD / "RSCFT_PREFLIGHT_ADDENDUM_EMA_ORDERING.json"
DECAY = 0.995


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class _TinyActor(torch.nn.Module):
    """A stand-in with real parameters. The ordering property under test is a property of
    EMATeacher.update and the runner's call sequence, not of any particular architecture."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(4, 3)


def main() -> int:
    from rl.retention_stabilizer import EMATeacher, RetentionRunner

    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")
    checks = []

    def check(name, passed, detail):
        checks.append({"check": name, "PASS": bool(passed), "detail": detail})
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {detail}")

    print(f"RSCFT PREFLIGHT ADDENDUM -- EMA ORDERING  {_now()}\n")

    torch.manual_seed(0)
    model = _TinyActor()
    teacher = EMATeacher(model, decay=DECAY)

    theta_pre = [p.detach().clone() for p in model.parameters()]
    bar0 = [p.detach().clone() for p in teacher.model.parameters()]

    # simulate an actor optimizer step: parameters change
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.1)
    theta_post = [p.detach().clone() for p in model.parameters()]

    teacher.update(model)
    bar1 = [p.detach().clone() for p in teacher.model.parameters()]

    expect_post = [DECAY * b + (1 - DECAY) * t for b, t in zip(bar0, theta_post)]
    expect_pre = [DECAY * b + (1 - DECAY) * t for b, t in zip(bar0, theta_pre)]
    matches_post = all(torch.allclose(a, b, atol=1e-12) for a, b in zip(bar1, expect_post))
    matches_pre = all(torch.allclose(a, b, atol=1e-12) for a, b in zip(bar1, expect_pre))
    pre_post_differ = not all(torch.allclose(a, b) for a, b in zip(theta_pre, theta_post))

    check("13a_EMA_consumes_POST_update_parameters",
          matches_post and not matches_pre and pre_post_differ,
          f"theta_bar_1 == decay*theta_bar_0 + (1-decay)*theta_POST: {matches_post}; "
          f"equals the theta_PRE form instead: {matches_pre}; "
          f"pre/post genuinely differ: {pre_post_differ}")

    src = inspect.getsource(RetentionRunner.note_ppo_minibatch)
    i_step = src.find("opt.step()")
    i_ema = src.find("self.teacher.update(")
    check("13b_runner_calls_optimizer_step_before_EMA_update",
          i_step != -1 and i_ema != -1 and i_step < i_ema,
          f"opt.step() at source offset {i_step}, teacher.update() at {i_ema} "
          f"(step must come first)")

    all_pass = all(c["PASS"] for c in checks)
    OUT.write_text(json.dumps({
        "record": "RSCFT preflight addendum -- EMA update ordering",
        "status": "FROZEN_RESULT", "utc": _now(),
        "amends": "RSCFT_PREFLIGHT.json (12/12), which was frozen before this 13th check was "
                  "raised; that record is not edited in place",
        "ordering_under_test": "theta_k -> actor optimizer step -> theta_{k+1} -> EMA update "
                               "-> theta_bar_{k+1}",
        "why_it_matters": "an EMA consuming PRE-update parameters would lag by exactly one "
                          "step and still show a healthy 1:1 update count, while quietly "
                          "regularising toward a stale target",
        "runtime_ordering_in_the_full_loop": "the updater calls "
            "retention_runner.note_ppo_minibatch(batch) AFTER the PPO actor minibatch update "
            "has completed; the runner then performs its own retention optimizer step and only "
            "then updates the EMA, so the teacher always consumes the most recent actor "
            "parameters",
        "VERDICT": "PASS" if all_pass else "FAIL",
        "checks": checks,
        "device": "cpu (deliberately, so this cannot contend with a training run for the GPU)",
    }, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {'PASS' if all_pass else 'FAIL'}")
    print(f"  -> {OUT}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
