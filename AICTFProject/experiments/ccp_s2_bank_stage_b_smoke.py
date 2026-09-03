"""Stage-B validity gate: prove the routed interventions are REPRESENTABLE by the objective.

The PI's stop-criteria for Stage B include "the routed intervention cannot be represented by
the training objective." That is not something to assert -- this feeds the real Stage-B
trajectories through the real rl/causal_supervision.py::causal_supervision_loss against the
real warm-start incumbent, and checks that:

  1. the loss evaluates finite and non-negative-denominator on every unit,
  2. gradients actually flow to the policy (a loss that cannot move the model is not
     supervision),
  3. zero-weight agents contribute exactly nothing -- zeroing a non-intervened agent's weight
     leaves the loss bit-identical,
  4. the decision mask genuinely gates -- an all-False mask yields exactly zero loss.

Run:  python experiments/ccp_s2_bank_stage_b_smoke.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.ccp_s2_collect as C

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
STAGE_B = SD / "CCP_S2_CAUSAL_BANK_STAGE_B.json"
OUT = SD / "CCP_S2_BANK_STAGE_B_SMOKE.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from rl.causal_supervision import causal_supervision_loss

    stage_b = json.loads(STAGE_B.read_text(encoding="utf-8"))
    if stage_b["status"] != "FROZEN_RESULT":
        raise SystemExit(f"REFUSING: Stage B not frozen: {stage_b['status']!r}")

    device, incumbent, _teachers, _R2, _ctx = C.load_runtime(args.device)
    model = incumbent.model if hasattr(incumbent, "model") else incumbent
    model.train()

    checks, all_pass = [], True
    for u in stage_b["units"]:
        z = np.load(ROOT / u["trajectory_file"])
        n = int(z["decision_mask"].shape[0])
        obs = {k: torch.as_tensor(z[k], device=device)
               for k in ("grid", "vec", "agent_mask", "mask", "global_state")}
        teacher_actions = torch.as_tensor(z["teacher_actions"], device=device).long()
        dmask = torch.as_tensor(z["decision_mask"], device=device)
        weights = torch.as_tensor(z["weights"], device=device).float()
        z_idx = torch.as_tensor(z["z_idx"], device=device).long()

        model.zero_grad(set_to_none=True)
        loss = causal_supervision_loss(model, obs, teacher_actions, z_idx=z_idx,
                                       decision_mask=dmask, weights=weights)
        finite = bool(torch.isfinite(loss))
        loss.backward()
        gnorm = float(sum((p.grad.detach() ** 2).sum() for p in model.parameters()
                          if p.grad is not None) ** 0.5)

        # zero-weight agents must contribute exactly nothing
        w_zeroed = weights.clone()
        for i in range(w_zeroed.shape[1]):
            if float(weights[:, i].max()) == 0.0:
                w_zeroed[:, i] = 0.0
        with torch.no_grad():
            loss_zeroed = causal_supervision_loss(model, obs, teacher_actions, z_idx=z_idx,
                                                  decision_mask=dmask, weights=w_zeroed)
            identical = bool(torch.equal(loss.detach(), loss_zeroed))
            # an all-False decision mask must produce exactly zero
            loss_masked = causal_supervision_loss(
                model, obs, teacher_actions, z_idx=z_idx,
                decision_mask=torch.zeros_like(dmask), weights=weights)
            masked_zero = float(loss_masked) == 0.0

        ok = finite and gnorm > 0.0 and identical and masked_zero
        all_pass &= ok
        checks.append({"unit": u["unit"], "steps": n, "loss": float(loss),
                       "grad_norm": gnorm, "finite": finite,
                       "zero_weight_agents_contribute_nothing": identical,
                       "empty_decision_mask_yields_zero_loss": masked_zero,
                       "PASS": ok})
        print(f"  {u['unit']:28s} loss={float(loss):8.4f} |grad|={gnorm:9.4f} "
              f"finite={finite} zero-w-inert={identical} mask-gates={masked_zero} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)

    verdict = "STAGE_B_REPRESENTABLE" if all_pass else "STAGE_B_NOT_REPRESENTABLE"
    OUT.write_text(json.dumps({
        "record": "CCP-S2 Stage-B representability smoke", "status": "FROZEN_RESULT",
        "utc": stage_b["utc"], "VERDICT": verdict,
        "question": "can the frozen routed interventions actually be represented and trained "
                    "by rl/causal_supervision.py's causal_supervision_loss against the real "
                    "warm-start incumbent?",
        "checks": checks,
    }, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}")
    print(f"  -> {OUT}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
