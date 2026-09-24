"""Tiny smoke: log (agent_id, d_i, r_i, role_age) over reassignment cycles.

Does NOT train. Exercises RoleHoldState against a live BatchedCTFCore.
Automated contracts in tests/test_rule_role_conditioning.py remain authoritative.

Run:
  python experiments/smoke_rule_role_conditioning.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from game_field_gpu import BatchedCTFCore, GPUFieldConfig
from rl.custom_ppo.rule_role_assignment import RoleHoldState, roles_from_core


def main() -> int:
    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=4,
        max_red_agents=4,
        device="cpu",
        max_decision_steps=64,
        stalemate_max_steps=64,
        map_layout="map_a_open",
    )
    core = BatchedCTFCore(cfg)
    core.reset_all()
    hold = RoleHoldState(1, 4, hold_ticks=8, device="cpu")

    print("tick agent_id d_i r_i role_age")
    for t in range(24):
        roles = roles_from_core(core, hold, force=(t == 0))
        # Advance env with zeros (scripted blue optional).
        act = torch.zeros((1, 8), dtype=torch.int64)
        core.step(act)
        d = hold.distances[0]
        age = int(hold.age[0].item())
        for i in range(4):
            print(
                f"{t:4d} {i:8d} {float(d[i]):8.3f} {int(roles[0, i].item()):4d} {age:8d}"
            )
        if t == 10:
            # Force a death to exercise immediate reassignment.
            core.blue_alive[0, 0] = False
            print("# injected death of agent 0")
    print("OK smoke_rule_role_conditioning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
