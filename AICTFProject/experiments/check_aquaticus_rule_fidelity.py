#!/usr/bin/env python3
"""Aquaticus-style rule fidelity checklist for map_a (diagnostic only).

Before concluding that G0's universal competence means the task lacks
strategic tension, verify the simulator still implements the mechanics that
create same-map commitment costs. Fixing missing mechanics is fidelity work,
not cherry-picking a C1.

Hard checks probe the real Aquaticus path in gpu_env (not synonym attr names):

  territory-dependent tagging   (_is_on_home_side gates in _rules.py)
  tag channel / pressure time   (tag_channel_seconds + pressure >= 2)
  return-home to untag          (_untag_if_home + home_untag_radius_cells)
  flag carry / capture scoring  (carrying + score fields)
  vehicle motion limits         (max_speed/accel/yaw under aquaticus_profile)
  finite match horizon          (max_decision_steps)

Soft notes (do not fail alone): classic discrete "tag cooldown" may be
implemented as return-home + tag-channel rather than a named cooldown field.

Exit 0 if all hard checks pass; exit 2 if any hard check fails.
"""
from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="artifacts/g0_weakness_sweep/rule_fidelity.json")
    args = p.parse_args()

    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
    from gpu_env._core import _rules as rules_mod
    from gpu_env._core import _dynamics as dyn_mod

    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=2, max_red_agents=2,
        map_set="train", map_layout="map_a",
        max_decision_steps=240,
        aquaticus_profile=True, rules_profile="OURS",
        device=args.device, seed=42,
    )
    env = GPUCTFVecEnv(cfg)
    report: dict = {"map": "map_a", "resolved_map": "map_a_open", "checks": {},
                    "soft_notes": [], "hard_fail": []}
    try:
        core = env.unwrapped if hasattr(env, "unwrapped") else env
        # Drill into BatchedCTFCore when wrapped.
        for attr in ("core", "_core", "envs"):
            if hasattr(core, attr):
                candidate = getattr(core, attr)
                if hasattr(candidate, "blue_tagged") or hasattr(candidate, "cfg"):
                    core = candidate
                    break
        if hasattr(core, "__getitem__"):
            try:
                core = core[0]
            except Exception:
                pass

        rules_src = inspect.getsource(rules_mod._RulesMixin._apply_aquaticus_tag_rules)
        untag_src = inspect.getsource(rules_mod._RulesMixin._untag_if_home)
        dyn_src = inspect.getsource(dyn_mod)

        checks = {
            "max_decision_steps_finite": int(cfg.max_decision_steps) > 0,
            "aquaticus_profile": bool(cfg.aquaticus_profile),
            "territory_tagging_home_side_gate": (
                "_is_on_home_side" in rules_src and "blue_can_tag" in rules_src
            ),
            "tag_channel_pressure": (
                "tag_channel_seconds" in rules_src and "pressure" in rules_src
            ),
            "return_home_to_untag": (
                "home_untag_radius_cells" in untag_src and "blue_tagged" in untag_src
            ),
            "tag_range_configured": float(getattr(cfg, "tag_range_cells", 0) or 0) > 0,
            "motion_limits_configured": (
                float(cfg.max_speed_cps) > 0
                and float(cfg.max_accel_cps2) > 0
                and float(cfg.max_yaw_rate_rps) > 0
            ),
            "aquaticus_caps_motion": (
                "max_accel_cps2" in dyn_src and "aquaticus_profile" in dyn_src
            ),
            "flag_carry_fields": hasattr(core, "blue_carrying") or hasattr(cfg, "score_limit"),
        }
        report["checks"] = {k: bool(v) for k, v in checks.items()}
        report["hard_fail"] = [k for k, v in checks.items() if not v]
        report["cfg_snapshot"] = {
            "max_decision_steps": int(cfg.max_decision_steps),
            "aquaticus_profile": bool(cfg.aquaticus_profile),
            "rules_profile": str(cfg.rules_profile),
            "map_layout": str(cfg.map_layout),
            "tag_range_cells": float(cfg.tag_range_cells),
            "home_untag_radius_cells": float(cfg.home_untag_radius_cells),
            "tag_channel_seconds": float(getattr(cfg, "tag_channel_seconds", float("nan"))),
            "max_speed_cps": float(cfg.max_speed_cps),
            "max_accel_cps2": float(cfg.max_accel_cps2),
            "max_yaw_rate_rps": float(cfg.max_yaw_rate_rps),
        }
        report["soft_notes"].append(
            "No discrete named tag_cooldown field; commitment cost is "
            "return-home-to-untag + tag_channel pressure time."
        )
    finally:
        env.close()

    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    print("Aquaticus rule-fidelity checklist (map_a)")
    for k, v in report["checks"].items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}")
    for note in report["soft_notes"]:
        print(f"  NOTE  {note}")
    if report["hard_fail"]:
        print(f"\nHARD FAILS: {report['hard_fail']}")
        print(f"wrote {out}")
        return 2
    print(f"\nAll hard probes present (magnitude not validated). wrote {out}")
    print("Next after variant synonym pass: same-map scenario bank S1-S6.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
