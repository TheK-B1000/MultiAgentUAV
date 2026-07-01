#!/usr/bin/env python3
"""Matched-seed communication corruption evaluation for V6I3 checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AICTF_DIR = os.path.dirname(_SCRIPT_DIR)
if _AICTF_DIR not in sys.path:
    sys.path.insert(0, _AICTF_DIR)

from rl.custom_ppo.communication.corruption import CommCorruptionMode
from rl.presets import apply_preset
from rl.config.ppo_config import PPOConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V6I3 communication corruption ablation report.")
    parser.add_argument("--checkpoint", required=True, help="Path to a V6I3 checkpoint (.zip).")
    parser.add_argument(
        "--modes",
        default="normal,silence,shuffle,random,extra_delay,constant",
        help="Comma-separated corruption modes.",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preset", default="v6i3")
    parser.add_argument("--output", default="", help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    modes = [CommCorruptionMode(m.strip()) for m in str(args.modes).split(",") if m.strip()]
    cfg = apply_preset(PPOConfig(), str(args.preset))
    cfg.seed = int(args.seed)
    if not bool(getattr(cfg, "communication_enabled", False)):
        raise SystemExit(f"Preset {args.preset!r} does not enable communication.")

    # Lightweight wiring report until full eval_rollout integration lands.
    report = {
        "checkpoint": str(args.checkpoint),
        "preset": str(args.preset),
        "experiment_id": str(getattr(cfg, "experiment_id", "")),
        "gate_protocol_version": str(getattr(cfg, "gate_protocol_version", "")),
        "comm_protocol_version": str(getattr(cfg, "comm_protocol_version", "")),
        "modes_requested": [m.value for m in modes],
        "episodes_per_mode": int(args.episodes),
        "status": "scaffold_ready",
        "note": (
            "Corruption runtime is wired on CommRolloutRuntime.set_corruption_mode; "
            "hook this script to eval_rollout for matched-seed win-rate tables."
        ),
    }
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)


if __name__ == "__main__":
    main()
