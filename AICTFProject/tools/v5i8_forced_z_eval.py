#!/usr/bin/env python3
"""v5i8 forced-z latent strategy evaluation harness.

Pure evaluation. This script does not train, tune, backpropagate, or assign
semantic labels to z. It freezes a named checkpoint as the evaluated artifact,
runs natural-router and fixed-z rollout modes, and writes a manifest alongside
the qualitative rollout outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools import qualitative_rollout


def _read_last_csv_row(path: Path) -> dict[str, str]:
    import csv

    last: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            last = dict(row)
    return last


def _health_snapshot(metrics_csv: Path | None) -> dict[str, Any]:
    if metrics_csv is None:
        return {"metrics_csv": None, "available": False}
    if not metrics_csv.exists():
        return {"metrics_csv": str(metrics_csv), "available": False}
    row = _read_last_csv_row(metrics_csv)
    keys = [
        "timesteps",
        "mean_ep_return",
        "win_rate",
        "latent_sampled_z_occupancy_ratio",
        "latent_sampled_z_effective_num",
        "strategy_grad_norm",
        "main_loop_q_phi_grad_norm",
        "main_loop_q_phi_train_active",
        "q_phi_grad_norm",
        "latent_q_phi_train_active",
        "actor_z_jsd_mean",
        "actor_z_argmax_disagree",
        "forced_z_macro_jsd_mean",
    ]
    return {
        "metrics_csv": str(metrics_csv),
        "available": True,
        "last_row": {k: row.get(k, "") for k in keys if k in row},
    }


def _write_protocol_md(path: Path, manifest: dict[str, Any]) -> None:
    checkpoint = Path(str(manifest["checkpoint"])).name
    outputs = manifest.get("outputs", {})
    lines = [
        f"# v5i8 forced-z evaluation: `{checkpoint}`",
        "",
        "This is post-training evaluation only. It does not change the Summer-plan training objective and it does not train z meanings.",
        "",
        "## Protocol",
        "",
        "1. Train normally with MARL reward, persistence, and entropy/coverage.",
        "2. Check training health from the frozen metrics CSV.",
        "3. Freeze the checkpoint used here.",
        "4. Run natural-router rollout and fixed-z rollouts for z0..zK-1.",
        "5. Compare win rate, captures, pickups, score deltas, distances to flags, team spread, attack/defense ratio, q_phi probabilities, and positions.",
        "6. Interpret fixed-z behavior differences as the causal evidence surface.",
        "",
        "## Run",
        "",
        f"- checkpoint: `{manifest['checkpoint']}`",
        f"- map_layout: `{manifest['map_layout']}`",
        f"- opponents: `{', '.join(manifest['opponents'])}`",
        f"- episodes_per_mode: `{manifest['episodes_per_mode']}`",
        f"- deterministic: `{manifest['deterministic']}`",
        f"- seed: `{manifest['seed']}`",
        "",
        "## Outputs",
        "",
    ]
    for name, value in outputs.items():
        lines.append(f"- {name}: `{value}`")
    lines.extend(
        [
            "",
            "## Interpretation Gates",
            "",
            "Latent strategy evidence requires both forced-z behavioral differences and opponent-dependent performance or macro-behavior differences.",
            "Occupancy, entropy, and MI are health diagnostics, not causal strategy evidence.",
            "If forced z changes behavior metrics and trajectories for the same opponent, latent strategies emerged.",
            "If forced z changes only win rate or return, z affects performance but strategy meaning remains unclear.",
            "If forced z changes neither behavior nor outcome, z usage is cosmetic for that checkpoint.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Frozen custom PPO checkpoint .zip")
    parser.add_argument("--metrics-csv", default=None, help="Optional training metrics CSV for health snapshot")
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["OP5", "OP6", "OP7"],
        help="Opponent labels for the forced-z matrix.",
    )
    parser.add_argument(
        "--map-layout",
        default="map_b_split_lane_v2",
        choices=["map_a_open", "map_b_split_lane", "map_b_split_lane_v2", "open", "split_lane", "split_lane_v2"],
        help="Map layout for the evaluation environment.",
    )
    parser.add_argument("--episodes-per-mode", type=int, default=25)
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args(argv)

    ckpt = Path(args.checkpoint).expanduser().resolve()
    if not ckpt.suffix:
        ckpt = ckpt.with_suffix(".zip")
    if not ckpt.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt}")

    if args.agents is None:
        from rl.custom_ppo.inference import read_custom_ppo_metadata

        meta = read_custom_ppo_metadata(str(ckpt))
        agents = int(meta.get("n_blue", 4))
    else:
        agents = int(args.agents)

    out_dir = Path(args.out_dir).expanduser() if args.out_dir else ckpt.parent / "v5i8_forced_z_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = qualitative_rollout.run(
        checkpoint=ckpt,
        opponents=list(args.opponents),
        episodes_per_mode=int(args.episodes_per_mode),
        agents=agents,
        device=str(args.device),
        seed=int(args.seed),
        out_dir=out_dir,
        modes=["natural", "fixed_z"],
        deterministic=not bool(args.stochastic),
        max_steps=int(args.max_steps),
        map_layout=str(args.map_layout),
    )

    metrics_csv = Path(args.metrics_csv).expanduser().resolve() if args.metrics_csv else None
    manifest = {
        "protocol": "v5i8_forced_z_eval",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(ckpt),
        "metrics_health": _health_snapshot(metrics_csv),
        "map_layout": str(args.map_layout),
        "opponents": list(args.opponents),
        "episodes_per_mode": int(args.episodes_per_mode),
        "agents": agents,
        "device": str(args.device),
        "seed": int(args.seed),
        "deterministic": not bool(args.stochastic),
        "outputs": {name: str(path) for name, path in outputs.items()},
    }
    manifest_path = out_dir / f"{ckpt.stem}_v5i8_forced_z_manifest.json"
    protocol_path = out_dir / f"{ckpt.stem}_v5i8_forced_z_protocol.md"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_protocol_md(protocol_path, manifest)
    print(f"[v5i8_forced_z_eval] manifest: {manifest_path}")
    print(f"[v5i8_forced_z_eval] protocol: {protocol_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
