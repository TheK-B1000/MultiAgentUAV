#!/usr/bin/env python3
"""V6I26 target-KL early-stop ladder (actor-step 3× LR held fixed).

Scientific question
-------------------
2× under-floor and 3× over-ceiling showed nonlinear fixed-batch KL vs LR.
Holding the 3× actor LR fixed, checkpoint every 1 PPO update, and stop at the
first checkpoint whose init→ckpt fixed-batch KL lands in [1e-3, 1e-2].

Does **not** change: OP9/z0 surface, mixture, separate clip, critic LR,
architecture, reward, residual α, opponents, or router.

Classification: DIAGNOSTIC.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.v6i26_lro_core import (  # noqa: E402
    ACTOR_STEP_FIXED_BATCH_KL_MAX,
    ACTOR_STEP_FIXED_BATCH_KL_MIN,
    MARGIN_PILOT_LOCKED,
    select_target_kl_ladder_rung,
    target_kl_ladder_contract,
    write_json,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I26 actor-step target-KL ladder")
    p.add_argument(
        "--locked-target-json",
        default="artifacts/v6i26_margin_actor_step_3x_5u_seed1/current_response_target.json",
        help="Locked margin mixture / target JSON (causal surface lock).",
    )
    p.add_argument(
        "--checkpoint",
        default="artifacts/v6i23_population_birth_5u_seed1/final_v6i23_population_birth_5u_seed1_2v2.zip",
    )
    p.add_argument(
        "--output-dir",
        default="artifacts/v6i26_margin_actor_step_3x_kl_ladder_seed1",
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--n-envs", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=256)
    p.add_argument("--max-updates", type=int, default=5)
    p.add_argument("--z-actor-lr-mult", type=float, default=3.0)
    p.add_argument("--branch", type=int, default=0)
    p.add_argument(
        "--skip-post-eval",
        action="store_true",
        help="Skip strategic forced-z on the selected rung (movement-only).",
    )
    return p.parse_args()


def _mixture_cells(mixture: dict) -> list[tuple[str, str, float]]:
    cells: list[tuple[str, str, float]] = []
    for ctx, w in mixture.items():
        if "|" not in ctx:
            continue
        opp, mp = ctx.split("|", 1)
        cells.append((opp.upper(), mp, float(w)))
    return cells


def _probe_fixed_batch_kl(
    *,
    init_ckpt: Path,
    trained_ckpt: Path,
    out_json: Path,
    branch: int,
    device: str,
    seed: int,
) -> float:
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_v6i26_logit_control_authority_probe.py"),
        "--init-checkpoint",
        str(init_ckpt),
        "--trained-checkpoint",
        str(trained_ckpt),
        "--output",
        str(out_json),
        "--branch",
        str(branch),
        "--opponent",
        str(MARGIN_PILOT_LOCKED["opponent"]),
        "--map",
        str(MARGIN_PILOT_LOCKED["map"]),
        "--device",
        str(device),
        "--seed",
        str(seed),
        "--n-obs",
        "128",
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    probe = json.loads(out_json.read_text(encoding="utf-8"))
    return float(
        (
            probe.get("birth_graph_vs_trained")
            or probe.get("checkpoint_compare_init_vs_trained")
            or {}
        ).get("mean_kl")
        or 0.0
    )


def main() -> int:
    args = _parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    init_ckpt = Path(args.checkpoint)
    if not init_ckpt.is_file():
        print(f"ERROR: missing init checkpoint {init_ckpt}")
        return 2

    target = json.loads(Path(args.locked_target_json).read_text(encoding="utf-8"))
    branch = int(args.branch)
    locked = MARGIN_PILOT_LOCKED
    if branch != int(locked["branch"]):
        print(f"ERROR: ladder requires branch={locked['branch']}, got {branch}")
        return 2
    if str(target.get("branch_to_train_index")) not in {str(branch), str(float(branch))}:
        # tolerate int/float JSON; still require explicit lock match on context
        pass
    tgt_ctx = str(
        (target.get("target_contexts") or [None])[0]
        or target.get("target_context")
        or ""
    )
    if tgt_ctx and tgt_ctx != locked["target_context"]:
        print(
            f"ERROR: locked target mismatch: {tgt_ctx!r} != {locked['target_context']!r}"
        )
        return 2

    cells = _mixture_cells(dict(target.get("mixture_weights") or {}))
    if not cells:
        print("ERROR: empty locked mixture")
        return 2
    opponents = sorted({o for o, _, _ in cells})
    maps = sorted({m for _, m, _ in cells})

    contract = target_kl_ladder_contract(
        z_actor_lr_mult=float(args.z_actor_lr_mult),
        max_updates=int(args.max_updates),
        checkpoint_every_updates=1,
    )
    write_json(out_dir / "target_kl_ladder_contract.json", contract)
    write_json(out_dir / "locked_response_target.json", target)

    from rl.config.ppo_config import PPOConfig
    from rl.presets import PRESET_REGISTRY
    from rl.train_ppo import train_ppo

    steps_per_update = int(args.n_envs) * int(args.n_steps)
    load_path = init_ckpt
    rungs: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    stop_reason = "max_updates_exhausted"

    print("=" * 72)
    print("V6I26 target-KL ladder (3× LR held fixed)")
    print("=" * 72)
    print(
        f"init={init_ckpt}  max_u={args.max_updates}  "
        f"z_actor_lr_mult={args.z_actor_lr_mult:g}  "
        f"window=[{ACTOR_STEP_FIXED_BATCH_KL_MIN:g}, {ACTOR_STEP_FIXED_BATCH_KL_MAX:g}]",
        flush=True,
    )

    for update in range(1, int(args.max_updates) + 1):
        run_tag = f"v6i26_lro_kl_ladder_z{branch}_u{update}_seed{args.seed}"
        cfg = PRESET_REGISTRY["v6i26_actor_step"](PPOConfig())
        cfg.latent_lro_z_actor_lr_mult = float(args.z_actor_lr_mult)
        cfg.seed = int(args.seed) + 17 * branch
        cfg.device = str(args.device)
        cfg.n_envs = int(args.n_envs)
        cfg.n_steps = int(args.n_steps)
        cfg.load_path = str(load_path)
        cfg.load_weights_only = True
        cfg.additional_timesteps = int(steps_per_update)
        cfg.checkpoint_dir = str(out_dir)
        cfg.run_tag = run_tag
        cfg.fresh_metrics_csv = True
        cfg.fixed_latent_strategy = True
        cfg.fixed_latent_strategy_id = int(branch)
        cfg.training_cell_distribution = tuple(cells)
        cfg.opponent_pool = tuple(opponents)
        cfg.freeze_return_norm_after_load = True
        cfg.periodic_checkpoint_steps = 0

        print(
            f"[ladder] train update={update}/{args.max_updates} "
            f"from={load_path.name} (+{steps_per_update} steps)",
            flush=True,
        )
        train_ppo(cfg)

        finals = sorted(out_dir.glob(f"final_{run_tag}*.zip"))
        if not finals:
            print(f"ERROR: missing final checkpoint for update {update}")
            return 3
        ckpt_u = finals[-1]
        # Stable alias for the rung.
        alias = out_dir / f"ckpt_ladder_u{update}.zip"
        alias.write_bytes(ckpt_u.read_bytes())

        probe_path = out_dir / f"probe_u{update}.json"
        print(f"[ladder] probe fixed-batch KL for u={update} ...", flush=True)
        fixed_kl = _probe_fixed_batch_kl(
            init_ckpt=init_ckpt,
            trained_ckpt=ckpt_u,
            out_json=probe_path,
            branch=branch,
            device=str(args.device),
            seed=int(args.seed),
        )
        rung = {
            "update": update,
            "checkpoint": str(ckpt_u),
            "alias": str(alias),
            "fixed_batch_kl": fixed_kl,
            "in_window": bool(
                ACTOR_STEP_FIXED_BATCH_KL_MIN <= fixed_kl <= ACTOR_STEP_FIXED_BATCH_KL_MAX
            ),
            "above_ceiling": bool(fixed_kl > ACTOR_STEP_FIXED_BATCH_KL_MAX),
            "below_floor": bool(fixed_kl < ACTOR_STEP_FIXED_BATCH_KL_MIN),
        }
        rungs.append(rung)
        write_json(out_dir / "kl_ladder_rungs.json", {"rungs": rungs})
        print(
            f"  u={update} fixed_batch_kl={fixed_kl:.3e} "
            f"in_window={rung['in_window']} above_ceiling={rung['above_ceiling']}",
            flush=True,
        )

        decision = select_target_kl_ladder_rung(rungs)
        if decision["status"] == "SELECTED":
            selected = decision
            stop_reason = "selected_in_window"
            print(
                f"[ladder] SELECTED u={update} KL={fixed_kl:.3e} — early stop",
                flush=True,
            )
            break
        if decision["status"] == "OVERSHOOT_BEFORE_WINDOW":
            selected = decision
            stop_reason = "overshoot_before_window"
            print(
                f"[ladder] OVERSHOOT at u={update} KL={fixed_kl:.3e} — stop",
                flush=True,
            )
            break

        load_path = ckpt_u

    if selected is None:
        selected = select_target_kl_ladder_rung(rungs)

    report = {
        "protocol": "v6i26_actor_step_target_kl_ladder",
        "z_actor_lr_mult": float(args.z_actor_lr_mult),
        "branch": branch,
        "init_checkpoint": str(init_ckpt),
        "locked_target_json": str(args.locked_target_json),
        "kl_window": [ACTOR_STEP_FIXED_BATCH_KL_MIN, ACTOR_STEP_FIXED_BATCH_KL_MAX],
        "stop_reason": stop_reason,
        "selection": selected,
        "rungs": rungs,
        "maps": maps,
        "opponents": opponents,
    }
    write_json(out_dir / "kl_ladder_report.json", report)
    print(
        f"[ladder] status={selected.get('status')} "
        f"selected_update={selected.get('selected_update')} "
        f"stop_reason={stop_reason}",
        flush=True,
    )

    if selected.get("status") != "SELECTED":
        print(
            "[ladder] no in-window checkpoint — do not escalate LR; "
            "no strategic promotion path from this ladder.",
            flush=True,
        )
        return 0

    if args.skip_post_eval:
        print("[ladder] selected; skipping post-eval (--skip-post-eval)")
        return 0

    sel_ckpt = Path((selected.get("rung") or {}).get("checkpoint") or "")
    if not sel_ckpt.is_file():
        print(f"ERROR: selected checkpoint missing: {sel_ckpt}")
        return 4

    print(
        f"[ladder] strategic post-eval ONLY on selected u="
        f"{selected.get('selected_update')} ({sel_ckpt.name})",
        flush=True,
    )
    eval_dir = out_dir / "forced_z_selected"
    cmd = [
        "uv",
        "run",
        "python",
        "experiments/run_forced_z_eval.py",
        "--checkpoint",
        str(sel_ckpt),
        "--out-dir",
        str(eval_dir),
        "--inherit-training-config",
        "--episodes",
        "4",
        "--device",
        str(args.device),
        "--base-seed",
        str(args.seed),
        "--oracle-metric",
        "win_margin",
        "--max-decision-steps",
        "240",
        "--progress-every",
        "8",
        "--opponents",
        "OP11_ADAPTIVE_EXPLOITER",
        "OP12_LATE_CONVERTER",
        "OP7_DEEP_FORTRESS",
        "OP9_SPLIT_LANE_FEINT",
        "--maps",
        "map_b_split_lane",
        "map_b_split_lane_v2",
    ]
    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
    report["forced_z_selected_dir"] = str(eval_dir)
    write_json(out_dir / "kl_ladder_report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
