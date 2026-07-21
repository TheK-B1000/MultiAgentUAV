#!/usr/bin/env python3
"""V6I13 delayed-commit opening-window advantage-router diagnostic.

V6I13 changes when the router decides:

* steps 0..31 execute a uniformly sampled warmup latent,
* step 32 commits one router-selected latent,
* the committed latent is held to terminal,
* the external V/A model trains on post-commit return and
  ``opening_context = [state_0, state_commit, state_commit - state_0]``.

No labels, opponent-ID target, oracle-z target, or actor update is introduced.
The opponent one-hot remains an input feature for the diagnostic model, matching
V6I12's hard-pool setup.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.dump_router_rollout_audit import _build_audit_trainer  # noqa: E402
from rl.custom_ppo.diagnostics.arc_credit_smoke import frozen_actor_z_fingerprint  # noqa: E402
from rl.global_state import GLOBAL_STATE_DIM  # noqa: E402
from rl.router.advantage_router import (  # noqa: E402
    AdvantageRouter,
    ContextualVBaseline,
    advantage_gap_ci,
    advantage_matrix_from_replay,
    train_advantage_router,
)
from rl.router.q_value_router import (  # noqa: E402
    ArcIntegrityError,
    QRouterReplayBuffer,
    check_arc_guards,
    copy_arc_record,
    decide_verdict,
)

_PRESET = "v6i13_opening_window_advantage_router"
_N_OPPONENTS = 3
_OPPONENT_ID_TO_IDX = {7: 0, 8: 1, 9: 2}
_OPP_NAMES = {0: "OP8", 1: "OP9", 2: "OP10"}
_LATENT_K = 4
_OPENING_STATE_DIM = GLOBAL_STATE_DIM * 3
_CONTEXT_DIM = _OPENING_STATE_DIM + _N_OPPONENTS


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I13 opening-window advantage router")
    p.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/2v2/"
            "final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip"
        ),
    )
    p.add_argument("--n-updates", type=int, default=5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out-dir", default="artifacts/v6i13_opening_window_advantage_router")
    p.add_argument("--v-lr", type=float, default=3e-4)
    p.add_argument("--a-lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--replay-capacity", type=int, default=10_000)
    p.add_argument("--train-steps", type=int, default=100)
    p.add_argument("--spread-threshold", type=float, default=0.05)
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def build_opening_context_from_record(rec: dict) -> torch.Tensor:
    """Return ``[opening_summary, opponent_onehot]`` for one arc record."""
    raw = rec.get("opening_context")
    if raw is None:
        gs = rec["global_state_0"].float()
        raw = torch.cat([gs, gs, torch.zeros_like(gs)], dim=0)
    opening = raw.float().flatten()
    if opening.numel() < _OPENING_STATE_DIM:
        opening = torch.cat([opening, torch.zeros(_OPENING_STATE_DIM - opening.numel())])
    opening = opening[:_OPENING_STATE_DIM]
    opp = torch.zeros(_N_OPPONENTS, dtype=torch.float32)
    idx = _OPPONENT_ID_TO_IDX.get(int(rec.get("opponent_id", -1)))
    if idx is not None:
        opp[idx] = 1.0
    return torch.cat([opening, opp], dim=0)


def _extract_arc_records(trainer) -> list[dict]:
    ls = getattr(trainer, "latent_state", None)
    if ls is None:
        return []
    return list(getattr(ls, "rollout_strategy_arc_records", []))


def _abort_invalid(out_dir: Path, reason: str, update_idx: int, args, checkpoint) -> None:
    summary = {
        "preset": _PRESET,
        "checkpoint": str(checkpoint),
        "n_updates": args.n_updates,
        "routing_verdict": "INVALID",
        "invalid_reason": reason,
        "invalid_at_update": update_idx,
        "promotion_status": "NOT_A_CANDIDATE",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[v6i13] INVALID at update {update_idx}: {reason}")
    sys.exit(2)


def _spread_summary(adv_mat: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for i in range(adv_mat.shape[0]):
        row = adv_mat[i]
        valid = ~np.isnan(row)
        key = f"adv_spread_{_OPP_NAMES.get(i, i)}"
        out[key] = float(np.nanmax(row) - np.nanmin(row)) if valid.sum() >= 2 else float("nan")
    return out


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and not args.force and list(out_dir.glob("update_*.json")):
        print(f"[v6i13] Output dir {out_dir} already has results. Pass --force to overwrite.")
        sys.exit(0)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print("=" * 72)
    print(f"[v6i13] preset      = {_PRESET}")
    print(f"[v6i13] checkpoint  = {checkpoint}")
    print(f"[v6i13] n_updates   = {args.n_updates}")
    print(f"[v6i13] hidden      = {args.hidden}")
    print(f"[v6i13] train_steps = {args.train_steps}")
    print(f"[v6i13] out_dir     = {out_dir}")
    print("=" * 72)

    cfg, _resolved, _env, trainer = _build_audit_trainer(
        preset=_PRESET,
        checkpoint=str(checkpoint),
        device=args.device,
        seed=args.seed,
    )
    if not bool(getattr(trainer, "latent_arc_credit_enabled", False)):
        raise RuntimeError("latent_arc_credit_enabled is False; v6i13 cannot collect records")

    latent_k = int(getattr(cfg, "latent_k", _LATENT_K) or _LATENT_K)
    v_baseline = ContextualVBaseline(
        state_dim=_OPENING_STATE_DIM,
        n_opponents=_N_OPPONENTS,
        hidden=args.hidden,
    ).to(args.device)
    a_router = AdvantageRouter(
        state_dim=_OPENING_STATE_DIM,
        n_opponents=_N_OPPONENTS,
        opponent_id_to_idx=_OPPONENT_ID_TO_IDX,
        latent_k=latent_k,
        hidden=args.hidden,
    ).to(args.device)
    replay = QRouterReplayBuffer(
        capacity=args.replay_capacity,
        context_dim=_CONTEXT_DIM,
        latent_k=latent_k,
    )
    v_opt = torch.optim.Adam(v_baseline.parameters(), lr=args.v_lr)
    a_opt = torch.optim.Adam(a_router.parameters(), lr=args.a_lr)
    init_actor_hash = frozen_actor_z_fingerprint(trainer.model)

    updates: list[dict] = []
    total_arcs = 0
    for update_idx in range(1, args.n_updates + 1):
        t0 = time.time()
        print(f"[v6i13] --- update {update_idx}/{args.n_updates} ---")
        buffer = trainer.collect_rollout()
        arc_records = [copy_arc_record(r) for r in _extract_arc_records(trainer)]
        records_before = len(arc_records)
        push_stats = replay.push_many(
            arc_records,
            rollout_index=update_idx,
            opponent_id_to_idx=_OPPONENT_ID_TO_IDX,
            build_context=build_opening_context_from_record,
        )
        _ = trainer.update(buffer, total_timesteps=trainer.global_step)
        records_after = len(_extract_arc_records(trainer))

        try:
            check_arc_guards(
                records_before_update=records_before,
                inserted=push_stats["inserted"],
                size_before=push_stats["size_before"],
                size_after=push_stats["size_after"],
            )
        except ArcIntegrityError as exc:
            _abort_invalid(out_dir, str(exc), update_idx, args, checkpoint)
        if records_after != 0:
            _abort_invalid(out_dir, f"arc buffer not drained after update: {records_after}", update_idx, args, checkpoint)

        total_arcs += int(push_stats["inserted"])
        train_tel = train_advantage_router(
            v_baseline,
            a_router,
            replay,
            v_opt,
            a_opt,
            batch_size=min(256, len(replay)),
            n_steps=args.train_steps,
            device=args.device,
        )
        adv_mat, count_mat = advantage_matrix_from_replay(
            replay,
            v_baseline,
            n_opponents=_N_OPPONENTS,
            latent_k=latent_k,
            opponent_id_to_idx=_OPPONENT_ID_TO_IDX,
            device=args.device,
        )
        terminal_frac = (
            float(np.mean([str(r.get("reason", "")) == "episode_end" for r in arc_records]))
            if arc_records else float("nan")
        )
        commit_steps = [int(r.get("commit_step", -1)) for r in arc_records]
        update_record = {
            "update": update_idx,
            "records_before_update": records_before,
            "records_after_update": records_after,
            "replay_inserted": int(push_stats["inserted"]),
            "replay_duplicates_rejected": int(push_stats["duplicates_rejected"]),
            "replay_size": len(replay),
            "terminal_finalized_fraction": terminal_frac,
            "commit_step_min": int(min(commit_steps)) if commit_steps else -1,
            "commit_step_max": int(max(commit_steps)) if commit_steps else -1,
            "total_arcs": total_arcs,
            "count_by_z": replay.count_by_z(),
            **train_tel,
            **_spread_summary(adv_mat),
            **{f"count_OP{8+oi}_z{zi}": int(count_mat[oi, zi])
               for oi in range(_N_OPPONENTS) for zi in range(latent_k)},
            "elapsed_s": round(time.time() - t0, 1),
        }
        updates.append(update_record)
        (out_dir / f"update_{update_idx:04d}.json").write_text(json.dumps(update_record, indent=2))
        print(
            f"  arcs={records_before} inserted={push_stats['inserted']} replay={len(replay)} "
            f"commit=[{update_record['commit_step_min']},{update_record['commit_step_max']}] "
            f"r2={train_tel.get('baseline_r2_mean', float('nan')):+.4f} "
            f"adv_std={train_tel.get('advantage_target_std_mean', float('nan')):.4f}"
        )

    actor_ok = frozen_actor_z_fingerprint(trainer.model) == init_actor_hash
    gap_ci = advantage_gap_ci(
        replay,
        v_baseline,
        a_router,
        n_opponents=_N_OPPONENTS,
        latent_k=latent_k,
        opponent_id_to_idx=_OPPONENT_ID_TO_IDX,
        device=args.device,
    )
    adv_mat_final, count_mat_final = advantage_matrix_from_replay(
        replay,
        v_baseline,
        n_opponents=_N_OPPONENTS,
        latent_k=latent_k,
        opponent_id_to_idx=_OPPONENT_ID_TO_IDX,
        device=args.device,
    )
    spread_final = _spread_summary(adv_mat_final)
    validity = replay.validity_report(
        n_opponents=_N_OPPONENTS,
        latent_k=latent_k,
        opponent_id_to_idx=_OPPONENT_ID_TO_IDX,
    )
    min_cell_arcs = float(np.nanmin(count_mat_final[count_mat_final > 0])) if np.any(count_mat_final > 0) else 0.0
    verdict, separating = decide_verdict(
        validity=validity,
        gap_ci=gap_ci,
        spread={k.replace("adv_spread_", "spread_"): v for k, v in spread_final.items()},
        spread_threshold=args.spread_threshold,
        min_cell_arcs=min_cell_arcs,
        n_opponents=_N_OPPONENTS,
        opp_names=_OPP_NAMES,
    )
    if not actor_ok:
        verdict = "INVALID"
    promotion_status = "SEPARATING_CANDIDATE" if verdict in {"SEPARATING", "WEAK_SEPARATION"} else "NOT_A_CANDIDATE"

    torch.save(v_baseline.state_dict(), out_dir / "v_baseline_final.pt")
    torch.save(a_router.state_dict(), out_dir / "a_router_final.pt")
    summary = {
        "preset": _PRESET,
        "checkpoint": str(checkpoint),
        "n_updates": args.n_updates,
        "total_arcs": total_arcs,
        "routing_verdict": verdict,
        "promotion_status": promotion_status,
        "reliably_separating_opponents": int(separating),
        "reliability_gap_ci": gap_ci,
        "spread_final_adv": spread_final,
        "replay_validity": validity,
        "min_cell_arcs": min_cell_arcs,
        "frozen_actor_ok": bool(actor_ok),
        "updates": updates,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[v6i13] verdict={verdict} promotion={promotion_status} arcs={total_arcs}")
    print(f"[v6i13] Summary written -> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
