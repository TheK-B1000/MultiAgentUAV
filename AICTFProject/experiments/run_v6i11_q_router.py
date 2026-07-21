#!/usr/bin/env python3
"""V6I11 contextual Q-value router experiment.

Replaces BPTT PPO routing with direct return regression over a replay buffer.
The frozen v6i9 actor is never touched; only the external ContextualQRouter MLP
is trained.

Algorithm
---------
1. Build trainer with preset ``v6i11_q_router_hardpool``:
   - Actor + adapters frozen.
   - GRU router still exists but all gradient losses are zeroed
     (latent_strategy_ppo_coef=0, router_ent_coef=0, latent_lam_p=0,
      latent_arc_credit_coef=0).
   - Arc-credit data collection ENABLED (latent_arc_credit_enabled=True).
   - 50 % uniform exploration (router_uniform_exploration_prob=0.5).

2. Per update:
   a. Run standard rollout (actor frozen, 50/50 epsilon-greedy z selection).
   b. Skip PPO router update (all router losses zero).
   c. Extract arc records: global_state_0, z, arc_return, opponent_id.
   d. Build enriched context: geometry (35d) + opponent onehot (3d).
   e. Push to QRouterReplayBuffer (ring, capacity 10k).
   f. Train ContextualQRouter for 20 Huber-loss steps.
   g. Log: arc count, Q-loss, Q-matrix (opp x z mean return), row spread.

3. Decision gate (configurable via --n-updates):
   - SEPARATING: >=2 opponents show a best-vs-second-best mean-return gap whose
     bootstrap CI excludes zero AND row-spread >= threshold (magnitude AND
     reliability, not spread alone).
   - WEAK_SEPARATION: exactly 1 opponent passes.
   - FLAT: no usable separation UNDER THIS dataset/horizon/context/budget. This
     does NOT re-open repertoire diversity (already proven via counterfactual
     actor logits, forced-z separation, +2.37 oracle gap); it means the current
     Q-formulation failed to resolve the latents.
   - INSUFFICIENT_SAMPLES: replay validity failed, or <20 arcs in the smallest cell.
   The decisive promotion gate is the held-out prospective test (Q-router > cross-
   episode-shuffled-Q, then > uniform, then approaching fixed_z2); it is a
   separate post-training step, NOT computed here.

Target horizon: this preset inherits v6i10's EPISODE-PERSISTENT contract, so each
arc == one episode (episode-start context, total episode return), matching Probe A
and the forced-z oracle. arc_length telemetry per update confirms this.

Example
-------
    uv run python experiments/run_v6i11_q_router.py \\
      --checkpoint checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip \\
      --n-updates 15 --device cuda --seed 1 \\
      --out-dir artifacts/v6i11_q_router
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
from rl.router.q_value_router import (  # noqa: E402
    ArcIntegrityError,
    ContextualQRouter,
    QRouterReplayBuffer,
    check_arc_guards,
    copy_arc_record,
    decide_verdict,
    train_q_router,
)
from rl.custom_ppo.diagnostics.arc_credit_smoke import (  # noqa: E402
    frozen_actor_z_fingerprint,
    router_fingerprint,
)
from rl.global_state import GLOBAL_STATE_DIM  # noqa: E402

_PRESET = "v6i11_q_router_hardpool"
_N_OPPONENTS = 3
# Arc records stamp the canonical ``_opponent_id_int_from_info`` id, which maps
# OP8->7, OP9->8, OP10->9 (see ``rl/custom_ppo/csv_writers.py::_OPPONENT_TAG_TO_ID``,
# scheme OP_N -> N-1). These raw ids must map to contiguous one-hot rows here.
_OPPONENT_ID_TO_IDX = {7: 0, 8: 1, 9: 2}
_LATENT_K = 4
_REPLAY_CAPACITY = 10_000
_Q_BATCH_SIZE = 256
_Q_TRAIN_STEPS = 20


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I11 Q-value router diagnostic")
    p.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/2v2/"
            "final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip"
        ),
    )
    p.add_argument("--n-updates", type=int, default=15)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out-dir", default="artifacts/v6i11_q_router")
    p.add_argument("--q-lr", type=float, default=3e-4)
    p.add_argument("--q-hidden", type=int, default=128)
    p.add_argument("--replay-capacity", type=int, default=_REPLAY_CAPACITY)
    p.add_argument("--q-train-steps", type=int, default=_Q_TRAIN_STEPS)
    p.add_argument(
        "--spread-threshold",
        type=float,
        default=0.10,
        help="Minimum Q-value row-spread (max-min) to declare routing learnable",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a previously completed output directory",
    )
    return p.parse_args()


def _extract_arc_records(trainer) -> list[dict]:
    """Pull completed arc records from the trainer's latent state."""
    ls = getattr(trainer, "latent_state", None)
    if ls is None:
        return []
    return list(getattr(ls, "rollout_strategy_arc_records", []))


def _spread_summary(q_matrix_np: np.ndarray) -> dict[str, float]:
    """Row-wise max-min spread of the Q-value matrix."""
    out: dict[str, float] = {}
    opp_names = {0: "OP8", 1: "OP9", 2: "OP10"}
    for i in range(q_matrix_np.shape[0]):
        row = q_matrix_np[i]
        valid = ~np.isnan(row)
        if valid.sum() >= 2:
            out[f"spread_{opp_names.get(i, i)}"] = float(np.nanmax(row) - np.nanmin(row))
        else:
            out[f"spread_{opp_names.get(i, i)}"] = float("nan")
    return out


def _abort_invalid(out_dir, reason: str, update_idx: int, args, checkpoint) -> None:
    """Write an INVALID summary and abort — never emit FLAT on a broken pipeline."""
    print()
    print("=" * 72)
    print(f"[v6i11] INVALID at update {update_idx}: {reason}")
    print("[v6i11] Aborting — a broken arc pipeline is a tooling failure, not")
    print("[v6i11] evidence about the repertoire.  No FLAT verdict is emitted.")
    print("=" * 72)
    summary = {
        "preset": _PRESET,
        "checkpoint": str(checkpoint),
        "n_updates": args.n_updates,
        "routing_verdict": "INVALID",
        "invalid_reason": reason,
        "invalid_at_update": update_idx,
        "promotion_status": "NOT_A_CANDIDATE",
    }
    (Path(out_dir) / "summary.json").write_text(json.dumps(summary, indent=2))
    sys.exit(2)


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and not args.force:
        existing = list(out_dir.glob("update_*.json"))
        if existing:
            print(f"[v6i11] Output dir {out_dir} already has results. Pass --force to overwrite.")
            sys.exit(0)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print("=" * 72)
    print(f"[v6i11] preset       = {_PRESET}")
    print(f"[v6i11] checkpoint   = {checkpoint}")
    print(f"[v6i11] n_updates    = {args.n_updates}")
    print(f"[v6i11] device       = {args.device}")
    print(f"[v6i11] out_dir      = {out_dir}")
    print(f"[v6i11] q_lr         = {args.q_lr}")
    print(f"[v6i11] q_hidden     = {args.q_hidden}")
    print(f"[v6i11] replay_cap   = {args.replay_capacity}")
    print(f"[v6i11] q_steps/upd  = {args.q_train_steps}")
    print("=" * 72)

    # Build trainer — use load_weights_only=True (the v6i11 preset doesn't need
    # optimizer state; we're not continuing a BPTT training session).
    cfg, resolved, env, trainer = _build_audit_trainer(
        preset=_PRESET,
        checkpoint=str(checkpoint),
        device=args.device,
        seed=args.seed,
    )

    latent_k = int(getattr(cfg, "latent_k", _LATENT_K) or _LATENT_K)
    context_dim = GLOBAL_STATE_DIM + _N_OPPONENTS

    # Instantiate the Q-router and replay buffer (external to the PPO trainer).
    q_router = ContextualQRouter(
        state_dim=GLOBAL_STATE_DIM,
        n_opponents=_N_OPPONENTS,
        opponent_id_to_idx=_OPPONENT_ID_TO_IDX,
        latent_k=latent_k,
        hidden=args.q_hidden,
    ).to(args.device)

    replay = QRouterReplayBuffer(
        capacity=args.replay_capacity,
        context_dim=context_dim,
        latent_k=latent_k,
    )

    q_optimizer = torch.optim.Adam(q_router.parameters(), lr=args.q_lr)

    # Fingerprint initial state.
    init_router_hash = router_fingerprint(trainer.model)
    init_actor_hash = frozen_actor_z_fingerprint(trainer.model)
    print(f"[v6i11] initial router hash      : {init_router_hash}")
    print(f"[v6i11] initial frozen actor hash: {init_actor_hash}")
    print()

    all_updates: list[dict] = []
    total_arcs = 0

    for update_idx in range(1, args.n_updates + 1):
        t0 = time.time()
        print(f"[v6i11] --- update {update_idx}/{args.n_updates} ---")

        # ------------------------------------------------------------------ #
        # 1. Run one rollout.  All router losses are zeroed by the preset, so
        #    the point is only to run the environment and collect arc records.
        # ------------------------------------------------------------------ #
        buffer = trainer.collect_rollout()

        # ------------------------------------------------------------------ #
        # 2. Copy finalized arc records and push them into replay — BEFORE the
        #    PPO update.  ``trainer.update`` runs post_update, which drains
        #    ``rollout_strategy_arc_records`` via reset_arc_credit_rollout_state();
        #    reading after update() would yield an EMPTY list every step and
        #    silently train the Q-router on zero arcs.  copy_arc_record clones
        #    the tensors so the later reset cannot mutate the copies.
        # ------------------------------------------------------------------ #
        records_before_update = len(_extract_arc_records(trainer))
        arc_records = [copy_arc_record(r) for r in _extract_arc_records(trainer)]
        n_arcs = len(arc_records)
        total_arcs += n_arcs

        size_before = len(replay)
        push_stats = replay.push_many(
            arc_records,
            rollout_index=update_idx,
            opponent_id_to_idx=_OPPONENT_ID_TO_IDX,
            build_context=q_router.build_context_from_record,
        )
        size_after = len(replay)

        # Now run the (no-op) PPO update for bookkeeping / global_step advance.
        # This drains the arc buffer we already copied above.
        _ = trainer.update(buffer, total_timesteps=trainer.global_step)
        records_after_update = len(_extract_arc_records(trainer))

        arc_lengths = [int(r.get("arc_length", 0)) for r in arc_records]
        terminal_frac = (
            float(np.mean([str(r.get("reason", "")) == "episode_end" for r in arc_records]))
            if arc_records else float("nan")
        )
        print(
            f"  integrity: records_before={records_before_update} "
            f"records_after={records_after_update} "
            f"inserted={push_stats['inserted']} "
            f"dup_rejected={push_stats['duplicates_rejected']} "
            f"replay {size_before}->{size_after}"
        )
        if arc_lengths:
            arc_len_arr = np.asarray(arc_lengths, dtype=float)
            print(
                f"  arc_length: mean={arc_len_arr.mean():.1f} "
                f"median={np.median(arc_len_arr):.0f} "
                f"min={arc_len_arr.min():.0f} max={arc_len_arr.max():.0f}  "
                f"terminal_finalized={terminal_frac:.3f}"
            )

        # ---- HARD GUARDS: abort (never emit FLAT) on a broken pipeline. ---- #
        try:
            check_arc_guards(
                records_before_update=records_before_update,
                inserted=push_stats["inserted"],
                size_before=size_before,
                size_after=size_after,
            )
        except ArcIntegrityError as exc:
            _abort_invalid(out_dir, str(exc), update_idx, args, checkpoint)
        if records_after_update != 0:
            _abort_invalid(
                out_dir,
                f"arc buffer not drained after update (records_after={records_after_update}); "
                "extraction/ordering assumption violated",
                update_idx, args, checkpoint,
            )

        if True:
            print(f"  arc_records: {n_arcs}  (replay size: {len(replay)})")

            # ------------------------------------------------------------------ #
            # 3. Train Q-router from replay.
            # ------------------------------------------------------------------ #
            q_tel = train_q_router(
                q_router,
                replay,
                q_optimizer,
                batch_size=min(_Q_BATCH_SIZE, len(replay)),
                n_steps=args.q_train_steps,
                device=args.device,
            )
            print(f"  q_loss: {q_tel['q_loss_mean']:.5f}  "
                  f"q_grad_norm: {q_tel.get('q_grad_norm', 0):.4f}  "
                  f"q_steps: {q_tel.get('q_steps', 0)}")

            # ------------------------------------------------------------------ #
            # 4. Log Q-value matrix and row spread.
            # ------------------------------------------------------------------ #
            mean_mat, count_mat = replay.mean_return_matrix(
                n_opponents=_N_OPPONENTS,
                latent_k=latent_k,
                opponent_id_to_idx=_OPPONENT_ID_TO_IDX,
            )
            spread = _spread_summary(mean_mat)

            # Q-router's learned matrix (zero-geometry context, only opp onehot).
            with torch.no_grad():
                q_matrix_learned = q_router.q_matrix(device=args.device).cpu().numpy()

            print("  empirical mean arc return (opp × z):")
            opp_labels = {0: "OP8", 1: "OP9", 2: "OP10"}
            for oi in range(_N_OPPONENTS):
                cells = []
                for zi in range(latent_k):
                    v = mean_mat[oi, zi]
                    n = int(count_mat[oi, zi])
                    cells.append(f"{v:+.3f}(n={n:3d})" if not np.isnan(v) else "  nan( n=  0)")
                sp = spread.get(f"spread_{opp_labels[oi]}", float("nan"))
                print(f"    {opp_labels[oi]}: {' '.join(cells)}  spread={sp:+.4f}")

            print("  Q-router learned values (zero-geometry context):")
            for oi in range(_N_OPPONENTS):
                row = q_matrix_learned[oi]
                print(f"    {opp_labels[oi]}: {' '.join(f'{v:+.4f}' for v in row)}")

            # Decision diagnostic.
            max_spread = max(
                (v for v in spread.values() if not np.isnan(v)), default=float("nan")
            )
            min_arcs_per_cell = float(
                np.nanmin(count_mat[count_mat > 0]) if np.any(count_mat > 0) else 0
            )
            print(f"  max_row_spread={max_spread:+.4f}  "
                  f"min_cell_arcs={min_arcs_per_cell:.0f}  "
                  f"total_arcs={total_arcs}")

            # Replay validity snapshot (count-by-z + duplicate guard).
            cnt_by_z = replay.count_by_z()
            print(f"  count_by_z: {cnt_by_z}  "
                  f"dup_rejected={replay.duplicates_rejected}/{replay.total_offered}")

            # Coverage gate: by update >= 3 all four z must be represented; severe
            # starvation before the Q-router can even estimate a cell is INVALID.
            if update_idx >= 3 and not all(cnt_by_z.get(zi, 0) > 0 for zi in range(latent_k)):
                missing = [zi for zi in range(latent_k) if cnt_by_z.get(zi, 0) == 0]
                print(f"  [warn] z starvation by update {update_idx}: missing z={missing}")

            update_record = {
                "update": update_idx,
                "n_arcs_this_update": n_arcs,
                "total_arcs": total_arcs,
                "replay_size": len(replay),
                "records_before_update": int(records_before_update),
                "records_after_update": int(records_after_update),
                "replay_inserted": int(push_stats["inserted"]),
                "replay_duplicates_rejected": int(push_stats["duplicates_rejected"]),
                "arc_length_mean": float(np.mean(arc_lengths)) if arc_lengths else float("nan"),
                "terminal_finalized_fraction": float(terminal_frac),
                "count_by_z": cnt_by_z,
                **q_tel,
                **{f"mean_return_OP{8+oi}_z{zi}": float(mean_mat[oi, zi])
                   for oi in range(_N_OPPONENTS) for zi in range(latent_k)},
                **{f"count_OP{8+oi}_z{zi}": int(count_mat[oi, zi])
                   for oi in range(_N_OPPONENTS) for zi in range(latent_k)},
                **{f"q_learned_OP{8+oi}_z{zi}": float(q_matrix_learned[oi, zi])
                   for oi in range(_N_OPPONENTS) for zi in range(latent_k)},
                **{f"spread_{k}": float(v) for k, v in spread.items()},
                "max_row_spread": float(max_spread),
                "elapsed_s": round(time.time() - t0, 1),
            }
            all_updates.append(update_record)
            (out_dir / f"update_{update_idx:04d}.json").write_text(
                json.dumps(update_record, indent=2)
            )

    # Frozen-actor sanity check.
    final_actor_hash = frozen_actor_z_fingerprint(trainer.model)
    actor_ok = final_actor_hash == init_actor_hash
    print()
    print(f"[v6i11] Frozen-actor check : {'PASS' if actor_ok else 'FAIL'}")
    print(f"[v6i11] Total arc records  : {total_arcs}")

    # Final Q-matrix printout.
    print()
    print(q_router.q_matrix_summary(replay))

    # ------------------------------------------------------------------ #
    # Separation verdict.
    #
    # The verdict describes ONLY what the online replay dataset supports.  It
    # deliberately does NOT re-litigate repertoire diversity, which was already
    # established by counterfactual actor-logit differences, forced-z behavioural
    # separation, and the +2.37 oracle gap.  Semantics:
    #
    #   SEPARATING       online context+targets support value-based routing:
    #                    >=2 opponents show a best-vs-second mean-return gap whose
    #                    bootstrap CI excludes zero AND spread >= threshold.
    #   WEAK_SEPARATION  partial predictive structure: exactly 1 opponent passes.
    #   FLAT             no usable separation learned UNDER THIS dataset, target
    #                    horizon, context, and training budget.  This means the
    #                    current Q-formulation failed to resolve the latents — NOT
    #                    that the latents do not differ.
    #
    # A raw Q-spread alone is NOT sufficient (a network can invent small spreads):
    # magnitude (spread) AND reliability (CI excludes zero) are both required, and
    # the decisive gate remains the held-out prospective test (not run here).
    # ------------------------------------------------------------------ #
    mean_mat_final, count_mat_final = replay.mean_return_matrix(
        n_opponents=_N_OPPONENTS,
        latent_k=latent_k,
        opponent_id_to_idx=_OPPONENT_ID_TO_IDX,
    )
    spread_final = _spread_summary(mean_mat_final)
    gap_ci = replay.best_second_gap_ci(
        n_opponents=_N_OPPONENTS,
        latent_k=latent_k,
        opponent_id_to_idx=_OPPONENT_ID_TO_IDX,
    )
    validity = replay.validity_report(
        n_opponents=_N_OPPONENTS,
        latent_k=latent_k,
        opponent_id_to_idx=_OPPONENT_ID_TO_IDX,
    )

    # Predicted-Q matrix (zero-geometry context) + empirical best-z agreement.
    opp_names = {0: "OP8", 1: "OP9", 2: "OP10"}
    with torch.no_grad():
        q_pred_mat = q_router.q_matrix(device=args.device).cpu().numpy()
    q_pred_spread = {
        f"spread_{opp_names.get(oi, oi)}": float(np.max(q_pred_mat[oi]) - np.min(q_pred_mat[oi]))
        for oi in range(_N_OPPONENTS)
    }
    best_z_agreement = {}
    for oi in range(_N_OPPONENTS):
        emp_row = mean_mat_final[oi]
        if np.all(np.isnan(emp_row)):
            best_z_agreement[opp_names.get(oi, oi)] = None
        else:
            emp_best = int(np.nanargmax(emp_row))
            q_best = int(np.argmax(q_pred_mat[oi]))
            best_z_agreement[opp_names.get(oi, oi)] = bool(emp_best == q_best)

    min_cell_arcs = float(
        np.nanmin(count_mat_final[count_mat_final > 0])
        if np.any(count_mat_final > 0) else 0
    )

    verdict, reliably_separating = decide_verdict(
        validity=validity,
        gap_ci=gap_ci,
        spread=spread_final,
        spread_threshold=args.spread_threshold,
        min_cell_arcs=min_cell_arcs,
        n_opponents=_N_OPPONENTS,
        opp_names=opp_names,
    )
    # A drifted frozen repertoire invalidates the whole experiment regardless
    # of what the replay stats look like.
    if not actor_ok:
        verdict = "INVALID"

    print()
    print("=" * 72)
    print("[v6i11] Replay validity")
    print(f"  all_z_represented        : {validity['all_z_represented']}  "
          f"count_by_z={validity['count_by_z']}")
    print(f"  all_opponents_represented: {validity['all_opponents_represented']}  "
          f"count_by_opponent={validity['count_by_opponent']}")
    print(f"  return_variance_nonzero  : {validity['return_variance_nonzero']}  "
          f"(var={validity['return_variance']:.4f})")
    print(f"  no_duplicate_arcs        : {validity['no_duplicate_arcs']}  "
          f"(rejected={validity['duplicates_rejected']}/{validity['total_offered']})")
    print(f"  terminal_finalized_frac  : {validity['terminal_finalized_fraction']:.3f}  "
          f"mean_arc_length={validity['mean_arc_length']:.1f}")
    print(f"  map_coverage             : {validity['map_coverage']}")
    print()
    print(f"[v6i11] Spread threshold   : {args.spread_threshold:.3f}")
    for oi in range(_N_OPPONENTS):
        name = opp_names.get(oi, str(oi))
        g = gap_ci.get(name, {})
        sp = spread_final.get(f"spread_{name}", float("nan"))
        qsp = q_pred_spread.get(f"spread_{name}", float("nan"))
        pc = validity["per_cell"]
        counts = [pc.get(f"{name}_z{zi}", {}).get("n", 0) for zi in range(latent_k)]
        sems = [pc.get(f"{name}_z{zi}", {}).get("sem", float("nan")) for zi in range(latent_k)]
        print(f"  {name}: emp_spread={sp:+.4f} q_pred_spread={qsp:+.4f}  "
              f"best_z={g.get('best_z')} vs z{g.get('second_z')} gap={g.get('gap', float('nan')):+.4f} "
              f"CI=[{g.get('ci_low', float('nan')):+.4f},{g.get('ci_high', float('nan')):+.4f}] "
              f"CI_excl_0={g.get('ci_excludes_zero')}  best_z_agree={best_z_agreement.get(name)}")
        print(f"        counts={counts}  sem={[round(s,3) if not np.isnan(s) else None for s in sems]}")
    print(f"[v6i11] Reliably-separating opponents: {reliably_separating}/{_N_OPPONENTS}")
    print(f"[v6i11] Min arcs per cell  : {min_cell_arcs:.0f}")

    # Positive data verdicts are only CANDIDATES until the held-out gate passes.
    if verdict in ("SEPARATING", "WEAK_SEPARATION"):
        promotion_status = "SEPARATING_CANDIDATE"
    else:
        promotion_status = "NOT_A_CANDIDATE"

    print(f"[v6i11] Routing verdict    : {verdict}")
    print(f"[v6i11] Promotion status   : {promotion_status}")
    print("[v6i11] NOTE: a positive verdict is only a CANDIDATE. The decisive gate is")
    print("[v6i11]       the held-out prospective test (argmax-Q > cross-episode-shuffled-Q,")
    print("[v6i11]       then > uniform, then approaching fixed_z2). NOT run by this script.")
    print("=" * 72)

    # Save Q-router weights and summary.
    torch.save(q_router.state_dict(), out_dir / "q_router_final.pt")

    summary = {
        "preset": _PRESET,
        "checkpoint": str(checkpoint),
        "n_updates": args.n_updates,
        "total_arcs": total_arcs,
        "routing_verdict": verdict,
        "promotion_status": promotion_status,
        "verdict_semantics": {
            "SEPARATING": "online context+targets support value-based routing "
                          "(>=2 opponents: spread>=thr AND best-vs-second CI excludes 0)",
            "WEAK_SEPARATION": "partial predictive structure (exactly 1 opponent passes)",
            "FLAT": "coverage OK but no usable separation learned UNDER THIS dataset/horizon/"
                    "context/budget; the Q-formulation failed to resolve the latents, NOT "
                    "proof the latents do not differ (repertoire diversity already established)",
            "INSUFFICIENT_DATA": "replay coverage or sample count too weak to judge "
                                 "(missing z, missing opponent, zero variance, or <20 arcs/cell)",
            "INVALID": "zero arcs, duplicate contamination, horizon mismatch "
                       "(terminal-finalized fraction too low), or integrity failure",
        },
        "spread_final": {k: float(v) for k, v in spread_final.items()},
        "q_pred_spread": q_pred_spread,
        "best_z_agreement": best_z_agreement,
        "reliability_gap_ci": gap_ci,
        "reliably_separating_opponents": int(reliably_separating),
        "replay_validity": validity,
        "min_cell_arcs": float(min_cell_arcs),
        "frozen_actor_ok": bool(actor_ok),
        "heldout_gate": {
            "status": "REQUIRED_NOT_RUN",
            "decisive": "Q-router > cross-episode-shuffled-Q",
            "then": ["Q-router > uniform", "Q-router approaches/beats fixed_z2"],
            "note": "Do NOT promote on Q-spread alone. Run argmax-Q routing on "
                    "fresh held-out seeds against fixed_z2 / uniform / shuffled-Q / oracle.",
        },
        "updates": all_updates,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[v6i11] Summary written -> {out_dir / 'summary.json'}")
    print(f"[v6i11] Q-router saved  -> {out_dir / 'q_router_final.pt'}")


if __name__ == "__main__":
    main()
