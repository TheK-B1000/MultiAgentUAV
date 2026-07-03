#!/usr/bin/env python3
"""V6I12 paired-advantage router experiment.

Fixes the noise problem in V6I11 raw-return Q targets by separating:
  V(context)  — context baseline absorbing episode-level return variance
  A(context, z) = norm_return - stopgrad(V(context))  — latent residual

Algorithm
---------
1. Build trainer with preset ``v6i11_q_router_hardpool`` (episode-persistent,
   50 % uniform exploration, frozen actor, arc credit data collection).

2. Per update:
   a. Rollout (frozen actor, 50/50 epsilon-greedy z).
   b. Extract arc records: (global_state_0, z, episode_return, opponent_id).
   c. Push to QRouterReplayBuffer.
   d. Double-centering training:
      * Normalize returns globally in each minibatch.
      * Train V(context) to predict normalized return (MSE).
      * target = norm_return - stopgrad(V(context)).
      * Train A(context, z) with Huber loss on advantage target.
   e. Log: baseline R², advantage target std, advantage matrix per (opp, z),
      advantage gap CI.

3. Verdict:
   SEPARATING       ≥2 opponents: advantage gap CI excludes zero AND
                    gap ≥ spread_threshold.
   WEAK_SEPARATION  exactly 1 opponent passes.
   FLAT             coverage OK but no reliable advantage separation.
   INSUFFICIENT_DATA  replay coverage too weak to judge.
   INVALID          zero arcs / duplicate contamination.

Relation to V6I11
-----------------
V6I11 used raw Q-values (unpaired, between-episode targets; return std ~2.6–3.9
per rollout).  The V6I11 run produced 0 arc records in 6 updates due to a
message-ordering bug in the original script; the training infrastructure is
correct (latent_arc_credit_enabled=True is confirmed via TrainerHyperparams).
V6I12 inherits the same infrastructure and adds the V(context) baseline to
suppress the return variance that drowns the latent signal.

Example
-------
    uv run python experiments/run_v6i12_advantage_router.py \\
      --checkpoint checkpoints/2v2/final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip \\
      --n-updates 20 --device cuda --seed 1 \\
      --out-dir artifacts/v6i12_advantage_router
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
from rl.custom_ppo.diagnostics.arc_credit_smoke import (  # noqa: E402
    frozen_actor_z_fingerprint,
    router_fingerprint,
)
from rl.global_state import GLOBAL_STATE_DIM  # noqa: E402

_PRESET = "v6i11_q_router_hardpool"
_N_OPPONENTS = 3
_OPPONENT_ID_TO_IDX = {7: 0, 8: 1, 9: 2}
_LATENT_K = 4
_REPLAY_CAPACITY = 10_000
_A_BATCH_SIZE = 256
_TRAIN_STEPS = 20
_OPP_LABELS = {0: "OP8", 1: "OP9", 2: "OP10"}
_OPP_NAMES = {0: "OP8", 1: "OP9", 2: "OP10"}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V6I12 paired-advantage router diagnostic")
    p.add_argument(
        "--checkpoint",
        default=(
            "checkpoints/2v2/"
            "final_v6i9-mapaware-repertoire-hardpool-refactor-r1-seed1_2v2.zip"
        ),
    )
    p.add_argument("--n-updates", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out-dir", default="artifacts/v6i12_advantage_router")
    p.add_argument("--v-lr", type=float, default=3e-4)
    p.add_argument("--a-lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--replay-capacity", type=int, default=_REPLAY_CAPACITY)
    p.add_argument("--train-steps", type=int, default=_TRAIN_STEPS)
    p.add_argument("--spread-threshold", type=float, default=0.05,
                   help="Minimum advantage gap to declare routing learnable "
                        "(lower than Q-router because advantages are centered)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite a previously completed output directory")
    return p.parse_args()


def _extract_arc_records(trainer) -> list[dict]:
    """Pull completed arc records from trainer.latent_state.rollout_strategy_arc_records."""
    ls = getattr(trainer, "latent_state", None)
    if ls is None:
        return []
    return list(getattr(ls, "rollout_strategy_arc_records", []))


def _abort_invalid(out_dir: Path, reason: str, update_idx: int, args, checkpoint) -> None:
    print()
    print("=" * 72)
    print(f"[v6i12] INVALID at update {update_idx}: {reason}")
    print("[v6i12] Aborting — a broken arc pipeline is a tooling failure, not")
    print("[v6i12] evidence about the repertoire.  No FLAT verdict is emitted.")
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
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    sys.exit(2)


def _advantage_spread_summary(adv_mat: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    for i in range(adv_mat.shape[0]):
        row = adv_mat[i]
        valid = ~np.isnan(row)
        if valid.sum() >= 2:
            out[f"adv_spread_{_OPP_LABELS.get(i, i)}"] = float(np.nanmax(row) - np.nanmin(row))
        else:
            out[f"adv_spread_{_OPP_LABELS.get(i, i)}"] = float("nan")
    return out


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    if out_dir.exists() and not args.force:
        existing = list(out_dir.glob("update_*.json"))
        if existing:
            print(f"[v6i12] Output dir {out_dir} already has results. Pass --force to overwrite.")
            sys.exit(0)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print("=" * 72)
    print(f"[v6i12] preset       = {_PRESET}")
    print(f"[v6i12] checkpoint   = {checkpoint}")
    print(f"[v6i12] n_updates    = {args.n_updates}")
    print(f"[v6i12] device       = {args.device}")
    print(f"[v6i12] out_dir      = {out_dir}")
    print(f"[v6i12] v_lr         = {args.v_lr}   a_lr = {args.a_lr}")
    print(f"[v6i12] hidden       = {args.hidden}")
    print(f"[v6i12] replay_cap   = {args.replay_capacity}")
    print(f"[v6i12] train_steps  = {args.train_steps}")
    print("=" * 72)

    cfg, resolved, env, trainer = _build_audit_trainer(
        preset=_PRESET,
        checkpoint=str(checkpoint),
        device=args.device,
        seed=args.seed,
    )

    # Verify arc credit is enabled.
    arc_enabled = bool(getattr(trainer, "latent_arc_credit_enabled", False))
    print(f"[v6i12] trainer.latent_arc_credit_enabled = {arc_enabled}")
    if not arc_enabled:
        print("[v6i12] FATAL: latent_arc_credit_enabled is False — arc records will never be collected.")
        print("[v6i12] Check that the preset sets latent_arc_credit_enabled=True and that")
        print("[v6i12] fixed_latent_strategy=False and use_latent_strategy=True.")
        sys.exit(2)

    latent_k = int(getattr(cfg, "latent_k", _LATENT_K) or _LATENT_K)
    context_dim = GLOBAL_STATE_DIM + _N_OPPONENTS

    v_baseline = ContextualVBaseline(
        state_dim=GLOBAL_STATE_DIM,
        n_opponents=_N_OPPONENTS,
        hidden=args.hidden,
    ).to(args.device)
    a_router = AdvantageRouter(
        state_dim=GLOBAL_STATE_DIM,
        n_opponents=_N_OPPONENTS,
        opponent_id_to_idx=_OPPONENT_ID_TO_IDX,
        latent_k=latent_k,
        hidden=args.hidden,
    ).to(args.device)
    replay = QRouterReplayBuffer(
        capacity=args.replay_capacity,
        context_dim=context_dim,
        latent_k=latent_k,
    )
    v_optimizer = torch.optim.Adam(v_baseline.parameters(), lr=args.v_lr)
    a_optimizer = torch.optim.Adam(a_router.parameters(), lr=args.a_lr)

    init_router_hash = router_fingerprint(trainer.model)
    init_actor_hash = frozen_actor_z_fingerprint(trainer.model)
    print(f"[v6i12] initial router hash      : {init_router_hash}")
    print(f"[v6i12] initial frozen actor hash: {init_actor_hash}")
    print()

    all_updates: list[dict] = []
    total_arcs = 0

    for update_idx in range(1, args.n_updates + 1):
        t0 = time.time()
        print(f"[v6i12] --- update {update_idx}/{args.n_updates} ---")

        # 1. Rollout (frozen actor, 50/50 epsilon-greedy z).
        buffer = trainer.collect_rollout()

        # 2. Extract arc records BEFORE update (update drains them via post_update).
        records_before_update = len(_extract_arc_records(trainer))
        arc_records = [copy_arc_record(r) for r in _extract_arc_records(trainer)]
        n_arcs = len(arc_records)
        total_arcs += n_arcs

        size_before = len(replay)
        push_stats = replay.push_many(
            arc_records,
            rollout_index=update_idx,
            opponent_id_to_idx=_OPPONENT_ID_TO_IDX,
            build_context=a_router.build_context_from_record,
        )
        size_after = len(replay)

        # Run the (no-op) PPO update — drains arc buffer.
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
                f"terminal_frac={terminal_frac:.3f}"
            )

        # Hard guard: abort on tooling failures (never emit FLAT on broken pipeline).
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

        print(f"  arc_records: {n_arcs}  (replay size: {len(replay)})")

        # 3. Train V-baseline and A-router.
        train_tel = train_advantage_router(
            v_baseline, a_router, replay, v_optimizer, a_optimizer,
            batch_size=min(_A_BATCH_SIZE, len(replay)),
            n_steps=args.train_steps,
            device=args.device,
        )
        print(f"  v_loss={train_tel['v_loss_mean']:.5f}  baseline_r2={train_tel['baseline_r2_mean']:+.4f}")
        print(f"  a_loss={train_tel['a_loss_mean']:.5f}  adv_target_std={train_tel['advantage_target_std_mean']:.4f}")

        # 4. Log empirical advantage matrix (after current training state).
        adv_mat, count_mat = advantage_matrix_from_replay(
            replay, v_baseline,
            n_opponents=_N_OPPONENTS, latent_k=latent_k,
            opponent_id_to_idx=_OPPONENT_ID_TO_IDX,
            device=args.device,
        )
        adv_spread = _advantage_spread_summary(adv_mat)

        print("  empirical advantage (norm_return - V(ctx)) per opp × z:")
        for oi in range(_N_OPPONENTS):
            cells = []
            for zi in range(latent_k):
                v = adv_mat[oi, zi]
                n = int(count_mat[oi, zi])
                cells.append(f"{v:+.3f}(n={n:3d})" if not np.isnan(v) else "  nan( n=  0)")
            sp = adv_spread.get(f"adv_spread_{_OPP_LABELS[oi]}", float("nan"))
            print(f"    {_OPP_LABELS[oi]}: {' '.join(cells)}  spread={sp:+.4f}")

        # Learned A-router matrix (zero-geometry context).
        with torch.no_grad():
            a_mat_learned = a_router.advantage_matrix(device=args.device).cpu().numpy()
        print("  A-router learned values (zero-geometry context):")
        for oi in range(_N_OPPONENTS):
            row = a_mat_learned[oi]
            print(f"    {_OPP_LABELS[oi]}: {' '.join(f'{v:+.4f}' for v in row)}")

        max_spread = max(
            (v for v in adv_spread.values() if not np.isnan(v)), default=float("nan")
        )
        min_arcs_per_cell = float(
            np.nanmin(count_mat[count_mat > 0]) if np.any(count_mat > 0) else 0
        )
        print(f"  max_adv_spread={max_spread:+.4f}  "
              f"min_cell_arcs={min_arcs_per_cell:.0f}  total_arcs={total_arcs}")

        cnt_by_z = replay.count_by_z()
        print(f"  count_by_z: {cnt_by_z}")

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
            **train_tel,
            **{f"adv_mat_OP{8+oi}_z{zi}": float(adv_mat[oi, zi])
               for oi in range(_N_OPPONENTS) for zi in range(latent_k)},
            **{f"count_OP{8+oi}_z{zi}": int(count_mat[oi, zi])
               for oi in range(_N_OPPONENTS) for zi in range(latent_k)},
            **{f"a_learned_OP{8+oi}_z{zi}": float(a_mat_learned[oi, zi])
               for oi in range(_N_OPPONENTS) for zi in range(latent_k)},
            **{k: float(v) for k, v in adv_spread.items()},
            "max_adv_spread": float(max_spread),
            "elapsed_s": round(time.time() - t0, 1),
        }
        all_updates.append(update_record)
        (out_dir / f"update_{update_idx:04d}.json").write_text(json.dumps(update_record, indent=2))

    # Final summary.
    final_actor_hash = frozen_actor_z_fingerprint(trainer.model)
    actor_ok = final_actor_hash == init_actor_hash
    print()
    print(f"[v6i12] Frozen-actor check : {'PASS' if actor_ok else 'FAIL'}")
    print(f"[v6i12] Total arc records  : {total_arcs}")

    # Final advantage matrix and CI.
    adv_mat_final, count_mat_final = advantage_matrix_from_replay(
        replay, v_baseline,
        n_opponents=_N_OPPONENTS, latent_k=latent_k,
        opponent_id_to_idx=_OPPONENT_ID_TO_IDX,
        device=args.device,
    )
    gap_ci = advantage_gap_ci(
        replay, v_baseline, a_router,
        n_opponents=_N_OPPONENTS, latent_k=latent_k,
        opponent_id_to_idx=_OPPONENT_ID_TO_IDX,
        device=args.device,
    )
    spread_final = _advantage_spread_summary(adv_mat_final)
    validity = replay.validity_report(
        n_opponents=_N_OPPONENTS, latent_k=latent_k,
        opponent_id_to_idx=_OPPONENT_ID_TO_IDX,
    )

    min_cell_arcs = float(
        np.nanmin(count_mat_final[count_mat_final > 0])
        if np.any(count_mat_final > 0) else 0
    )

    # Use Q-router verdict logic with advantage spread.
    # Remap spread keys from adv_spread_ prefix to spread_ for compatibility.
    spread_for_verdict = {
        k.replace("adv_spread_", "spread_"): v for k, v in spread_final.items()
    }
    verdict, reliably_separating = decide_verdict(
        validity=validity,
        gap_ci=gap_ci,
        spread=spread_for_verdict,
        spread_threshold=args.spread_threshold,
        min_cell_arcs=min_cell_arcs,
        n_opponents=_N_OPPONENTS,
        opp_names=_OPP_NAMES,
    )
    if not actor_ok:
        verdict = "INVALID"

    print()
    print("=" * 72)
    print("[v6i12] Replay validity")
    print(f"  all_z_represented        : {validity['all_z_represented']}")
    print(f"  all_opponents_represented: {validity['all_opponents_represented']}")
    print(f"  return_variance_nonzero  : {validity['return_variance_nonzero']}")
    print(f"  terminal_frac            : {validity['terminal_finalized_fraction']:.3f}")
    print()
    print(f"[v6i12] Advantage gaps (norm_return - V(ctx)):")
    for oi in range(_N_OPPONENTS):
        name = _OPP_NAMES.get(oi, str(oi))
        g = gap_ci.get(name, {})
        sp = spread_final.get(f"adv_spread_{name}", float("nan"))
        counts = [int(count_mat_final[oi, zi]) for zi in range(latent_k)]
        print(f"  {name}: best_z={g.get('best_z')} vs z{g.get('second_z')} "
              f"gap={g.get('gap', float('nan')):+.4f} "
              f"CI=[{g.get('ci_low', float('nan')):+.4f},{g.get('ci_high', float('nan')):+.4f}] "
              f"CI_excl_0={g.get('ci_excludes_zero')}  adv_spread={sp:+.4f}")
        print(f"        counts={counts}")
    print(f"[v6i12] Reliably-separating opponents: {reliably_separating}/{_N_OPPONENTS}")
    print(f"[v6i12] Min arcs per cell  : {min_cell_arcs:.0f}")
    print(f"[v6i12] Advantage verdict  : {verdict}")

    if verdict in ("SEPARATING", "WEAK_SEPARATION"):
        promotion_status = "SEPARATING_CANDIDATE"
    else:
        promotion_status = "NOT_A_CANDIDATE"
    print(f"[v6i12] Promotion status   : {promotion_status}")
    print("[v6i12] NOTE: SEPARATING only promotes after held-out prospective test")
    print("[v6i12]       (A-router > cross-episode-shuffled baseline, then > uniform).")
    print("=" * 72)

    torch.save(v_baseline.state_dict(), out_dir / "v_baseline_final.pt")
    torch.save(a_router.state_dict(), out_dir / "a_router_final.pt")

    summary = {
        "preset": _PRESET,
        "checkpoint": str(checkpoint),
        "n_updates": args.n_updates,
        "total_arcs": total_arcs,
        "routing_verdict": verdict,
        "promotion_status": promotion_status,
        "verdict_semantics": {
            "SEPARATING": "advantage CI excludes zero + gap>=threshold for >=2 opponents",
            "WEAK_SEPARATION": "exactly 1 opponent reliably separates",
            "FLAT": "coverage OK but no reliable advantage separation — "
                    "V(context) cannot absorb return variance, or latents "
                    "have equal value given context",
            "INSUFFICIENT_DATA": "replay coverage too weak to judge",
            "INVALID": "zero arcs / duplicate contamination / integrity failure",
        },
        "spread_final_adv": {k: float(v) for k, v in spread_final.items()},
        "reliability_gap_ci": gap_ci,
        "reliably_separating_opponents": int(reliably_separating),
        "replay_validity": validity,
        "min_cell_arcs": float(min_cell_arcs),
        "frozen_actor_ok": bool(actor_ok),
        "heldout_gate": {
            "status": "REQUIRED_NOT_RUN",
            "decisive": "A-router > cross-episode-shuffled-A-router",
            "then": ["A-router > uniform", "A-router approaches/beats fixed_z2"],
            "note": "Do NOT promote on advantage spread alone.",
        },
        "updates": all_updates,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[v6i12] Summary written -> {out_dir / 'summary.json'}")
    print(f"[v6i12] V-baseline saved -> {out_dir / 'v_baseline_final.pt'}")
    print(f"[v6i12] A-router saved  -> {out_dir / 'a_router_final.pt'}")


if __name__ == "__main__":
    main()
