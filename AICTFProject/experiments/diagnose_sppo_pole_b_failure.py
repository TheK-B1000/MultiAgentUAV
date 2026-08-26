"""SPPPO V1 postmortem D0 -- where does pole-B strategic identity break?

NO TRAINING. NO NEW SEEDS. NO GATE RE-SCORING. This replays the already-scored
z1|B episodes (seeds 10300001..10300192) under the frozen production checkpoint
and decomposes the FROZEN identity metric per decision state. Precedent:
EXP2B_BRANCH_PLAN_FROZEN.json case B4 -- "descriptive postmortem using EXISTING
checkpoints and rows only; no new training".

WHAT THE FROZEN METRIC ACTUALLY SAYS. From eval_exp2_k2_terminal.action_identity:

    margin_A_bits = KL(pi_A || z1) - KL(pi_A || z0)     +0.960  PASS
    margin_B_bits = KL(pi_B || z0) - KL(pi_B || z1)     -0.031  FAIL

margin_B_bits < 0 does NOT merely mean "z1 drifted from pi_B". It means
**z0 is CLOSER to pi_B than z1 is** -- the wrong mode resembles the BREACH
teacher more than the assigned one does. Combined with jsd PASS (the modes are
different) and margin_A PASS (z0 is strongly pi_A-like), the picture is
undirected diversity: z1 is different from both teachers rather than aligned
with its own.

This file reuses the FROZEN helpers (`action_identity`, `_dist_probs`) rather
than re-deriving a parallel metric, so what it decomposes is the actual gate
number and not a lookalike. Q_psi is queried through the unchanged
`rl.scorer.ranking.strategic_contrast`.

THE FORK, evaluated on the states where margin_B is worst:

    Q_psi ranks z1 above z0 there   -> OPTIMISATION_INTERFERENCE
        the surrogate was right; the ranking pressure failed to transfer
    Q_psi does not rank z1 there    -> SURROGATE_FAILURE
        the scorer itself cannot distinguish the failing states

Run:  python experiments/diagnose_sppo_pole_b_failure.py --device cuda
      python experiments/diagnose_sppo_pole_b_failure.py --device cuda --seeds 8
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand"
SPPO = SD / "sppo"
OUT = SPPO / "D0_pole_b_diagnostic.json"
OUT_ROWS = SPPO / "D0_pole_b_decision_rows.csv"

SEED_BASE, N_SEEDS, MAX_STEPS = 10_300_001, 192, 240
QPSI_PATH = SD / "phase0_scorer_data" / "qpsi_frozen.pt"
QPSI_SHA = "930051a725e55e4f14e05dfe178e5f1dc7bd8f3d7e3adeba01187958bb7417bf"
STUDENT_SHA = "1260e420b85ad01b1aecf433d65688adf153538195d30511e26a3585b7285f1f"
RANKING_MARGIN = 0.04


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    a = ap.parse_args()
    device = a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"

    # Import the SPPPO wrapper FIRST so the shared evaluator module carries SPPPO
    # identity, then reuse its frozen helpers. (Wrapper rebinding is shared-module
    # mutation -- see eval_sppo_v1_terminal._assert_bindings_intact.)
    from experiments import eval_sppo_v1_terminal as W
    from experiments import eval_exp2_k2_terminal as E
    W._assert_bindings_intact()

    from experiments.opponent_spec import assert_live_opponent_batch, install_keyed_opponent_overlays
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.scorer.ranking import POLE_B, load_frozen_qpsi, strategic_contrast

    for name, path, sha in (("student", E.STUDENT, STUDENT_SHA),
                            ("pi_A", E.TEACHER_A, E.EXPECTED_HASHES["pi_A"]),
                            ("pi_B", E.TEACHER_B, E.EXPECTED_HASHES["pi_B"])):
        if E._sha256(path) != sha:
            raise SystemExit(f"REFUSING: {name} checkpoint hash mismatch")
    qpsi = load_frozen_qpsi(QPSI_PATH, expected_sha256=QPSI_SHA, device=device)

    probe = E.build_env(device, SEED_BASE)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    policy_map = {
        "student": load_custom_ppo_policy(str(E.STUDENT), obs_space, act_space, device=device),
        "pi_A": load_custom_ppo_policy(str(E.TEACHER_A), obs_space, act_space, device=device),
        "pi_B": load_custom_ppo_policy(str(E.TEACHER_B), obs_space, act_space, device=device),
    }
    inner_student = getattr(policy_map["student"], "model", policy_map["student"])

    seeds = list(range(SEED_BASE, SEED_BASE + a.seeds))
    print(f"D0 POLE-B DIAGNOSTIC  {_now()}")
    print(f"  seeds  {seeds[0]}..{seeds[-1]}  (n={len(seeds)}) -- ALREADY-SCORED block, no growth")
    print(f"  reuses frozen action_identity(); Q_psi {QPSI_SHA[:16]}... measurement only\n",
          flush=True)

    rows = []
    for si, seed in enumerate(seeds):
        env = E.build_env(device, seed)
        core = env.core
        try:
            student = policy_map["student"]
            student.fixed_latent_strategy = True          # frozen evaluator convention
            student.fixed_latent_strategy_id = 1          # z1 is the assigned mode on pole B
            student.reset_strategy()
            core._bt_profile_override = None
            core._sds_opening_hold_steps = 0
            install_keyed_opponent_overlays(core, {})     # pole B (OP7) takes no genome
            env.env_method("set_phase", phase_from_tag("OP7"))
            env.env_method("set_next_opponent", "SCRIPTED", "OP7")
            obs = env.reset()
            obs["global_state"] = env.state()
            assert_live_opponent_batch(core, {}, allowed_keys=("OP7",),
                                       context=f"D0 z1|B seed {seed}")
            didx = 0
            for step in range(MAX_STEPS):
                ident = E.action_identity(policy_map, obs)       # THE FROZEN METRIC
                if ident["count"] > 0:
                    obs_t = {k: torch.as_tensor(np.asarray(obs[k]), dtype=torch.float32,
                                                device=device)
                             for k in ("grid", "vec", "agent_mask", "mask")}
                    pole_t = torch.full((1,), POLE_B, dtype=torch.long, device=device)
                    with torch.no_grad():
                        d_hat, _, _ = strategic_contrast(inner_student, qpsi, obs_t, pole_t)
                    rows.append({
                        "seed": seed, "step": step, "decision_idx": didx,
                        "jsd_bits": ident["jsd_bits"],
                        "margin_A_bits": ident["margin_A_bits"],
                        "margin_B_bits": ident["margin_B_bits"],
                        "argmax_disagree": ident["argmax_disagree"],
                        "eligible_heads": ident["count"],
                        "delta_B_hat_qpsi": float(d_hat.item()),
                        "qpsi_ranks_z1_correct": int(float(d_hat.item()) > RANKING_MARGIN),
                        "blue_carrying": int(bool(core.blue_carrying[0].any().item())),
                        "red_carrying": int(bool(core.red_carrying[0].any().item())),
                        "own_flag_home": int(bool(torch.allclose(
                            core.blue_flag_pos[0], core.blue_flag_home[0], atol=1e-3))),
                        "blue_score": int(core.blue_score[0].item()),
                        "red_score": int(core.red_score[0].item()),
                    })
                    didx += 1
                action, _ = student.predict(obs, deterministic=True)
                env.step_async(action)
                obs, _r, done, _info = env.step_wait()
                obs["global_state"] = env.state()
                if bool(np.asarray(done).any()):
                    break
        finally:
            env.close()
        if (si + 1) % 24 == 0 or si == len(seeds) - 1:
            print(f"  {si+1}/{len(seeds)}   decisions: {len(rows)}", flush=True)

    by_seed = defaultdict(list)
    for r in rows:
        by_seed[r["seed"]].append(r)
    for rs in by_seed.values():
        n = len(rs)
        for i, r in enumerate(rs):
            r["tertile"] = "early" if i < n / 3 else ("late" if i >= 2 * n / 3 else "mid")

    with open(OUT_ROWS, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    mB = np.array([r["margin_B_bits"] for r in rows])
    dq = np.array([r["delta_B_hat_qpsi"] for r in rows])
    ok = np.array([r["qpsi_ranks_z1_correct"] for r in rows], dtype=bool)

    worst = mB <= np.percentile(mB, 25)          # the states dragging the gate negative
    fork = ("OPTIMISATION_INTERFERENCE" if float(ok[worst].mean()) >= 0.5
            else "SURROGATE_FAILURE")

    def stats(mask, label):
        if not mask.any():
            return None
        return {"label": label, "n": int(mask.sum()),
                "mean_margin_B_bits": float(mB[mask].mean()),
                "frac_margin_B_negative": float((mB[mask] < 0).mean()),
                "mean_delta_B_hat": float(dq[mask].mean()),
                "qpsi_correct_rate": float(ok[mask].mean())}

    cats = {
        "ALL": stats(np.ones(len(rows), bool), "all decisions"),
        "worst_quartile_margin_B": stats(worst, "margin_B in worst 25%"),
        "best_quartile_margin_B": stats(mB >= np.percentile(mB, 75), "margin_B in best 25%"),
        "carrying": stats(np.array([r["blue_carrying"] for r in rows], bool), "blue carrying flag"),
        "not_carrying": stats(~np.array([r["blue_carrying"] for r in rows], bool), "not carrying"),
        "own_flag_home": stats(np.array([r["own_flag_home"] for r in rows], bool), "own flag home"),
        "own_flag_stolen": stats(~np.array([r["own_flag_home"] for r in rows], bool), "own flag stolen"),
        "early": stats(np.array([r["tertile"] == "early" for r in rows], bool), "early tertile"),
        "mid": stats(np.array([r["tertile"] == "mid" for r in rows], bool), "mid tertile"),
        "late": stats(np.array([r["tertile"] == "late" for r in rows], bool), "late tertile"),
    }

    rec = {
        "record": "SPPPO V1 postmortem D0 -- pole-B strategic identity decomposition",
        "status": "DIAGNOSTIC_ONLY -- no training, no new seeds, no gate re-scoring",
        "utc": _now(),
        "reuses_unchanged": ["eval_exp2_k2_terminal.action_identity (THE frozen metric)",
                             "rl.scorer.ranking.strategic_contrast"],
        "what_margin_B_negative_means": ("margin_B_bits = KL(pi_B||z0) - KL(pi_B||z1). "
                                         "Negative means z0 is CLOSER to pi_B than z1 is: "
                                         "the wrong mode resembles the BREACH teacher."),
        "n_seeds": len(seeds), "n_decision_points": len(rows),
        "categories": cats,
        "fraction_of_all_decisions_with_negative_margin_B": float((mB < 0).mean()),
        "DIAGNOSTIC_FORK_VERDICT": fork,
        "fork_basis": {
            "states": "worst quartile of margin_B_bits",
            "qpsi_correct_rate_there": float(ok[worst].mean()),
            "OPTIMISATION_INTERFERENCE": "Q_psi right, pressure did not transfer",
            "SURROGATE_FAILURE": "Q_psi itself cannot distinguish the failing states",
        },
        "raw_rows": str(OUT_ROWS.relative_to(ROOT)),
    }
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    print(f"\n  decisions scanned           : {len(rows)}")
    print(f"  margin_B negative fraction  : {(mB < 0).mean():.3f}")
    print(f"  worst-quartile mean margin_B: {mB[worst].mean():+.4f}")
    print(f"  Q_psi correct-rate there    : {ok[worst].mean():.3f}")
    print(f"  FORK VERDICT                : {fork}")
    print(f"\n  -> {OUT}\n  -> {OUT_ROWS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
