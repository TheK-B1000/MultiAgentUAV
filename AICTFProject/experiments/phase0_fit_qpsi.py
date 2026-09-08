"""Phase 0 -- fit Q_psi on the 160 TRAIN seeds and freeze it.

Target (PHASE0_SCORER_TARGET_AMENDMENT.json, frozen before this file existed):

    Q_psi(o, a1, a2, p)  ->  E[ terminal win margin | o, a, p ],   margin = blue - red

Monte-Carlo return is EXCLUDED. Policy identity is not an input.

Development selection lives entirely inside the 160 training seeds via a
prospectively frozen inner split (128 fit / 32 inner-val, contiguous, mirroring
the outer split rule). The 96 held-out seeds are the scientific gate, not a
development set, and this script asserts it never opens one.

WEIGHTING. Plain rows carry an EPISODE-level label: every decision point in an
episode shares one terminal margin. Each plain row is therefore weighted
1/len(episode) so an episode contributes one unit of loss, while each branch row
-- which has its own distinct continuation and terminal margin -- carries unit
weight. This implements the frozen effective-sample-size caveat directly in the
objective instead of asserting it only in prose.

EARLY STOPPING is on inner-val weighted MSE, deliberately NOT on the strategic
ordering. Selecting the checkpoint by ordering quality, even on train-side data,
would tune the estimator toward the gate's quantity. Fit quality is the
selection criterion; ordering is reported afterwards as a diagnostic.

Run:  python experiments/phase0_fit_qpsi.py --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase0_scorer_common import (            # noqa: E402
    COLL, HELDOUT_SEEDS, INNER_FIT_SEEDS, INNER_VAL_SEEDS, SD, TRAIN_SEEDS,
    assert_teacher_query_valid, load_split, sha256_file, teacher_action_dists,
)
from rl.scorer.qpsi import QPsi, QPsiConfig, joint_action_index   # noqa: E402

OUT_DIR = SD / "phase0_scorer_data"
WEIGHTS = OUT_DIR / "qpsi_frozen.pt"
RECORD = SD / "PHASE0_SCORER_FROZEN.json"
SC = SD / "sappo_continuation"
TEACHERS = {
    "pi_A": SC / "sappo_pi_A_specialist_1p5M_seed7100001/ckpts/final_sappo_pi_A_specialist_1p5M_seed7100001.zip",
    "pi_B": SC / "sappo_pi_B_specialist_1p5M_seed7200001/ckpts/final_sappo_pi_B_specialist_1p5M_seed7200001.zip",
}

# ---- frozen training configuration (fixed before the first fit) -------------
TRAIN_CFG = {
    "optimizer": "Adam", "lr": 3e-4, "weight_decay": 0.0,
    "batch_size": 512, "max_epochs": 60, "early_stop_patience": 8,
    "torch_seed": 7, "loss": "weighted MSE on terminal win margin",
    "plain_row_weight": "1 / (decision points in that episode)",
    "branch_row_weight": "1.0",
    "early_stop_criterion": "inner-val weighted MSE (NOT strategic ordering)",
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tensors(split, device):
    """Pool plain + branch rows into one weighted regression dataset."""
    t = lambda a, dt: torch.as_tensor(a, dtype=dt)
    grid = torch.cat([t(split.p_grid, torch.float32), t(split.b_grid, torch.float32)])
    vec = torch.cat([t(split.p_vec, torch.float32), t(split.b_vec, torch.float32)])
    am = torch.cat([t(split.p_amask, torch.float32), t(split.b_amask, torch.float32)])
    pole = torch.cat([t(split.p_pole, torch.long), t(split.b_pole, torch.long)])
    act = torch.cat([t(split.p_action, torch.long), t(split.b_action, torch.long)])
    y = torch.cat([t(split.p_margin, torch.float32), t(split.b_margin, torch.float32)])
    w = torch.cat([t(split.p_weight, torch.float32),
                   torch.ones(len(split.b_margin), dtype=torch.float32)])
    a1, a2 = joint_action_index(act)
    return grid, vec, am, pole, a1, a2, y, w


def _eval_loss(model, data, device, bs=1024) -> float:
    grid, vec, am, pole, a1, a2, y, w = data
    model.eval()
    tot_l, tot_w = 0.0, 0.0
    with torch.no_grad():
        for i in range(0, len(y), bs):
            s = slice(i, i + bs)
            pred = model(grid[s].to(device), vec[s].to(device), am[s].to(device),
                         pole[s].to(device), a1[s].to(device), a2[s].to(device))
            ww = w[s].to(device)
            tot_l += float((ww * (pred - y[s].to(device)) ** 2).sum())
            tot_w += float(ww.sum())
    return tot_l / max(tot_w, 1e-9)


def _ordering_diagnostic(model, split, teachers, device):
    """TRAIN-SIDE ONLY. d_A / d_B on inner-val branch states. NOT Gate 0B."""
    # Refuse to compute any value from a teacher distribution that does not
    # reproduce the policy's own recorded deterministic actions. The first run
    # of this script silently queried UNMASKED logits (10-17% argmax agreement)
    # because the inference wrapper exposes no masking path; the resulting
    # ordering numbers were meaningless. This guard makes that unrepeatable.
    checks = {tag: assert_teacher_query_valid(teachers[tag], split, ti, device)
              for ti, tag in enumerate(("pi_A", "pi_B"))}
    grid = torch.as_tensor(split.b_grid, dtype=torch.float32)
    vec = torch.as_tensor(split.b_vec, dtype=torch.float32)
    am = torch.as_tensor(split.b_amask, dtype=torch.float32)
    mk = torch.as_tensor(split.b_mask, dtype=torch.float32)
    pole = torch.as_tensor(split.b_pole, dtype=torch.long)
    # branch rows duplicate each state per teacher; keep one copy per state
    keep = split.b_teacher == 0
    idx = np.nonzero(keep)[0]
    model.eval()
    V = {}
    with torch.no_grad():
        for tag in ("pi_A", "pi_B"):
            vals = []
            for i in range(0, len(idx), 512):
                j = idx[i:i + 512]
                g, v, a, m, p = (grid[j].to(device), vec[j].to(device), am[j].to(device),
                                 mk[j].to(device), pole[j].to(device))
                p1, p2 = teacher_action_dists(teachers[tag], g, v, a, m)
                vals.append(model.expected_value(g, v, a, p, p1, p2).cpu().numpy())
            V[tag] = np.concatenate(vals)
    pl = split.b_pole[idx]
    sd = split.b_seed[idx]
    out = {}
    for name, sel, sign in (("A", pl == 0, 1.0), ("B", pl == 1, -1.0)):
        d = sign * (V["pi_A"][sel] - V["pi_B"][sel])
        seeds = sd[sel]
        per = np.array([d[seeds == s].mean() for s in np.unique(seeds)])
        out[f"d_{name}"] = {"mean": round(float(d.mean()), 6),
                            "per_seed_mean": round(float(per.mean()), 6),
                            "n_states": int(sel.sum()), "n_seeds": int(len(per))}
    out["teacher_query_validity"] = checks
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if RECORD.is_file():
        raise SystemExit(f"REFUSING: {RECORD} exists; the scorer is already frozen")

    device = a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"
    torch.manual_seed(TRAIN_CFG["torch_seed"])
    np.random.seed(TRAIN_CFG["torch_seed"])

    print(f"PHASE 0 -- FIT Q_psi  {_now()}")
    print(f"  target   E[terminal win margin | o,a,p]   (margin = blue - red)")
    print(f"  fit      {len(INNER_FIT_SEEDS)} seeds  {INNER_FIT_SEEDS[0]}..{INNER_FIT_SEEDS[-1]}")
    print(f"  inner-val{len(INNER_VAL_SEEDS):4d} seeds  {INNER_VAL_SEEDS[0]}..{INNER_VAL_SEEDS[-1]}")
    print(f"  held-out {len(HELDOUT_SEEDS)} seeds  NOT OPENED\n")

    t0 = time.perf_counter()
    fit_s = load_split(INNER_FIT_SEEDS)
    val_s = load_split(INNER_VAL_SEEDS)
    if fit_s.heldout_opened or val_s.heldout_opened:
        raise SystemExit("REFUSING: a held-out shard was opened during fitting")
    print(f"  loaded in {time.perf_counter()-t0:.1f}s  "
          f"fit plain={len(fit_s.p_margin)} branch={len(fit_s.b_margin)}  "
          f"val plain={len(val_s.p_margin)} branch={len(val_s.b_margin)}")

    fit_d, val_d = _tensors(fit_s, device), _tensors(val_s, device)
    grid, vec, am, pole, a1, a2, y, w = fit_d
    n = len(y)

    model = QPsi().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_CFG["lr"],
                           weight_decay=TRAIN_CFG["weight_decay"])
    bs = TRAIN_CFG["batch_size"]
    best, best_state, bad, history = float("inf"), None, 0, []

    for ep in range(TRAIN_CFG["max_epochs"]):
        model.train()
        perm = torch.randperm(n)
        run_l, run_w = 0.0, 0.0
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            pred = model(grid[j].to(device), vec[j].to(device), am[j].to(device),
                         pole[j].to(device), a1[j].to(device), a2[j].to(device))
            ww = w[j].to(device)
            loss = (ww * (pred - y[j].to(device)) ** 2).sum() / ww.sum().clamp_min(1e-9)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            bw = float(ww.sum()); run_l += float(loss.detach()) * bw; run_w += bw
        vl = _eval_loss(model, val_d, device)
        history.append({"epoch": ep, "train_wmse": round(run_l / max(run_w, 1e-9), 6),
                        "val_wmse": round(vl, 6)})
        print(f"  epoch {ep:2d}  train {run_l/max(run_w,1e-9):.5f}  val {vl:.5f}"
              + ("  *" if vl < best else ""))
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= TRAIN_CFG["early_stop_patience"]:
                print(f"  early stop at epoch {ep} (patience {TRAIN_CFG['early_stop_patience']})")
                break

    model.load_state_dict(best_state)
    torch.save({"config": model.cfg.to_dict(), "state_dict": best_state}, WEIGHTS)
    wsha = sha256_file(WEIGHTS)

    # ---- train-side calibration ------------------------------------------
    from rl.custom_ppo import load_custom_ppo_policy
    import experiments.r2_learned_crossover as R2
    probe = R2.build_env(device, TRAIN_SEEDS[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()
    teachers = {k: load_custom_ppo_policy(str(v), obs_space, act_space, device=device)
                for k, v in TEACHERS.items()}
    order = _ordering_diagnostic(model, val_s, teachers, device)

    model.eval()
    with torch.no_grad():
        g, v, amk, pl, b1, b2, yy, _ = val_d
        preds = []
        for i in range(0, len(yy), 1024):
            s = slice(i, i + 1024)
            preds.append(model(g[s].to(device), v[s].to(device), amk[s].to(device),
                               pl[s].to(device), b1[s].to(device), b2[s].to(device)).cpu())
        pred = torch.cat(preds).numpy()
    yv = yy.numpy()
    corr = float(np.corrcoef(pred, yv)[0, 1])

    rec = {
        "record": "PHASE0 frozen action-conditioned scorer Q_psi",
        "status": "FROZEN_BEFORE_ANY_HELD_OUT_ACCESS",
        "utc": _now(),
        "target": {
            "definition": "E[terminal win margin | o, a, p], margin = blue_score - red_score",
            "amendment": "PHASE0_SCORER_TARGET_AMENDMENT.json",
            "monte_carlo_return": "EXCLUDED from fitting",
            "orientation": "blue is always the evaluated policy's team; positive = better for it",
        },
        "architecture": {
            "form": "Q = b(o,p) + u(o,p)^T e(a1) + v(o,p)^T e(a2) + e(a1)^T M(o,p) e(a2)",
            "M_factorisation": "low-rank M = P Q^T, never materialised",
            "per_agent_action": "single categorical over 250 = 5 macros x 50 waypoints",
            "policy_identity_input": "ABSENT -- prohibited by protocol",
            "config": model.cfg.to_dict(),
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
            "module": "rl/scorer/qpsi.py",
        },
        "training_config": TRAIN_CFG,
        "seed_sets": {
            "train_total": [TRAIN_SEEDS[0], TRAIN_SEEDS[-1]],
            "inner_fit": [INNER_FIT_SEEDS[0], INNER_FIT_SEEDS[-1], len(INNER_FIT_SEEDS)],
            "inner_val": [INNER_VAL_SEEDS[0], INNER_VAL_SEEDS[-1], len(INNER_VAL_SEEDS)],
            "inner_split_frozen_before_fitting": True,
            "held_out_range": [HELDOUT_SEEDS[0], HELDOUT_SEEDS[-1], len(HELDOUT_SEEDS)],
        },
        "held_out_seeds_opened": 0,
        "data_hashes": {
            "inner_fit_shards_sha256": fit_s.data_sha256,
            "inner_val_shards_sha256": val_s.data_sha256,
            "n_fit_plain_rows": int(len(fit_s.p_margin)),
            "n_fit_branch_rows": int(len(fit_s.b_margin)),
            "n_val_plain_rows": int(len(val_s.p_margin)),
            "n_val_branch_rows": int(len(val_s.b_margin)),
        },
        "teacher_checkpoints": {k: sha256_file(v) for k, v in TEACHERS.items()},
        "weights": {"path": str(WEIGHTS.relative_to(ROOT)), "sha256": wsha},
        "effective_sample_size": {
            "statistical_unit": "episode / seed, NOT state",
            "inner_fit_seeds": len(INNER_FIT_SEEDS),
            "inner_fit_plain_episodes": int(len(INNER_FIT_SEEDS) * 4),
            "inner_fit_branch_labelled_rows": int(len(fit_s.b_margin)),
            "note": "plain state rows share one episode label and are weighted 1/len(episode); they are not independent payoff observations",
        },
        "train_side_calibration_NOT_gate_0B": {
            "scope": "inner-val seeds only (train-side); the 96 held-out seeds were not opened",
            "best_inner_val_weighted_mse": round(best, 6),
            "pred_vs_actual_margin_corr_inner_val": round(corr, 6),
            "pred_margin_mean": round(float(pred.mean()), 6),
            "pred_margin_std": round(float(pred.std()), 6),
            "actual_margin_mean": round(float(yv.mean()), 6),
            "ordering_diagnostic": order,
            "warning": "this is a development diagnostic. It is NOT Gate 0B and must never be reported as such.",
        },
        "epoch_history": history,
        "next_action": "Gate 0B on the 96 held-out seeds, seed-level bootstrap, one shot",
    }
    RECORD.write_text(json.dumps(rec, indent=2), encoding="utf-8")

    print(f"\n  best inner-val weighted MSE : {best:.5f}")
    print(f"  pred/actual margin corr     : {corr:+.4f}")
    print(f"  train-side ordering (NOT 0B): {order}")
    print(f"  weights sha256              : {wsha[:16]}...")
    print(f"  held-out seeds opened       : 0")
    print(f"  -> {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
