"""D2 -- fit Q_psi at each density rung. Implements D2_SPEC_FROZEN.json +
D2_SPEC_AMENDMENT_1.json + D2_SPEC_AMENDMENT_3.json.

Reuses phase0_fit_qpsi.py's architecture, target, loss, optimizer, schedule and
torch_seed UNCHANGED. The only thing that varies across rungs is the branch-row
set fed to fitting, via the frozen mixture-mass-control rule:

    stolen pole-B branch ROWS get weight 100.0 / n_stolen_rows_this_rung each
    (1.0 / 0.5 / 0.25 at rungs 1/2/3 respectively -- see D2_SPEC_AMENDMENT_3:
    the baseline's 50 stolen STATES are 100 stolen ROWS, two per state)
    every other row (all plain rows, pole-A branches, non-stolen pole-B
    branches) keeps its ORIGINAL weight and count, unmodified at every rung

Rung 1 is a FRESH refit at the original density (n=50), not the original frozen
checkpoint -- isolates density from run-to-run CUDA nondeterminism.

Run:  python experiments/d2_fit_density_ablation.py --device cuda
"""
from __future__ import annotations

import glob
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.phase0_scorer_common import (            # noqa: E402
    INNER_FIT_SEEDS, INNER_VAL_SEEDS, SD, load_split,
)
from rl.scorer.qpsi import QPsi, QPsiConfig, joint_action_index, N_ACTIONS  # noqa: E402

SPPO = SD / "sppo"
D2_DIR = SPPO / "d2_density"
SUPPLEMENT = D2_DIR / "supplement_shards"
SELECTION = SPPO / "D2_POINT_SELECTION.json"
OUT_DIR = D2_DIR / "fits"
RECORD = SPPO / "D2_FIT_RESULT.json"

# UNITS (D2_SPEC_AMENDMENT_3): load_split emits TWO rows per branch state, one
# per teacher. The baseline's 50 stolen pole-B STATES are 100 stolen pole-B ROWS.
# Everything below is counted in ROWS so both sides of the guard match.
BASELINE_STOLEN_ROWS = 100             # 50 states x 2 teachers
RUNGS_NEW_STATES = {1: 0, 2: 50, 3: 150}   # rung -> cumulative NEW states added
TRAIN_CFG = {"lr": 3e-4, "batch_size": 512, "max_epochs": 60,
            "early_stop_patience": 8, "torch_seed": 7}
POLE_B = 1


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_supplement_branch_rows(seeds_and_steps: list[tuple[int, int]]):
    """Expand collected (seed, step) supplement points into per-teacher rows,
    matching load_split's branch-row convention exactly (2 rows per point)."""
    by_seed: dict[int, list[int]] = {}
    for s, t in seeds_and_steps:
        by_seed.setdefault(s, []).append(t)

    B = {k: [] for k in ("grid", "vec", "amask", "mask", "pole", "action", "margin", "teacher")}
    for seed in by_seed:
        z = np.load(SUPPLEMENT / f"seed_{seed}.npz", allow_pickle=True)
        # the supplement file holds ALL points collected for this seed, in
        # collection order; match by count (collector wrote them in the same
        # order as D2_POINT_SELECTION.json's per-seed occurrences)
        n = z["branch_pole"].shape[0]
        for i in range(n):
            for ti, tag in enumerate(("pi_A", "pi_B")):
                # strip the vec-env leading dim, matching load_split's [:, 0]
                # convention exactly -- supplement shards store (n,1,2,...) as
                # written by branch_at()'s obs snapshot
                B["grid"].append(z["branch_obs_grid"][i, 0])
                B["vec"].append(z["branch_obs_vec"][i, 0])
                B["amask"].append(z["branch_obs_agent_mask"][i, 0])
                B["mask"].append(z["branch_obs_mask"][i, 0])
                B["pole"].append(POLE_B)
                B["action"].append(z[f"branch_{tag}_action"][i])
                blue = float(z[f"branch_{tag}_blue"][i]); red = float(z[f"branch_{tag}_red"][i])
                B["margin"].append(blue - red)
                B["teacher"].append(ti)
    cat = lambda k, dt: np.asarray(B[k], dtype=dt)
    return {
        "grid": cat("grid", np.float32), "vec": cat("vec", np.float32),
        "amask": cat("amask", np.float32), "mask": cat("mask", np.float32),
        "pole": cat("pole", np.int64), "action": cat("action", np.int64),
        "margin": cat("margin", np.float32), "teacher": cat("teacher", np.int64),
    }


def _is_stolen(vec_row: np.ndarray, model: QPsi) -> np.ndarray:
    v = torch.as_tensor(vec_row, dtype=torch.float32)
    return (model.regime_from_vec(v).numpy() >= 2)


def build_rung_tensors(rung: int, baseline_split, supplement_all, tagger_model):
    """Baseline rows UNCHANGED, plus the first RUNGS_NEW_STATES[rung] new
    supplement states (2 rows each), with the frozen weight-rescaling rule."""
    t = lambda a, dt: torch.as_tensor(a, dtype=dt)

    # plain rows: untouched at every rung
    p_grid, p_vec = t(baseline_split.p_grid, torch.float32), t(baseline_split.p_vec, torch.float32)
    p_amask, p_pole = t(baseline_split.p_amask, torch.float32), t(baseline_split.p_pole, torch.long)
    p_act = t(baseline_split.p_action, torch.long)
    p_y, p_w = t(baseline_split.p_margin, torch.float32), t(baseline_split.p_weight, torch.float32)
    p_a1, p_a2 = joint_action_index(p_act)

    # baseline branch rows: untouched set, but re-weighted if stolen pole-B
    b_grid, b_vec = baseline_split.b_grid, baseline_split.b_vec
    b_amask, b_mask = baseline_split.b_amask, baseline_split.b_mask
    b_pole, b_act, b_y = baseline_split.b_pole, baseline_split.b_action, baseline_split.b_margin
    b_stolen = np.zeros(len(b_pole), dtype=bool)
    on_b = b_pole == POLE_B
    if on_b.any():
        b_stolen[on_b] = _is_stolen(b_vec[on_b], tagger_model)

    n_new_states = RUNGS_NEW_STATES[rung]
    n_new_rows = n_new_states * 2
    if n_new_states > 0:
        sup = {k: v[:n_new_rows] for k, v in supplement_all.items()}   # 2 rows/state
        s_stolen = _is_stolen(sup["vec"], tagger_model)                # all pole-B by construction
        if not s_stolen.all():
            raise RuntimeError("supplement rows were selected as stolen but re-tag disagrees")
        grid = np.concatenate([b_grid, sup["grid"]]); vec = np.concatenate([b_vec, sup["vec"]])
        amask = np.concatenate([b_amask, sup["amask"]]); mask = np.concatenate([b_mask, sup["mask"]])
        pole = np.concatenate([b_pole, sup["pole"]]); act = np.concatenate([b_act, sup["action"]])
        y = np.concatenate([b_y, sup["margin"]])
        stolen = np.concatenate([b_stolen, s_stolen])
    else:
        grid, vec, amask, mask, pole, act, y, stolen = (
            b_grid, b_vec, b_amask, b_mask, b_pole, b_act, b_y, b_stolen)

    n_stolen_total = int(stolen.sum())
    expected_rows = BASELINE_STOLEN_ROWS + n_new_rows
    if n_stolen_total != expected_rows:
        raise RuntimeError(f"rung {rung}: expected {expected_rows} stolen ROWS, got {n_stolen_total}")
    w = np.ones(len(pole), dtype=np.float32)
    # mixture-mass control: total stolen mass held at the baseline's own 100.0,
    # giving 1.0 / 0.5 / 0.25 per row at rungs 1 / 2 / 3
    w[stolen] = BASELINE_STOLEN_ROWS / n_stolen_total

    b_grid_t, b_vec_t = t(grid, torch.float32), t(vec, torch.float32)
    b_amask_t, b_pole_t = t(amask, torch.float32), t(pole, torch.long)
    b_act_t = t(act, torch.long)
    b_y_t, b_w_t = t(y, torch.float32), t(w, torch.float32)
    b_a1, b_a2 = joint_action_index(b_act_t)

    all_grid = torch.cat([p_grid, b_grid_t]); all_vec = torch.cat([p_vec, b_vec_t])
    all_amask = torch.cat([p_amask, b_amask_t]); all_pole = torch.cat([p_pole, b_pole_t])
    all_a1 = torch.cat([p_a1, b_a1]); all_a2 = torch.cat([p_a2, b_a2])
    all_y = torch.cat([p_y, b_y_t]); all_w = torch.cat([p_w, b_w_t])
    return (all_grid, all_vec, all_amask, all_pole, all_a1, all_a2, all_y, all_w,
            n_stolen_total, float(w[stolen][0]) if n_stolen_total else 1.0)


def fit_one_rung(rung: int, data, val_data, device: str):
    grid, vec, am, pole, a1, a2, y, w, n_stolen, per_row_w = data
    torch.manual_seed(TRAIN_CFG["torch_seed"]); np.random.seed(TRAIN_CFG["torch_seed"])
    model = QPsi().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=TRAIN_CFG["lr"])
    bs, n = TRAIN_CFG["batch_size"], len(y)
    best, best_state, bad = float("inf"), None, 0
    history = []

    def eval_loss(vd):
        vg, vv, va, vp, va1, va2, vy, vw = vd
        model.eval(); tot_l = tot_w = 0.0
        with torch.no_grad():
            for i in range(0, len(vy), 1024):
                s = slice(i, i + 1024)
                pred = model(vg[s].to(device), vv[s].to(device), va[s].to(device),
                            vp[s].to(device), va1[s].to(device), va2[s].to(device))
                ww = vw[s].to(device)
                tot_l += float((ww * (pred - vy[s].to(device)) ** 2).sum()); tot_w += float(ww.sum())
        return tot_l / max(tot_w, 1e-9)

    for ep in range(TRAIN_CFG["max_epochs"]):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            j = perm[i:i + bs]
            pred = model(grid[j].to(device), vec[j].to(device), am[j].to(device),
                        pole[j].to(device), a1[j].to(device), a2[j].to(device))
            ww = w[j].to(device)
            loss = (ww * (pred - y[j].to(device)) ** 2).sum() / ww.sum().clamp_min(1e-9)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        vl = eval_loss(val_data)
        history.append({"epoch": ep, "val_wmse": round(vl, 6)})
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= TRAIN_CFG["early_stop_patience"]:
                break
    model.load_state_dict(best_state)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wpath = OUT_DIR / f"qpsi_rung{rung}.pt"
    torch.save({"config": model.cfg.to_dict(), "state_dict": best_state}, wpath)
    return model, {"rung": rung, "n_stolen_pole_B_rows": n_stolen,
                   "per_stolen_row_weight": per_row_w, "best_val_wmse": best,
                   "n_epochs_run": len(history), "weights_path": str(wpath.relative_to(ROOT))}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    device = a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"

    if RECORD.is_file():
        raise SystemExit(f"REFUSING: {RECORD} exists; D2 fitting is one-shot")
    sel = json.loads(SELECTION.read_text(encoding="utf-8"))
    ordered = [tuple(x) for x in sel["all_selected_in_draw_order"]]
    if len(ordered) != 150:
        raise SystemExit("REFUSING: point selection is not 150 points")

    tagger = QPsi(QPsiConfig())
    print(f"D2 FIT DENSITY ABLATION  {_now()}")
    fit_split = load_split(INNER_FIT_SEEDS)
    val_split = load_split(INNER_VAL_SEEDS)
    if fit_split.heldout_opened or val_split.heldout_opened:
        raise SystemExit("REFUSING: a held-out shard was opened")
    supplement_all = _load_supplement_branch_rows(ordered)
    print(f"  baseline fit rows: plain={len(fit_split.p_margin)} branch={len(fit_split.b_margin)}")
    print(f"  supplement rows available: {len(supplement_all['margin'])} (2/point x 150 points)\n")

    vt = lambda a, dt: torch.as_tensor(a, dtype=dt)
    val_data = (vt(val_split.p_grid, torch.float32), vt(val_split.p_vec, torch.float32),
               vt(val_split.p_amask, torch.float32), vt(val_split.p_pole, torch.long),
               *joint_action_index(vt(val_split.p_action, torch.long)),
               vt(val_split.p_margin, torch.float32), vt(val_split.p_weight, torch.float32))

    results = {}
    for rung in (1, 2, 3):
        print(f"  fitting rung {rung} (+{RUNGS_NEW_STATES[rung]} new states)...", flush=True)
        data = build_rung_tensors(rung, fit_split, supplement_all, tagger)
        model, r = fit_one_rung(rung, data, val_data, device)
        results[str(rung)] = r
        results[str(rung)]["model_ref"] = model   # kept in memory for the eval step
        print(f"    n_stolen={r['n_stolen_pole_B_rows']} weight/row={r['per_stolen_row_weight']:.3f} "
              f"val_wmse={r['best_val_wmse']:.5f} epochs={r['n_epochs_run']}")

    rec = {
        "record": "D2 fit result", "status": "DIAGNOSTIC_ONLY", "utc": _now(),
        "rungs": {k: {kk: vv for kk, vv in v.items() if kk != "model_ref"} for k, v in results.items()},
    }
    RECORD.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n  -> {RECORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
