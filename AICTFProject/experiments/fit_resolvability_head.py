"""Probe 2: add a resolvability head. Implements TWO_HEAD_RESOLVABILITY_PROBE_SPEC.

Probe 1 could predict WHICH strategy is preferred but not WHETHER a preference
exists -- its direction confidence separated resolvable from tied states at AUC
0.415, pointing the wrong way. It had never been trained to make that distinction.

Probe 2 supervises it explicitly, and changes nothing else:

    frozen Q_psi trunk -> h(256) -+-> direction head    (probe 1 weights, FROZEN)
                                  +-> resolvability head (the only new trainable part)

    direction head      trained on resolvable FIT only, ties give ZERO pressure
    resolvability head  trained on ALL FIT states, label = (Delta != 0)

Tied states still never say "become A" or "become B". They say only "there is not
evidence here to establish either preference".

kappa will then threshold P(resolvable | s) instead of P(my guess is right).

FIT only. CALIB is not loaded here. EVAL is never touched.

Run:  python experiments/fit_resolvability_head.py --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fit_k2_preference_probe import (                 # noqa: E402
    FIT_HI, FIT_LO, HEAD_HIDDEN, OUT_DIR, TRUNK_SHA,
    _digest_params, _sha256, build, features, load_split,
)
from rl.launch_gate import LaunchGateError                        # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand"
SPEC = SD / "sppo" / "TWO_HEAD_RESOLVABILITY_PROBE_SPEC.json"
DIR_HEAD = OUT_DIR / "k2_probe_head.pt"
RES_HEAD = OUT_DIR / "resolvability_head.pt"
RECORD = OUT_DIR / "PROBE2_FIT.json"

EPOCHS, BATCH, LR, WEIGHT_DECAY = 60, 64, 1e-3, 1e-2
RNG_SEED = 23


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_spec() -> dict:
    if not SPEC.is_file():
        raise LaunchGateError("REFUSING: probe-2 spec is not frozen")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN -- PROBE_2_SPEC_FROZEN_BEFORE_IMPLEMENTATION":
        raise LaunchGateError(f"REFUSING: unexpected spec status {spec['status']!r}")
    if spec["SCOPE"]["EVAL"] != "NOT TOUCHED":
        raise LaunchGateError("REFUSING: spec does not bar EVAL")
    return spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from torch import nn
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    spec = _require_spec()
    if RECORD.is_file():
        raise SystemExit(f"REFUSING: {RECORD} exists; the probe-2 fit is one-shot")
    if not DIR_HEAD.is_file():
        raise SystemExit("REFUSING: probe-1 direction head not found")

    # every FIT state, ties included -- that is the whole point of this probe
    fit = load_split(FIT_LO, FIT_HI, resolvable_only=False)
    y_res = (fit["delta"] != 0).astype(np.int64)
    n_res, n_tie = int(y_res.sum()), int((1 - y_res).sum())
    print(f"PROBE 2 -- RESOLVABILITY HEAD  {_now()}")
    print(f"  FIT {FIT_LO}..{FIT_HI}: {len(y_res)} states, {n_res} resolvable, {n_tie} tied")
    print(f"  positive rate {y_res.mean():.4f}   (no class weighting, per the frozen spec)\n")

    qpsi, dir_head, trunk_names, trunk_before = build(device)
    dir_ck = torch.load(DIR_HEAD, map_location=device, weights_only=False)
    dir_head.load_state_dict(dir_ck["state_dict"])
    dir_head.eval()
    for p in dir_head.parameters():
        p.requires_grad_(False)                      # probe 1's head never trains again
    dir_before = _digest_params(list(dir_head.named_parameters()))

    torch.manual_seed(RNG_SEED)
    res_head = nn.Sequential(
        nn.Linear(256, HEAD_HIDDEN), nn.ReLU(), nn.Linear(HEAD_HIDDEN, 1)).to(device)
    n_new = sum(p.numel() for p in res_head.parameters())
    print(f"  trunk frozen, direction head frozen ({sum(p.numel() for p in dir_head.parameters()):,} params)")
    print(f"  resolvability head {n_new:,} trainable   ({n_new / len(y_res):.1f} params/example)\n")

    opt = torch.optim.AdamW(res_head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    if {id(p) for g in opt.param_groups for p in g["params"]} != {id(p) for p in res_head.parameters()}:
        raise LaunchGateError("REFUSING: optimizer holds parameters outside the resolvability head")

    h_all = features(qpsi, fit, device)
    t_res = torch.as_tensor(y_res, dtype=torch.float32, device=device)
    rng = np.random.default_rng(RNG_SEED)
    n = len(y_res)
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        total = 0.0
        for i in range(0, n, BATCH):
            idx = torch.as_tensor(order[i:i + BATCH], device=device)
            loss = nn.functional.binary_cross_entropy_with_logits(
                res_head(h_all[idx]).squeeze(-1), t_res[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        if epoch % 15 == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                pr = torch.sigmoid(res_head(h_all).squeeze(-1)).cpu().numpy()
            print(f"    epoch {epoch:3d}  loss {total / n:.4f}  FIT AUC {_auc(pr, y_res):.3f}")

    if _digest_params([(nm, p) for nm, p in qpsi.named_parameters() if nm in trunk_names]) != trunk_before:
        raise LaunchGateError("REFUSING: frozen trunk changed during probe-2 fitting")
    if _digest_params(list(dir_head.named_parameters())) != dir_before:
        raise LaunchGateError("REFUSING: probe-1 direction head changed during probe-2 fitting")

    with torch.no_grad():
        pr = torch.sigmoid(res_head(h_all).squeeze(-1)).cpu().numpy()
    fit_auc = _auc(pr, y_res)
    torch.save({"state_dict": res_head.state_dict(), "hidden": HEAD_HIDDEN,
                "trunk_sha256": TRUNK_SHA,
                "direction_head_sha256": _sha256(DIR_HEAD)}, RES_HEAD)

    per_cell = defaultdict(lambda: [0, 0])
    for c, r in zip(fit["cell"], y_res):
        per_cell[c][0] += 1
        per_cell[c][1] += int(r)

    record = {
        "record": "Probe 2 -- resolvability head, FIT only",
        "status": "FIT_COMPLETE", "utc": _now(),
        "implements": "TWO_HEAD_RESOLVABILITY_PROBE_SPEC.json",
        "the_only_change_from_probe_1": "a small binary resolvability head; everything else held fixed",
        "architecture": {
            "trunk": "Q_psi.encode -> h(256), FROZEN",
            "trunk_sha256": TRUNK_SHA,
            "direction_head": "probe 1 weights, REUSED BIT-IDENTICALLY, not trained",
            "direction_head_sha256": _sha256(DIR_HEAD),
            "resolvability_head": f"Linear(256,{HEAD_HIDDEN}) ReLU Linear({HEAD_HIDDEN},1)",
            "resolvability_head_params": n_new,
            "params_per_example": round(n_new / len(y_res), 2)},
        "data": {
            "split": "FIT only", "seed_range": [FIT_LO, FIT_HI],
            "states": int(len(y_res)), "resolvable": n_res, "tied": n_tie,
            "positive_rate": round(float(y_res.mean()), 4),
            "CALIB_loaded": False, "EVAL_touched": False},
        "tie_states_role": (
            "Tied states train the RESOLVABILITY head only. They still exert zero "
            "direction pressure -- the direction head is frozen and received no "
            "gradient at all in this probe."),
        "guards_passed": [
            "frozen spec verified before fitting",
            "trunk sha256 matches; trunk bit-identical after fitting",
            "probe-1 direction head bit-identical after fitting",
            "optimizer holds only resolvability-head parameters",
            "FIT only; CALIB not loaded; EVAL not touched",
            "no class weighting despite 428 vs 681 imbalance"],
        "per_cell_fit_support": {c: {"states": n, "resolvable": r} for c, (n, r) in sorted(per_cell.items())},
        "fit_auc_resolvability": round(fit_auc, 4),
        "fit_auc_caveat": (
            "In-sample. Probe 1's direction confidence achieved AUC 0.415 on CALIB for "
            "this same task without being trained for it. CALIB gets the first real say."),
        "optimisation": {"epochs": EPOCHS, "batch": BATCH, "lr": LR,
                         "weight_decay": WEIGHT_DECAY, "rng_seed": RNG_SEED},
        "resolvability_head_weights": str(RES_HEAD.relative_to(ROOT)),
    }
    RECORD.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\n  FIT AUC (in-sample) {fit_auc:.4f}")
    print("  trunk bit-identical: True   direction head bit-identical: True")
    print(f"  -> {RECORD}")
    return 0


def _auc(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = float(label.sum())
    n0 = float(len(label) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


if __name__ == "__main__":
    raise SystemExit(main())
