"""State-level resolvability representation audit. Implements the frozen spec.

Three arms, all predicting resolvable vs preference-not-established, all trained on
FIT and scored on CALIB:

    arm 1  cell identity only          -- what the global AUC gets for free
    arm 2  frozen Q_psi features       -- probe 2's representation
    arm 3  small dedicated encoder     -- IDENTICAL raw inputs to arm 2

Arms 2 and 3 differ ONLY in representation. Same grid, same vec, same agent_mask,
same pole. Giving arm 3 more inputs would confound representation with information.

PRIMARY metric is pair-weighted WITHIN-cell AUC, not global AUC. Global is inflated
by between-cell base rates that per-cell thresholds provably cannot exploit
(o*_min = 0.3455). Within-cell is the quantity that determines whether abstention is
achievable at all.

Diagnostic only. No kappa. No gates. No EVAL. No new seeds.

Run:  python experiments/resolvability_representation_audit.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.fit_k2_preference_probe import (                 # noqa: E402
    CALIB_HI, CALIB_LO, FIT_HI, FIT_LO, HEAD_HIDDEN, OUT_DIR, build, features, load_split,
)
from rl.launch_gate import LaunchGateError                        # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand"
SPEC = SD / "sppo" / "RESOLVABILITY_REPRESENTATION_AUDIT_SPEC.json"
OUT = SD / "sppo" / "RESOLVABILITY_REPRESENTATION_AUDIT.json"

EPOCHS, BATCH, LR, WEIGHT_DECAY = 60, 64, 1e-3, 1e-2
RNG_SEED = 29


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def auc(score: np.ndarray, label: np.ndarray) -> float:
    order = np.argsort(score)
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = float(label.sum()); n0 = float(len(label) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[label == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def within_cell_auc(score, label, cells):
    """Pair-weighted mean within-cell AUC, plus the per-cell table."""
    per, num, den = {}, 0.0, 0.0
    for c in sorted(set(cells)):
        m = cells == c
        s, y = score[m], label[m]
        n1, n0 = int(y.sum()), int((1 - y).sum())
        if n1 == 0 or n0 == 0:
            per[c] = None
            continue
        a = auc(s, y)
        per[c] = round(a, 4)
        num += a * n1 * n0
        den += n1 * n0
    return (num / den if den else float("nan")), per


def _train_head(h_fit, y_fit, in_dim, device, seed):
    import torch
    from torch import nn
    torch.manual_seed(seed)
    head = nn.Sequential(nn.Linear(in_dim, HEAD_HIDDEN), nn.ReLU(),
                         nn.Linear(HEAD_HIDDEN, 1)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    t = torch.as_tensor(y_fit, dtype=torch.float32, device=device)
    rng = np.random.default_rng(seed)
    n = len(y_fit)
    for _ in range(EPOCHS):
        for i in range(0, n, BATCH):
            idx = torch.as_tensor(rng.permutation(n)[i:i + BATCH], device=device)
            loss = nn.functional.binary_cross_entropy_with_logits(
                head(h_fit[idx]).squeeze(-1), t[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    return head, sum(p.numel() for p in head.parameters())


class SmallEncoder:
    """Arm 3: a deliberately small conv+MLP on the same raw inputs as arm 2."""

    def __init__(self, device, seed):
        import torch
        from torch import nn
        torch.manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv2d(14, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 16, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten()).to(device)
        self.mlp = nn.Sequential(nn.Linear(16 + 40 + 2 + 4, 16), nn.ReLU(),
                                 nn.Linear(16, 1)).to(device)
        self.device = device

    def parameters(self):
        return list(self.net.parameters()) + list(self.mlp.parameters())

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def __call__(self, batch):
        import torch
        t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=self.device)
        n = len(batch["pole"])
        g = self.net(t(batch["grid"], torch.float32).reshape(n, 14, 20, 20))
        pole = torch.nn.functional.one_hot(t(batch["pole"], torch.long), 2).float()
        x = torch.cat([g, t(batch["vec"], torch.float32).reshape(n, -1),
                       t(batch["amask"], torch.float32).reshape(n, -1),
                       pole, pole], dim=-1)
        return self.mlp(x).squeeze(-1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from torch import nn
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN -- SPEC_FROZEN_BEFORE_IMPLEMENTATION":
        raise LaunchGateError("REFUSING: audit spec is not frozen")
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; this audit is one-shot")

    fit = load_split(FIT_LO, FIT_HI, resolvable_only=False)
    calib = load_split(CALIB_LO, CALIB_HI, resolvable_only=False)
    y_fit = (fit["delta"] != 0).astype(np.int64)
    y_cal = (calib["delta"] != 0).astype(np.int64)
    print(f"RESOLVABILITY REPRESENTATION AUDIT  {_now()}")
    print(f"  FIT {len(y_fit)} states ({y_fit.sum()} resolvable)   "
          f"CALIB {len(y_cal)} states ({y_cal.sum()} resolvable)")
    print("  PRIMARY metric: pair-weighted WITHIN-cell AUC on CALIB\n")

    arms = {}

    # ---- arm 1: cell identity only -------------------------------------------
    rate = {}
    for c in sorted(set(fit["cell"])):
        m = fit["cell"] == c
        rate[c] = float(y_fit[m].mean()) if m.any() else 0.0
    s1 = np.array([rate.get(c, 0.0) for c in calib["cell"]])
    w1, per1 = within_cell_auc(s1, y_cal, calib["cell"])
    arms["arm_1_cell_only"] = {
        "input": "cell identity (pole x regime x horizon) -> FIT base rate",
        "trainable_params": 0,
        "global_auc": round(auc(s1, y_cal), 4),
        "within_cell_auc_pair_weighted": round(w1, 4),
        "note": "within-cell AUC is 0.5 by construction; the score is constant inside a cell"}

    # ---- arm 2: frozen Q_psi features ----------------------------------------
    qpsi, _, _, _ = build(device)
    h_fit = features(qpsi, fit, device)
    h_cal = features(qpsi, calib, device)
    head2, n2 = _train_head(h_fit, y_fit, h_fit.shape[1], device, RNG_SEED)
    with torch.no_grad():
        s2 = torch.sigmoid(head2(h_cal).squeeze(-1)).cpu().numpy()
    w2, per2 = within_cell_auc(s2, y_cal, calib["cell"])
    arms["arm_2_frozen_qpsi"] = {
        "input": "Q_psi.encode(grid, vec, agent_mask, pole) -> h(256), FROZEN",
        "trainable_params": n2,
        "global_auc": round(auc(s2, y_cal), 4),
        "within_cell_auc_pair_weighted": round(w2, 4),
        "probe2_reference": {"global_auc": 0.7041, "within_cell_auc": 0.5449}}

    # ---- arm 3: small dedicated encoder, identical inputs ---------------------
    enc = SmallEncoder(device, RNG_SEED)
    opt = torch.optim.AdamW(enc.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    t_fit = torch.as_tensor(y_fit, dtype=torch.float32, device=device)
    rng = np.random.default_rng(RNG_SEED)
    n = len(y_fit)
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        for i in range(0, n, BATCH):
            idx = order[i:i + BATCH]
            sub = {k: fit[k][idx] for k in ("grid", "vec", "amask", "pole")}
            loss = nn.functional.binary_cross_entropy_with_logits(
                enc(sub), t_fit[torch.as_tensor(idx, device=device)])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    with torch.no_grad():
        s3 = torch.sigmoid(enc({k: calib[k] for k in ("grid", "vec", "amask", "pole")})).cpu().numpy()
    w3, per3 = within_cell_auc(s3, y_cal, calib["cell"])
    arms["arm_3_dedicated_encoder"] = {
        "input": "IDENTICAL raw inputs to arm 2 (grid, vec, agent_mask, pole)",
        "architecture": "Conv(14->16,s2) ReLU Conv(16->16,s2) ReLU GAP -> MLP(62->16->1)",
        "trainable_params": enc.n_params(),
        "global_auc": round(auc(s3, y_cal), 4),
        "within_cell_auc_pair_weighted": round(w3, 4)}

    print(f"  {'arm':28s} {'params':>8s} {'global AUC':>11s} {'WITHIN-cell AUC':>16s}")
    for k, v in arms.items():
        print(f"  {k:28s} {v['trainable_params']:8d} {v['global_auc']:11.4f} "
              f"{v['within_cell_auc_pair_weighted']:16.4f}")

    delta = w3 - w2
    if w3 >= 0.65:
        verdict = "QPSI_REPRESENTATION_WAS_THE_BOTTLENECK"
        reading = (
            f"Arm 3 reaches within-cell AUC {w3:.4f} from the SAME inputs arm 2 had, "
            f"against {w2:.4f}. The state-level signal is present in the observation and "
            "was being lost by the frozen Q_psi representation. A dedicated resolvability "
            "representation is justified.")
    elif w3 <= 0.58:
        verdict = "REPRESENTATION_IS_NOT_THE_BINDING_LIMIT"
        reading = (
            f"Arm 3 reaches only {w3:.4f} within-cell, against arm 2's {w2:.4f}, from "
            "identical inputs. Enlarging or unfreezing Q_psi is unlikely to help. The "
            "limit lies in observation sufficiency or in the target itself -- and per the "
            "frozen spec's ARM_4 note, those two remain ENTANGLED because no "
            "privileged-state collection exists.")
    else:
        verdict = "AMBIGUOUS"
        reading = (f"Arm 3 at {w3:.4f} against arm 2's {w2:.4f} falls between the "
                   "preregistered forks. Neither conclusion is licensed.")

    rec = {
        "record": "State-level resolvability representation audit",
        "status": "DIAGNOSTIC_RESULT -- no kappa, no gates, no EVAL, no new seeds",
        "utc": _now(),
        "implements": "RESOLVABILITY_REPRESENTATION_AUDIT_SPEC.json",
        "primary_metric": "pair-weighted within-cell AUC on CALIB",
        "why_within_cell": (
            "Global AUC is inflated by between-cell base rates, which per-cell thresholds "
            "provably cannot exploit -- the oracle bound was o*_min = 0.3455 against a "
            "0.10 ceiling."),
        "arms": arms,
        "arm3_minus_arm2_within_cell": round(delta, 4),
        "per_cell_within_auc": {"arm_1": per1, "arm_2": per2, "arm_3": per3},
        "VERDICT": verdict,
        "reading": reading,
        "arm_4_full_state_oracle": "INFEASIBLE -- no privileged state was ever collected; see the frozen spec",
        "does_not_alter": "the CALIBRATION_FAILED verdicts of probe 1 or probe 2",
        "EVAL_touched": False,
    }
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n  arm3 - arm2 (within-cell) = {delta:+.4f}")
    print(f"  VERDICT: {verdict}\n  {reading}")
    print(f"  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
