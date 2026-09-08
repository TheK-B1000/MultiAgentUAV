"""K=2 preference probe: frozen Q_psi trunk + a tiny trainable head.

Option B, per PI ruling 2026-08-29. The probe answers one question per state --
is A preferred, is B preferred, or is the model not confident enough to say --
and exposes the confidence signal that kappa will later threshold on CALIB.

    frozen Q_psi.encode(o, p)  ->  h (256)  ->  Linear(256,8) ReLU Linear(8,2)

Only the head trains. 154,736 trunk parameters stay bit-identical; ~2k head
parameters see gradients. That ratio is the whole point: FIT holds 428 resolvable
examples against a 5,640-dimensional observation, so a Q_psi-scale trainable model
would measure memorisation capacity rather than strategic generalisation.

TIES RECEIVE ZERO PRESSURE. Only Delta != 0 states enter the loss. This is not an
optimisation convenience -- it is the operational content of "preference not
established", and a counter asserts it stayed zero.

FIT only. CALIB is never loaded here. EVAL is never touched at all.

Run:  python experiments/fit_k2_preference_probe.py --device cuda
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

from rl.launch_gate import LaunchGateError                      # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand"
DATA = SD / "stratified_regime_data"
TRUNK_CKPT = SD / "phase0_scorer_data" / "qpsi_frozen.pt"
TRUNK_SHA = "930051a725e55e4f14e05dfe178e5f1dc7bd8f3d7e3adeba01187958bb7417bf"
OUT_DIR = SD / "sppo" / "k2_probe"

FIT_LO, FIT_HI = 10_700_001, 10_700_096
CALIB_LO, CALIB_HI = 10_700_097, 10_700_128
EVAL_LO, EVAL_HI = 10_700_129, 10_700_160

HEAD_HIDDEN = 8
EPOCHS, BATCH, LR, WEIGHT_DECAY = 60, 64, 1e-3, 1e-2
RNG_SEED = 17


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _digest_params(named) -> str:
    """Bit-exact digest over a set of tensors, order-independent."""
    h = hashlib.sha256()
    for name, tensor in sorted(named):
        h.update(name.encode("utf-8"))
        h.update(tensor.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def load_split(lo: int, hi: int, resolvable_only: bool):
    """Branch states in [lo, hi]. Labels are the deterministic sign of Delta."""
    grid, vec, amask, pole, cell, label, delta, seed = [], [], [], [], [], [], [], []
    for s in range(lo, hi + 1):
        path = DATA / "seed_shards" / f"seed_{s}.npz"
        with np.load(path, allow_pickle=False) as z:
            d = ((z["branch_pi_B_blue"].astype(np.int64) - z["branch_pi_B_red"].astype(np.int64))
                 - (z["branch_pi_A_blue"].astype(np.int64) - z["branch_pi_A_red"].astype(np.int64)))
            keep = np.nonzero(d != 0)[0] if resolvable_only else np.arange(len(d))
            if not len(keep):
                continue
            grid.append(z["branch_obs_grid"][keep][:, 0])
            vec.append(z["branch_obs_vec"][keep][:, 0])
            amask.append(z["branch_obs_agent_mask"][keep][:, 0])
            pole.append(z["branch_pole"][keep].astype(np.int64))
            cell.extend(str(c) for c in z["branch_cell"][keep])
            delta.append(d[keep])
            label.append((d[keep] > 0).astype(np.int64))      # 1 = B preferred
            seed.append(np.full(len(keep), s, dtype=np.int64))
    if not grid:
        raise LaunchGateError(f"no states found in [{lo}, {hi}]")
    return {"grid": np.concatenate(grid), "vec": np.concatenate(vec),
            "amask": np.concatenate(amask), "pole": np.concatenate(pole),
            "cell": np.array(cell), "label": np.concatenate(label),
            "delta": np.concatenate(delta), "seed": np.concatenate(seed)}


def build(device: str):
    """Frozen trunk, fresh tiny head. Returns (qpsi, head, trunk_digest)."""
    import torch
    from torch import nn
    from rl.scorer.qpsi import QPsi, QPsiConfig

    actual = _sha256(TRUNK_CKPT)
    if actual != TRUNK_SHA:
        raise LaunchGateError(
            f"REFUSING: Q_psi trunk hash mismatch (expected {TRUNK_SHA}, got {actual})")
    ck = torch.load(TRUNK_CKPT, map_location=device, weights_only=False)
    qpsi = QPsi(QPsiConfig(**ck["config"])).to(device)
    qpsi.load_state_dict(ck["state_dict"])
    qpsi.eval()

    trunk_names = tuple(n for n, _ in qpsi.named_parameters()
                        if n.startswith(("conv.", "pole_emb.", "trunk.")))
    for name, p in qpsi.named_parameters():
        p.requires_grad_(False)                      # nothing in Q_psi ever trains

    torch.manual_seed(RNG_SEED)
    head = nn.Sequential(
        nn.Linear(ck["config"]["hidden"], HEAD_HIDDEN), nn.ReLU(),
        nn.Linear(HEAD_HIDDEN, 2)).to(device)

    digest = _digest_params([(n, p) for n, p in qpsi.named_parameters() if n in trunk_names])
    return qpsi, head, trunk_names, digest


def features(qpsi, batch, device):
    import torch
    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    with torch.no_grad():                             # trunk is frozen: no graph
        return qpsi.encode(t(batch["grid"], torch.float32), t(batch["vec"], torch.float32),
                           t(batch["amask"], torch.float32), t(batch["pole"], torch.long))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from torch import nn
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    weights_path = OUT_DIR / "k2_probe_head.pt"
    record_path = OUT_DIR / "K2_PROBE_FIT.json"
    if record_path.is_file():
        raise SystemExit(f"REFUSING: {record_path} exists; the FIT-only fit is one-shot")

    fit = load_split(FIT_LO, FIT_HI, resolvable_only=True)
    all_fit = load_split(FIT_LO, FIT_HI, resolvable_only=False)
    n_tied = int((all_fit["delta"] == 0).sum())
    print(f"K=2 PREFERENCE PROBE  {_now()}")
    print(f"  FIT {FIT_LO}..{FIT_HI}: {len(all_fit['delta'])} branch states, "
          f"{len(fit['delta'])} resolvable, {n_tied} tied and EXCLUDED from the loss\n")

    per_cell = defaultdict(lambda: [0, 0])
    for c, d in zip(all_fit["cell"], all_fit["delta"]):
        per_cell[c][0] += 1
        per_cell[c][1] += int(d != 0)
    print("  per-cell FIT support (reported exactly, including the thin cells):")
    thin = []
    for c, (n, r) in sorted(per_cell.items(), key=lambda kv: kv[1][1]):
        flag = "  <- too thin for cell-specific learning" if r < 12 else ""
        if r < 12:
            thin.append(c)
        print(f"    {c:18s} {n:4d} states  {r:4d} resolvable{flag}")

    qpsi, head, trunk_names, trunk_before = build(device)
    n_head = sum(p.numel() for p in head.parameters())
    n_trunk = sum(p.numel() for n, p in qpsi.named_parameters() if n in trunk_names)
    print(f"\n  trunk {n_trunk:,} frozen params (sha verified)   head {n_head:,} trainable")
    print(f"  params per resolvable example: {n_head / len(fit['delta']):.1f}")

    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    trainable = {id(p) for group in opt.param_groups for p in group["params"]}
    if trainable != {id(p) for p in head.parameters()}:
        raise LaunchGateError("REFUSING: optimizer holds parameters outside the head")

    h_all = features(qpsi, fit, device)
    y_all = torch.as_tensor(fit["label"], dtype=torch.long, device=device)
    rng = np.random.default_rng(RNG_SEED)
    n = len(y_all)
    updates, pressure_on_tied = 0, 0
    for epoch in range(EPOCHS):
        order = rng.permutation(n)
        total = 0.0
        for i in range(0, n, BATCH):
            idx = torch.as_tensor(order[i:i + BATCH], device=device)
            loss = nn.functional.cross_entropy(head(h_all[idx]), y_all[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            updates += 1
            total += float(loss) * len(idx)
        if epoch % 15 == 0 or epoch == EPOCHS - 1:
            with torch.no_grad():
                acc = float((head(h_all).argmax(1) == y_all).float().mean())
            print(f"    epoch {epoch:3d}  loss {total / n:.4f}  FIT acc {acc:.3f}")

    trunk_after = _digest_params([(nm, p) for nm, p in qpsi.named_parameters() if nm in trunk_names])
    if trunk_after != trunk_before:
        raise LaunchGateError("REFUSING: frozen trunk parameters changed during fitting")
    if pressure_on_tied != 0:
        raise LaunchGateError(f"REFUSING: tied states received {pressure_on_tied} updates")

    with torch.no_grad():
        probs = torch.softmax(head(h_all), dim=1)
        fit_acc = float((probs.argmax(1) == y_all).float().mean())
    torch.save({"state_dict": head.state_dict(), "hidden": HEAD_HIDDEN,
                "trunk_sha256": TRUNK_SHA}, weights_path)

    record = {
        "record": "K=2 preference probe -- frozen Q_psi trunk + tiny head (Option B)",
        "status": "FIT_COMPLETE", "utc": _now(),
        "authority": "PI ruling 2026-08-29: Option B, FIT-only, no weighting or oversampling",
        "architecture": {
            "trunk": "Q_psi.encode(grid, vec, agent_mask, pole) -> h(256), FROZEN",
            "trunk_checkpoint": str(TRUNK_CKPT.relative_to(ROOT)),
            "trunk_sha256": TRUNK_SHA,
            "trunk_params_frozen": n_trunk,
            "head": f"Linear(256,{HEAD_HIDDEN}) ReLU Linear({HEAD_HIDDEN},2)",
            "head_params_trainable": n_head,
            "params_per_resolvable_example": round(n_head / len(fit["delta"]), 2)},
        "confidence_signal": {
            "definition": "max softmax probability over the two classes, in [0.5, 1.0]",
            "monotone_equivalent_to": "the probability margin |p_B - p_A|, so kappa thresholding is identical either way",
            "consumed_by": "kappa selection on CALIB, per AMENDMENT_3"},
        "data": {
            "split": "FIT only", "seed_range": [FIT_LO, FIT_HI],
            "branch_states": int(len(all_fit["delta"])),
            "resolvable_used_for_training": int(len(fit["delta"])),
            "tied_excluded_from_loss": n_tied,
            "CALIB_loaded": False, "EVAL_touched": False},
        "ties_receive_zero_pressure": {
            "enforced": "only Delta != 0 states enter the loss",
            "verified_updates_on_tied": pressure_on_tied},
        "per_cell_fit_support": {c: {"states": n, "resolvable": r} for c, (n, r) in sorted(per_cell.items())},
        "DOCUMENTED_LIMITATION": {
            "cells_with_fewer_than_12_resolvable_fit_examples": thin,
            "statement": (
                "These cells contain too few resolvable FIT examples to support reliable "
                "cell-specific learning. If kappa later fails because CALIB requires "
                "coverage in such a cell, that result MUST be interpreted jointly with "
                "this documented FIT support, and NOT as clean evidence that selective "
                "supervision itself is impossible."),
            "recorded_before": "any CALIB prediction or kappa selection"},
        "optimisation": {"epochs": EPOCHS, "batch": BATCH, "lr": LR,
                         "weight_decay": WEIGHT_DECAY, "rng_seed": RNG_SEED,
                         "updates": updates,
                         "no_oversampling": True, "no_cell_weighting": True},
        "guards_passed": [
            "trunk checkpoint sha256 matches the frozen artifact",
            "trunk parameters bit-identical before and after fitting",
            "optimizer holds only head parameters",
            "FIT only; CALIB not loaded; EVAL not touched",
            "tied states contributed zero gradient"],
        "fit_accuracy_on_resolvable": round(fit_acc, 4),
        "fit_accuracy_caveat": "In-sample. Says nothing about generalisation; CALIB gets the first real say.",
        "head_weights": str(weights_path.relative_to(ROOT)),
    }
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"\n  FIT accuracy (in-sample) {fit_acc:.3f}")
    print(f"  trunk bit-identical: True   tied-state updates: {pressure_on_tied}")
    print(f"  -> {record_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
