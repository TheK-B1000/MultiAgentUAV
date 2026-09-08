"""H-OG-PSP smoke 1: is latent-private capacity real, reachable, and isolated?

Implements HOG_PSP_V3_SPEC.json#MECHANISM_SMOKE_MUST_PROVE (architecture half).

The failure mode this exists to catch is the one V3 turns on. If the "private"
branches are not actually isolated, a z0 gradient and a z1 gradient still compete for
the same weights, we reproduce OG-PSP's shared-parameter compromise wearing a new
architecture, and every downstream number looks plausible while measuring nothing.

PASS requires BOTH halves. "Nothing moved" trivially satisfies isolation, so the
positive half is not optional:

    z0 update:  z0 private trunk/head MOVES      <- positive
                z1 private trunk/head BIT-IDENTICAL   <- isolation
                shared backbone MAY MOVE         <- expected, not a failure
    z1 update:  mirrored

Shared parameters moving during a z0 update is CORRECT. Asserting otherwise would
produce a smoke that fails for the wrong reason.

Negative controls, because a guard that cannot fail proves nothing:

    no-LRO fixture      -> zero private tensors; isolation is VACUOUS; must REFUSE
    alpha = 0 fixture   -> gated zero-init path (the V6I22E degenerate equilibrium);
                           must be DETECTED, not silently accepted

Diagnostic. Authorizes nothing. EVAL 11300101..11300132 untouched.

Run:  python experiments/smoke_hog_psp_branch_isolation.py --device cuda
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

SD = ROOT / "artifacts" / "strategic_demand"
SPEC = SD / "sppo" / "HOG_PSP_V3_SPEC.json"
OUT = SD / "sppo" / "HOG_PSP_BRANCH_ISOLATION_SMOKE.json"

LRO_FLAGS = {
    "enable_latent_z_residual": True,
    "latent_population_birth_per_z_action_heads": True,
    "latent_lro_deep_branches": True,
    "latent_z_residual_alpha": 1.0,
}
PRIVATE_MARKERS = ("latent_branch_trunks", "latent_action_heads", "latent_adapters")
SMOKE_SEED, BATCH_STATES, LR = 7311, 48, 1e-3


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def build(device: str, overrides: dict):
    """Fresh K=2 with the given arch overrides, via the existing smoke builder."""
    import experiments.oracle_rehearsal_smoke as S
    original = dict(S.EXP2C_ARCH)
    S.EXP2C_ARCH.update(overrides)
    try:
        return S.build_fresh_k2(device)
    finally:
        S.EXP2C_ARCH.clear()
        S.EXP2C_ARCH.update(original)


def private_names(model, z: int) -> list[str]:
    """Parameter names private to latent z. Indexed by the ModuleList position."""
    return [n for n, _ in model.named_parameters()
            if any(f"{m}.{z}." in n for m in PRIVATE_MARKERS)]


def shared_names(model) -> list[str]:
    return [n for n, _ in model.named_parameters()
            if not any(m in n for m in PRIVATE_MARKERS)]


def snapshot(model) -> dict[str, np.ndarray]:
    return {n: p.detach().cpu().numpy().copy() for n, p in model.named_parameters()}


def changed(before: dict, after: dict, names: list[str]) -> list[str]:
    return [n for n in names if not np.array_equal(before[n], after[n])]


def single_latent_update(model, bank, z: int, device: str) -> dict:
    """One optimizer step driven ONLY by latent z, on real bank states."""
    import torch
    from rl.custom_ppo.strategy_anchor import anchor_loss

    batch = bank.sample(BATCH_STATES)
    rows = np.nonzero(batch["z_idx"] == z)[0]          # this latent's rows only
    t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=device)
    obs = {k: t(v[rows], torch.float32) for k, v in batch["obs"].items()}

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    opt.zero_grad(set_to_none=True)
    loss = anchor_loss(
        model, obs,
        t(batch["teacher_action"][rows], torch.long),
        decision_mask=t(batch["obs"]["agent_mask"][rows], torch.float32).bool(),
        z_idx=t(batch["z_idx"][rows], torch.long))
    loss.backward()

    grads = {n: (None if p.grad is None else float(p.grad.abs().sum()))
             for n, p in model.named_parameters()}
    opt.step()
    return {"loss": float(loss.detach()), "grads": grads, "n_rows": int(len(rows))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; this smoke is one-shot")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: V3 spec is not frozen: {spec['status']!r}")

    import torch
    from rl.paired_rehearsal import load_paired_bank

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    failures: list[str] = []
    print(f"H-OG-PSP BRANCH ISOLATION SMOKE  {_now()}")

    # ---------------------------------------------------------------- structure
    cfg, model = build(device, LRO_FLAGS)
    la = model.latent_actor
    n_priv = sum(p.numel() for n, p in model.named_parameters()
                 if any(m in n for m in PRIVATE_MARKERS))
    n_tot = sum(p.numel() for p in model.parameters())
    structure = {
        "latent_k": int(la.latent_k),
        "branch_trunks": None if la.latent_branch_trunks is None else len(la.latent_branch_trunks),
        "private_action_heads": None if la.latent_action_heads is None else len(la.latent_action_heads),
        "adapters": None if la.latent_adapters is None else len(la.latent_adapters),
        "fixed_alpha_mode": la.latent_adapter_gates is None,
        "latent_z_alpha": float(la._latent_z_alpha),
        "private_params": n_priv, "total_params": n_tot,
        "private_frac": round(n_priv / n_tot, 4),
    }
    print(f"  private capacity: {n_priv:,} of {n_tot:,} params "
          f"({100*n_priv/n_tot:.1f}%), fixed_alpha={structure['fixed_alpha_mode']}")
    if structure["branch_trunks"] != 2 or structure["private_action_heads"] != 2:
        failures.append("LRO deep branches did not instantiate for both latents")
    if not structure["fixed_alpha_mode"]:
        failures.append("gated mode is active; fixed-alpha is required (V6I22E equilibrium)")

    bank = load_paired_bank(include_v2=True, rng_seed=SMOKE_SEED)

    # ------------------------------------------------------------- isolation
    directions = {}
    for z, other in ((0, 1), (1, 0)):
        _, m = build(device, LRO_FLAGS)                 # fresh model per direction
        own, foreign = private_names(m, z), private_names(m, other)
        shared = shared_names(m)
        if not own or not foreign:
            failures.append(f"z{z}: no private parameters found; isolation is vacuous")
            continue

        before = snapshot(m)
        info = single_latent_update(m, bank, z, device)
        after = snapshot(m)

        own_moved = changed(before, after, own)
        foreign_moved = changed(before, after, foreign)
        shared_moved = changed(before, after, shared)
        own_grad = sum(1 for n in own if info["grads"][n])
        foreign_grad = [n for n in foreign if info["grads"][n]]

        if not own_moved:
            failures.append(f"z{z} update did not move z{z}'s private parameters; "
                            "the private path is unreachable")
        if foreign_moved:
            failures.append(f"z{z} update LEAKED into z{other}'s private parameters: "
                            f"{foreign_moved[:3]}")
        if foreign_grad:
            failures.append(f"z{z} update produced gradients on z{other}'s private "
                            f"parameters: {foreign_grad[:3]}")

        directions[f"z{z}_update"] = {
            "rows_used": info["n_rows"], "loss": info["loss"],
            "own_private_tensors": len(own), "own_moved": len(own_moved),
            "own_with_gradient": own_grad,
            "foreign_private_tensors": len(foreign),
            "foreign_moved": len(foreign_moved),
            "foreign_with_gradient": len(foreign_grad),
            "shared_tensors": len(shared), "shared_moved": len(shared_moved),
            "shared_may_move": True,
            "isolated": not foreign_moved and not foreign_grad,
            "reachable": bool(own_moved),
        }
        print(f"  z{z} update: own {len(own_moved)}/{len(own)} moved, "
              f"foreign {len(foreign_moved)}/{len(foreign)} moved, "
              f"shared {len(shared_moved)}/{len(shared)} moved "
              f"-> {'OK' if not foreign_moved and own_moved else 'FAIL'}")

    # ------------------------------------------------- negative control 1: no LRO
    _, plain = build(device, {})
    plain_priv = [n for n, _ in plain.named_parameters()
                  if any(m in n for m in PRIVATE_MARKERS)]
    nc1_detects = len(plain_priv) == 0
    if not nc1_detects:
        failures.append("no-LRO fixture unexpectedly has private tensors; "
                        "the private-capacity check cannot distinguish the treatment")
    print(f"  [NC1] no-LRO fixture: {len(plain_priv)} private tensors "
          f"-> {'DETECTED as vacuous' if nc1_detects else 'NOT DETECTED'}")

    # -------------------------------------- negative control 2: alpha = 0 (gated)
    _, gated = build(device, {**LRO_FLAGS, "latent_z_residual_alpha": 0.0})
    gl = gated.latent_actor
    zero_init = bool(np.array_equal(
        gl.latent_adapters[0].weight.detach().cpu().numpy(),
        np.zeros_like(gl.latent_adapters[0].weight.detach().cpu().numpy())))
    nc2_detects = (gl.latent_adapter_gates is not None) and zero_init
    if not nc2_detects:
        failures.append("alpha=0 fixture was not detected as the gated zero-init path; "
                        "the V6I22E degenerate equilibrium would pass unnoticed")
    print(f"  [NC2] alpha=0 fixture: gates={'present' if gl.latent_adapter_gates is not None else 'None'}, "
          f"adapters zero-init={zero_init} -> "
          f"{'DETECTED as degenerate' if nc2_detects else 'NOT DETECTED'}")

    verdict = "PASS" if not failures else "FAIL"
    OUT.write_text(json.dumps({
        "record": "H-OG-PSP smoke 1: latent-private branch isolation",
        "status": "SMOKE_RESULT", "utc": _now(), "VERDICT": verdict,
        "implements": "HOG_PSP_V3_SPEC.json#MECHANISM_SMOKE_MUST_PROVE (architecture half)",
        "proves": ("Private capacity exists, is reachable by its own latent's gradient, and "
                   "cannot be moved by the other latent's update. Says NOTHING about whether "
                   "the treatment learns."),
        "structure": structure,
        "isolation": directions,
        "shared_backbone_note": ("Shared parameters moving during a single-latent update is "
                                 "CORRECT and is not asserted against. The invariant is that "
                                 "the OTHER latent's private branch stays bit-identical."),
        "negative_controls": {
            "no_LRO_fixture": {"private_tensors": len(plain_priv), "detected": nc1_detects,
                               "why": "without LRO there are no private branches, so an "
                                      "isolation claim would be vacuously true"},
            "alpha_zero_fixture": {"gates_present": gl.latent_adapter_gates is not None,
                                   "adapters_zero_init": zero_init, "detected": nc2_detects,
                                   "why": "zero-init adapters under gating form the "
                                          "zero-gradient degenerate equilibrium that was the "
                                          "V6I22E root cause"},
        },
        "failures": failures,
        "authorizes": "nothing; trajectory-objective liveness is a separate smoke",
        "EVAL_touched": False,
    }, indent=2), encoding="utf-8")

    print(f"\n  VERDICT: {verdict}")
    for f in failures:
        print(f"    FAIL: {f}")
    print(f"  -> {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
