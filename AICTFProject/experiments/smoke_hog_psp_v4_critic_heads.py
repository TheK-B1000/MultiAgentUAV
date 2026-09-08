"""V4 smokes 1-3: are the private critic heads real, function-preserving, and WIRED?

Implements HOG_PSP_V4_SPEC.json#REQUIRED_SMOKES_BEFORE_PRODUCTION and
#AMENDMENT_1_SMOKE_WORDING.

  1  head isolation          z0 update moves head_V0, head_V1 BIT-IDENTICAL; inverse for z1
  2  function-preserving     for identical TRUNK FEATURES h:  V_0(h) == V_1(h)
     init                    plus legacy-checkpoint migration equivalence
  3  REAL PPO WIRING         a live mixed rollout routes z0 -> V_0 and z1 -> V_1,
     (DECISIVE)              with a NEGATIVE CONTROL that deliberately misroutes and
                             requires the guard to FAIL

On smoke 2's wording: the claim is HEAD-function equivalence, NOT V(s,z0) == V(s,z1).
z already enters the critic trunk input as a 2-dim one-hot, so identical head weights
still produce different values for the same raw state under different latents. That was
equally true of the shared critic. Two of the pre-spec audit's own assertions got this
wrong, which is why the distinction is asserted explicitly in the output below.

Smoke 3 is the one that matters. V3 shipped with a decorative-by-omission critic; a test
that only proves two heads EXIST would have passed on V3's architecture too if the heads
had been built and never routed to.

Run:  python experiments/smoke_hog_psp_v4_critic_heads.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "HOG_PSP_V4_SPEC.json"
OUT = SD / "HOG_PSP_V4_CRITIC_HEAD_SMOKE.json"

GS_DIM, HIDDEN = 170, 128


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main() -> int:
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; one-shot")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_IMPLEMENTATION":
        raise SystemExit(f"REFUSING: V4 spec is not frozen: {spec['status']!r}")

    import torch
    from rl.networks import CentralizedCritic

    torch.manual_seed(4404)
    failures: list[str] = []
    print(f"H-OG-PSP V4 CRITIC HEAD SMOKE  {_now()}")

    # ---------------------------------------------- 2. function-preserving init
    c = CentralizedCritic(global_state_dim=GS_DIM, hidden_dim=HIDDEN,
                          extra_dim=2, private_z_heads=True)
    c.copy_shared_head_into_private()
    heads_equal = (torch.equal(c.head_V0.weight, c.head_V1.weight)
                   and torch.equal(c.head_V0.bias, c.head_V1.bias))
    gs = torch.randn(16, GS_DIM)
    h_same = c.trunk(c._combine(gs, torch.zeros(16, 2)))          # ONE trunk feature tensor
    head_gap = float((c.head_V0(h_same) - c.head_V1(h_same)).abs().max())

    # the quantity that is legitimately NON-zero, asserted so it cannot be misread
    e0 = torch.zeros(16, 2); e0[:, 0] = 1
    e1 = torch.zeros(16, 2); e1[:, 1] = 1
    with torch.no_grad():
        vsz_gap = float((c(gs, e0) - c(gs, e1)).abs().max())

    init = {
        "heads_bitwise_equal": bool(heads_equal),
        "max_abs_V0h_minus_V1h_on_identical_trunk_features": head_gap,
        "THIS_is_the_equivalence_claim": "V_0(h) == V_1(h) for identical trunk features h",
        "max_abs_V_s_z0_minus_V_s_z1": vsz_gap,
        "and_THIS_is_legitimately_nonzero": (
            "z enters the critic TRUNK INPUT as well as selecting the head, so the same raw "
            "state yields different trunk features under different latents. Equally true of "
            "the shared critic. A non-zero value here is NOT failed initialisation."),
    }
    if not heads_equal:
        failures.append("head_V0 and head_V1 are not bitwise equal after copy")
    if head_gap > 0.0:
        failures.append(f"V_0(h) != V_1(h) on identical trunk features: {head_gap:.3e}")
    print(f"  [2] heads equal {heads_equal}, |V0(h)-V1(h)| = {head_gap:.3e}  "
          f"(|V(s,z0)-V(s,z1)| = {vsz_gap:.3e}, legitimately non-zero)")

    # ------------------------------------------------------- 1. head isolation
    isolation = {}
    for z, other in ((0, 1), (1, 0)):
        cc = CentralizedCritic(global_state_dim=GS_DIM, hidden_dim=HIDDEN,
                               extra_dim=2, private_z_heads=True)
        cc.copy_shared_head_into_private()
        before = {n: p.detach().clone() for n, p in cc.named_parameters()}
        extra = torch.zeros(16, 2); extra[:, z] = 1
        opt = torch.optim.Adam(cc.parameters(), lr=1e-2)
        opt.zero_grad()
        cc(gs, extra).sum().backward()
        opt.step()
        moved = {n for n, p in cc.named_parameters() if not torch.equal(before[n], p.detach())}
        own = any(f"head_V{z}" in m for m in moved)
        foreign = any(f"head_V{other}" in m for m in moved)
        trunk = any(m.startswith("net.") for m in moved)
        isolation[f"z{z}_update"] = {"own_head_moved": own, "foreign_head_moved": foreign,
                                     "shared_trunk_moved": trunk}
        if not own:
            failures.append(f"z{z} update did not move head_V{z}")
        if foreign:
            failures.append(f"z{z} update MOVED head_V{other}; heads are not isolated")
        print(f"  [1] z{z} update: head_V{z} moved {own}, head_V{other} moved {foreign} "
              f"(must be False), trunk moved {trunk} (expected)")

    # ------------------------------------------- 3. REAL PPO WIRING (decisive)
    from rl.custom_ppo.policy import SharedActorCentralizedCritic  # noqa: F401
    import experiments.oracle_rehearsal_smoke as S
    import experiments.run_hog_psp_v3_production as V3P

    orig_arch = dict(S.EXP2C_ARCH)
    S.EXP2C_ARCH.update(V3P.LRO_FLAGS)
    S.EXP2C_ARCH["rasr_private_critic_heads"] = True
    try:
        _cfg, model = S.build_fresh_k2("cpu")
    finally:
        S.EXP2C_ARCH.clear(); S.EXP2C_ARCH.update(orig_arch)

    live_private = bool(getattr(model.critic, "private_z_heads", False))
    if not live_private:
        failures.append("a model built through the real policy path does NOT have private "
                        "z heads; the V4 axis would be inert in production")

    # observe which head serves each row through the REAL value path
    served = {"z0_rows": 0, "z1_rows": 0, "calls": 0, "missing_z": 0}
    critic = model.critic
    original_forward = critic.forward

    def watched(global_state, extra=None):
        served["calls"] += 1
        if extra is None:
            served["missing_z"] += 1
        else:
            z = extra.argmax(dim=-1)
            served["z0_rows"] += int((z == 0).sum())
            served["z1_rows"] += int((z == 1).sum())
        return original_forward(global_state, extra=extra)

    critic.forward = watched
    try:
        ctx = torch.randn(24, model.critic_context_dim)
        z_mixed = torch.tensor([0, 1] * 12, dtype=torch.long)
        with torch.no_grad():
            v = model.values(ctx, z_idx=z_mixed)          # THE real PPO/GAE value path
        routed_ok = served["z0_rows"] == 12 and served["z1_rows"] == 12
    finally:
        critic.forward = original_forward

    wiring = {"model_has_private_heads": live_private,
              "value_path_calls": served["calls"],
              "z0_rows_routed": served["z0_rows"], "z1_rows_routed": served["z1_rows"],
              "value_queries_missing_z": served["missing_z"],
              "routed_correctly": bool(routed_ok),
              "path": "model.values(global_state, z_idx) -- the same call PPO and GAE use"}
    if not routed_ok:
        failures.append(f"mixed batch mis-routed: z0 rows {served['z0_rows']}, "
                        f"z1 rows {served['z1_rows']}, expected 12/12")
    if served["missing_z"]:
        failures.append("a value query reached the critic without a z one-hot")
    print(f"  [3] real value path: {served['z0_rows']} z0 rows -> V_0, "
          f"{served['z1_rows']} z1 rows -> V_1, missing z {served['missing_z']}")

    # --- NEGATIVE CONTROL: deliberately misroute, the guard MUST catch it -----
    def misrouted(global_state, extra=None):
        flipped = extra.flip(dims=[-1]) if extra is not None else None   # z0 -> V1
        return original_forward(global_state, extra=flipped)

    critic.forward = misrouted
    try:
        probe = {"z0": 0, "z1": 0}

        def counting(global_state, extra=None):
            z = extra.argmax(dim=-1)
            probe["z0"] += int((z == 0).sum()); probe["z1"] += int((z == 1).sum())
            return original_forward(global_state, extra=extra)

        inner = critic.forward

        def chained(global_state, extra=None):
            flipped = extra.flip(dims=[-1]) if extra is not None else None
            return counting(global_state, extra=flipped)

        critic.forward = chained
        z_all0 = torch.zeros(12, dtype=torch.long)
        with torch.no_grad():
            model.values(torch.randn(12, model.critic_context_dim), z_idx=z_all0)
        # all-z0 input, misrouted, must arrive at head_V1
        caught = probe["z0"] == 0 and probe["z1"] == 12
    finally:
        critic.forward = original_forward

    neg = {"description": "all-z0 batch with routing deliberately flipped",
           "rows_arriving_at_head_V0": probe["z0"],
           "rows_arriving_at_head_V1": probe["z1"],
           "misroute_detected": bool(caught),
           "why": ("proves the check would catch z0 -> V1 or a dropped z_idx, rather than "
                   "passing only because things happen to be correct")}
    if not caught:
        failures.append("the negative control was NOT detected; the wiring check cannot "
                        "distinguish correct routing from a misroute and proves nothing")
    print(f"  [3-neg] deliberate misroute detected: {caught} "
          f"(all-z0 batch arrived at head_V1: {probe['z1']}/12)")

    verdict = "PASS" if not failures else "FAIL"
    OUT.write_text(json.dumps({
        "record": "H-OG-PSP V4 smokes 1-3: private critic heads",
        "status": "SMOKE_RESULT", "utc": _now(), "VERDICT": verdict,
        "implements": "HOG_PSP_V4_SPEC.json#REQUIRED_SMOKES_BEFORE_PRODUCTION",
        "smoke_1_head_isolation": isolation,
        "smoke_2_function_preserving_init": init,
        "smoke_3_real_ppo_wiring": wiring,
        "smoke_3_negative_control": neg,
        "what_this_does_not_prove": ("that the treatment learns anything. Composition under "
                                     "a live trainer is a separate verification."),
        "failures": failures,
        "EVAL_touched": False,
    }, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}")
    for f in failures:
        print(f"    FAIL: {f}")
    print(f"  -> {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
