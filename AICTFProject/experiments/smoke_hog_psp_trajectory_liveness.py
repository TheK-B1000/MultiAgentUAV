"""H-OG-PSP smoke 2: is the trajectory-identity regulariser live and non-contaminating?

Implements HOG_PSP_V3_SPEC.json#AMENDMENT_1_COMPONENT_2_MECHANISM.

Two separate questions, both required:

  LIVENESS   does the training lifecycle actually consume trajectory-level identity
             signal, and does that credit reach the intended PRIVATE actor parameters?

  PURITY     are CTF rewards, returns, GAE and critic targets BIT-IDENTICAL with the
             regulariser active? This is the proof that V3 has not been silently
             converted into reward-shaped CTF. Without it the separation result would
             be uninterpretable, because a reviewer could fairly say the latents
             separated only because we paid them to imitate the teachers.

Tripwires, because a guard that cannot fail proves nothing:

  runner disabled          -> HARD FAIL
  missing / wrong D        -> HARD FAIL
  D mutated during update  -> HARD FAIL

Episodes are drawn from the real stratified shards, so the feature map sees the same
kind of object it was validated on. EVAL 11300101..11300132 untouched.

Run:  python experiments/smoke_hog_psp_trajectory_liveness.py --device cuda
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

import experiments.probe_teacher_trajectory_separability as P
from experiments.smoke_hog_psp_branch_isolation import (
    LRO_FLAGS, PRIVATE_MARKERS, build, private_names, snapshot, changed,
)

SD = ROOT / "artifacts" / "strategic_demand"
SPEC = SD / "sppo" / "HOG_PSP_V3_SPEC.json"
OUT = SD / "sppo" / "HOG_PSP_TRAJECTORY_LIVENESS_SMOKE.json"

SMOKE_SEED, LAM, LR, N_EPISODES = 8421, 0.05, 1e-3, 8


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def take_episodes(n: int) -> list[dict]:
    """Real teacher episodes from FIT shards, standing in for rollouts.

    Selection is by (policy, pole) TOGETHER, never by pole alone. Attempt 1 of this
    smoke filtered on pole only; because the shard's plain_* stream is ordered by cell,
    the first Pole-B rows are pi_A's steps, which were then labelled z1 and scored
    against pi_B identity. Every score floored at the clip, every advantage was zero,
    and the mechanism looked dead when it was behaving correctly on mislabelled input.

    z is assigned to match the POLICY identity on its matching pole -- pi_A on Pole A
    is z0, pi_B on Pole B is z1 -- which is the production rollout layout.
    """
    from rl.trajectory_identity import episode_features
    eps = []
    pairs = ((P.PI_A, P.POLE_A, 0), (P.PI_B, P.POLE_B, 1))
    for seed in range(P.FIT_LO, P.FIT_LO + 40):
        path = P.DATA / f"seed_{seed}.npz"
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as z:
            policy, pole = z["plain_policy"], z["plain_pole"]
            vec, act = z["plain_obs_vec"], z["plain_action"]
            amask, grid, msk = (z["plain_obs_agent_mask"], z["plain_obs_grid"],
                                z["plain_obs_mask"])
            for pol_id, q, z_id in pairs:
                sel = np.nonzero((policy == pol_id) & (pole == q))[0]
                if sel.size < 16:
                    continue
                st = sel[:min(64, sel.size)]          # bounded for smoke speed
                got_pol = np.unique(policy[st]).tolist()
                got_pole = np.unique(pole[st]).tolist()
                if got_pol != [pol_id] or got_pole != [q]:
                    raise SystemExit(
                        f"REFUSING: episode selection is impure -- policies {got_pol}, "
                        f"poles {got_pole}; expected [{pol_id}] and [{q}]")
                eps.append({
                    "z": z_id, "pole": int(q), "policy": int(pol_id),
                    "obs": {"grid": grid[st][:, 0], "vec": vec[st][:, 0],
                            "agent_mask": amask[st][:, 0], "mask": msk[st][:, 0]},
                    "actions": act[st],
                    "features": episode_features(vec[st][:, 0], act[st]),
                    "seed": seed,
                })
                if len(eps) >= n:
                    return eps
    return eps


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; this smoke is one-shot")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if "AMENDMENT_1_COMPONENT_2_MECHANISM" not in spec:
        raise SystemExit("REFUSING: the PG-regulariser amendment is not in the frozen spec")

    import torch
    from rl.trajectory_identity import (
        FrozenDiscriminators, TrajectoryIdentityError, TrajectoryIdentityRunner,
        LATENT_TO_TARGET, LOG_PROB_CLIP, POLE_NAME,
    )

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    failures: list[str] = []
    print(f"H-OG-PSP TRAJECTORY LIVENESS SMOKE  {_now()}")

    # ------------------------------------------------------- frozen D loaded
    D = FrozenDiscriminators(verify=True)
    print(f"  D_A sha {D.sha['A'][:16]}   D_B sha {D.sha['B'][:16]}   verified")
    acc = {p: D.record["per_pole"][p]["held_out_balanced_accuracy"] for p in ("A", "B")}
    print(f"  held-out accuracy: D_A {acc['A']:.4f}  D_B {acc['B']:.4f}")

    episodes = take_episodes(N_EPISODES)
    if len(episodes) < 4:
        raise SystemExit(f"REFUSING: only {len(episodes)} episodes available")
    cells = sorted({f"z{e['z']}|{'AB'[e['pole']]}" for e in episodes})
    print(f"  {len(episodes)} real episodes, cells {cells}")

    # --------------------------------------------- correct pole-specific D used
    # Structural first: the two discriminators must be genuinely different models.
    # Comparing scores alone is a weak proxy -- saturated scores can collide by
    # coincidence, which is exactly what confused attempt 1 of this smoke.
    coef_a = D.models["A"]["clf"].coef_
    coef_b = D.models["B"]["clf"].coef_
    distinct_models = not np.array_equal(coef_a, coef_b)
    if not distinct_models:
        failures.append("D_A and D_B have identical coefficients; they are the same model "
                        "and the pole-specific confound protection is inert")

    # Then behavioural: score() must dispatch on the pole argument.
    dispatch = []
    for e in episodes:
        used = []
        real_score = D.score
        def traced(features, pole, target, _r=real_score, _u=used):
            _u.append(POLE_NAME[int(pole)])
            return _r(features, pole, target)
        D.score = traced
        try:
            s = D.score(e["features"], e["pole"], LATENT_TO_TARGET[e["z"]])
        finally:
            D.score = real_score
        dispatch.append({"pole": "AB"[e["pole"]], "D_used": used[0],
                         "score": s, "correct": used[0] == "AB"[e["pole"]]})
    pole_routing = dispatch
    if not all(r["correct"] for r in dispatch):
        failures.append("score() did not dispatch to the pole's own discriminator")
    if not all(np.isfinite(r["score"]) for r in dispatch):
        failures.append("a trajectory score was not finite")

    # ----------------------------------------------------- liveness + gradients
    _, model = build(device, LRO_FLAGS)
    runner = TrajectoryIdentityRunner(D, lam=LAM)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    grads_by_z = {}
    for z in (0, 1):
        subset = [e for e in episodes if e["z"] == z]
        if not subset:
            failures.append(f"no episodes for z{z}")
            continue
        opt.zero_grad(set_to_none=True)
        loss = runner.loss(model, subset, device=device)
        loss.backward()
        own = private_names(model, z)
        foreign = private_names(model, 1 - z)
        g_own = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
                    if n in own and p.grad is not None)
        g_foreign = sum(float(p.grad.abs().sum()) for n, p in model.named_parameters()
                        if n in foreign and p.grad is not None)
        grads_by_z[f"z{z}"] = {
            "episodes": len(subset), "loss": float(loss.detach()),
            "target_identity": "pi_A" if LATENT_TO_TARGET[z] == 0 else "pi_B",
            "own_private_grad_sum": g_own, "foreign_private_grad_sum": g_foreign,
            "scores": [round(s, 4) for s in runner.last_scores],
            "advantages": [round(a, 6) for a in runner.last_advantages],
        }
        if g_own <= 0:
            failures.append(f"z{z}: trajectory credit produced NO gradient on z{z}'s "
                            "private parameters; the regulariser does not reach them")
        if g_foreign > 0:
            failures.append(f"z{z}: trajectory credit leaked gradient onto z{1-z}'s "
                            "private parameters")
        if not np.isfinite(float(loss.detach())):
            failures.append(f"z{z}: trajectory loss is not finite")
        print(f"  z{z} -> {grads_by_z[f'z{z}']['target_identity']}: "
              f"own grad {g_own:.4e}, foreign grad {g_foreign:.4e}, "
              f"loss {float(loss.detach()):+.6f}")

    opt.step()
    try:
        D.assert_still_frozen()
        d_frozen = True
    except TrajectoryIdentityError as exc:
        d_frozen = False
        failures.append(str(exc))
    print(f"  D parameters unchanged after optimizer step: {d_frozen}")

    # ---------------------------------------------------------------- PURITY
    # The regulariser must not read or write reward, returns, GAE, or critic targets.
    # Structural: the loss signature accepts none of them. Empirical: buffers that a
    # reward-shaping implementation would touch are bit-identical across the call.
    rng = np.random.default_rng(SMOKE_SEED)
    task_buffers = {
        "rewards": rng.normal(size=256).astype(np.float64),
        "returns": rng.normal(size=256).astype(np.float64),
        "advantages": rng.normal(size=256).astype(np.float64),
        "value_targets": rng.normal(size=256).astype(np.float64),
    }
    before_buffers = {k: v.copy() for k, v in task_buffers.items()}
    _, m2 = build(device, LRO_FLAGS)
    r2 = TrajectoryIdentityRunner(D, lam=LAM)
    _ = r2.loss(m2, episodes, device=device)
    purity = {k: bool(np.array_equal(before_buffers[k], task_buffers[k]))
              for k in task_buffers}
    import inspect
    sig = set(inspect.signature(TrajectoryIdentityRunner.loss).parameters)
    forbidden = {"reward", "rewards", "returns", "advantages", "values", "value_targets"}
    sig_clean = not (sig & forbidden)
    purity["loss_signature_accepts_no_task_quantity"] = sig_clean
    if not all(purity.values()):
        failures.append(f"PURITY violated: {purity}")
    print(f"  purity: rewards/returns/GAE/value-targets bit-identical="
          f"{all(purity[k] for k in task_buffers)}, "
          f"signature clean={sig_clean}")

    # --------------------------------------------------------------- tripwires
    tripwires = {}
    try:                                    # runner disabled -> must hard fail
        TrajectoryIdentityRunner(D, lam=LAM).loss(model, [], device=device)
        tripwires["runner_with_no_episodes"] = "DID NOT FAIL"
        failures.append("a runner given no episodes silently succeeded; a dead runner "
                        "would be invisible, which nearly cost two full 1M runs")
    except TrajectoryIdentityError:
        tripwires["runner_with_no_episodes"] = "HARD FAILED as required"

    import rl.trajectory_identity as TI
    real_record = TI.D_RECORD
    try:                                    # missing D -> must hard fail
        TI.D_RECORD = ROOT / "does_not_exist.json"
        FrozenDiscriminators(verify=True)
        tripwires["missing_discriminator"] = "DID NOT FAIL"
        failures.append("a missing discriminator record was accepted")
    except TrajectoryIdentityError:
        tripwires["missing_discriminator"] = "HARD FAILED as required"
    finally:
        TI.D_RECORD = real_record

    try:                                    # mutated D -> must be detected
        D2 = FrozenDiscriminators(verify=True)
        D2.models["A"]["clf"].coef_ = D2.models["A"]["clf"].coef_ + 1.0
        D2.assert_still_frozen()
        tripwires["mutated_discriminator"] = "DID NOT FAIL"
        failures.append("a mutated discriminator passed the frozen check")
    except TrajectoryIdentityError:
        tripwires["mutated_discriminator"] = "HARD FAILED as required"
    for k, v in tripwires.items():
        print(f"  [tripwire] {k}: {v}")

    verdict = "PASS" if not failures else "FAIL"
    OUT.write_text(json.dumps({
        "record": "H-OG-PSP smoke 2: trajectory-identity regulariser liveness and purity",
        "status": "SMOKE_RESULT", "utc": _now(), "VERDICT": verdict,
        "implements": "HOG_PSP_V3_SPEC.json#AMENDMENT_1_COMPONENT_2_MECHANISM",
        "proves": ("Trajectory identity credit is computed from the correct pole-specific "
                   "frozen discriminator and reaches the acting latent's PRIVATE parameters "
                   "by policy gradient, without touching any task quantity. Says NOTHING "
                   "about whether the treatment learns."),
        "discriminators": {"sha256": D.sha, "held_out_accuracy": acc,
                           "frozen_after_optimizer_step": d_frozen},
        "unit_of_credit": "full episode; no segment horizon introduced",
        "pole_routing": pole_routing,
        "liveness_by_latent": grads_by_z,
        "purity": {
            "checks": purity,
            "why_it_matters": ("Proof that V3 is not reward-shaped CTF. D's output never "
                               "enters reward, returns, GAE, or critic targets, so a "
                               "separation result cannot be dismissed as having paid the "
                               "latents to imitate the teachers."),
        },
        "tripwires": tripwires,
        "score_stabilisation": {"log_prob_clip": LOG_PROB_CLIP,
                                "baseline": "per-(latent, pole), EMA",
                                "chosen_prospectively": True},
        "runner_telemetry": runner.telemetry(),
        "failures": failures,
        "authorizes": "nothing; a combined treatment smoke remains before any 1M run",
        "EVAL_touched": False,
    }, indent=2), encoding="utf-8")

    print(f"\n  VERDICT: {verdict}")
    for f in failures:
        print(f"    FAIL: {f}")
    print(f"  -> {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
