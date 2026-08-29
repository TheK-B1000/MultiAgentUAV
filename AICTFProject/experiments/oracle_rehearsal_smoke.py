"""Live treatment smoke for oracle-gated rehearsal, on a FRESH K=2 model.

Fresh EXP2C-style architecture, step 0, no checkpoint weights. An existing EXP2C
checkpoint would carry whatever specialization, collapse or interference the earlier
treatment produced, so a success could not be attributed to this treatment.

Proves the treatment path exists end to end:

    exact FIT labels -> gated batch -> anchor loss -> optimizer -> parameter update

and that the parts which must NOT move did not.

Tiny by construction: a handful of rehearsal batches and one optimizer step per
batch. It says nothing about whether learning works.

Run:  python experiments/oracle_rehearsal_smoke.py --device cuda
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rl.launch_gate import LaunchGateError, check_fresh_training, check_opponent_mode  # noqa: E402
from rl.oracle_rehearsal import load_bank, rehearsal_anchor_loss                        # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand"
SPEC = SD / "sppo" / "ORACLE_GATED_REHEARSAL_SPEC.json"
OUT = SD / "sppo" / "ORACLE_REHEARSAL_SMOKE.json"

# EXP2C architecture, read from its checkpoint config. Weights are NOT loaded.
EXP2C_ARCH = {
    "use_latent_strategy": True, "latent_k": 2,
    "latent_assignment_mode": "static_env",
    "forced_latent_env_ids": tuple([0] * 16 + [1] * 16),
    "actor_cnn_feature_dim": 128, "actor_hidden_dim": 256,
    "latent_z_embed_dim": 16, "latent_actor_conditioning": "concat",
    "latent_strategy_hidden": 128, "latent_vf_hidden": 128,
}
# corrected from the archived run, which had opponent_randomize=True
OPPONENT_OVERRIDES = {"mode": "FIXED_OPPONENT", "opponent_randomize": False}

SMOKE_SEED, BATCHES, BATCH_SIZE, LR = 4242, 6, 48, 1e-4


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _digest(named) -> str:
    h = hashlib.sha256()
    for name, t in sorted(named):
        h.update(name.encode()); h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def build_fresh_k2(device: str):
    """EXP2C architecture, fresh weights, fixed seed. No checkpoint is opened."""
    import torch
    from rl.config.ppo_config import PPOConfig
    from rl.custom_ppo.policy import SharedActorCentralizedCritic
    from rl.custom_ppo.trainer_config import TrainerHyperparams, build_model_kwargs
    import experiments.r2_learned_crossover as R2

    cfg = PPOConfig()
    for k, v in {**EXP2C_ARCH, **OPPONENT_OVERRIDES}.items():
        setattr(cfg, k, v)
    cfg.load_path = None                       # fresh means step 0

    # TrainerHyperparams carries fields PPOConfig does not expose under the same
    # name. Fill from cfg where possible, fall back to the dataclass default, and
    # only invent a value where the field has no default at all.
    import dataclasses
    fields = TrainerHyperparams.__dataclass_fields__
    explicit = {"value_clip_range": 0.2, "reward_dense_weight": 1.0, "reward_scale": 1.0,
                "reward_clip": 10.0, "reward_stalemate_penalty": 0.0, "run_id": "smoke",
                "run_pid": 0, "opponent_randomize_training": False}
    kw = {}
    for name, f in fields.items():
        if hasattr(cfg, name):
            kw[name] = getattr(cfg, name)
        elif name in explicit:
            kw[name] = explicit[name]
        elif f.default is not dataclasses.MISSING:
            kw[name] = f.default
        elif f.default_factory is not dataclasses.MISSING:      # type: ignore[misc]
            kw[name] = f.default_factory()                       # type: ignore[misc]
        else:
            raise LaunchGateError(
                f"TrainerHyperparams field {name!r} has no cfg source and no default; "
                "the smoke will not invent one")
    hp = TrainerHyperparams(**kw)

    torch.manual_seed(SMOKE_SEED)
    env = R2.build_env(device, 10_700_001)
    try:
        model = SharedActorCentralizedCritic(
            env.observation_space, env.action_space, **build_model_kwargs(cfg, hp)).to(device)
    finally:
        env.close()
    return cfg, model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN -- SPEC_FROZEN_BEFORE_IMPLEMENTATION":
        raise LaunchGateError("REFUSING: rehearsal spec is not frozen")
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; the smoke is one-shot")

    print(f"ORACLE-GATED REHEARSAL SMOKE  {_now()}")
    failures: list[str] = []

    # ---- 1. config guards, via the REAL launch gate ------------------------
    cfg, model = build_fresh_k2(device)
    opp, fresh = check_opponent_mode(cfg), check_fresh_training(cfg)
    print(f"  [{opp.status}] {opp.name}: {opp.detail}")
    print(f"  [{fresh.status}] {fresh.name}: {fresh.detail}")
    for c in (opp, fresh):
        if not c.passed:
            failures.append(f"{c.name}: {c.detail}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  fresh K=2 model: {n_params:,} params, latent_k={model.latent_k}, "
          f"uses_latent={model.uses_latent_strategy}, weights NOT loaded")
    if model.latent_k != 2 or not model.uses_latent_strategy:
        failures.append("model is not a latent K=2 policy")

    # ---- 2. rehearsal bank --------------------------------------------------
    bank = load_bank(rng_seed=SMOKE_SEED)
    comp = bank.composition()
    print(f"  bank: {comp['eligible']} eligible "
          f"({comp['A_preferred']} A-pref, {comp['B_preferred']} B-pref), "
          f"{comp['tied_excluded_from_sampling']} tied excluded")

    # ---- 3. snapshot before ------------------------------------------------
    before = {n: p.detach().cpu().numpy().copy() for n, p in model.named_parameters()}
    # NOTE on teacher freezing: no teacher NETWORK is loaded here. The rehearsal
    # anchors against pre-recorded teacher ACTIONS stored in the collection
    # (branch_pi_A_action / branch_pi_B_action), which were produced by the
    # SHA-verified SAPPO checkpoints during collection. Teachers therefore cannot
    # drift during this smoke because they are not present -- a stronger property
    # than freezing them, and the reason there is no teacher digest to compare.

    # ---- 4. gated rehearsal + optimizer steps -------------------------------
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    z_pressure = {0: 0, 1: 0}
    losses = []
    for _ in range(BATCHES):
        batch = bank.sample(BATCH_SIZE)
        for z in batch["z_idx"]:
            z_pressure[int(z)] += 1
        loss = rehearsal_anchor_loss(model, batch, device=device)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach()))
    print(f"  ran {BATCHES} gated batches, loss {losses[0]:.4f} -> {losses[-1]:.4f}")

    # ---- 5. assertions ------------------------------------------------------
    after = {n: p.detach().cpu().numpy() for n, p in model.named_parameters()}
    changed = sorted(n for n in after if not np.array_equal(before[n], after[n]))
    unchanged = sorted(n for n in after if n not in changed)
    print(f"  parameters moved: {len(changed)} of {len(after)}")

    # A parameter must be unchanged EXACTLY WHEN the loss never reached it. Anything
    # else means either a dead gradient path or an unintended one.
    grad_none = sorted(n for n, p in model.named_parameters() if p.grad is None)
    if grad_none != unchanged:
        only_stuck = sorted(set(unchanged) - set(grad_none))
        only_ungrad = sorted(set(grad_none) - set(unchanged))
        failures.append(
            "unchanged parameters do not match the no-gradient set; "
            f"received gradient but did not move: {only_stuck[:4]}; "
            f"moved without gradient: {only_ungrad[:4]}")

    if not changed:
        failures.append("no parameter changed; the optimizer path is dead")
    if z_pressure[0] == 0:
        failures.append("z0 received no rehearsal pressure")
    if z_pressure[1] == 0:
        failures.append("z1 received no rehearsal pressure")
    try:
        bank.assert_zero_tied_pressure()
    except Exception as exc:                       # noqa: BLE001
        failures.append(str(exc))

    tel = bank.telemetry()
    print(f"  z0 pressure {z_pressure[0]}   z1 pressure {z_pressure[1]}   "
          f"tied exposures {tel['tied_exposures']}   replay factor {tel['replay_factor']}")

    verdict = "PASS" if not failures else "FAIL"
    rec = {
        "record": "Oracle-gated rehearsal live smoke",
        "status": "SMOKE_RESULT", "utc": _now(),
        "implements": "ORACLE_GATED_REHEARSAL_SPEC.json",
        "VERDICT": verdict,
        "meaning": ("Proves the treatment path exists: exact FIT labels -> gated batch -> "
                    "anchor loss -> optimizer -> parameter update. Says NOTHING about "
                    "whether learning works."),
        "model": {"fresh_initialisation": True, "checkpoint_weights_loaded": False,
                  "seed": SMOKE_SEED, "params": n_params, "latent_k": int(model.latent_k),
                  "architecture_source": "EXP2C checkpoint cfg (architecture only, weights discarded)"},
        "config_guards": {"opponent_mode": opp.detail, "fresh_training": fresh.detail,
                          "opponent_randomize": bool(getattr(cfg, "opponent_randomize")),
                          "note": "the archived EXP2C run had opponent_randomize=True; corrected here"},
        "rehearsal": {"batches": BATCHES, "batch_size": BATCH_SIZE,
                      "z0_pressure": z_pressure[0], "z1_pressure": z_pressure[1],
                      "tied_exposures": tel["tied_exposures"],
                      "replay_factor": tel["replay_factor"],
                      "bank": comp},
        "parameters": {"changed": len(changed), "unchanged": len(unchanged),
                       "changed_groups": sorted({n.split(".")[0] for n in changed}),
                       "unchanged_groups": sorted({n.split(".")[0] for n in unchanged}),
                       "unchanged_equals_no_gradient_set": grad_none == unchanged,
                       "why_those_are_unchanged": (
                           "the anchor loss touches only the actor path; critic, "
                           "phase_predictor and strategy_encoder receive no gradient "
                           "from it, and strategy_encoder is the router which "
                           "static_env assignment bypasses entirely")},
        "teacher_freezing": {
            "teacher_network_loaded": False,
            "why": ("the rehearsal anchors against pre-recorded teacher ACTIONS stored in "
                    "the SHA-verified collection, so no teacher network exists in this "
                    "process and teachers cannot drift -- stronger than freezing them"),
            "teacher_action_provenance": "branch_pi_A_action / branch_pi_B_action, produced by the SHA-pinned SAPPO checkpoints during collection"},
        "loss_first_last": [losses[0], losses[-1]],
        "failures": failures,
        "COVERAGE_LIMITS": {
            "not_covered_here": "z -> pole persistence across environment resets",
            "why": ("That property lives in the rollout loop, not the rehearsal path. The "
                    "runtime auditors are already wired at the per-episode close in "
                    "rl/custom_ppo/latent/credit/episode/manager.py and activate during "
                    "actual training, where they hard-fail on drift."),
            "open_question_for_the_PI": (
                "Under FIXED_OPPONENT every env faces ONE scripted opponent, so 'z -> pole' "
                "may not even be well-defined in the rollout the way it was for EXP2B, "
                "which used a two-tag pool. The pole distinction currently lives in the "
                "REHEARSAL data (branch_pole), not in the rollout. Whether the production "
                "run should assign z0 and z1 to different opponents is a scientific choice "
                "the frozen spec does not pin down."),
        },
        "EVAL_touched": False,
        "authorizes": "nothing; PPO launch remains a separate PI decision",
    }
    OUT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}")
    for f in failures:
        print(f"    FAIL: {f}")
    print(f"  -> {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
