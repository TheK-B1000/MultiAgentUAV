"""Evaluate the five SPPPO development candidates on the DEVELOPMENT block.

    train      10100001..10100032   (98,304 steps per candidate)
    evaluate   10200001..10200032   <- THIS FILE
    select     smallest qualifying lambda_R, terminal values only

Selection must be OUT OF SAMPLE. An earlier driver trained the candidates on the
development block and read Delta_A/Delta_B from training-time telemetry, which
would have chosen lambda_R from the same trajectories the candidates were
optimised on. Training telemetry remains diagnostic; the selector consumes only
what this file produces.

The environment is built with build_training_env -- the SAME 32-env construction
training uses -- not the single-env Phase 0 scoring builder. An earlier version
of this file used the latter and would have produced a length-1 pole vector
against a 32-wide batch. The per-env z/pole assertion from
configure_exp2b_live_environment runs here too, so a broken cell layout aborts
the evaluation instead of silently mis-scoring it.

Actions are taken under the env's ASSIGNED z (forced per cell, matching training)
via the masked logits path. Delta is computed by evaluating BOTH z at each
decision state and orienting by the TRUE pole from the live opponent key:

    Delta_A(o) = V_hat(o, z0, A) - V_hat(o, z1, A)      on pole-A envs
    Delta_B(o) = V_hat(o, z1, B) - V_hat(o, z0, B)      on pole-B envs

Q_psi is frozen, SHA-verified, and never updated -- this is measurement only.

Run:  python experiments/eval_sppo_dev_candidates.py --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand"
RESULT = SD / "sppo" / "SPPPO_DEV_EVALUATION.json"

DEV_SEED = 10_200_001
DEV_RANGE = (10_200_001, 10_200_032)
MAX_STEPS = 240


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _masked_argmax(inner, obs_t, z_idx):
    """Deterministic action under the ASSIGNED z, using PPO's own masking."""
    flat = inner._mask_logits(inner.policy_logits(obs_t, z_idx=z_idx), obs_t["mask"])
    heads = torch.split(flat, list(inner.action_dims), dim=-1)
    return torch.stack([h.argmax(dim=-1) for h in heads], dim=-1)


def evaluate_candidate(lam, qpsi, device, n_episodes):
    from experiments.run_sppo_lambda_sweep import build_candidate, _tag, OUT
    from experiments.run_exp2b_specialization_preserving_compression import (
        CELL_Z, configure_exp2b_live_environment,
    )
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.scorer.ranking import POLE_A, POLE_B, strategic_contrast
    from rl.training.env_factory import build_training_env

    cfg, _, parent_contract = build_candidate(lam)
    cfg.seed = DEV_SEED                       # DEVELOPMENT block
    ck_dir = OUT / _tag(lam) / "ckpts"
    ckpts = sorted(ck_dir.glob("final_*")) or sorted(ck_dir.glob("*.zip"))
    if not ckpts:
        raise SystemExit(f"REFUSING: no terminal checkpoint for lambda={lam} in {ck_dir}")

    env = build_training_env(cfg, initial_phase="OP6", initial_opponent_tag="OP6")
    try:
        manifest = configure_exp2b_live_environment(
            env, cfg, contract=parent_contract, allow_development_seed=True,
            training_seed_range=(10_100_001, 10_100_032),
            development_seed_range=DEV_RANGE,
            manifest_key="sppo_dev_eval",
            context_label=f"SPPPO dev evaluation lambda={lam}")
        rows = manifest["sppo_dev_eval"]["resolved_opponent_rows"]
        pole = torch.tensor(
            [POLE_A if r["live_opponent_key"] == "OP6" else POLE_B for r in rows],
            dtype=torch.long, device=device)
        z = torch.tensor(CELL_Z, dtype=torch.long, device=device)
        if pole.shape[0] != int(env.core.B):
            raise RuntimeError(f"pole vector {pole.shape[0]} != {int(env.core.B)} envs")
        expected = torch.where(z == 0, torch.full_like(z, POLE_A), torch.full_like(z, POLE_B))
        if not bool((pole == expected).all()):
            raise RuntimeError("live z/pole assignment is broken at evaluation")

        model = load_custom_ppo_policy(str(ckpts[-1]), env.observation_space,
                                       env.action_space, device=device)
        inner = getattr(model, "model", model)
        dA, dB, ep_returns = [], [], []
        obs = env.reset()
        for _ep in range(n_episodes):
            rew = np.zeros(int(env.core.B), dtype=np.float64)
            for _t in range(MAX_STEPS):
                o = {k: torch.as_tensor(np.asarray(obs[k]), dtype=torch.float32,
                                        device=device)
                     for k in ("grid", "vec", "agent_mask", "mask")}
                decision = (env.core.blue_commit_ticks_left[:, 0] <= 0)
                if bool(decision.any()):
                    with torch.no_grad():
                        delta, _, _ = strategic_contrast(inner, qpsi, o, pole)
                    d = delta.cpu().numpy(); m = decision.cpu().numpy()
                    p = pole.cpu().numpy()
                    dA.extend(d[(p == POLE_A) & m].tolist())
                    dB.extend(d[(p == POLE_B) & m].tolist())
                with torch.no_grad():
                    act = _masked_argmax(inner, o, z).cpu().numpy()
                obs, r, done, _i = env.step(act)
                rew += np.asarray(r, dtype=np.float64)
                if bool(np.asarray(done).any()):
                    break
            ep_returns.append(float(rew.mean()))
        return {
            "delta_A": float(np.mean(dA)) if dA else float("nan"),
            "delta_B": float(np.mean(dB)) if dB else float("nan"),
            "ep_rew_mean": float(np.mean(ep_returns)),
            "n_decision_states_A": len(dA), "n_decision_states_B": len(dB),
            "n_episodes": n_episodes, "n_envs": int(env.core.B),
            "checkpoint": ckpts[-1].name,
        }
    finally:
        env.close()


def main() -> int:
    from experiments.run_sppo_lambda_sweep import LAMBDA_GRID
    from rl.scorer.attach import SPPPO_QPSI_PATH, SPPPO_QPSI_SHA256
    from rl.scorer.ranking import load_frozen_qpsi

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--episodes", type=int, default=4)
    a = ap.parse_args()
    if RESULT.is_file():
        raise SystemExit(f"REFUSING: {RESULT} exists; the development evaluation is one-shot")

    device = a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"
    qpsi = load_frozen_qpsi(ROOT / SPPPO_QPSI_PATH,
                            expected_sha256=SPPPO_QPSI_SHA256, device=device)
    sha_before = {k: v.detach().cpu().clone() for k, v in qpsi.state_dict().items()}

    print(f"SPPPO DEVELOPMENT EVALUATION  {_now()}")
    print(f"  block  {DEV_RANGE[0]}..{DEV_RANGE[1]}   32 envs x {a.episodes} episodes")
    print(f"  qpsi   {SPPPO_QPSI_SHA256[:16]}...  frozen, measurement only\n")

    results = {}
    for lam in LAMBDA_GRID:
        print(f"  lambda_R = {lam} ...", flush=True)
        r = evaluate_candidate(lam, qpsi, device, a.episodes)
        results[str(lam)] = r
        print(f"     delta_A {r['delta_A']:+.6f}   delta_B {r['delta_B']:+.6f}   "
              f"return {r['ep_rew_mean']:+.4f}   states {r['n_decision_states_A']}/"
              f"{r['n_decision_states_B']}", flush=True)

    for k, v in qpsi.state_dict().items():
        if not torch.equal(v.detach().cpu(), sha_before[k]):
            raise RuntimeError("Q_psi mutated during evaluation")

    rec = {
        "record": "SPPPO V1 development evaluation of lambda_R candidates",
        "status": "FROZEN_RESULT",
        "utc": _now(),
        "train_block": "10100001..10100032",
        "evaluation_block": f"{DEV_RANGE[0]}..{DEV_RANGE[1]}",
        "out_of_sample": "candidates were TRAINED on the training block; measured here on the development block",
        "env_construction": "build_training_env, 32 envs, same as training; per-env z/pole asserted",
        "actions": "deterministic masked argmax under the ASSIGNED z per cell",
        "qpsi_sha256": SPPPO_QPSI_SHA256,
        "qpsi_mutated": False,
        "episodes_per_candidate": a.episodes,
        "results": results,
        "note": "training-time telemetry is diagnostic only and is NOT the selection quantity",
    }
    RESULT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n  Q_psi unchanged after all candidates\n  -> {RESULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
