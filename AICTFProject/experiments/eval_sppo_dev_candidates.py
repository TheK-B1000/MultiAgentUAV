"""Evaluate the five SPPPO development candidates on the DEVELOPMENT block.

    train      10100001..10100032   (98,304 steps per candidate)
    evaluate   10200001..10200032   <- THIS FILE
    select     smallest qualifying lambda_R, terminal values only

Selection must be OUT OF SAMPLE. An earlier driver trained the candidates on the
development block and read Delta_A/Delta_B from training-time telemetry, which
would have chosen lambda_R from the same trajectories the candidates were
optimised on. Training telemetry remains diagnostic; the selector consumes only
what this file produces.

For each terminal checkpoint, on each development seed, under the frozen
16 x z0|A / 16 x z1|B cell layout:

    Delta_A(o) = V_hat(o, z0, A) - V_hat(o, z1, A)      on pole-A envs
    Delta_B(o) = V_hat(o, z1, B) - V_hat(o, z0, B)      on pole-B envs

V_hat is the analytic expectation under the MASKED policy distribution, scored
by the frozen Q_psi whose SHA is verified before use. Q_psi is never updated
here -- this is measurement only.

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
OUT = SD / "sppo" / "lambda_sweep"
RESULT = SD / "sppo" / "SPPPO_DEV_EVALUATION.json"

DEV_SEEDS = list(range(10_200_001, 10_200_033))       # 32 development seeds
MAX_STEPS = 240


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def evaluate_candidate(ckpt_path, qpsi, seeds, device):
    """Roll the terminal policy on development seeds and score contrasts."""
    from rl.custom_ppo import load_custom_ppo_policy
    from rl.scorer.ranking import POLE_A, POLE_B, strategic_contrast
    from experiments.run_exp2b_specialization_preserving_compression import (
        CELL_KEYS, CELL_Z, pole_A_genome,
    )
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays,
    )
    from rl.curriculum import phase_from_tag
    import experiments.r2_learned_crossover as R2

    per_seed = {"A": [], "B": [], "ret": []}
    for seed in seeds:
        env = R2.build_env(device, seed)
        core = env.core
        try:
            model = load_custom_ppo_policy(str(ckpt_path), env.observation_space,
                                           env.action_space, device=device)
            inner = getattr(model, "model", model)
            core._bt_profile_override = None
            core._sds_opening_hold_steps = 0
            genomes = {"OP6": pole_A_genome()}
            install_keyed_opponent_overlays(core, genomes)
            for i, key in enumerate(CELL_KEYS):
                env.env_method("set_phase", phase_from_tag(key), indices=[i])
                env.env_method("set_next_opponent", "SCRIPTED", key, indices=[i])
            obs = env.reset()
            rows = assert_live_opponent_batch(
                core, genomes, allowed_keys=("OP6", "OP7"),
                context=f"SPPPO dev eval seed {seed}")
            # TRUE pole per env from the live opponent, not from z
            pole = torch.tensor(
                [POLE_A if r["live_opponent_key"] == "OP6" else POLE_B for r in rows],
                dtype=torch.long, device=device)
            z = torch.tensor(CELL_Z, dtype=torch.long, device=device)
            expected = torch.where(z == 0, torch.full_like(z, POLE_A),
                                   torch.full_like(z, POLE_B))
            if not bool((pole == expected).all()):
                raise RuntimeError(f"seed {seed}: live z/pole assignment is broken")

            dA, dB, rew = [], [], np.zeros(int(core.B))
            for t in range(MAX_STEPS):
                decision = (core.blue_commit_ticks_left[:, 0] <= 0)
                if bool(decision.any()):
                    o = {"grid": torch.as_tensor(obs["grid"], dtype=torch.float32, device=device),
                         "vec": torch.as_tensor(obs["vec"], dtype=torch.float32, device=device),
                         "agent_mask": torch.as_tensor(obs["agent_mask"], dtype=torch.float32, device=device),
                         "mask": torch.as_tensor(obs["mask"], dtype=torch.float32, device=device)}
                    with torch.no_grad():
                        delta, _, _ = strategic_contrast(inner, qpsi, o, pole)
                    d = delta.cpu().numpy()
                    m = decision.cpu().numpy()
                    p = pole.cpu().numpy()
                    dA.extend(d[(p == POLE_A) & m].tolist())
                    dB.extend(d[(p == POLE_B) & m].tolist())
                act, _ = model.predict(obs, deterministic=True)
                obs, r, done, _info = env.step(act)
                rew = rew + np.asarray(r, dtype=np.float64)
                if bool(np.asarray(done).all()):
                    break
            if dA:
                per_seed["A"].append(float(np.mean(dA)))
            if dB:
                per_seed["B"].append(float(np.mean(dB)))
            per_seed["ret"].append(float(rew.mean()))
        finally:
            env.close()
    return {
        "delta_A": float(np.mean(per_seed["A"])) if per_seed["A"] else float("nan"),
        "delta_B": float(np.mean(per_seed["B"])) if per_seed["B"] else float("nan"),
        "ep_rew_mean": float(np.mean(per_seed["ret"])) if per_seed["ret"] else float("nan"),
        "n_seeds": len(per_seed["ret"]),
        "per_seed_delta_A": [round(v, 6) for v in per_seed["A"]],
        "per_seed_delta_B": [round(v, 6) for v in per_seed["B"]],
    }


def main() -> int:
    from experiments.run_sppo_lambda_sweep import LAMBDA_GRID, _tag
    from rl.scorer.attach import SPPPO_QPSI_PATH, SPPPO_QPSI_SHA256
    from rl.scorer.ranking import load_frozen_qpsi

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if RESULT.is_file():
        raise SystemExit(f"REFUSING: {RESULT} exists; the development evaluation is one-shot")

    device = a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"
    qpsi = load_frozen_qpsi(ROOT / SPPPO_QPSI_PATH,
                            expected_sha256=SPPPO_QPSI_SHA256, device=device)
    print(f"SPPPO DEVELOPMENT EVALUATION  {_now()}")
    print(f"  seeds  {DEV_SEEDS[0]}..{DEV_SEEDS[-1]}  ({len(DEV_SEEDS)})")
    print(f"  qpsi   {SPPPO_QPSI_SHA256[:16]}...  frozen, measurement only\n")

    results = {}
    for lam in LAMBDA_GRID:
        ck_dir = OUT / _tag(lam) / "ckpts"
        ckpts = sorted(ck_dir.glob("final_*")) or sorted(ck_dir.glob("*.zip"))
        if not ckpts:
            raise SystemExit(f"REFUSING: no terminal checkpoint for lambda={lam} in {ck_dir}")
        print(f"  lambda_R = {lam}  <- {ckpts[-1].name}", flush=True)
        r = evaluate_candidate(ckpts[-1], qpsi, DEV_SEEDS, device)
        r["checkpoint"] = ckpts[-1].name
        results[str(lam)] = r
        print(f"     delta_A {r['delta_A']:+.6f}   delta_B {r['delta_B']:+.6f}   "
              f"return {r['ep_rew_mean']:+.4f}")

    rec = {
        "record": "SPPPO V1 development evaluation of lambda_R candidates",
        "status": "FROZEN_RESULT",
        "utc": _now(),
        "train_block": "10100001..10100032",
        "evaluation_block": f"{DEV_SEEDS[0]}..{DEV_SEEDS[-1]}",
        "out_of_sample": "candidates were TRAINED on the training block; this measures them on the development block",
        "qpsi_sha256": SPPPO_QPSI_SHA256,
        "qpsi_mutated": False,
        "results": results,
        "note": "training-time telemetry is diagnostic only and is NOT the selection quantity",
    }
    RESULT.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(f"\n  -> {RESULT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
