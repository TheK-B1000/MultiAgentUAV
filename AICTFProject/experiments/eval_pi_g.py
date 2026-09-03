"""pi_G single-generalist (no-latent) baseline EVAL.

Implements PI_G_EVAL_SPEC.json. Answers the cleanest reviewer objection to a latent-conditioned
architecture: "do we need latent specialization at all?" pi_G is a single policy trained on the
mixed OP6+OP7 opponent distribution with no strategic-mode mechanism (latent_k=0,
uses_latent_strategy=False, verified by loading the checkpoint, not by trusting its
run_config.json). Strictly inference-only: no fine-tuning, no additional steps.

Reuses the SAME sealed 64-seed block RSCFT's EVAL used (11704001..11704064), deliberately, for
direct pairing against whichever latent-system result is designated the paper's final
comparison target -- this does not reopen or contaminate RSCFT's own one-shot result; it scores
a different, never-before-evaluated checkpoint on an already-opened seed range.

GATED: refuses to run until the RSCFT sealed EVAL has produced a terminal artifact (either
RSCFT_EVAL_RESULT.json or RSCFT_EVAL_INTEGRITY_REQUIRED.json), per PI_G_EVAL_SPEC.json's
explicit DO_NOT_RUN_UNTIL clause -- running concurrently would contend for the same GPU as the
still-open RSCFT result.

Run:  python experiments/eval_pi_g.py --device cuda
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.eval_hog_psp_v3 import _mean_ci

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "PI_G_EVAL_SPEC.json"
RSCFT_RESULT = SD / "RSCFT_EVAL_RESULT.json"
RSCFT_FLAG = SD / "RSCFT_EVAL_INTEGRITY_REQUIRED.json"
OUT = SD / "PI_G_EVAL_RESULT.json"
ROWS_CSV = SD / "pi_g_eval_rows.csv"
PREAUDIT_FLAG = SD / "PI_G_EVAL_INTEGRITY_REQUIRED.json"

EVAL_SEEDS = list(range(11_704_001, 11_704_065))
N_BOOT, ALPHA, BOOTSTRAP_SEED = 20_000, 0.05, 7
POLES = ("A", "B")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _preflight() -> dict:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_EVAL_IS_OPENED":
        raise SystemExit(f"REFUSING: pi_G eval spec not frozen: {spec['status']!r}")
    if not (RSCFT_RESULT.is_file() or RSCFT_FLAG.is_file()):
        raise SystemExit(
            "REFUSING: neither RSCFT_EVAL_RESULT.json nor "
            "RSCFT_EVAL_INTEGRITY_REQUIRED.json exists yet. PI_G_EVAL_SPEC.json's "
            "DO_NOT_RUN_UNTIL clause requires the RSCFT sealed EVAL to have fully "
            "completed (either a clean verdict or a flagged tie/reversal) before this "
            "runs -- it must not contend for the GPU with a still-open sealed result.")
    ck = ROOT / spec["MODEL_UNDER_TEST"]["checkpoint"]
    if not ck.is_file():
        raise SystemExit(f"REFUSING: pi_G checkpoint missing: {ck}")
    if _sha(ck) != spec["MODEL_UNDER_TEST"]["sha256"]:
        raise SystemExit("REFUSING: pi_G checkpoint sha mismatch")
    if OUT.is_file() or ROWS_CSV.is_file() or PREAUDIT_FLAG.is_file():
        raise SystemExit("REFUSING: a pi_G EVAL output already exists; EVAL is one-shot")
    block = spec["PROTOCOL"]["block"]
    lo, hi = (int(x) for x in block.split(".."))
    if [EVAL_SEEDS[0], EVAL_SEEDS[-1]] != [lo, hi] or len(EVAL_SEEDS) != 64:
        raise SystemExit(f"REFUSING: evaluator seeds do not match the frozen block {block}")
    return spec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
    )
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    spec = _preflight()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    rscft_state = "RESULT" if RSCFT_RESULT.is_file() else "INTEGRITY_REQUIRED (tie/reversal)"

    print(f"PI_G EVAL  {_now()}")
    print(f"  RSCFT sealed EVAL state at launch: {rscft_state}")
    print("  checkpoint sha256 VERIFIED against PI_G_EVAL_SPEC.json")
    print(f"  seeds {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]} (n={len(EVAL_SEEDS)}), reused from RSCFT's block")
    print(f"  no z-conditioning: pi_G uses_latent_strategy=False")
    print(f"  bootstrap n={N_BOOT}, alpha={ALPHA}, rng_seed={BOOTSTRAP_SEED}\n", flush=True)

    ck = ROOT / spec["MODEL_UNDER_TEST"]["checkpoint"]
    probe = R2.build_env(device, EVAL_SEEDS[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()

    policy = load_custom_ppo_policy(str(ck), obs_space, act_space, device=device)
    model = policy.model if hasattr(policy, "model") else policy
    if getattr(model, "latent_k", None) not in (0, None) or getattr(model, "uses_latent_strategy", False):
        raise SystemExit("REFUSING: loaded checkpoint IS latent-conditioned; this is not pi_G "
                         "as specified")

    def run_cell(pole: str, seed: int) -> dict:
        env = R2.build_env(device, seed)
        core = env.core
        try:
            policy.reset_strategy()          # no fixed_latent_strategy call -- pi_G has no z
            core._bt_profile_override = None
            core._sds_opening_hold_steps = 0
            genomes = {"OP6": pole_A_genome()} if pole == "A" else {}
            install_keyed_opponent_overlays(core, genomes)
            key = P0.POLES[pole]
            env.env_method("set_phase", phase_from_tag(key))
            env.env_method("set_next_opponent", "SCRIPTED", key)
            obs = env.reset()
            obs["global_state"] = env.state()
            assert_live_opponent_batch(core, genomes, allowed_keys=(key,),
                                       context=f"pi_G eval {pole} seed {seed}")
            terminal = None
            for _ in range(R2.MAX_STEPS):
                action, _ = policy.predict(obs, deterministic=True)
                env.step_async(action)
                obs, _r, done, info = env.step_wait()
                obs["global_state"] = env.state()
                if bool(np.asarray(done).any()):
                    i0 = info[0] if isinstance(info, (list, tuple)) else info
                    res = (i0 or {}).get("episode_result") or {}
                    terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                    break
            if terminal is None:
                terminal = (int(core.blue_score[0]), int(core.red_score[0]))
            blue, red = terminal
            return {"blue": blue, "red": red, "win": int(blue > red), "margin": blue - red}
        finally:
            env.close()

    rows = []
    for pole in POLES:
        for seed in EVAL_SEEDS:
            rows.append({"policy": "pi_G", "pole": pole, "seed": seed, **run_cell(pole, seed)})
        wr = np.mean([r["win"] for r in rows if r["pole"] == pole])
        print(f"  pi_G on Pole {pole}: win rate {wr:.4f}", flush=True)

    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def wins(pole):
        by = {r["seed"]: r["win"] for r in rows if r["pole"] == pole}
        return np.array([by[s] for s in EVAL_SEEDS], dtype=np.float64)

    v_a, v_b = _mean_ci(wins("A")), _mean_ci(wins("B"))

    print(f"\n  V(pi_G, A) = {v_a['mean']:.4f} [{v_a['lcb95']:.4f}, {v_a['ucb95']:.4f}]")
    print(f"  V(pi_G, B) = {v_b['mean']:.4f} [{v_b['lcb95']:.4f}, {v_b['ucb95']:.4f}]")

    OUT.write_text(json.dumps({
        "record": "pi_G single-generalist baseline EVAL", "status": "FROZEN_RESULT",
        "one_shot": True, "utc": _now(), "implements": "PI_G_EVAL_SPEC.json",
        "rscft_state_at_launch": rscft_state,
        "checkpoint": {"path": spec["MODEL_UNDER_TEST"]["checkpoint"],
                       "sha256": spec["MODEL_UNDER_TEST"]["sha256"]},
        "seeds": {"block": [EVAL_SEEDS[0], EVAL_SEEDS[-1]], "n": len(EVAL_SEEDS),
                  "reused_from": "RSCFT sealed EVAL block"},
        "V_pi_G_A": v_a, "V_pi_G_B": v_b,
        "bootstrap": {"samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOTSTRAP_SEED},
        "comparison_against_latent_system": "NOT computed here -- PI_G_EVAL_SPEC.json's "
            "COMPARISON_RULE requires a designated champion result file, which does not exist "
            "at pi_G-eval time in general. Run experiments/compare_pi_g_vs_latent.py against "
            "the chosen champion once it is designated.",
        "total_episodes": len(rows),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
