"""One-shot held-out payoff EVAL for the frozen H-OG-PSP V3 terminal.

The trajectory-identity result was frozen and committed before this evaluator and
cannot substitute for its strategic payoff gate. The evaluator carries forward the
OG-PSP six-cell design and paired-bootstrap criterion unchanged:

    delta_A = wins(z0,A) - wins(z1,A)      PASS iff lcb95 > 0
    delta_B = wins(z1,B) - wins(z0,B)      PASS iff lcb95 > 0

BOTH sides must pass. Terminal checkpoint only. One deterministic episode per
(cell, seed), 32 held-out seeds, 192 episodes total.

Run:  python experiments/eval_hog_psp_v3.py --device cuda
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

SD = ROOT / "artifacts" / "strategic_demand"
SPEC = SD / "sppo" / "HOG_PSP_V3_EVAL_SPEC.json"
FROZEN = SD / "sppo" / "HOG_PSP_V3_MODEL_FROZEN.json"
OUT = SD / "sppo" / "HOG_PSP_V3_EVAL_RESULT.json"
ROWS_CSV = SD / "sppo" / "hog_psp_v3_eval_rows.csv"

EVAL_SEEDS = list(range(11_300_101, 11_300_133))
N_BOOT, ALPHA, BOOTSTRAP_SEED = 20_000, 0.05, 7
POLES = ("A", "B")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mean_ci(values: np.ndarray) -> dict:
    """Paired percentile bootstrap, seed as the resampling unit."""
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"mean": float(values.mean()), "lcb95": float(lo), "ucb95": float(hi)}


def _ratio_ci(numer: np.ndarray, denom: np.ndarray) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(numer), size=(N_BOOT, len(numer)))
    d = denom[idx].mean(axis=1)
    boot = np.divide(numer[idx].mean(axis=1), d, out=np.full(N_BOOT, np.nan), where=d > 0)
    boot = boot[~np.isnan(boot)]
    lo, hi = np.percentile(boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"mean": float(np.nanmean(boot)), "lcb95": float(lo), "ucb95": float(hi)}


def _preflight() -> tuple[dict, dict, Path]:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "FROZEN_BEFORE_EVAL_IS_OPENED":
        raise SystemExit(f"REFUSING: eval spec is not frozen: {spec['status']!r}")
    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    if frozen["TERMINAL_RECORD_VALIDITY"]["verdict"].split()[0] != "VALID":
        raise SystemExit("REFUSING: the training run was not established VALID")
    if frozen["EVAL_STATE_AT_FREEZE"]["touched"]:
        raise SystemExit("REFUSING: the V3 EVAL block was not untouched at model freeze")
    ck = ROOT / frozen["TERMINAL_CHECKPOINT"]["path"]
    if not ck.is_file():
        raise SystemExit(f"REFUSING: terminal checkpoint missing: {ck}")
    actual = hashlib.sha256(ck.read_bytes()).hexdigest()
    expected = frozen["TERMINAL_CHECKPOINT"]["sha256"]
    if actual != expected:
        raise SystemExit(
            f"REFUSING: checkpoint identity mismatch.\n  expected {expected}\n  actual   {actual}")
    if actual != spec["MODEL_UNDER_TEST"]["sha256"]:
        raise SystemExit("REFUSING: freeze record and eval spec name different checkpoints")
    if OUT.is_file() or ROWS_CSV.is_file():
        raise SystemExit("REFUSING: an H-OG-PSP V3 EVAL output already exists; EVAL is one-shot")
    if spec["SEEDS"]["block"] != "11300101..11300132":
        raise SystemExit("REFUSING: eval seed block drifted from the frozen spec")
    if [EVAL_SEEDS[0], EVAL_SEEDS[-1]] != [11_300_101, 11_300_132]:
        raise SystemExit("REFUSING: scorer seed range does not match the frozen block")
    gate = spec["PRIMARY_GATE_CROSSOVER"]
    if (gate["n_boot"], gate["alpha"], gate["rng_seed"]) != (
            N_BOOT, ALPHA, BOOTSTRAP_SEED):
        raise SystemExit("REFUSING: bootstrap settings drifted from the frozen spec")
    return spec, frozen, ck


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

    spec, frozen, ck_path = _preflight()
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    print(f"H-OG-PSP V3 HELD-OUT PAYOFF EVAL  {_now()}")
    print("  terminal checkpoint sha256 VERIFIED against freeze record and eval spec")
    print(f"  seeds {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]}  (n={len(EVAL_SEEDS)})")
    print(f"  gate: BOTH delta lcb95 > 0, paired bootstrap n={N_BOOT}, alpha={ALPHA}")
    print("  trajectory identity is already frozen; this run measures payoff only\n", flush=True)

    probe = R2.build_env(device, EVAL_SEEDS[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()

    student = load_custom_ppo_policy(str(ck_path), obs_space, act_space, device=device)
    if student.model.latent_k != 2 or not student.model.uses_latent_strategy:
        raise SystemExit("REFUSING: checkpoint is not a latent K=2 policy")
    teachers = {k: load_custom_ppo_policy(str(v), obs_space, act_space, device=device)
                for k, v in P0.TEACHERS.items()}

    def run_cell(policy, pole: str, z: int | None, seed: int) -> dict:
        env = R2.build_env(device, seed)
        core = env.core
        try:
            if z is not None:
                policy.fixed_latent_strategy = True
                policy.fixed_latent_strategy_id = int(z)
            policy.reset_strategy()

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
                                       context=f"hog v3 eval {pole} z{z} seed {seed}")
            terminal = None
            for _ in range(R2.MAX_STEPS):
                action, _ = policy.predict(obs, deterministic=True)
                env.step_async(action)
                obs, _reward, done, info = env.step_wait()
                obs["global_state"] = env.state()
                if bool(np.asarray(done).any()):
                    i0 = info[0] if isinstance(info, (list, tuple)) else info
                    result = (i0 or {}).get("episode_result") or {}
                    terminal = (int(result.get("blue_score", 0)),
                                int(result.get("red_score", 0)))
                    break
            if terminal is None:
                terminal = (int(core.blue_score[0]), int(core.red_score[0]))
            blue, red = terminal
            return {"blue": blue, "red": red, "win": int(blue > red), "margin": blue - red}
        finally:
            env.close()

    cells = [("z0", 0, student), ("z1", 1, student),
             ("pi_A", None, teachers["pi_A"]), ("pi_B", None, teachers["pi_B"])]
    rows = []
    for name, z, policy in cells:
        for pole in POLES:
            if name == "pi_A" and pole != "A":
                continue
            if name == "pi_B" and pole != "B":
                continue
            for seed in EVAL_SEEDS:
                result = run_cell(policy, pole, z, seed)
                rows.append({"policy": name, "pole": pole, "seed": seed, **result})
            wr = np.mean([r["win"] for r in rows
                          if r["policy"] == name and r["pole"] == pole])
            print(f"  {name:5s} on Pole {pole}: win rate {wr:.4f}", flush=True)

    def wins(name: str, pole: str) -> np.ndarray:
        by_seed = {r["seed"]: r["win"] for r in rows
                   if r["policy"] == name and r["pole"] == pole}
        return np.array([by_seed[s] for s in EVAL_SEEDS], dtype=np.float64)

    delta_a = _mean_ci(wins("z0", "A") - wins("z1", "A"))
    delta_b = _mean_ci(wins("z1", "B") - wins("z0", "B"))
    delta_a["passes"] = delta_a["lcb95"] > 0
    delta_b["passes"] = delta_b["lcb95"] > 0

    matrix = {f"{name}|{pole}": _mean_ci(wins(name, pole))
              for name, pole in (("z0", "A"), ("z1", "A"), ("z0", "B"),
                                 ("z1", "B"), ("pi_A", "A"), ("pi_B", "B"))}
    retention = _ratio_ci(0.5 * (wins("z0", "A") + wins("z1", "B")),
                          0.5 * (wins("pi_A", "A") + wins("pi_B", "B")))

    passed = bool(delta_a["passes"] and delta_b["passes"])
    labels = spec["PRIMARY_GATE_CROSSOVER"]["verdict_labels"]
    verdict = labels["pass"] if passed else labels["fail"]

    print("\n  CROSS-EVALUATION MATRIX (win rate, seed-level CI)")
    for key in ("z0|A", "z1|A", "z0|B", "z1|B", "pi_A|A", "pi_B|B"):
        metric = matrix[key]
        print(f"    {key:8s} {metric['mean']:.4f}  "
              f"[{metric['lcb95']:.4f}, {metric['ucb95']:.4f}]")
    print(f"\n  delta_A = z0|A - z1|A : {delta_a['mean']:+.4f} "
          f"[{delta_a['lcb95']:+.4f}, {delta_a['ucb95']:+.4f}]  "
          f"{'PASS' if delta_a['passes'] else 'FAIL'}")
    print(f"  delta_B = z1|B - z0|B : {delta_b['mean']:+.4f} "
          f"[{delta_b['lcb95']:+.4f}, {delta_b['ucb95']:+.4f}]  "
          f"{'PASS' if delta_b['passes'] else 'FAIL'}")
    print(f"  retention (context only): {retention['mean']:.4f} "
          f"[{retention['lcb95']:.4f}, {retention['ucb95']:.4f}]")

    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    OUT.write_text(json.dumps({
        "record": "H-OG-PSP V3 held-out payoff EVAL",
        "status": "FROZEN_RESULT",
        "one_shot": True,
        "utc": _now(),
        "implements": "HOG_PSP_V3_EVAL_SPEC.json",
        "VERDICT": verdict,
        "trajectory_identity_frozen_before_eval": "TRAJECTORY_IDENTITY_CONFIRMED",
        "checkpoint": {
            "path": frozen["TERMINAL_CHECKPOINT"]["path"],
            "sha256": frozen["TERMINAL_CHECKPOINT"]["sha256"],
            "verified_at_eval_time": True,
            "terminal_only": True,
        },
        "seeds": {"block": [EVAL_SEEDS[0], EVAL_SEEDS[-1]], "n": len(EVAL_SEEDS)},
        "payoff_matrix": matrix,
        "delta_A": delta_a,
        "delta_B": delta_b,
        "gate": "BOTH delta lcb95 > 0",
        "retention_context_only_not_a_gate": retention,
        "bootstrap": {
            "procedure": "paired percentile bootstrap over seed-level episode outcomes",
            "samples": N_BOOT,
            "alpha": ALPHA,
            "rng_seed": BOOTSTRAP_SEED,
            "unit": "evaluation seed",
            "provenance": "unchanged from OG-PSP, V1, and EXP2C",
        },
        "interpretive_context_frozen_before_the_result":
            spec["INTERPRETIVE_CONTEXT_FROZEN_BEFORE_THE_RESULT"],
        "no_model_selection_occurred": True,
        "total_episodes": len(rows),
    }, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}")
    print(f"  -> {OUT}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
