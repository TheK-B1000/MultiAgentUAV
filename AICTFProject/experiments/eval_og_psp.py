"""One-shot held-out EVAL for the OG-PSP run. Implements the frozen spec.

Scores the cross-evaluation matrix on seeds never used for training, rehearsal, V2
bank collection, calibration, or the mechanism diagnostic:

    V(z0, A)   V(z1, A)        Pole A = OP6 + SDS2_A_payoff_INIT_3 overlay
    V(z0, B)   V(z1, B)        Pole B = OP7, no overlay

Crossover gate, carried over UNCHANGED from eval_oracle_gated_k2.py -- which itself
adopted it unchanged from eval_exp2_k2_terminal.py -- so the criterion cannot be
accused of having been chosen for this result:

    delta_A = wins(z0,A) - wins(z1,A)      PASS iff lcb95 > 0
    delta_B = wins(z1,B) - wins(z0,B)      PASS iff lcb95 > 0

BOTH must pass. The mechanism diagnostic showed the differentiation is one-sided, so
a B-passes/A-fails outcome is the specifically foreseeable partial result; the spec
pre-classifies it as NOT CONFIRMED.

Paired percentile bootstrap over seed-level outcomes, 20000 samples, alpha 0.05.
Point estimates alone do not pass.

Terminal checkpoint only, verified by sha256 against the freeze record. Retention
against the teachers is reported for CONTEXT ONLY and does not gate.

This is the FINAL latent attempt under the stopping rule.

Run:  python experiments/eval_og_psp.py --device cuda
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
SPEC = SD / "sppo" / "OG_PSP_EVAL_SPEC.json"
FROZEN = SD / "sppo" / "OG_PSP_MODEL_FROZEN.json"
OUT = SD / "sppo" / "OG_PSP_EVAL_RESULT.json"
ROWS_CSV = SD / "sppo" / "og_psp_eval_rows.csv"

EVAL_SEEDS = list(range(11_200_001, 11_200_033))
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
    if OUT.is_file():
        raise SystemExit(f"REFUSING: {OUT} exists; EVAL is opened ONCE")
    if spec["SEEDS"]["block"] != "11200001..11200032":
        raise SystemExit("REFUSING: eval seed block drifted from the frozen spec")
    if [EVAL_SEEDS[0], EVAL_SEEDS[-1]] != [11_200_001, 11_200_032]:
        raise SystemExit("REFUSING: scorer seed range does not match the frozen block")
    return spec, frozen, ck


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
    )
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo import load_custom_ppo_policy

    spec, frozen, ck_path = _preflight()
    device = a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu"

    print(f"OG-PSP HELD-OUT EVAL  {_now()}")
    print(f"  checkpoint sha256 VERIFIED against the freeze record and the eval spec")
    print(f"  seeds {EVAL_SEEDS[0]}..{EVAL_SEEDS[-1]}  (n={len(EVAL_SEEDS)})")
    print(f"  gate: BOTH delta lcb95 > 0, paired bootstrap n={N_BOOT}, alpha={ALPHA}")
    print(f"  one passing side is NOT a pass (pre-classified in the frozen spec)\n",
          flush=True)

    probe = R2.build_env(device, EVAL_SEEDS[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()

    student = load_custom_ppo_policy(str(ck_path), obs_space, act_space, device=device)
    if student.model.latent_k != 2 or not student.model.uses_latent_strategy:
        raise SystemExit("REFUSING: checkpoint is not a latent K=2 policy")
    teachers = {k: load_custom_ppo_policy(str(v), obs_space, act_space, device=device)
                for k, v in P0.TEACHERS.items()}

    def run_cell(policy, pole: str, z: int | None, seed: int) -> dict:
        """One deterministic episode.

        Latent forcing uses the SAME mechanism as the frozen EXP2 and V1 evals --
        fixed_latent_strategy / fixed_latent_strategy_id then reset_strategy(). An
        earlier draft of the V1 scorer guessed a set_forced_latent() that does not
        exist, which would have run z0 and z1 as the SAME policy and produced a
        meaningless crossover.

        global_state must be injected into obs on reset and every step, or the router
        receives all-zero context.
        """
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
                                       context=f"eval {pole} seed {seed}")
            terminal = None
            for _ in range(R2.MAX_STEPS):
                action, _ = policy.predict(obs, deterministic=True)
                env.step_async(action)
                obs, _reward, done, info = env.step_wait()
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

    cells = [("z0", 0, student), ("z1", 1, student),
             ("pi_A", None, teachers["pi_A"]), ("pi_B", None, teachers["pi_B"])]
    rows = []
    for name, z, pol in cells:
        for pole in POLES:
            if name == "pi_A" and pole != "A":
                continue
            if name == "pi_B" and pole != "B":
                continue
            for seed in EVAL_SEEDS:
                r = run_cell(pol, pole, z, seed)
                rows.append({"policy": name, "pole": pole, "seed": seed, **r})
            wr = np.mean([r["win"] for r in rows if r["policy"] == name and r["pole"] == pole])
            print(f"  {name:5s} on Pole {pole}: win rate {wr:.4f}", flush=True)

    def wins(name: str, pole: str) -> np.ndarray:
        by = {r["seed"]: r["win"] for r in rows if r["policy"] == name and r["pole"] == pole}
        return np.array([by[s] for s in EVAL_SEEDS], dtype=np.float64)

    delta_a = _mean_ci(wins("z0", "A") - wins("z1", "A"))
    delta_b = _mean_ci(wins("z1", "B") - wins("z0", "B"))
    delta_a["passes"] = delta_a["lcb95"] > 0
    delta_b["passes"] = delta_b["lcb95"] > 0

    matrix = {f"{n}|{p}": _mean_ci(wins(n, p))
              for n, p in (("z0", "A"), ("z1", "A"), ("z0", "B"), ("z1", "B"),
                           ("pi_A", "A"), ("pi_B", "B"))}
    retention = _ratio_ci(0.5 * (wins("z0", "A") + wins("z1", "B")),
                          0.5 * (wins("pi_A", "A") + wins("pi_B", "B")))

    passed = bool(delta_a["passes"] and delta_b["passes"])
    verdict = ("OG_PSP_CROSSOVER_CONFIRMED" if passed
               else "OG_PSP_CROSSOVER_NOT_CONFIRMED")
    if delta_b["passes"] and not delta_a["passes"]:
        asymmetry = ("B side passed, A side did not. Pre-classified in the frozen spec as "
                     "NOT CONFIRMED. OG-PSP induced generalisable latent differentiation, "
                     "but asymmetrically, primarily as a B-specialist rather than two "
                     "complementary specialists.")
    elif delta_a["passes"] and not delta_b["passes"]:
        asymmetry = ("A side passed, B side did not. One passing side is not a pass. This "
                     "is the opposite asymmetry from the one the mechanism diagnostic "
                     "predicted, which is itself worth recording.")
    else:
        asymmetry = None

    print("\n  CROSS-EVALUATION MATRIX (win rate, seed-level CI)")
    for k in ("z0|A", "z1|A", "z0|B", "z1|B", "pi_A|A", "pi_B|B"):
        m = matrix[k]
        print(f"    {k:8s} {m['mean']:.4f}  [{m['lcb95']:.4f}, {m['ucb95']:.4f}]")
    print(f"\n  delta_A = z0|A - z1|A : {delta_a['mean']:+.4f} "
          f"[{delta_a['lcb95']:+.4f}, {delta_a['ucb95']:+.4f}]  "
          f"{'PASS' if delta_a['passes'] else 'FAIL'}")
    print(f"  delta_B = z1|B - z0|B : {delta_b['mean']:+.4f} "
          f"[{delta_b['lcb95']:+.4f}, {delta_b['ucb95']:+.4f}]  "
          f"{'PASS' if delta_b['passes'] else 'FAIL'}")
    print(f"  retention (context only): {retention['mean']:.4f} "
          f"[{retention['lcb95']:.4f}, {retention['ucb95']:.4f}]")

    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    OUT.write_text(json.dumps({
        "record": "OG-PSP held-out EVAL",
        "status": "FROZEN_RESULT", "one_shot": True, "utc": _now(),
        "implements": "OG_PSP_EVAL_SPEC.json",
        "VERDICT": verdict,
        "checkpoint": {"path": frozen["TERMINAL_CHECKPOINT"]["path"],
                       "sha256": frozen["TERMINAL_CHECKPOINT"]["sha256"],
                       "verified_at_eval_time": True,
                       "terminal_only": True},
        "seeds": {"block": [EVAL_SEEDS[0], EVAL_SEEDS[-1]], "n": len(EVAL_SEEDS)},
        "payoff_matrix": matrix,
        "delta_A": delta_a, "delta_B": delta_b,
        "gate": "BOTH delta lcb95 > 0",
        "asymmetry_note": asymmetry,
        "retention_context_only_not_a_gate": retention,
        "bootstrap": {"procedure": "paired percentile bootstrap over seed-level episode outcomes",
                      "samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOTSTRAP_SEED,
                      "unit": "evaluation seed",
                      "provenance": "unchanged from V1, which took it unchanged from EXP2C"},
        "interpretive_context_frozen_before_the_result":
            spec["INTERPRETIVE_CONTEXT_FROZEN_BEFORE_THE_RESULT"],
        "no_model_selection_occurred": True,
        "total_episodes": len(rows),
        "stopping_rule": (
            "Final latent attempt. A negative verdict closes latent-strategy work and "
            "pivots to the SAPPO-centered ICRA paper."),
    }, indent=2), encoding="utf-8")
    print(f"\n  VERDICT: {verdict}")
    print(f"  -> {OUT}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
