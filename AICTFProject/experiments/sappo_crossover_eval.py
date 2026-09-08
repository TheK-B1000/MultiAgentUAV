"""SAPPO V1 — the crossover gate, on the untouched block 7800001..7800192.

IDENTICAL procedure to R2 (experiments/r2_learned_crossover.py): same paired
design, same frozen bootstrap (rng 7, 20000 resamples, alpha 0.05), same gate,
one block, no extension, no new statistic. Only the checkpoints and the block
differ, so R2 and SAPPO are directly comparable.

R2 result for reference (vanilla 1M specialists):
    delta_A +0.0833  LCB95 -0.0156  FAIL
    delta_B +0.0625  LCB95 -0.0260  FAIL

Nothing from training can rescue this: anchor loss, teacher agreement and the
measured cadence are health diagnostics. A policy can imitate its teacher well
on the teacher's own states and still drift on its own state distribution
against the live opponent, which is exactly where R2 failed.


The first experiment that can tell us whether PPO actually discovered the two
strategies the environment was engineered to demand.

Frozen gate (REPERTOIRE_LADDER_FROZEN.json):

    delta_A = WR(pi_A, A) - WR(pi_B, A)
    delta_B = WR(pi_B, B) - WR(pi_A, B)

    PASS iff  LCB95(delta_A) > 0  AND  LCB95(delta_B) > 0

No magnitude floor, deliberately. R2 asks only whether the specialists learned
different best responses; whether that difference is worth anything is R3's
question, and putting a second threshold here would measure value twice.

Discipline enforced by this script:

  * TERMINAL checkpoints only. Intermediate checkpoints are diagnostics and
    cannot be substituted.
  * The generalist does NOT participate in the crossover contrasts. R2 is a
    statement about pi_A vs pi_B. pi_G is evaluated here only so R3 has its
    matrix row on the same seeds, and its cells are reported, never gated on.
  * Paired seeds within each pole: all three policies see the SAME episode
    seeds, so the bootstrap is over seed-level differences.
  * Block 7500001..7500192 is spent once. No extension, no re-run.

Both poles are instantiated through the asserted R0 seam, so a policy cannot be
scored against the wrong opponent.

Run:  python experiments/r2_learned_crossover.py --device cuda
      python experiments/r2_learned_crossover.py --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.m1_payoff_assay import paired_ci                      # noqa: E402
from experiments.opponent_spec import (                                # noqa: E402
    assert_live_opponent_batch, install_keyed_opponent_overlays,
    pole_A_genome,
)
from rl.curriculum import phase_from_tag                                # noqa: E402

SD = ROOT / "artifacts/strategic_demand"
R1 = SD / "r1_training"
OUT_DIR = SD / "sappo_crossover"

SC = SD / "sappo_continuation"
POLICIES = {
    "pi_A": SC / "sappo_pi_A_specialist_1p5M_seed7100001/ckpts/final_sappo_pi_A_specialist_1p5M_seed7100001.zip",
    "pi_B": SC / "sappo_pi_B_specialist_1p5M_seed7200001/ckpts/final_sappo_pi_B_specialist_1p5M_seed7200001.zip",
}
POLES = {"A": "OP6", "B": "OP7"}

SEED_BASE = 7_800_001
N_PAIRED = 192
BOOTSTRAP_RNG = 7
MAP = "map_a_open"
MAX_STEPS = 240
AGENTS = 2
RULESET = dict(taggers_required=1, tag_min_interval_seconds=10.0,
               tag_nearest_only=True, tag_channel_seconds=0.0,
               suppression_attackers_required=2)
SPENT_OR_DISQUALIFIED = {2500001, 2600001, 5000001, 6000001,
                         7000001, 7100001, 7200001, 7500001,
                         7600001, 7900001, 8000001}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def guard_rails() -> None:
    lo, hi = SEED_BASE, SEED_BASE + N_PAIRED - 1
    for bad in SPENT_OR_DISQUALIFIED:
        if lo <= bad <= hi:
            raise SystemExit(f"REFUSING: block {lo}..{hi} touches spent/disqualified {bad}")
    if (OUT_DIR / "summary.json").is_file():
        raise SystemExit(f"REFUSING: {OUT_DIR/'summary.json'} exists. The block is "
                         "spent once; no re-run, no extension.")
    for name, ck in POLICIES.items():
        if not ck.is_file():
            raise SystemExit(f"REFUSING: terminal checkpoint missing for {name}: {ck}")
        if not ck.name.startswith("final_"):
            raise SystemExit(f"REFUSING: {name} is not a terminal checkpoint: {ck.name}")


def build_env(device: str, seed: int):
    """One env per episode, seeded with THAT episode's seed.

    Seeding must happen at GPUFieldConfig construction: there is no
    core.set_seed, so a post-construction reseed would be a silent no-op and
    every episode in the block would be the identical rollout.
    """
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=AGENTS, max_red_agents=AGENTS, map_set="train",
        map_layout=MAP, max_decision_steps=MAX_STEPS, aquaticus_profile=True,
        rules_profile="OURS", device=device, seed=int(seed),
        # obstacle_obs_channel is deliberately NOT set: R1 trained with the
        # default (7 CNN input channels). Forcing True here produced an 8th
        # channel and made the loader zero-init an expansion the policies never
        # trained on. Behaviourally equivalent (mean_kl=0), but evaluating under
        # a different observation space than training is not a shim to rely on.
        tag_telemetry_enabled=True,
        own_flag_home_required_to_score=True, **RULESET)
    return GPUCTFVecEnv(cfg)


def score_cell(model, pole: str, *, device: str, n: int) -> list[dict]:
    """One (policy, pole) cell across the paired seed block."""
    base_key = POLES[pole]
    genomes = {"OP6": pole_A_genome()} if pole == "A" else {}
    rows = []
    for i in range(n):
        seed = SEED_BASE + i
        env = build_env(device, seed)
        core = env.core
        try:
            core._bt_profile_override = None
            core._sds_opening_hold_steps = 0
            install_keyed_opponent_overlays(core, genomes)
            env.env_method("set_phase", phase_from_tag(base_key))
            env.env_method("set_next_opponent", "SCRIPTED", base_key)
            obs = env.reset()
            assert_live_opponent_batch(core, genomes, allowed_keys=(base_key,),
                                       context=f"R2 pole {pole} seed {seed}")
            term = None
            steps = 0
            for _ in range(MAX_STEPS):
                action, _ = model.predict(obs, deterministic=True)
                env.step_async(action)
                obs, _r, done, info = env.step_wait()
                steps += 1
                if bool(np.asarray(done).any()):
                    i0 = info[0] if isinstance(info, (list, tuple)) else info
                    er = (i0 or {}).get("episode_result") or {}
                    term = (int(er.get("blue_score", 0)), int(er.get("red_score", 0)))
                    break
            if term is None:
                term = (int(core.blue_score[0]), int(core.red_score[0]))
            b, r = term
            rows.append({"pole": pole, "opponent": base_key, "episode_seed": seed,
                         "blue_score": b, "red_score": r, "win": int(b > r),
                         "draw": int(b == r), "steps": steps,
                         "zero_zero": int(b == 0 and r == 0),
                         "total_score": b + r})
        finally:
            env.close()
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=N_PAIRED)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    guard_rails()
    print(f"SAPPO V1 CROSSOVER GATE  {_now()}")
    print(f"  block    {SEED_BASE}..{SEED_BASE + a.n - 1}  ({a.n} paired seeds)")
    print(f"  policies {list(POLICIES)}  (TERMINAL checkpoints only)")
    print(f"  poles    A=OP6+overlay, B=OP7 canonical")
    print(f"  gate     LCB95(delta_A) > 0 AND LCB95(delta_B) > 0, no magnitude floor")
    print(f"  episodes {a.n * len(POLICIES) * len(POLES)} total")
    if a.dry_run:
        print("\nDRY RUN -- no episode run, block untouched.")
        return 0

    from rl.custom_ppo import load_custom_ppo_policy
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    probe = build_env(a.device, SEED_BASE)
    obs_space, act_space = probe.observation_space, probe.action_space
    probe.close()

    matrix, all_rows = {}, []
    for pname, ckpt in POLICIES.items():
        model = load_custom_ppo_policy(str(ckpt), obs_space, act_space,
                                       device=a.device)
        for pole in POLES:
            print(f"  scoring {pname} vs pole {pole} ...", flush=True)
            rows = score_cell(model, pole, device=a.device, n=a.n)
            for r in rows:
                r["policy"] = pname
            all_rows.extend(rows)
            wr = float(np.mean([r["win"] for r in rows]))
            matrix[f"{pname}|{pole}"] = {
                "win_rate": wr,
                "frac_0_0": float(np.mean([r["zero_zero"] for r in rows])),
                "mean_total_score": float(np.mean([r["total_score"] for r in rows])),
            }
            print(f"    WR({pname}, {pole}) = {wr:.4f}", flush=True)

    def wins(p, pole):
        return np.array([r["win"] for r in all_rows
                         if r["policy"] == p and r["pole"] == pole], dtype=float)

    dA = wins("pi_A", "A") - wins("pi_B", "A")
    dB = wins("pi_B", "B") - wins("pi_A", "B")
    mA, loA, hiA = paired_ci(dA, np.random.default_rng(BOOTSTRAP_RNG))
    mB, loB, hiB = paired_ci(dB, np.random.default_rng(BOOTSTRAP_RNG))
    passA, passB = bool(loA > 0.0), bool(loB > 0.0)
    verdict = ("SAPPO_V1_CROSSOVER_CONFIRMED" if (passA and passB)
               else "SAPPO_V1_CROSSOVER_NOT_CONFIRMED")

    summary = {
        "record": "SAPPO V1 crossover gate", "utc": _now(),
        "protocol": "artifacts/strategic_demand/STRATEGY_ANCHORED_PPO_V1_FROZEN.json",
        "terminals": "artifacts/strategic_demand/SAPPO_V1_TERMINALS_LOCKED.json",
        "r2_reference": {"delta_A": 0.0833, "delta_A_lcb": -0.0156,
                         "delta_B": 0.0625, "delta_B_lcb": -0.0260,
                         "verdict": "R2_LEARNED_CROSSOVER_NOT_CONFIRMED"},
        "block": f"{SEED_BASE}..{SEED_BASE + a.n - 1}", "n_paired": a.n,
        "total_episodes": len(all_rows),
        "checkpoints": {k: str(v.relative_to(ROOT)) for k, v in POLICIES.items()},
        "payoff_matrix": matrix,
        "delta_A": {"contrast": "WR(pi_A,A) - WR(pi_B,A)", "mean": mA,
                    "lcb95": loA, "ucb95": hiA, "passes": passA},
        "delta_B": {"contrast": "WR(pi_B,B) - WR(pi_A,B)", "mean": mB,
                    "lcb95": loB, "ucb95": hiB, "passes": passB},
        "gate": "LCB95 > 0 in BOTH directions; no magnitude floor by design",
        "generalist_not_evaluated": ("pi_G was NOT continued under SAPPO, so it has no "
                                    "counterpart here. R3 remains gated behind this test."),
        "verdict": verdict,
        "bootstrap": {"n_boot": 20000, "alpha": 0.05, "rng_seed": BOOTSTRAP_RNG,
                      "procedure": "paired percentile bootstrap over seed-level differences"},
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with open(OUT_DIR / "episode_rows.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    print("\n" + "=" * 66)
    print(f"{'':<8}{'A':>10}{'B':>10}")
    for p in POLICIES:
        print(f"{p:<8}{matrix[f'{p}|A']['win_rate']:>10.4f}{matrix[f'{p}|B']['win_rate']:>10.4f}")
    print("-" * 66)
    print(f"delta_A = {mA:+.4f}  LCB95 {loA:+.4f}  {'PASS' if passA else 'FAIL'}")
    print(f"delta_B = {mB:+.4f}  LCB95 {loB:+.4f}  {'PASS' if passB else 'FAIL'}")
    print("=" * 66)
    print(f"VERDICT: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
