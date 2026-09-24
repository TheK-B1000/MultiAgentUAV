"""Standalone crossover evaluation for a scaled latent-dispatched (Rung-format) policy.

Asks the SAME question eval_specialist_crossover_scaled.py asks of two separate specialist
checkpoints, but for ONE latent-conditioned policy dispatched by z:

    delta_A = WR(z0,A) - WR(z1,A)
    delta_B = WR(z1,B) - WR(z0,B)

PASS iff both means > 0 AND both LCB95 > 0 -- identical criterion, identical bootstrap
(n_boot=20000, alpha=0.05, rng_seed=7) to every other gate in this program.

Deliberately NOT eval_ladder_rung1_matched.py's ladder-relative D_A/D_B comparison against a
Rung-0 reference -- there is no Rung-0-at-scale, and this evaluator asks the simpler, absolute
question: does this ONE frozen latent policy still express two behaviorally distinct,
oppositely-useful strategies. That is sufficient evidence the latent mechanism scales; it is
not a claim about how sharing compares to zero sharing at this team size.

Run:  python experiments/eval_rung1_crossover_scaled.py --team-size 6 --rung 1 \\
          --checkpoint <path> --spec <frozen spec> --seed-base <base> --n-seeds 128 \\
          --label RUNG1_6V6 --device cuda
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

from experiments.eval_hog_psp_v3 import _mean_ci  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
N_BOOT, ALPHA, BOOTSTRAP_SEED = 20_000, 0.05, 7
BASE_KEY = {"A": "OP6", "B": "OP7"}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team-size", type=int, required=True, choices=(2, 4, 6))
    ap.add_argument("--rung", type=int, required=True, choices=(0, 1, 2, 3))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--spec", required=True, help="frozen spec governing this evaluation")
    ap.add_argument("--seed-base", type=int, required=True)
    ap.add_argument("--n-seeds", type=int, default=128)
    ap.add_argument("--label", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    N = int(args.team_size)
    label = str(args.label)
    seeds = list(range(int(args.seed_base), int(args.seed_base) + int(args.n_seeds)))

    OUT = SD / f"{label}_CROSSOVER_EVAL_RESULT.json"
    ROWS_CSV = SD / f"{label.lower()}_crossover_eval_rows.csv"
    PREAUDIT_FLAG = SD / f"{label}_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json"

    spec_path = Path(args.spec)
    if not spec_path.is_file():
        raise SystemExit(f"REFUSING: spec not found: {spec_path}")
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: spec not frozen: {spec.get('status')!r}")

    ck = Path(args.checkpoint)
    if not ck.is_file():
        raise SystemExit(f"REFUSING: checkpoint missing: {ck}")
    if OUT.is_file() or ROWS_CSV.is_file() or PREAUDIT_FLAG.is_file():
        raise SystemExit(f"REFUSING: an output for label {label!r} already exists; one-shot")

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays,
        pole_A_genome, pole_B_genome,
    )
    import experiments.r2_learned_crossover as R2
    from rl import ladder_rung1 as L1
    from rl.curriculum import phase_from_tag

    R2.AGENTS = N
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    genomes_by_pole = {
        "A": {"OP6": pole_A_genome(N)},
        "B": {"OP7": pole_B_genome(N)} if N != 2 else {},
    }

    print(f"RUNG-{args.rung} CROSSOVER EVAL  {label}  {N}v{N}  {_now()}")
    print(f"  spec       {spec_path.name}  [{spec.get('status')}]  arm={spec.get('arm', 'n/a')}")
    print(f"  checkpoint {ck}  sha {_sha(ck)[:12]}...")
    print(f"  seeds      {seeds[0]}..{seeds[-1]} (n={len(seeds)}), SHARED across z and poles")
    print(f"  poles      A: OP6+{dict(pole_A_genome(N).overlay or {})}   "
          f"B: OP7+{dict(pole_B_genome(N).overlay or {})}")
    print(f"  gate       delta_A > 0 & LCB95 > 0; delta_B symmetric")
    print(f"  bootstrap  n={N_BOOT}, alpha={ALPHA}, rng_seed={BOOTSTRAP_SEED}\n", flush=True)

    probe = R2.build_env(device, seeds[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    grid_agents = int(obs_space.spaces["grid"].shape[0])
    probe.close()
    if grid_agents != N:
        raise SystemExit(f"FAIL-CLOSED: env grid agent dim {grid_agents} != team size {N}")

    model, branch_cfg, _ = L1.load_rung(args.rung, str(ck), obs_space, act_space, device=device)
    policy = L1.make_dispatch_policy(model, branch_cfg, device=device)

    def run_cell(z: int, pole: str, seed: int) -> dict:
        env = R2.build_env(device, seed)
        core = env.core
        try:
            policy.fixed_latent_strategy = True
            policy.fixed_latent_strategy_id = int(z)
            policy.reset_strategy()
            core._bt_profile_override = None
            core._sds_opening_hold_steps = 0
            genomes = genomes_by_pole[pole]
            install_keyed_opponent_overlays(core, genomes)
            key = BASE_KEY[pole]
            env.env_method("set_phase", phase_from_tag(key))
            env.env_method("set_next_opponent", "SCRIPTED", key)
            obs = env.reset()
            obs["global_state"] = env.state()
            assert_live_opponent_batch(core, genomes, allowed_keys=(key,),
                                       context=f"{label} z{z}@{pole} seed {seed}")
            resolved = core._bt_resolved_profile_tensors()
            got = resolved.get("min_alive_for_defender")
            got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
            if got_val != N:
                raise SystemExit(f"FAIL-CLOSED: live pole {pole} resolves "
                                 f"min_alive_for_defender={got_val}, expected {N}")
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

    if args.dry_run:
        for pole in ("A", "B"):
            env = R2.build_env(device, 99_990_000 + N)
            try:
                core = env.core
                core._bt_profile_override = None
                install_keyed_opponent_overlays(core, genomes_by_pole[pole])
                key = BASE_KEY[pole]
                env.env_method("set_phase", phase_from_tag(key))
                env.env_method("set_next_opponent", "SCRIPTED", key)
                env.reset()
                resolved = core._bt_resolved_profile_tensors()
                got = resolved.get("min_alive_for_defender")
                got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
                print(f"  dry-run pole {pole}: live min_alive_for_defender={got_val} "
                      f"(expect {N}) {'OK' if got_val == N else 'MISMATCH'}")
                if got_val != N:
                    raise SystemExit("FAIL-CLOSED: dry-run live pole mismatch")
            finally:
                env.close()
        print("\n  --dry-run: spec frozen, checkpoint present, env at N, both poles resolve at N. "
              "NO episodes run, NOTHING written.")
        return 0

    from experiments.tqdm_loop import set_postfix, tqdm_iter

    cells = [(z, pole, seed) for z in (0, 1) for pole in ("A", "B") for seed in seeds]
    rows = []
    bar = tqdm_iter(cells, desc=f"{label} crossover", unit="ep")
    for z, pole, seed in bar:
        set_postfix(bar, f"z{z}@Pole{pole} seed={seed}")
        rows.append({"z": z, "pole": pole, "seed": seed, **run_cell(z, pole, seed)})
        if seed == seeds[-1]:
            wr = np.mean([r["win"] for r in rows if r["z"] == z and r["pole"] == pole])
            print(f"  z{z} on Pole {pole}: win rate {wr:.4f}", flush=True)

    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def wins(z, pole):
        by = {r["seed"]: r["win"] for r in rows if r["z"] == z and r["pole"] == pole}
        return np.array([by[s] for s in seeds], dtype=np.float64)

    delta_a = _mean_ci(wins(0, "A") - wins(1, "A"))
    delta_b = _mean_ci(wins(1, "B") - wins(0, "B"))
    delta_a["passes"] = bool(delta_a["mean"] > 0 and delta_a["lcb95"] > 0)
    delta_b["passes"] = bool(delta_b["mean"] > 0 and delta_b["lcb95"] > 0)

    tie_or_reversal = [k for k, d in (("delta_A", delta_a), ("delta_B", delta_b))
                       if d["mean"] <= 0.0]
    if tie_or_reversal:
        PREAUDIT_FLAG.write_text(json.dumps({
            "record": f"{label} crossover EVAL integrity audit REQUIRED",
            "status": "FLAGGED", "utc": _now(),
            "implements": f"{spec_path.name}#EVALUATION.tie_or_reversal",
            "triggered_by": tie_or_reversal,
            "point_estimates": {"delta_A": delta_a["mean"], "delta_B": delta_b["mean"]},
            "rule": "requires a row-level integrity audit before any verdict-bearing result. "
                    f"Raw rows: {ROWS_CSV.name}.",
        }, indent=2), encoding="utf-8")
        print(f"\n  TIE/REVERSAL on {tie_or_reversal} -- integrity audit REQUIRED.")
        print(f"  -> {PREAUDIT_FLAG}")
        return 0

    gate_passes = bool(delta_a["passes"] and delta_b["passes"])
    print("\n  PRIMARY GATE")
    print(f"    delta_A {delta_a['mean']:+.4f} [{delta_a['lcb95']:+.4f}, {delta_a['ucb95']:+.4f}]"
          f" {'PASS' if delta_a['passes'] else 'FAIL'}")
    print(f"    delta_B {delta_b['mean']:+.4f} [{delta_b['lcb95']:+.4f}, {delta_b['ucb95']:+.4f}]"
          f" {'PASS' if delta_b['passes'] else 'FAIL'}")
    print(f"\n  GATE: {'PASS' if gate_passes else 'FAIL'}")

    OUT.write_text(json.dumps({
        "record": f"{label} crossover EVAL", "status": "FROZEN_RESULT",
        "one_shot": True, "utc": _now(),
        "arm": spec.get("arm", "n/a"), "confirmatory": bool(spec.get("confirmatory", False)),
        "implements": f"{spec_path.name}#EVALUATION",
        "team_size": N, "rung": int(args.rung), "device": device,
        "checkpoint_sha256": _sha(ck),
        "seeds": {"block": [seeds[0], seeds[-1]], "n": len(seeds), "shared_across_z_and_poles": True},
        "poles": {p: {"base": BASE_KEY[p],
                      "overlay": dict((pole_A_genome(N) if p == "A" else pole_B_genome(N)).overlay or {})}
                  for p in ("A", "B")},
        "PRIMARY_GATE": {"delta_A": delta_a, "delta_B": delta_b, "passes": gate_passes},
        "bootstrap": {"procedure": "paired percentile bootstrap over evaluation seeds",
                      "samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOTSTRAP_SEED},
        "no_model_selection_occurred": True, "total_episodes": len(rows),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0 if gate_passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
