"""Forced-z crossover eval for the suite's distilled sharing arms, at any team size.

Loads suite ``.pt`` students (not PPO ``.zip``), forces z, augments entity
tensors, and applies the program gate:

    delta_A = V(z0, A) - V(z1, A)
    delta_B = V(z1, B) - V(z0, B)

PASS iff both means > 0 AND both LCB95 > 0 (n_boot=20000, alpha=0.05, rng=7).

Implements SUITE_SHARING_<N>V<N>_CROSSOVER_EVAL_SPEC.json. Team size is an argument;
there is no module-level team size (CROSS_SCALE_CANONICAL_RECIPE_V1.json#STAGE_IMPLEMENTATIONS_required).

Run:
  python experiments/eval_suite_sharing_crossover.py --team-size 4 --arm fully_shared --dry-run
  python experiments/eval_suite_sharing_crossover.py --team-size 4 --arm share_encoder --device cuda
  python experiments/eval_suite_sharing_crossover.py --team-size 4 --arm share_backbone --device cuda
  python experiments/eval_suite_sharing_crossover.py --team-size 4 --arm share_macro --device cuda
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
SUPPORTED_TEAM_SIZES = (2, 4, 6)
#: CLOSEST_DEFENDS defender count per scale -- the one scale knob besides N.
K_DEFEND_BY_SCALE = {2: 1, 4: 2, 6: 1}


def _spec_path(n: int) -> Path:
    return SD / f"SUITE_SHARING_{n}V{n}_CROSSOVER_EVAL_SPEC.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        required=True,
        choices=("fully_shared", "share_encoder", "share_backbone", "share_macro"),
    )
    ap.add_argument(
        "--team-size", type=int, required=True, choices=SUPPORTED_TEAM_SIZES,
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--n-seeds", type=int, default=None, help="override SPEC n (smoke only)")
    ap.add_argument("--seed-base", type=int, default=None, help="override SPEC seed base")
    args = ap.parse_args()

    N_AGENTS = int(args.team_size)
    SPEC_PATH = _spec_path(N_AGENTS)
    if not SPEC_PATH.is_file():
        raise SystemExit(
            f"FAIL-CLOSED: {SPEC_PATH.name} not found. Each scale needs its own frozen "
            f"crossover eval spec pinning that scale's arms, seeds and poles."
        )
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: {SPEC_PATH.name} not frozen: {spec.get('status')!r}")

    arm_key = {
        "fully_shared": "fully_shared_z",
        "share_encoder": "share_encoder",
        "share_backbone": "share_backbone",
        "share_macro": "share_macro",
    }[args.arm]
    if arm_key not in spec["ARMS"]:
        raise SystemExit(f"REFUSING: SPEC missing ARMS[{arm_key!r}] — pin after distill freeze")
    arm = spec["ARMS"][arm_key]
    label = str(arm["label_exploratory"])
    seed_key = f"{arm_key}_exploratory"
    seed_range = str(spec["SEEDS"][seed_key])
    seed_base = int(args.seed_base) if args.seed_base else int(seed_range.split("..")[0])
    n_seeds = int(args.n_seeds) if args.n_seeds else int(spec["SEEDS"]["n_exploratory"])
    seeds = list(range(seed_base, seed_base + n_seeds))

    OUT = SD / f"{label}_CROSSOVER_EVAL_RESULT.json"
    ROWS_CSV = SD / f"{label.lower()}_crossover_eval_rows.csv"
    PREAUDIT_FLAG = SD / f"{label}_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json"
    LOG = SD / "suite_sharing" / f"{N_AGENTS}v{N_AGENTS}" / args.arm / "crossover_eval.log"

    ck = ROOT / arm["checkpoint"]
    if not ck.is_file():
        raise SystemExit(f"REFUSING: checkpoint missing: {ck}")
    if _sha(ck) != arm["sha256"]:
        raise SystemExit(f"REFUSING: checkpoint sha mismatch vs SPEC pin")
    if not args.dry_run and (OUT.is_file() or ROWS_CSV.is_file() or PREAUDIT_FLAG.is_file()):
        raise SystemExit(f"REFUSING: an output for label {label!r} already exists; one-shot")

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch,
        install_keyed_opponent_overlays,
        pole_A_genome,
        pole_B_genome,
    )
    import experiments.r2_learned_crossover as R2
    from experiments.tqdm_loop import set_postfix, tqdm_iter
    from gpu_env._core._entity_obs import augment_obs_with_entities
    from rl.curriculum import phase_from_tag

    R2.AGENTS = N_AGENTS
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    genomes_by_pole = {
        "A": {"OP6": pole_A_genome(N_AGENTS)},
        "B": {"OP7": pole_B_genome(N_AGENTS)},
    }

    print(f"SUITE {N_AGENTS}V{N_AGENTS} CROSSOVER EVAL  {label}  {_now()}  device={device}")
    print(f"  arm        {args.arm}")
    print(f"  checkpoint {ck.relative_to(ROOT)}  sha {_sha(ck)[:12]}...")
    print(f"  seeds      {seeds[0]}..{seeds[-1]} (n={len(seeds)})")
    print(f"  gate       delta_A>0 & LCB95>0; delta_B symmetric")
    print(f"  bootstrap  n={N_BOOT}, alpha={ALPHA}, rng_seed={BOOTSTRAP_SEED}\n", flush=True)

    probe = R2.build_env(device, seeds[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    if int(obs_space.spaces["grid"].shape[0]) != N_AGENTS:
        raise SystemExit(f"FAIL-CLOSED: env agent dim != {N_AGENTS}")
    probe.close()

    if args.arm == "fully_shared":
        from rl.custom_ppo.inference_policy import CustomPPOInferencePolicy
        from rl.suite_fully_shared_distill import load_fully_shared

        model, payload = load_fully_shared(str(ck), obs_space, act_space, device=device)
        cfg = dict(payload.get("cfg") or {})
        cfg["fixed_latent_strategy"] = True
        policy = CustomPPOInferencePolicy(model, device=device, cfg=cfg)
        needs_entity = getattr(model, "entity_encoder", None) is not None
    else:
        from rl import ladder_rung1 as L1

        rung = {"share_encoder": 1, "share_backbone": 2, "share_macro": 3}[args.arm]
        if rung == 1:
            model, branch_cfg, _ = L1.load_rung1(str(ck), obs_space, act_space, device=device)
        else:
            model, branch_cfg, _ = L1.load_rung(
                rung, str(ck), obs_space, act_space, device=device,
            )
        policy = L1.make_dispatch_policy(model, branch_cfg, device=device)
        needs_entity = bool(getattr(model, "entity_repair_enabled", False))

    if not bool(getattr(model, "uses_latent_strategy", False)) and not hasattr(model, "branch"):
        # Fully shared sets uses_latent_strategy; Rung1Model also sets it.
        pass
    if int(getattr(model, "latent_k", 0) or 0) != 2:
        raise SystemExit(f"REFUSING: latent_k must be 2; got {getattr(model, 'latent_k', None)}")
    if not needs_entity:
        raise SystemExit(f"REFUSING: suite {N_AGENTS}v{N_AGENTS} students are entity-repair; refusing non-entity eval")

    def run_cell(z: int, pole: str, seed: int) -> dict:
        env = R2.build_env(device, seed)
        core = env.core
        try:
            policy.fixed_latent_strategy = True
            policy.fixed_latent_strategy_id = int(z)
            if hasattr(policy, "reset_strategy"):
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
            obs = augment_obs_with_entities(obs, core, side="blue")
            assert_live_opponent_batch(
                core, genomes, allowed_keys=(key,),
                context=f"{label} z{z}@Pole{pole} seed {seed}",
            )
            resolved = core._bt_resolved_profile_tensors()
            got = resolved.get("min_alive_for_defender")
            got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
            if got_val != N_AGENTS:
                raise SystemExit(
                    f"FAIL-CLOSED: pole {pole} min_alive_for_defender={got_val}, expected {N_AGENTS}"
                )
            terminal = None
            for _ in range(R2.MAX_STEPS):
                action, _ = policy.predict(obs, deterministic=True)
                env.step_async(action)
                obs, _r, done, info = env.step_wait()
                obs["global_state"] = env.state()
                obs = augment_obs_with_entities(obs, core, side="blue")
                if bool(np.asarray(done).any()):
                    i0 = info[0] if isinstance(info, (list, tuple)) else info
                    res = (i0 or {}).get("episode_result") or {}
                    terminal = (
                        int(res.get("blue_score", 0)),
                        int(res.get("red_score", 0)),
                    )
                    break
            if terminal is None:
                terminal = (int(core.blue_score[0]), int(core.red_score[0]))
            blue, red = terminal
            return {"blue": blue, "red": red, "win": int(blue > red), "margin": blue - red}
        finally:
            env.close()

    if args.dry_run:
        # One forced-z smoke step to prove entity+predict path.
        env = R2.build_env(device, 99_991_004)
        try:
            core = env.core
            install_keyed_opponent_overlays(core, genomes_by_pole["A"])
            env.env_method("set_phase", phase_from_tag("OP6"))
            env.env_method("set_next_opponent", "SCRIPTED", "OP6")
            obs = env.reset()
            obs["global_state"] = env.state()
            obs = augment_obs_with_entities(obs, core, side="blue")
            policy.fixed_latent_strategy = True
            policy.fixed_latent_strategy_id = 0
            if hasattr(policy, "reset_strategy"):
                policy.reset_strategy()
            action, _ = policy.predict(obs, deterministic=True)
            print(f"  dry-run predict OK  action_shape={np.asarray(action).shape}")
        finally:
            env.close()
        for pole in ("A", "B"):
            env = R2.build_env(device, 99_990_000 + N_AGENTS)
            try:
                core = env.core
                install_keyed_opponent_overlays(core, genomes_by_pole[pole])
                key = BASE_KEY[pole]
                env.env_method("set_phase", phase_from_tag(key))
                env.env_method("set_next_opponent", "SCRIPTED", key)
                env.reset()
                resolved = core._bt_resolved_profile_tensors()
                got = resolved.get("min_alive_for_defender")
                got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
                print(f"  dry-run pole {pole}: min_alive={got_val} "
                      f"{'OK' if got_val == N_AGENTS else 'MISMATCH'}")
                if got_val != N_AGENTS:
                    raise SystemExit("FAIL-CLOSED: dry-run pole mismatch")
            finally:
                env.close()
        print("\n  --dry-run PASS: nothing written.")
        return 0

    LOG.parent.mkdir(parents=True, exist_ok=True)
    cells = [(z, pole, seed) for z in (0, 1) for pole in ("A", "B") for seed in seeds]
    rows = []
    bar = tqdm_iter(cells, desc=f"{label}", unit="ep")
    for z, pole, seed in bar:
        set_postfix(bar, f"z{z}@Pole{pole} seed={seed}")
        rows.append({"z": z, "pole": pole, "seed": seed, **run_cell(z, pole, seed)})
        if seed == seeds[-1]:
            wr = float(np.mean([r["win"] for r in rows if r["z"] == z and r["pole"] == pole]))
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

    tie_or_reversal = [
        k for k, d in (("delta_A", delta_a), ("delta_B", delta_b)) if d["mean"] <= 0.0
    ]
    if tie_or_reversal:
        PREAUDIT_FLAG.write_text(json.dumps({
            "record": f"{label} crossover EVAL integrity audit REQUIRED",
            "status": "FLAGGED", "utc": _now(),
            "triggered_by": tie_or_reversal,
            "point_estimates": {"delta_A": delta_a["mean"], "delta_B": delta_b["mean"]},
            "raw_rows": str(ROWS_CSV.relative_to(ROOT)),
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
        "record": f"{label} crossover EVAL",
        "status": "FROZEN_RESULT",
        "one_shot": True,
        "utc": _now(),
        "arm": "EXPLORATORY",
        "confirmatory": False,
        "implements": f"{SPEC_PATH.name}#EVALUATION",
        "suite_arm": args.arm,
        "team_size": N_AGENTS,
        "device": device,
        "checkpoint": str(ck.relative_to(ROOT)),
        "checkpoint_sha256": _sha(ck),
        "seeds": {"block": [seeds[0], seeds[-1]], "n": len(seeds),
                  "shared_across_z_and_poles": True},
        "poles": {
            p: {
                "base": BASE_KEY[p],
                "overlay": dict(
                    (pole_A_genome(N_AGENTS) if p == "A" else pole_B_genome(N_AGENTS)).overlay or {}
                ),
            }
            for p in ("A", "B")
        },
        "PRIMARY_GATE": {"delta_A": delta_a, "delta_B": delta_b, "passes": gate_passes},
        "bootstrap": {
            "procedure": "paired percentile bootstrap over evaluation seeds",
            "samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOTSTRAP_SEED,
        },
        "no_model_selection_occurred": True,
        "total_episodes": len(rows),
        "claim_boundary": "EXPLORATORY n=64; not confirmatory; not PAPER-FAITHFUL",
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {OUT}")
    return 0 if gate_passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
