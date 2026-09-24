r"""Closed-loop Pole-A geometry ablation.

Governed by artifacts/strategic_demand/sppo/POLE_A_GEOMETRY_ABLATION_SPEC.json
(frozen before any episode). Answers ONE question:

    Is the entity pathway load-bearing AT INFERENCE for the recovered
    Pole-A specialization?

It does NOT answer whether geometry caused the specialization to emerge during
training -- that is the scaffold hypothesis and needs the no-entity +1M-step
continuation control.

Four cells, ALL on the same seed block, Pole A only:

    pi_A3 FULL   pi_A3 ZERO   pi_B3 FULL   pi_B3 ZERO

    delta_A_full = V(A3,FULL) - V(B3,FULL)
    delta_A_zero = V(A3,ZERO) - V(B3,ZERO)
    C_A          = delta_A_full - delta_A_zero      <- primary
    L_A          = V(A3,FULL) - V(A3,ZERO)
    L_B          = V(B3,FULL) - V(B3,ZERO)

All bootstrapped as SEED-LEVEL PAIRED contrasts on one shared set of bootstrap
resamples, never by differencing independently-bootstrapped win rates.

    python -m experiments.eval_pole_a_geometry_ablation [--dry-run]
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
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
LABEL = "POLE_A_GEOMETRY_ABLATION"
SPEC = SD / f"{LABEL}_SPEC.json"

SEED_LO, N_SEEDS = 18_000_001, 64
N_BOOT, ALPHA, BOOTSTRAP_SEED = 20_000, 0.05, 7

CKPTS = {
    "pi_A3": ROOT / "artifacts/scale_4v4_specialists/pi_A_specialist_4v4_b3_entity_repair/ckpts/final_pi_A_specialist_4v4_b3_entity_repair.zip",
    "pi_B3": ROOT / "artifacts/scale_4v4_specialists/pi_B_specialist_4v4_b3_entity_repair/ckpts/final_pi_B_specialist_4v4_b3_entity_repair.zip",
}
ARMS = ("FULL", "ZERO")
CELLS = [(p, a) for p in ("pi_A3", "pi_B3") for a in ARMS]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _apply_arm(obs: dict, arm: str) -> dict:
    """ZERO deletes the entity FEATURE tensors exactly (bias-free encoder -> g == 0).
    Validity masks are left untouched. Returns a new dict; never mutates `obs`."""
    if arm == "FULL":
        return obs
    out = dict(obs)
    out["teammates"] = np.zeros_like(obs["teammates"])
    out["enemies"] = np.zeros_like(obs["enemies"])
    return out


def _build_pole_a_env(device, seed):
    import experiments.r2_learned_crossover as R2
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
    )
    from rl.curriculum import phase_from_tag
    R2.AGENTS = 4
    genomes = {"OP6": pole_A_genome(4)}
    env = R2.build_env(device, seed)
    core = env.core
    core._bt_profile_override = None
    core._sds_opening_hold_steps = 0
    install_keyed_opponent_overlays(core, genomes)
    env.env_method("set_phase", phase_from_tag("OP6"))
    env.env_method("set_next_opponent", "SCRIPTED", "OP6")
    obs = env.reset()
    obs["global_state"] = env.state()
    assert_live_opponent_batch(core, genomes, allowed_keys=("OP6",),
                               context=f"{LABEL} PoleA seed {seed}")
    resolved = core._bt_resolved_profile_tensors()
    got = resolved.get("min_alive_for_defender")
    got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
    if got_val != 4:                                        # contract 5
        raise SystemExit(f"FAIL-CLOSED: live Pole A min_alive_for_defender={got_val}, expected 4")
    return env, core, obs


def _contracts(policies, device) -> dict:
    """Rule 12: all six contracts, executed, before a single episode is spent."""
    from gpu_env._core._entity_obs import augment_obs_with_entities, build_entity_tensors, flatten_for_policy
    res = {}

    # 4. checkpoints genuinely distinct, both entity-enabled
    sha_a, sha_b = _sha(CKPTS["pi_A3"]), _sha(CKPTS["pi_B3"])
    res["checkpoints_are_distinct"] = bool(sha_a != sha_b)
    res["both_have_entity_encoder"] = all(
        policies[n].model.entity_encoder is not None for n in CKPTS)

    env, core, obs = _build_pole_a_env(device, SEED_LO)
    try:
        res["live_pole_A_min_alive_is_4"] = True            # _build_pole_a_env already fail-closed
        obs_full = augment_obs_with_entities(obs, core, side="blue")
        obs_zero = _apply_arm(obs_full, "ZERO")

        # 3. intervention isolation: base observation bit-identical across arms
        base_identical = all(
            np.array_equal(np.asarray(obs_full[k]), np.asarray(obs_zero[k]))
            for k in ("grid", "vec", "agent_mask", "mask"))
        entities_differ = not np.array_equal(np.asarray(obs_full["teammates"]),
                                             np.asarray(obs_zero["teammates"]))
        res["intervention_isolated_base_obs_bit_identical"] = bool(base_identical)
        res["intervention_actually_changed_entities"] = bool(entities_differ)

        d_full = build_entity_tensors(core, "blue")
        d_zero = {**d_full,
                  "teammates": torch.zeros_like(d_full["teammates"]),
                  "enemies": torch.zeros_like(d_full["enemies"])}
        ok_zero, ok_nonzero, ok_det = [], [], []
        for name in CKPTS:
            model = policies[name].model
            with torch.no_grad():
                g0 = model.entity_encoder(*flatten_for_policy(d_zero))
                gf = model.entity_encoder(*flatten_for_policy(d_full))
                ok_zero.append(bool(torch.equal(g0, torch.zeros_like(g0))))      # 1
                ok_nonzero.append(bool(gf.abs().sum().item() > 0.0))             # 2
                l1 = model.policy_logits(
                    {k: torch.as_tensor(np.asarray(obs_full[k]), dtype=torch.float32,
                                        device=model.entity_encoder.proj.weight.device)
                     for k in ("grid", "vec", "agent_mask", "mask")},
                    teammates=d_full["teammates"], teammates_valid=d_full["teammates_valid"],
                    enemies=d_full["enemies"], enemies_valid=d_full["enemies_valid"])
                l2 = model.policy_logits(
                    {k: torch.as_tensor(np.asarray(obs_full[k]), dtype=torch.float32,
                                        device=model.entity_encoder.proj.weight.device)
                     for k in ("grid", "vec", "agent_mask", "mask")},
                    teammates=d_full["teammates"], teammates_valid=d_full["teammates_valid"],
                    enemies=d_full["enemies"], enemies_valid=d_full["enemies_valid"])
                ok_det.append(bool(torch.equal(l1, l2)))                          # 6
        res["ZERO_residual_is_exactly_zero_trained_weights"] = all(ok_zero)
        res["FULL_residual_is_nonzero_trained_weights"] = all(ok_nonzero)
        res["logits_deterministic"] = all(ok_det)
    finally:
        env.close()
    return res


def run_cell(inference, policy_name: str, arm: str, seeds, device, rows_writer, rows_fh) -> dict:
    from gpu_env._core._entity_obs import augment_obs_with_entities
    from experiments.tqdm_loop import set_postfix, tqdm_iter
    import experiments.r2_learned_crossover as R2

    wins = {}
    bar = tqdm_iter(seeds, desc=f"{LABEL} {policy_name}/{arm}", unit="ep")
    for seed in bar:
        set_postfix(bar, f"seed={seed}")
        env, core, obs = _build_pole_a_env(device, seed)
        try:
            inference.reset_strategy()
            obs = augment_obs_with_entities(obs, core, side="blue")
            terminal = None
            for _ in range(R2.MAX_STEPS):
                action, _ = inference.predict(_apply_arm(obs, arm), deterministic=True)
                env.step_async(action)
                obs, _r, done, info = env.step_wait()
                obs["global_state"] = env.state()
                obs = augment_obs_with_entities(obs, core, side="blue")
                if bool(np.asarray(done).any()):
                    i0 = info[0] if isinstance(info, (list, tuple)) else info
                    r = (i0 or {}).get("episode_result") or {}
                    terminal = (int(r.get("blue_score", 0)), int(r.get("red_score", 0)))
                    break
            if terminal is None:
                terminal = (int(core.blue_score[0]), int(core.red_score[0]))
            blue, red = terminal
            wins[seed] = int(blue > red)
            rows_writer.writerow({"policy": policy_name, "arm": arm, "seed": seed,
                                  "blue": blue, "red": red, "win": wins[seed],
                                  "margin": blue - red})
            rows_fh.flush()                                  # Rule 2
        finally:
            env.close()
    return wins


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from experiments.run_lock import RunLock
    from rl.custom_ppo import load_custom_ppo_policy

    if not SPEC.is_file():
        raise SystemExit(f"REFUSING: frozen spec missing: {SPEC}")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: spec not frozen: {spec.get('status')!r}")

    out_path = SD / f"{LABEL}_RESULT.json"
    rows_csv = SD / f"{LABEL.lower()}_rows.csv"
    if not args.dry_run and (out_path.is_file() or rows_csv.is_file()):
        raise SystemExit(f"REFUSING: output for {LABEL} already exists; one-shot")

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    seeds = list(range(SEED_LO, SEED_LO + N_SEEDS))

    print(f"{LABEL}  {_now()}  device={device}")
    print(f"  spec       {SPEC.name} [{spec.get('status')}]")
    print(f"  question   {spec.get('THE_QUESTION_THIS_ANSWERS')}")
    print(f"  seeds      {seeds[0]}..{seeds[-1]} (n={len(seeds)}), SHARED across all 4 cells")
    print(f"  cells      {CELLS}")
    print(f"  bootstrap  n={N_BOOT}, alpha={ALPHA}, rng_seed={BOOTSTRAP_SEED}, unit=seed, PAIRED")
    for n, p in CKPTS.items():
        if not p.is_file():
            raise SystemExit(f"REFUSING: checkpoint missing: {p}")
        print(f"  {n:6s}     sha {_sha(p)[:12]}...")

    env, _core, _obs = _build_pole_a_env(device, seeds[0])
    obs_space, act_space = env.observation_space, env.action_space
    env.close()
    policies = {n: load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
                for n, p in CKPTS.items()}
    for pol in policies.values():
        pol.model.eval()

    print("\n  known-answer contracts ...", flush=True)
    contracts = _contracts(policies, device)
    for k, v in contracts.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    failed = [k for k, v in contracts.items() if not v]
    if failed:
        raise SystemExit(f"FAIL-CLOSED: contracts failed {failed}; refusing to spend an episode.")

    if args.dry_run:
        print("\n  --dry-run: spec frozen, checkpoints present, pole resolves at N=4, "
              "all contracts pass. NO episodes run, NOTHING written.")
        return 0

    results: dict[str, dict[int, int]] = {}
    lock = SD / f"{LABEL}.run.lock"
    with RunLock(lock, run_id=LABEL):
        with rows_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["policy", "arm", "seed", "blue", "red", "win", "margin"])
            w.writeheader()
            for policy_name, arm in CELLS:
                wins = run_cell(policies[policy_name], policy_name, arm, seeds, device, w, fh)
                results[f"{policy_name}/{arm}"] = wins
                wr = float(np.mean([wins[s] for s in seeds]))
                print(f"\n  {policy_name}/{arm}: win rate {wr:.4f}", flush=True)
                out_path.write_text(json.dumps(
                    {"record": f"{LABEL} (partial)", "status": "RUNNING", "utc": _now(),
                     "cells_done": sorted(results), "win_rates": {
                         k: float(np.mean([v[s] for s in seeds])) for k, v in results.items()}},
                    indent=2), encoding="utf-8")

    # ---- seed-level paired bootstrap, ONE shared resample matrix ---------------
    def vec(key):
        return np.array([results[key][s] for s in seeds], dtype=np.float64)

    aF, aZ = vec("pi_A3/FULL"), vec("pi_A3/ZERO")
    bF, bZ = vec("pi_B3/FULL"), vec("pi_B3/ZERO")
    d_full, d_zero = aF - bF, aZ - bZ
    c_seed = d_full - d_zero
    l_a, l_b = aF - aZ, bF - bZ

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(seeds), size=(N_BOOT, len(seeds)))

    def est(v, name):
        boot = v[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        return {"name": name, "mean": float(v.mean()), "lcb95": float(lo), "ucb95": float(hi)}

    delta_full = est(d_full, "delta_A_full")
    delta_zero = est(d_zero, "delta_A_zero")
    C_A = est(c_seed, "C_A")
    L_A = est(l_a, "L_A")
    L_B = est(l_b, "L_B")
    corr = float(np.corrcoef(d_full, d_zero)[0, 1]) if d_full.std() > 0 and d_zero.std() > 0 else float("nan")

    print("\n  " + "=" * 68)
    for e in (delta_full, delta_zero, C_A, L_A, L_B):
        print(f"    {e['name']:13s} {e['mean']:+.4f}  [{e['lcb95']:+.4f}, {e['ucb95']:+.4f}]")
    print(f"    corr(d_full, d_zero) = {corr:+.4f}   (pairing efficiency)")
    print("  " + "=" * 68)

    out_path.write_text(json.dumps({
        "record": f"{LABEL} closed-loop Pole-A geometry ablation",
        "status": "COMPLETE_DIAGNOSTIC", "utc": _now(), "device": device,
        "arm": "DIAGNOSTIC", "confirmatory": False,
        "implements": f"{SPEC.name}#ESTIMANDS",
        "question_answered": spec.get("THE_QUESTION_THIS_ANSWERS"),
        "question_NOT_answered": spec.get("THE_QUESTION_THIS_DOES_NOT_ANSWER"),
        "seeds": {"block": [seeds[0], seeds[-1]], "n": len(seeds), "shared_across_cells": True},
        "checkpoints": {n: _sha(p) for n, p in CKPTS.items()},
        "contracts": contracts,
        "win_rates": {k: float(np.mean([v[s] for s in seeds])) for k, v in results.items()},
        "estimates": {e["name"]: e for e in (delta_full, delta_zero, C_A, L_A, L_B)},
        "pairing_correlation_d_full_d_zero": corr,
        "bootstrap": {"procedure": "paired percentile bootstrap over evaluation seeds, single shared "
                                   "resample matrix across all estimands",
                      "samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOTSTRAP_SEED},
        "total_episodes": len(seeds) * len(CELLS),
        "NOT_A_CLAIM": spec.get("WHAT_THIS_EXPERIMENT_MAY_NOT_CLAIM"),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {out_path}\n  -> {rows_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
