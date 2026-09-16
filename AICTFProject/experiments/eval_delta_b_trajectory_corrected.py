r"""delta_B across the CORRECTED pi_B3's training trajectory, against a FROZEN pi_A3.

Governed by artifacts/strategic_demand/sppo/DELTA_B_TRAJECTORY_CORRECTED_SPEC.json
(frozen before any episode). Supersedes eval_delta_b_trajectory.py, whose
checkpoints trained against canonical Pole B instead of the certified candidate
(PI_B3_TRAIN_EVAL_POLE_MISMATCH_INVALIDATION.json). Answers ONE question:

    Did B ever become good on its own pole during CORRECT training, even
    temporarily -- and if so, did it improve, peak-then-decline, or stay flat?

    delta_B(t) = V(pi_B3_corrected^(t), B) - V(pi_A3_final, B)

for t in {0 (warm-start), 100k, 200k, ..., 900k, final(~1.001M)}, all on Pole B,
all on ONE shared seed block, with pi_A3 evaluated once as the shared comparator.

Every episode ATTESTS its live Pole B against the governing certification
before running (Rule 14) -- the exact class of check that was missing when the
contaminated checkpoints were produced.

    python -m experiments.eval_delta_b_trajectory_corrected [--dry-run]
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
LABEL = "DELTA_B_TRAJECTORY_CORRECTED"
SPEC = SD / f"{LABEL}_SPEC.json"
EXPERIMENT_ID = "DELTA_B_TRAJECTORY_CORRECTED"

SEED_LO, N_SEEDS = 18_400_001, 64
N_BOOT, ALPHA, BOOTSTRAP_SEED = 20_000, 0.05, 7

SCALE = ROOT / "artifacts" / "scale_4v4_specialists"
CORR = SCALE / "pi_B_specialist_4v4_b3_entity_repair_corrected" / "ckpts"
POLE_B_GENOME = SD / "pole_b2_candidates" / "B3-3_lockdef10_2v1.json"

COMPARATOR = ("pi_A3_final",
              SCALE / "pi_A_specialist_4v4_b3_entity_repair" / "ckpts"
              / "final_pi_A_specialist_4v4_b3_entity_repair.zip")

#: (label, t_in_millions_or_None, path). t=None marks the warm-start anchor,
#: excluded from the slope fit (different architecture -- no entity_encoder).
TRAJECTORY = [
    ("t0_warmstart", None, SCALE / "pi_B_specialist_4v4_b3" / "ckpts" / "final_pi_B_specialist_4v4_b3.zip"),
    ("t100k", 0.1, CORR / "ckpt_pi_B_specialist_4v4_b3_entity_repair_corrected_100000.zip"),
    ("t200k", 0.2, CORR / "ckpt_pi_B_specialist_4v4_b3_entity_repair_corrected_200000.zip"),
    ("t300k", 0.3, CORR / "ckpt_pi_B_specialist_4v4_b3_entity_repair_corrected_300000.zip"),
    ("t400k", 0.4, CORR / "ckpt_pi_B_specialist_4v4_b3_entity_repair_corrected_400000.zip"),
    ("t500k", 0.5, CORR / "ckpt_pi_B_specialist_4v4_b3_entity_repair_corrected_500000.zip"),
    ("t600k", 0.6, CORR / "ckpt_pi_B_specialist_4v4_b3_entity_repair_corrected_600000.zip"),
    ("t700k", 0.7, CORR / "ckpt_pi_B_specialist_4v4_b3_entity_repair_corrected_700000.zip"),
    ("t800k", 0.8, CORR / "ckpt_pi_B_specialist_4v4_b3_entity_repair_corrected_800000.zip"),
    ("t900k", 0.9, CORR / "ckpt_pi_B_specialist_4v4_b3_entity_repair_corrected_900000.zip"),
    ("tfinal", 1.001472, CORR / "final_pi_B_specialist_4v4_b3_entity_repair_corrected.zip"),
]

SEALED_PI_B3_CORRECTED_SHA_PREFIX = "021342c84bbe"
SEALED_PI_A3_SHA_PREFIX = "94dde69d091a"
SEALED_DELTA_B_FINAL = -0.15625
REPLICATION_OK_RANGE = (-0.40, 0.05)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _pole_b_genome():
    from experiments.opponent_spec import _with_full_team_defender_gate
    from experiments.sds_genome import SDSGenome
    return _with_full_team_defender_gate(
        SDSGenome.from_dict(json.loads(POLE_B_GENOME.read_text(encoding="utf-8"))), 4)


def _build_pole_b_env(device, seed, genome):
    import experiments.r2_learned_crossover as R2
    from experiments.opponent_spec import assert_live_opponent_batch, install_keyed_opponent_overlays
    from rl.curriculum import phase_from_tag
    R2.AGENTS = 4
    genomes = {"OP7": genome}
    env = R2.build_env(device, seed)
    core = env.core
    core._bt_profile_override = None
    core._sds_opening_hold_steps = 0
    install_keyed_opponent_overlays(core, genomes)
    env.env_method("set_phase", phase_from_tag("OP7"))
    env.env_method("set_next_opponent", "SCRIPTED", "OP7")
    obs = env.reset()
    obs["global_state"] = env.state()
    assert_live_opponent_batch(core, genomes, allowed_keys=("OP7",),
                               context=f"{LABEL} PoleB seed {seed}")
    return env, core, obs


def _attest_pole_b(cert_path, genome, core, seed, device):
    """Rule 14: this evaluator proves its own Pole B against the governing
    certification before running a single episode -- the exact check absent
    when the contaminated pi_B3 was trained and first evaluated."""
    from experiments.pole_attestation import assert_resolved_matches_certification, attest_live_pole
    att = assert_resolved_matches_certification("B", 4, cert_path, genome, is_smoke=False)
    env, core, _obs = _build_pole_b_env(device, seed, genome)
    try:
        att = attest_live_pole(core, "B", 4, att, context=f"{LABEL} preflight")
    finally:
        env.close()
    return att


def _contracts(paths: dict, policies: dict, cert_path, genome, device) -> dict:
    from gpu_env._core._entity_obs import build_entity_tensors, flatten_for_policy
    res = {}
    shas = {k: _sha(p) for k, p in paths.items()}

    res["K1_all_checkpoints_distinct"] = bool(len(set(shas.values())) == len(shas))
    res["K2_final_matches_corrected_crossover_pi_B3"] = shas["tfinal"].startswith(
        SEALED_PI_B3_CORRECTED_SHA_PREFIX)
    res["K3_comparator_matches_corrected_crossover_pi_A3"] = shas[COMPARATOR[0]].startswith(
        SEALED_PI_A3_SHA_PREFIX)

    enc_ok = all(policies[lbl].model.entity_encoder is not None
                 for lbl, t, _ in TRAJECTORY if lbl != "t0_warmstart")
    t0_none = policies["t0_warmstart"].model.entity_encoder is None
    res["K5_entity_encoders_as_expected"] = bool(enc_ok and t0_none)

    att = _attest_pole_b(cert_path, genome, None, SEED_LO, device)
    res["K4_rule14_live_pole_attestation_passed"] = bool(att.get("live_attestation_passed", False))

    env, core, _obs = _build_pole_b_env(device, SEED_LO, genome)
    try:
        d = build_entity_tensors(core, "blue")
        flat = flatten_for_policy(d)
        with torch.no_grad():
            g_early = policies["t100k"].model.entity_encoder(*flat)
            g_late = policies["tfinal"].model.entity_encoder(*flat)
        res["K6_checkpoints_are_genuinely_different_states"] = bool(not torch.equal(g_early, g_late))
    finally:
        env.close()
    return res, att


def _require_registered_block() -> None:
    import experiments.seed_registry as R
    doc = R.load()
    b = next((x for x in doc["blocks"] if x["experiment_id"] == EXPERIMENT_ID), None)
    if b is None:
        raise SystemExit(
            f"FAIL-CLOSED (Rule 9): no seed block registered for experiment_id "
            f"{EXPERIMENT_ID!r}. Reserve it with seed_registry.allocate before spending a seed.")
    if b["lo"] != SEED_LO or b["hi"] != SEED_LO + N_SEEDS - 1:
        raise SystemExit(
            f"FAIL-CLOSED (Rule 9): registered block {b['lo']}..{b['hi']} does not match this "
            f"run's {SEED_LO}..{SEED_LO + N_SEEDS - 1}.")


def run_cell(inference, label, seeds, device, genome, w, fh) -> dict:
    from gpu_env._core._entity_obs import augment_obs_with_entities
    from experiments.tqdm_loop import set_postfix, tqdm_iter
    import experiments.r2_learned_crossover as R2

    wins = {}
    bar = tqdm_iter(seeds, desc=f"{LABEL} {label}", unit="ep")
    for seed in bar:
        set_postfix(bar, f"seed={seed}")
        env, core, obs = _build_pole_b_env(device, seed, genome)
        try:
            inference.reset_strategy()
            obs = augment_obs_with_entities(obs, core, side="blue")
            terminal = None
            for _ in range(R2.MAX_STEPS):
                action, _ = inference.predict(obs, deterministic=True)
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
            w.writerow({"cell": label, "seed": seed, "blue": blue, "red": red,
                        "win": wins[seed], "margin": blue - red})
            fh.flush()
        finally:
            env.close()
    return wins


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from experiments.run_lock import RunLock
    from experiments.pole_attestation import governing_certification
    from rl.custom_ppo import load_custom_ppo_policy

    if not SPEC.is_file():
        raise SystemExit(f"REFUSING: frozen spec missing: {SPEC}")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: spec not frozen: {spec.get('status')!r}")

    out_path = SD / f"{LABEL}_RESULT.json"
    rows_csv = SD / f"{LABEL.lower()}_rows.csv"
    if not args.dry_run:
        _require_registered_block()
        if out_path.is_file() or rows_csv.is_file():
            raise SystemExit(f"REFUSING: output for {LABEL} already exists; one-shot")

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    seeds = list(range(SEED_LO, SEED_LO + N_SEEDS))
    genome = _pole_b_genome()
    _verdict, cert_path = governing_certification(4)

    paths = {COMPARATOR[0]: COMPARATOR[1]}
    for lbl, _t, p in TRAJECTORY:
        paths[lbl] = p
    for lbl, p in paths.items():
        if not p.is_file():
            raise SystemExit(f"REFUSING: checkpoint missing for {lbl}: {p}")

    print(f"{LABEL}  {_now()}  device={device}")
    print(f"  spec        {SPEC.name} [{spec.get('status')}]")
    print(f"  question    {spec.get('THE_QUESTION')}")
    print(f"  pole B      {genome.genome_id!r} overlay={dict(genome.overlay or {})}  "
          f"(certification: {cert_path.name})")
    print(f"  seeds       {seeds[0]}..{seeds[-1]} (n={len(seeds)}), SHARED across all cells")
    print(f"  comparator  {COMPARATOR[0]} (FROZEN) sha {_sha(COMPARATOR[1])[:12]}...")
    for lbl, t, p in TRAJECTORY:
        print(f"  {lbl:14s} t={'warm-start' if t is None else f'{t:.3f}M':>10s}  sha {_sha(p)[:12]}...")

    env, _c, _o = _build_pole_b_env(device, seeds[0], genome)
    obs_space, act_space = env.observation_space, env.action_space
    env.close()
    policies = {lbl: load_custom_ppo_policy(str(p), obs_space, act_space, device=device)
                for lbl, p in paths.items()}
    for pol in policies.values():
        pol.model.eval()

    print("\n  known-answer contracts (incl. Rule 14 live pole attestation) ...", flush=True)
    contracts, attestation = _contracts(paths, policies, cert_path, genome, device)
    for k, v in contracts.items():
        print(f"    {'OK  ' if v else 'FAIL'} {k}")
    failed = [k for k, v in contracts.items() if not v]
    if failed:
        raise SystemExit(f"FAIL-CLOSED: contracts failed {failed}; refusing to spend an episode.")

    if args.dry_run:
        print("\n  --dry-run: spec frozen, 12 checkpoints present and distinct, Pole B attested "
              "against the governing certification, all contracts pass. NO episodes run, NOTHING written.")
        print(f"  NOTE: seed block {SEED_LO}..{SEED_LO + N_SEEDS - 1} must be reserved under "
              f"experiment_id {EXPERIMENT_ID!r} before the real run (enforced at launch).")
        return 0

    order = [COMPARATOR[0]] + [lbl for lbl, _t, _p in TRAJECTORY]
    results: dict[str, dict[int, int]] = {}
    with RunLock(SD / f"{LABEL}.run.lock", run_id=LABEL):
        with rows_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["cell", "seed", "blue", "red", "win", "margin"])
            w.writeheader()
            for lbl in order:
                results[lbl] = run_cell(policies[lbl], lbl, seeds, device, genome, w, fh)
                wr = float(np.mean([results[lbl][s] for s in seeds]))
                print(f"\n  {lbl}: win rate {wr:.4f}", flush=True)
                out_path.write_text(json.dumps(
                    {"record": f"{LABEL} (partial)", "status": "RUNNING", "utc": _now(),
                     "win_rates": {k: float(np.mean([v[s] for s in seeds]))
                                   for k, v in results.items()}}, indent=2), encoding="utf-8")

    # ---- seed-level paired bootstrap, one shared resample matrix ---------------
    wA = np.array([results[COMPARATOR[0]][s] for s in seeds], dtype=np.float64)
    d_by_label = {lbl: np.array([results[lbl][s] for s in seeds], dtype=np.float64) - wA
                  for lbl, _t, _p in TRAJECTORY}

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, len(seeds), size=(N_BOOT, len(seeds)))

    def est(v):
        boot = v[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
        return {"mean": float(v.mean()), "lcb95": float(lo), "ucb95": float(hi)}

    deltas = {lbl: est(d_by_label[lbl]) for lbl, _t, _p in TRAJECTORY}

    fit_labels = [(lbl, t) for lbl, t, _p in TRAJECTORY if t is not None]
    tv = np.array([t for _lbl, t in fit_labels], dtype=np.float64)
    tc = tv - tv.mean()
    Y = np.stack([d_by_label[lbl][idx].mean(axis=1) for lbl, _t in fit_labels], axis=1)
    beta_boot = (Y @ tc) / float(tc @ tc)
    y_point = np.array([d_by_label[lbl].mean() for lbl, _t in fit_labels])
    beta_point = float((y_point - y_point.mean()) @ tc / (tc @ tc))
    b_lo, b_hi = np.percentile(beta_boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    slope = {"mean": beta_point, "lcb95": float(b_lo), "ucb95": float(b_hi),
             "units": "delta_B per 1,000,000 training steps"}

    endpoint = est(d_by_label["tfinal"] - d_by_label["t100k"])

    # peak detection among INTERIOR post-warm-start points (excludes tfinal itself
    # as its own peak, and excludes t0 which is architecturally different)
    interior = [(lbl, t) for lbl, t in fit_labels if lbl != "tfinal"]
    peak_lbl, _peak_t = max(interior, key=lambda kv: d_by_label[kv[0]].mean())
    peak_vs_final = est(d_by_label[peak_lbl] - d_by_label["tfinal"])

    d_final = deltas["tfinal"]["mean"]
    steps_to_zero = (float("inf") if beta_point <= 0 else max(0.0, -d_final) / beta_point)
    replicates = REPLICATION_OK_RANGE[0] <= d_final <= REPLICATION_OK_RANGE[1]

    print("\n  " + "=" * 72)
    for lbl, t, _p in TRAJECTORY:
        e = deltas[lbl]
        tag = "warm-start" if t is None else f"{t:.3f}M"
        marker = "  <-- PEAK (interior)" if lbl == peak_lbl else ""
        print(f"    delta_B({tag:>10s})  {e['mean']:+.4f}  [{e['lcb95']:+.4f}, {e['ucb95']:+.4f}]{marker}")
    print(f"    slope             {slope['mean']:+.4f}  [{slope['lcb95']:+.4f}, {slope['ucb95']:+.4f}]  per 1M steps")
    print(f"    endpoint final-100k {endpoint['mean']:+.4f}  [{endpoint['lcb95']:+.4f}, {endpoint['ucb95']:+.4f}]")
    print(f"    peak({peak_lbl})-final {peak_vs_final['mean']:+.4f}  "
          f"[{peak_vs_final['lcb95']:+.4f}, {peak_vs_final['ucb95']:+.4f}]")
    print(f"    steps_to_zero     {steps_to_zero:.2f}M (optimistic linear; inf = never)")
    print(f"    replication       delta_B(final)={d_final:+.4f} vs corrected-crossover "
          f"{SEALED_DELTA_B_FINAL:+.4f} -> {'COMPATIBLE' if replicates else 'INCOMPATIBLE -- STOP'}")
    print("  " + "=" * 72)

    out_path.write_text(json.dumps({
        "record": f"{LABEL} delta_B across CORRECTED pi_B3 training trajectory",
        "status": "COMPLETE_DIAGNOSTIC", "utc": _now(), "device": device,
        "arm": "DIAGNOSTIC", "confirmatory": False,
        "implements": f"{SPEC.name}#ESTIMANDS",
        "question_answered": spec.get("THE_QUESTION"),
        "seeds": {"block": [seeds[0], seeds[-1]], "n": len(seeds), "shared_across_cells": True},
        "checkpoints": {lbl: _sha(p) for lbl, p in paths.items()},
        "contracts": contracts,
        "rule14_attestation": attestation,
        "win_rates": {k: float(np.mean([v[s] for s in seeds])) for k, v in results.items()},
        "delta_B_by_t": {lbl: {**deltas[lbl], "t_millions": t} for lbl, t, _p in TRAJECTORY},
        "slope": slope,
        "endpoint_contrast_final_minus_100k": endpoint,
        "peak_label": peak_lbl,
        "peak_minus_final_contrast": peak_vs_final,
        "steps_to_zero_millions_optimistic_linear": (None if steps_to_zero == float("inf")
                                                     else steps_to_zero),
        "replication_check": {"delta_B_final_fresh": d_final, "corrected_crossover_delta_B": SEALED_DELTA_B_FINAL,
                              "ok_range": list(REPLICATION_OK_RANGE), "compatible": bool(replicates)},
        "bootstrap": {"procedure": "paired percentile bootstrap over seeds; slope REFIT on every "
                                   "resample; single shared resample matrix across all estimands",
                      "samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOTSTRAP_SEED},
        "total_episodes": len(seeds) * len(order),
        "NOT_A_CLAIM": spec.get("WHAT_THIS_EXPERIMENT_MAY_NOT_CLAIM"),
    }, indent=2), encoding="utf-8")
    print(f"\n  -> {out_path}\n  -> {rows_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
