"""2v2 Fully Shared+z forced-z crossover (suite task 1), one-shot sealed eval.

Implements SUITE_FULLY_SHARED_Z_2V2_CROSSOVER_EVAL_SPEC.json. Loads the frozen
distilled student (ONE actor parameter set, concat-conditioned on z), forces z,
and runs the four cells on the spec's fresh registered seed block:

    delta_A = V(z0, A) - V(z1, A)
    delta_B = V(z1, B) - V(z0, B)

PASS iff both means > 0 AND both LCB95 > 0. The 2v2 environment, poles and episode
loop are exactly those of experiments/eval_ladder_rung1_matched.py (the ruler behind
the other 2v2 rows); only the seed block differs (see the spec's SEEDS section).

Finished episodes are appended to an append-only PARTIAL.jsonl and skipped on relaunch.
Sealing goes through experiments.run_state.seal; status is never hand-written.

Run:
  python experiments/eval_suite_fully_shared_z_2v2.py --dry-run --device cuda
  python experiments/eval_suite_fully_shared_z_2v2.py --device cuda
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.eval_hog_psp_v3 import _mean_ci  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / "SUITE_FULLY_SHARED_Z_2V2_CROSSOVER_EVAL_SPEC.json"
LABEL = "SUITE_FULLY_SHARED_Z_2V2"
EXP_ID = "SUITE_FULLY_SHARED_Z_2V2_CROSSOVER"
OUT = SD / f"{LABEL}_CROSSOVER_EVAL_RESULT.json"
ROWS_CSV = SD / f"{LABEL.lower()}_crossover_eval_rows.csv"
PREAUDIT_FLAG = SD / f"{LABEL}_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json"
PARTIAL = SD / "suite_sharing" / "2v2" / "fully_shared_z" / "crossover_eval_PARTIAL.jsonl"
N_BOOT, ALPHA, BOOTSTRAP_SEED = 20_000, 0.05, 7
N_AGENTS = 2


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_partial(ck_sha: str) -> dict:
    """Finished (z, pole, seed) -> row. The first line pins the checkpoint sha."""
    done: dict = {}
    if not PARTIAL.is_file():
        return done
    lines = [ln for ln in PARTIAL.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return done
    head = json.loads(lines[0])
    if head.get("header") is not True or head.get("checkpoint_sha256") != ck_sha:
        raise SystemExit("REFUSING: PARTIAL header missing or checkpoint sha differs; "
                         "a partial from another checkpoint must never be resumed")
    for ln in lines[1:]:
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue                    # torn final line from a kill mid-write; re-run that episode
        done[(int(r["z"]), str(r["pole"]), int(r["seed"]))] = r
    return done


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"REFUSING: {SPEC_PATH.name} not frozen: {spec.get('status')!r}")
    arm = spec["ARM"]
    lo, hi = (int(x) for x in spec["SEEDS"]["block"].split(".."))
    seeds = list(range(lo, hi + 1))
    if len(seeds) != int(spec["SEEDS"]["n"]):
        raise SystemExit("REFUSING: spec seed block length != spec n")

    ck = ROOT / arm["checkpoint"]
    if not ck.is_file():
        raise SystemExit(f"REFUSING: checkpoint missing: {ck}")
    ck_sha = _sha(ck)
    if ck_sha != arm["sha256"]:
        raise SystemExit("REFUSING: checkpoint sha mismatch vs spec pin")

    from experiments import seed_registry as sr
    ok, msg = sr.check_block(lo, hi, "sealed_confirmatory", experiment_id=EXP_ID)
    if not ok:
        raise SystemExit(f"REFUSING (Rule 9): {msg}")
    if not args.dry_run and (OUT.is_file() or PREAUDIT_FLAG.is_file()):
        raise SystemExit("REFUSING: an output for this label already exists; one-shot")
    if args.device != "cuda":
        raise SystemExit("REFUSING: the spec fixes cuda")

    import torch
    from experiments.opponent_spec import (
        assert_live_opponent_batch, install_keyed_opponent_overlays, pole_A_genome,
    )
    import experiments.phase0_collect_scorer_data as P0
    import experiments.r2_learned_crossover as R2
    from experiments.tqdm_loop import set_postfix, tqdm_iter
    from rl.curriculum import phase_from_tag
    from rl.custom_ppo.inference_policy import CustomPPOInferencePolicy
    from rl.suite_fully_shared_distill import assert_fully_shared_structure, load_fully_shared

    if not torch.cuda.is_available():
        raise SystemExit("REFUSING: cuda unavailable")
    device = args.device
    R2.AGENTS = N_AGENTS

    print(f"SUITE 2V2 FULLY SHARED+Z CROSSOVER EVAL  {LABEL}  {_now()}  device={device}")
    print(f"  checkpoint {ck.relative_to(ROOT)}  sha {ck_sha[:12]}... VERIFIED")
    print(f"  seeds      {seeds[0]}..{seeds[-1]} (n={len(seeds)}, sealed_confirmatory, registry OK)")
    print(f"  gate       delta_A>0 & LCB95>0; delta_B>0 & LCB95>0")
    print(f"  bootstrap  n={N_BOOT}, alpha={ALPHA}, rng_seed={BOOTSTRAP_SEED}\n", flush=True)

    probe = R2.build_env(device, seeds[0])
    obs_space, act_space = probe.observation_space, probe.action_space
    if int(obs_space.spaces["grid"].shape[0]) != N_AGENTS:
        raise SystemExit("FAIL-CLOSED: env agent dim != 2")
    probe.close()

    model, payload = load_fully_shared(str(ck), obs_space, act_space, device=device)
    assert_fully_shared_structure(model)
    if int(getattr(model, "latent_k", 0) or 0) != 2:
        raise SystemExit(f"REFUSING: latent_k must be 2; got {getattr(model, 'latent_k', None)}")
    if getattr(model, "entity_encoder", None) is not None or bool(getattr(model, "entity_repair_enabled", False)):
        raise SystemExit("REFUSING: 2v2 student carries an entity encoder; this runner feeds plain 2v2 obs")
    cfg = dict(payload.get("cfg") or {})
    cfg["fixed_latent_strategy"] = True
    policy = CustomPPOInferencePolicy(model, device=device, cfg=cfg)

    def prepare(env, z: int, pole: str, seed: int):
        core = env.core
        policy.fixed_latent_strategy = True
        policy.fixed_latent_strategy_id = int(z)
        if hasattr(policy, "reset_strategy"):
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
                                   context=f"{LABEL} z{z}@Pole{pole} seed {seed}")
        return core, obs

    def run_cell(z: int, pole: str, seed: int) -> dict:
        env = R2.build_env(device, seed)
        try:
            core, obs = prepare(env, z, pole, seed)
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
        # (1) forcing z must change the emitted actions on a real rollout (bind to the live object).
        acts = {}
        for z in (0, 1):
            env = R2.build_env(device, 99_990_002)
            try:
                _core, obs = prepare(env, z, "A", 99_990_002)
                seq = []
                for _ in range(40):
                    a, _ = policy.predict(obs, deterministic=True)
                    seq.append(np.asarray(a).copy())
                    env.step_async(a)
                    obs, _r, done, _i = env.step_wait()
                    obs["global_state"] = env.state()
                    if bool(np.asarray(done).any()):
                        break
                acts[z] = seq
            finally:
                env.close()
        n = min(len(acts[0]), len(acts[1]))
        differ = sum(int(not np.array_equal(acts[0][i], acts[1][i])) for i in range(n))
        print(f"  dry-run z-forcing: actions differ on {differ}/{n} of the first steps")
        if differ == 0:
            raise SystemExit("FAIL-CLOSED: forcing z0 vs z1 produced identical actions; z is not reaching the policy")
        # (2) both poles install and resolve.
        for pole in ("A", "B"):
            env = R2.build_env(device, 99_990_000 + N_AGENTS)
            try:
                prepare(env, 0, pole, 99_990_000 + N_AGENTS)
                print(f"  dry-run pole {pole}: overlay installed, opponent batch live OK")
            finally:
                env.close()
        # (3) the audit's bootstrap must reproduce the evaluator's CI, or the seal would fail after hours.
        import experiments.run_state as rs
        v = np.random.default_rng(3).integers(-1, 2, size=len(seeds)).astype(np.float64)
        ev, au = _mean_ci(v), rs._bootstrap(v, N_BOOT, ALPHA, BOOTSTRAP_SEED)
        worst = max(abs(ev[k] - au[k]) for k in ("mean", "lcb95", "ucb95"))
        print(f"  dry-run audit-vs-evaluator bootstrap: max |diff| = {worst:.2e}")
        if worst > 1e-6:
            raise SystemExit("FAIL-CLOSED: run_state audit bootstrap does not reproduce _mean_ci")
        print("\n  --dry-run PASS: nothing written.")
        return 0

    import experiments.run_state as rs
    PARTIAL.parent.mkdir(parents=True, exist_ok=True)
    done = _load_partial(ck_sha)
    if not PARTIAL.is_file() or PARTIAL.stat().st_size == 0:
        with PARTIAL.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"header": True, "checkpoint_sha256": ck_sha, "utc": _now(),
                                 "spec": SPEC_PATH.name}) + "\n")
    state = rs.RunState(SD, LABEL)
    state.begin(checkpoint=str(ck), seed_base=seeds[0], n_seeds=len(seeds), team_size=N_AGENTS,
                resumed_from_partial=len(done))
    if done:
        print(f"  resuming: {len(done)} episodes already in {PARTIAL.name}", flush=True)

    cells = [(z, pole, seed) for z in (0, 1) for pole in ("A", "B") for seed in seeds]
    bar = tqdm_iter(cells, desc=LABEL, unit="ep")
    for z, pole, seed in bar:
        set_postfix(bar, f"z{z}@Pole{pole} seed={seed}")
        if (z, pole, seed) in done:
            continue
        row = {"z": z, "pole": pole, "seed": seed, **run_cell(z, pole, seed)}
        with PARTIAL.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        done[(z, pole, seed)] = row
        if seed == seeds[-1]:
            wr = float(np.mean([r["win"] for k, r in done.items() if k[0] == z and k[1] == pole]))
            print(f"  z{z} on Pole {pole}: win rate {wr:.4f}", flush=True)

    rows = [done[(z, pole, seed)] for z in (0, 1) for pole in ("A", "B") for seed in seeds]
    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["z", "pole", "seed", "blue", "red", "win", "margin"])
        w.writeheader()
        w.writerows(rows)

    def wins(z, pole):
        by = {r["seed"]: r["win"] for r in rows if r["z"] == z and r["pole"] == pole}
        return np.array([by[s] for s in seeds], dtype=np.float64)

    delta_a = _mean_ci(wins(0, "A") - wins(1, "A"))
    delta_b = _mean_ci(wins(1, "B") - wins(0, "B"))
    for d in (delta_a, delta_b):
        d["passes"] = bool(d["mean"] > 0 and d["lcb95"] > 0)
    gate_passes = bool(delta_a["passes"] and delta_b["passes"])
    cell_rates = {f"z{z}_pole{p}": float(wins(z, p).mean()) for z in (0, 1) for p in ("A", "B")}

    print("")
    for k, v in cell_rates.items():
        print(f"    {k:10s} win rate {v:.4f}")
    print(f"    delta_A {delta_a['mean']:+.4f} [{delta_a['lcb95']:+.4f}, {delta_a['ucb95']:+.4f}] "
          f"{'PASS' if delta_a['passes'] else 'FAIL'}")
    print(f"    delta_B {delta_b['mean']:+.4f} [{delta_b['lcb95']:+.4f}, {delta_b['ucb95']:+.4f}] "
          f"{'PASS' if delta_b['passes'] else 'FAIL'}")
    print(f"\n  GATE: {'PASS' if gate_passes else 'FAIL'}")

    tie_or_reversal = [k for k, d in (("delta_A", delta_a), ("delta_B", delta_b)) if d["mean"] <= 0.0]
    if tie_or_reversal:
        PREAUDIT_FLAG.write_text(json.dumps({
            "record": f"{LABEL} crossover EVAL integrity audit REQUIRED", "status": "FLAGGED", "utc": _now(),
            "triggered_by": tie_or_reversal,
            "point_estimates": {"delta_A": delta_a["mean"], "delta_B": delta_b["mean"]},
            "raw_rows": str(ROWS_CSV.relative_to(ROOT)),
        }, indent=2), encoding="utf-8")
        print(f"  TIE/REVERSAL on {tie_or_reversal} -- row-level audit required before interpretation.")

    claims = [
        rs.Claim(name="delta_A", recorded={k: delta_a[k] for k in ("mean", "lcb95", "ucb95")},
                 minuend={"z": 0, "pole": "A"}, subtrahend={"z": 1, "pole": "A"}, value_field="win"),
        rs.Claim(name="delta_B", recorded={k: delta_b[k] for k in ("mean", "lcb95", "ucb95")},
                 minuend={"z": 1, "pole": "B"}, subtrahend={"z": 0, "pole": "B"}, value_field="win"),
    ]
    plan = rs.AuditPlan(
        rows_csv=ROWS_CSV, expected_rows=len(rows), expected_seeds=seeds,
        group_by=("z", "pole"), seed_field="seed",
        int_fields=("z", "seed", "blue", "red", "margin"), binary_fields=("win",), derived={},
        checkpoints={"pi_phi": (ck, ck_sha)}, spec_path=SPEC_PATH, claims=claims,
        n_boot=N_BOOT, alpha=ALPHA, rng_seed=BOOTSTRAP_SEED,
        seed_class="sealed_confirmatory", experiment_id=EXP_ID,
    )
    # status is owned by seal(); do not set it here.
    result = {
        "record": f"{LABEL} crossover EVAL", "one_shot": True, "utc": _now(),
        "implements": SPEC_PATH.name, "team_size": N_AGENTS, "device": device,
        "architecture": "fully_shared_pi_phi_a_given_o_z_forced_z_no_router",
        "checkpoint_sha256": ck_sha,
        "seeds": {"block": [seeds[0], seeds[-1]], "n": len(seeds), "shared_across_z_and_poles": True,
                  "seed_class": "sealed_confirmatory", "registry_experiment_id": EXP_ID,
                  "matched_to_ladder_rows": False},
        "cell_win_rates": cell_rates,
        "PRIMARY_GATE": {"delta_A": delta_a, "delta_B": delta_b, "passes": gate_passes},
        "bootstrap": {"procedure": "paired percentile bootstrap over evaluation seeds",
                      "samples": N_BOOT, "alpha": ALPHA, "rng_seed": BOOTSTRAP_SEED},
        "no_model_selection_occurred": True, "total_episodes": len(rows),
    }
    rs.seal(out_path=OUT, payload=result, plan=plan, state=state, strict=False)
    sealed = json.loads(OUT.read_text(encoding="utf-8"))
    sr.set_status(EXP_ID, "SPENT", note=f"sealed {sealed.get('status')}; gate_passes={gate_passes}")
    print(f"\n  -> {OUT} ({sealed.get('status')})")
    return 0 if gate_passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
