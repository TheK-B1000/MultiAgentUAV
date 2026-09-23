"""CLOSEST_DEFENDS_SCREEN_6V6_V1_SPEC.json.

DESCRIPTIVE / EXPLORATORY screen (n=64, no confirmatory label, no LCB95 gate) on the
FROZEN, EXISTING 6v6 learned specialists (no training, no parameter change): does
forcing DEFEND onto the one ACTIVE agent closest to its own flag at t=0
(state-dependent, fixed for the episode) create the desired A/B specialization
separation? Mirrors experiments/run_closest_defends_screen_4v4.py (SEALED
41575677, pattern_match=TRUE) with k_D=1 instead of N/2=2, matching the sealed
scripted 6v6 composition finding (5A_1D beats 3A_3D).

This is a DIFFERENT, disjoint experiment from the already-sealed
DEFENDER_INJECTION_CAUSAL_BRIDGE (which used seed-rotated, not closest-to-home,
defender selection and found ONE_DEFENDER_HARM_ONLY concentrated on Pole B). See
the frozen spec's IMPORTANT_PRIOR_RESULT_TO_WEIGH for why selection rule is
expected to matter.

  contracts   Mechanism correctness (closest-to-home selection, determinism,
              state-dependence, non-degeneracy, checkpoint pins, seed freshness,
              read-only fence). Spends no seed.
  run         384 full episodes (64 seeds x 2 poles x 3 arms), resumable,
              append-only partial file, deterministic policy actions, no grad.
  analyze     Refuses unless all 384 cells are present. Seals via run_state.seal
              (Rule 7/8/9: never hand-write a status).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.probe_learned_composition as P  # noqa: E402  (mechanism, unchanged, pinned)
import experiments.run_learned_composition_probe as L  # noqa: E402  (checkpoint pins, digests)
from experiments.run_routed_composition_outcome import ALPHA, N_BOOT, RNG_SEED, _bootstrap  # noqa: E402
from experiments.tqdm_loop import tqdm_iter  # noqa: E402
from macro_actions import MacroAction  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
STEM = "CLOSEST_DEFENDS_SCREEN_6V6"
EXP_ID = STEM
SPEC_PATH = SD / f"{STEM}_V1_SPEC.json"
CONTRACT_PATH = SD / f"{STEM}_CONTRACT_RESULT.json"
RESULT_PATH = SD / f"{STEM}_RESULT.json"
ROWS_PATH = SD / f"{STEM}_ROWS.csv"
PARTIAL_PATH = SD / f"{STEM}_PARTIAL.jsonl"

DEVICE = "cuda"
N = P.N_AGENTS
SEED_BASE, SEED_N, SEED_CLASS = 22_000_001, 64, "exploratory"
SEEDS = list(range(SEED_BASE, SEED_BASE + SEED_N))
POLES = P.POLES
ARMS = ("pi_A", "A_closest", "pi_B")
ARM_POLICY = {"pi_A": "pi_A", "A_closest": "pi_A", "pi_B": "pi_B"}
K_DEFEND = 1

CONTRASTS = {
    "Delta_A_closest": (("A_closest", "A"), ("pi_B", "A")),
    "Delta_B_closest": (("pi_B", "B"), ("A_closest", "B")),
    "I_A": (("A_closest", "A"), ("pi_A", "A")),
    "I_B": (("A_closest", "B"), ("pi_A", "B")),
    "native_Delta_A": (("pi_A", "A"), ("pi_B", "A")),
    "native_Delta_B": (("pi_B", "B"), ("pi_A", "B")),
}
GUARD_SENTENCE = ("A_closest is pi_A with an externally imposed, state-dependent single-defender assignment "
                  "(closest-to-home ACTIVE agent at t=0), not a learned assignment. DESCRIPTIVE screen, not a "
                  "confirmatory claim: no LCB95 gate, no terminal pass/fail label. Answers whether the pair-selection "
                  "RULE (closest-to-home vs the already-sealed seed-rotated DEFENDER_INJECTION_CAUSAL_BRIDGE) "
                  "changes the outcome, not whether PPO can learn to select or execute it.")


def _load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


# ============================================================================== CLOSEST_DEFENDS (k=1)

def closest_defend_id(core, k: int = K_DEFEND) -> tuple[int, ...]:
    """State-dependent, computed ONCE at the state right after env.reset() (before
    any action). Among ACTIVE agents (alive, not tagged, not carrying), returns
    the k closest to own_flag_home by Euclidean distance, ties broken by lower
    agent index (np.argsort is stable)."""
    alive = core.blue_alive[0].detach().cpu().numpy()
    tagged = core.blue_tagged[0].detach().cpu().numpy()
    carrying = core.blue_carrying[0].detach().cpu().numpy()
    active = alive & ~tagged & ~carrying
    idx = np.where(active)[0]
    if len(idx) < k:
        raise SystemExit(f"FAIL-CLOSED: only {len(idx)} ACTIVE agents at t=0, need >= {k}")
    x = core.blue_x[0].detach().cpu().numpy()
    y = core.blue_y[0].detach().cpu().numpy()
    home = core.blue_flag_home[0].detach().cpu().numpy()
    d = np.hypot(x[idx] - home[0], y[idx] - home[1])
    order = idx[np.argsort(d, kind="stable")]
    return tuple(int(i) for i in order[:k])


def run_closest_episode(policy, pole: str, seed: int, genomes: dict[str, dict], device: str,
                        max_ticks: int | None = None) -> dict[str, Any]:
    """Mirrors experiments.probe_learned_composition.run_learned_episode_with_forced_defender
    action for action; the ONLY difference is that the defender id is computed HERE
    from the post-reset state instead of taken as a seed-derived parameter."""
    import experiments.r2_learned_crossover as R2
    from gpu_env._core._entity_obs import augment_obs_with_entities
    env = P.build_probe_env(device, seed, n_macros=5)
    core = env.core
    try:
        policy.reset_strategy()
        obs = P.setup_episode(env, pole, genomes)
        (defender_id,) = closest_defend_id(core)
        P.install_forced_defend_target(core, defender_id)
        terminal, steps = None, 0
        horizon = R2.MAX_STEPS if max_ticks is None else min(int(max_ticks), R2.MAX_STEPS)
        for _ in range(horizon):
            action, _ = policy.predict(obs, deterministic=True)
            env.step_async(action)
            obs, _r, done, info = env.step_wait()
            steps += 1
            obs["global_state"] = env.state()
            obs = augment_obs_with_entities(obs, core, side="blue")
            if bool(np.asarray(done).any()):
                i0 = info[0] if isinstance(info, (list, tuple)) else info
                res = (i0 or {}).get("episode_result") or {}
                terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                break
        if terminal is None:
            terminal = (int(core.blue_score[0]), int(core.red_score[0]))
        blue, red = terminal
        return {"steps": steps, "blue": blue, "red": red, "win": int(blue > red),
               "margin": blue - red, "defender_id": defender_id}
    finally:
        env.close()


def run_cell(pols: dict, genomes: dict, arm: str, pole: str, seed: int, device: str) -> dict[str, Any]:
    if arm == "A_closest":
        r = run_closest_episode(pols["pi_A"], pole, seed, genomes, device)
    else:
        r = P.run_learned_episode(pols[ARM_POLICY[arm]], pole, seed, genomes, device)
        r = {"steps": r["steps"], "blue": r["blue"], "red": r["red"], "win": r["win"],
             "margin": r["margin"], "defender_id": -1}
    return {"arm": arm, "pole": pole, "seed": seed, **r}


# ============================================================================== contracts

def contracts() -> dict:
    import torch
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    spec = _load_json(SPEC_PATH)
    add("C0_SPEC_FROZEN", spec.get("status") == "FROZEN_BEFORE_SEED_SPEND", f"spec status = {spec.get('status')!r}")

    # C1: checkpoints match the already-established pins
    pins = L._checkpoint_pins()
    got = {n: L._sha(p) for n, p in P.CHECKPOINTS.items()}
    add("C1_CHECKPOINTS_MATCH_THE_PINNED_HASHES", got == pins, f"{got} vs pinned {pins}")

    # C2: seed block reserved, exploratory, unspent, disjoint from every prior block
    from experiments import seed_registry as sr
    ok, msg = sr.check_block(SEED_BASE, SEED_BASE + SEED_N - 1, SEED_CLASS, experiment_id=EXP_ID)
    entry = next((b for b in sr.load()["blocks"] if b["experiment_id"] == EXP_ID), None)
    reg_ok = bool(ok and entry is not None and entry["status"] == "RESERVED"
                  and entry["lo"] == SEED_BASE and entry["hi"] == SEED_BASE + SEED_N - 1
                  and entry["seed_class"] == SEED_CLASS)
    add("C2_SEED_BLOCK_RESERVED_EXPLORATORY_UNSPENT", reg_ok, f"{msg}; registry status={entry and entry['status']}")

    # C3: closest_defend_id matches an independent brute-force recomputation
    class FakeCore:
        pass
    rng = np.random.default_rng(11)
    import torch as T
    n_ok = n_eligible = 0
    for _ in range(500):
        fc = FakeCore()
        alive = rng.random(N) < 0.9
        alive[rng.integers(0, N)] = True
        tagged = rng.random(N) < 0.2
        carrying = rng.random(N) < 0.2
        x, y = rng.uniform(0, 19, N), rng.uniform(0, 19, N)
        home = rng.uniform(0, 19, 2)
        fc.blue_alive = T.tensor([alive]); fc.blue_tagged = T.tensor([tagged]); fc.blue_carrying = T.tensor([carrying])
        fc.blue_x = T.tensor([x]); fc.blue_y = T.tensor([y]); fc.blue_flag_home = T.tensor([home])
        active = alive & ~tagged & ~carrying
        idx = np.where(active)[0]
        if len(idx) < 1:
            continue
        n_eligible += 1
        d = np.hypot(x[idx] - home[0], y[idx] - home[1])
        want = (int(idx[np.argsort(d, kind="stable")][0]),)
        got_id = closest_defend_id(fc)
        n_ok += int(got_id == want)
    add("C3_CLOSEST_SELECTION_MATCHES_INDEPENDENT_BRUTE_FORCE", n_eligible > 400 and n_ok == n_eligible,
        f"{n_ok}/{n_eligible} eligible synthetic states (of 500 drawn) matched an independent argsort recomputation")

    # C4: spawn non-degeneracy on the ACTUAL 64 seeds x 2 poles that will be used
    genomes = P.pole_genomes()
    gaps = []
    for seed in SEEDS:
        for pole in POLES:
            env = P.build_probe_env("cpu", seed, n_macros=5)
            core = env.core
            P.setup_episode(env, pole, genomes)
            x, y = core.blue_x[0].detach().cpu().numpy(), core.blue_y[0].detach().cpu().numpy()
            home = core.blue_flag_home[0].detach().cpu().numpy()
            d = np.sort(np.hypot(x - home[0], y - home[1]))
            gaps.append(float(d[1] - d[0]))
            env.close()
    add("C4_SPAWN_GEOMETRY_NON_DEGENERATE_ON_ALL_REAL_CELLS", min(gaps) > 1e-3,
        f"gap between 1st/2nd-closest agent across all {len(gaps)} (seed,pole) cells: min {min(gaps):.4f}, median {float(np.median(gaps)):.4f}")

    # C5: determinism + state-dependence (not a hidden fixed id)
    pols = P.load_policies(DEVICE)
    before = {n: L._param_digest(p) for n, p in pols.items()}
    with torch.no_grad():
        r1 = run_closest_episode(pols["pi_A"], "A", SEEDS[0], genomes, DEVICE)
        r2 = run_closest_episode(pols["pi_A"], "A", SEEDS[0], genomes, DEVICE)
        ids4 = {run_closest_episode(pols["pi_A"], "A", s, genomes, DEVICE)["defender_id"] for s in SEEDS[:4]}
    add("C5_A_CLOSEST_IS_DETERMINISTIC_AND_STATE_DEPENDENT",
        r1["defender_id"] == r2["defender_id"] and r1["blue"] == r2["blue"] and r1["red"] == r2["red"] and len(ids4) >= 2,
        f"rerun of (A, {SEEDS[0]}) identical defender_id/outcome: "
        f"{r1['defender_id'] == r2['defender_id'] and (r1['blue'], r1['red']) == (r2['blue'], r2['red'])}; "
        f"{len(ids4)} distinct ids across the first 4 seeds: {sorted(ids4)}")
    after = {n: L._param_digest(p) for n, p in pols.items()}
    add("C5b_POLICY_PARAMETERS_UNCHANGED_BY_CONTRACT_EPISODES", before == after, "parameter digests equal before/after")

    # C6: no seed-table side door
    src = Path(__file__).read_text(encoding="utf-8")
    banned = ["defender_id_for_" + "seed", "seed %" + " N", "seed %" + " 6"]
    hits = [b for b in banned if b in src]
    add("C6_NO_SEED_TABLE_SIDE_DOOR", not hits,
        "closest_defend_id never calls a seed-to-id helper or a seed-modulo lookup" if not hits else f"found {hits}")

    # C7: read-only fence
    src2 = Path(__file__).read_text(encoding="utf-8")
    forbidden = [".le" + "arn(", "optimizer" + ".step", ".back" + "ward(", "PP" + "O(", "zero_" + "grad"]
    fh_ = [t for t in forbidden if t in src2]
    add("C7_READ_ONLY_FENCE", not fh_, "no training/optimizer call in the source" if not fh_ else f"found {fh_}")

    # C8: pattern-match decision-rule self-test
    def pm(a, b, ib):
        return bool(a > 0 and b > 0 and ib < 0)
    cases = [((0.1, 0.1, -0.1), True), ((-0.1, 0.1, -0.1), False), ((0.1, -0.1, -0.1), False),
             ((0.1, 0.1, 0.1), False), ((0.0, 0.1, -0.1), False)]
    lab_ok = all(pm(*args) == want for args, want in cases)
    add("C8_PATTERN_MATCH_SELFTEST", lab_ok, f"{len(cases)}/{len(cases)} synthetic cases")

    decision = "CONTRACTS_PASS" if all(c["pass"] for c in checks) else "CONTRACTS_FAIL"
    result = {"record_id": f"{STEM}_CONTRACT_RESULT", "implements": SPEC_PATH.name, "utc": L._now(),
              "DECISION": decision, "spec_sha256": L._sha(SPEC_PATH), "script_sha256": L._sha(Path(__file__)), "checks": checks}
    CONTRACT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n  CLOSEST_DEFENDS_SCREEN_6V6 CONTRACTS: {decision}  ({sum(not c['pass'] for c in checks)}/{len(checks)} failed)", flush=True)
    return result


# ============================================================================== run stage

def _cells() -> list[tuple[str, str, int]]:
    return [(arm, pole, s) for s in SEEDS for pole in POLES for arm in ARMS]


def _load_partial(path: Path) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                out[tuple(o["key"])] = o["row"]
    return out


def run_shard(shard: int, n_shards: int) -> int:
    import torch
    if _load_json(CONTRACT_PATH).get("DECISION") != "CONTRACTS_PASS":
        raise SystemExit("REFUSING: contracts did not pass")
    if RESULT_PATH.exists():
        raise SystemExit(f"REFUSING: result already exists: {RESULT_PATH}")
    from experiments import seed_registry as sr
    ok, msg = sr.check_block(SEED_BASE, SEED_BASE + SEED_N - 1, SEED_CLASS, experiment_id=EXP_ID)
    entry = next((b for b in sr.load()["blocks"] if b["experiment_id"] == EXP_ID), None)
    if not (entry is not None and entry["status"] == "RESERVED"):
        raise SystemExit(f"REFUSING: seed block not reserved: {msg}")
    pins = L._checkpoint_pins()
    if {n: L._sha(p) for n, p in P.CHECKPOINTS.items()} != pins:
        raise SystemExit("REFUSING: checkpoint hashes differ from the frozen pins")

    mine = [c for i, c in enumerate(_cells()) if i % n_shards == shard]
    path = SD / f"{STEM}_SHARD{shard}OF{n_shards}_PARTIAL.jsonl"
    done = _load_partial(path)
    pending = [c for c in mine if c not in done]
    print(f"  shard {shard}/{n_shards}: {len(mine)} cells, {len(done)} already recorded, {len(pending)} to run", flush=True)
    if not pending:
        return 0
    genomes = P.pole_genomes()
    pols = P.load_policies(DEVICE)
    before = {n: L._param_digest(p) for n, p in pols.items()}
    with path.open("a", encoding="utf-8") as fh, torch.no_grad():
        for key in tqdm_iter(pending, desc=f"closest-screen-6v6 shard {shard}/{n_shards} (cuda)", total=len(pending), unit="ep"):
            arm, pole, seed = key
            row = run_cell(pols, genomes, arm, pole, seed, DEVICE)
            fh.write(json.dumps({"key": list(key), "row": row}) + "\n")
            fh.flush()
    after = {n: L._param_digest(p) for n, p in pols.items()}
    if before != after:
        raise SystemExit("ABORT: policy parameters changed during a read-only run")
    print(f"  shard {shard}/{n_shards} complete (parameter digests unchanged)", flush=True)
    return 0


# ============================================================================== analyze (seal)

def analyze() -> int:
    from experiments import run_state as rs
    from experiments import seed_registry as sr

    if _load_json(CONTRACT_PATH).get("DECISION") != "CONTRACTS_PASS":
        raise SystemExit("REFUSING: contracts did not pass")
    if RESULT_PATH.exists():
        raise SystemExit(f"REFUSING: result already exists: {RESULT_PATH}")

    merged: dict[tuple, dict] = {}
    shard_glob = f"{STEM}_SHARD*_PARTIAL.jsonl"
    for path in sorted(SD.glob(shard_glob)):
        for key, row in _load_partial(path).items():
            if key in merged and merged[key] != row:
                raise SystemExit(f"ABORT: conflicting rows for {key} across shards")
            merged[key] = row
    for key, row in _load_partial(PARTIAL_PATH).items():
        if key in merged and merged[key] != row:
            raise SystemExit(f"ABORT: conflicting rows for {key} between partial and shard files")
        merged[key] = row
    want = set(_cells())
    missing, extra = sorted(want - set(merged)), sorted(set(merged) - want)
    if missing or extra:
        raise SystemExit(f"ABORT: {len(missing)} cell(s) missing (e.g. {missing[:2]}), {len(extra)} unexpected. Run/resume the shards first.")

    rows = [merged[c] for c in _cells()]
    with ROWS_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    state = rs.RunState(SD, STEM).begin(spec=SPEC_PATH.name, n_episodes=len(rows))

    val = {f: {(r["arm"], r["pole"], int(r["seed"])): float(r[f]) for r in rows} for f in ("win", "blue", "margin")}

    def diff(f: str, plus: tuple[str, str], minus: tuple[str, str]) -> np.ndarray:
        return np.asarray([val[f][(plus[0], plus[1], s)] - val[f][(minus[0], minus[1], s)] for s in SEEDS], dtype=np.float64)

    contrasts = {f: {name: _bootstrap(diff(f, a, b)) for name, (a, b) in CONTRASTS.items()} for f in ("win", "blue", "margin")}
    cell_means = {f"{arm}_pole{pole}": {f: round(float(np.mean([val[f][(arm, pole, s)] for s in SEEDS])), 6) for f in ("win", "blue", "margin")}
                  for arm in ARMS for pole in POLES}
    win = contrasts["win"]
    pattern_match = bool(win["Delta_A_closest"]["mean"] > 0 and win["Delta_B_closest"]["mean"] > 0 and win["I_B"]["mean"] < 0)

    prior = _load_json(SD / "DEFENDER_INJECTION_CAUSAL_BRIDGE_RESULT.json")
    prior_ref = {k: prior["PER_CELL_DELTA"][k] for k in ("pi_A_B", "pi_B_B")}

    pins = L._checkpoint_pins()
    claims = [
        rs.Claim(name=name, recorded={"mean": win[name]["mean"], "lcb95": win[name]["lcb95"], "ucb95": win[name]["ucb95"]},
                minuend={"arm": a[0], "pole": a[1]}, subtrahend={"arm": b[0], "pole": b[1]}, value_field="win")
        for name, (a, b) in CONTRASTS.items()
    ]
    plan = rs.AuditPlan(rows_csv=ROWS_PATH, expected_rows=len(rows), expected_seeds=SEEDS,
                        group_by=("arm", "pole"), seed_field="seed", int_fields=("seed", "steps", "defender_id"),
                        binary_fields=("win",), derived={},
                        checkpoints={n: (p, pins[n]) for n, p in P.CHECKPOINTS.items()},
                        spec_path=SPEC_PATH, seed_class=SEED_CLASS, experiment_id=EXP_ID,
                        n_boot=N_BOOT, alpha=ALPHA, rng_seed=RNG_SEED, claims=claims)

    payload = {
        "record_id": f"{STEM}_RESULT", "implements": SPEC_PATH.name, "device": DEVICE,
        "classification": "DESCRIPTIVE / EXPLORATORY -- no confirmatory label, no LCB95 gate",
        "seed_block": {"base": SEED_BASE, "n": SEED_N, "class": SEED_CLASS},
        "WIN_RATE_CONTRASTS_point_estimates_and_bootstrap_CI": win,
        "DESCRIPTIVE_BLUE_GOALS_CONTRASTS": contrasts["blue"], "DESCRIPTIVE_MARGIN_CONTRASTS": contrasts["margin"],
        "CELL_MEANS": cell_means,
        "PRIOR_SEED_ROTATED_RESULT_FOR_COMPARISON_DEFENDER_INJECTION_CAUSAL_BRIDGE": prior_ref,
        "PATTERN_MATCH_point_estimate_only_not_a_significance_claim": {
            "rule": "Delta_A_closest_mean > 0 AND Delta_B_closest_mean > 0 AND I_B_mean < 0",
            "result": pattern_match,
        },
        "interpretation_guard": GUARD_SENTENCE,
        "claim_boundary": "Descriptive comparison only, on the frozen existing 6v6 learned specialists, fresh exploratory "
                          "seeds, k=1 closest-to-home defender injection. Not a confirmatory crossover claim; no LCB95 gate. " + GUARD_SENTENCE,
    }
    rs.seal(out_path=RESULT_PATH, payload=payload, plan=plan, state=state, strict=False)
    sealed = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    sr.set_status(EXP_ID, "SPENT", note=f"descriptive screen complete; pattern_match={pattern_match}")
    print(json.dumps({"status": sealed.get("status"), "pattern_match": pattern_match,
                      "win_rate_point_estimates": {n: {"mean": win[n]["mean"], "lcb95": win[n]["lcb95"], "ucb95": win[n]["ucb95"]} for n in CONTRASTS},
                      "prior_seed_rotated_reference": prior_ref}, indent=2))
    return 0 if sealed.get("status") == "SEALED" else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("contracts", "run", "analyze"), required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1, dest="n_shards")
    a = ap.parse_args()
    if a.stage == "contracts":
        return 0 if contracts()["DECISION"] == "CONTRACTS_PASS" else 2
    if a.stage == "run":
        if not 0 <= a.shard < a.n_shards:
            raise SystemExit(f"--shard must be in [0, {a.n_shards})")
        return run_shard(a.shard, a.n_shards)
    return analyze()


if __name__ == "__main__":
    raise SystemExit(main())
