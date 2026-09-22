"""CLOSEST_DEFENDS_SCREEN_4V4_V1_SPEC.json.

EXPLORATORY screen (n=64, DESCRIPTIVE only -- no confirmatory label, no LCB95 gate) on the FROZEN 4v4 learned
specialists (no training, no parameter change): does forcing DEFEND onto the two ACTIVE agents closest to their own
flag at the first decision tick (state-dependent, fixed for the whole episode) preserve the sealed scaffold bridge's
crossover pattern (Delta_A' > 0, Delta_B' > 0, I_B < 0), or does the specific seed-rotated pair choice matter more
than the compositional headcount (2A/2D)?

Arms on both certified poles, the same 64 fresh paired seeds:
    pi_A         native pi_A3 (identical arm to the sealed bridge)
    A_closest    pi_A3 + 2D: the two ACTIVE agents closest to own_flag_home at the state right after env.reset()
                 (before any action) have their resolved target forced to DEFEND's live-state target -- same
                 mechanism as the sealed A' (install_forced_defend_target, same pin), a different PAIR RULE
    pi_B         native corrected pi_B3
Reports Delta_A_closest, Delta_B_closest, I_A, I_B exactly as the sealed bridge does, compared descriptively
(point estimates only) against the sealed bridge's own sealed numbers.

  contracts   Pair-selection correctness, attestation, pole/checkpoint pins, parity against a known sealed bridge
              row, determinism. Spends no seed.
  run         One shard of the 384 episodes (64 seeds x 2 poles x 3 arms), resumable, append-only partial file.
  analyze     Refuses to run unless all 384 cells are present; writes a descriptive comparison, no confirmatory label.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.probe_learned_composition as P  # noqa: E402  (mechanism, unchanged, sha256 pinned)
import experiments.run_learned_composition_probe as L  # noqa: E402  (param digest)
import experiments.run_scaffolded_a_crossover_bridge_4v4 as S  # noqa: E402  (imported, never modified)
from experiments.run_routed_composition_outcome import ALPHA, N_BOOT, RNG_SEED, _bootstrap  # noqa: E402
from experiments.tqdm_loop import tqdm_iter  # noqa: E402

SD = S.SD
STEM = "CLOSEST_DEFENDS_SCREEN_4V4"
EXP_ID = STEM
SPEC_PATH = SD / f"{STEM}_V1_SPEC.json"
CONTRACT_PATH = SD / f"{STEM}_CONTRACT_RESULT.json"
RESULT_PATH = SD / f"{STEM}_RESULT.json"
ROWS_PATH = SD / f"{STEM}_ROWS.csv"
SHARD_GLOB = f"{STEM}_SHARD*_PARTIAL.jsonl"

DEVICE = S.DEVICE
N = S.N
SEED_BASE, SEED_N, SEED_CLASS = 21_300_001, 64, "exploratory"
SEEDS = list(range(SEED_BASE, SEED_BASE + SEED_N))
ARMS = ("pi_A", "A_closest", "pi_B")
POLES = S.POLES
ARM_POLICY = {"pi_A": "pi_A", "A_closest": "pi_A", "pi_B": "pi_B"}
K_DEFEND = 2

CONTRASTS = {
    "Delta_A_closest": (("A_closest", "A"), ("pi_B", "A")),
    "Delta_B_closest": (("pi_B", "B"), ("A_closest", "B")),
    "I_A": (("A_closest", "A"), ("pi_A", "A")),
    "I_B": (("A_closest", "B"), ("pi_A", "B")),
    "native_Delta_A": (("pi_A", "A"), ("pi_B", "A")),
    "native_Delta_B": (("pi_B", "B"), ("pi_A", "B")),
}
GUARD_SENTENCE = ("A_closest is pi_A with an externally imposed, state-dependent defender assignment (closest-to-home at t=0), not a "
                  "learned assignment. This is a DESCRIPTIVE screen, not a confirmatory claim: no LCB95 gate, no terminal pass/fail "
                  "label. It answers whether the pair-selection RULE matters, not whether PPO can learn to select or execute it.")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _sha(p: Path) -> str:
    return S._sha_file(Path(p))


def _cells() -> list[tuple[str, str, int]]:
    return [(arm, pole, s) for s in SEEDS for pole in POLES for arm in ARMS]


def _registered() -> tuple[bool, str]:
    from experiments import seed_registry as sr
    ok, msg = sr.check_block(SEED_BASE, SEED_BASE + SEED_N - 1, SEED_CLASS, experiment_id=EXP_ID)
    entry = next((b for b in sr.load()["blocks"] if b["experiment_id"] == EXP_ID), None)
    good = bool(ok and entry is not None and entry["status"] == "RESERVED" and entry["lo"] == SEED_BASE
                and entry["hi"] == SEED_BASE + SEED_N - 1 and entry["seed_class"] == SEED_CLASS)
    return good, f"{msg}; registry status={entry and entry['status']} class={entry and entry['seed_class']}"


def _checkpoint_paths(spec: dict) -> dict[str, Path]:
    return S._checkpoint_paths(spec)


def _sealed_inputs_ok() -> tuple[bool, str]:
    pins = _load_json(SPEC_PATH)["INTEGRITY_FROZEN"]["sealed_inputs_sha256"]
    now = {"scaffold_spec": _sha(S.SPEC_PATH), "scaffold_result": _sha(S.RESULT_PATH), "scaffold_rows": _sha(S.ROWS_PATH),
           "scaffold_runner": _sha(Path(S.__file__)), "mechanism_source": _sha(S.MECHANISM_FILE)}
    bad = [k for k in pins if pins[k] != now.get(k)]
    return (not bad, f"all {len(pins)} pinned inputs equal their pins" if not bad else f"DIFFER: {bad}")


# ============================================================================================ CLOSEST_DEFENDS

def closest_defends_pair(core, k: int = K_DEFEND) -> tuple[int, ...]:
    """State-dependent, computed ONCE at the state right after env.reset() (before any action). Among ACTIVE agents
    (alive, not tagged, not carrying -- the audit's own eligibility convention), returns the k closest to
    own_flag_home by Euclidean distance, ties broken by lower agent index (np.argsort is stable)."""
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


def run_closest_episode(setup: dict, policy, pole: str, seed: int, device: str) -> dict[str, Any]:
    """Mirrors experiments.run_scaffolded_a_crossover_bridge_4v4.run_episode action for action; the ONLY difference
    is that forced_ids is computed HERE from the post-reset state instead of looked up from a seed table."""
    from experiments.opponent_spec import assert_live_opponent_batch
    from gpu_env._core._entity_obs import augment_obs_with_entities
    R2 = setup["R2"]
    env = R2.build_env(device, seed)
    core = env.core
    try:
        policy.reset_strategy()
        gen, key = S._open_opponent(env, core, setup["genomes"], pole, "closest-defends screen")
        obs = env.reset()
        obs["global_state"] = env.state()
        obs = augment_obs_with_entities(obs, core, side="blue")
        assert_live_opponent_batch(core, gen, allowed_keys=(key,), context=f"closest screen {pole} seed {seed}")
        got = core._bt_resolved_profile_tensors().get("min_alive_for_defender")
        got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
        if got_val != N:
            raise SystemExit(f"FAIL-CLOSED: live pole {pole} resolves min_alive_for_defender={got_val}, expected {N}")
        forced_ids = closest_defends_pair(core)
        for i in forced_ids:
            P.install_forced_defend_target(core, int(i))
        terminal, steps = None, 0
        for _ in range(R2.MAX_STEPS):
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
        return {"steps": steps, "blue": blue, "red": red, "win": int(blue > red), "margin": blue - red, "pair": forced_ids}
    finally:
        env.close()


def run_cell(setup: dict, pols: dict, arm: str, pole: str, seed: int, device: str) -> dict[str, Any]:
    if arm == "A_closest":
        r = run_closest_episode(setup, pols["pi_A"], pole, seed, device)
        pair = "-".join(map(str, r.pop("pair")))
    else:
        r = S.run_episode(setup, pols[ARM_POLICY[arm]], pole, seed, (), device)
        pair = ""
    return {"arm": arm, "pole": pole, "seed": seed, "pair": pair, **r}


# ============================================================================================ shards

def _shard_path(i: int, k: int) -> Path:
    return SD / f"{STEM}_SHARD{i}OF{k}_PARTIAL.jsonl"


def _load_partial(path: Path) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                out[tuple(o["key"])] = o["row"]
    return out


def run_shard(shard: int, n_shards: int) -> int:
    if _load_json(CONTRACT_PATH).get("DECISION") != "CONTRACTS_PASS":
        raise SystemExit("REFUSING: contracts did not pass")
    if RESULT_PATH.exists():
        raise SystemExit(f"REFUSING: result already exists: {RESULT_PATH}")
    ok, msg = _registered()
    if not ok:
        raise SystemExit(f"REFUSING: seed block not reserved as frozen: {msg}")
    ok, msg = _sealed_inputs_ok()
    if not ok:
        raise SystemExit(f"REFUSING: {msg}")
    spec = _load_json(SPEC_PATH)
    paths = _checkpoint_paths(spec)
    pins = {n: spec["INTEGRITY_FROZEN"]["checkpoints"][n]["sha256"] for n in paths}
    if {n: _sha(p) for n, p in paths.items()} != pins:
        raise SystemExit("REFUSING: checkpoint hashes differ from the frozen pins")
    mine = [c for i, c in enumerate(_cells()) if i % n_shards == shard]
    path = _shard_path(shard, n_shards)
    done = _load_partial(path)
    pending = [c for c in mine if c not in done]
    print(f"  shard {shard}/{n_shards}: {len(mine)} cells, {len(done)} already recorded, {len(pending)} to run", flush=True)
    if not pending:
        return 0
    import torch
    setup = S._setup_poles()
    pols = S._load_policies(setup["R2"], paths, DEVICE)
    before = {n: L._param_digest(p) for n, p in pols.items()}
    with path.open("a", encoding="utf-8") as fh, torch.no_grad():
        for key in tqdm_iter(pending, desc=f"closest-screen shard {shard}/{n_shards} (cuda)", total=len(pending), unit="ep"):
            arm, pole, seed = key
            row = run_cell(setup, pols, arm, pole, seed, DEVICE)
            fh.write(json.dumps({"key": list(key), "row": row}) + "\n")
            fh.flush()
    if before != {n: L._param_digest(p) for n, p in pols.items()}:
        raise SystemExit("ABORT: policy parameters changed during a read-only run")
    print(f"  shard {shard}/{n_shards} complete (parameter digests unchanged)", flush=True)
    return 0


# ============================================================================================ analyze (descriptive)

def analyze() -> int:
    from experiments import seed_registry as sr
    if _load_json(CONTRACT_PATH).get("DECISION") != "CONTRACTS_PASS":
        raise SystemExit("REFUSING: contracts did not pass")
    if RESULT_PATH.exists():
        raise SystemExit(f"REFUSING: result already exists: {RESULT_PATH}")
    ok, msg = _registered()
    if not ok:
        raise SystemExit(f"REFUSING: seed block not reserved as frozen: {msg}")
    ok, msg = _sealed_inputs_ok()
    if not ok:
        raise SystemExit(f"REFUSING: {msg}")

    merged: dict[tuple, dict] = {}
    for path in sorted(SD.glob(SHARD_GLOB)):
        for key, row in _load_partial(path).items():
            if key in merged and merged[key] != row:
                raise SystemExit(f"ABORT: conflicting rows for {key} across shards")
            merged[key] = row
    want = set(_cells())
    missing, extra = sorted(want - set(merged)), sorted(set(merged) - want)
    if missing or extra:
        raise SystemExit(f"ABORT: {len(missing)} cell(s) missing (e.g. {missing[:2]}), {len(extra)} unexpected. Run/resume the shards first.")
    bad_pair = [k for k, r in merged.items() if (r["pair"] == "") != (k[0] != "A_closest")]
    if bad_pair:
        raise SystemExit(f"ABORT: {len(bad_pair)} row(s) whose pair field disagrees with its arm, e.g. {bad_pair[0]}")
    pairs_used = sorted({r["pair"] for k, r in merged.items() if k[0] == "A_closest"})
    if len(pairs_used) < 3:
        raise SystemExit(f"ABORT: A_closest used only {len(pairs_used)} distinct pair(s) across {SEED_N} seeds -- selection may not be state-dependent: {pairs_used}")

    rows = [merged[c] for c in _cells()]
    with ROWS_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    val = {f: {(r["arm"], r["pole"], int(r["seed"])): float(r[f]) for r in rows} for f in ("win", "blue", "margin")}

    def diff(f: str, plus: tuple[str, str], minus: tuple[str, str]) -> np.ndarray:
        return np.asarray([val[f][(plus[0], plus[1], s)] - val[f][(minus[0], minus[1], s)] for s in SEEDS], dtype=np.float64)

    contrasts = {f: {name: _bootstrap(diff(f, a, b)) for name, (a, b) in CONTRASTS.items()} for f in ("win", "blue", "margin")}
    cell_means = {f"{arm}_pole{pole}": {f: round(float(np.mean([val[f][(arm, pole, s)] for s in SEEDS])), 6) for f in ("win", "blue", "margin")}
                  for arm in ARMS for pole in POLES}
    win = contrasts["win"]
    sealed = _load_json(S.RESULT_PATH)
    sealed_win = sealed["PRIMARY_WIN_RATE_CONTRASTS"] | sealed["MECHANISM_DIAGNOSTICS_NOT_GATES_win_rate"]
    pattern_match = bool(win["Delta_A_closest"]["mean"] > 0 and win["Delta_B_closest"]["mean"] > 0 and win["I_B"]["mean"] < 0)
    payload = {
        "record_id": f"{STEM}_RESULT", "implements": SPEC_PATH.name, "utc": _now(), "device": DEVICE,
        "classification": "DESCRIPTIVE / EXPLORATORY -- no confirmatory label, no LCB95 gate",
        "seed_block": {"base": SEED_BASE, "n": SEED_N, "class": SEED_CLASS},
        "WIN_RATE_CONTRASTS_point_estimates_and_bootstrap_CI": win,
        "DESCRIPTIVE_BLUE_GOALS_CONTRASTS": contrasts["blue"], "DESCRIPTIVE_MARGIN_CONTRASTS": contrasts["margin"],
        "CELL_MEANS": cell_means,
        "PAIRS_SELECTED_BY_A_CLOSEST": {p: sum(1 for k, r in merged.items() if k[0] == "A_closest" and r["pair"] == p) for p in pairs_used},
        "sealed_bridge_reference_n128_seed_rotated_pair": {
            k: {kk: sealed_win[k][kk] for kk in ("mean", "lcb95", "ucb95")} for k in ("Delta_A_prime", "Delta_B_prime", "I_A", "I_B")
        },
        "PATTERN_MATCH_point_estimate_only_not_a_significance_claim": {
            "rule": "Delta_A_closest_mean > 0 AND Delta_B_closest_mean > 0 AND I_B_mean < 0 (the sealed bridge's own defining pattern)",
            "result": pattern_match,
        },
        "interpretation_guard": GUARD_SENTENCE,
        "provenance": {"spec_sha256": _sha(SPEC_PATH), "script_sha256": _sha(Path(__file__)),
                       "mechanism_source_sha256": _sha(S.MECHANISM_FILE), "rows_sha256": _sha(ROWS_PATH)},
        "claim_boundary": "Descriptive comparison only, on the frozen 4v4 learned specialists, fresh exploratory seeds. Not a confirmatory "
                          "crossover claim; no LCB95 gate is applied. " + GUARD_SENTENCE,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sr.set_status(EXP_ID, "SPENT", note=f"descriptive screen complete; pattern_match={pattern_match}")
    print(json.dumps({"pattern_match": pattern_match,
                      "win_rate_point_estimates": {n: {"mean": win[n]["mean"], "lcb95": win[n]["lcb95"], "ucb95": win[n]["ucb95"]} for n in CONTRASTS},
                      "sealed_bridge_reference": payload["sealed_bridge_reference_n128_seed_rotated_pair"]}, indent=2))
    return 0


# ============================================================================================ contracts

def contracts() -> dict:
    import torch
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    spec = _load_json(SPEC_PATH)
    add("C0_SPEC_FROZEN", spec.get("status") == "FROZEN_BEFORE_SEED_SPEND", f"spec status = {spec.get('status')!r}")
    ok, msg = _sealed_inputs_ok()
    add("C1_SEALED_INPUTS_EQUAL_THEIR_PINS", ok, msg)
    paths = _checkpoint_paths(spec)
    pins = {n: spec["INTEGRITY_FROZEN"]["checkpoints"][n]["sha256"] for n in paths}
    now = {n: _sha(p) for n, p in paths.items()}
    add("C2_CHECKPOINTS_EQUAL_THE_PINNED_HASHES", now == pins, f"{now} vs pinned {pins}")
    setup = S._setup_poles()
    add("C3_BOTH_POLES_ATTESTED_AND_POLE_B_IS_THE_CERTIFIED_B3_3_GENOME", set(setup["att"]) == set(POLES) and setup["pole_b"] is not None,
        f"attested {sorted(setup['att'])}; Pole B genome file {S.B33_GENOME.name}")
    ok, msg = _registered()
    add("C4_BLOCK_RESERVED_EXPLORATORY_UNSPENT_AND_DISJOINT", ok, msg)

    # C5: closest_defends_pair matches an independent brute-force recomputation, incl. inactive-agent exclusion and tie-breaking
    class FakeCore:
        pass
    rng = np.random.default_rng(9)
    import torch as T
    n_ok = n_eligible = 0
    for _ in range(500):
        fc = FakeCore()
        alive = rng.random(4) < 0.9
        alive[rng.integers(0, 4)] = True  # avoid the degenerate all-dead case colliding with the < k check below
        tagged = rng.random(4) < 0.2
        carrying = rng.random(4) < 0.2
        x, y = rng.uniform(0, 19, 4), rng.uniform(0, 19, 4)
        home = rng.uniform(0, 19, 2)
        fc.blue_alive = T.tensor([alive]); fc.blue_tagged = T.tensor([tagged]); fc.blue_carrying = T.tensor([carrying])
        fc.blue_x = T.tensor([x]); fc.blue_y = T.tensor([y]); fc.blue_flag_home = T.tensor([home])
        active = alive & ~tagged & ~carrying
        idx = np.where(active)[0]
        if len(idx) < 2:
            continue
        n_eligible += 1
        d = np.hypot(x[idx] - home[0], y[idx] - home[1])
        want = tuple(int(i) for i in idx[np.argsort(d, kind="stable")][:2])
        got = closest_defends_pair(fc)
        n_ok += int(got == want)
    add("C5_PAIR_SELECTION_MATCHES_INDEPENDENT_BRUTE_FORCE", n_eligible > 400 and n_ok == n_eligible,
        f"{n_ok}/{n_eligible} eligible synthetic states (of 500 drawn) matched an independent argsort recomputation exactly")

    # C6: spawn non-degeneracy on the ACTUAL 64 seeds x 2 poles that will be used
    R2 = setup["R2"]
    gaps = []
    for seed in SEEDS:
        for pole in POLES:
            env = R2.build_env("cpu", seed)
            core = env.core
            env.reset()
            x, y = core.blue_x[0].detach().cpu().numpy(), core.blue_y[0].detach().cpu().numpy()
            home = core.blue_flag_home[0].detach().cpu().numpy()
            d = np.sort(np.hypot(x - home[0], y - home[1]))
            gaps.append(float(d[2] - d[1]))
            env.close()
    add("C6_SPAWN_GEOMETRY_NON_DEGENERATE_ON_ALL_REAL_CELLS", min(gaps) > 1e-3,
        f"gap between 2nd/3rd-closest agent across all {len(gaps)} (seed,pole) cells: min {min(gaps):.4f}, median {float(np.median(gaps)):.4f}")

    # C7: harness parity against a KNOWN sealed bridge row (proves S._setup_poles/_open_opponent/run_episode plumbing is wired correctly)
    ck_path_A, ck_path_B = paths["pi_A"], paths["pi_B"]
    pols = S._load_policies(setup["R2"], {"pi_A": ck_path_A, "pi_B": ck_path_B}, DEVICE)
    before = {n: L._param_digest(p) for n, p in pols.items()}
    sealed_rows = {}
    with S.ROWS_PATH.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sealed_rows[(r["arm"], r["pole"], int(r["seed"]))] = (int(r["steps"]), int(r["blue"]), int(r["red"]))
    par = {}
    with torch.no_grad():
        for arm in ("pi_A", "pi_B"):
            for pole in POLES:
                seed = S.SEEDS[0]
                r = S.run_episode(setup, pols[arm], pole, seed, (), DEVICE)
                par[(arm, pole, seed)] = (r["steps"], r["blue"], r["red"]) == sealed_rows[(arm, pole, seed)]
    add("C7_NATIVE_ARM_HARNESS_REPRODUCES_A_KNOWN_SEALED_BRIDGE_ROW", all(par.values()), f"{sum(par.values())}/{len(par)} (arm,pole) cells at seed {S.SEEDS[0]} equal their sealed bridge row")

    # C8: A_closest determinism + state-dependence (not a hidden fixed pair)
    with torch.no_grad():
        r1 = run_closest_episode(setup, pols["pi_A"], "A", SEEDS[0], DEVICE)
        r2 = run_closest_episode(setup, pols["pi_A"], "A", SEEDS[0], DEVICE)
        pairs4 = {run_closest_episode(setup, pols["pi_A"], "A", s, DEVICE)["pair"] for s in SEEDS[:4]}
    add("C8_A_CLOSEST_IS_DETERMINISTIC_AND_STATE_DEPENDENT", r1 == r2 and len(pairs4) >= 2,
        f"rerun of (A, {SEEDS[0]}) identical: {r1 == r2}; {len(pairs4)} distinct pairs across the first 4 seeds: {sorted(pairs4)}")
    add("C8b_POLICY_PARAMETERS_UNCHANGED_BY_CONTRACT_EPISODES", before == {n: L._param_digest(p) for n, p in pols.items()}, "parameter digests equal before/after")

    # C9: no side door -- state-dependent selection, never a seed-table lookup. Fragments are split so this check's
    # own declaration (a literal string containing the banned token) does not self-match.
    src = Path(__file__).read_text(encoding="utf-8")
    banned = ["pair_for_seed" + "(seed)", "PAIRS[" + "seed"]
    hits = [b for b in banned if b in src]
    add("C9_NO_SEED_TABLE_SIDE_DOOR", not hits, "run_closest_episode never calls pair_for_seed/PAIRS[...]" if not hits else f"found {hits}")

    # C10: descriptive decision-rule self-test
    def pm(a, b, ib):
        return bool(a > 0 and b > 0 and ib < 0)
    cases = [((0.1, 0.1, -0.1), True), ((-0.1, 0.1, -0.1), False), ((0.1, -0.1, -0.1), False), ((0.1, 0.1, 0.1), False), ((0.0, 0.1, -0.1), False)]
    lab_ok = all(pm(*args) == want for args, want in cases)
    add("C10_PATTERN_MATCH_SELFTEST", lab_ok, f"{len(cases)}/{len(cases)} synthetic cases")

    # C11: read-only fence
    src2 = Path(__file__).read_text(encoding="utf-8")
    forbidden = [".le" + "arn(", "optimizer" + ".step", ".back" + "ward(", "PP" + "O(", "zero_" + "grad"]
    fh_ = [t for t in forbidden if t in src2]
    add("C11_READ_ONLY_FENCE", not fh_, "no training/optimizer call in the source" if not fh_ else f"found {fh_}")

    decision = "CONTRACTS_PASS" if all(c["pass"] for c in checks) else "CONTRACTS_FAIL"
    result = {"record_id": f"{STEM}_CONTRACT_RESULT", "implements": SPEC_PATH.name, "utc": _now(), "DECISION": decision,
              "spec_sha256": _sha(SPEC_PATH), "script_sha256": _sha(Path(__file__)), "checks": checks}
    CONTRACT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  CLOSEST_DEFENDS CONTRACTS: {decision}  ({sum(not c['pass'] for c in checks)}/{len(checks)} failed)", flush=True)
    return result


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
