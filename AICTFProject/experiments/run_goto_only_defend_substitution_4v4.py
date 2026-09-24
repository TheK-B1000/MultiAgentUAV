"""GOTO_ONLY_DEFEND_SUBSTITUTION_4V4_V1_SPEC.json.

CONFIRMATORY env-level necessity test on the FROZEN 4v4 learned specialists (no training, no parameter change):
is instantaneous DEFEND heading NECESSARY for the scaffold's crossover effect, or does a defender restricted to the existing native
(GO_TO, legal waypoint) surface -- made to follow the DEFEND path as closely as GO_TO permits -- preserve it?

Arms on both certified poles, the same 128 fresh paired seeds:
    pi_A     native pi_A3
    A_prime  pi_A3 + 2D: two agents' resolved targets forced to DEFEND's live-state target (the sealed scaffold, unchanged)
    N_prime  pi_A3 + 2 agents whose ACTIONS are replaced by a causal controller emitting only (GO_TO, legal waypoint)
    pi_B     native corrected pi_B3
The N' controller is a pure function of the agent's own recorded state snapshot and the legal-action mask; its selection is the audit's
own greedy path-oracle (Engine.path_oracle) applied to the isolated DEFEND rollout recomputed from the CURRENT state at each commit
boundary. It never calls the scaffold, never overrides a resolved target, and adds no macro.

  contracts   Structural, attestation, purity, determinism, parity and decision-tree contracts. Spends no confirmatory seed.
  run         One shard of the 1024 episodes (resumable, append-only partial file per shard). Prints COUNTS only.
  analyze     Refuses unless all 1024 cells are present and N' attestation holds; runs the frozen crossover contrasts through the formal
              audit, the in-env path check (with the A' positive control) and the heading diagnostic, then the frozen interpretation tree.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.audit_scaffold_to_native_representability_4v4 as A  # noqa: E402  (imported, never modified)
import experiments.run_learned_composition_probe as L  # noqa: E402  (parameter digest)
import experiments.run_scaffolded_a_crossover_bridge_4v4 as S  # noqa: E402  (imported, never modified)
from experiments.run_routed_composition_outcome import ALPHA, N_BOOT, RNG_SEED, _bootstrap  # noqa: E402
from experiments.tqdm_loop import tqdm_iter  # noqa: E402

SD = S.SD
STEM = "GOTO_ONLY_DEFEND_SUBSTITUTION_4V4"
EXP_ID = STEM
SPEC_PATH = SD / f"{STEM}_V1_SPEC.json"
CONTRACT_PATH = SD / f"{STEM}_CONTRACT_RESULT.json"
RESULT_PATH = SD / f"{STEM}_RESULT.json"
ROWS_PATH = SD / f"{STEM}_ROWS.csv"
CHECK_CELLS_PATH = SD / f"{STEM}_PATHCHECK_CELLS.csv"
SHARD_GLOB = f"{STEM}_SHARD*_PARTIAL.jsonl"

DEVICE = S.DEVICE
N = S.N
SEED_BASE, SEED_N, SEED_CLASS = 21_200_001, 128, "sealed_confirmatory"
SEEDS = list(range(SEED_BASE, SEED_BASE + SEED_N))
ARMS = ("pi_A", "A_prime", "N_prime", "pi_B")
POLES = S.POLES
PAIRS = S.PAIRS
NATIVE_ARMS = ("pi_A", "pi_B")
RECORDED_ARMS = ("A_prime", "N_prime")
H, STRIDE = A.HORIZON, A.STRIDE
RADIUS_TOL, COS_MIN, COVER = A.RADIUS_TOL, A.COS_MIN, A.COVER
GO_TO = A.GO_TO
IX = A.IX

# (contrast name) -> (minuend (arm, pole), subtrahend (arm, pole)); paired by seed
CONTRASTS = {
    "Delta_A_N": (("N_prime", "A"), ("pi_B", "A")),
    "Delta_B_N": (("pi_B", "B"), ("N_prime", "B")),
    "Delta_A_Aprime": (("A_prime", "A"), ("pi_B", "A")),
    "Delta_B_Aprime": (("pi_B", "B"), ("A_prime", "B")),
    "I_A_Aprime": (("A_prime", "A"), ("pi_A", "A")),
    "I_B_Aprime": (("A_prime", "B"), ("pi_A", "B")),
    "I_A_N": (("N_prime", "A"), ("pi_A", "A")),
    "I_B_N": (("N_prime", "B"), ("pi_A", "B")),
    "N_minus_Aprime_A": (("N_prime", "A"), ("A_prime", "A")),
    "N_minus_Aprime_B": (("N_prime", "B"), ("A_prime", "B")),
    "native_Delta_A": (("pi_A", "A"), ("pi_B", "A")),
    "native_Delta_B": (("pi_B", "B"), ("pi_A", "B")),
}

LABEL_CONFIRMED = "GOTO_ONLY_CROSSOVER_CONFIRMED"
LABEL_LOST_HIGH_FIDELITY = "GOTO_ONLY_CROSSOVER_LOST_WITH_HIGH_PATH_FIDELITY"
LABEL_INCONCLUSIVE_CONTROLLER = "INCONCLUSIVE_CONTROLLER"
LABEL_INCONCLUSIVE_INSTRUMENT = "INCONCLUSIVE_INSTRUMENT"
LABEL_INVALID = "AUDIT_INVALID"

STATEMENTS = {
    "confirmed_direction_manipulated": (
        "Instantaneous DIRECTION is not necessary for crossover: a defender restricted to native (GO_TO, legal waypoint) actions, which removes per-tick heading "
        "fidelity relative to the scaffold, preserved the crossover. This does not mean the DIRECTION metric was wrong; it measured a real mismatch that was not "
        "behaviorally necessary for the endpoint."),
    "confirmed_direction_not_shown": (
        "Crossover survives GO_TO-only defense, but the test did not demonstrably remove per-tick heading fidelity relative to the scaffold (the paired "
        "heading-agreement difference A' minus N' was not above zero on both poles); no statement about the necessity of instantaneous DIRECTION is made."),
    "path_check_not_passed_suffix": (
        " N' also did not meet the pre-declared in-env path criterion, so it is not claimed to reproduce the scaffold's path."),
    "lost_high_fidelity": (
        "High path fidelity under native GO_TO was insufficient to preserve crossover, strengthening evidence that behavior omitted by the GO_TO approximation, "
        "including instantaneous directional control, may be causally important. This is NOT a finding that DIRECTION is necessary: N' could still differ from A' in "
        "another unmeasured way. It authorizes the next directional/interface experiment only; it does not authorize PPO."),
    "inconclusive_controller": (
        "N' lost crossover but its in-env path did not meet the pre-declared criterion (RMSE <= 2.5 cells, coverage >= 0.90 on each pole). No claim about direction or vocabulary."),
    "inconclusive_instrument": (
        "The in-env path instrument could not certify the DEFEND-injected teacher itself (A' positive control below the pre-declared coverage), so a failed N' path "
        "check is uninformative. No claim about direction or vocabulary."),
}
GUARD_SENTENCE = ("N' is an oracle-informed GO_TO controller: it uses the DEFEND teacher law as information to choose waypoints, so a positive result says the native "
                  "vocabulary suffices for the crossover effect, NOT that PPO will find that behavior. Neither outcome authorizes PPO.")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _sha(p: Path) -> str:
    return S._sha_file(Path(p))


def pair_for_seed(seed: int) -> tuple[int, int]:
    return S.pair_for_seed(seed)


def _cells() -> list[tuple[str, str, int]]:
    return [(arm, pole, s) for s in SEEDS for pole in POLES for arm in ARMS]


def _registered() -> tuple[bool, str]:
    from experiments import seed_registry as sr
    ok, msg = sr.check_block(SEED_BASE, SEED_BASE + SEED_N - 1, SEED_CLASS, experiment_id=EXP_ID)
    entry = next((b for b in sr.load()["blocks"] if b["experiment_id"] == EXP_ID), None)
    good = bool(ok and entry is not None and entry["status"] == "RESERVED" and entry["lo"] == SEED_BASE
                and entry["hi"] == SEED_BASE + SEED_N - 1 and entry["seed_class"] == SEED_CLASS)
    return good, f"{msg}; registry status={entry and entry['status']} class={entry and entry['seed_class']}"


# ============================================================================================ frozen decision tree

def terminal_label(*, integrity_ok: bool, delta_a_lcb: float, delta_b_lcb: float, instrument_valid: bool,
                   check_pass_n: bool, direction_manipulated: bool) -> tuple[str, str]:
    """The frozen interpretation tree. integrity_ok is False for any attestation/audit/completeness failure (it overrides everything)."""
    if not integrity_ok:
        return LABEL_INVALID, "Integrity failure; no scientific claim."
    if delta_a_lcb > 0 and delta_b_lcb > 0:
        text = STATEMENTS["confirmed_direction_manipulated" if direction_manipulated else "confirmed_direction_not_shown"]
        if not check_pass_n:
            text += STATEMENTS["path_check_not_passed_suffix"]
        return LABEL_CONFIRMED, text
    if not instrument_valid:
        return LABEL_INCONCLUSIVE_INSTRUMENT, STATEMENTS["inconclusive_instrument"]
    if not check_pass_n:
        return LABEL_INCONCLUSIVE_CONTROLLER, STATEMENTS["inconclusive_controller"]
    return LABEL_LOST_HIGH_FIDELITY, STATEMENTS["lost_high_fidelity"]


# ============================================================================================ the N' controller

CONTROLLER_FIELDS = ("alive", "tagged", "carrying", "x", "y", "h", "v", "Fx", "Fy", "cl")


def controller_active(row: np.ndarray) -> bool:
    return bool(row[IX["alive"]] > 0.5 and row[IX["tagged"]] < 0.5 and row[IX["carrying"]] < 0.5)


def controller_select(eng: "A.Engine", row: np.ndarray, bits: int) -> int:
    """Pure function of the CONTROLLER_FIELDS entries of `row` (the agent's own state snapshot and the flag it defends) and the legal-action
    bits. Recomputes the isolated DEFEND rollout from the CURRENT state (flag held at its current position) and returns the legal GO_TO
    waypoint chosen by the audit's own greedy path-oracle at the first commit boundary. No RNG, no outcome, no opponent information."""
    st = np.zeros((H, len(A.STATE_FIELDS)))
    for f in ("x", "y", "h", "v", "Fx", "Fy"):
        st[:, IX[f]] = row[IX[f]]
    ref_pos, _ts, _pre = eng.rollout_ref(st)
    _pos, _tgt, _pre2, chosen = eng.path_oracle(st, int(bits), ref_pos, interruptible=False)
    macro, idx = chosen[0]
    if macro != GO_TO:
        raise RuntimeError("controller produced a non-GO_TO macro")
    return int(idx)


def run_recorded_episode(setup: dict, policy, pole: str, seed: int, eng: "A.Engine", device: str, *, keep_actions: bool = False) -> dict[str, Any]:
    """One N' episode: the sealed A' episode's loop with the forced agents' ACTIONS replaced by the controller's native actions."""
    import torch  # noqa: F401
    from experiments.opponent_spec import assert_live_opponent_batch
    from gpu_env._core._entity_obs import augment_obs_with_entities
    R2 = setup["R2"]
    env = R2.build_env(device, seed)
    core = env.core
    forced = S.pair_for_seed(seed)
    nM, nT = int(core.cfg.n_macros), int(core.cfg.n_targets)
    try:
        policy.reset_strategy()
        gen, key = S._open_opponent(env, core, setup["genomes"], pole, "goto-only substitution")
        obs = env.reset()
        obs["global_state"] = env.state()
        obs = augment_obs_with_entities(obs, core, side="blue")
        assert_live_opponent_batch(core, gen, allowed_keys=(key,), context=f"N' {pole} seed {seed}")
        got = core._bt_resolved_profile_tensors().get("min_alive_for_defender")
        got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
        if got_val != N:
            raise SystemExit(f"FAIL-CLOSED: live pole {pole} resolves min_alive_for_defender={got_val}, expected {N}")
        if "_build_targets_from_action" in core.__dict__ or getattr(core, "_forced_defend_targets", None) is not None:
            raise SystemExit("FAIL-CLOSED: a scaffold patch is present on an N' core")
        rt = {k: float(v.reshape(-1)[0]) for k, v in core.__dict__.items() if k.startswith("rt_") and hasattr(v, "numel") and v.numel() == 1}
        home = core.blue_flag_home[0].detach().cpu().numpy().astype(np.float64).tolist()
        last = {int(i): 0 for i in forced}
        att = {"n_emitted": 0, "n_native": 0, "n_boundary": 0, "n_commit_match": 0, "n_controller_calls": 0,
               "n_boundary_unverifiable_at_episode_end": 0, "scaffold_patched": False}
        ticks, bits_l, emitted = [], [], []
        terminal, steps = None, 0
        for _ in range(R2.MAX_STEPS):
            st, bt = A._snap(core, forced, nM, nT)
            ticks.append(st)
            bits_l.append(bt)
            action, _ = policy.predict(obs, deterministic=True)
            a = np.asarray(action, dtype=np.int64).reshape(-1).copy()
            boundary: dict[int, int] = {}
            for k, i in enumerate(forced):
                i, row, b = int(i), st[k], int(bt[k])
                is_boundary = bool(row[IX["cl"]] <= 0 and row[IX["alive"]] > 0.5)
                if is_boundary and controller_active(row):
                    last[i] = controller_select(eng, row, b)
                    att["n_controller_calls"] += 1
                a[2 * i], a[2 * i + 1] = GO_TO, last[i]                # the ONLY writes to the action array for a forced agent
                att["n_emitted"] += 1
                att["n_native"] += int(eng.is_legal(b, GO_TO, last[i]))
                if is_boundary:
                    boundary[i] = last[i]
                if keep_actions:
                    emitted.append((k, GO_TO, last[i]))
            env.step_async(a)
            obs, _r, done, info = env.step_wait()
            steps += 1
            ep_done = bool(np.asarray(done).any())
            if not ep_done:
                obs["global_state"] = env.state()
                obs = augment_obs_with_entities(obs, core, side="blue")
                # GPUCTFVecEnv.step_wait() calls core.reset_indices() internally the instant an episode ends, so a
                # boundary on the FINAL tick cannot be checked against post-step commit state: that read would see
                # the NEXT episode's freshly reset buffers, not this tick's effect. Such boundaries are excluded
                # from BOTH counters (never silently counted as a match); their count is kept separately.
                for i, idx in boundary.items():
                    att["n_boundary"] += 1
                    att["n_commit_match"] += int(int(core.blue_commit_macro[0, i]) == GO_TO and int(core.blue_commit_target[0, i]) == idx)
            else:
                att["n_boundary_unverifiable_at_episode_end"] += len(boundary)
                i0 = info[0] if isinstance(info, (list, tuple)) else info
                res = (i0 or {}).get("episode_result") or {}
                terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                break
        if terminal is None:
            terminal = (int(core.blue_score[0]), int(core.red_score[0]))
        att["scaffold_patched"] = bool("_build_targets_from_action" in core.__dict__ or getattr(core, "_forced_defend_targets", None) is not None)
        meta = {"pole": pole, "seed": int(seed), "forced": list(map(int, forced)), "steps": steps, "blue": terminal[0], "red": terminal[1],
                "rt": rt, "dt": float(core.dt), "home": home, "attest": att}
        out = {"meta": meta, "state": np.transpose(np.stack(ticks), (1, 0, 2)).tolist(), "legal": np.stack(bits_l).T.tolist()}
        if keep_actions:
            out["emitted"] = emitted
        return out
    finally:
        env.close()


def run_cell(setup: dict, pols: dict, eng: "A.Engine", arm: str, pole: str, seed: int, *, keep_actions: bool = False) -> tuple[dict, dict | None]:
    if arm in NATIVE_ARMS:
        r = S.run_episode(setup, pols[arm], pole, seed, (), DEVICE)
        return {"arm": arm, "pole": pole, "seed": seed, "pair": "", **r}, None
    forced = S.pair_for_seed(seed)
    if arm == "A_prime":
        cell = A.replay_cell(setup, pols["pi_A"], pole, seed, DEVICE)
    else:
        cell = run_recorded_episode(setup, pols["pi_A"], pole, seed, eng, DEVICE, keep_actions=keep_actions)
    m = cell["meta"]
    row = {"arm": arm, "pole": pole, "seed": seed, "pair": "-".join(map(str, forced)), "steps": m["steps"], "blue": m["blue"], "red": m["red"],
           "win": int(m["blue"] > m["red"]), "margin": m["blue"] - m["red"]}
    return row, cell


def _attestation_ok(meta: dict) -> tuple[bool, str]:
    a = meta.get("attest")
    if not isinstance(a, dict):
        return False, "missing attestation block"
    bad = []
    if a["n_emitted"] <= 0 or a["n_emitted"] != a["n_native"]:
        bad.append(f"emitted {a['n_emitted']} != native-surface {a['n_native']}")
    if a["n_boundary"] != a["n_commit_match"]:
        bad.append(f"boundary {a['n_boundary']} != env commit match {a['n_commit_match']}")
    if a["n_controller_calls"] <= 0:
        bad.append("controller never called")
    if a["scaffold_patched"]:
        bad.append("scaffold patch present")
    return not bad, "; ".join(bad) if bad else "ok"


# ============================================================================================ shards

def _shard_path(i: int, k: int) -> Path:
    return SD / f"{STEM}_SHARD{i}OF{k}_PARTIAL.jsonl"


def _load_partial(path: Path) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                out[tuple(o["key"])] = o
    return out


def _checkpoint_paths(spec: dict) -> dict[str, Path]:
    return S._checkpoint_paths(spec)


def _sealed_inputs_ok() -> tuple[bool, str]:
    pins = _load_json(SPEC_PATH)["INTEGRITY_FROZEN"]["sealed_inputs_sha256"]
    now = {"scaffold_spec": _sha(S.SPEC_PATH), "scaffold_result": _sha(S.RESULT_PATH), "scaffold_rows": _sha(S.ROWS_PATH),
           "scaffold_runner": _sha(Path(S.__file__)), "mechanism_source": _sha(S.MECHANISM_FILE),
           "audit_spec": _sha(A.SPEC_PATH), "audit_result": _sha(A.RESULT_PATH), "audit_runner": _sha(Path(A.__file__))}
    bad = [k for k in pins if pins[k] != now.get(k)]
    return (not bad, f"all {len(pins)} pinned inputs equal their pins" if not bad else f"DIFFER: {bad}")


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
    eng = A.Engine()
    with path.open("a", encoding="utf-8") as fh, torch.no_grad():
        for arm, pole, seed in tqdm_iter(pending, desc=f"goto-only shard {shard}/{n_shards} (cuda)", total=len(pending), unit="ep"):
            row, cell = run_cell(setup, pols, eng, arm, pole, seed)
            fh.write(json.dumps({"key": [arm, pole, seed], "row": row, "cell": cell}) + "\n")
            fh.flush()
    if before != {n: L._param_digest(p) for n, p in pols.items()}:
        raise SystemExit("ABORT: policy parameters changed during a read-only run")
    print(f"  shard {shard}/{n_shards} complete (parameter digests unchanged)", flush=True)
    return 0


# ============================================================================================ in-env path check + heading diagnostic

CHECK_FIELDS = ("arm", "pole", "seed", "n_win", "win_pass", "n_win_free", "win_free_pass", "n_t", "head_pass")


def check_cell(arm: str, pole: str, seed: int, cell: dict) -> dict[str, Any]:
    """In-env path check: the LIVE recorded defender path over 16 ticks vs the ISOLATED DEFEND reference from the window-start state,
    pass iff RMSE <= 2.5 (the audit's TRAJECTORY criterion). Heading diagnostic: on ACTIVE ticks with a non-zero DEFEND vector, does the
    REALIZED displacement point within 8 degrees of the direction to the DEFEND target (zero realized move counts as -1)."""
    eng = A._engine()
    row = {k: 0 for k in CHECK_FIELDS}
    row.update(arm=arm, pole=pole, seed=seed)
    for a in range(2):
        st = np.asarray(cell["state"][a], dtype=np.float64)
        T = len(st)
        act = (st[:, IX["alive"]] > 0.5) & (st[:, IX["tagged"]] < 0.5) & (st[:, IX["carrying"]] < 0.5)
        for t0 in range(0, T - H, STRIDE):                 # needs the record at t0+16 (16 live positions)
            if not act[t0:t0 + H].all():
                continue
            ref_pos, _ts, _pre = eng.rollout_ref(st[t0:t0 + H])
            live = st[t0 + 1:t0 + H + 1, [IX["x"], IX["y"]]]
            ok = float(np.sqrt(np.mean(np.sum((ref_pos - live) ** 2, axis=1)))) <= RADIUS_TOL
            free = bool((st[t0:t0 + H, IX["dmin"]] >= eng.reach).all())
            row["n_win"] += 1
            row["win_pass"] += int(ok)
            row["n_win_free"] += int(free)
            row["win_free_pass"] += int(free and ok)
        for t in np.flatnonzero(act[:-1]):
            s = st[t]
            tx, ty, _ = eng.defend(s[IX["x"]], s[IX["y"]], s[IX["h"]], (s[IX["Fx"]], s[IX["Fy"]]))
            g = np.array([tx - s[IX["x"]], ty - s[IX["y"]]])
            if np.hypot(*g) <= 1e-8:
                continue
            d = np.array([st[t + 1, IX["x"]] - s[IX["x"]], st[t + 1, IX["y"]] - s[IX["y"]]])
            cos = -1.0 if np.hypot(*d) <= 1e-8 else float(np.clip(np.dot(d, g) / (np.hypot(*d) * np.hypot(*g)), -1.0, 1.0))
            row["n_t"] += 1
            row["head_pass"] += int(cos >= COS_MIN)
    return row


def _check_cell_star(args: tuple) -> dict[str, Any]:
    return check_cell(*args)


def derive_check(rows: list[dict]) -> dict[str, Any]:
    """rows: per-cell dicts (CSV strings or numbers). GATING check = interaction-free windows only (no other live agent within shove
    reach for the whole 16-tick window; the audit's own pre-declared robustness definition). The full population is reported as a
    non-gating diagnostic: it contrasts an isolated reference against LIVE recorded positions, which inherit the live step's 0.5-cell
    avoid-collision shove (quantified in the audit's contract C9) -- a confound shared identically by A' and N', unrelated to vocabulary,
    that made even A' BORDERLINE at calibration (see cal.json). PI-approved 2026-09-21: gate on interaction-free, do not build a
    multi-agent-aware reference. Threshold is unchanged (RMSE <= 2.5, LCB95 >= 0.90); only the eligible window population changed,
    exactly as the audit itself excluded windows where the agent stopped being ACTIVE."""
    out: dict[str, Any] = {"path_check": {}, "path_check_status": {}, "path_check_full_population_diagnostic": {}, "heading": {}, "heading_contrast_A_minus_N": {}}
    for arm in RECORDED_ARMS:
        for pole in POLES:
            rp = [r for r in rows if r["arm"] == arm and r["pole"] == pole]
            col = lambda k: np.asarray([float(r[k]) for r in rp])  # noqa: E731
            rate, lo, hi = A.cluster_rate_ci(col("win_free_pass"), col("n_win_free"))
            out["path_check"].setdefault(arm, {})[pole] = {"rate": rate, "lcb95": lo, "ucb95": hi, "windows": int(col("n_win_free").sum()), "episodes": len(rp)}
            out["path_check_status"].setdefault(arm, {})[pole] = A.classify(lo, hi)
            fr, flo, fhi = A.cluster_rate_ci(col("win_pass"), col("n_win"))
            out["path_check_full_population_diagnostic"].setdefault(arm, {})[pole] = {"rate": fr, "lcb95": flo, "ucb95": fhi, "windows": int(col("n_win").sum())}
            hr, hlo, hhi = A.cluster_rate_ci(col("head_pass"), col("n_t"))
            out["heading"].setdefault(arm, {})[pole] = {"rate": hr, "lcb95": hlo, "ucb95": hhi, "ticks": int(col("n_t").sum())}
    for pole in POLES:
        by = {arm: {int(r["seed"]): r for r in rows if r["arm"] == arm and r["pole"] == pole} for arm in RECORDED_ARMS}
        diffs = []
        for s in sorted(set(by["A_prime"]) & set(by["N_prime"])):
            ra, rn = by["A_prime"][s], by["N_prime"][s]
            if float(ra["n_t"]) > 0 and float(rn["n_t"]) > 0:
                diffs.append(float(ra["head_pass"]) / float(ra["n_t"]) - float(rn["head_pass"]) / float(rn["n_t"]))
        out["heading_contrast_A_minus_N"][pole] = _bootstrap(np.asarray(diffs))
        out["heading_contrast_A_minus_N"][pole]["n_pairs_used"] = len(diffs)
    return out


def decision_inputs(chk: dict, delta_lcbs: tuple[float, float], integrity_ok: bool) -> dict[str, Any]:
    st = chk["path_check_status"]
    instrument_valid = all(st["A_prime"][p] == "PASS" for p in POLES)
    check_pass_n = all(st["N_prime"][p] == "PASS" for p in POLES)
    hc = chk["heading_contrast_A_minus_N"]
    direction_manipulated = all(hc[p]["lcb95"] is not None and hc[p]["lcb95"] > 0 for p in POLES)
    return {"integrity_ok": integrity_ok, "delta_a_lcb": delta_lcbs[0], "delta_b_lcb": delta_lcbs[1], "instrument_valid": instrument_valid,
            "check_pass_n": check_pass_n, "direction_manipulated": direction_manipulated}


# ============================================================================================ analyze

def analyze() -> int:
    from concurrent.futures import ProcessPoolExecutor
    from experiments import run_state as rs
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
        for key, o in _load_partial(path).items():
            if key in merged and merged[key]["row"] != o["row"]:
                raise SystemExit(f"ABORT: conflicting rows for {key} across shards")
            merged[key] = o
    want = set(_cells())
    missing, extra = sorted(want - set(merged)), sorted(set(merged) - want)
    if missing or extra:
        raise SystemExit(f"ABORT: {len(missing)} cell(s) missing (e.g. {missing[:2]}), {len(extra)} unexpected. Run/resume the shards first.")
    bad_pair = [k for k, o in merged.items() if o["row"]["pair"] != ("-".join(map(str, pair_for_seed(k[2]))) if k[0] in RECORDED_ARMS else "")]
    if bad_pair:
        raise SystemExit(f"ABORT: {len(bad_pair)} row(s) whose pair does not follow the frozen rotation, e.g. {bad_pair[0]}")
    att_fail = [(k, _attestation_ok(o["cell"]["meta"])[1]) for k, o in merged.items() if k[0] == "N_prime" and not _attestation_ok(o["cell"]["meta"])[0]]
    rows = [merged[c]["row"] for c in _cells()]
    with ROWS_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if att_fail:
        RESULT_PATH.write_text(json.dumps({"record_id": f"{STEM}_RESULT", "TERMINAL_OUTCOME": LABEL_INVALID, "status": "INVALID",
                                           "reason": "N' attestation failed", "failures": att_fail[:20], "n_failures": len(att_fail)}, indent=2), encoding="utf-8")
        print(f"AUDIT_INVALID: {len(att_fail)} N' cell(s) failed attestation, e.g. {att_fail[:2]}")
        return 2

    print("running the in-env path check on the recorded arms ...", flush=True)
    jobs = [(arm, pole, seed, merged[(arm, pole, seed)]["cell"]) for (arm, pole, seed) in _cells() if arm in RECORDED_ARMS]
    with ProcessPoolExecutor(max_workers=6) as pool:
        check_rows = list(tqdm_iter(pool.map(_check_cell_star, jobs, chunksize=2), desc="path check", total=len(jobs), unit="cell"))
    with CHECK_CELLS_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CHECK_FIELDS))
        w.writeheader()
        w.writerows(check_rows)
    with CHECK_CELLS_PATH.open(newline="", encoding="utf-8") as fh:              # derive from the FILE, as the audit did
        chk = derive_check(list(csv.DictReader(fh)))

    spec = _load_json(SPEC_PATH)
    paths = _checkpoint_paths(spec)
    pins = {n: spec["INTEGRITY_FROZEN"]["checkpoints"][n]["sha256"] for n in paths}
    state = rs.RunState(SD, EXP_ID).begin(spec=SPEC_PATH.name, n_episodes=len(rows))
    val = {f: {(r["arm"], r["pole"], int(r["seed"])): float(r[f]) for r in rows} for f in ("win", "blue", "margin")}

    def diff(f: str, plus: tuple[str, str], minus: tuple[str, str]) -> np.ndarray:
        return np.asarray([val[f][(plus[0], plus[1], s)] - val[f][(minus[0], minus[1], s)] for s in SEEDS], dtype=np.float64)

    contrasts = {f: {name: _bootstrap(diff(f, a, b)) for name, (a, b) in CONTRASTS.items()} for f in ("win", "blue", "margin")}
    cell_means = {f"{arm}_pole{pole}": {f: round(float(np.mean([val[f][(arm, pole, s)] for s in SEEDS])), 6) for f in ("win", "blue", "margin")}
                  for arm in ARMS for pole in POLES}
    win = contrasts["win"]
    dec = decision_inputs(chk, (win["Delta_A_N"]["lcb95"], win["Delta_B_N"]["lcb95"]), integrity_ok=True)
    label, statement = terminal_label(**dec)
    claims = [rs.Claim(name=f"{f}_{name}", recorded={k: contrasts[f][name][k] for k in ("mean", "lcb95", "ucb95")},
                       minuend={"arm": a[0], "pole": a[1]}, subtrahend={"arm": b[0], "pole": b[1]}, value_field=f)
              for f in ("win", "blue", "margin") for name, (a, b) in CONTRASTS.items()]
    att_summary = {"n_cells": sum(1 for k in merged if k[0] == "N_prime"), "all_native_surface": True,
                   "n_emitted_total": int(sum(o["cell"]["meta"]["attest"]["n_emitted"] for k, o in merged.items() if k[0] == "N_prime")),
                   "n_boundary_total": int(sum(o["cell"]["meta"]["attest"]["n_boundary"] for k, o in merged.items() if k[0] == "N_prime")),
                   "n_controller_calls_total": int(sum(o["cell"]["meta"]["attest"]["n_controller_calls"] for k, o in merged.items() if k[0] == "N_prime")),
                   "n_boundary_unverifiable_at_episode_end_total": int(sum(o["cell"]["meta"]["attest"]["n_boundary_unverifiable_at_episode_end"]
                                                                            for k, o in merged.items() if k[0] == "N_prime"))}
    payload = {
        "record_id": f"{STEM}_RESULT", "implements": SPEC_PATH.name, "utc": _now(), "device": DEVICE,
        "seed_block": {"base": SEED_BASE, "n": SEED_N, "class": SEED_CLASS},
        "PRIMARY_WIN_RATE_CONTRASTS": {n: {**win[n], "lcb95_above_zero": bool(win[n]["lcb95"] > 0)} for n in ("Delta_A_N", "Delta_B_N")},
        "SCAFFOLD_KNOWN_CROSSOVER_REPRODUCTION_win_rate": {n: win[n] for n in ("Delta_A_Aprime", "Delta_B_Aprime")},
        "MECHANISM_DIAGNOSTICS_NOT_GATES_win_rate": {n: win[n] for n in ("I_A_Aprime", "I_B_Aprime", "I_A_N", "I_B_N", "N_minus_Aprime_A", "N_minus_Aprime_B")},
        "NATIVE_CROSSOVER_ON_THE_FRESH_BLOCK_DESCRIPTIVE_win_rate": {n: win[n] for n in ("native_Delta_A", "native_Delta_B")},
        "DESCRIPTIVE_BLUE_GOALS_CONTRASTS": contrasts["blue"], "DESCRIPTIVE_MARGIN_CONTRASTS": contrasts["margin"], "CELL_MEANS": cell_means,
        "IN_ENV_PATH_CHECK": {"criterion": f"GATING: interaction-free-window live path vs isolated DEFEND reference, RMSE <= {RADIUS_TOL}, coverage LCB95 >= {COVER} per pole "
                                          "(the audit's own robustness definition; excludes the live avoid-collision-shove confound, not a threshold change). "
                                          "Full-population rate reported as a non-gating diagnostic.",
                              "csv_sha256": _sha(CHECK_CELLS_PATH), **{k: chk[k] for k in ("path_check", "path_check_status", "path_check_full_population_diagnostic")}},
        "HEADING_DIAGNOSTIC": {"realized_heading_agreement": chk["heading"], "paired_A_prime_minus_N_prime": chk["heading_contrast_A_minus_N"]},
        "DECISION_INPUTS": dec, "N_PRIME_ATTESTATION": att_summary,
        "TERMINAL_OUTCOME": label, "INTERPRETATION": statement, "interpretation_guard": GUARD_SENTENCE,
        "provenance": {"spec_sha256": _sha(SPEC_PATH), "script_sha256": _sha(Path(__file__)), "audit_script_sha256": _sha(Path(A.__file__)),
                       "scaffold_runner_sha256": _sha(Path(S.__file__)), "mechanism_source_sha256": _sha(S.MECHANISM_FILE),
                       "pole_b_genome_sha256": _sha(S.B33_GENOME)},
        "claim_boundary": "Two frozen 4v4 learned specialists, the two certified 4v4 poles, fresh sealed_confirmatory seeds. TERMINAL_OUTCOME is certified ONLY if this "
                          "record's own status is SEALED; integrity/audit failure takes precedence over every label. " + GUARD_SENTENCE,
    }
    plan = rs.AuditPlan(
        rows_csv=ROWS_PATH, expected_rows=len(rows), expected_seeds=SEEDS, group_by=("arm", "pole"), seed_field="seed",
        int_fields=("seed", "steps", "blue", "red", "margin"), binary_fields=("win",),
        derived={"margin": rs.Derived("margin == blue - red", lambda r: r["blue"] - r["red"]),
                 "win": rs.Derived("win == (blue > red)", lambda r: int(r["blue"] > r["red"]))},
        checkpoints={n: (paths[n], pins[n]) for n in paths}, spec_path=SPEC_PATH, seed_class=SEED_CLASS, experiment_id=EXP_ID,
        n_boot=N_BOOT, alpha=ALPHA, rng_seed=RNG_SEED, claims=claims)
    rs.seal(out_path=RESULT_PATH, payload=payload, plan=plan, state=state, strict=False)
    sealed = _load_json(RESULT_PATH)
    sr.set_status(EXP_ID, "SPENT", note=f"sealed {sealed.get('status')}; {label}")
    print(json.dumps({"status": sealed.get("status"), "TERMINAL_OUTCOME": label, "decision_inputs": dec,
                      "win_rate_contrasts": {n: {k: win[n][k] for k in ("mean", "lcb95", "ucb95")} for n in CONTRASTS}}, indent=2))
    return 0 if sealed.get("status") == "SEALED" else 2


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
    add("C1_PINNED_INPUTS_EQUAL_THEIR_PINS", ok, msg)
    paths = _checkpoint_paths(spec)
    pins = {n: spec["INTEGRITY_FROZEN"]["checkpoints"][n]["sha256"] for n in paths}
    now = {n: _sha(p) for n, p in paths.items()}
    add("C2_CHECKPOINTS_EQUAL_THE_PINNED_HASHES", now == pins, f"{now} vs pinned {pins}")
    setup = S._setup_poles()
    add("C3_BOTH_POLES_ATTESTED_AND_POLE_B_IS_THE_CERTIFIED_B3_3_GENOME", set(setup["att"]) == set(POLES) and setup["pole_b"] is not None,
        f"attested {sorted(setup['att'])}; Pole B genome file {S.B33_GENOME.name}")
    ok, msg = _registered()
    counts: dict = {}
    for s in SEEDS:
        counts[pair_for_seed(s)] = counts.get(pair_for_seed(s), 0) + 1
    add("C4_BLOCK_RESERVED_SEALED_CONFIRMATORY_UNSPENT_DISJOINT_AND_ROTATION_BALANCED", ok and set(counts) == set(PAIRS) and max(counts.values()) - min(counts.values()) <= 1,
        f"{msg}; pair counts {dict(sorted(counts.items()))}")
    reg_before = json.dumps([b for b in json.loads((ROOT / "artifacts" / "SEED_REGISTRY.json").read_text(encoding="utf-8"))["blocks"] if b["experiment_id"] != EXP_ID], sort_keys=True)

    eng = A.Engine()
    from experiments.run_scaffolded_a_crossover_bridge_4v4 import _load_policies
    pols = _load_policies(setup["R2"], paths, DEVICE)
    before = {n: L._param_digest(p) for n, p in pols.items()}
    sealed_rows = {}
    with S.ROWS_PATH.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sealed_rows[(r["arm"], r["pole"], int(r["seed"]))] = (int(r["steps"]), int(r["blue"]), int(r["red"]))
    # C5: the unchanged arms reproduce the sealed BRIDGE episodes exactly through THIS runner's code path
    par_seeds = (S.SEEDS[0], S.SEEDS[1])
    par = {}
    n5 = {}
    with torch.no_grad():
        for seed in par_seeds:
            for pole in POLES:
                for arm in ("pi_A", "A_prime", "pi_B"):
                    row, _cell = run_cell(setup, pols, eng, arm, pole, seed)
                    par[(arm, pole, seed)] = (row["steps"], row["blue"], row["red"]) == sealed_rows[(arm, pole, seed)]
    add("C5_UNCHANGED_ARMS_REPRODUCE_SEALED_BRIDGE_EPISODES_EXACTLY", all(par.values()), f"{sum(par.values())}/{len(par)} episodes (pi_A, A', pi_B x 2 poles x 2 seeds) equal their sealed rows")
    # N' episodes for the remaining contracts
    ncell: dict = {}
    with torch.no_grad():
        for seed in par_seeds:
            for pole in POLES:
                _row, cell = run_cell(setup, pols, eng, "N_prime", pole, seed, keep_actions=True)
                ncell[(pole, seed)] = cell
    # C6: same selection logic as the audit's oracle (constant-flag windows)
    n_eq = n_tot = 0
    for cell in ncell.values():
        for k in range(2):
            st = np.asarray(cell["state"][k], dtype=np.float64)
            lg = cell["legal"][k]
            for t0 in range(0, len(st) - H, 7):
                w = st[t0:t0 + H]
                if not np.allclose(w[:, IX["Fx"]], w[0, IX["Fx"]]) or not np.allclose(w[:, IX["Fy"]], w[0, IX["Fy"]]):
                    continue
                if not controller_active(w[0]):
                    continue
                ref_pos, _ts, _pre = eng.rollout_ref(w)
                oracle_first = eng.path_oracle(w, int(lg[t0]), ref_pos, interruptible=False)[3][0][1]
                n_tot += 1
                n_eq += int(controller_select(eng, w[0], int(lg[t0])) == oracle_first)
    add("C6_CONTROLLER_SELECTION_EQUALS_THE_AUDIT_ORACLES_FIRST_BOUNDARY_CHOICE", n_tot > 0 and n_eq == n_tot, f"{n_eq}/{n_tot} constant-flag recorded windows: identical waypoint")
    # C7: native-surface attestation from the real action array and the env's own commit state
    att = {k: _attestation_ok(c["meta"]) for k, c in ncell.items()}
    tot = {f: sum(c["meta"]["attest"][f] for c in ncell.values())
           for f in ("n_emitted", "n_native", "n_boundary", "n_commit_match", "n_controller_calls", "n_boundary_unverifiable_at_episode_end")}
    add("C7_EVERY_N_PRIME_ACTION_IS_ON_THE_NATIVE_SURFACE_AND_THE_ENV_COMMIT_STATE_AGREES", all(v[0] for v in att.values()),
        f"{len(ncell)} N' episodes: emitted {tot['n_emitted']} == native-surface {tot['n_native']}; boundary {tot['n_boundary']} == env commit match {tot['n_commit_match']}; "
        f"controller calls {tot['n_controller_calls']}; excluded as unverifiable at episode-end reset {tot['n_boundary_unverifiable_at_episode_end']} "
        f"(GPUCTFVecEnv auto-resets on done; those boundaries are counted in neither n_boundary nor n_commit_match)")
    # C8: no side door
    src_ep, src_ctl = inspect.getsource(run_recorded_episode), inspect.getsource(controller_select)
    banned = ["install_forced" + "_defend_target", "_forced_defend" + "_targets =", "P." + "install", "_build_targets_from_action(" ]
    hits = [b for b in banned if b in src_ep + src_ctl]
    runtime_clean = all(not c["meta"]["attest"]["scaffold_patched"] for c in ncell.values())
    add("C8_NO_SIDE_DOOR_STATIC_AND_RUNTIME", not hits and runtime_clean, f"N' episode + controller sources contain none of {banned}: {not hits}; no scaffold patch on any N' core: {runtime_clean}")
    # C9: purity / outcome-blindness
    rng = np.random.default_rng(3)
    n_pure = n_ok = 0
    forbidden = [i for i, f in enumerate(A.STATE_FIELDS) if f not in CONTROLLER_FIELDS]
    for cell in ncell.values():
        for k in range(2):
            st = np.asarray(cell["state"][k], dtype=np.float64)
            lg = cell["legal"][k]
            for t in np.flatnonzero(rng.random(len(st)) < 0.15):
                if not (st[t, IX["cl"]] <= 0 and controller_active(st[t])):
                    continue
                base = controller_select(eng, st[t], int(lg[t]))
                pert = st[t].copy()
                for i in forbidden:
                    pert[i] = rng.uniform(-50, 50)
                pert[IX["cl"]] = st[t, IX["cl"]]
                n_pure += 1
                n_ok += int(controller_select(eng, pert, int(lg[t])) == base)
    # every emitted boundary action is reproduced offline from the recorded snapshot
    n_rep = n_rep_ok = 0
    for cell in ncell.values():
        emitted = cell["emitted"]
        for k in range(2):
            st = np.asarray(cell["state"][k], dtype=np.float64)
            lg = cell["legal"][k]
            for t in range(len(st)):
                if st[t, IX["cl"]] <= 0 and st[t, IX["alive"]] > 0.5 and controller_active(st[t]):
                    n_rep += 1
                    n_rep_ok += int(emitted[2 * t + k][2] == controller_select(eng, st[t], int(lg[t])))
    add("C9_CONTROLLER_IS_A_PURE_FUNCTION_OF_ITS_SNAPSHOT_AND_MASK", n_pure > 0 and n_ok == n_pure and n_rep > 0 and n_rep_ok == n_rep,
        f"perturbing every non-controller field (mines, enemy flag, commit ids, neighbour distance) changed {n_pure - n_ok}/{n_pure} decisions; offline recomputation from the recorded snapshots reproduced {n_rep_ok}/{n_rep} emitted boundary actions")
    # C10: determinism
    (pole0, seed0), first = next(iter(ncell.items()))
    _row2, second = run_cell(setup, pols, eng, "N_prime", pole0, seed0, keep_actions=True)
    same = (first["emitted"] == second["emitted"] and first["meta"]["blue"] == second["meta"]["blue"] and first["meta"]["red"] == second["meta"]["red"]
            and first["meta"]["steps"] == second["meta"]["steps"] and first["state"] == second["state"])
    add("C10_N_PRIME_IS_DETERMINISTIC_GIVEN_STATE_AND_SEED", same, f"cell ({pole0}, {seed0}) rerun: emitted actions, recorded states and outcome identical: {same}")
    # C11: planted truth for the controller
    j0 = 20
    row = np.zeros(len(A.STATE_FIELDS))
    row[IX["alive"]] = 1.0
    row[IX["Fx"]], row[IX["Fy"]] = eng.wp[j0]
    row[IX["x"]], row[IX["y"]], row[IX["h"]], row[IX["v"]] = float(eng.wp[j0][0]) - 9.0, float(eng.wp[j0][1]), 0.0, 1.0
    all_legal = (1 << 0) | (((1 << eng.nT) - 1) << eng.nM)
    picked = controller_select(eng, row, all_legal)
    add("C11_PLANTED_TRUTH_CONTROLLER_PICKS_THE_WAYPOINT_THE_DEFEND_PATH_IS_HEADING_FOR", picked == j0, f"flag placed on waypoint {j0}, defender 9 cells away: chose waypoint {picked}")
    # C12: decision tree + coverage classes
    T_, F_ = True, False
    cases = [
        (dict(integrity_ok=F_, delta_a_lcb=1, delta_b_lcb=1, instrument_valid=T_, check_pass_n=T_, direction_manipulated=T_), LABEL_INVALID),
        (dict(integrity_ok=T_, delta_a_lcb=0.05, delta_b_lcb=0.04, instrument_valid=T_, check_pass_n=T_, direction_manipulated=T_), LABEL_CONFIRMED),
        (dict(integrity_ok=T_, delta_a_lcb=0.05, delta_b_lcb=0.04, instrument_valid=F_, check_pass_n=F_, direction_manipulated=F_), LABEL_CONFIRMED),
        (dict(integrity_ok=T_, delta_a_lcb=-0.01, delta_b_lcb=0.2, instrument_valid=T_, check_pass_n=T_, direction_manipulated=T_), LABEL_LOST_HIGH_FIDELITY),
        (dict(integrity_ok=T_, delta_a_lcb=0.0, delta_b_lcb=0.2, instrument_valid=T_, check_pass_n=T_, direction_manipulated=T_), LABEL_LOST_HIGH_FIDELITY),
        (dict(integrity_ok=T_, delta_a_lcb=-0.1, delta_b_lcb=-0.1, instrument_valid=T_, check_pass_n=F_, direction_manipulated=T_), LABEL_INCONCLUSIVE_CONTROLLER),
        (dict(integrity_ok=T_, delta_a_lcb=-0.1, delta_b_lcb=-0.1, instrument_valid=F_, check_pass_n=T_, direction_manipulated=T_), LABEL_INCONCLUSIVE_INSTRUMENT),
        (dict(integrity_ok=T_, delta_a_lcb=-0.1, delta_b_lcb=-0.1, instrument_valid=F_, check_pass_n=F_, direction_manipulated=T_), LABEL_INCONCLUSIVE_INSTRUMENT),
    ]
    lab_ok = all(terminal_label(**kw)[0] == want for kw, want in cases)
    t_yes = terminal_label(integrity_ok=T_, delta_a_lcb=1, delta_b_lcb=1, instrument_valid=T_, check_pass_n=T_, direction_manipulated=T_)[1]
    t_no = terminal_label(integrity_ok=T_, delta_a_lcb=1, delta_b_lcb=1, instrument_valid=T_, check_pass_n=T_, direction_manipulated=F_)[1]
    t_lost = terminal_label(integrity_ok=T_, delta_a_lcb=-1, delta_b_lcb=1, instrument_valid=T_, check_pass_n=T_, direction_manipulated=T_)[1]
    # "necess" (not "necessary") because STATEMENTS["confirmed_direction_not_shown"] correctly reads "the necessITY of
    # instantaneous DIRECTION" -- a different word FORM conveying the same claim, not a wording defect.
    word_checks = {"t_yes has 'not necessary'": "not necessary" in t_yes, "t_no has a necessity/necessary word": "necess" in t_no,
                   "t_no has 'no statement'": "no statement" in t_no, "t_lost withholds the necessity claim": "NOT a finding that DIRECTION is necessary" in t_lost,
                   "t_lost withholds PPO authorization": "does not authorize PPO" in t_lost, "t_lost states causal importance": "may be causally important" in t_lost}
    wording_ok = all(word_checks.values())
    n_lab_ok = sum(terminal_label(**kw)[0] == want for kw, want in cases)
    add("C12_INTERPRETATION_TREE_AND_WORDING_SELFTESTS", lab_ok and wording_ok,
        f"{n_lab_ok}/{len(cases)} branch cases return the frozen label; wording checks: " +
        "; ".join(f"{k}={v}" for k, v in word_checks.items()))
    # C13: plumbing of the in-env check on recorded contract cells (invariants only; no rates printed)
    rows_chk = []
    for (pole, seed), cell in ncell.items():
        rows_chk.append(check_cell("N_prime", pole, seed, cell))
    with torch.no_grad():
        for pole in POLES:
            for seed in par_seeds:
                _r, acell = run_cell(setup, pols, eng, "A_prime", pole, seed)
                rows_chk.append(check_cell("A_prime", pole, seed, acell))
    inv = [all(np.isfinite(float(r[k])) for k in CHECK_FIELDS if k not in ("arm", "pole")) and r["win_pass"] <= r["n_win"] and r["n_win_free"] <= r["n_win"]
           and r["win_free_pass"] <= r["n_win_free"] and r["head_pass"] <= r["n_t"] for r in rows_chk]
    dchk = derive_check([{k: str(v) for k, v in r.items()} for r in rows_chk])
    add("C13_IN_ENV_CHECK_AND_HEADING_DIAGNOSTIC_PLUMBING", all(inv) and sum(r["n_win"] for r in rows_chk) > 0 and sum(r["n_t"] for r in rows_chk) > 0
        and set(dchk["path_check_status"]) == set(RECORDED_ARMS), f"{sum(inv)}/{len(inv)} per-cell invariants hold on {len(rows_chk)} recorded contract cells; derivation runs. Rates deliberately not printed.")
    # C14: fence
    src = Path(__file__).read_text(encoding="utf-8")
    forbidden_tokens = [".le" + "arn(", "optimizer" + ".step", ".back" + "ward(", "PP" + "O(", "zero_" + "grad"]
    fh_ = [t for t in forbidden_tokens if t in src]
    after = {n: L._param_digest(p) for n, p in pols.items()}
    reg_after = json.dumps([b for b in json.loads((ROOT / "artifacts" / "SEED_REGISTRY.json").read_text(encoding="utf-8"))["blocks"] if b["experiment_id"] != EXP_ID], sort_keys=True)
    add("C14_READ_ONLY_FENCE_DIGESTS_AND_OTHER_REGISTRY_ENTRIES_UNCHANGED", not fh_ and before == after and reg_before == reg_after,
        f"no training/optimizer call in the source: {not fh_}; both policy parameter digests unchanged: {before == after}; every other registry entry byte-identical: {reg_before == reg_after}")
    decision = "CONTRACTS_PASS" if all(c["pass"] for c in checks) else "CONTRACTS_FAIL"
    result = {"record_id": f"{STEM}_CONTRACT_RESULT", "implements": SPEC_PATH.name, "utc": _now(), "DECISION": decision,
              "spec_sha256": _sha(SPEC_PATH), "script_sha256": _sha(Path(__file__)), "checks": checks}
    CONTRACT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  N' CONTRACTS: {decision}  ({sum(not c['pass'] for c in checks)}/{len(checks)} failed)", flush=True)
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
