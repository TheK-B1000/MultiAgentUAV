"""CLOSEST_COMPOSITION_CONTRAST_6V6_V1_SPEC.json.

DESCRIPTIVE / EXPLORATORY (n=64, no confirmatory label): does the sealed scripted
5A/1D vs 3A/3D contrast survive when role assignment is CLOSEST_DEFENDS geometry
instead of FIXED_IDENTITY_PREFIX?

Scripted macros only (same make_env_n / action_for_roles_n as the sealed 6v6
composition sweep and FIXED_ATTACK_HEAVY confirmation). No learned checkpoints,
no env-level forced-DEFEND monkeypatch, no PPO.

  contracts   Mechanism correctness; spends no seed.
  run         256 full episodes (64 x 2 poles x 2 arms), resumable partial, CPU.
  analyze     Refuses unless all 256 cells present; seals via run_state.seal.
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

from experiments.run_pyquaticus_6v6_role_composition_sweep import (  # noqa: E402
    N_AGENTS,
    POLES,
    _ids_n,
    action_for_roles_n,
    composition_roles_n,
    make_env_n,
)
from experiments.run_pyquaticus_4v4_team_evaluation import (  # noqa: E402
    HORIZON,
    _telemetry_for_tick,
)
from experiments.run_routed_composition_outcome import ALPHA, N_BOOT, RNG_SEED, _bootstrap  # noqa: E402
from experiments.tqdm_loop import tqdm_iter  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
STEM = "CLOSEST_COMPOSITION_CONTRAST_6V6"
EXP_ID = STEM
SPEC_PATH = SD / f"{STEM}_V1_SPEC.json"
CONTRACT_PATH = SD / f"{STEM}_CONTRACT_RESULT.json"
RESULT_PATH = SD / f"{STEM}_RESULT.json"
ROWS_PATH = SD / f"{STEM}_ROWS.csv"
PARTIAL_PATH = SD / f"{STEM}_PARTIAL.jsonl"

SEED_BASE, SEED_N, SEED_CLASS = 22_100_001, 64, "exploratory"
SEEDS = list(range(SEED_BASE, SEED_BASE + SEED_N))
ARMS = ("5A_1D_closest", "3A_3D_closest")
ARM_K = {"5A_1D_closest": 1, "3A_3D_closest": 3}

CONTRASTS = {
    "Delta_B": (("5A_1D_closest", "B"), ("3A_3D_closest", "B")),
    "Delta_A": (("5A_1D_closest", "A"), ("3A_3D_closest", "A")),
    "H_A": (("3A_3D_closest", "A"), ("5A_1D_closest", "A")),
}
GUARD_SENTENCE = (
    "Both arms are SCRIPTED role macros with CLOSEST_DEFENDS assignment at t=0, "
    "not learned policies and not FIXED_IDENTITY. DESCRIPTIVE screen: no LCB95 "
    "gate, no terminal CONFIRMED label. Answers whether the sealed 5A/1D vs 3A/3D "
    "composition contrast survives geometry assignment; does not authorize pi_D."
)


def _load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with Path(p).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================== CLOSEST_DEFENDS roles

def closest_roles(core, k_defend: int) -> tuple[int, ...]:
    """Episode-static roles: k_defend closest ACTIVE agents to own_flag_home DEFEND.

    ACTIVE = alive & ~tagged & ~carrying. Ties broken by lower agent index
    (np.argsort stable). Computed once at the post-reset state.
    """
    alive = core.blue_alive[0].detach().cpu().numpy()
    tagged = core.blue_tagged[0].detach().cpu().numpy()
    carrying = core.blue_carrying[0].detach().cpu().numpy()
    active = alive & ~tagged & ~carrying
    idx = np.where(active)[0]
    if len(idx) < k_defend:
        raise SystemExit(f"FAIL-CLOSED: only {len(idx)} ACTIVE agents at t=0, need >= {k_defend}")
    x = core.blue_x[0].detach().cpu().numpy()
    y = core.blue_y[0].detach().cpu().numpy()
    home = core.blue_flag_home[0].detach().cpu().numpy()
    d = np.hypot(x[idx] - home[0], y[idx] - home[1])
    order = idx[np.argsort(d, kind="stable")]
    defenders = set(int(i) for i in order[:k_defend])
    roles = tuple(1 if i in defenders else 0 for i in range(N_AGENTS))
    if roles.count(1) != k_defend or roles.count(0) != N_AGENTS - k_defend:
        raise AssertionError((k_defend, roles))
    return roles


def run_closest_composition_episode(pole: str, arm: str, seed: int) -> dict[str, Any]:
    k = ARM_K[arm]
    env, core, genome, _live = make_env_n(pole, seed)
    try:
        roles = closest_roles(core, k)
        attackers, defenders = _ids_n(roles)
        counters = {"attack_ticks": 0, "defend_ticks": 0,
                    "attack_enemy_flag_branch_count": 0, "carrier_home_branch_count": 0,
                    "defend_inward_count": 0, "defend_outward_count": 0,
                    "tagged_ticks_by_role": 0}
        terminal_info, steps = None, 0
        for _ in range(HORIZON):
            _telemetry_for_tick(core, roles, counters)
            env.step_async(action_for_roles_n(core, roles))
            _obs, _rew, done, infos = env.step_wait()
            steps += 1
            if bool(np.asarray(done).any()):
                terminal_info = dict(infos[0])
                break
        if terminal_info is None:
            terminal_info = {"episode_result": {"blue_score": int(core.blue_score[0].item()),
                                                "red_score": int(core.red_score[0].item())},
                             "terminal_observation": {}}
        r = dict(terminal_info.get("episode_result") or {})
        blue_score, red_score = int(r.get("blue_score", 0)), int(r.get("red_score", 0))
        return {
            "arm": arm, "pole": pole, "seed": int(seed), "k_defend": int(k),
            "defender_ids": ",".join(map(str, defenders)),
            "attacker_ids": ",".join(map(str, attackers)),
            "blue_score": blue_score, "red_score": red_score,
            "win": int(blue_score > red_score), "draw": int(blue_score == red_score),
            "margin": blue_score - red_score, "steps": int(steps),
            "genome_id": str(genome.genome_id), **counters, "role_switch_count": 0,
        }
    finally:
        env.close()


# ============================================================================== contracts

def contracts() -> dict:
    checks: list[dict] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    spec = _load_json(SPEC_PATH)
    add("C0_SPEC_FROZEN", spec.get("status") == "FROZEN_BEFORE_SEED_SPEND",
        f"spec status = {spec.get('status')!r}")

    from experiments import seed_registry as sr
    ok, msg = sr.check_block(SEED_BASE, SEED_BASE + SEED_N - 1, SEED_CLASS, experiment_id=EXP_ID)
    entry = next((b for b in sr.load()["blocks"] if b["experiment_id"] == EXP_ID), None)
    reg_ok = bool(ok and entry is not None and entry["status"] == "RESERVED"
                  and entry["lo"] == SEED_BASE and entry["hi"] == SEED_BASE + SEED_N - 1
                  and entry["seed_class"] == SEED_CLASS)
    add("C1_SEED_BLOCK_RESERVED_EXPLORATORY_UNSPENT", reg_ok,
        f"{msg}; registry status={entry and entry['status']}")

    class FakeCore:
        pass

    rng = np.random.default_rng(11)
    import torch as T
    n_ok = n_eligible = 0
    for k in (1, 3):
        for _ in range(300):
            fc = FakeCore()
            alive = rng.random(N_AGENTS) < 0.95
            alive[rng.integers(0, N_AGENTS)] = True
            tagged = rng.random(N_AGENTS) < 0.05
            carrying = rng.random(N_AGENTS) < 0.05
            x, y = rng.uniform(0, 19, N_AGENTS), rng.uniform(0, 19, N_AGENTS)
            home = rng.uniform(0, 19, 2)
            fc.blue_alive = T.tensor([alive])
            fc.blue_tagged = T.tensor([tagged])
            fc.blue_carrying = T.tensor([carrying])
            fc.blue_x = T.tensor([x])
            fc.blue_y = T.tensor([y])
            fc.blue_flag_home = T.tensor([home])
            active = alive & ~tagged & ~carrying
            idx = np.where(active)[0]
            if len(idx) < k:
                continue
            n_eligible += 1
            d = np.hypot(x[idx] - home[0], y[idx] - home[1])
            want_def = set(int(i) for i in idx[np.argsort(d, kind="stable")][:k])
            got = closest_roles(fc, k)
            got_def = {i for i, r in enumerate(got) if r == 1}
            n_ok += int(got_def == want_def)
    add("C2_CLOSEST_ROLES_MATCH_INDEPENDENT_BRUTE_FORCE",
        n_eligible >= 400 and n_ok == n_eligible,
        f"{n_ok}/{n_eligible} eligible synthetic states (k in {{1,3}}) matched independent argsort")

    gaps_k1, gaps_k3 = [], []
    for seed in SEEDS:
        for pole in POLES:
            env, core, _g, _l = make_env_n(pole, seed)
            try:
                x = core.blue_x[0].detach().cpu().numpy()
                y = core.blue_y[0].detach().cpu().numpy()
                home = core.blue_flag_home[0].detach().cpu().numpy()
                d = np.sort(np.hypot(x - home[0], y - home[1]))
                gaps_k1.append(float(d[1] - d[0]))
                gaps_k3.append(float(d[3] - d[2]))
            finally:
                env.close()
    add("C3_SPAWN_GEOMETRY_NON_DEGENERATE",
        min(gaps_k1) > 1e-3 and min(gaps_k3) > 1e-3,
        f"k1 gap min={min(gaps_k1):.4f} med={float(np.median(gaps_k1)):.4f}; "
        f"k3 gap min={min(gaps_k3):.4f} med={float(np.median(gaps_k3)):.4f}")

    r1 = run_closest_composition_episode("A", "5A_1D_closest", SEEDS[0])
    r2 = run_closest_composition_episode("A", "5A_1D_closest", SEEDS[0])
    ids4 = {run_closest_composition_episode("A", "5A_1D_closest", s)["defender_ids"] for s in SEEDS[:4]}
    ids4_k3 = {run_closest_composition_episode("A", "3A_3D_closest", s)["defender_ids"] for s in SEEDS[:4]}
    add("C4_SELECTION_DETERMINISTIC_AND_STATE_DEPENDENT",
        r1["defender_ids"] == r2["defender_ids"]
        and (r1["blue_score"], r1["red_score"]) == (r2["blue_score"], r2["red_score"])
        and len(ids4) >= 2 and len(ids4_k3) >= 2,
        f"rerun identical; {len(ids4)} distinct k=1 defender sets / {len(ids4_k3)} k=3 sets "
        f"across first 4 seeds")

    from macro_actions import MacroAction
    roles_probe = (1, 0, 0, 0, 0, 0)
    env, core, _g, _l = make_env_n("A", SEEDS[0])
    try:
        a = action_for_roles_n(core, roles_probe)
        ok_macro = (int(a[0, 0, 0]) == int(MacroAction.DEFEND)
                    and all(int(a[0, i, 0]) in (int(MacroAction.GET_FLAG), int(MacroAction.GO_HOME))
                            for i in range(1, N_AGENTS)))
    finally:
        env.close()
    add("C5_ACTION_MACROS_MATCH_SEALED_SWEEP_VOCABULARY", ok_macro,
        "DEFEND on defenders; GET_FLAG/GO_HOME on attackers via action_for_roles_n")

    src = Path(__file__).read_text(encoding="utf-8")
    # Build ban tokens without embedding the literal call-site phrases in this
    # source line (a prior screen failed C6 by matching its own ban list).
    banned = [
        "defender_id_for_" + "seed",
        "seed %" + " N",
        "seed %" + " 6",
        "install_forced" + "_defend_target",
        ".le" + "arn(",
        "optimizer" + ".step",
        ".back" + "ward(",
        "PP" + "O(",
        "zero_" + "grad",
    ]
    # Exclude this contracts fence itself: only fail if a token appears outside
    # the banned-list construction / this comment block.
    fence_start = src.find("banned = [")
    fence_end = src.find("hits = [b for b in banned", fence_start)
    body = src[:fence_start] + src[fence_end:] if fence_start >= 0 and fence_end > fence_start else src
    hits = [b for b in banned if b in body]
    add("C6_NO_SEED_TABLE_NO_FORCED_TARGET_NO_TRAINING", not hits,
        "source fence clean" if not hits else f"found {hits}")

    def pm(db, da):
        return bool(db > 0 and da > 0)
    cases = [((0.1, 0.1), True), ((-0.1, 0.1), False), ((0.1, -0.1), False),
             ((0.0, 0.1), False), ((0.1, 0.0), False)]
    add("C7_PATTERN_MATCH_SELFTEST",
        all(pm(*args) == want for args, want in cases),
        f"{len(cases)}/{len(cases)} synthetic cases")

    prefix = composition_roles_n("5A_1D", N_AGENTS)
    add("C8_FIXED_IDENTITY_PREFIX_STILL_AVAILABLE_NOT_AN_ARM",
        prefix == (1, 0, 0, 0, 0, 0) and composition_roles_n("3A_3D", N_AGENTS) == (1, 1, 1, 0, 0, 0),
        f"5A_1D prefix={prefix}; 3A_3D prefix={composition_roles_n('3A_3D', N_AGENTS)}")

    decision = "CONTRACTS_PASS" if all(c["pass"] for c in checks) else "CONTRACTS_FAIL"
    result = {"record_id": f"{STEM}_CONTRACT_RESULT", "implements": SPEC_PATH.name,
              "utc": _now(), "DECISION": decision, "spec_sha256": _sha(SPEC_PATH),
              "script_sha256": _sha(Path(__file__)), "checks": checks}
    CONTRACT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"\n  {STEM} CONTRACTS: {decision}  "
          f"({sum(not c['pass'] for c in checks)}/{len(checks)} failed)", flush=True)
    return result


# ============================================================================== run

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


def run_stage() -> int:
    if _load_json(CONTRACT_PATH).get("DECISION") != "CONTRACTS_PASS":
        raise SystemExit("REFUSING: contracts did not pass")
    if RESULT_PATH.exists():
        raise SystemExit(f"REFUSING: result already exists: {RESULT_PATH}")
    from experiments import seed_registry as sr
    ok, msg = sr.check_block(SEED_BASE, SEED_BASE + SEED_N - 1, SEED_CLASS, experiment_id=EXP_ID)
    entry = next((b for b in sr.load()["blocks"] if b["experiment_id"] == EXP_ID), None)
    if not (entry is not None and entry["status"] == "RESERVED"):
        raise SystemExit(f"REFUSING: seed block not reserved: {msg}")

    done = _load_partial(PARTIAL_PATH)
    pending = [c for c in _cells() if c not in done]
    print(f"  {STEM} run: {len(_cells())} cells, {len(done)} recorded, {len(pending)} to run",
          flush=True)
    if not pending:
        return 0
    with PARTIAL_PATH.open("a", encoding="utf-8") as fh:
        for key in tqdm_iter(pending, desc=f"{STEM} (cpu)", total=len(pending), unit="ep"):
            arm, pole, seed = key
            row = run_closest_composition_episode(pole, arm, seed)
            fh.write(json.dumps({"key": list(key), "row": row}) + "\n")
            fh.flush()
    print(f"  {STEM} run complete", flush=True)
    return 0


# ============================================================================== analyze

def analyze() -> int:
    from experiments import run_state as rs
    from experiments import seed_registry as sr

    if _load_json(CONTRACT_PATH).get("DECISION") != "CONTRACTS_PASS":
        raise SystemExit("REFUSING: contracts did not pass")
    if RESULT_PATH.exists():
        raise SystemExit(f"REFUSING: result already exists: {RESULT_PATH}")

    merged = _load_partial(PARTIAL_PATH)
    want = set(_cells())
    missing, extra = sorted(want - set(merged)), sorted(set(merged) - want)
    if missing or extra:
        raise SystemExit(f"ABORT: {len(missing)} missing (e.g. {missing[:2]}), "
                         f"{len(extra)} unexpected. Resume --stage run first.")

    rows = [merged[c] for c in _cells()]
    with ROWS_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    state = rs.RunState(SD, STEM).begin(spec=SPEC_PATH.name, n_episodes=len(rows))
    val = {f: {(r["arm"], r["pole"], int(r["seed"])): float(r[f]) for r in rows}
           for f in ("win", "blue_score", "margin")}

    def diff(f: str, plus: tuple[str, str], minus: tuple[str, str]) -> np.ndarray:
        return np.asarray(
            [val[f][(plus[0], plus[1], s)] - val[f][(minus[0], minus[1], s)] for s in SEEDS],
            dtype=np.float64)

    contrasts = {f: {name: _bootstrap(diff(f, a, b)) for name, (a, b) in CONTRASTS.items()}
                 for f in ("win", "blue_score", "margin")}
    cell_means = {
        f"{arm}_pole{pole}": {
            f: round(float(np.mean([val[f][(arm, pole, s)] for s in SEEDS])), 6)
            for f in ("win", "blue_score", "margin")
        }
        for arm in ARMS for pole in POLES
    }
    win = contrasts["win"]
    pattern_match = bool(win["Delta_B"]["mean"] > 0 and win["Delta_A"]["mean"] > 0)

    prior_path = SD / "FIXED_ATTACK_HEAVY_6V6_OUTCOME_RESULT.json"
    prior_ref = None
    if prior_path.is_file():
        prior = _load_json(prior_path)
        # sealed confirmatory used different field names; keep a thin pointer only
        prior_ref = {
            "record": prior_path.name,
            "status": prior.get("status"),
            "label": (prior.get("TERMINAL_INTERPRETATION") or {}).get("label")
                     or prior.get("label"),
        }

    claims = [
        rs.Claim(name=name,
                 recorded={"mean": win[name]["mean"], "lcb95": win[name]["lcb95"],
                           "ucb95": win[name]["ucb95"]},
                 minuend={"arm": a[0], "pole": a[1]},
                 subtrahend={"arm": b[0], "pole": b[1]},
                 value_field="win")
        for name, (a, b) in CONTRASTS.items()
    ]
    plan = rs.AuditPlan(
        rows_csv=ROWS_PATH, expected_rows=len(rows), expected_seeds=SEEDS,
        group_by=("arm", "pole"), seed_field="seed",
        int_fields=("seed", "steps", "k_defend", "blue_score", "red_score", "draw",
                    "role_switch_count"),
        binary_fields=("win",), derived={}, checkpoints={},
        spec_path=SPEC_PATH, seed_class=SEED_CLASS, experiment_id=EXP_ID,
        n_boot=N_BOOT, alpha=ALPHA, rng_seed=RNG_SEED, claims=claims,
    )

    payload = {
        "record_id": f"{STEM}_RESULT", "implements": SPEC_PATH.name, "device": "cpu",
        "classification": "DESCRIPTIVE / EXPLORATORY -- no confirmatory label, no LCB95 gate",
        "seed_block": {"base": SEED_BASE, "n": SEED_N, "class": SEED_CLASS},
        "WIN_RATE_CONTRASTS_point_estimates_and_bootstrap_CI": win,
        "DESCRIPTIVE_BLUE_GOALS_CONTRASTS": contrasts["blue_score"],
        "DESCRIPTIVE_MARGIN_CONTRASTS": contrasts["margin"],
        "CELL_MEANS": cell_means,
        "PRIOR_FIXED_IDENTITY_CONFIRMATORY_POINTER": prior_ref,
        "PATTERN_MATCH_point_estimate_only_not_a_significance_claim": {
            "rule": "Delta_B_mean > 0 AND Delta_A_mean > 0",
            "result": pattern_match,
        },
        "interpretation_guard": GUARD_SENTENCE,
        "claim_boundary": (
            "Descriptive comparison only: scripted 5A/1D vs 3A/3D under CLOSEST_DEFENDS "
            "assignment on fresh exploratory seeds. Not confirmatory; not a learned claim; "
            "does not authorize pi_D. " + GUARD_SENTENCE
        ),
        "frozen_next_step_rule": (
            "pattern_match=TRUE -> PI gate on confirmatory n / heuristic+learned bridge; "
            "pattern_match=FALSE -> stop assignment-trick rescue of old 6v6 objects; "
            "escalate. Never auto-start pi_D from this record."
        ),
    }
    rs.seal(out_path=RESULT_PATH, payload=payload, plan=plan, state=state, strict=False)
    sealed = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    sr.set_status(EXP_ID, "SPENT", note=f"descriptive screen complete; pattern_match={pattern_match}")
    print(json.dumps({
        "status": sealed.get("status"),
        "pattern_match": pattern_match,
        "win_rate_point_estimates": {
            n: {"mean": win[n]["mean"], "lcb95": win[n]["lcb95"], "ucb95": win[n]["ucb95"]}
            for n in CONTRASTS
        },
        "cell_means": cell_means,
    }, indent=2))
    return 0 if sealed.get("status") == "SEALED" else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("contracts", "run", "analyze"), required=True)
    a = ap.parse_args()
    if a.stage == "contracts":
        return 0 if contracts()["DECISION"] == "CONTRACTS_PASS" else 2
    if a.stage == "run":
        return run_stage()
    return analyze()


if __name__ == "__main__":
    raise SystemExit(main())
