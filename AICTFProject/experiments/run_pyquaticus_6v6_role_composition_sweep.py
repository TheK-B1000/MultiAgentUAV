"""PYQUATICUS_6V6_ROLE_COMPOSITION_SWEEP_SPEC.json -- contracts, then the sweep.

Seven fixed compositions (6A/0D through 0A/6D) on both certified 6v6 poles,
paired seeds, PPO off. The deliverable is the per-pole argmax composition: the
pair a 6v6 router would switch between.

The n-agent helpers below generalize the frozen 4v4 ones. C2 requires them to
reproduce those 4v4 helpers EXACTLY at n=4 -- pure-function equality for the
role mapping, and tick-by-tick array equality for the action adapter driven on a
live 4v4 core. Without that, no cross-scale statement from this sweep is sound.

CLAIM BOUNDARY, frozen: the 6v6 Pole B is canonical SDS_PARENT_OP7, not the 4v4
B3-3 SDS2_B3_LOCKDEF10_2V1 construction. Cross-scale differences cannot be
attributed solely to team size.
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

import experiments.strategic_demand_searcher as searcher  # noqa: E402
from experiments.opponent_spec import assert_opponent_resolved  # noqa: E402
from experiments.pole_attestation import resolve_pole_genome  # noqa: E402
from experiments.run_pyquaticus_4v4_role_composition_sweep import (  # noqa: E402
    COMPOSITIONS as COMPOSITIONS_4V4,
    composition_roles as composition_roles_4v4,
)
from experiments.run_pyquaticus_4v4_team_evaluation import (  # noqa: E402
    HORIZON,
    N_MACROS_EVAL,
    _action_for_roles as action_for_roles_4v4,
    _telemetry_for_tick,
)
from experiments.run_routed_composition_outcome import (  # noqa: E402
    ALPHA,
    N_BOOT,
    PPO_FORBIDDEN_PREFIXES,
    RNG_SEED,
    _bootstrap,
)
from experiments.sds_genome import apply_genome_to_core  # noqa: E402
from experiments.tqdm_loop import tqdm_iter  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / "PYQUATICUS_6V6_ROLE_COMPOSITION_SWEEP_SPEC.json"
CERT_6V6 = ROOT / "artifacts" / "6v6_results" / "specs" / "STRATEGIC_DEMAND_6v6_GUARD_DISTRIBUTED_V2_CERTIFICATION.json"
CONTRACT_PATH = SD / "PYQUATICUS_6V6_ROLE_COMPOSITION_SWEEP_CONTRACT_RESULT.json"
RESULT_PATH = SD / "PYQUATICUS_6V6_ROLE_COMPOSITION_SWEEP_RESULT.json"
EPISODE_CSV = SD / "PYQUATICUS_6V6_ROLE_COMPOSITION_SWEEP_EPISODES.csv"
PARTIAL = SD / "PYQUATICUS_6V6_ROLE_COMPOSITION_SWEEP_PARTIAL.jsonl"
LABEL = "PYQUATICUS_6V6_ROLE_COMPOSITION_SWEEP"
EXPERIMENT_ID = "PYQUATICUS_6V6_ROLE_COMPOSITION_SWEEP"

N_AGENTS = 6
POLES = ("A", "B")
COMPOSITIONS = ("6A_0D", "5A_1D", "4A_2D", "3A_3D", "2A_4D", "1A_5D", "0A_6D")
BASELINE = "3A_3D"
SEED_BASE, SEED_N = 20_500_001, 64
SWEEP_SEEDS = list(range(SEED_BASE, SEED_BASE + SEED_N))
SEED_CLASS = "exploratory"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------- n-agent helpers

def composition_roles_n(composition: str, n: int) -> tuple[int, ...]:
    """FIXED_IDENTITY_PREFIX_DEFENDERS at arbitrary team size.

    Generalizes run_pyquaticus_4v4_role_composition_sweep.composition_roles,
    which hardcodes range(4). Must equal it exactly at n=4 (C2).
    """
    attackers_s, defenders_s = composition.split("_")
    attackers, defenders = int(attackers_s[:-1]), int(defenders_s[:-1])
    if attackers + defenders != n:
        raise ValueError(f"composition {composition!r} does not cover a team of {n}")
    roles = tuple(1 if i < defenders else 0 for i in range(n))
    if roles.count(1) != defenders or roles.count(0) != attackers:
        raise AssertionError((composition, roles))
    return roles


def action_for_roles_n(core, roles: tuple[int, ...]) -> np.ndarray:
    """Generalizes run_pyquaticus_4v4_team_evaluation._action_for_roles, which
    hardcodes the (1, 4, 2) action shape. Must equal it tick for tick at n=4 (C2)."""
    from macro_actions import MacroAction

    carrying = bool(core.blue_carrying[0].any().item())
    n = len(roles)
    action = np.zeros((1, n, 2), dtype=np.int64)
    for i, role in enumerate(roles):
        if role == 1:
            macro = int(MacroAction.DEFEND)
        elif carrying:
            macro = int(MacroAction.GO_HOME)
        else:
            macro = int(MacroAction.GET_FLAG)
        action[0, i, 0] = macro
        action[0, i, 1] = 0
    if int(action[..., 0].max()) >= N_MACROS_EVAL:
        raise AssertionError("evaluation action escaped the frozen n_macros=8 vocabulary")
    return action


def _ids_n(roles: tuple[int, ...]) -> tuple[list[int], list[int]]:
    return ([i for i, r in enumerate(roles) if r == 0],
            [i for i, r in enumerate(roles) if r == 1])


def make_env_n(pole: str, seed: int, n: int = N_AGENTS):
    """Pole construction at arbitrary team size, mirroring the validated 4v4
    _make_env: set phase -> set opponent -> apply genome -> reset -> reapply ->
    assert the LIVE resolved profile."""
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig

    genome = resolve_pole_genome(pole, n)
    cfg = GPUFieldConfig(
        n_envs=1, max_blue_agents=n, max_red_agents=n, n_macros=N_MACROS_EVAL,
        map_set="train", map_layout=searcher.MAP, max_decision_steps=HORIZON,
        score_limit=1_000_000, aquaticus_profile=True, rules_profile="OURS",
        device="cpu", seed=int(seed), obstacle_obs_channel=True,
        tag_telemetry_enabled=True, own_flag_home_required_to_score=True,
        **searcher.RULESET,
    )
    env = GPUCTFVecEnv(cfg)
    core = env.core
    env.env_method("set_phase", genome.base_opponent)
    env.env_method("set_next_opponent", "SCRIPTED", genome.base_opponent)
    apply_genome_to_core(core, genome)
    core.blue_scripted = False
    env.reset()
    apply_genome_to_core(core, genome)
    core.blue_scripted = False
    core.drain_tag_events()
    live = assert_opponent_resolved(core, genome.base_opponent, genome,
                                    context=f"6v6 composition sweep pole {pole}")
    return env, core, genome, live


def run_episode(pole: str, composition: str, seed: int) -> dict:
    env, core, genome, _live = make_env_n(pole, seed)
    try:
        roles = composition_roles_n(composition, N_AGENTS)
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
        agent_mask = (terminal_info.get("terminal_observation") or {}).get("agent_mask")
        return {
            "seed": int(seed), "pole": pole, "composition": composition,
            "defender_ids": ",".join(map(str, defenders)),
            "attacker_ids": ",".join(map(str, attackers)),
            "blue_score": blue_score, "red_score": red_score,
            "blue_win": int(blue_score > red_score), "draw": int(blue_score == red_score),
            "steps": int(steps),
            "blue_alive_end": int(np.asarray(agent_mask).sum()) if agent_mask is not None else None,
            "genome_id": str(genome.genome_id), **counters, "role_switch_count": 0,
        }
    finally:
        env.close()


# ---------------------------------------------------------------- contracts

def run_contracts() -> dict:
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str, **data: Any) -> None:
        checks.append({"name": name, "gating": True, "passed": bool(passed), "detail": detail, **data})

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    record("C0_SPEC_FROZEN", str(spec.get("status", "")).startswith("FROZEN"),
           f"spec status = {spec.get('status')!r}")

    # C2a -- pure-function parity of the role mapping at n=4
    mism = [c for c in COMPOSITIONS_4V4
            if composition_roles_n(c, 4) != composition_roles_4v4(c)]
    record("C2a_ROLE_MAPPING_PARITY_AT_N4", not mism and len(COMPOSITIONS_4V4) == 5,
           f"{len(COMPOSITIONS_4V4)} 4v4 compositions compared against the frozen helper; "
           f"mismatches: {mism or 'none'}")

    # C2b -- tick-by-tick parity of the action adapter on a LIVE 4v4 core.
    # Driven by the FROZEN helper so the trajectory is the validated one.
    from experiments.run_pyquaticus_4v4_team_evaluation import _make_env as make_env_4v4
    diffs, ticks_cmp, carry_seen = [], 0, False
    for comp in ("2A_2D", "4A_0D"):
        env, core, _g, _l = make_env_4v4("A", 20100001)
        try:
            roles4 = composition_roles_4v4(comp)
            for _ in range(HORIZON):
                a_old = action_for_roles_4v4(core, roles4)
                a_new = action_for_roles_n(core, roles4)
                ticks_cmp += 1
                carry_seen = carry_seen or bool(core.blue_carrying[0].any().item())
                if not np.array_equal(a_old, a_new):
                    diffs.append(f"{comp}@{ticks_cmp}")
                env.step_async(a_old)
                _o, _r, d, _i = env.step_wait()
                if bool(np.asarray(d).any()):
                    break
        finally:
            env.close()
    record("C2b_ACTION_ADAPTER_PARITY_AT_N4", not diffs and ticks_cmp > 0 and carry_seen,
           f"{ticks_cmp} live 4v4 ticks compared array-for-array; mismatches: {diffs or 'none'}; "
           f"a friendly-carrier tick was exercised: {carry_seen} (so the GO_HOME branch is covered, "
           f"not just GET_FLAG/DEFEND)")

    # C4 -- every composition covers the team, prefix rule holds
    bad = []
    for c in COMPOSITIONS:
        roles = composition_roles_n(c, N_AGENTS)
        d = int(c.split("_")[1][:-1])
        if len(roles) != N_AGENTS or roles[:d] != (1,) * d or roles[d:] != (0,) * (N_AGENTS - d):
            bad.append(c)
    record("C4_COMPOSITION_COVERS_THE_TEAM", not bad and len(COMPOSITIONS) == 7,
           f"7 compositions, identity-prefix rule verified at n=6; offenders: {bad or 'none'}")

    # C1 + C3 -- team size is really 6, and each pole resolves as certified
    cert = json.loads(CERT_6V6.read_text(encoding="utf-8"))
    if str(cert.get("status")) != "FROZEN_RESULT" or int(cert.get("team_size", 0)) != 6:
        record("C3_POLE_RESOLVES_AS_CERTIFIED", False,
               f"certification not usable: status={cert.get('status')!r} team_size={cert.get('team_size')!r}")
    else:
        shape_ok, pole_bad, shapes = True, [], {}
        for pole in POLES:
            env, core, genome, live = make_env_n(pole, 99_900_612)
            try:
                shapes[pole] = [int(core.blue_x.shape[1]), int(core.red_x.shape[1])]
                if shapes[pole] != [N_AGENTS, N_AGENTS]:
                    shape_ok = False
                want = cert["poles"][pole]
                for field in ("defender_zone_frac", "threat_radius"):
                    if abs(float(live.get(field, float("nan"))) - float(want[field])) > 1e-6:
                        pole_bad.append(f"{pole}.{field}: live={live.get(field)!r} cert={want[field]!r}")
                mad_live = int(round(float(live.get("min_alive_for_defender", -1))))
                mad_cert = int(want["overlay"]["min_alive_for_defender"])
                if mad_live != mad_cert:
                    pole_bad.append(f"{pole}.min_alive_for_defender: live={mad_live} cert={mad_cert}")
            finally:
                env.close()
        record("C1_TEAM_SIZE_IS_REALLY_SIX", shape_ok,
               f"live core agent counts read back from the constructed env: {shapes}")
        record("C3_POLE_RESOLVES_AS_CERTIFIED", not pole_bad,
               f"both poles checked against {CERT_6V6.name} on the fields it records; "
               f"disagreements: {pole_bad or 'none'}")

    # C5 -- action vocabulary
    env, core, _g, _l = make_env_n("A", 99_900_612)
    try:
        worst = max(int(action_for_roles_n(core, composition_roles_n(c, N_AGENTS))[..., 0].max())
                    for c in COMPOSITIONS)
    finally:
        env.close()
    record("C5_ACTION_VOCABULARY", worst < N_MACROS_EVAL,
           f"max emitted macro id across all 7 compositions = {worst} < n_macros {N_MACROS_EVAL}")

    # C7 -- PPO unreachable
    live_ppo = sorted(m for m in sys.modules if m.startswith(PPO_FORBIDDEN_PREFIXES))
    record("C7_NO_PPO_REACHABLE", not live_ppo, f"imported PPO/trainer modules: {live_ppo or 'none'}")

    # C8 -- the seed block is still free (allocation happens after contracts pass)
    from experiments import seed_registry as sr
    free, msg = sr.check_block(SEED_BASE, SEED_BASE + SEED_N - 1, SEED_CLASS,
                               experiment_id=EXPERIMENT_ID)
    record("C8_SEED_BLOCK_FREE_OR_OURS", free, msg)

    n_failed = sum(1 for c in checks if not c["passed"])
    report = {
        "record_id": "PYQUATICUS_6V6_ROLE_COMPOSITION_SWEEP_CONTRACT_RESULT",
        "utc": _now(), "implements": SPEC_PATH.name,
        "team_size": N_AGENTS, "compositions": list(COMPOSITIONS), "baseline": BASELINE,
        "n_checks": len(checks), "n_gating": len(checks), "n_failed": n_failed,
        "checks": checks,
        "DECISION": "CONTRACTS_PASS" if n_failed == 0 else "CONTRACT_FAILURE",
        "claim_boundary": "Contracts only. No sweep episode was run and no seed was allocated. "
                          "C2b reuses spent 4v4 seed 20100001 for deterministic parity only.",
    }
    CONTRACT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'=' * 66}\n  CONTRACTS: {report['DECISION']}  ({n_failed}/{len(checks)} gating failed)\n{'=' * 66}")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print(f"  -> {CONTRACT_PATH}")
    return report


# -------------------------------------------------------------------- sweep

def run_sweep() -> int:
    from experiments import run_state as rs
    from experiments import seed_registry as sr

    contracts = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contracts.get("DECISION") != "CONTRACTS_PASS":
        raise SystemExit(f"REFUSING: contracts did not pass ({contracts.get('DECISION')!r})")
    if RESULT_PATH.exists():
        raise SystemExit(f"REFUSING: result already exists: {RESULT_PATH}")

    entry = next((b for b in sr.load()["blocks"] if b["experiment_id"] == EXPERIMENT_ID), None)
    if entry is None:
        free, msg = sr.check_block(SEED_BASE, SEED_BASE + SEED_N - 1, SEED_CLASS,
                                   experiment_id=EXPERIMENT_ID)
        if not free:
            raise SystemExit(f"REFUSING: seed block not free at reservation time: {msg}")
        sr.allocate(EXPERIMENT_ID, SEED_BASE, SEED_BASE + SEED_N - 1, SEED_CLASS,
                    purpose="6v6 role-composition sweep, 7 compositions x 2 certified poles, paired, PPO off",
                    spec=SPEC_PATH.name)
        print(f"  reserved {SEED_BASE}-{SEED_BASE + SEED_N - 1} for {EXPERIMENT_ID}", flush=True)

    total = len(POLES) * len(COMPOSITIONS) * SEED_N
    state = rs.RunState(SD, LABEL).begin(spec=SPEC_PATH.name, n_episodes=total)

    rows: dict[tuple[str, str, int], dict] = {}
    if PARTIAL.is_file():
        for line in PARTIAL.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[(r["pole"], r["composition"], int(r["seed"]))] = r
        print(f"  resuming: {len(rows)}/{total} episodes already recorded", flush=True)

    pending = [
        (pole, comp, seed)
        for pole in POLES
        for comp in COMPOSITIONS
        for seed in SWEEP_SEEDS
        if (pole, comp, seed) not in rows
    ]
    with PARTIAL.open("a", encoding="utf-8") as fh:
        for pole, comp, seed in tqdm_iter(
            pending,
            desc="6v6 role-composition sweep",
            total=len(pending),
            unit="ep",
        ):
            row = run_episode(pole, comp, seed)
            rows[(pole, comp, seed)] = row
            fh.write(json.dumps(row) + "\n")
            fh.flush()

    fields = list(next(iter(rows.values())).keys())
    with EPISODE_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for k in sorted(rows):
            w.writerow(rows[k])

    def wr(pole: str, comp: str) -> float:
        return float(np.mean([rows[(pole, comp, s)]["blue_win"] for s in SWEEP_SEEDS]))

    def paired(pole: str, left: str, right: str, field: str = "blue_win") -> np.ndarray:
        return np.asarray([float(rows[(pole, left, s)][field]) - float(rows[(pole, right, s)][field])
                           for s in SWEEP_SEEDS])

    win_rates = {p: {c: round(wr(p, c), 6) for c in COMPOSITIONS} for p in POLES}
    margins = {p: {c: round(float(np.mean([rows[(p, c, s)]["blue_score"] - rows[(p, c, s)]["red_score"]
                                           for s in SWEEP_SEEDS])), 4) for c in COMPOSITIONS} for p in POLES}
    contrasts = {p: {c: _bootstrap(paired(p, c, BASELINE)) for c in COMPOSITIONS if c != BASELINE}
                 for p in POLES}
    regime = {c: _bootstrap(np.asarray([float(rows[("B", c, s)]["blue_win"]) -
                                        float(rows[("A", c, s)]["blue_win"]) for s in SWEEP_SEEDS]))
              for c in COMPOSITIONS}

    # primary deliverable: per-pole argmax, with its separation from runner-up and baseline
    deliverable = {}
    for p in POLES:
        order = sorted(COMPOSITIONS, key=lambda c: win_rates[p][c], reverse=True)
        best, runner = order[0], order[1]
        tied = [c for c in COMPOSITIONS if abs(win_rates[p][c] - win_rates[p][best]) < 1e-12]
        deliverable[p] = {
            "argmax_composition": best, "argmax_win_rate": win_rates[p][best],
            "tied_at_argmax": tied if len(tied) > 1 else None,
            "runner_up": runner, "runner_up_win_rate": win_rates[p][runner],
            "argmax_minus_runner_up": _bootstrap(paired(p, best, runner)),
            "argmax_minus_baseline": (_bootstrap(paired(p, best, BASELINE)) if best != BASELINE else None),
            "ranking": [{"composition": c, "win_rate": win_rates[p][c]} for c in order],
        }

    gate_details, regime_signal = [], False
    for c in COMPOSITIONS:
        if c == BASELINE:
            continue
        a, b = contrasts["A"][c], contrasts["B"][c]
        a_ex = a["lcb95"] > 0 or a["ucb95"] < 0
        b_ex = b["lcb95"] > 0 or b["ucb95"] < 0
        opp = (a["mean"] > 0) != (b["mean"] > 0)
        gate_details.append({"composition": c, "A_excludes_zero": a_ex,
                             "B_excludes_zero": b_ex, "opposite_sign": opp})
        regime_signal = regime_signal or (a_ex and b_ex and opp)
    any_comp = any((contrasts[p][c]["lcb95"] > 0 or contrasts[p][c]["ucb95"] < 0)
                   for p in POLES for c in contrasts[p])
    label = ("REGIME_DEPENDENT_COMPOSITION_SIGNAL" if regime_signal else
             "COMPOSITION_DESCRIPTIVE_SIGNAL" if any_comp else
             "NO_DEMONSTRATED_COMPOSITION_EFFECT")

    payload = {
        "record_id": "PYQUATICUS_6V6_ROLE_COMPOSITION_SWEEP_RESULT",
        "implements": SPEC_PATH.name, "utc": _now(),
        "team_size": N_AGENTS, "poles": list(POLES), "compositions": list(COMPOSITIONS),
        "baseline": BASELINE,
        "seed_block": {"base": SEED_BASE, "last": SEED_BASE + SEED_N - 1, "n": SEED_N, "class": SEED_CLASS},
        "contracts": {"path": CONTRACT_PATH.name, "decision": contracts["DECISION"],
                      "n_gating": contracts["n_gating"], "n_failed": contracts["n_failed"]},
        "PRIMARY_DELIVERABLE_per_pole_argmax": deliverable,
        "win_rates": win_rates, "score_margin_blue_minus_red": margins,
        "composition_contrasts_vs_baseline": contrasts,
        "regime_B_minus_A_contrasts": regime,
        "regime_gate_details": gate_details,
        "decision_label": label,
        "bootstrap": {"n_boot": N_BOOT, "alpha": ALPHA, "rng_seed": RNG_SEED,
                      "procedure": "paired percentile bootstrap over seeds"},
        "FROZEN_CLAIM_BOUNDARY_POLE_B": (
            "The 6v6 composition sweep uses the certified 6v6 pole pair. Pole B is canonical "
            "SDS_PARENT_OP7, not the 4v4 B3-3 SDS2_B3_LOCKDEF10_2V1 construction. Therefore "
            "cross-scale differences cannot be attributed solely to team size. If 6v6 prefers a "
            "different composition than 4v4, that could be team size, opponent construction, or "
            "their interaction."),
        "multiplicity": "7 compositions x 2 poles, unadjusted intervals, exploratory. A nominal "
                        "interval exclusion does not authorize a router, a confirmation or PPO.",
        "claim_boundary": "Fixed scripted role composition at 6v6 under the certified poles. Does not "
                          "establish learned roles, authorize a 6v6 router or PPO, or transfer the 4v4 "
                          "router's confirmed result to 6v6.",
    }

    plan = rs.AuditPlan(
        rows_csv=EPISODE_CSV, expected_rows=total, expected_seeds=SWEEP_SEEDS,
        group_by=("pole", "composition"), seed_field="seed",
        int_fields=("seed", "blue_score", "red_score", "blue_win", "draw", "steps"),
        binary_fields=("blue_win", "draw"),
        derived={"blue_win": rs.Derived("blue_win == int(blue_score > red_score)",
                                        lambda r: int(int(r["blue_score"]) > int(r["red_score"]))),
                 "draw": rs.Derived("draw == int(blue_score == red_score)",
                                    lambda r: int(int(r["blue_score"]) == int(r["red_score"])))},
        spec_path=SPEC_PATH, seed_class=SEED_CLASS, experiment_id=EXPERIMENT_ID,
        n_boot=N_BOOT, alpha=ALPHA, rng_seed=RNG_SEED,
        claims=tuple(
            rs.Claim(name=f"{p}_{c}_minus_{BASELINE}",
                     recorded=contrasts[p][c],
                     minuend={"pole": p, "composition": c},
                     subtrahend={"pole": p, "composition": BASELINE}, value_field="blue_win")
            for p in POLES for c in COMPOSITIONS if c != BASELINE),
    )
    rs.seal(out_path=RESULT_PATH, payload=payload, plan=plan, state=state, strict=False)
    sealed = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    sr.set_status(EXPERIMENT_ID, "SPENT", note=f"sealed {sealed.get('status')}; {label}")
    print(json.dumps({"decision_label": label,
                      "argmax": {p: deliverable[p]["argmax_composition"] for p in POLES},
                      "win_rates": win_rates}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("contracts", "sweep"), required=True)
    args = ap.parse_args()
    if args.stage == "contracts":
        return 0 if run_contracts()["DECISION"] == "CONTRACTS_PASS" else 2
    return run_sweep()


if __name__ == "__main__":
    raise SystemExit(main())
