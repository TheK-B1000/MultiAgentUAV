"""ROUTED_COMPOSITION_OUTCOME_V1_SPEC.json -- contracts, then the outcome arms.

Three arms on paired seeds, both poles: STATE_B_TRIGGER (the live frozen V2
router picks the composition per tick), FIXED_2A2D (A-safe baseline, and the
control for both primary gates) and FIXED_4A0D (descriptive reference).

The FIXED arms delegate to the already-validated sweep episode runner. The
STATE arm uses the routed runner below, and C5 proves that pinning the routed
runner to a constant composition reproduces the sweep runner bit-for-bit -- so
the arms differ only in the composition sequence.

PPO is off and unreachable. Sealing goes through experiments/run_state.py::seal.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_pyquaticus_4v4_role_composition_sweep import (  # noqa: E402
    _action_for_roles,
    _ids,
    composition_roles,
    run_episode as run_fixed_episode,
)
from experiments.run_pyquaticus_4v4_team_evaluation import (  # noqa: E402
    HORIZON,
    _make_env,
    _telemetry_for_tick,
)

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / "ROUTED_COMPOSITION_OUTCOME_V1_SPEC.json"
V2_RESULT = SD / "COMPOSITION_SELECTOR_TWO_FEATURE_UNCERTAINTY_AWARE_V2_RESULT.json"
V2_ROWS = SD / "composition_selector_two_feature_v2_raw_ticks.csv"
ORACLE_ROWS = SD / "PYQUATICUS_4V4_ORACLE_COMPOSITION_EPISODES.csv"
CONTRACT_PATH = SD / "ROUTED_COMPOSITION_OUTCOME_CONTRACT_RESULT.json"
RESULT_PATH = SD / "ROUTED_COMPOSITION_OUTCOME_RESULT.json"
EPISODE_CSV = SD / "ROUTED_COMPOSITION_OUTCOME_EPISODES.csv"
LABEL = "ROUTED_COMPOSITION_OUTCOME"
EXPERIMENT_ID = "ROUTED_COMPOSITION_OUTCOME_V1"

POLES = ("A", "B")
ARMS = ("STATE_B_TRIGGER", "FIXED_2A2D", "FIXED_4A0D")
DEFAULT_COMPOSITION = "2A_2D"
TRIGGERED_COMPOSITION = "4A_0D"
FIXED_COMPOSITION = {"FIXED_2A2D": "2A_2D", "FIXED_4A0D": "4A_0D"}

# frozen router operating point -- verified against the sealed V2 result by C6
R_BLUE = 4.0
R_RED = 4.0
WINDOW = 40
HYSTERESIS = 0.2
DWELL = 10
THRESHOLD = -0.8333333333333334

SEED_BASE, SEED_N = 20_200_001, 64
OUTCOME_SEEDS = list(range(SEED_BASE, SEED_BASE + SEED_N))
SEED_CLASS = "sealed_confirmatory"
TAU_A_HARM = 0.10
N_BOOT, ALPHA, RNG_SEED = 20_000, 0.05, 7
ORACLE_PARITY_SEEDS = list(range(20_100_001, 20_100_009))

PPO_FORBIDDEN_PREFIXES = ("rl.custom_ppo", "rl.train_ppo", "rl.trainer", "stable_baselines3")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------- router

class OnlineRouter:
    """The frozen V2 selector, run live.

    Deliberately mirrors calibrate_asymmetric_b_trigger.windowed +
    run_state_machine tick by tick. Its ONLY input is the scalar D_t: there is
    no parameter through which a pole label, genome, opponent id or outcome
    could reach it. C1 and C7 pin both properties.
    """

    def __init__(self, window: int = WINDOW, threshold: float = THRESHOLD,
                 hysteresis: float = HYSTERESIS, dwell: int = DWELL,
                 pinned: str | None = None) -> None:
        self.window = int(window)
        self.threshold = float(threshold)
        self.hysteresis = float(hysteresis)
        self.dwell = int(dwell)
        self.pinned = pinned
        self._buf: deque[float] = deque(maxlen=self.window)
        self._state = 0                  # 0 = 2A_2D (default), 1 = 4A_0D
        self._last_switch = -10_000
        self._t = -1
        self.switches = 0

    @property
    def composition(self) -> str:
        if self.pinned is not None:
            return self.pinned
        return TRIGGERED_COMPOSITION if self._state == 1 else DEFAULT_COMPOSITION

    def update(self, d_value: float) -> str:
        """Feed one tick's D_t, return the composition to execute this tick."""
        self._t += 1
        self._buf.append(float(d_value))
        v = sum(self._buf) / len(self._buf)
        self.windowed = v
        if self.pinned is None:
            elapsed = self._t - self._last_switch
            if self._state == 0 and v < self.threshold - self.hysteresis and elapsed >= self.dwell:
                self._state, self._last_switch, self.switches = 1, self._t, self.switches + 1
            elif self._state == 1 and v > self.threshold + self.hysteresis and elapsed >= self.dwell:
                self._state, self._last_switch, self.switches = 0, self._t, self.switches + 1
        return self.composition


def _count_red_near(core, cx: float, cy: float, radius: float) -> int:
    count = 0
    for i in range(core.red_x.shape[1]):
        if not bool(core.red_alive[0, i].item()):
            continue
        dx = float(core.red_x[0, i].item()) - cx
        dy = float(core.red_y[0, i].item()) - cy
        if (dx * dx + dy * dy) ** 0.5 <= radius:
            count += 1
    return count


def observe_d(core) -> float:
    """The declared feature. Reads only quantities already used by the legal
    macro-resolution path; no pole, genome, phase or outcome field is touched."""
    p_blue = _count_red_near(core, float(core.blue_flag_pos[0, 0].item()),
                             float(core.blue_flag_pos[0, 1].item()), R_BLUE)
    p_red = _count_red_near(core, float(core.red_flag_pos[0, 0].item()),
                            float(core.red_flag_pos[0, 1].item()), R_RED)
    return float(p_blue - p_red)


# ------------------------------------------------------------ routed episode

def run_routed_episode(pole: str, seed: int, pinned: str | None = None,
                       label: str = "ROUTED") -> tuple[dict, dict]:
    """Mirrors run_pyquaticus_4v4_role_composition_sweep.run_episode, with the
    roles recomputed each tick from the router. With ``pinned`` set it must
    reproduce that function bit-for-bit (C5)."""
    env, core, genome, live = _make_env(pole, seed)
    try:
        router = OnlineRouter(pinned=pinned)
        counters = {
            "attack_ticks": 0, "defend_ticks": 0,
            "attack_enemy_flag_branch_count": 0, "carrier_home_branch_count": 0,
            "defend_inward_count": 0, "defend_outward_count": 0,
            "tagged_ticks_by_role": 0,
        }
        terminal_info = None
        steps = 0
        first_roles: tuple[int, ...] | None = None
        ticks_triggered = 0
        for _ in range(HORIZON):
            composition = router.update(observe_d(core))
            roles = composition_roles(composition)
            if first_roles is None:
                first_roles = roles
            ticks_triggered += int(composition == TRIGGERED_COMPOSITION)
            _telemetry_for_tick(core, roles, counters)
            env.step_async(_action_for_roles(core, roles))
            _obs, _rew, done, infos = env.step_wait()
            steps += 1
            if bool(np.asarray(done).any()):
                terminal_info = dict(infos[0])
                break
        if terminal_info is None:
            terminal_info = {"episode_result": {"blue_score": int(core.blue_score[0].item()),
                                                "red_score": int(core.red_score[0].item())},
                             "terminal_observation": {}}
        result = dict(terminal_info.get("episode_result") or {})
        blue_score = int(result.get("blue_score", 0))
        red_score = int(result.get("red_score", 0))
        terminal_obs = terminal_info.get("terminal_observation") or {}
        agent_mask = terminal_obs.get("agent_mask")
        blue_alive_end = int(np.asarray(agent_mask).sum()) if agent_mask is not None else None
        attackers, defenders = _ids(first_roles or composition_roles(DEFAULT_COMPOSITION))
        row = {
            "seed": int(seed), "pole": pole, "composition": label,
            "defender_ids": ",".join(map(str, defenders)),
            "attacker_ids": ",".join(map(str, attackers)),
            "blue_score": blue_score, "red_score": red_score,
            "blue_win": int(blue_score > red_score),
            "draw": int(blue_score == red_score),
            "steps": int(steps), "blue_alive_end": blue_alive_end, "red_alive_end": None,
            "genome_id": str(genome.genome_id),
            "pole_config_hash": str(live.get("live_config_hash", "")),
            **counters,
            "role_switch_count": int(router.switches),
        }
        mapping = {"seed": int(seed), "pole": pole, "composition": label,
                   "defender_ids": defenders, "attacker_ids": attackers,
                   "router_switches": int(router.switches),
                   "router_ticks_4A0D": int(ticks_triggered),
                   "router_tick_fraction_4A0D": float(ticks_triggered / max(steps, 1))}
        return row, mapping
    finally:
        env.close()


def run_job(pole: str, arm: str, seed: int) -> tuple[dict, dict]:
    if arm == "STATE_B_TRIGGER":
        row, mapping = run_routed_episode(pole, seed, pinned=None, label="ROUTED")
    else:
        composition = FIXED_COMPOSITION[arm]
        row, mapping = run_fixed_episode(pole, composition, seed)
        mapping["router_switches"] = 0
        mapping["router_ticks_4A0D"] = 0 if composition == DEFAULT_COMPOSITION else int(row["steps"])
        mapping["router_tick_fraction_4A0D"] = 0.0 if composition == DEFAULT_COMPOSITION else 1.0
    row["arm"] = arm
    mapping["arm"] = arm
    return row, mapping


# ---------------------------------------------------------------- contracts

def _offline_sequence(values: list[float]) -> list[int]:
    """The calibration-side path, imported so C7 compares against the real thing."""
    from experiments.calibrate_asymmetric_b_trigger import run_state_machine, windowed
    return [int(x) for x in run_state_machine(windowed(values, WINDOW), THRESHOLD, HYSTERESIS, DWELL)]


def _raw_state_machine(values: list[float]) -> list[int]:
    """The frozen state machine fed values that ARE the windowed statistic."""
    from experiments.calibrate_asymmetric_b_trigger import run_state_machine
    return [int(x) for x in run_state_machine(np.asarray(values, dtype=np.float64),
                                              THRESHOLD, HYSTERESIS, DWELL)]


def _online_sequence(values: list[float]) -> list[int]:
    r = OnlineRouter()
    return [1 if r.update(v) == TRIGGERED_COMPOSITION else 0 for v in values]


def run_contracts() -> dict:
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str, gating: bool = True, **data) -> None:
        checks.append({"name": name, "gating": gating, "passed": bool(passed),
                       "detail": detail, **data})

    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    record("C0_SPEC_FROZEN", spec.get("status") == "FROZEN_BEFORE_SEED_ALLOCATION",
           f"spec status = {spec.get('status')!r}")

    # -- C6: the operating point is the sealed one, byte for byte
    v2 = json.loads(V2_RESULT.read_text(encoding="utf-8"))
    sel = v2.get("selected_config") or {}
    ok = (v2.get("DECISION") == "CONSERVATIVE_B_TRIGGER_CALIBRATED_V2"
          and sel.get("window") == WINDOW and sel.get("hysteresis") == HYSTERESIS
          and sel.get("dwell") == DWELL and sel.get("threshold") == THRESHOLD)
    record("C6_CALIBRATION_PASS_SEALED", ok,
           f"V2 DECISION={v2.get('DECISION')!r}; selected={{W:{sel.get('window')}, "
           f"m:{sel.get('hysteresis')}, d:{sel.get('dwell')}, thr:{sel.get('threshold')!r}}}; "
           f"executor={{W:{WINDOW}, m:{HYSTERESIS}, d:{DWELL}, thr:{THRESHOLD!r}}}")

    # -- C1: no regime leak. Identical features => identical compositions.
    rng = np.random.default_rng(11)
    probe = list(rng.normal(-0.5, 1.5, 240))
    seq_a, seq_b = _online_sequence(probe), _online_sequence(probe)
    import inspect
    router_params = set(inspect.signature(OnlineRouter.update).parameters) - {"self"}
    leaky = {"pole", "genome", "opponent", "phase", "win", "score", "reward", "return"}
    record("C1_NO_REGIME_LEAK", seq_a == seq_b and not (router_params & leaky),
           f"identical feature sequence -> identical composition sequence ({seq_a == seq_b}); "
           f"router update() parameters = {sorted(router_params)}",
           n_ticks=len(probe))

    # -- C2: default is 2A_2D at t0 and under above-trigger evidence
    r = OnlineRouter()
    first = r.update(0.0)
    stay = all(OnlineRouter().update(v) == DEFAULT_COMPOSITION for v in (0.0, 1.0, 5.0, -0.5))
    high = _online_sequence([2.0] * 240)
    record("C2_DEFAULT_IS_2A2D", first == DEFAULT_COMPOSITION and stay and sum(high) == 0,
           f"tick0 composition = {first}; never triggers on high-D evidence (sum={sum(high)})")

    # -- C3: strong B evidence (LOW D) triggers, and dwell spaces the switches.
    #
    # An earlier draft of this contract asserted that the FIRST trigger must be
    # delayed to tick >= DWELL. That assertion was wrong, and the router was
    # right: the frozen state machine seeds last_switch = -10000 precisely so
    # the first switch is exempt from dwell, and the offline reference that
    # produced the V2 calibration triggers at tick 0 on this same fixture.
    # Dwell is a minimum spacing BETWEEN switches, not a warm-up before the
    # first one. The contract below tests that property instead, on both the
    # online router and the offline reference -- strictly more than the
    # original check, since it now also pins the tick-0 behaviour as intended
    # rather than merely tolerating it.
    low_feed = [-6.0] * 240
    low_on, low_off = _online_sequence(low_feed), _offline_sequence(low_feed)
    first_on = low_on.index(1) if 1 in low_on else None
    first_off = low_off.index(1) if 1 in low_off else None

    def _switch_gaps(seq: list[int]) -> list[int]:
        at = [t for t in range(1, len(seq)) if seq[t] != seq[t - 1]]
        return [b - a for a, b in zip(at, at[1:])]

    # Dwell spacing is a property of the state machine, so it has to be driven
    # at the level of the WINDOWED statistic. A raw oscillation cannot do it:
    # the production W=40 trailing mean smooths a fast oscillation to ~0, which
    # never re-crosses the trigger bound, so such a fixture produces one switch
    # and tests nothing. Driving with window=1 makes the fed values the windowed
    # statistic itself, at the real threshold / hysteresis / dwell, and lets both
    # implementations be compared on a sequence that genuinely wants to switch
    # more often than dwell allows.
    osc = [(-6.0 if (t // 3) % 2 == 0 else 6.0) for t in range(240)]
    osc_off = [int(x) for x in _raw_state_machine(osc)]
    r_osc = OnlineRouter(window=1)
    osc_on = [1 if r_osc.update(v) == TRIGGERED_COMPOSITION else 0 for v in osc]
    gaps_on, gaps_off = _switch_gaps(osc_on), _switch_gaps(osc_off)
    dwell_ok = (len(gaps_on) >= 4 and min(gaps_on) >= DWELL
                and gaps_on == gaps_off and osc_on == osc_off)
    record("C3_FORCED_B_EVIDENCE_TRIGGERS_AND_DWELL_SPACES_SWITCHES",
           sum(low_on) > 0 and first_on == first_off and low_on == low_off and dwell_ok,
           f"strong-B fixture at the production window: {sum(low_on)}/{len(low_on)} ticks "
           f"triggered, first trigger online={first_on} offline={first_off} (tick 0 is correct "
           f"-- dwell exempts the first switch by design); dwell-spacing fixture at window=1: "
           f"{len(gaps_on)} inter-switch gaps, min={min(gaps_on) if gaps_on else None} >= dwell "
           f"{DWELL}, online==offline={osc_on == osc_off}",
           min_switch_gap=(min(gaps_on) if gaps_on else None), n_switch_gaps=len(gaps_on))

    # -- C4: A-like evidence releases (or never leaves)
    release = _online_sequence([-6.0] * 120 + [6.0] * 120)
    released = release[-1] == 0 and 1 in release
    record("C4_FORCED_A_EVIDENCE_RELEASES", released,
           f"triggered then released: last tick state = {release[-1]}, "
           f"ticks triggered = {sum(release)}/{len(release)}")

    # -- C7: online router == offline calibration state machine, tick for tick
    by_ep: dict[tuple[str, int], list[tuple[int, float]]] = {}
    with V2_ROWS.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["split"] != "holdout":
                continue
            by_ep.setdefault((row["pole"], int(row["seed"])), []).append(
                (int(row["tick"]), float(row["p_blue"]) - float(row["p_red_r4"])))
    mismatched, compared_ticks = [], 0
    for key, pairs in sorted(by_ep.items()):
        vals = [v for _t, v in sorted(pairs)]
        on, off = _online_sequence(vals), _offline_sequence(vals)
        compared_ticks += len(vals)
        if on != off:
            mismatched.append(f"{key}:{sum(1 for a, b in zip(on, off) if a != b)}")
    record("C7_ONLINE_OFFLINE_ROUTER_EQUIVALENCE",
           bool(by_ep) and not mismatched and compared_ticks > 0,
           f"{len(by_ep)} held-out episodes, {compared_ticks} ticks compared, "
           f"mismatches: {mismatched or 'none'}",
           n_episodes=len(by_ep), n_ticks=compared_ticks)

    # -- C5: control parity. Pinned router == the sweep runner, bit for bit.
    parity_diffs = []
    for pole in POLES:
        for composition in (DEFAULT_COMPOSITION, TRIGGERED_COMPOSITION):
            seed = ORACLE_PARITY_SEEDS[0]
            pinned, _ = run_routed_episode(pole, seed, pinned=composition, label=composition)
            fixed, _ = run_fixed_episode(pole, composition, seed)
            diff = {k: [pinned.get(k), fixed.get(k)] for k in fixed
                    if pinned.get(k) != fixed.get(k)}
            if diff:
                parity_diffs.append({"pole": pole, "composition": composition, "diff": diff})
    record("C5_CONTROL_PARITY", not parity_diffs,
           f"4 pinned-vs-fixed episode comparisons, field-by-field; "
           f"disagreements: {parity_diffs or 'none'}", n_comparisons=4)

    # -- C8/C9: known-answer parity against the SEALED oracle rows
    sealed: dict[tuple[str, int], dict] = {}
    with ORACLE_ROWS.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["arm"] == "FIXED_2A2D" and int(row["seed"]) in ORACLE_PARITY_SEEDS:
                sealed[(row["pole"], int(row["seed"]))] = row
    anchor_diffs, anchor_rows, wins = [], 0, []
    for pole in POLES:
        for seed in ORACLE_PARITY_SEEDS:
            ref = sealed.get((pole, seed))
            if ref is None:
                anchor_diffs.append(f"{pole}/{seed}: missing from sealed rows")
                continue
            got, _ = run_fixed_episode(pole, DEFAULT_COMPOSITION, seed)
            anchor_rows += 1
            wins.append(int(got["blue_win"]))
            for fld in ("blue_score", "red_score", "blue_win", "steps"):
                if int(got[fld]) != int(ref[fld]):
                    anchor_diffs.append(f"{pole}/{seed}/{fld}: {got[fld]} != {ref[fld]}")
    record("C8_KNOWN_ANSWER_MEASUREMENT_PARITY",
           anchor_rows == 2 * len(ORACLE_PARITY_SEEDS) and not anchor_diffs,
           f"{anchor_rows} episodes re-run against sealed oracle FIXED_2A2D rows; "
           f"disagreements: {anchor_diffs or 'none'}", n_compared=anchor_rows)
    record("C9_TERMINAL_SCORING_NOT_RESET_STATE",
           anchor_rows > 0 and 0 < sum(wins) < len(wins),
           f"win rate on the parity subset = {sum(wins)}/{len(wins)} -- non-degenerate, "
           f"which post-auto-reset zeroed scores could not produce, and exact-matched above")

    # -- C10: PPO unreachable
    live_ppo = sorted(m for m in sys.modules if m.startswith(PPO_FORBIDDEN_PREFIXES))
    record("C10_NO_PPO_REACHABLE", not live_ppo,
           f"imported PPO/trainer modules: {live_ppo or 'none'}")

    # -- seed hygiene
    from experiments import seed_registry as sr
    free, msg = sr.check_block(SEED_BASE, SEED_BASE + SEED_N - 1, SEED_CLASS,
                               experiment_id=EXPERIMENT_ID)
    record("C11_OUTCOME_SEED_BLOCK_FREE", free, msg)

    gating = [c for c in checks if c["gating"]]
    passed = all(c["passed"] for c in gating)
    report = {
        "record_id": "ROUTED_COMPOSITION_OUTCOME_CONTRACT_RESULT",
        "utc": _now(),
        "implements": SPEC_PATH.name,
        "router_operating_point": {"statistic": "D_t = P_blue(4.0) - P_red(4.0)",
                                   "window": WINDOW, "hysteresis": HYSTERESIS,
                                   "dwell": DWELL, "threshold": THRESHOLD},
        "n_checks": len(checks), "n_gating": len(gating),
        "n_failed": sum(1 for c in gating if not c["passed"]),
        "checks": checks,
        "DECISION": "CONTRACTS_PASS" if passed else "CONTRACT_FAILURE",
        "contract_amendment_disclosed": {
            "what_happened": "The first execution of this contract stage returned CONTRACT_FAILURE on C3, "
                             "which asserted that the router's FIRST trigger must be delayed to tick >= dwell. "
                             "No outcome seed was allocated or spent.",
            "diagnosis": "The assertion was wrong; the router was right. The frozen state machine seeds "
                         "last_switch = -10000 so the first switch is exempt from dwell, and the offline "
                         "reference that produced the sealed V2 calibration triggers at tick 0 on the same "
                         "fixture. Verified directly against experiments/calibrate_asymmetric_b_trigger.py::"
                         "run_state_machine before any edit. Dwell is a minimum spacing BETWEEN switches.",
            "resolution": "C3 was replaced by a strictly stronger contract: strong-B evidence must trigger, "
                          "the online first-trigger tick must equal the offline reference's, the full "
                          "sequences must be identical, and on a fixture that wants to switch more often "
                          "than dwell allows, no two switches may be closer than dwell in either "
                          "implementation. The router was NOT modified; changing it would have changed the "
                          "calibrated operating point and invalidated the V2 held-out result.",
            "second_failure_also_disclosed": {
                "what_happened": "The rewritten C3 failed again on its second execution, reporting 0 "
                                 "inter-switch gaps. Again no outcome seed was allocated or spent.",
                "diagnosis": "The dwell-spacing FIXTURE was at fault, not the router and not the assertion. "
                             "A 2-tick raw oscillation is smoothed by the production W=40 trailing mean to "
                             "exactly 0.0, which never re-crosses the trigger bound, so the fixture produced "
                             "a single switch and could not exercise dwell at all. Verified by printing the "
                             "post-warmup windowed range (0.0000 .. 0.0000).",
                "resolution": "Dwell spacing is a property of the state machine, so it is now driven at the "
                              "level of the windowed statistic (window=1, so the fed values ARE that "
                              "statistic) at the real threshold / hysteresis / dwell. Both implementations "
                              "are compared on a sequence that genuinely wants to switch more often than "
                              "dwell permits, and the contract additionally requires at least 4 gaps so it "
                              "cannot pass vacuously on a fixture that produces none.",
            },
            "superseded_record": "the two earlier ROUTED_COMPOSITION_OUTCOME_CONTRACT_RESULT.json writes, overwritten by this one",
        },
        "smoke_block_used": None,
        "smoke_note": "No fresh smoke episodes were required: C5 and C8 exercise the full "
                      "episode path on already-spent oracle seeds, and C7 replays persisted "
                      "calibration rows. 99900601-616 remains unallocated.",
        "claim_boundary": "Contracts only. No outcome seed was allocated or spent by this stage.",
    }
    CONTRACT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'=' * 66}\n  CONTRACTS: {report['DECISION']}  "
          f"({report['n_failed']}/{report['n_gating']} gating failed)\n{'=' * 66}")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print(f"  -> {CONTRACT_PATH}")
    return report


# ------------------------------------------------------------------ outcome

def _bootstrap(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"mean": None, "lcb95": None, "ucb95": None, "n": 0}
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, values.size, size=(N_BOOT, values.size))
    means = values[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return {"mean": round(float(values.mean()), 6), "lcb95": round(float(lo), 6),
            "ucb95": round(float(hi), 6), "n": int(values.size)}


def _paired(rows: dict, pole: str, left: str, right: str, field: str = "blue_win") -> np.ndarray:
    return np.asarray([float(rows[(pole, left, s)][field]) - float(rows[(pole, right, s)][field])
                       for s in OUTCOME_SEEDS])


def run_outcome() -> int:
    from experiments import run_state as rs
    from experiments import seed_registry as sr

    contracts = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if contracts.get("DECISION") != "CONTRACTS_PASS":
        raise SystemExit(f"REFUSING: contracts did not pass ({contracts.get('DECISION')!r})")
    if RESULT_PATH.exists():
        raise SystemExit(f"REFUSING: result already exists: {RESULT_PATH}")

    sr.allocate(EXPERIMENT_ID, SEED_BASE, SEED_BASE + SEED_N - 1, SEED_CLASS,
                purpose="routed composition outcome, three arms, paired, 4v4, PPO off",
                spec=SPEC_PATH.name)
    state = rs.RunState(SD, LABEL).begin(spec=SPEC_PATH.name, n_episodes=len(POLES) * len(ARMS) * SEED_N)

    rows: dict[tuple[str, str, int], dict] = {}
    mappings: list[dict] = []
    total = len(POLES) * len(ARMS) * SEED_N
    done_n = 0
    for pole in POLES:
        for arm in ARMS:
            for seed in OUTCOME_SEEDS:
                row, mapping = run_job(pole, arm, seed)
                rows[(pole, arm, seed)] = row
                mappings.append(mapping)
                done_n += 1
                if done_n % 16 == 0 or done_n == total:
                    print(f"  {done_n}/{total} episodes", flush=True)

    fields = list(next(iter(rows.values())).keys())
    with EPISODE_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for key in sorted(rows, key=lambda k: (k[0], k[1], k[2])):
            w.writerow(rows[key])

    b_gain = _bootstrap(_paired(rows, "B", "STATE_B_TRIGGER", "FIXED_2A2D"))
    a_harm = _bootstrap(_paired(rows, "A", "FIXED_2A2D", "STATE_B_TRIGGER"))
    win_rates = {f"{pole}_{arm}": round(float(np.mean([rows[(pole, arm, s)]["blue_win"]
                                                       for s in OUTCOME_SEEDS])), 6)
                 for pole in POLES for arm in ARMS}

    pass_b = b_gain["lcb95"] is not None and b_gain["lcb95"] > 0
    pass_a = a_harm["ucb95"] is not None and a_harm["ucb95"] <= TAU_A_HARM
    if not pass_b:
        decision = "NO_DEMONSTRATED_B_OUTCOME_GAIN"
    elif pass_a:
        decision = "ROUTED_COMPOSITION_OUTCOME_PASS"
    else:
        decision = "B_GAIN_WITH_EXCESS_A_HARM"

    ceiling_b = win_rates["B_FIXED_4A0D"] - win_rates["B_FIXED_2A2D"]
    floor_a = win_rates["A_FIXED_2A2D"] - win_rates["A_FIXED_4A0D"]
    telemetry = {
        f"{pole}_router_tick_fraction_4A0D": round(float(np.mean(
            [m["router_tick_fraction_4A0D"] for m in mappings
             if m["arm"] == "STATE_B_TRIGGER" and m["pole"] == pole])), 6)
        for pole in POLES}
    telemetry.update({
        f"{pole}_router_switches_per_episode": round(float(np.mean(
            [m["router_switches"] for m in mappings
             if m["arm"] == "STATE_B_TRIGGER" and m["pole"] == pole])), 6)
        for pole in POLES})

    payload = {
        "record_id": "ROUTED_COMPOSITION_OUTCOME_RESULT",
        "implements": SPEC_PATH.name,
        "utc": _now(),
        "arms": list(ARMS), "poles": list(POLES),
        "seed_block": {"base": SEED_BASE, "last": SEED_BASE + SEED_N - 1, "n": SEED_N,
                       "class": SEED_CLASS},
        "router_operating_point": {"statistic": "D_t = P_blue(4.0) - P_red(4.0)",
                                   "window": WINDOW, "hysteresis": HYSTERESIS,
                                   "dwell": DWELL, "threshold": THRESHOLD,
                                   "retuned": False},
        "contracts": {"path": CONTRACT_PATH.name, "decision": contracts["DECISION"],
                      "n_gating": contracts["n_gating"], "n_failed": contracts["n_failed"]},
        "primary_gates": {
            "B_improvement_paired_STATE_minus_FIXED_2A2D_on_pole_B": b_gain,
            "B_gate_LCB95>0": pass_b,
            "A_harm_paired_FIXED_2A2D_minus_STATE_on_pole_A": a_harm,
            "tau_A_harm": TAU_A_HARM,
            "A_gate_UCB95<=tau": pass_a,
        },
        "win_rates": win_rates,
        "descriptive_not_gating": {
            "FIXED_4A0D_is_reference_only": True,
            "B_ceiling_4A0D_minus_2A2D": round(float(ceiling_b), 6),
            "A_floor_2A2D_minus_4A0D": round(float(floor_a), 6),
            "B_capture_fraction": (round(float((win_rates["B_STATE_B_TRIGGER"]
                                                - win_rates["B_FIXED_2A2D"]) / ceiling_b), 6)
                                   if abs(ceiling_b) > 1e-12 else None),
            "A_damage_avoided_fraction": (round(float((win_rates["A_STATE_B_TRIGGER"]
                                                       - win_rates["A_FIXED_4A0D"]) / floor_a), 6)
                                          if abs(floor_a) > 1e-12 else None),
            "ratio_caveat": "Ratios of noisy estimates; quote with the numerator and denominator intervals, never alone.",
            "router_telemetry": telemetry,
        },
        "router_calibration_caveat_context_only": (
            "The router's held-out Pole-A one-sided episode UCB95 was 0.1026, marginally above "
            "the 0.10 calibration limit (P(true A_FP <= 0.10) about 0.937). Promisingly "
            "conservative, not proven safe. Context only; not a gate here and it does not "
            "rewrite the V2 router verdict."),
        "DECISION": decision,
        "claim_boundary": ("Two frozen opponents, 4v4, PPO off, n=64 paired seeds. Does not "
                           "authorize PPO, learned coordination, or 6v6."),
    }

    plan = rs.AuditPlan(
        rows_csv=EPISODE_CSV, expected_rows=total, expected_seeds=OUTCOME_SEEDS,
        # one audit CELL is (pole, arm); each must carry the whole frozen seed
        # block exactly once, which is what makes the paired claims seed-matched
        group_by=("pole", "arm"), seed_field="seed",
        int_fields=("seed", "blue_score", "red_score", "blue_win", "draw", "steps"),
        binary_fields=("blue_win", "draw"),
        derived={"blue_win": rs.Derived("blue_win == int(blue_score > red_score)",
                                        lambda r: int(int(r["blue_score"]) > int(r["red_score"]))),
                 "draw": rs.Derived("draw == int(blue_score == red_score)",
                                    lambda r: int(int(r["blue_score"]) == int(r["red_score"])))},
        spec_path=SPEC_PATH, seed_class=SEED_CLASS, n_boot=N_BOOT, alpha=ALPHA, rng_seed=RNG_SEED,
        claims=(
            rs.Claim(name="B_improvement_STATE_minus_FIXED_2A2D_pole_B", recorded=b_gain,
                     minuend={"pole": "B", "arm": "STATE_B_TRIGGER"},
                     subtrahend={"pole": "B", "arm": "FIXED_2A2D"}, value_field="blue_win"),
            rs.Claim(name="A_harm_FIXED_2A2D_minus_STATE_pole_A", recorded=a_harm,
                     minuend={"pole": "A", "arm": "FIXED_2A2D"},
                     subtrahend={"pole": "A", "arm": "STATE_B_TRIGGER"}, value_field="blue_win"),
        ),
    )
    rs.seal(out_path=RESULT_PATH, payload=payload, plan=plan, state=state, strict=False)
    sealed = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    sr.set_status(EXPERIMENT_ID, "SPENT", note=f"sealed {sealed.get('status')}; {decision}")
    print(json.dumps({"DECISION": decision, "status": sealed.get("status"),
                      "B_gain": b_gain, "A_harm": a_harm, "win_rates": win_rates}, indent=2))
    return 0 if decision == "ROUTED_COMPOSITION_OUTCOME_PASS" else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("contracts", "outcome"), required=True)
    args = ap.parse_args()
    if args.stage == "contracts":
        return 0 if run_contracts()["DECISION"] == "CONTRACTS_PASS" else 2
    return run_outcome()


if __name__ == "__main__":
    raise SystemExit(main())
