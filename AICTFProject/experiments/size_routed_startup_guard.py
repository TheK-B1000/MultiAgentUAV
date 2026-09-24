"""DESCRIPTIVE SIZING ONLY -- ROUTED_COMPOSITION_STARTUP_GUARD_V1_SPEC.json.

Closed-loop re-simulation of the ALREADY-SPENT outcome block 20200001-20200064
with the frozen startup guard active, to decide whether a fresh confirmatory
experiment is worth funding.

NOT EVIDENCE. Not a gate, not a promotion, not a scientific claim. Every headline
number carries the optimism disclosure: these are the same 64 seeds that revealed
the tick-0 defect, so a rule targeting that locus will look better here than on
fresh seeds even though nothing was tuned.

No new seeds. No knob is varied: exactly one guard, the frozen evidence-completeness
rule, is simulated. No alternative guard length is compared.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.localize_routed_a_harm import _runs_of_ones  # noqa: E402
from experiments.run_pyquaticus_4v4_role_composition_sweep import (  # noqa: E402
    _action_for_roles,
    _ids,
    composition_roles,
)
from experiments.run_pyquaticus_4v4_team_evaluation import (  # noqa: E402
    HORIZON,
    _make_env,
    _telemetry_for_tick,
)
from experiments.run_routed_composition_outcome import (  # noqa: E402
    DEFAULT_COMPOSITION,
    EPISODE_CSV,
    OUTCOME_SEEDS,
    TRIGGERED_COMPOSITION,
    OnlineRouter,
    _bootstrap,
    observe_d,
)

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC = SD / "ROUTED_COMPOSITION_STARTUP_GUARD_V1_SPEC.json"
PASS2_TICKS = SD / "routed_composition_switch_safety_ticks.csv"
OUT = SD / "ROUTED_COMPOSITION_STARTUP_GUARD_SIZING.json"
ROWS_CSV = SD / "routed_composition_startup_guard_sizing_episodes.csv"
POLES = ("A", "B")

OPTIMISM = ("DESCRIPTIVE SIZING ONLY. Same 64 seeds that revealed the tick-0 defect. The guard "
            "rule was not fitted -- it has no tunable parameter -- but the MECHANISM it targets "
            "was identified on these episodes, so these numbers are optimistic by an unknown "
            "amount. Planning estimate, not a prediction, not evidence.")


class GuardedRouter(OnlineRouter):
    """The frozen router plus the frozen startup guard.

    Mirrors OnlineRouter.update exactly, with ONE added precondition: a
    departure from the default composition requires a full evidence window.
    Returns to the default are never blocked -- the guard protects leaving the
    safe state, not re-entering it.

    With guard=False this must reproduce OnlineRouter bit-for-bit; that is
    checked against real traces before any sizing number is produced.
    """

    def __init__(self, *args: Any, guard: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.guard = bool(guard)
        self.blocked_departures = 0
        self.first_block_tick: int | None = None

    def update(self, d_value: float) -> str:
        self._t += 1
        self._buf.append(float(d_value))
        v = sum(self._buf) / len(self._buf)
        self.windowed = v
        if self.pinned is None:
            elapsed = self._t - self._last_switch
            window_full = len(self._buf) >= self.window
            if self._state == 0 and v < self.threshold - self.hysteresis and elapsed >= self.dwell:
                if window_full or not self.guard:
                    self._state, self._last_switch = 1, self._t
                    self.switches += 1
                else:
                    self.blocked_departures += 1
                    if self.first_block_tick is None:
                        self.first_block_tick = self._t
            elif self._state == 1 and v > self.threshold + self.hysteresis and elapsed >= self.dwell:
                self._state, self._last_switch = 0, self._t
                self.switches += 1
        return self.composition


def _equivalence_check() -> dict[str, Any]:
    """GuardedRouter(guard=False) must equal OnlineRouter on real D_t traces.
    Without this, the subclass could have drifted and the 'guard on' arm would
    differ from the frozen router for reasons other than the guard."""
    if not PASS2_TICKS.is_file():
        raise SystemExit(f"ABORT: pass-2 traces missing: {PASS2_TICKS}. Run "
                         "experiments/contrast_routed_switch_safety.py first.")
    series: dict[tuple[str, int], list[tuple[int, float]]] = {}
    with PASS2_TICKS.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            series.setdefault((row["pole"], int(row["seed"])), []).append(
                (int(row["tick"]), float(row["d"])))
    mismatch, n_ticks = [], 0
    for key, pairs in sorted(series.items()):
        vals = [v for _t, v in sorted(pairs)]
        a = OnlineRouter()
        b = GuardedRouter(guard=False)
        sa = [a.update(v) for v in vals]
        sb = [b.update(v) for v in vals]
        n_ticks += len(vals)
        if sa != sb or a.switches != b.switches:
            mismatch.append(str(key))
    if mismatch:
        raise SystemExit(f"ABORT: GuardedRouter(guard=False) diverges from the frozen router: "
                         f"{mismatch[:5]}")
    return {"episodes": len(series), "ticks": n_ticks, "mismatches": 0,
            "detail": "GuardedRouter(guard=False) reproduces OnlineRouter exactly on real traces"}


def run_guarded_episode(pole: str, seed: int) -> tuple[dict, dict]:
    env, core, genome, live = _make_env(pole, seed)
    try:
        router = GuardedRouter(guard=True)
        counters = {"attack_ticks": 0, "defend_ticks": 0,
                    "attack_enemy_flag_branch_count": 0, "carrier_home_branch_count": 0,
                    "defend_inward_count": 0, "defend_outward_count": 0,
                    "tagged_ticks_by_role": 0}
        flags: list[int] = []
        terminal_info, steps = None, 0
        first_roles = None
        for _ in range(HORIZON):
            composition = router.update(observe_d(core))
            roles = composition_roles(composition)
            if first_roles is None:
                first_roles = roles
            flags.append(int(composition == TRIGGERED_COMPOSITION))
            _telemetry_for_tick(core, roles, counters)
            env.step_async(_action_for_roles(core, roles))
            _obs, _rew, done, infos = env.step_wait()
            steps += 1
            if bool(np.asarray(done).any()):
                terminal_info = dict(infos[0])
                break
        if terminal_info is None:
            terminal_info = {"episode_result": {"blue_score": int(core.blue_score[0].item()),
                                                "red_score": int(core.red_score[0].item())}}
        r = dict(terminal_info.get("episode_result") or {})
        b, rd = int(r.get("blue_score", 0)), int(r.get("red_score", 0))
        attackers, defenders = _ids(first_roles or composition_roles(DEFAULT_COMPOSITION))
        bursts = _runs_of_ones(flags)
        row = {"seed": int(seed), "pole": pole, "arm": "STATE_GUARDED",
               "composition": "ROUTED_GUARDED",
               "defender_ids": ",".join(map(str, defenders)),
               "attacker_ids": ",".join(map(str, attackers)),
               "blue_score": b, "red_score": rd, "blue_win": int(b > rd),
               "draw": int(b == rd), "steps": steps,
               "blue_alive_end": None, "red_alive_end": None,
               "genome_id": str(genome.genome_id),
               "pole_config_hash": str(live.get("live_config_hash", "")),
               **counters, "role_switch_count": int(router.switches)}
        diag = {"pole": pole, "seed": int(seed),
                "blocked_departures": int(router.blocked_departures),
                "first_block_tick": router.first_block_tick,
                "n_bursts": len(bursts),
                "first_onset": (bursts[0][0] if bursts else None),
                "ticks_in_4A0D": int(sum(flags)),
                "fraction_in_4A0D": float(sum(flags)) / max(steps, 1)}
        return row, diag
    finally:
        env.close()


def main() -> int:
    if not SPEC.is_file():
        raise SystemExit(f"REFUSING: frozen guard spec missing: {SPEC}")
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec.get("status") != "FROZEN_BEFORE_SIZING":
        raise SystemExit(f"REFUSING: guard spec not frozen: {spec.get('status')!r}")

    print("Checking GuardedRouter(guard=False) == frozen OnlineRouter on real traces...", flush=True)
    equiv = _equivalence_check()
    print(f"  equivalence PASS ({equiv['episodes']} episodes, {equiv['ticks']} ticks)", flush=True)

    sealed: dict[tuple[str, str, int], dict] = {}
    with EPISODE_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sealed[(row["pole"], row["arm"], int(row["seed"]))] = row
    if not sealed:
        raise SystemExit("ABORT: sealed outcome rows missing -- absence is an error state")

    print("Re-simulating both poles with the guard ACTIVE (no new seeds)...", flush=True)
    rows: dict[tuple[str, int], dict] = {}
    diags: dict[tuple[str, int], dict] = {}
    for pole in POLES:
        for i, seed in enumerate(OUTCOME_SEEDS, 1):
            row, diag = run_guarded_episode(pole, seed)
            rows[(pole, seed)] = row
            diags[(pole, seed)] = diag
            if i % 16 == 0:
                print(f"  {pole} {i}/{len(OUTCOME_SEEDS)}", flush=True)

    with ROWS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(next(iter(rows.values()))))
        w.writeheader()
        for k in sorted(rows, key=lambda x: (x[0], x[1])):
            w.writerow(rows[k])

    def wr(pole: str, arm: str) -> float:
        if arm == "STATE_GUARDED":
            return float(np.mean([rows[(pole, s)]["blue_win"] for s in OUTCOME_SEEDS]))
        return float(np.mean([int(sealed[(pole, arm, s)]["blue_win"]) for s in OUTCOME_SEEDS]))

    def paired(pole: str, left: str, right: str) -> np.ndarray:
        def val(arm: str, s: int) -> float:
            return (float(rows[(pole, s)]["blue_win"]) if arm == "STATE_GUARDED"
                    else float(sealed[(pole, arm, s)]["blue_win"]))
        return np.asarray([val(left, s) - val(right, s) for s in OUTCOME_SEEDS])

    b_gain_guarded = _bootstrap(paired("B", "STATE_GUARDED", "FIXED_2A2D"))
    a_harm_guarded = _bootstrap(paired("A", "FIXED_2A2D", "STATE_GUARDED"))
    b_gain_unguarded = _bootstrap(paired("B", "STATE_B_TRIGGER", "FIXED_2A2D"))
    a_harm_unguarded = _bootstrap(paired("A", "FIXED_2A2D", "STATE_B_TRIGGER"))
    b_delta = _bootstrap(paired("B", "STATE_GUARDED", "STATE_B_TRIGGER"))
    a_delta = _bootstrap(paired("A", "STATE_GUARDED", "STATE_B_TRIGGER"))

    def occupancy(pole: str) -> dict[str, Any]:
        g = [diags[(pole, s)] for s in OUTCOME_SEEDS]
        onsets = [d["first_onset"] for d in g if d["first_onset"] is not None]
        return {
            "guarded_tick_fraction_4A0D": round(float(np.mean([d["fraction_in_4A0D"] for d in g])), 6),
            "guarded_episodes_with_any_trigger": int(sum(1 for d in g if d["n_bursts"] > 0)),
            "guarded_switches_per_episode": round(float(np.mean([rows[(pole, s)]["role_switch_count"]
                                                                 for s in OUTCOME_SEEDS])), 6),
            "guarded_first_onset_mean": (round(float(np.mean(onsets)), 2) if onsets else None),
            "guarded_first_onset_min": (int(min(onsets)) if onsets else None),
            "episodes_with_a_blocked_departure": int(sum(1 for d in g if d["blocked_departures"] > 0)),
            "blocked_then_triggered_later": int(sum(1 for d in g if d["blocked_departures"] > 0
                                                    and d["n_bursts"] > 0)),
            "blocked_and_never_triggered": int(sum(1 for d in g if d["blocked_departures"] > 0
                                                   and d["n_bursts"] == 0)),
        }

    report = {
        "record_id": "ROUTED_COMPOSITION_STARTUP_GUARD_SIZING",
        "label": "DESCRIPTIVE SIZING ONLY",
        "implements": SPEC.name,
        "NOT_EVIDENCE": OPTIMISM,
        "declares_no_pass_fail": True,
        "no_new_seeds": "re-simulation of the already-spent block 20200001-20200064",
        "no_knob_varied": "exactly one guard simulated; no alternative guard length compared",
        "controls": {
            "subclass_equivalence": equiv,
            "unguarded_baseline_determinism": ("verified by ROUTED_COMPOSITION_SWITCH_SAFETY_CONTRAST, "
                                               "which re-simulated both poles unguarded and gated "
                                               "every episode against the sealed rows"),
            "fixed_arms": "read from the sealed rows, not re-simulated",
        },
        "S1_pole_A_harm": {
            "unguarded": a_harm_unguarded, "guarded": a_harm_guarded,
            "guarded_minus_unguarded_win_rate_on_A": a_delta,
            "frozen_tolerance_for_reference_only": 0.10,
        },
        "S2_pole_B_gain": {
            "unguarded": b_gain_unguarded, "guarded": b_gain_guarded,
            "guarded_minus_unguarded_win_rate_on_B": b_delta,
        },
        "win_rates": {
            pole: {"FIXED_2A2D": round(wr(pole, "FIXED_2A2D"), 6),
                   "FIXED_4A0D": round(wr(pole, "FIXED_4A0D"), 6),
                   "STATE_unguarded": round(wr(pole, "STATE_B_TRIGGER"), 6),
                   "STATE_guarded": round(wr(pole, "STATE_GUARDED"), 6)}
            for pole in POLES},
        "S3_occupancy_and_timing": {pole: occupancy(pole) for pole in POLES},
        "S4_does_later_switching_compensate": {
            pole: {
                "episodes_whose_first_departure_was_blocked":
                    occupancy(pole)["episodes_with_a_blocked_departure"],
                "of_those_triggered_later": occupancy(pole)["blocked_then_triggered_later"],
                "of_those_never_triggered": occupancy(pole)["blocked_and_never_triggered"],
            } for pole in POLES},
        "claim_boundary": ("Planning estimate on one already-spent block. The guard's outcome claim "
                           "requires a fresh sealed_confirmatory block under the unchanged gates."),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 70)
    print("  DESCRIPTIVE SIZING ONLY -- not evidence, no pass/fail declared")
    print("=" * 70)
    for pole in POLES:
        w = report["win_rates"][pole]
        print(f"  pole {pole}: 2A2D={w['FIXED_2A2D']:.4f}  4A0D={w['FIXED_4A0D']:.4f}  "
              f"STATE={w['STATE_unguarded']:.4f}  STATE+guard={w['STATE_guarded']:.4f}")
    print(f"\n  S1 A harm   unguarded {a_harm_unguarded['mean']:+.4f} "
          f"[{a_harm_unguarded['lcb95']:+.4f},{a_harm_unguarded['ucb95']:+.4f}]"
          f"  ->  guarded {a_harm_guarded['mean']:+.4f} "
          f"[{a_harm_guarded['lcb95']:+.4f},{a_harm_guarded['ucb95']:+.4f}]")
    print(f"  S2 B gain   unguarded {b_gain_unguarded['mean']:+.4f} "
          f"[{b_gain_unguarded['lcb95']:+.4f},{b_gain_unguarded['ucb95']:+.4f}]"
          f"  ->  guarded {b_gain_guarded['mean']:+.4f} "
          f"[{b_gain_guarded['lcb95']:+.4f},{b_gain_guarded['ucb95']:+.4f}]")
    print(f"     paired guarded-minus-unguarded: A {a_delta['mean']:+.4f} "
          f"[{a_delta['lcb95']:+.4f},{a_delta['ucb95']:+.4f}] | "
          f"B {b_delta['mean']:+.4f} [{b_delta['lcb95']:+.4f},{b_delta['ucb95']:+.4f}]")
    for pole in POLES:
        o = report["S3_occupancy_and_timing"][pole]
        print(f"  S3 pole {pole}: tick fraction 4A0D {o['guarded_tick_fraction_4A0D']:.4f}, "
              f"triggered {o['guarded_episodes_with_any_trigger']}/64, "
              f"first onset mean {o['guarded_first_onset_mean']}, min {o['guarded_first_onset_min']}")
        s4 = report["S4_does_later_switching_compensate"][pole]
        print(f"  S4 pole {pole}: blocked departures in {s4['episodes_whose_first_departure_was_blocked']} "
              f"episodes -> triggered later {s4['of_those_triggered_later']}, "
              f"never {s4['of_those_never_triggered']}")
    print(f"\n  {OPTIMISM}")
    print(f"\n-> {OUT}\n-> {ROWS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
