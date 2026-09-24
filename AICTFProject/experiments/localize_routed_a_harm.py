"""Read-only A-harm localization on the already-spent ROUTED_COMPOSITION_OUTCOME episodes.

DIAGNOSTIC. Post-hoc. Descriptive. Gates nothing, decides nothing, and does not
change ROUTED_COMPOSITION_OUTCOME_RESULT.json (which stays AUDIT_FAILED) or the
B_GAIN_WITH_EXCESS_A_HARM verdict.

No new seeds. The Pole-A STATE_B_TRIGGER episodes of block 20200001-20200064 are
re-simulated deterministically to recover per-tick router traces the outcome CSV
did not persist. A hard determinism gate requires the re-run to reproduce the
sealed rows exactly -- otherwise the traces describe a different episode than the
one the verdict was computed from.

No knob is searched. No threshold, window, hysteresis or dwell is varied.

The four questions below are declared here, before the numbers, and they are the
PI's list verbatim. They are hypothesis-generating; none is a test.
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

from experiments.run_pyquaticus_4v4_role_composition_sweep import (  # noqa: E402
    _action_for_roles,
    composition_roles,
)
from experiments.run_pyquaticus_4v4_team_evaluation import (  # noqa: E402
    HORIZON,
    _make_env,
    _telemetry_for_tick,
)
from experiments.run_routed_composition_outcome import (  # noqa: E402
    EPISODE_CSV,
    OUTCOME_SEEDS,
    TRIGGERED_COMPOSITION,
    OnlineRouter,
    observe_d,
)

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
OUT = SD / "ROUTED_COMPOSITION_A_HARM_LOCALIZATION.json"
TICKS_CSV = SD / "routed_composition_a_harm_ticks.csv"

QUESTIONS = [
    "Q1 when do the Pole-A router triggers occur",
    "Q2 how long do the 4A_0D bursts last",
    "Q3 what game state precedes a trigger",
    "Q4 are losses concentrated in episodes with early / long / particular 4A_0D excursions",
]


def trace_pole_a_state_episode(seed: int) -> tuple[dict, list[dict]]:
    """Re-simulate one Pole-A STATE_B_TRIGGER episode, recording per-tick state.

    The action path is identical to run_routed_composition_outcome.run_routed_episode;
    only the recording is added, so the trajectory is bit-identical.
    """
    env, core, genome, live = _make_env("A", seed)
    try:
        router = OnlineRouter()
        counters = {"attack_ticks": 0, "defend_ticks": 0,
                    "attack_enemy_flag_branch_count": 0, "carrier_home_branch_count": 0,
                    "defend_inward_count": 0, "defend_outward_count": 0,
                    "tagged_ticks_by_role": 0}
        ticks: list[dict] = []
        terminal_info, steps = None, 0
        for _ in range(HORIZON):
            d_value = observe_d(core)
            composition = router.update(d_value)
            roles = composition_roles(composition)
            ticks.append({
                "seed": seed, "tick": steps, "d": d_value, "windowed": router.windowed,
                "state_4A0D": int(composition == TRIGGERED_COMPOSITION),
                "blue_score": int(core.blue_score[0].item()),
                "red_score": int(core.red_score[0].item()),
                "blue_carrying": int(bool(core.blue_carrying[0].any().item())),
                "blue_alive": int(core.blue_alive[0].sum().item()),
                "red_alive": int(core.red_alive[0].sum().item()),
                "blue_tagged": int(core.blue_tagged[0].sum().item()),
            })
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
        return ({"seed": seed, "blue_score": b, "red_score": rd,
                 "blue_win": int(b > rd), "steps": steps,
                 "switches": int(router.switches)}, ticks)
    finally:
        env.close()


def _runs_of_ones(flags: list[int]) -> list[tuple[int, int]]:
    """(start_tick, length) for every maximal 4A_0D burst."""
    out, start = [], None
    for t, f in enumerate(flags):
        if f and start is None:
            start = t
        elif not f and start is not None:
            out.append((start, t - start))
            start = None
    if start is not None:
        out.append((start, len(flags) - start))
    return out


def _describe(name: str, a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Descriptive contrast with a bootstrap interval. Not a test; no p-value."""
    if a.size == 0 or b.size == 0:
        return {"metric": name, "n_loss": int(a.size), "n_other": int(b.size),
                "note": "one group empty; not computed"}
    rng = np.random.default_rng(7)
    diffs = (a[rng.integers(0, a.size, size=(20000, a.size))].mean(axis=1)
             - b[rng.integers(0, b.size, size=(20000, b.size))].mean(axis=1))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"metric": name, "loss_mean": round(float(a.mean()), 4),
            "other_mean": round(float(b.mean()), 4),
            "difference": round(float(a.mean() - b.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "n_loss": int(a.size), "n_other": int(b.size)}


def main() -> int:
    sealed: dict[tuple[str, str, int], dict] = {}
    with EPISODE_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sealed[(row["pole"], row["arm"], int(row["seed"]))] = row
    if not sealed:
        raise SystemExit("ABORT: outcome rows missing -- absence is an error state")

    print("Re-simulating Pole-A STATE_B_TRIGGER episodes (no new seeds)...", flush=True)
    per_ep: dict[int, dict] = {}
    all_ticks: list[dict] = []
    drift: list[str] = []
    for i, seed in enumerate(OUTCOME_SEEDS, 1):
        summary, ticks = trace_pole_a_state_episode(seed)
        ref = sealed[("A", "STATE_B_TRIGGER", seed)]
        for fld in ("blue_score", "red_score", "blue_win", "steps"):
            if int(summary[fld]) != int(ref[fld]):
                drift.append(f"{seed}/{fld}: {summary[fld]} != {ref[fld]}")
        if int(summary["switches"]) != int(ref["role_switch_count"]):
            drift.append(f"{seed}/switches: {summary['switches']} != {ref['role_switch_count']}")
        per_ep[seed] = {"summary": summary, "flags": [t["state_4A0D"] for t in ticks],
                        "ticks": ticks}
        all_ticks.extend(ticks)
        if i % 16 == 0:
            print(f"  {i}/{len(OUTCOME_SEEDS)}", flush=True)

    if drift:
        raise SystemExit("ABORT: re-simulation does not reproduce the sealed rows; the traces "
                         f"would describe a different episode. {drift[:5]}")
    print(f"  determinism gate PASS: {len(OUTCOME_SEEDS)}/{len(OUTCOME_SEEDS)} episodes "
          f"reproduce the sealed rows exactly (scores, wins, steps, switch counts)", flush=True)

    with TICKS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_ticks[0]))
        w.writeheader()
        w.writerows(all_ticks)

    # ---- per-episode trigger features -------------------------------------
    feats: dict[int, dict] = {}
    for seed, ep in per_ep.items():
        flags, ticks = ep["flags"], ep["ticks"]
        bursts = _runs_of_ones(flags)
        onsets = [s for s, _L in bursts]
        pre = []
        for s, _L in bursts:
            t = ticks[max(s - 1, 0)]
            pre.append({"score_diff": t["blue_score"] - t["red_score"],
                        "carrying": t["blue_carrying"], "red_alive": t["red_alive"],
                        "blue_alive": t["blue_alive"], "tagged": t["blue_tagged"]})
        feats[seed] = {
            "n_bursts": len(bursts),
            "ticks_in_4A0D": int(sum(flags)),
            "fraction_in_4A0D": float(sum(flags)) / max(len(flags), 1),
            "first_onset": (min(onsets) if onsets else None),
            "longest_burst": (max(L for _s, L in bursts) if bursts else 0),
            "mean_burst": (float(np.mean([L for _s, L in bursts])) if bursts else 0.0),
            "pre_trigger": pre,
            "win_state": int(ep["summary"]["blue_win"]),
            "win_fixed_2a2d": int(sealed[("A", "FIXED_2A2D", seed)]["blue_win"]),
        }

    triggered = [s for s in OUTCOME_SEEDS if feats[s]["n_bursts"] > 0]
    loss = [s for s in triggered if feats[s]["win_fixed_2a2d"] == 1 and feats[s]["win_state"] == 0]
    other = [s for s in triggered if s not in loss]
    gain = [s for s in triggered if feats[s]["win_fixed_2a2d"] == 0 and feats[s]["win_state"] == 1]

    def arr(seeds: list[int], key: str) -> np.ndarray:
        return np.asarray([float(feats[s][key]) for s in seeds], dtype=np.float64)

    contrasts = [_describe(k, arr(loss, k), arr(other, k))
                 for k in ("first_onset", "longest_burst", "mean_burst",
                           "ticks_in_4A0D", "fraction_in_4A0D", "n_bursts")]

    all_bursts = [(s, L) for seed in triggered for s, L in
                  _runs_of_ones(per_ep[seed]["flags"])]
    lens = np.asarray([L for _s, L in all_bursts], dtype=np.float64)
    onsets_all = np.asarray([s for s, _L in all_bursts], dtype=np.float64)
    pre_all = [p for seed in triggered for p in feats[seed]["pre_trigger"]]

    report = {
        "record_id": "ROUTED_COMPOSITION_A_HARM_LOCALIZATION",
        "classification": "DIAGNOSTIC. Post-hoc. Descriptive. Gates nothing.",
        "reads": "ROUTED_COMPOSITION_OUTCOME_EPISODES.csv (block 20200001-20200064)",
        "does_not_change": ("ROUTED_COMPOSITION_OUTCOME_RESULT.json remains AUDIT_FAILED and its "
                            "verdict remains B_GAIN_WITH_EXCESS_A_HARM. Nothing here is a gate."),
        "no_new_seeds": "Pole-A STATE_B_TRIGGER episodes of the already-spent block were "
                        "re-simulated deterministically to recover per-tick traces.",
        "no_knob_searched": "No threshold, window, hysteresis or dwell was varied.",
        "questions_declared_before_the_numbers": QUESTIONS,
        "determinism_gate": {
            "episodes": len(OUTCOME_SEEDS), "drift": 0,
            "fields_compared": ["blue_score", "red_score", "blue_win", "steps", "role_switch_count"],
            "detail": "every re-simulated episode reproduces the sealed row exactly",
        },
        "cohort": {
            "pole_A_episodes": len(OUTCOME_SEEDS),
            "episodes_with_any_trigger": len(triggered),
            "episodes_never_triggered": len(OUTCOME_SEEDS) - len(triggered),
            "untriggered_are_bit_identical_to_FIXED_2A2D": "verified in ROUTED_COMPOSITION_OUTCOME_READING",
            "discordant_losses_2A2D_win_STATE_lose": len(loss),
            "discordant_gains_2A2D_lose_STATE_win": len(gain),
        },
        "Q1_trigger_timing": {
            "n_bursts_total": len(all_bursts),
            "onset_tick": {"mean": round(float(onsets_all.mean()), 2),
                           "median": round(float(np.median(onsets_all)), 2),
                           "min": int(onsets_all.min()), "max": int(onsets_all.max()),
                           "deciles": [round(float(x), 1) for x in
                                       np.percentile(onsets_all, [10, 25, 50, 75, 90])]},
            "fraction_of_bursts_starting_in_first_40_ticks":
                round(float((onsets_all < 40).mean()), 4),
        },
        "Q2_burst_length": {
            "length_ticks": {"mean": round(float(lens.mean()), 2),
                             "median": round(float(np.median(lens)), 2),
                             "min": int(lens.min()), "max": int(lens.max()),
                             "deciles": [round(float(x), 1) for x in
                                         np.percentile(lens, [10, 25, 50, 75, 90])]},
            "bursts_per_triggered_episode": round(len(all_bursts) / max(len(triggered), 1), 3),
            "fraction_of_bursts_running_to_episode_end":
                round(float(np.mean([1.0 if (s + L) >= len(per_ep[sd]["flags"]) else 0.0
                                     for sd in triggered
                                     for s, L in _runs_of_ones(per_ep[sd]["flags"])])), 4),
        },
        "Q3_state_preceding_a_trigger": {
            "n_onsets": len(pre_all),
            "score_diff_blue_minus_red": {
                "mean": round(float(np.mean([p["score_diff"] for p in pre_all])), 3),
                "fraction_while_ahead": round(float(np.mean([p["score_diff"] > 0 for p in pre_all])), 4),
                "fraction_while_level": round(float(np.mean([p["score_diff"] == 0 for p in pre_all])), 4),
                "fraction_while_behind": round(float(np.mean([p["score_diff"] < 0 for p in pre_all])), 4)},
            "fraction_while_carrying": round(float(np.mean([p["carrying"] for p in pre_all])), 4),
            "mean_red_alive": round(float(np.mean([p["red_alive"] for p in pre_all])), 3),
            "mean_blue_alive": round(float(np.mean([p["blue_alive"] for p in pre_all])), 3),
            "mean_blue_tagged": round(float(np.mean([p["tagged"] for p in pre_all])), 3),
        },
        "Q4_are_losses_concentrated_in_particular_excursions": {
            "comparison": "discordant-loss episodes vs all other triggered episodes",
            "contrasts": contrasts,
            "multiplicity_disclosed": (f"6 descriptive contrasts on n_loss={len(loss)} vs "
                                       f"n_other={len(other)}. Intervals are unadjusted and the "
                                       "loss group is small; treat any separation as a lead, "
                                       "never as a finding."),
        },
        "claim_boundary": ("Descriptive localization on one already-spent block, one pole, one "
                           "frozen router. Generates hypotheses about WHERE the Pole-A cost sits. "
                           "Establishes no mechanism and authorizes no change."),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("\n== Q1 trigger timing ==")
    print(f"   {len(all_bursts)} bursts; onset tick mean {onsets_all.mean():.1f} "
          f"median {np.median(onsets_all):.0f} range {int(onsets_all.min())}-{int(onsets_all.max())}; "
          f"{(onsets_all < 40).mean():.1%} start in the first 40 ticks")
    print("== Q2 burst length ==")
    print(f"   mean {lens.mean():.1f} median {np.median(lens):.0f} max {int(lens.max())} ticks; "
          f"{report['Q2_burst_length']['bursts_per_triggered_episode']} bursts per triggered episode; "
          f"{report['Q2_burst_length']['fraction_of_bursts_running_to_episode_end']:.1%} run to episode end")
    print("== Q3 state preceding a trigger ==")
    q3 = report["Q3_state_preceding_a_trigger"]
    print(f"   score diff {q3['score_diff_blue_minus_red']['mean']:+.2f}; "
          f"ahead {q3['score_diff_blue_minus_red']['fraction_while_ahead']:.1%} / "
          f"level {q3['score_diff_blue_minus_red']['fraction_while_level']:.1%} / "
          f"behind {q3['score_diff_blue_minus_red']['fraction_while_behind']:.1%}; "
          f"carrying {q3['fraction_while_carrying']:.1%}; red alive {q3['mean_red_alive']:.2f}")
    print(f"== Q4 discordant losses (n={len(loss)}) vs other triggered (n={len(other)}) ==")
    for c in contrasts:
        if "difference" in c:
            print(f"   {c['metric']:18s} loss {c['loss_mean']:8.3f} | other {c['other_mean']:8.3f} "
                  f"| diff {c['difference']:+8.3f} {c['ci95']}")
    print(f"\n-> {OUT}\n-> {TICKS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
