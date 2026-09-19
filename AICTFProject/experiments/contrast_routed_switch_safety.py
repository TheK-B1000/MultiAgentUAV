"""Pass 2: what separates a HARMFUL Pole-A trigger from a HELPFUL Pole-B trigger?

DIAGNOSTIC. Post-hoc. Descriptive. Gates nothing and authorizes nothing.

Pass 1 (localize_routed_a_harm.py) characterised Pole-A triggers on their own.
This pass adds the two comparisons the PI named that pass 1 could not make:

  * the 39 triggered Pole-A episodes vs the 25 untriggered Pole-A episodes
    (which are bit-identical to FIXED_2A2D), and
  * Pole-A triggers, where switching HURTS, vs Pole-B triggers, where it HELPS.

No new seeds: both poles' STATE_B_TRIGGER episodes of the already-spent block
20200001-20200064 are re-simulated deterministically, gated against the sealed
rows. No knob is varied.

CANDIDATE INTERLOCK OBSERVABLES ARE DECLARED BELOW, BEFORE THE NUMBERS. They are
the PI's four named possibilities, expressed as observables the router could
legally read. This script REPORTS their separation. It does not choose one, does
not fit a threshold, and does not rank them by outcome. Choosing a guard is a PI
decision that needs its own freeze and fresh held-out seeds -- selecting the
best-separating observable here and then testing it on these same rows would be
exactly the post-hoc fitting this program has spent three freezes avoiding.
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
OUT = SD / "ROUTED_COMPOSITION_SWITCH_SAFETY_CONTRAST.json"
TICKS_CSV = SD / "routed_composition_switch_safety_ticks.csv"

# Declared before any number is computed. Each is an observable the router could
# legally read at the instant of a switch, mapped to the PI's four possibilities.
CANDIDATE_INTERLOCKS = {
    "entry_guard_onset_tick": "PI option 1 (early triggers): how far into the episode the switch happens",
    "defensive_advantage_score_diff": "PI option 2 (favourable defensive state): blue_score - red_score at the switch",
    "home_pressure_reds_near_blue_flag": "PI option 2/4 (protected condition / home pressure): P_blue at the switch",
    "burst_length": "PI option 3 (long bursts): how long the 4A_0D excursion runs",
    "carrier_state": "PI option 4 (carrier event): whether a blue agent is carrying at the switch",
    "red_alive_at_switch": "PI option 4 (force balance): live red agents at the switch",
    "blue_tagged_at_switch": "PI option 4 (own-team disruption): tagged blue agents at the switch",
}


def trace_state_episode(pole: str, seed: int) -> tuple[dict, list[dict]]:
    env, core, genome, live = _make_env(pole, seed)
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
            bfx = float(core.blue_flag_pos[0, 0].item())
            bfy = float(core.blue_flag_pos[0, 1].item())
            p_blue = 0
            for i in range(core.red_x.shape[1]):
                if not bool(core.red_alive[0, i].item()):
                    continue
                dx = float(core.red_x[0, i].item()) - bfx
                dy = float(core.red_y[0, i].item()) - bfy
                p_blue += int((dx * dx + dy * dy) ** 0.5 <= 4.0)
            composition = router.update(d_value)
            roles = composition_roles(composition)
            ticks.append({
                "pole": pole, "seed": seed, "tick": steps, "d": d_value,
                "windowed": router.windowed,
                "state_4A0D": int(composition == TRIGGERED_COMPOSITION),
                "p_blue": p_blue,
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
        return ({"pole": pole, "seed": seed, "blue_score": b, "red_score": rd,
                 "blue_win": int(b > rd), "steps": steps,
                 "switches": int(router.switches)}, ticks)
    finally:
        env.close()


def _contrast(name: str, a: np.ndarray, b: np.ndarray, a_lab: str, b_lab: str) -> dict[str, Any]:
    if a.size == 0 or b.size == 0:
        return {"observable": name, "note": "one group empty; not computed",
                "n_" + a_lab: int(a.size), "n_" + b_lab: int(b.size)}
    rng = np.random.default_rng(7)
    d = (a[rng.integers(0, a.size, size=(20000, a.size))].mean(axis=1)
         - b[rng.integers(0, b.size, size=(20000, b.size))].mean(axis=1))
    lo, hi = np.percentile(d, [2.5, 97.5])
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2.0) if (a.size > 1 and b.size > 1) else np.nan
    return {"observable": name,
            f"{a_lab}_mean": round(float(a.mean()), 4),
            f"{b_lab}_mean": round(float(b.mean()), 4),
            "difference": round(float(a.mean() - b.mean()), 4),
            "ci95": [round(float(lo), 4), round(float(hi), 4)],
            "separation_excludes_zero": bool(lo > 0 or hi < 0),
            "standardized_difference": (None if not np.isfinite(pooled) or pooled == 0
                                        else round(float((a.mean() - b.mean()) / pooled), 3)),
            f"n_{a_lab}": int(a.size), f"n_{b_lab}": int(b.size)}


def main() -> int:
    sealed: dict[tuple[str, str, int], dict] = {}
    with EPISODE_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sealed[(row["pole"], row["arm"], int(row["seed"]))] = row
    if not sealed:
        raise SystemExit("ABORT: outcome rows missing -- absence is an error state")

    eps: dict[tuple[str, int], dict] = {}
    all_ticks: list[dict] = []
    drift: list[str] = []
    for pole in ("A", "B"):
        print(f"Re-simulating pole {pole} STATE_B_TRIGGER (no new seeds)...", flush=True)
        for i, seed in enumerate(OUTCOME_SEEDS, 1):
            summary, ticks = trace_state_episode(pole, seed)
            ref = sealed[(pole, "STATE_B_TRIGGER", seed)]
            for fld in ("blue_score", "red_score", "blue_win", "steps"):
                if int(summary[fld]) != int(ref[fld]):
                    drift.append(f"{pole}/{seed}/{fld}")
            if int(summary["switches"]) != int(ref["role_switch_count"]):
                drift.append(f"{pole}/{seed}/switches")
            eps[(pole, seed)] = {"summary": summary, "ticks": ticks,
                                 "flags": [t["state_4A0D"] for t in ticks]}
            all_ticks.extend(ticks)
            if i % 16 == 0:
                print(f"  {pole} {i}/{len(OUTCOME_SEEDS)}", flush=True)
    if drift:
        raise SystemExit(f"ABORT: re-simulation drifted from the sealed rows: {drift[:6]}")
    print(f"  determinism gate PASS: {len(eps)}/{len(eps)} episodes reproduce the sealed rows",
          flush=True)

    with TICKS_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_ticks[0]))
        w.writeheader()
        w.writerows(all_ticks)

    # ---- per-switch records, both poles ------------------------------------
    switches: list[dict] = []
    per_ep: dict[tuple[str, int], dict] = {}
    for (pole, seed), ep in eps.items():
        bursts = _runs_of_ones(ep["flags"])
        win_state = int(ep["summary"]["blue_win"])
        win_fixed = int(sealed[(pole, "FIXED_2A2D", seed)]["blue_win"])
        for onset, length in bursts:
            t = ep["ticks"][max(onset - 1, 0)]
            switches.append({
                "pole": pole, "seed": seed,
                "entry_guard_onset_tick": float(onset),
                "burst_length": float(length),
                "defensive_advantage_score_diff": float(t["blue_score"] - t["red_score"]),
                "home_pressure_reds_near_blue_flag": float(t["p_blue"]),
                "carrier_state": float(t["blue_carrying"]),
                "red_alive_at_switch": float(t["red_alive"]),
                "blue_tagged_at_switch": float(t["blue_tagged"]),
                "windowed_at_switch": float(t["windowed"]),
                "episode_harmful": int(win_fixed == 1 and win_state == 0),
                "episode_helpful": int(win_fixed == 0 and win_state == 1),
            })
        per_ep[(pole, seed)] = {
            "n_bursts": len(bursts), "ticks_in_4A0D": int(sum(ep["flags"])),
            "fraction_in_4A0D": float(sum(ep["flags"])) / max(len(ep["flags"]), 1),
            "triggered": len(bursts) > 0, "win_state": win_state, "win_fixed_2a2d": win_fixed,
            "mean_windowed": float(np.mean([t["windowed"] for t in ep["ticks"]])),
            "min_windowed": float(np.min([t["windowed"] for t in ep["ticks"]])),
            "mean_p_blue": float(np.mean([t["p_blue"] for t in ep["ticks"]])),
        }

    sw = {k: np.asarray([s[k] for s in switches], dtype=np.float64) for k in CANDIDATE_INTERLOCKS}
    pole_of = np.asarray([s["pole"] for s in switches])
    a_mask, b_mask = pole_of == "A", pole_of == "B"

    # --- comparison 1: triggered vs untriggered Pole-A episodes -------------
    a_eps = [(p, s) for (p, s) in per_ep if p == "A"]
    trig = [k for k in a_eps if per_ep[k]["triggered"]]
    untrig = [k for k in a_eps if not per_ep[k]["triggered"]]
    ep_keys = ("mean_windowed", "min_windowed", "mean_p_blue")
    cmp_trig_untrig = [
        _contrast(k, np.asarray([per_ep[x][k] for x in trig]),
                  np.asarray([per_ep[x][k] for x in untrig]), "triggered", "untriggered")
        for k in ep_keys]
    cmp_trig_untrig.append(_contrast(
        "baseline_win_rate_FIXED_2A2D",
        np.asarray([float(per_ep[x]["win_fixed_2a2d"]) for x in trig]),
        np.asarray([float(per_ep[x]["win_fixed_2a2d"]) for x in untrig]),
        "triggered", "untriggered"))

    # --- comparison 2: Pole-A switches vs Pole-B switches -------------------
    cmp_a_vs_b = [_contrast(k, sw[k][a_mask], sw[k][b_mask], "poleA", "poleB")
                  for k in CANDIDATE_INTERLOCKS]

    # --- comparison 3: harmful vs non-harmful Pole-A switches ---------------
    harm = np.asarray([s["episode_harmful"] for s in switches])[a_mask].astype(bool)
    cmp_harm = [_contrast(k, sw[k][a_mask][harm], sw[k][a_mask][~harm], "harmful", "other")
                for k in CANDIDATE_INTERLOCKS]

    report = {
        "record_id": "ROUTED_COMPOSITION_SWITCH_SAFETY_CONTRAST",
        "classification": "DIAGNOSTIC. Post-hoc. Descriptive. Gates nothing, authorizes nothing.",
        "reads": "ROUTED_COMPOSITION_OUTCOME_EPISODES.csv (block 20200001-20200064)",
        "does_not_change": ("ROUTED_COMPOSITION_OUTCOME_RESULT.json remains AUDIT_FAILED with "
                            "verdict B_GAIN_WITH_EXCESS_A_HARM."),
        "no_new_seeds": "both poles' STATE episodes of the spent block re-simulated deterministically",
        "no_knob_searched": "no threshold, window, hysteresis or dwell was varied",
        "candidate_interlocks_declared_before_the_numbers": CANDIDATE_INTERLOCKS,
        "determinism_gate": {"episodes": len(eps), "drift": 0,
                             "fields": ["blue_score", "red_score", "blue_win", "steps",
                                        "role_switch_count"]},
        "cohort": {
            "pole_A_switches": int(a_mask.sum()), "pole_B_switches": int(b_mask.sum()),
            "pole_A_triggered_episodes": len(trig), "pole_A_untriggered_episodes": len(untrig),
            "pole_A_harmful_switches": int(harm.sum()),
        },
        "comparison_1_poleA_triggered_vs_untriggered": cmp_trig_untrig,
        "comparison_2_poleA_switches_vs_poleB_switches": cmp_a_vs_b,
        "comparison_3_harmful_vs_other_poleA_switches": cmp_harm,
        "MULTIPLICITY_AND_SELECTION_WARNING": (
            f"{len(CANDIDATE_INTERLOCKS)} observables x 3 comparisons, unadjusted intervals, on "
            "one already-spent block. Some will separate by chance. This record deliberately does "
            "NOT rank them by outcome, does NOT fit a threshold, and does NOT nominate a guard. "
            "Picking the best-separating observable here and then evaluating it on these same rows "
            "would be the post-hoc fitting the selector freezes exist to prevent. Any guard must be "
            "chosen by the PI, frozen with its threshold declared in advance, and tested on fresh "
            "held-out seeds."),
        "claim_boundary": ("Descriptive contrast on one block, one frozen router, two frozen "
                           "opponents. Generates hypotheses about which safety interlock to "
                           "consider. Establishes no mechanism."),
    }
    OUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def show(title: str, rows: list[dict], a_lab: str, b_lab: str) -> None:
        print(f"\n== {title} ==")
        for c in rows:
            if "difference" not in c:
                print(f"   {c['observable']:36s} {c.get('note')}")
                continue
            star = " *" if c["separation_excludes_zero"] else "  "
            print(f"   {c['observable']:36s} {a_lab} {c[a_lab + '_mean']:8.3f} | "
                  f"{b_lab} {c[b_lab + '_mean']:8.3f} | diff {c['difference']:+8.3f} "
                  f"{c['ci95']}{star}")

    show("1. Pole-A triggered vs untriggered episodes", cmp_trig_untrig, "triggered", "untriggered")
    show("2. Pole-A switches vs Pole-B switches", cmp_a_vs_b, "poleA", "poleB")
    show("3. Harmful vs other Pole-A switches", cmp_harm, "harmful", "other")
    print("\n   (* = interval excludes zero; unadjusted, see MULTIPLICITY_AND_SELECTION_WARNING)")
    print(f"\n-> {OUT}\n-> {TICKS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
