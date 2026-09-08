"""Measure what each reward family actually pays per episode, then derive V3 weights.

Three single-constant ablations each moved the failure to whichever shaping
channel still paid. The completed runs show why: the win/loss signal is 1.4-1.9%
of total reward mass in EVERY run, competent and collapsed alike. So the fix is
not another constant -- it is a budget.

This measures, from runs already on disk:

  1. how often each rewardable event actually occurs per episode
  2. what each reward family therefore pays per episode at current values
  3. the resulting hierarchy (which is inverted today)

and then DERIVES weights that satisfy a declared budget, rather than picking
numbers by hand. Per-event value is not the constraint; expected cumulative
per-episode value is what PPO optimises.

Run:  python experiments/measure_reward_budget.py
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Current values (game_manager.py / GPUFieldConfig defaults).
CURRENT = {
    "capture": 100.0,          # SPARSE_FLAG_CAPTURE_POINTS, per capture, +/-
    "tag_no_flag": 100.0,      # SPARSE_TAG_NO_FLAG_POINTS, symmetric
    "tag_carrier": 50.0,       # SPARSE_TAG_WITH_FLAG_POINTS
    "oob": -100.0,             # SPARSE_OOB_POINTS
    "failed_commit": -0.2,     # ACTION_FAILED_PUNISHMENT (NOT /100 scaled)
}
SPARSE_DIVISOR = 100.0         # reward_sparse = sparse_weight * points / 100

# Runs with a full event ledger (tag carrier/non-carrier split + captures).
LEDGER_RUNS = {
    "notag_2700001": "artifacts/g0_v2_tagreward_ablation/g0_v2_notag_seed2700001",
    "notag_2700002": "artifacts/g0_v2_tagreward_ablation/g0_v2_notag_seed2700002",
    "notag_2700003": "artifacts/g0_v2_tagreward_ablation/g0_v2_notag_seed2700003",
}
# Baseline runs carry aggregate counts only, in the training report.
BASELINE_RUNS = {
    "baseline_2500001": "artifacts/g0_v2_seed2500001",
    "baseline_2500002": "artifacts/g0_v2_seed2500002",
    "baseline_2500003": "artifacts/g0_v2_seed2500003",
}

# --- declared budget --------------------------------------------------------
# The invariant: expected CUMULATIVE shaping per episode must sit well below the
# value of accomplishing the objective.
BUDGET = {
    "objective_reference": "capture",
    "tactical_shaping_max_share": 0.15,   # all tag-type shaping, combined
    "safety_penalty_max_share": 0.30,     # OOB + failed-commit, combined
    "tags_start_at_zero": True,           # mechanics already make tags valuable
}


def _episodes(run: Path) -> int:
    p = run / "episode_rows.csv"
    if not p.is_file():
        return 0
    with open(p, encoding="utf-8", newline="") as f:
        return sum(1 for _ in csv.DictReader(f))


def ledger_rates(name: str, run: Path) -> dict | None:
    rep = run / "ablation_report.json"
    if not rep.is_file():
        return None
    d = json.loads(rep.read_text(encoding="utf-8"))
    sd = d.get("sparse_decomposition") or {}
    eps = _episodes(run)
    if not eps:
        return None
    per = lambda k: sd.get(k, 0) / eps  # noqa: E731
    return {
        "run": name,
        "episodes": eps,
        "tag_no_flag_per_ep": round(per("tag_blue_noncarrier") + per("tag_against_blue"), 3),
        "tag_carrier_per_ep": round(per("tag_blue_carrier"), 3),
        "captures_per_ep": round(per("capture_blue") + per("capture_red"), 3),
        "blue_captures_per_ep": round(per("capture_blue"), 3),
    }


def baseline_rates(name: str, run: Path) -> dict | None:
    rep = run / "g0_v2_training_report.json"
    if not rep.is_file():
        return None
    d = json.loads(rep.read_text(encoding="utf-8"))
    h = d["runtime_health"]
    eps = h.get("resets_observed") or _episodes(run)
    if not eps:
        return None
    return {
        "run": name,
        "episodes": eps,
        # Ledger split unavailable here; tag_success covers both kinds.
        "tag_all_per_ep": round(h["tag_success_events"] / eps, 3),
        "captures_per_ep": round(h["capture_events"] / eps, 3),
    }


def failed_commits_per_episode(run: Path) -> float | None:
    """Back out failed-commit count from the reward channel PPO actually saw."""
    p = run / "metrics.csv"
    if not p.is_file():
        return None
    with open(p, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    vals = [
        float(r["reward_failure_mean"]) for r in rows
        if r.get("reward_failure_mean") not in (None, "", "nan")
    ]
    if not vals:
        return None
    # reward_failure_mean is per-step mean of (-0.2 * n_failed). Report the
    # per-step failure count; per-episode needs the horizon.
    return statistics.fmean(vals) / CURRENT["failed_commit"]


def channel_per_episode(run: Path, horizon: int = 240) -> dict:
    """Per-episode total for every composed reward channel, in ``raw`` units.

    ``_compose_training_reward_components`` sums terminal + sparse + failure +
    offense + dense_weight*(pbrs+team) before squashing, so these are directly
    comparable -- which is the whole point of a budget.
    """
    p = run / "metrics.csv"
    if not p.is_file():
        return {}
    with open(p, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    def m(col: str) -> float:
        vals = [float(r[col]) for r in rows
                if r.get(col) not in (None, "", "nan")]
        return statistics.fmean(vals) if vals else 0.0

    return {
        "terminal": round(m("reward_terminal_mean") * horizon, 3),
        "sparse_total": round(m("reward_sparse_mean") * horizon, 3),
        "failure": round(m("reward_failure_mean") * horizon, 3),
        "offense": round(m("reward_offense_mean") * horizon, 3),
        "pbrs": round(m("reward_pbrs_mean") * horizon, 3),
        "team": round(m("reward_team_mean") * horizon, 3),
    }


def implied_oob_per_episode(rates: dict, sparse_total: float,
                            tag_no_flag_points: float,
                            tag_carrier_points: float) -> float | None:
    """Back OOB out of the sparse total, since it has no event ledger.

    sparse = (100*(capB-capR) + tag_pts - 100*oob) / 100
    """
    capb = rates.get("blue_captures_per_ep")
    caps = rates.get("captures_per_ep")
    if capb is None or caps is None:
        return None
    capr = caps - capb
    tag_pts = (tag_no_flag_points * rates.get("tag_no_flag_per_ep", 0.0)
               + tag_carrier_points * rates.get("tag_carrier_per_ep", 0.0))
    net_cap_pts = CURRENT["capture"] * (capb - capr)
    oob_pts = net_cap_pts + tag_pts - sparse_total * SPARSE_DIVISOR
    return round(oob_pts / abs(CURRENT["oob"]), 3)


def current_per_episode(rates: dict, failed_per_step: float | None,
                        horizon: int = 240) -> dict:
    """What each family pays per episode at TODAY's values."""
    cap = abs(CURRENT["capture"]) * rates.get("captures_per_ep", 0.0) / SPARSE_DIVISOR
    tag_nf = abs(CURRENT["tag_no_flag"]) * rates.get("tag_no_flag_per_ep", 0.0) / SPARSE_DIVISOR
    tag_c = abs(CURRENT["tag_carrier"]) * rates.get("tag_carrier_per_ep", 0.0) / SPARSE_DIVISOR
    fail = (abs(CURRENT["failed_commit"]) * failed_per_step * horizon
            if failed_per_step else None)
    out = {
        "capture_per_ep": round(cap, 3),
        "tag_no_flag_per_ep": round(tag_nf, 3),
        "tag_carrier_per_ep": round(tag_c, 3),
        "tag_total_per_ep": round(tag_nf + tag_c, 3),
        "failed_commit_per_ep": None if fail is None else round(fail, 3),
    }
    if cap > 0:
        out["tag_to_capture_ratio"] = round((tag_nf + tag_c) / cap, 2)
        if fail is not None:
            out["failure_to_capture_ratio"] = round(fail / cap, 2)
    return out


def derive_v3(rates_all: list[dict], current_all: list[dict]) -> dict:
    """Solve for values that satisfy the budget, using observed frequencies."""
    tags = [r.get("tag_no_flag_per_ep", 0.0) + r.get("tag_carrier_per_ep", 0.0)
            for r in rates_all if r.get("captures_per_ep")]
    caps = [r["captures_per_ep"] for r in rates_all if r.get("captures_per_ep")]
    fails = [c["failed_commit_per_ep"] for c in current_all
             if c.get("failed_commit_per_ep") is not None]
    cap_reward_per_ep = [abs(CURRENT["capture"]) * c / SPARSE_DIVISOR for c in caps]

    mean_tags = statistics.fmean(tags) if tags else 0.0
    mean_caps = statistics.fmean(caps) if caps else 0.0
    mean_cap_reward = statistics.fmean(cap_reward_per_ep) if cap_reward_per_ep else 0.0
    mean_fail_reward = statistics.fmean(fails) if fails else 0.0

    # Tactical shaping budget, spread over observed tag frequency.
    tag_budget = BUDGET["tactical_shaping_max_share"] * mean_cap_reward
    max_tag_points = (tag_budget * SPARSE_DIVISOR / mean_tags) if mean_tags else 0.0

    # Safety penalties rescaled so they cannot dominate the objective.
    safety_budget = BUDGET["safety_penalty_max_share"] * mean_cap_reward
    fail_rescale = (safety_budget / mean_fail_reward) if mean_fail_reward else None
    derived_failed_commit = (CURRENT["failed_commit"] * fail_rescale
                             if fail_rescale else None)

    return {
        "observed": {
            "mean_tags_per_episode": round(mean_tags, 3),
            "mean_captures_per_episode": round(mean_caps, 3),
            "tags_per_capture": round(mean_tags / mean_caps, 2) if mean_caps else None,
            "mean_capture_reward_per_episode": round(mean_cap_reward, 3),
            "mean_failed_commit_cost_per_episode": round(mean_fail_reward, 3),
        },
        "budget": BUDGET,
        "derived": {
            "tag_points_ceiling_if_nonzero": round(max_tag_points, 2),
            "recommended_tag_points": 0.0,
            "recommended_tag_points_rationale": (
                "start at zero: tagging already serves the objective through its "
                "consequences (carrier drops the flag, attack is interrupted, "
                "cooldown opens a window). The ceiling above is what the budget "
                "would permit IF an explicit tag reward turns out to be needed."
            ),
            "failed_commit_rescale_factor": (
                None if fail_rescale is None else round(fail_rescale, 4)
            ),
            "derived_failed_commit_points": (
                None if derived_failed_commit is None else round(derived_failed_commit, 4)
            ),
        },
        "warning": (
            "Derived from 300k-step probes where the tag reward was ALREADY zeroed, "
            "so tag frequencies reflect a policy not being paid to tag. Frequencies "
            "under a restored tag reward would be higher, making these ceilings "
            "conservative in the wrong direction if tags are ever re-enabled."
        ),
    }


def main() -> int:
    rates_all, current_all, rows = [], [], []
    for name, path in {**LEDGER_RUNS}.items():
        run = PROJECT_ROOT / path
        r = ledger_rates(name, run)
        if not r:
            continue
        fps = failed_commits_per_episode(run)
        c = current_per_episode(r, fps)
        ch = channel_per_episode(run)
        # These probes ran with tag_no_flag zeroed; carrier tags still paid 50.
        c["implied_oob_events_per_ep"] = implied_oob_per_episode(
            r, ch.get("sparse_total", 0.0), 0.0, CURRENT["tag_carrier"]
        )
        rates_all.append(r)
        current_all.append(c)
        rows.append({"rates": r, "current_per_episode": c, "channel_per_episode": ch})

    for name, path in BASELINE_RUNS.items():
        run = PROJECT_ROOT / path
        b = baseline_rates(name, run)
        if b:
            rows.append({"rates": b, "current_per_episode": None})

    v3 = derive_v3(rates_all, current_all)
    report = {"current_values": CURRENT, "runs": rows, "v3_derivation": v3}
    out = PROJECT_ROOT / "artifacts" / "reward_budget_v3.json"
    out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    print("=" * 78)
    print("REWARD BUDGET MEASUREMENT (from completed runs)")
    print("=" * 78)
    for row in rows:
        r, c = row["rates"], row["current_per_episode"]
        print(f"\n{r['run']}  ({r['episodes']} episodes)")
        for k, v in r.items():
            if k not in ("run", "episodes"):
                print(f"    {k:26s} {v}")
        if c:
            print("  per-episode reward at CURRENT values:")
            for k, v in c.items():
                print(f"    {k:26s} {v}")
        ch = row.get("channel_per_episode")
        if ch:
            tot = sum(abs(v) for v in ch.values()) or 1e-12
            print("  composed channel totals per episode (raw units, |share|):")
            for k, v in sorted(ch.items(), key=lambda kv: -abs(kv[1])):
                print(f"    {k:26s} {v:>10.3f}   {abs(v)/tot*100:5.1f}%")
    print("\n" + "=" * 78)
    print("V3 DERIVATION")
    print("=" * 78)
    print(json.dumps(v3, indent=2))
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
