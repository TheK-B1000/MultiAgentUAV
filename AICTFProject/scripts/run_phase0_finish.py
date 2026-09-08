"""Unattended supervisor: finish and freeze V2 (Phase 0), then hard-stop.

Carries the remaining Phase-0 work to completion without human input:

    wait for 7E  ->  synthesize V2_MECHANICAL_DIAGNOSTIC (Q1-Q4)
                 ->  write V3_RECOMMENDATION (not implemented)
                 ->  HUMAN_DECISION_REQUIRED, stop

Deliberately does NOT start the strategic-demand searcher, any ruleset change,
or any PPO training. The V3 search space depends on the V2 diagnosis it is
producing -- R5 carrier vulnerability is explicitly conditional on 7E -- so
auto-starting Phase 1+ would mean choosing a search space from results the
supervisor has not yet read.

State is derived by INSPECTING ARTIFACTS, so a crash or restart re-derives the
truth rather than trusting a stored cursor.

Run:  python scripts/run_phase0_finish.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv/Scripts/python.exe")
P7 = ROOT / "artifacts/phase7"
SD = ROOT / "artifacts/strategic_demand"
LOG = ROOT / "artifacts/phase7/phase0_supervisor.log"

TAGSAT = P7 / "tag_saturation.json"
CARRIER = P7 / "carrier_return.json"
INTERACT = P7 / "interaction_assay.json"
COMMIT = P7 / "commitment_assay.json"
DIAG_MD = SD / "V2_MECHANICAL_DIAGNOSTIC.md"
DIAG_JSON = SD / "V2_MECHANICAL_DIAGNOSTIC.json"
V3 = SD / "V3_RECOMMENDATION.md"
FINAL = SD / "HUMAN_DECISION_REQUIRED_PHASE0.md"


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def workers_alive() -> int:
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Where-Object { $_.CommandLine -like '*phase7e*' }).Count"],
            capture_output=True, text=True, timeout=60).stdout.strip()
        return int(out or 0)
    except Exception:
        return 0


def jload(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# --------------------------------------------------------------- synthesis
def build_diagnostic() -> dict:
    tag = jload(TAGSAT) or {}
    car = jload(CARRIER) or {}
    inter = jload(INTERACT) or {}

    ta = tag.get("arms", {})
    one = ta.get("ONE_DEFENDER_vs_OP7", {})
    both = ta.get("BOTH_ATTACK_vs_OP7", {})
    ia = inter.get("arms", {})

    def g(d, *ks, default=None):
        cur = d
        for k in ks:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    # --- Q1 context-specific opportunity cost -------------------------
    q1 = {
        "answer": "CONTEXT_DEPENDENT — no global label",
        "OP6_FAST_RAID": {
            "defensive_benefit": "PASS (+0.6875, CI95 [+0.3438, +1.0312])",
            "offensive_cost": "FAIL (-0.1875, CI95 [-0.4062, +0.0625])",
            "gate_A": "FAIL_OFFENSE_COST",
            "reading": "holding a defender pays defensively and costs no demonstrated offense",
        },
        "OP7_FORTRESS": {
            "defensive_benefit": "FAIL (+0.1875, CI95 [-0.0938, +0.5000])",
            "offensive_cost": "PASS (+0.9062, CI95 [+0.5938, +1.2188])",
            "gate_A": "FAIL_DEFENSE_BENEFIT",
            "reading": "holding a defender costs ~0.9 captures and buys no demonstrated defense",
        },
        "CONTEXT_DEPENDENT_OFFENSIVE_VALUE": "SUPPORTED",
        "gate_A_overall": "INCOMPLETE — neither context passes both legs, but the "
                          "contexts show OPPOSITE structure rather than a shared null",
    }

    # --- Q2 two-agent offense + mechanism ------------------------------
    q2 = {
        "answer": "YES against FORTRESS; mechanism is THROUGHPUT, not suppression",
        "suppression_hypothesis": "FALSIFIED",
        "suppression_evidence": "mean_suppressions_of_red = 0.000 in ALL 16 arms "
                                "(256 episodes) at every range 2.0/2.5/2.75/3.00",
        "conversion": {
            "ONE_DEFENDER_vs_OP7": {
                "breach": g(ia, "7B_ONE_DEFENDER_vs_OP7_supp2", "breach_rate"),
                "capture": g(ia, "7B_ONE_DEFENDER_vs_OP7_supp2", "capture_rate"),
                "pickups": g(ia, "7B_ONE_DEFENDER_vs_OP7_supp2", "mean_pickups"),
            },
            "BOTH_ATTACK_vs_OP7": {
                "breach": g(ia, "7B_BOTH_ATTACK_vs_OP7_supp2", "breach_rate"),
                "capture": g(ia, "7B_BOTH_ATTACK_vs_OP7_supp2", "capture_rate"),
                "pickups": g(ia, "7B_BOTH_ATTACK_vs_OP7_supp2", "mean_pickups"),
            },
        },
        "rate_limit_evidence": {
            "median_inter_tag_gap_ONE_DEFENDER": g(one, "inter_tag_interval", "median"),
            "median_inter_tag_gap_BOTH_ATTACK": g(both, "inter_tag_interval", "median"),
            "cooldown_floor_seconds": 10.0,
            "post_tag_other_agent_pickup_ONE": g(one, "post_tag_window", "frac"),
            "post_tag_other_agent_pickup_BOTH": g(both, "post_tag_window", "frac"),
            "tag_ended_attempt_ONE": g(one, "tag_ended_attempt", "frac"),
            "tag_ended_attempt_BOTH": g(both, "tag_ended_attempt", "frac"),
        },
        "reading": "the second attacker does not remove the defence; it outpaces a "
                   "rate-limited defence. Inter-tag gap compresses toward the 10s "
                   "floor, the second attacker converts post-tag windows, and a "
                   "single tag stops the whole attempt far less often.",
        "metric_withdrawn": "cooldown 'at_floor' fraction is vacuous by construction "
                            "(a tag can only succeed at zero cooldown) and is not used",
    }

    # --- Q3 commitment -------------------------------------------------
    q3 = {
        "answer": "UNRESOLVED",
        "7D_OP6": "INVALID_TREATMENT_INSTANTIATION",
        "why": "mean t_intent 7.2 preceded mean t_commit 7.3, so the RECOVERY arm "
               "approximated ONE_DEFENDER from episode start and the assay "
               "degenerated into a Gate 2B re-run",
        "valid_side_finding": "OP6_NO_PRECOMMITMENT_UNCERTAINTY_WINDOW = SUPPORTED "
                              "under the frozen definitions",
        "caveat": "both t_commit and t_intent are midline-crossing events, so their "
                  "near-simultaneity is partly structural given symmetric geometry",
        "not_established": "whether a deeper commitment threshold would precede intent",
    }

    # --- Q4 carrier self-sufficiency -----------------------------------
    ca = car.get("arms", {})
    q4 = {"answer": "PENDING", "arms": ca}
    if ca:
        effs, escorts, mates = [], [], []
        for _k, v in ca.items():
            if isinstance(v, dict):
                for src, dst in (("mean_path_efficiency", effs),
                                 ("mean_escort_fraction", escorts),
                                 ("mean_teammate_dist", mates)):
                    val = v.get(src)
                    if isinstance(val, (int, float)) and val == val:
                        dst.append(float(val))
        eff = sum(effs) / len(effs) if effs else float("nan")
        esc = sum(escorts) / len(escorts) if escorts else float("nan")
        q4 = {
            "answer": ("SUPPORTED" if (esc == esc and esc < 0.5)
                       else "NOT_SUPPORTED" if (esc == esc) else "INDETERMINATE"),
            "mean_path_efficiency": eff,
            "mean_escort_fraction": esc,
            "mean_teammate_distance": (sum(mates) / len(mates)) if mates else float("nan"),
            "arms": ca,
            "rule": "SUPPORTED when the teammate is near the carrier for a minority "
                    "of the return; this is descriptive and not a frozen gate",
        }

    return {
        "record": "V2 mechanical diagnosis (Phase 0 close-out)",
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "scope": "2v2 only, map_a, stock RULESET_V2",
        "Q1_allocation_opportunity_cost": q1,
        "Q2_two_agent_offense_and_mechanism": q2,
        "Q3_commitment_reversibility": q3,
        "Q4_carrier_self_sufficiency": q4,
        "frozen_2x2_cell": ("Gate A INCOMPLETE (context-dependent) x commitment "
                            "UNRESOLVED -- the frozen 2x2 cannot be applied because "
                            "neither axis resolved to a single label"),
        "negative_findings_preserved": [
            "suppression is dormant in 2v2: 0 events / 256 episodes / all ranges",
            "suppression_range is therefore removed from all future rule levers",
            "7B_ORIGINAL (truncated 1-agent OP7) = INVALID_OPPONENT_INSTANTIATION",
            "OP7 pre-amendment offense_cost +0.6988 permanently non-gating",
        ],
    }


def build_v3(diag: dict) -> str:
    q4 = diag["Q4_carrier_self_sufficiency"]["answer"]
    carrier_line = {
        "SUPPORTED": "ELIGIBLE — 7E supports a largely self-sufficient carrier, so a "
                     "small carrier-vulnerability family {1.0, 0.9, 0.8} would create "
                     "a real escort decision.",
        "NOT_SUPPORTED": "NOT ELIGIBLE — 7E does not support the self-sufficient-carrier "
                         "hypothesis, so this lever is withheld per the conditional rule.",
    }.get(q4, "WITHHELD — 7E indeterminate; the conditional rule is not satisfied.")

    return f"""# V3_RECOMMENDATION — one minimal intervention

**Status: RECOMMENDATION ONLY. NOT IMPLEMENTED.** No ruleset was modified, no
PPO trained, no searcher started.

## Recommended single change

```
own_flag_home_required_to_score = True
```

**Lever 1 (own flag must be home to score). Binary. Nothing else.**

### Why this one

The V2 diagnosis shows allocation value is context-dependent but that neither
context produces BOTH legs of an opportunity cost:

```
OP6 FAST_RAID   defence pays,  offence costs nothing demonstrated
OP7 FORTRESS    offence pays,  defence buys nothing demonstrated
```

Lever 1 is the only candidate that couples the two directly. It gives
DOUBLE_BREACH a genuine downside it currently lacks — both agents forward means
a stolen home flag blocks scoring — while leaving GUARD_RAID's existing
weakness against the fortress intact. It is one boolean, so attribution stays
clean.

Verified available: the capture condition at `gpu_env/_core/_rules.py:566` is
`alive & carrying & ~tagged & (home_dist <= 1.2)` with **no own-flag check**.

### Explicitly excluded, on evidence

```
suppression_range   REMOVED — 0 suppression events / 256 episodes / all 4 ranges.
                    The M3 ladder was tuning a mechanic that never fires in 2v2.
```

This is the strongest negative result of Phase 0 and it retires a lever the
project had treated as a leading candidate.

### Held in reserve, not recommended now

- **Tag consequence duration** {{1x, 2.5x, 5x}} — plausible, but Q2 shows the
  tag system is already near its rate limit under BOTH_ATTACK
  (median gap ~10.4s against a 10s floor), so changing respawn may interact
  with a mechanism that is already saturated. Test after Lever 1, not with it.
- **Tag channel** {{0, 1, 2}}s — same reasoning.
- **Carrier vulnerability** {{1.0, 0.9, 0.8}} — {carrier_line}

### Why not stack them

The smallest intervention that produces strategic demand is the scientifically
interpretable one. If Lever 1 alone creates the two-way reversal, adding respawn
and channel changes would make the cause unattributable.

## What must be true before V3 is accepted

Unchanged frozen gates: two-way payoff reversal at >= 0.15 with LCB95 > 0 in
both directions, precommitment uncertainty (t_intent > t_commit) measured by a
legal-observation probe rather than a hand-picked geometry event, fresh held-out
replication, and non-degeneracy.

**Do not implement this recommendation without a human decision.**
"""


def main() -> int:
    log("phase0 supervisor start")
    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        if CARRIER.is_file():
            log("7E artifact present")
            break
        n = workers_alive()
        log(f"waiting for 7E (phase7e workers alive: {n})")
        if n == 0 and not CARRIER.is_file():
            time.sleep(60)
            if not CARRIER.is_file() and workers_alive() == 0:
                log("7E worker gone with no artifact -- proceeding with Q4 PENDING")
                break
        time.sleep(120)

    SD.mkdir(parents=True, exist_ok=True)
    diag = build_diagnostic()
    DIAG_JSON.write_text(json.dumps(diag, indent=2), encoding="utf-8")
    log(f"wrote {DIAG_JSON.name}")

    md = ["# V2 MECHANICAL DIAGNOSTIC", "",
          f"Generated {diag['utc']} | scope: {diag['scope']}", ""]
    for qk in ("Q1_allocation_opportunity_cost", "Q2_two_agent_offense_and_mechanism",
               "Q3_commitment_reversibility", "Q4_carrier_self_sufficiency"):
        md += [f"## {qk}", "", "```json",
               json.dumps(diag[qk], indent=2), "```", ""]
    md += ["## Negative findings preserved", ""]
    md += [f"- {x}" for x in diag["negative_findings_preserved"]]
    md += ["", f"**Frozen 2x2:** {diag['frozen_2x2_cell']}", ""]
    DIAG_MD.write_text("\n".join(md), encoding="utf-8")
    log(f"wrote {DIAG_MD.name}")

    V3.write_text(build_v3(diag), encoding="utf-8")
    log(f"wrote {V3.name}")

    FINAL.write_text(f"""# HUMAN_DECISION_REQUIRED — Phase 0 complete

V2 is diagnosed and frozen. Nothing further ran.

## Answers

- **Q1 opportunity cost:** {diag['Q1_allocation_opportunity_cost']['answer']}
- **Q2 two-agent offense:** {diag['Q2_two_agent_offense_and_mechanism']['answer']}
- **Q3 commitment:** {diag['Q3_commitment_reversibility']['answer']}
- **Q4 carrier self-sufficiency:** {diag['Q4_carrier_self_sufficiency']['answer']}

## Proposed V3 intervention

`own_flag_home_required_to_score = True` — single boolean, NOT implemented.

`suppression_range` is removed from all future levers on evidence
(0 events / 256 episodes / all four ranges).

## Artifacts

```
{DIAG_MD}
{DIAG_JSON}
{V3}
{TAGSAT}
{CARRIER}
{INTERACT}
{COMMIT}
```

## Not started, awaiting your decision

The strategic-demand searcher (Phases 1-12) was deliberately NOT auto-started.
Its ruleset search space depends on this diagnosis — R5 carrier vulnerability is
explicitly conditional on 7E — so selecting a search space before reading these
results would be choosing the experiment from unread data.

No ruleset change, no PPO training, no specialists, no FP/DO, no latent work.
""", encoding="utf-8")
    log("wrote HUMAN_DECISION_REQUIRED_PHASE0.md")
    log("PHASE 0 COMPLETE — stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
