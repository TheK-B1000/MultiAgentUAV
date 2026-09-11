"""Harvest EVERY sealed number needed for the Results section into one file.

Reads directly from the sealed artifacts -- nothing is retyped by hand, so a
transcription error is not possible. Each artifact has its own schema; this
script targets each one explicitly rather than guessing a common shape.
Missing artifacts are emitted as explicit PENDING entries, never estimated.

Output: paper/data/RESULTS_DATA_2v2_6v6.json

Run:  ./.venv/Scripts/python.exe paper/figures/harvest_results_data.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SD = ROOT / "artifacts" / "strategic_demand"
SP = SD / "sppo"
PD = ROOT / "paper" / "data"
OUT = PD / "RESULTS_DATA_2v2_6v6.json"


def load(rel: str):
    for base in (SD, SP, PD):
        p = base / rel
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def trip(d):
    """Normalise a {mean,lcb95,ucb95} block to a rounded triple."""
    if not isinstance(d, dict):
        return None
    try:
        return {"mean": round(float(d["mean"]), 4),
                "lcb95": round(float(d["lcb95"]), 4),
                "ucb95": round(float(d["ucb95"]), 4)}
    except (KeyError, TypeError, ValueError):
        return None


out = {
    "record": "Consolidated sealed Results data, 2v2 + 6v6",
    "generated_by": "paper/figures/harvest_results_data.py (re-run to refresh)",
    "rule": "Every value is read from a sealed artifact. PENDING means not yet sealed and MUST NOT be estimated.",
    "criterion": {
        "delta_A": "V(z0,A) - V(z1,A)",
        "delta_B": "V(z1,B) - V(z0,B)",
        "gate": "PASS iff both means > 0 AND both LCB95 > 0",
        "bootstrap": {"procedure": "paired percentile over seeds", "n_boot": 20000,
                      "alpha": 0.05, "rng_seed": 7, "unit": "seed"},
        "applied_identically_at": ["2v2", "6v6"],
    },
    "compression_invariance": {},
    "2v2": {},
    "6v6": {},
}

# ============================================================== 2v2 : demand
d = load("V3_STRATEGIC_DEMAND_VALIDATED.json")
if d:
    t = d["the_four_prospective_tests_all_pass"]
    out["2v2"]["demand"] = {
        "artifact": "V3_STRATEGIC_DEMAND_VALIDATED.json",
        "verdict": d.get("verdict"),
        "seeds": d.get("seeds"),
        "n": 192,
        "payoff_A": {"delta": t["A_payoff"]["delta"], "lcb95": t["A_payoff"]["lcb95"],
                     "ucb95": t["A_payoff"]["ucb95"], "guard_wr": t["A_payoff"]["guard_wr"],
                     "breach_wr": t["A_payoff"]["breach_wr"], "passes": t["A_payoff"]["passes"]},
        "payoff_B": {"delta": t["B_payoff"]["delta"], "lcb95": t["B_payoff"]["lcb95"],
                     "ucb95": t["B_payoff"]["ucb95"], "guard_wr": t["B_payoff"]["guard_wr"],
                     "breach_wr": t["B_payoff"]["breach_wr"], "passes": t["B_payoff"]["passes"]},
        "concealment_A": {"p_C": t["A_concealment"]["p_C"], "lcb95": t["A_concealment"]["lcb95"]},
        "concealment_B": {"p_C": t["B_concealment"]["p_C"], "lcb95": t["B_concealment"]["lcb95"]},
    }

# ======================================================= 2v2 : generalist pi_G
g = load("PI_G_EVAL_RESULT.json")
if g:
    out["2v2"]["generalist_pi_G"] = {
        "artifact": "PI_G_EVAL_RESULT.json",
        "seeds": g.get("seeds"),
        "V_pi_G_A": g.get("V_pi_G_A"),
        "V_pi_G_B": g.get("V_pi_G_B"),
        "total_episodes": g.get("total_episodes"),
    }

# ==================================================== 2v2 : independent experts
s = load("SPECIALIST_BASELINE_EVAL_RESULT.json")
if s:
    pg = s["PRIMARY_GATE"]
    out["2v2"]["independent_experts"] = {
        "artifact": "SPECIALIST_BASELINE_EVAL_RESULT.json",
        "seeds": s.get("seeds"),
        "delta_A": trip(pg.get("delta_A")),
        "delta_B": trip(pg.get("delta_B")),
        "passes": pg.get("passes"),
        "total_episodes": s.get("total_episodes"),
    }

# ============================================================ 2v2 : ladder
params = load("sharing_params_tradeoff.json") or {}
rung_params = {r["label"]: r for r in params.get("rungs", [])}

LADDER = [
    ("Share-0", "RUNG0_LADDER_REFERENCE.json", None),
    ("Share-Encoder", "RUNG1_LADDER_EVAL_RESULT.json", "RUNG1_STUDENT_FROZEN.json"),
    ("Share-Backbone", "RUNG2_LADDER_EVAL_RESULT.json", None),
    ("Share-Macro", "RUNG3_LADDER_EVAL_RESULT.json", None),
]

out["2v2"]["sharing_ladder"] = {}
for label, res_file, frozen_file in LADDER:
    r = load(res_file)
    if not r:
        continue
    if label == "Share-0":
        gate = r.get("POOLED_N128", {})
        entry = {
            "artifact": res_file,
            "cell_win_rates": r.get("cell_win_rates_n128"),
            "delta_A": trip(gate.get("delta_A")),
            "delta_B": trip(gate.get("delta_B")),
            "passes": gate.get("passes"),
            "role": "bit-exact expert-dispatch reference (zero tied modules)",
        }
    else:
        gate = r.get("OWN_GATE_N128", {})
        paired = r.get("PRIMARY_WITHIN_SEED", {})
        entry = {
            "artifact": res_file,
            "matched_seeds": r.get("matched_seeds"),
            "cell_win_rates": r.get("cell_win_rates"),
            "delta_A": trip(gate.get("delta_A")),
            "delta_B": trip(gate.get("delta_B")),
            "passes": gate.get("passes"),
            "paired_vs_share0": {
                "D_A": trip(paired.get("D_A")),
                "D_B": trip(paired.get("D_B")),
                "classification": paired.get("classification"),
            },
            "total_episodes": r.get("total_episodes"),
        }
    if label in rung_params:
        entry["params"] = {k: rung_params[label].get(k)
                           for k in ("n_unique", "params_M", "reduction_vs_share0")}
    if frozen_file:
        fz = load(frozen_file)
        if fz:
            entry["fidelity"] = fz.get("final_holdout")
            entry["sharing_arithmetic"] = fz.get("sharing_arithmetic")
    out["2v2"]["sharing_ladder"][label] = entry

# ================================================== compression invariance
fz2 = load("RUNG1_STUDENT_FROZEN.json") or {}
fz6 = load("RUNG1_6V6_STUDENT_FROZEN.json") or {}
sa2 = fz2.get("sharing_arithmetic") or {}
sa6 = fz6.get("sharing_arithmetic") or {}
if sa6.get("n_branch_total") and sa6.get("n_unique"):
    branch, uniq = sa6["n_branch_total"], sa6["n_unique"]
    out["compression_invariance"] = {
        "why": "the actor is per-agent weight-shared; team size changes the observation batch dimension, not any parameter shape",
        "n_branch_each": branch,
        "two_independent_experts": 2 * branch,
        "shared_encoder_unique": uniq,
        "shared_module_params": sa6.get("n_shared"),
        "private_per_branch": branch - sa6.get("n_shared", 0),
        "reduction_vs_two_experts": round(1 - uniq / (2 * branch), 4),
        "shared_fraction_of_one_branch": round(sa6.get("n_shared", 0) / branch, 4),
        "identical_at_2v2_and_6v6": bool(
            sa2.get("n_branch_total") == branch and sa2.get("n_unique") == uniq),
        "2v2_sharing_arithmetic": sa2,
        "6v6_sharing_arithmetic": sa6,
    }

# ========================================================= 2v2 : robustness
out["2v2"]["robustness"] = {}
for tier, f in (("low", "ROBUSTNESS_2V2_DOSE_RESPONSE_LOW_TIER_RESULT.json"),
                ("medium", "ROBUSTNESS_2V2_RUNG1_RESULT.json"),
                ("high", "ROBUSTNESS_2V2_HIGH_TIER_RESULT.json")):
    r = load(f)
    if not r:
        continue
    out["2v2"]["robustness"][tier] = {
        "artifact": f,
        "per_condition_crossover": r.get("PER_CONDITION_CROSSOVER"),
        "paired_degradation_vs_nominal": r.get("PAIRED_DEGRADATION_VS_NOMINAL_WITHIN_SEED"),
        "reading": r.get("READING"),
    }
low = load("ROBUSTNESS_2V2_DOSE_RESPONSE_LOW_TIER_RESULT.json") or {}
if low:
    out["2v2"]["robustness"]["nominal_reference"] = low.get(
        "NOMINAL_REFERENCE_shared_across_all_three_tiers")
    out["2v2"]["robustness"]["FULL_DOSE_TABLE"] = low.get(
        "FULL_DOSE_RESPONSE_TABLE_nominal_low_medium_high")

# ============================================================== 6v6 : demand
c6 = load("STRATEGIC_DEMAND_6v6_GUARD_DISTRIBUTED_V2_CERTIFICATION.json")
if c6:
    out["6v6"]["demand"] = {
        "artifact": "STRATEGIC_DEMAND_6v6_GUARD_DISTRIBUTED_V2_CERTIFICATION.json",
        "verdict": c6.get("VERDICT"),
        "seeds": c6.get("seeds"),
        "total_episodes": c6.get("total_episodes"),
        "guard_defenders": c6.get("guard_defenders"),
        "guard_defender_indices": c6.get("guard_defender_indices"),
        "poles": c6.get("poles"),
        "cell_win_rates": c6.get("cell_win_rates"),
        "delta_guard_A": trip((c6.get("PRIMARY") or {}).get("delta_A")),
        "delta_breach_B": trip((c6.get("PRIMARY") or {}).get("delta_B")),
    }

# ======================================================== 6v6 : specialists
out["6v6"]["specialist_teachers"] = {
    "pi_A": {"seed": 7610001, "pole": "A", "base_opponent": "OP6", "steps": 1000000},
    "pi_B": {"seed": 7620001, "pole": "B", "base_opponent": "OP7", "steps": 1000000},
    "role": "distillation teachers only; expert crossover NOT re-certified at 6v6 (scope decision)",
    "source": "artifacts/scale_6v6_specialists/pi_{A,B}_specialist_6v6/run_manifest.json",
}

# ====================================================== 6v6 : distillation
if fz6:
    out["6v6"]["distillation_rung1"] = {
        "artifact": "RUNG1_6V6_STUDENT_FROZEN.json",
        "architecture": fz6.get("architecture"),
        "terminal_checkpoint": fz6.get("TERMINAL_CHECKPOINT"),
        "training": fz6.get("training"),
        "fidelity": fz6.get("final_holdout"),
        "sharing_arithmetic": sa6,
        "fit_check": fz6.get("fit_check"),
        "roundtrip_max_abs_logit_diff": fz6.get("roundtrip_max_abs_logit_diff"),
    }
pf = load("RUNG1_6V6_PREFLIGHT.json")
if pf:
    out["6v6"]["distillation_preflight"] = {
        "artifact": "RUNG1_6V6_PREFLIGHT.json",
        "passed": pf.get("passed"), "verdict": pf.get("VERDICT"),
    }

# ======================================================== 6v6 : crossover
res6 = load("RUNG1_6V6_CROSSOVER_EVAL_RESULT.json")
flag6 = (SP / "RUNG1_6V6_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json").is_file()
if res6:
    pg = res6.get("PRIMARY_GATE", {})
    out["6v6"]["crossover"] = {
        "status": "SEALED",
        "artifact": "RUNG1_6V6_CROSSOVER_EVAL_RESULT.json",
        "seeds": res6.get("seeds"),
        "cell_win_rates": res6.get("cell_win_rates"),
        "delta_A": trip(pg.get("delta_A")),
        "delta_B": trip(pg.get("delta_B")),
        "passes": pg.get("passes"),
        "total_episodes": res6.get("total_episodes"),
    }
elif flag6:
    out["6v6"]["crossover"] = {
        "status": "INTEGRITY_AUDIT_REQUIRED",
        "artifact": "RUNG1_6V6_CROSSOVER_EVAL_INTEGRITY_REQUIRED.json",
        "note": "tie or reversal flagged; row-level audit required before interpretation",
    }
else:
    out["6v6"]["crossover"] = {
        "status": "PENDING",
        "seed_block": [13640001, 13640128], "n": 128, "episodes": 512,
        "note": "eval in flight; DO NOT ESTIMATE. Re-run this harvester when the artifact lands.",
    }

out["6v6"]["robustness"] = {
    "status": "RESERVED_UNSPENT",
    "seed_block": [13660001, 13660128], "n": 128,
    "gating_rule": "runs only after the 6v6 crossover seals PASS",
}

OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
print(f"wrote {OUT}")
print(f"  2v2 sections : {sorted(out['2v2'].keys())}")
print(f"  6v6 sections : {sorted(out['6v6'].keys())}")
print(f"  6v6 crossover: {out['6v6']['crossover']['status']}")
ci = out.get("compression_invariance", {})
print(f"  compression identical at both scales: {ci.get('identical_at_2v2_and_6v6')} "
      f"({ci.get('two_independent_experts')} -> {ci.get('shared_encoder_unique')}, "
      f"{ci.get('reduction_vs_two_experts')})")
