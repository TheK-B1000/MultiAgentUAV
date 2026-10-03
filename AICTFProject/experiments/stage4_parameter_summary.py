r"""Provenance record for the Stage-4 parameter table, built from the source files (nothing typed in).

    python experiments/stage4_parameter_summary.py --team-size 2

Writes artifacts/strategic_demand/sppo/dual_branch_v1/matched128_<N>v<N>/<N>v<N>_stage4_parameter_summary.json
(+ .md). For every system: checkpoint path(s) + sha256, unique actor (policy) parameters with the critic
excluded -- the same count the Stage-4 trainer records (TD.actor_parameters) -- and holdout agreement with the
A and B teachers. For the Stage-4 top-50 diagnostic: the seed-list file + sha256, each result file + sha256,
Delta_A / Delta_B with 95% paired bootstrap intervals recomputed from the sealed rows and cross-checked against
each sealed result. Fails closed on any missing file, a seed-list mismatch, or a recomputed Delta that
disagrees with its sealed result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"

#: the repaired 1M specialists = No-Role Specialists (and the dual-branch warm starts), per scale
NO_ROLE = {
    2: ("artifacts/scale_2v2_specialists/pi_A_specialist_2v2_std_entity_repair/ckpts/final_pi_A_specialist_2v2_std_entity_repair.zip",
        "artifacts/scale_2v2_specialists/pi_B_specialist_2v2_std_entity_repair/ckpts/final_pi_B_specialist_2v2_std_entity_repair.zip"),
    4: ("artifacts/scale_4v4_specialists/pi_A_specialist_4v4_b3_entity_repair/ckpts/final_pi_A_specialist_4v4_b3_entity_repair.zip",
        "artifacts/scale_4v4_specialists/pi_B_specialist_4v4_b3_entity_repair_corrected/ckpts/final_pi_B_specialist_4v4_b3_entity_repair_corrected.zip"),
    6: ("artifacts/scale_6v6_specialists/pi_A_specialist_6v6_c2_entity_repair/ckpts/final_pi_A_specialist_6v6_c2_entity_repair.zip",
        "artifacts/scale_6v6_specialists/pi_B_specialist_6v6_c2_entity_repair/ckpts/final_pi_B_specialist_6v6_c2_entity_repair.zip"),
}
STUDENTS = {  # paper name -> (student dir tag, OWN50 label suffix)
    "Share-Encoder": ("share_encoder", "SHARE_ENCODER"),
    "Fully Shared+z+r": ("fully_shared_z_r", "FULLY_SHARED_ZR"),
    "Role-only (r)": ("role_only", "ROLE_ONLY"),
}


def sha(p: Path) -> str:
    if not p.is_file():
        raise SystemExit(f"FAIL-CLOSED: missing {p}")
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p: Path) -> str:
    return str(p.relative_to(ROOT)).replace("\\", "/")


def stat(x) -> dict:
    from experiments.eval_hog_psp_v3 import _mean_ci
    x = np.asarray(x, dtype=np.float64)
    c = _mean_ci(x)
    return {"mean": float(x.mean()), "std": float(x.std(ddof=1)), "lcb95": float(c["lcb95"]), "ucb95": float(c["ucb95"]),
            "n": int(x.size)}


def actor_params(paths: list[str], n: int) -> list[dict]:
    import experiments.collect_distillation_states as C
    import experiments.r2_learned_crossover as R2
    from rl import teacher_distillation as TD
    from rl.custom_ppo import load_custom_ppo_policy
    C.N_AGENTS, R2.AGENTS = n, n
    env = R2.build_env("cpu", 99_999_001)
    os_, as_ = env.observation_space, env.action_space
    env.close()
    out = []
    for p in paths:
        m = load_custom_ppo_policy(str(ROOT / p), os_, as_, device="cpu").model
        out.append({"path": p, "sha256": sha(ROOT / p),
                    "actor_params": sum(int(t.numel()) for _, t in TD.actor_parameters(m)),
                    "all_params_incl_critic": sum(int(t.numel()) for t in m.parameters())})
    return out


def cells(path: Path, key: str, seeds: list[int]) -> dict:
    by: dict = {}
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by.setdefault((r[key], r["pole"]), {})[int(r["seed"])] = float(r["win"])
    for k, d in by.items():
        if sorted(d) != seeds:
            raise SystemExit(f"FAIL-CLOSED: {path.name} cell {k} is not on exactly the top-50 seeds")
    return {k: np.array([d[s] for s in seeds]) for k, d in by.items()}


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team-size", type=int, required=True, choices=(2, 4, 6))
    n = ap.parse_args().team_size
    N = f"{n}V{n}"
    out_dir = SD / "dual_branch_v1" / f"matched128_{n}v{n}"
    man = json.loads((ROOT / f"{n}v{n}" / "dual_branch_deploy_manifest.json").read_text(encoding="utf-8"))

    # ---- parameters
    ours_paths = [man[k]["path"] for k in ("pi_A_attack", "pi_A_defend", "pi_B_attack", "pi_B_defend")]
    for k in ("pi_A_attack", "pi_A_defend", "pi_B_attack", "pi_B_defend"):
        if sha(ROOT / man[k]["path"]) != man[k]["sha256"]:
            raise SystemExit(f"FAIL-CLOSED: {k} sha differs from the deploy manifest")
    ours = actor_params(ours_paths, n)
    norole = actor_params(list(NO_ROLE[n]), n)
    ours_total = sum(x["actor_params"] for x in ours)
    systems = {
        "Ours": {"group": "Proposed", "organization": "4 separate policy networks: A-ATTACK, A-DEFEND, B-ATTACK, B-DEFEND",
                 "checkpoints": ours, "unique_actor_params": ours_total, "holdout_agreement": None,
                 "deploy_manifest": f"{n}v{n}/dual_branch_deploy_manifest.json"},
    }
    for name, (tag, _lab) in STUDENTS.items():
        d = SD / "suite_sharing_std" / f"{n}v{n}_stage4" / tag
        fz = json.loads((d / "STUDENT_FROZEN.json").read_text(encoding="utf-8"))
        if fz.get("status") != "FROZEN_STUDENT":
            raise SystemExit(f"FAIL-CLOSED: {name} not FROZEN_STUDENT")
        ck = d / "ckpts" / f"final_{tag}_{n}v{n}.pt"
        h = fz["final_holdout"]
        systems[name] = {
            "group": "Proposed" if name == "Fully Shared+z+r" else "Comparisons",
            "organization": {"Share-Encoder": "shared encoder, separate per-strategy heads",
                             "Fully Shared+z+r": "1 shared policy network conditioned on z and r",
                             "Role-only (r)": "1 shared policy network conditioned on r only (no z)"}[name],
            "checkpoints": [{"path": rel(ck), "sha256": sha(ck)}],
            "student_frozen_record": {"path": rel(d / "STUDENT_FROZEN.json"), "sha256": sha(d / "STUDENT_FROZEN.json")},
            "unique_actor_params": int(fz["unique_actor_params"]),
            "holdout_agreement": {"A_teacher": h["holdout_agree_z0_vs_piA"], "B_teacher": h["holdout_agree_z1_vs_piB"]},
        }
    systems["No-Role Specialists"] = {"group": "Comparisons", "organization": "2 separate policy networks: pi_A, pi_B",
                                      "checkpoints": norole, "unique_actor_params": sum(x["actor_params"] for x in norole),
                                      "holdout_agreement": None}
    for s in systems.values():
        s["params_M"] = round(s["unique_actor_params"] / 1e6, 2)
        s["reduction_vs_ours_pct"] = round(100 * (1 - s["unique_actor_params"] / ours_total), 1)

    # ---- Stage-4 top-50 diagnostic
    sel = json.loads((out_dir / f"DUAL_BRANCH_{N}_OWN_TOP50.json").read_text(encoding="utf-8"))
    seed_file = out_dir / f"DUAL_BRANCH_{N}_OWN_TOP50_seed_ids.json"
    seeds = sorted(int(s) for s in json.loads(seed_file.read_text(encoding="utf-8")))
    m128 = SD / f"posthoc_matched128_{n}v{n}_dual_branch_specialist_crossover_eval_rows.csv"
    by: dict = {}
    with m128.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by.setdefault((r["policy"][-1], r["pole"]), {})[int(r["seed"])] = float(r["win"])
    O = {k: np.array([d[s] for s in seeds]) for k, d in by.items()}
    diag = {"Ours": {"source_rows": {"path": rel(m128), "sha256": sha(m128)},
                     "note": "matched-128 rows restricted to exactly the top-50 seed IDs (deterministic per seed)",
                     "V": {f"{p}@{q}": float(O[(p, q)].mean()) for p in "AB" for q in "AB"},
                     "Delta_A": stat(O[("A", "A")] - O[("B", "A")]), "Delta_B": stat(O[("B", "B")] - O[("A", "B")])}}
    for name, (_tag, lab) in STUDENTS.items():
        label = f"OWN50_{N}_STAGE4_{lab}"
        res_p, rows_p = SD / f"{label}_CROSSOVER_EVAL_RESULT.json", SD / f"{label.lower()}_crossover_eval_rows.csv"
        res = json.loads(res_p.read_text(encoding="utf-8"))
        if res.get("status") != "SEALED" or sorted(res["seeds"].get("seed_ids") or []) != seeds:
            raise SystemExit(f"FAIL-CLOSED: {label} not SEALED on exactly the top-50 seeds")
        V = cells(rows_p, "z", seeds)
        e = {"result": {"path": rel(res_p), "sha256": sha(res_p)}, "rows": {"path": rel(rows_p), "sha256": sha(rows_p)},
             "episodes": res.get("total_episodes")}
        if ("1", "A") in V:
            e["V"] = {"A@A": float(V[("0", "A")].mean()), "B@A": float(V[("1", "A")].mean()),
                      "A@B": float(V[("0", "B")].mean()), "B@B": float(V[("1", "B")].mean())}
            e["Delta_A"] = stat(V[("0", "A")] - V[("1", "A")])
            e["Delta_B"] = stat(V[("1", "B")] - V[("0", "B")])
            for k in ("delta_A", "delta_B"):
                if abs(res["PRIMARY_GATE"][k]["mean"] - e[k.replace("delta", "Delta")]["mean"]) > 1e-9:
                    raise SystemExit(f"FAIL-CLOSED: recomputed {k} for {label} differs from its sealed result")
        else:
            e["win_rate_pole_A"] = float(V[("0", "A")].mean())
            e["win_rate_pole_B"] = float(V[("0", "B")].mean())
            e["note"] = "no z: one policy on both poles, so no crossover Delta (report Pole A / Pole B win rates)"
        diag[name] = e

    rec = {
        "record": f"{n}v{n}_stage4_parameter_summary", "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_head": git_head(), "generator": "experiments/stage4_parameter_summary.py",
        "naming": "EXPERIMENTAL_FRAMING_OURS_TEACHERS_SHARED_V1_AMENDMENT_1/2 (Ours; Fully Shared+z+r proposed, being evaluated)",
        "parameter_definition": "unique actor (policy) parameters of the deployed networks, CRITIC EXCLUDED; the same count "
                                "the Stage-4 trainer records (rl.teacher_distillation.actor_parameters). Reduction is vs Ours.",
        "systems": systems,
        "stage4_top50_diagnostic": {
            "title": "Stage-4 Top-50 diagnostic (post-hoc; matched-128 is the primary evidence for Ours)",
            "seed_list": {"path": rel(seed_file), "sha256": sha(seed_file), "n": len(seeds)},
            "same_seeds_for_every_row": True,
            "selection": {"rule": sel["rule"], "score_distribution": sel["score_distribution"], "cutoff": sel["cutoff"],
                          "composition": f"{sel['cutoff']['seeds_above']} score-2 seeds + "
                                         f"{len(seeds) - sel['cutoff']['seeds_above']} score-1 seeds chosen by the "
                                         f"pre-specified ascending-seed-ID tie-break"},
            "statistics": "Delta = mean of per-seed paired differences; 95% paired percentile bootstrap (n=20000, rng 7)",
            "rows": diag,
        },
        "reporting_rules": ["no PASS/FAIL column: effects + intervals only",
                            "Role-only reported as Pole A / Pole B win rates, not A@A / A@B",
                            "Fully Shared+z+r described as proposed, outcome-level separation only, until the "
                            "behavioral-signature check is sealed (amendment 2)"],
    }
    js = out_dir / f"{n}v{n}_stage4_parameter_summary.json"
    js.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
    f = lambda s: f"{s['mean']:+.2f} [{s['lcb95']:+.2f}, {s['ucb95']:+.2f}]"  # noqa: E731
    md = [f"# {n}v{n} Stage-4 parameter table + Stage-4 Top-50 diagnostic", "",
          "| Condition | Params (M) | Reduction | Agreement A / B | Δ_A | Δ_B |", "|---|---|---|---|---|---|"]
    for name in ("Ours", "Share-Encoder", "Fully Shared+z+r", "Role-only (r)"):
        s, d = systems[name], diag[name]
        ag = s["holdout_agreement"]
        ag_txt = "N/A" if ag is None else f"{ag['A_teacher']:.3f} / {ag['B_teacher']:.3f}"
        md.append(f"| {name} | {s['params_M']:.2f} | {s['reduction_vs_ours_pct']:.1f}% | {ag_txt} | "
                  f"{f(d['Delta_A']) if 'Delta_A' in d else 'N/A'} | {f(d['Delta_B']) if 'Delta_B' in d else 'N/A'} |")
    r = diag["Role-only (r)"]
    md += ["", f"Role-only (r): wins {r['win_rate_pole_A']:.2f} on Pole A, {r['win_rate_pole_B']:.2f} on Pole B.",
           f"No-Role Specialists: {systems['No-Role Specialists']['params_M']:.2f}M "
           f"({systems['No-Role Specialists']['reduction_vs_ours_pct']:.1f}% vs Ours).",
           f"Seeds: {rec['stage4_top50_diagnostic']['selection']['composition']}; identical for every row."]
    js.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n-> {js}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
