r"""Role stage of the specialist qualification procedure (SPECIALIST_QUALIFICATION_V1_SPEC.json#ROLE_STAGE_unchanged).

Called by experiments/specialist_qualification.py only after the selected pair (A*, B*) QUALIFIED at D_gate.
Unchanged machinery, applied to the qualified pair:
  1. defenders  defender-only recipe for A* and B* (k = 1, frozen specialist ATTACK, N' teacher 0.1 -> 0 over
                50k-150k, cadence 4, 200k), pre-reserved seeds, authorized by a derived spec that only records the
                selected checkpoints (no new choices)
  2. shared k   strategy-conditioned-k amendment 1: S(k) = [V(A,k,A) + V(B,k,B)] / 2 over K(4) = {0, 1} on the
                pre-reserved role-dev block; exact tie -> smaller k; k_A = k_B = k*
  3. confirm    four cells on the pre-reserved fresh 128-seed block; win + margin; Delta_A, Delta_B; paired
                bootstrap 95% and 98.33%; no PASS/FAIL wording. Then STOP (no Stage 4, no 2v2/6v6).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
DEF_SUFFIX = "_sq_def_k1"
ALPHA, N_ATTEMPTS, N_BOOT, BOOT_SEED = 0.05, 3, 20_000, 7


def _auth_spec(q, sel: dict) -> str:
    """Derived authorization for the two defender runs: records only the selected checkpoints + frozen seeds."""
    p = SD / "SPECIALIST_QUALIFICATION_V1_4V4_ROLE_AUTH_SPEC.json"
    rs = q.spec["ROLE_STAGE_unchanged"]
    if not p.is_file():
        auth = []
        for side in "AB":
            c = q.cands[sel["selected"][side]]
            auth.append({"team_size": 4, "policy": side, "role_k_defend": 1, "run_label_suffix": DEF_SUFFIX,
                         "specialist_path": c["final"], "specialist_sha256": q_sha(q, c["final"]),
                         "seed": int(rs["defenders"]["training_seeds"][side]), "experiment_id": f"SQ_4V4_DEF_{side}_TRAIN"})
        p.write_text(json.dumps({
            "record_id": p.stem, "status": "FROZEN_DERIVED", "utc": q_now(), "confirmatory": False,
            "parent": q.spec["record_id"], "selection_record_sha256": q_sha(q, (q.out / "SQ_4V4_SELECTION_SEALED.json").relative_to(ROOT)),
            "gate_record_sha256": q_sha(q, (q.out / "SQ_4V4_GATE_SEALED.json").relative_to(ROOT)),
            "purpose": "authorizes the two role-stage defender runs for the qualified pair; recipe = frozen defender-only recipe",
            "TRAINING_AUTHORIZED": auth}, indent=2) + "\n", encoding="utf-8")
    return str(p.relative_to(ROOT)).replace("\\", "/")


def q_sha(q, rel) -> str:
    from experiments.specialist_qualification import sha
    return sha(rel)


def q_now() -> str:
    from experiments.specialist_qualification import now
    return now()


def _def_dir(side: str) -> dict:
    d = f"artifacts/scale_4v4_specialists/pi_{side}_specialist_4v4{DEF_SUFFIX}"
    return {"dir": d, "final": f"{d}/ckpts/final_pi_{side}_specialist_4v4{DEF_SUFFIX}.zip"}


def _def_argv(q, sel: dict, side: str, auth_rel: str) -> list[str]:
    spec_path = q.cands[sel["selected"][side]]["final"]
    seed = int(q.spec["ROLE_STAGE_unchanged"]["defenders"]["training_seeds"][side])
    return ["experiments/train_specialist_scale.py", "--team-size", "4", "--policy", side, "--seed", str(seed), "--device", "cuda",
            "--total-timesteps", "200000", "--entity-repair-enabled", "--entity-hidden-dim", "32",
            "--role-conditioning-enabled", "--role-hold-ticks", "8", "--role-fixed-for-episode", "--role-k-defend", "1",
            "--split-attack-defend-enabled", "--split-attack-defend-frozen-ckpt", spec_path,
            "--split-attack-defend-frozen-ckpt-sha256", q_sha(q, spec_path), "--load-path", spec_path,
            "--defend-teacher-lambda", "0.1", "--defend-teacher-lambda-end", "0.0",
            "--defend-teacher-decay-start-step", "50000", "--defend-teacher-decay-end-step", "150000",
            "--defend-teacher-cadence", "4", "--run-label-suffix", DEF_SUFFIX, "--symmetric-role-spec", auth_rel,
            "--experiment-id", f"SQ_4V4_DEF_{side}_TRAIN"]


def _side_args(q, sel: dict, side: str, k: int) -> list[str]:
    spec_path = q.cands[sel["selected"][side]]["final"]
    flag = "--pi-a-path" if side == "A" else "--pi-b-path"
    if k == 0:
        return [flag, spec_path]
    fa = ("--frozen-attack-path", "--frozen-attack-path-sha256") if side == "A" else ("--frozen-attack-path-b", "--frozen-attack-path-b-sha256")
    return [flag, _def_dir(side)["final"], fa[0], spec_path, fa[1], q_sha(q, spec_path)]


def _eval_argv(q, sel, label, reg, lo, hi, k, spec_rel) -> list[str]:
    a = ["experiments/eval_specialist_crossover_scaled.py", "--team-size", "4", "--spec", spec_rel, "--seed-base", str(lo),
         "--n-seeds", str(hi - lo + 1), "--label", label, "--device", "cuda", *_side_args(q, sel, "A", k), *_side_args(q, sel, "B", k)]
    if k:
        a += ["--role-fixed-for-episode", "--role-k-defend", "1"]
    if reg:
        a += ["--registry-experiment-id", reg]
    return a


def _confirm_spec(q) -> str:
    p = SD / "SPECIALIST_QUALIFICATION_V1_4V4_CONFIRM_EVAL_SPEC.json"
    if not p.is_file():
        p.write_text(json.dumps({"record_id": p.stem, "status": "FROZEN_CONFIRMATION_EVAL", "utc": q_now(), "confirmatory": True,
                                 "parent": q.spec["record_id"], "purpose": "evaluator spec for the final fresh confirmation only"},
                                indent=2) + "\n", encoding="utf-8")
    return str(p.relative_to(ROOT)).replace("\\", "/")


def _boot(x, a: float) -> list[float]:
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    b = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1)
    lo, hi = np.percentile(b, [100 * a / 2, 100 * (1 - a / 2)])
    return [float(lo), float(hi)]


def run(q, sel: dict) -> int:
    rs = q.spec["ROLE_STAGE_unchanged"]
    auth = _auth_spec(q, sel)
    # 1. defenders (both at once)
    q.run_parallel([(f"def_{s}", _def_argv(q, sel, s, auth), (lambda s=s: (ROOT / _def_dir(s)["final"]).is_file())) for s in "AB"])
    # 2. shared-k role selection
    rec_p = q.out / "SQ_4V4_ROLE_SELECTION_SEALED.json"
    if not rec_p.is_file():
        lo, hi = rs["role_dev_block"]
        dev_spec = q.eval_spec()
        q.run_parallel([(f"role_dev_k{k}", _eval_argv(q, sel, f"SQ_4V4_ROLE_DEV_K{k}", "SQ_4V4_ROLE_DEV", lo, hi, k, dev_spec) + ["--resume"],
                         (lambda k=k: q.sealed(f"SQ_4V4_ROLE_DEV_K{k}"))) for k in (0, 1)])
        seeds = list(range(lo, hi + 1))
        V = {k: q.cells(f"SQ_4V4_ROLE_DEV_K{k}", seeds) for k in (0, 1)}
        S = {k: (V[k][("A", "A")] + V[k][("B", "B")]) / 2 for k in (0, 1)}
        kstar = min(k for k in (0, 1) if S[k] == max(S.values()))
        rec_p.write_text(json.dumps({"record_id": rec_p.stem, "status": "SEALED_SELECTION", "utc": q_now(), "block": [lo, hi],
                                     "rule": "S(k) = [V(A,k,A) + V(B,k,B)]/2, k* = argmax, tie -> smaller k; k_A = k_B = k*",
                                     "intended_pole": {f"A@A_k{k}": V[k][("A", "A")] for k in (0, 1)} | {f"B@B_k{k}": V[k][("B", "B")] for k in (0, 1)},
                                     "S": {str(k): S[k] for k in (0, 1)}, "selected_k": kstar}, indent=2) + "\n", encoding="utf-8")
        q.log(f"ROLE SELECTION SEALED: k* = {kstar}  S = {S}")
    kstar = json.loads(rec_p.read_text(encoding="utf-8"))["selected_k"]
    # 3. final confirmation
    lo, hi = rs["final_confirmation"]["block"]
    label = "SQ_4V4_CONFIRM"
    q.run_parallel([("confirm", _eval_argv(q, sel, label, None, lo, hi, kstar, _confirm_spec(q)) + ["--resume"],
                     lambda: q.sealed(label))])
    from experiments.eval_hog_psp_v3 import _mean_ci
    seeds = list(range(lo, hi + 1))
    by: dict = {}
    with (SD / f"{label.lower()}_specialist_crossover_eval_rows.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by.setdefault((r["policy"][-1], r["pole"]), {})[int(r["seed"])] = (float(r["win"]), float(r["margin"]))
    out = {"record": "SQ_4V4_CONFIRM_READOUT", "utc": q_now(), "k_star": kstar, "pair": sel["selected"], "block": [lo, hi],
           "result_sha256": q_sha(q, q.result(label).relative_to(ROOT))}
    for i, field in enumerate(("win", "margin")):
        v = {key: np.array([d[s][i] for s in seeds]) for key, d in by.items()}
        res = {f"{p}@{qq}": float(v[(p, qq)].mean()) for p in "AB" for qq in "AB"}
        for name, x in (("Delta_A", v[("A", "A")] - v[("B", "A")]), ("Delta_B", v[("B", "B")] - v[("A", "B")])):
            c = _mean_ci(x)
            res[name] = {"mean": float(x.mean()), "std": float(x.std(ddof=1)), "ci95": [c["lcb95"], c["ucb95"]],
                         "ci9833": _boot(x, ALPHA / N_ATTEMPTS)}
        out[field] = res
    f = lambda s: f"{s['mean']:+.3f} ± {s['std']:.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"  # noqa: E731
    md = [f"# 4v4 qualified specialist pair ({sel['selected']['A']}, {sel['selected']['B']}) + shared-k roles (k* = {kstar}): "
          f"fresh confirmation {lo}..{hi}", "", "| | A@A | B@A | A@B | B@B | Δ_A (95%) | Δ_B (95%) |", "|---|---|---|---|---|---|---|"]
    for field in ("win", "margin"):
        t = out[field]
        md.append(f"| {field} | {t['A@A']:.3f} | {t['B@A']:.3f} | {t['A@B']:.3f} | {t['B@B']:.3f} | {f(t['Delta_A'])} | {f(t['Delta_B'])} |")
    w = out["win"]
    md += ["", f"98.33%: Δ_A [{w['Delta_A']['ci9833'][0]:+.3f}, {w['Delta_A']['ci9833'][1]:+.3f}], "
               f"Δ_B [{w['Delta_B']['ci9833'][0]:+.3f}, {w['Delta_B']['ci9833'][1]:+.3f}]",
           "", "Specialist qualification procedure (frozen before the existing-pair test); motivation disclosed in the spec."]
    out["table_markdown"] = "\n".join(md)
    (q.out / "SQ_4V4_CONFIRM_READOUT.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    (q.out / "SQ_4V4_CONFIRM_READOUT.md").write_text(out["table_markdown"] + "\n", encoding="utf-8")
    q.log("CONFIRMATION READOUT WRITTEN -- STOP (no Stage 4, no 2v2/6v6 without approval)")
    return 0
