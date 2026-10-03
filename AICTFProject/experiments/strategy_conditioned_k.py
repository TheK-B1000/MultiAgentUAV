r"""Strategy-conditioned role allocation k = k(z): one team-size-generic procedure (2v2 / 4v4 / 6v6).

    python experiments/strategy_conditioned_k.py --team-size 4 --verify     # candidate checks only, no side effects
    python experiments/strategy_conditioned_k.py --team-size 4 --freeze     # frozen spec + dev/confirmation seeds
    python experiments/strategy_conditioned_k.py --team-size 4 --run        # dev selection -> freeze k -> confirmation -> readout -> STOP

Revised hypothesis motivated by the sealed 4v4 results (V1 k=2, R1 k=1): a strategy is its learned behavior AND
its team organization. Fairness is PROCEDURAL symmetry: A and B get the same candidate set, selection objective,
development budget, tie-break and confirmation procedure; the selected k_A and k_B may differ. Not the original
fixed-k method; no historical asymmetric system is compared or revived.

  k_role(N) = max(1, floor(N/3 + 1/2)) = max(1, (2N + 3) // 6);   K(N) = {0, k_role(N)}  (2v2 {0,1}, 4v4 {0,1}, 6v6 {0,2})
  k = 0 : the sealed repaired specialist controls all N agents (no DEFEND role)
  k > 0 : defender-only construction -- ATTACK slots = the sealed specialist (frozen), DEFEND slots = its trained
          defender; roles by CLOSEST_DEFENDS(k), fixed for the episode
  select: k_p* = argmax_{k in K(N)} V(Pi_{p,k}, p) on the intended pole only (A on Pole A, B on Pole B), one fresh
          development block for every candidate; exact tie -> the smaller k. Never crossover, never Delta.
  confirm: the selected pair on a SEPARATE fresh 128-seed block, four cells; Delta_A, Delta_B, win and margin,
           paired bootstrap 95% and the multiplicity-aware 98.33% (1 - 0.05/3).
The dev evaluator always plays both policies on both poles; the selection reads ONLY the intended-pole cells and
the other cells are recorded as excluded. Stage 4, 2v2 and 6v6 are never started here.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
N_DEV, N_CONF = 64, 128
ALPHA, N_ATTEMPTS, N_BOOT, BOOT_SEED = 0.05, 3, 20_000, 7

#: sealed repaired 1M specialists (k = 0 candidates and the frozen ATTACK of k > 0)
SPECIALISTS = {
    2: {"A": ("artifacts/scale_2v2_specialists/pi_A_specialist_2v2_std_entity_repair/ckpts/final_pi_A_specialist_2v2_std_entity_repair.zip",
              "858805dde3588a686868ae15cfcc4b4c2d63f30d23222dfe6ae421ab7a563b7f"),
        "B": ("artifacts/scale_2v2_specialists/pi_B_specialist_2v2_std_entity_repair/ckpts/final_pi_B_specialist_2v2_std_entity_repair.zip",
              "9ee024ad6356ea79ee765c7d55bba184a510a73fe8a4738a1f7203630636fd0d")},
    4: {"A": ("artifacts/scale_4v4_specialists/pi_A_specialist_4v4_b3_entity_repair/ckpts/final_pi_A_specialist_4v4_b3_entity_repair.zip",
              "94dde69d091a79344db3390d5464dbb4bcf51677a175df93969ab252b2e0f478"),
        "B": ("artifacts/scale_4v4_specialists/pi_B_specialist_4v4_b3_entity_repair_corrected/ckpts/final_pi_B_specialist_4v4_b3_entity_repair_corrected.zip",
              "021342c84bbe3aa37d70ed609d898bba3ea85affb53852324f92922c72f70e12")},
    6: {"A": ("artifacts/scale_6v6_specialists/pi_A_specialist_6v6_c2_entity_repair/ckpts/final_pi_A_specialist_6v6_c2_entity_repair.zip",
              "3298000480acd899e715eb78312938a9d20610a1eb0bcf73d382fd268577009e"),
        "B": ("artifacts/scale_6v6_specialists/pi_B_specialist_6v6_c2_entity_repair/ckpts/final_pi_B_specialist_6v6_c2_entity_repair.zip",
              "fc0043235d3abc23f87f5157a46920921f5eced0e9aa252347084d2b500258b5")},
}
#: candidate defender run dirs for k = k_role(N) (defender-only construction); must pass verify_defender()
DEFENDERS = {
    2: {"A": "artifacts/scale_2v2_specialists/pi_A_specialist_2v2_std_split_defend_k1",
        "B": "artifacts/scale_2v2_specialists/pi_B_specialist_2v2_sym_B_top50"},
    4: {"A": "artifacts/scale_4v4_specialists/pi_A_specialist_4v4_sym_defonly_k1",
        "B": "artifacts/scale_4v4_specialists/pi_B_specialist_4v4_sym_defonly_k1"},
    6: {"A": "artifacts/scale_6v6_specialists/pi_A_specialist_6v6_sym_A_k2_top50",
        "B": "artifacts/scale_6v6_specialists/pi_B_specialist_6v6_sym_B_top50"},
}
#: the frozen defender recipe every k > 0 candidate must match (identical for A and B, every N)
RECIPE = {"split_attack_defend_enabled": True, "role_fixed_for_episode": True, "role_hold_ticks": 8,
          "defend_teacher_lambda": 0.1, "defend_teacher_lambda_end": 0.0, "defend_teacher_decay_start_step": 50000,
          "defend_teacher_decay_end_step": 150000, "defend_teacher_cadence": 4, "entity_repair_enabled": True,
          "entity_hidden_dim": 32, "total_timesteps": 200000}
#: The ONLY resolved-config keys allowed to differ between the A and B defender runs of one scale (user 2026-10-03).
#: Every other key -- every scientific setting -- must match exactly.
AB_DIFF_OK = {
    # each strategy trains against its own intended opponent
    "fixed_opponent_tag", "opponent_pool", "episode_csv_path",
    # each side's own sealed specialist (verify_defender separately requires it to equal SPECIALISTS[n][side])
    "load_path", "split_attack_defend_frozen_ckpt", "split_attack_defend_frozen_ckpt_sha256",
    # per-run identity / output naming
    "seed", "checkpoint_dir", "metrics_csv_path", "run_tag",
}


def k_role(n: int) -> int:
    return max(1, (2 * int(n) + 3) // 6)


def menu(n: int) -> list[int]:
    return [0, k_role(n)]


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(p) -> str:
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def paths(n: int) -> dict:
    t = f"{n}V{n}"
    out = ROOT / "strategy_conditioned_k" / f"{n}v{n}"
    return {"out": out, "spec": SD / f"STRATEGY_CONDITIONED_K_{t}_SPEC.json",
            "dev_spec": SD / f"STRATEGY_CONDITIONED_K_{t}_DEV_EVAL_SPEC.json",
            "dev_block": f"SCK_{t}_DEV_SELECTION", "dev_labels": {0: f"SCK_{t}_DEV_K0", k_role(n): f"SCK_{t}_DEV_K{k_role(n)}"},
            "conf_label": f"SCK_{t}_CONFIRM", "selection": out / f"SCK_{t}_SELECTION_SEALED.json",
            "readout": out / f"SCK_{t}_CONFIRM_READOUT"}


# ------------------------------------------------------------------ candidate verification
def _run_config(run_dir: str) -> dict:
    f = glob.glob(str(ROOT / run_dir / "*_run_config.json"))
    if not f:
        raise FileNotFoundError(run_dir)
    return json.loads(Path(f[0]).read_text(encoding="utf-8")).get("resolved_ppo_config") or {}


def verify_defender(n: int, side: str) -> dict:
    """A k > 0 candidate is reusable only if its recorded recipe equals RECIPE, k = k_role(N), it is NOT dual-branch,
    its frozen ATTACK is exactly the sealed specialist (sha), it was warm-started from it, and its final exists."""
    d = DEFENDERS[n][side]
    spec_path, spec_sha = SPECIALISTS[n][side]
    final = f"{d}/ckpts/final_{Path(d).name}.zip"
    r = {"run_dir": d, "final": final, "problems": []}
    try:
        c = _run_config(d)
    except FileNotFoundError:
        r["problems"].append("defender run not present (not trained yet)")
        return r
    for k, v in RECIPE.items():
        if c.get(k) != v:
            r["problems"].append(f"{k}={c.get(k)!r} != {v!r}")
    if int(c.get("role_k_defend", -1)) != k_role(n):
        r["problems"].append(f"role_k_defend={c.get('role_k_defend')} != k_role({n})={k_role(n)}")
    if bool(c.get("dual_branch_role_composite_enabled", False)):
        r["problems"].append("dual-branch run (ATTACK was trained) -- not the defender-only construction")
    if str(c.get("split_attack_defend_frozen_ckpt_sha256", "")).lower() != spec_sha:
        r["problems"].append("frozen ATTACK is not the sealed specialist")
    if str(c.get("load_path", "")).replace("\\", "/") != spec_path:
        r["problems"].append(f"warm start {c.get('load_path')!r} is not the sealed specialist")
    if not (ROOT / final).is_file():
        r["problems"].append("final checkpoint missing")
    else:
        r["final_sha256"] = sha(final)
    r["recipe"] = {k: c.get(k) for k in [*RECIPE, "role_k_defend", "seed"]}
    return r


def verify(n: int) -> dict:
    out = {"team_size": n, "menu": menu(n), "k_role": k_role(n), "specialists": {}, "defenders": {}, "ab_config_diff": None}
    for side in "AB":
        p, s = SPECIALISTS[n][side]
        out["specialists"][side] = {"path": p, "sha_ok": (ROOT / p).is_file() and sha(p) == s, "sha256": s}
        out["defenders"][side] = verify_defender(n, side)
    if not any(out["defenders"][s]["problems"] for s in "AB"):
        a, b = _run_config(DEFENDERS[n]["A"]), _run_config(DEFENDERS[n]["B"])
        diff = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k) and k not in AB_DIFF_OK)
        out["ab_config_diff"] = diff
        if diff:
            for s in "AB":
                out["defenders"][s]["problems"].append(f"A/B defender configs differ beyond seed/paths: {diff[:8]}")
    out["reusable"] = all(out["specialists"][s]["sha_ok"] for s in "AB") and not any(out["defenders"][s]["problems"] for s in "AB")
    return out


# ------------------------------------------------------------------ freeze
def freeze(n: int) -> None:
    from experiments import seed_registry as SR
    P = paths(n)
    if P["spec"].exists() or P["dev_spec"].exists():
        raise SystemExit(f"REFUSING: {P['spec'].name} exists (frozen once)")
    v = verify(n)
    if not v["reusable"]:
        raise SystemExit(f"REFUSING: candidates not reusable: {json.dumps(v['defenders'], indent=1)[:1500]}")
    reg = {b["experiment_id"]: b for b in SR.load()["blocks"]}
    if P["dev_block"] not in reg:
        lo = SR.next_free(N_DEV, "exploratory")
        SR.allocate(P["dev_block"], lo, lo + N_DEV - 1, "exploratory", purpose=f"{n}v{n} strategy-conditioned k: development selection",
                    spec=P["dev_spec"].name, shared_by_labels=list(P["dev_labels"].values()))
    conf_id = f"{P['conf_label']}_SPECIALIST_CROSSOVER"
    if conf_id not in {b["experiment_id"] for b in SR.load()["blocks"]}:
        lo = SR.next_free(N_CONF, "sealed_confirmatory")
        SR.allocate(conf_id, lo, lo + N_CONF - 1, "sealed_confirmatory", purpose=f"{n}v{n} strategy-conditioned k: fresh confirmation",
                    spec=P["spec"].name)
    reg = {b["experiment_id"]: b for b in SR.load()["blocks"]}
    dev, conf = reg[P["dev_block"]], reg[conf_id]
    if not (dev["hi"] < conf["lo"] or conf["hi"] < dev["lo"]):
        raise SystemExit("REFUSING: development and confirmation seeds overlap")
    dev_ids = list(range(dev["lo"], dev["hi"] + 1))
    conf_ids = list(range(conf["lo"], conf["hi"] + 1))
    hl = lambda ids: hashlib.sha256(json.dumps(ids).encode()).hexdigest()  # noqa: E731
    cands = {}
    for side in "AB":
        p, s = SPECIALISTS[n][side]
        cands[side] = {"0": {"construction": f"sealed {side} specialist on all {n} agents", "path": p, "sha256": s},
                       str(k_role(n)): {"construction": f"defender-only: {n - k_role(n)} frozen {side}-specialist ATTACK + {k_role(n)} trained DEFEND",
                                        "defender": v["defenders"][side]["final"], "defender_sha256": v["defenders"][side]["final_sha256"],
                                        "frozen_attack": p, "frozen_attack_sha256": s}}
    common = {"status": "FROZEN_BEFORE_SELECTION", "utc": now(), "team_size": n, "git_head_at_freeze": git_head()}
    spec = {
        "record_id": P["spec"].stem, **common, "arm": "REVISED_METHOD_STRATEGY_CONDITIONED_ORGANIZATION", "confirmatory": True,
        "classification": "Revised hypothesis motivated by the sealed 4v4 scaling results (V1 k=2, R1 k=1). Strategy-conditioned "
                          "organization: not the original fixed-k method; no historical asymmetric system is compared or revived. "
                          "Not PAPER-FAITHFUL.",
        "decided_by": "user 2026-10-03: implement strategy-conditioned role allocation, 4v4 first",
        "MOTIVATION_disclosed": "Formulated AFTER observing the sealed 4v4 V1 (k=2: B collapsed, Delta_B -0.086) and R1 (k=1: A/B "
                                "converged, Delta_A +0.047, Delta_B -0.109) results and the PARTIAL defender-only behavior (stopped at "
                                "220/512, unsealed: A@A 0.648, A@B ~0.64). It was not conceived beforehand. Those seeds are never reused.",
        "HYPOTHESIS": "A strategy consists of its learned behavior and its selected team organization: k = k(z).",
        "FAIRNESS": "procedural symmetry: identical candidate set, selection objective, development budget, tie-break and "
                    "confirmation procedure for A and B; the selected k_A and k_B may differ and are never assigned by hand.",
        "MENU_locked": {"k_role": "max(1, floor(N/3 + 1/2)) = max(1, (2N + 3) // 6)", "K(N)": "{0, k_role(N)}", "this_scale": menu(n),
                        "k=0": "sealed repaired specialist on all N agents (no DEFEND role)",
                        "k>0": "defender-only: frozen specialist ATTACK slots + trained DEFEND slots, CLOSEST_DEFENDS(k), roles fixed per episode"},
        "CANDIDATES_locked": cands,
        "candidate_verification": {s: v["defenders"][s]["recipe"] for s in "AB"},
        "SELECTION_RULE_locked": {"objective": "k_p* = argmax_{k in K(N)} V(Pi_{p,k}, p): win rate on the INTENDED pole only "
                                               "(A on Pole A, B on Pole B), over the development block",
                                  "tie_break": "exact tie in measured intended-pole win rate -> the smaller k",
                                  "never_used": ["crossover cells", "Delta_A / Delta_B", "manual override"],
                                  "dev_block": {"experiment_id": P["dev_block"], "range": [dev["lo"], dev["hi"]], "n": N_DEV, "seed_ids_sha256": hl(dev_ids)},
                                  "dev_runs": {str(k): lab for k, lab in P["dev_labels"].items()},
                                  "note": "the evaluator plays both policies on both poles; non-intended-pole dev cells are recorded and EXCLUDED"},
        "CONFIRMATION_locked": {"label": P["conf_label"], "registry_experiment_id": conf_id, "range": [conf["lo"], conf["hi"]], "n": N_CONF,
                                "seed_ids_sha256": hl(conf_ids), "cells": "A@A, B@A, A@B, B@B with the selected k_A, k_B",
                                "report": "four win rates, score margins, Delta_A = A@A - B@A, Delta_B = B@B - A@B; paired bootstrap 95% and "
                                          f"{1 - ALPHA / N_ATTEMPTS:.4f} (1 - {ALPHA}/{N_ATTEMPTS}); n_boot {N_BOOT}, rng {BOOT_SEED}; no PASS/FAIL wording",
                                "k_frozen_before": "the selection record is sealed before the confirmation starts; k_A, k_B never change afterwards"},
        "INTERPRETATION_locked": {"k_A == k_B": "the selected organization happens to remain structurally symmetric",
                                  "k_A != k_B": "The deployed strategies use different organizations, but those organizations were selected by "
                                                "the same pre-specified candidate set, objective, evaluation budget, and tie-breaking rule. "
                                                "(Not called structurally symmetric.)"},
        "STOP": "after the confirmation readout: stop; no Stage 4, no 2v2/6v6 without PI/user approval; if no useful "
                "opponent-specific separation, stop the strategy-conditioned-k experiment",
        "NOT_AUTHORIZED": ["tuning the menu or rule after any result", "using crossover or Delta to pick k", "reusing these seeds",
                           "running 2v2 or 6v6", "Stage 4", "calling this the original fixed-k confirmatory method"],
    }
    P["spec"].write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    dev_spec = {"record_id": P["dev_spec"].stem, **common, "arm": "DEVELOPMENT_SELECTION", "confirmatory": False,
                "parent": P["spec"].name, "purpose": "evaluation spec for the development selection runs only"}
    P["dev_spec"].write_text(json.dumps(dev_spec, indent=2) + "\n", encoding="utf-8")
    print(f"-> {P['spec'].name}, {P['dev_spec'].name}; dev {dev['lo']}..{dev['hi']}, confirmation {conf['lo']}..{conf['hi']}")


# ------------------------------------------------------------------ evaluation plumbing
def log(n: int, msg: str) -> None:
    line = f"{now()} {msg}"
    print(line, flush=True)
    p = paths(n)["out"] / "sck.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def side_args(spec: dict, side: str, k: int) -> list[str]:
    c = spec["CANDIDATES_locked"][side][str(k)]
    flag = "--pi-a-path" if side == "A" else "--pi-b-path"
    if k == 0:
        return [flag, c["path"]]
    fa = ["--frozen-attack-path", "--frozen-attack-path-sha256"] if side == "A" else ["--frozen-attack-path-b", "--frozen-attack-path-b-sha256"]
    return [flag, c["defender"], fa[0], c["frozen_attack"], fa[1], c["frozen_attack_sha256"]]


def eval_cmd(n: int, spec_rel: str, label: str, lo: int, count: int, ka: int, kb: int, spec: dict, reg_id: str | None) -> list[str]:
    a = ["experiments/eval_specialist_crossover_scaled.py", "--team-size", str(n), "--spec", spec_rel, "--seed-base", str(lo),
         "--n-seeds", str(count), "--label", label, "--device", "cuda", *side_args(spec, "A", ka), *side_args(spec, "B", kb)]
    if ka or kb:
        a += ["--role-fixed-for-episode", "--role-k-defend", str(k_role(n))]
    if reg_id:
        a += ["--registry-experiment-id", reg_id]
    return a


def run_eval(n: int, argv: list[str], tag: str) -> int:
    out = paths(n)["out"]
    env = dict(os.environ, FOR_DISABLE_CONSOLE_CTRL_HANDLER="1", PYTHONUNBUFFERED="1")
    log(n, f"exec: {' '.join(argv)}")
    with (out / f"{tag}.log").open("a", encoding="utf-8") as fo, (out / f"{tag}.log.err").open("a", encoding="utf-8") as fe:
        return subprocess.run([PY, *argv], cwd=ROOT, env=env, stdout=fo, stderr=fe).returncode


def rows(label: str) -> dict:
    by: dict = {}
    with (SD / f"{label.lower()}_specialist_crossover_eval_rows.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            by.setdefault((r["policy"][-1], r["pole"]), {})[int(r["seed"])] = (float(r["win"]), float(r["margin"]))
    return by


def sealed(label: str) -> Path:
    p = SD / f"{label}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
    if not p.is_file() or json.loads(p.read_text(encoding="utf-8")).get("status") != "SEALED":
        raise SystemExit(f"FAIL-CLOSED: {label} not SEALED")
    return p


def boot(x, level_alpha: float) -> list[float]:
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(BOOT_SEED)
    b = x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1)
    lo, hi = np.percentile(b, [100 * level_alpha / 2, 100 * (1 - level_alpha / 2)])
    return [float(lo), float(hi)]


# ------------------------------------------------------------------ run: dev -> select -> confirm -> readout -> stop
def run(n: int) -> int:
    P = paths(n)
    spec = json.loads(P["spec"].read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit("REFUSING: spec not frozen")
    for side in "AB":                                      # candidates unchanged since freeze
        for k, c in spec["CANDIDATES_locked"][side].items():
            for key in (("path", "sha256"), ("defender", "defender_sha256"), ("frozen_attack", "frozen_attack_sha256")):
                if key[0] in c and sha(c[key[0]]) != c[key[1]]:
                    raise SystemExit(f"FAIL-CLOSED: candidate {side} k={k} {key[0]} changed since freeze")
    sel = spec["SELECTION_RULE_locked"]
    dlo, dhi = sel["dev_block"]["range"]
    spec_rel_dev = str(P["dev_spec"].relative_to(ROOT)).replace("\\", "/")
    spec_rel = str(P["spec"].relative_to(ROOT)).replace("\\", "/")
    if not P["selection"].is_file():
        for k, label in ((int(k), lab) for k, lab in sel["dev_runs"].items()):
            res = SD / f"{label}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
            if not res.is_file():
                base = eval_cmd(n, spec_rel_dev, label, dlo, dhi - dlo + 1, k, k, spec, sel["dev_block"]["experiment_id"])
                if run_eval(n, base + ["--dry-run"], f"dev_k{k}_dryrun") != 0:
                    log(n, f"STOPPED: dev k={k} dry-run failed")
                    return 1
                run_eval(n, base + ["--resume"], f"dev_k{k}")
            sealed(label)
        seeds = list(range(dlo, dhi + 1))
        V, files = {}, {}
        for k, label in ((int(k), lab) for k, lab in sel["dev_runs"].items()):
            r = rows(label)
            files[str(k)] = {"result_sha256": sha(sealed(label).relative_to(ROOT)),
                             "rows_sha256": sha((SD / f"{label.lower()}_specialist_crossover_eval_rows.csv").relative_to(ROOT))}
            for side in "AB":                               # intended pole only
                V[(side, k)] = float(np.mean([r[(side, side)][s][0] for s in seeds]))
        chosen = {}
        for side in "AB":
            best = max(V[(side, k)] for k in menu(n))
            chosen[side] = min(k for k in menu(n) if V[(side, k)] == best)   # exact tie -> smaller k
        rec = {"record_id": P["selection"].stem, "status": "SEALED_SELECTION", "utc": now(), "git_head": git_head(),
               "spec": {"path": spec_rel, "sha256": sha(spec_rel)}, "candidate_set": menu(n),
               "dev_block": sel["dev_block"], "selection_rule": sel["objective"], "tie_break": sel["tie_break"],
               "candidates": spec["CANDIDATES_locked"], "dev_result_files": files,
               "intended_pole_win_rates": {f"{s}@{s}_k{k}": V[(s, k)] for s in "AB" for k in menu(n)},
               "excluded": "non-intended-pole development cells (recorded by the evaluator, not read)",
               "selected": {"k_A": chosen["A"], "k_B": chosen["B"]},
               "frozen": "k_A and k_B never change after this record"}
        P["selection"].write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        log(n, f"SELECTION SEALED: k_A={chosen['A']} k_B={chosen['B']}  ({rec['intended_pole_win_rates']})")
    rec = json.loads(P["selection"].read_text(encoding="utf-8"))
    ka, kb = rec["selected"]["k_A"], rec["selected"]["k_B"]
    conf = spec["CONFIRMATION_locked"]
    clo, chi = conf["range"]
    if not (dhi < clo or chi < dlo):
        raise SystemExit("REFUSING: development and confirmation seeds overlap")
    res = SD / f"{conf['label']}_SPECIALIST_CROSSOVER_EVAL_RESULT.json"
    if not res.is_file():
        base = eval_cmd(n, spec_rel, conf["label"], clo, chi - clo + 1, ka, kb, spec, None)
        if run_eval(n, base + ["--dry-run"], "confirm_dryrun") != 0:
            log(n, "STOPPED: confirmation dry-run failed")
            return 1
        run_eval(n, base + ["--resume"], "confirm")
    sealed(conf["label"])
    readout(n, spec, rec)
    log(n, "CONFIRMATION READOUT WRITTEN -- STOP (no Stage 4, no 2v2/6v6 without approval)")
    return 0


def readout(n: int, spec: dict, sel: dict) -> dict:
    from experiments.eval_hog_psp_v3 import _mean_ci
    P = paths(n)
    conf = spec["CONFIRMATION_locked"]
    clo, chi = conf["range"]
    seeds = list(range(clo, chi + 1))
    r = rows(conf["label"])
    ka, kb = sel["selected"]["k_A"], sel["selected"]["k_B"]
    out = {"record": P["readout"].name, "utc": now(), "git_head": git_head(), "spec_sha256": sha(P["spec"].relative_to(ROOT)),
           "selection_record_sha256": sha(P["selection"].relative_to(ROOT)), "k_A": ka, "k_B": kb,
           "result_sha256": sha(sealed(conf["label"]).relative_to(ROOT)),
           "rows_sha256": sha((SD / f"{conf['label'].lower()}_specialist_crossover_eval_rows.csv").relative_to(ROOT)),
           "seeds": conf["range"], "bootstrap": {"n": N_BOOT, "rng": BOOT_SEED, "levels": [0.95, 1 - ALPHA / N_ATTEMPTS]}}
    for i, field in enumerate(("win", "margin")):
        v = {key: np.array([d[s][i] for s in seeds]) for key, d in r.items()}
        res = {f"{p}@{q}": float(v[(p, q)].mean()) for p in "AB" for q in "AB"}
        for name, x in (("Delta_A", v[("A", "A")] - v[("B", "A")]), ("Delta_B", v[("B", "B")] - v[("A", "B")])):
            c = _mean_ci(x)
            res[name] = {"mean": float(x.mean()), "std": float(x.std(ddof=1)), "ci95": [c["lcb95"], c["ucb95"]],
                         "ci9833": boot(x, ALPHA / N_ATTEMPTS)}
        out[field] = res
    interp = spec["INTERPRETATION_locked"]["k_A == k_B" if ka == kb else "k_A != k_B"]
    out["interpretation"] = interp
    f = lambda s: f"{s['mean']:+.3f} ± {s['std']:.3f} [{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"  # noqa: E731
    md = [f"# {n}v{n} strategy-conditioned role allocation: fresh confirmation ({clo}..{chi})", "",
          f"Selected on the development block (intended pole only): **k_A = {ka}, k_B = {kb}**. {interp}", "",
          "| | A@A | B@A | A@B | B@B | Δ_A (95%) | Δ_B (95%) |", "|---|---|---|---|---|---|---|"]
    for field in ("win", "margin"):
        t = out[field]
        md.append(f"| {field} | {t['A@A']:.3f} | {t['B@A']:.3f} | {t['A@B']:.3f} | {t['B@B']:.3f} | {f(t['Delta_A'])} | {f(t['Delta_B'])} |")
    w = out["win"]
    md += ["", f"98.33% (1 - 0.05/3): Δ_A [{w['Delta_A']['ci9833'][0]:+.3f}, {w['Delta_A']['ci9833'][1]:+.3f}], "
               f"Δ_B [{w['Delta_B']['ci9833'][0]:+.3f}, {w['Delta_B']['ci9833'][1]:+.3f}]",
           "", "Revised hypothesis motivated by the sealed 4v4 scaling results; not the original fixed-k method."]
    out["table_markdown"] = "\n".join(md)
    P["readout"].with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    P["readout"].with_suffix(".md").write_text(out["table_markdown"] + "\n", encoding="utf-8")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--team-size", type=int, required=True, choices=(2, 4, 6))
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--verify", action="store_true")
    g.add_argument("--freeze", action="store_true")
    g.add_argument("--run", action="store_true")
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if a.verify:
        print(json.dumps(verify(a.team_size), indent=1))
        return 0
    if a.team_size != 4 and not os.environ.get("SCK_CROSS_SCALE_APPROVED"):
        raise SystemExit("REFUSING: only 4v4 is authorized; 2v2/6v6 need PI/user approval after the 4v4 confirmation")
    if a.freeze:
        freeze(a.team_size)
        return 0
    return run(a.team_size)


if __name__ == "__main__":
    raise SystemExit(main())
