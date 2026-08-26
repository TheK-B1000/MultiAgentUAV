"""SPPPO V1 -- frozen five-candidate lambda_R development sweep.

Protocol: artifacts/strategic_demand/sppo/SPPPO_V1_PROTOCOL.json

    candidates   lambda_R in {0, 0.03, 0.1, 0.3, 1.0}
    budget       98,304 environment steps EACH  (= 24 PPO updates at 4096/update)
    train        10100001..10100032   (candidates are TRAINED here)
    select       10200001..10200032   (terminal checkpoints EVALUATED here)
    selection    computed ONLY after all five terminate, on TERMINAL values only

    qualify:  Delta_A(lambda) - Delta_A(0) > 0
          AND Delta_B(lambda) - Delta_B(0) > 0
          AND return degradation vs the lambda_R = 0 control <= 5%
    choose:   the SMALLEST qualifying lambda_R
    none:     SPPPO V1 = NOT WELL-POSED. STOP. No second grid, no larger budget.

WHAT THIS DRIVER ENFORCES RATHER THAN ASSUMES
---------------------------------------------
* every candidate is built from ONE base config; the resolved-config diff
  between any candidate and the control must contain nothing but lambda_R and
  the per-run paths. Anything else aborts the sweep before a step is spent.
* identical seed, identical PPO config, identical teacher rehearsal, identical
  Q_psi and SHA, identical margin, identical cadence, identical budget.
* lambda_R = 0 constructs NO runner (structural absence), verified by asserting
  the control's config carries lambda 0.0 and that the orchestrator hook returns
  without loading Q_psi.
* all five run to completion; selection is refused until five terminal records
  exist.
* TERMINAL values only -- interim rows are never read, so a candidate cannot be
  chosen for looking strong at 49k and regressing by 98k.

Run:  python experiments/run_sppo_lambda_sweep.py             # contract only
      python experiments/run_sppo_lambda_sweep.py --launch    # spend the sweep
      python experiments/run_sppo_lambda_sweep.py --select    # apply the rule
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_exp2b_specialization_preserving_compression import (  # noqa: E402
    configure_exp2b_live_environment,
)
from experiments.run_exp2c_mode_specific_actor_compression import (  # noqa: E402
    build_exp2c_config,
)

SD = ROOT / "artifacts" / "strategic_demand"
PROTOCOL = SD / "sppo" / "SPPPO_V1_PROTOCOL.json"
OUT = SD / "sppo" / "lambda_sweep"
SELECTION = SD / "sppo" / "SPPPO_LAMBDA_SELECTION.json"
DEV_EVAL = SD / "sppo" / "SPPPO_DEV_EVALUATION.json"

LAMBDA_GRID = [0.0, 0.03, 0.1, 0.3, 1.0]        # frozen; extension PROHIBITED
CONTROL = 0.0
DEV_STEPS = 98_304                               # 24 PPO updates at 4096/update
TRAIN_SEED = 10_100_001                          # candidates TRAIN here
DEV_SEED = 10_200_001                            # terminal checkpoints are SELECTED here
TRAIN_RANGE = (10_100_001, 10_100_032)
DEV_RANGE = (10_200_001, 10_200_032)
MARGIN = 0.04
RETURN_TOLERANCE = 0.05                          # <= 5% degradation vs control

# The ONLY fields that may differ between candidates. Everything else differing
# means the sweep is not a single-axis comparison and must not run.
ALLOWED_DIFFS = {
    "sppo_lambda_rank", "run_tag", "checkpoint_dir",
    "metrics_csv_path", "episode_csv_path",
}


def _stable_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _tag(lam: float) -> str:
    return f"sppo_dev_lambda{str(lam).replace('.', 'p')}_seed{TRAIN_SEED}"


def configure_sppo_live_environment(env, cfg, *, contract):
    """SPPPO's own frozen seed ranges -- NOT EXP2C's.

    The first launch reused EXP2C's configure function, which hardcodes
    8700001/8800001, and the guard correctly refused SPPPO's audited blocks.
    Everything else about the live contract is inherited unchanged: RULESET_V3
    M1, 32 environments, the 16/0/0/16 cell layout, and the per-env z/pole
    assertion.
    """
    return configure_exp2b_live_environment(
        env, cfg, contract=contract, allow_development_seed=False,
        training_seed_range=TRAIN_RANGE,
        development_seed_range=DEV_RANGE,
        manifest_key="sppo_protocol",
        context_label="SPPPO V1 lambda_R development sweep construction",
    )


def _load_protocol() -> dict:
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if p.get("status") != "FROZEN":
        raise RuntimeError(f"SPPPO protocol is not FROZEN (status={p.get('status')})")
    sel = p["lambda_R_SELECTION_EXPERIMENT_FROZEN_BEFORE_ANY_CANDIDATE_RUNS"]
    if [float(x) for x in sel["candidates"]] != LAMBDA_GRID:
        raise RuntimeError(f"grid drift: protocol {sel['candidates']} vs driver {LAMBDA_GRID}")
    budget = sel["development_budget_per_candidate"]["value_env_steps"]
    if int(budget) != DEV_STEPS:
        raise RuntimeError(f"budget drift: protocol {budget} vs driver {DEV_STEPS}")
    if float(p["margin_m_RATIFIED"]["value"]) != MARGIN:
        raise RuntimeError("margin drift vs frozen protocol")
    return p


def build_candidate(lam: float):
    """One candidate config, derived from the single shared EXP2C base."""
    cfg, parent_contract = build_exp2c_config()
    cfg.seed = TRAIN_SEED
    cfg.total_timesteps = DEV_STEPS
    cfg.sppo_ranking_margin = MARGIN
    cfg.sppo_ranking_cadence = 1
    base = dataclasses.asdict(cfg)              # identical for every candidate

    cfg.sppo_lambda_rank = float(lam)
    art = OUT / _tag(lam)
    cfg.run_tag = _tag(lam)
    cfg.checkpoint_dir = str(art / "ckpts")
    cfg.metrics_csv_path = str(art / "metrics.csv")
    cfg.episode_csv_path = str(art / "episode_rows.csv")
    return cfg, base, parent_contract


def _diff(left: dict, right: dict) -> dict:
    return {k: {"control": left.get(k), "candidate": right.get(k)}
            for k in sorted(set(left) | set(right)) if left.get(k) != right.get(k)}


def build_sweep_contract() -> dict:
    """Resolve all five, and prove lambda_R is the only scientific difference."""
    _load_protocol()
    resolved, bases = {}, {}
    for lam in LAMBDA_GRID:
        cfg, base, _ = build_candidate(lam)
        resolved[lam] = dataclasses.asdict(cfg)
        bases[lam] = base

    base_hashes = {lam: _stable_hash(b) for lam, b in bases.items()}
    if len(set(base_hashes.values())) != 1:
        raise RuntimeError(f"candidates do not share one base config: {base_hashes}")

    control = resolved[CONTROL]
    diffs, scientific = {}, {}
    for lam in LAMBDA_GRID:
        if lam == CONTROL:
            continue
        d = _diff(control, resolved[lam])
        unexpected = sorted(set(d) - ALLOWED_DIFFS)
        if unexpected:
            raise RuntimeError(
                f"lambda={lam} differs from the control outside the frozen axes: "
                f"{unexpected}. The sweep would not be a single-axis comparison.")
        sci = sorted(set(d) - {"run_tag", "checkpoint_dir",
                               "metrics_csv_path", "episode_csv_path"})
        if sci != ["sppo_lambda_rank"]:
            raise RuntimeError(f"lambda={lam} scientific diff is not single-axis: {sci}")
        diffs[str(lam)] = d
        scientific[str(lam)] = sci

    if float(control["sppo_lambda_rank"]) != 0.0:
        raise RuntimeError("the control must carry lambda_rank exactly 0.0")
    for lam in LAMBDA_GRID:
        r = resolved[lam]
        if int(r["total_timesteps"]) != DEV_STEPS or int(r["seed"]) != TRAIN_SEED:
            raise RuntimeError(f"lambda={lam} budget/seed drift")
        if float(r["sppo_ranking_margin"]) != MARGIN or int(r["sppo_ranking_cadence"]) != 1:
            raise RuntimeError(f"lambda={lam} margin/cadence drift")

    return {
        "record": "SPPPO V1 lambda_R development sweep contract",
        "protocol": "sppo/SPPPO_V1_PROTOCOL.json",
        "grid": LAMBDA_GRID,
        "grid_extension": "PROHIBITED",
        "control": CONTROL,
        "steps_per_candidate": DEV_STEPS,
        "train_seed": TRAIN_SEED,
        "selection_block": f"{DEV_RANGE[0]}..{DEV_RANGE[1]}",
        "selection_is_out_of_sample": True,
        "margin": MARGIN,
        "ranking_cadence": 1,
        "shared_base_config_sha256": next(iter(set(base_hashes.values()))),
        "resolved_config_sha256": {str(k): _stable_hash(v) for k, v in resolved.items()},
        "resolved_config_diff_vs_control": diffs,
        "scientific_diff_fields": scientific,
        "single_axis_verified": True,
        "control_is_structurally_absent": (
            "lambda_rank = 0.0 makes _maybe_attach_sppo_ranking return before "
            "loading Q_psi or constructing a runner; no attribute is set"),
        "run_tags": {str(k): _tag(k) for k in LAMBDA_GRID},
    }


# ------------------------------------------------------------------ selection
def _terminal_row(lam: float) -> dict:
    """Terminal DEVELOPMENT-BLOCK evaluation for one candidate.

    Selection reads the out-of-sample development evaluation, never the
    training-time telemetry. An earlier driver trained on the development block
    and selected from those same trajectories, which would have made the
    lambda_R choice in-sample. Training telemetry stays diagnostic.
    """
    if not DEV_EVAL.is_file():
        raise RuntimeError(
            f"{DEV_EVAL} is missing. Run experiments/eval_sppo_dev_candidates.py "
            "first: selection consumes the development evaluation, not training rows.")
    payload = json.loads(DEV_EVAL.read_text(encoding="utf-8"))
    results = payload["results"]
    key = str(lam)
    if key not in results:
        raise RuntimeError(f"lambda={lam} has no development evaluation; candidate did not run")
    r = results[key]
    steps = _terminal_training_steps(lam)
    if steps < DEV_STEPS:
        raise RuntimeError(
            f"lambda={lam} trained only {steps} < {DEV_STEPS} steps; the frozen "
            "budget requires every candidate to run to completion")
    return {"lambda": lam, "timesteps": steps,
            "delta_A": float(r["delta_A"]), "delta_B": float(r["delta_B"]),
            "ep_rew_mean": float(r["ep_rew_mean"]),
            "n_rank_updates": _terminal_rank_updates(lam)}


def _terminal_training_steps(lam: float) -> int:
    """Training completion is still verified from the run's own metrics.csv."""
    import csv
    path = OUT / _tag(lam) / "metrics.csv"
    if not path.is_file():
        raise RuntimeError(f"lambda={lam} has no metrics.csv; candidate did not run")
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise RuntimeError(f"lambda={lam} metrics.csv is empty")
    return int(float(rows[-1].get("timesteps") or 0))


def _terminal_rank_updates(lam: float) -> float:
    """Used only to prove the control was structurally absent."""
    import csv
    path = OUT / _tag(lam) / "metrics.csv"
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    v = rows[-1].get("sppo_n_rank_updates")
    return float(v) if v not in (None, "") else 0.0


def select() -> dict:
    import math
    if SELECTION.is_file():
        raise SystemExit(f"REFUSING: {SELECTION} exists; selection is one-shot")
    terminals = {lam: _terminal_row(lam) for lam in LAMBDA_GRID}   # all five or abort
    ctl = terminals[CONTROL]
    if math.isfinite(ctl["n_rank_updates"]) and ctl["n_rank_updates"] > 0:
        raise RuntimeError(
            "the lambda_R = 0 control recorded ranking updates; it was not "
            "structurally absent and the sweep is invalid")

    rows, qualifying = [], []
    for lam in LAMBDA_GRID:
        if lam == CONTROL:
            continue
        t = terminals[lam]
        dA = t["delta_A"] - ctl["delta_A"]
        dB = t["delta_B"] - ctl["delta_B"]
        deg = ((ctl["ep_rew_mean"] - t["ep_rew_mean"]) / abs(ctl["ep_rew_mean"])
               if ctl["ep_rew_mean"] else float("nan"))
        ok = bool(dA > 0 and dB > 0 and (deg <= RETURN_TOLERANCE))
        rows.append({"lambda": lam, "delta_A_gain": round(dA, 6),
                     "delta_B_gain": round(dB, 6),
                     "return_degradation": round(deg, 6),
                     "moves_both_contrasts": bool(dA > 0 and dB > 0),
                     "return_within_tolerance": bool(deg <= RETURN_TOLERANCE),
                     "qualifies": ok})
        if ok:
            qualifying.append(lam)

    chosen = min(qualifying) if qualifying else None
    rec = {
        "record": "SPPPO V1 lambda_R selection",
        "status": "FROZEN_RESULT",
        "rule": "smallest lambda_R with both contrast gains > 0 and return degradation <= 5%",
        "terminal_values_only": True,
        "selection_source": "SPPPO_DEV_EVALUATION.json (out-of-sample development block)",
        "training_telemetry_used_for_selection": False,
        "control_terminal": ctl,
        "candidates": rows,
        "qualifying": qualifying,
        "SELECTED_LAMBDA_R": chosen,
        "VERDICT": "LAMBDA_SELECTED" if chosen is not None else "SPPPO_V1_NOT_WELL_POSED",
        "consequence": (f"freeze lambda_R = {chosen} before production"
                        if chosen is not None else
                        "STOP. No second grid, no larger budget, no production run."),
    }
    SELECTION.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true", help="spend the five development runs")
    ap.add_argument("--select", action="store_true", help="apply the frozen selection rule")
    a = ap.parse_args()

    if a.select:
        rec = select()
        print(json.dumps(rec, indent=2))
        return 0

    contract = build_sweep_contract()
    print(json.dumps({k: v for k, v in contract.items()
                      if k != "resolved_config_diff_vs_control"}, indent=2, default=str))
    print("\nper-candidate scientific diff vs control:")
    for lam, sci in contract["scientific_diff_fields"].items():
        print(f"  lambda={lam:<5} {sci}")
    if not a.launch:
        print("\nCONTRACT ONLY. No environment constructed, no training step spent.")
        return 0

    for lam in LAMBDA_GRID:
        run_dir = OUT / _tag(lam)
        if run_dir.exists() and any(run_dir.iterdir()):
            raise SystemExit(f"REFUSING: candidate dir is not empty: {run_dir}")

    from rl.training.orchestrator import orchestrate_training_run
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "SWEEP_CONTRACT.json").write_text(json.dumps(contract, indent=2, default=str),
                                             encoding="utf-8")
    for lam in LAMBDA_GRID:
        cfg, _, parent_contract = build_candidate(lam)
        print(f"\n=== SPPPO development candidate lambda_R = {lam} "
              f"({DEV_STEPS} steps) ===", flush=True)
        orchestrate_training_run(
            cfg,
            pre_rollout_env_setup=partial(configure_sppo_live_environment,
                                          contract=parent_contract),
        )
    print("\nAll five candidates complete. Run --select to apply the frozen rule.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
