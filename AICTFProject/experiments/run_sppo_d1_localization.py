"""Run frozen SPPPO D1A/D1B diagnostics without replay, training, or new seeds."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts" / "strategic_demand"
SPPO = SD / "sppo"
PROTOCOL = SPPO / "D1_PROTOCOL_FROZEN.json"
D0_ROWS = SPPO / "D0_pole_b_decision_rows.csv"
D0_UNCERTAINTY = SPPO / "D0_pole_b_diagnostic_with_seed_uncertainty.json"
OUT = SPPO / "D1_RESULT.json"

PHASE0_SEEDS = list(range(6_500_001, 6_500_161))
D0_SEEDS = list(range(10_300_001, 10_300_193))
CATEGORIES = ("own_flag_stolen", "own_flag_home", "carrying", "not_carrying")
SAMPLES, ALPHA, RNG_SEED = 20_000, 0.05, 7
MAP_COLS = MAP_ROWS = 20
BLUE_HOME = np.asarray([2.0, 10.0])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _categories_from_vec(vec: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    carrying = (vec[..., 10] > 0.5).any(axis=1)
    x = vec[:, 0, 0] * (MAP_COLS - 1)
    y = vec[:, 0, 1] * (MAP_ROWS - 1)
    fx = vec[:, 0, 6] * MAP_COLS + x
    fy = vec[:, 0, 7] * MAP_ROWS + y
    x1 = vec[:, 1, 0] * (MAP_COLS - 1)
    y1 = vec[:, 1, 1] * (MAP_ROWS - 1)
    fx1 = vec[:, 1, 6] * MAP_COLS + x1
    fy1 = vec[:, 1, 7] * MAP_ROWS + y1
    recon_disagreement = np.sqrt((fx - fx1) ** 2 + (fy - fy1) ** 2)
    home = (np.isclose(fx, BLUE_HOME[0], atol=1e-3, rtol=1e-5)
            & np.isclose(fy, BLUE_HOME[1], atol=1e-3, rtol=1e-5))
    return {
        "own_flag_stolen": ~home,
        "own_flag_home": home,
        "carrying": carrying,
        "not_carrying": ~carrying,
    }, {
        "max_flag_reconstruction_disagreement_between_agents": float(recon_disagreement.max()),
        "home_x": float(BLUE_HOME[0]), "home_y": float(BLUE_HOME[1]),
    }


def _bootstrap_proportion(mask: np.ndarray, row_seed: np.ndarray, weights: np.ndarray,
                          seeds: list[int], rng: np.random.Generator) -> dict[str, float]:
    success = np.asarray([weights[(row_seed == s) & mask].sum() for s in seeds])
    total = np.asarray([weights[row_seed == s].sum() for s in seeds])
    draws = rng.multinomial(len(seeds), np.full(len(seeds), 1 / len(seeds)), size=SAMPLES)
    denom = draws @ total
    vals = (draws @ success) / np.maximum(denom, 1e-12)
    point = float((weights * mask).sum() / weights.sum())
    return {"point": point, "lcb95": float(np.quantile(vals, ALPHA / 2)),
            "ucb95": float(np.quantile(vals, 1 - ALPHA / 2))}


def _phase0_audit() -> dict[str, Any]:
    from experiments.phase0_scorer_common import TRAIN_SEEDS, load_split

    if TRAIN_SEEDS != PHASE0_SEEDS:
        raise RuntimeError("frozen Phase-0 train seed block drifted")
    split = load_split(PHASE0_SEEDS)
    if split.heldout_opened != 0 or split.seeds_opened != PHASE0_SEEDS:
        raise RuntimeError("D1A opened a non-training Phase-0 shard")

    vec = np.concatenate([split.p_vec, split.b_vec])
    pole = np.concatenate([split.p_pole, split.b_pole])
    row_seed = np.concatenate([split.p_seed, split.b_seed])
    objective_weights = np.concatenate([split.p_weight, np.ones(len(split.b_vec))])
    raw_weights = np.ones(len(vec))
    cats, reconstruction = _categories_from_vec(vec)
    rng = np.random.default_rng(RNG_SEED)
    reports = {}
    for name in CATEGORIES:
        reports[name] = {
            "raw_fit_rows": {
                "count": int(cats[name].sum()), "total": int(len(vec)),
                "proportion": _bootstrap_proportion(cats[name], row_seed, raw_weights,
                                                     PHASE0_SEEDS, rng),
            },
            "qpsi_loss_weighted": {
                "effective_count": float(objective_weights[cats[name]].sum()),
                "effective_total": float(objective_weights.sum()),
                "proportion": _bootstrap_proportion(cats[name], row_seed, objective_weights,
                                                     PHASE0_SEEDS, rng),
            },
            "by_pole_raw_rows": {
                label: {
                    "count": int((cats[name] & (pole == pi)).sum()),
                    "total": int((pole == pi).sum()),
                    "proportion": _bootstrap_proportion(
                        cats[name] & (pole == pi), row_seed, (pole == pi).astype(float),
                        PHASE0_SEEDS, rng),
                }
                for pi, label in ((0, "A"), (1, "B"))
            },
        }
    return {
        "train_seed_block": "6500001..6500160", "n_train_seeds": 160,
        "held_out_shards_opened": 0, "n_plain_fit_rows": int(len(split.p_vec)),
        "n_branch_fit_rows_including_teacher_action_pairs": int(len(split.b_vec)),
        "n_total_fit_rows": int(len(vec)), "data_sha256": split.data_sha256,
        "category_reconstruction": reconstruction, "categories": reports,
    }


def _read_d0() -> list[dict[str, Any]]:
    with D0_ROWS.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        for k in ("seed", "margin_B_bits", "qpsi_ranks_z1_correct", "own_flag_home",
                  "blue_carrying"):
            r[k] = float(r[k])
        r["seed"] = int(r["seed"])
    if sorted({r["seed"] for r in rows}) != D0_SEEDS:
        raise RuntimeError("D0 rows do not cover the exact frozen 192-seed block")
    return rows


def _worst_quartile_bootstrap(rows: list[dict[str, Any]]) -> dict[str, Any]:
    margin = np.asarray([r["margin_B_bits"] for r in rows])
    correct = np.asarray([r["qpsi_ranks_z1_correct"] for r in rows])
    seed_idx = np.asarray([D0_SEEDS.index(r["seed"]) for r in rows], dtype=np.int16)
    point_cut = float(np.percentile(margin, 25))
    point_mask = margin <= point_cut
    point = float(correct[point_mask].mean())
    order = np.argsort(margin, kind="stable")
    sorted_margin = margin[order]
    rng = np.random.default_rng(RNG_SEED)
    vals = np.empty(SAMPLES); cuts = np.empty(SAMPLES)
    for i in range(SAMPLES):
        counts = rng.multinomial(len(D0_SEEDS), np.full(len(D0_SEEDS), 1 / len(D0_SEEDS)))
        weights = counts[seed_idx].astype(float)
        sw = weights[order]; target = 0.25 * sw.sum()
        ci = min(int(np.searchsorted(np.cumsum(sw), target, side="left")), len(order) - 1)
        cut = float(sorted_margin[ci]); cuts[i] = cut
        mask = margin <= cut
        vals[i] = np.dot(weights * mask, correct) / np.maximum((weights * mask).sum(), 1e-12)
    return {
        "n_rows": len(rows), "point_quartile_cutoff": point_cut,
        "bootstrap_cutoff_lcb95": float(np.quantile(cuts, ALPHA / 2)),
        "bootstrap_cutoff_ucb95": float(np.quantile(cuts, 1 - ALPHA / 2)),
        "qpsi_correct_rate": {"point": point,
                              "lcb95": float(np.quantile(vals, ALPHA / 2)),
                              "ucb95": float(np.quantile(vals, 1 - ALPHA / 2))},
    }


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"REFUSING: D1 already exists: {OUT}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_ANY_COMPUTATION":
        raise SystemExit("REFUSING: D1 protocol is not prospectively frozen")
    d0_unc = json.loads(D0_UNCERTAINTY.read_text(encoding="utf-8"))
    d1a = _phase0_audit()
    d0_rows = _read_d0()
    home_rows = [r for r in d0_rows if bool(r["own_flag_home"])]
    d1b = _worst_quartile_bootstrap(home_rows)
    d1b["primary_excluding_own_flag_stolen"] = "identical to own_flag_home stratum by frozen complementary definitions"

    comparisons = {}
    d0_seed = np.asarray([r["seed"] for r in d0_rows])
    d0_rng = np.random.default_rng(RNG_SEED)
    for name in CATEGORIES:
        p0 = d1a["categories"][name]["qpsi_loss_weighted"]["proportion"]
        d0_mask = np.asarray([
            (not bool(r["own_flag_home"]) if name == "own_flag_stolen"
             else bool(r["own_flag_home"]) if name == "own_flag_home"
             else bool(r["blue_carrying"]) if name == "carrying"
             else not bool(r["blue_carrying"]))
            for r in d0_rows
        ])
        d0_prop = _bootstrap_proportion(d0_mask, d0_seed, np.ones(len(d0_rows)),
                                        D0_SEEDS, d0_rng)
        comparisons[name] = {
            "phase0_qpsi_loss_weighted_proportion": p0,
            "D0_z1_B_raw_decision_proportion": d0_prop,
            "point_difference_phase0_minus_D0": p0["point"] - d0_prop["point"],
        }

    result = {
        "record": "SPPPO D1 pole-B localization diagnostic",
        "status": "DIAGNOSTIC_ONLY_NOT_A_GATE",
        "protocol": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": _sha(PROTOCOL),
        "inputs": {"D0_rows_sha256": _sha(D0_ROWS),
                   "D0_uncertainty_sha256": _sha(D0_UNCERTAINTY)},
        "bootstrap": {"unit": "seed", "samples": SAMPLES, "alpha": ALPHA,
                      "rng_seed": RNG_SEED, "quartile_cutoff": "recomputed per replicate"},
        "D1A": d1a, "D1B": d1b, "phase0_vs_D0": comparisons,
        "branch_invariants": protocol["branch_invariants"],
        "interpretation": "Descriptive mechanism localization only; no retrospective threshold or authorization.",
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"D1 complete -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
