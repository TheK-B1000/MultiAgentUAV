"""Run frozen SPPPO D1A/D1B diagnostics without replay, training, or new seeds.

Scientific protocol: artifacts/strategic_demand/sppo/D1_PROTOCOL_FROZEN.json
(commit 4c0a8fd4). This module is the implementation lock; it must be committed
before any D1 output exists.

D1A primary reports are raw fit-row counts/proportions with seed-level
uncertainty. Loss-weighted coverage is a secondary_training_weight_diagnostic
only and must not redefine the off-support interpretation.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SD = ROOT / "artifacts" / "strategic_demand"
SPPO = SD / "sppo"
PROTOCOL = SPPO / "D1_PROTOCOL_FROZEN.json"
D0_ROWS = SPPO / "D0_pole_b_decision_rows.csv"
D0_UNCERTAINTY = SPPO / "D0_pole_b_diagnostic_with_seed_uncertainty.json"
OUT = SPPO / "D1_RESULT.json"
IMPL_FREEZE = SPPO / "D1_IMPLEMENTATION_FROZEN.json"

PHASE0_SEEDS = list(range(6_500_001, 6_500_161))
D0_SEEDS = list(range(10_300_001, 10_300_193))
CATEGORIES = ("own_flag_stolen", "own_flag_home", "carrying", "not_carrying")
SAMPLES, ALPHA, RNG_SEED = 20_000, 0.05, 7

# Hard observation-reconstruction contract for D1A.
EXPECTED_N_AGENTS = 2
EXPECTED_VEC_DIM = 20
CARRYING_FEATURE_INDEX = 10
FLAG_DX_INDEX, FLAG_DY_INDEX = 6, 7
POS_X_INDEX, POS_Y_INDEX = 0, 1
MAP_COLS = MAP_ROWS = 20
BLUE_HOME = np.asarray([2.0, 10.0], dtype=np.float64)
HOME_ATOL = 1e-3
HOME_RTOL = 1e-5
MAX_AGENT_FLAG_RECON_DISAGREE = 1e-3


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _assert_obs_schema(vec: np.ndarray) -> None:
    if vec.ndim != 3:
        raise RuntimeError(
            f"D1A obs schema fail: expected rank-3 vec [N, agents, dim], got shape {vec.shape}"
        )
    n, n_agents, dim = vec.shape
    if n_agents != EXPECTED_N_AGENTS:
        raise RuntimeError(
            f"D1A obs schema fail: expected n_agents={EXPECTED_N_AGENTS}, got {n_agents}"
        )
    if dim != EXPECTED_VEC_DIM:
        raise RuntimeError(
            f"D1A obs schema fail: expected vec_dim={EXPECTED_VEC_DIM}, got {dim}"
        )
    if MAP_COLS != 20 or MAP_ROWS != 20:
        raise RuntimeError("D1A map contract fail: frozen map must be 20x20")
    if not (np.isclose(BLUE_HOME[0], 2.0) and np.isclose(BLUE_HOME[1], 10.0)):
        raise RuntimeError("D1A home contract fail: blue home must be (2,10)")
    if n == 0:
        raise RuntimeError("D1A obs schema fail: empty fit-row tensor")


def _categories_from_vec(vec: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Classify Phase-0 fit rows from stored observation vectors.

    Contracts:
      carrying      = any agent with feature 10 > 0.5
      own_flag_home = reconstructed blue-flag absolute position is (2,10)
      own_flag_stolen = complement of own_flag_home
    """
    _assert_obs_schema(vec)
    carrying = (vec[..., CARRYING_FEATURE_INDEX] > 0.5).any(axis=1)
    x = vec[:, 0, POS_X_INDEX] * (MAP_COLS - 1)
    y = vec[:, 0, POS_Y_INDEX] * (MAP_ROWS - 1)
    fx = vec[:, 0, FLAG_DX_INDEX] * MAP_COLS + x
    fy = vec[:, 0, FLAG_DY_INDEX] * MAP_ROWS + y
    x1 = vec[:, 1, POS_X_INDEX] * (MAP_COLS - 1)
    y1 = vec[:, 1, POS_Y_INDEX] * (MAP_ROWS - 1)
    fx1 = vec[:, 1, FLAG_DX_INDEX] * MAP_COLS + x1
    fy1 = vec[:, 1, FLAG_DY_INDEX] * MAP_ROWS + y1
    recon_disagreement = np.sqrt((fx - fx1) ** 2 + (fy - fy1) ** 2)
    max_dis = float(recon_disagreement.max()) if len(recon_disagreement) else 0.0
    if max_dis > MAX_AGENT_FLAG_RECON_DISAGREE:
        raise RuntimeError(
            f"D1A flag-reconstruction disagreement {max_dis} exceeds "
            f"{MAX_AGENT_FLAG_RECON_DISAGREE}; observation schema likely drifted"
        )
    home = (
        np.isclose(fx, BLUE_HOME[0], atol=HOME_ATOL, rtol=HOME_RTOL)
        & np.isclose(fy, BLUE_HOME[1], atol=HOME_ATOL, rtol=HOME_RTOL)
    )
    return {
        "own_flag_stolen": ~home,
        "own_flag_home": home,
        "carrying": carrying,
        "not_carrying": ~carrying,
    }, {
        "expected_n_agents": EXPECTED_N_AGENTS,
        "expected_vec_dim": EXPECTED_VEC_DIM,
        "carrying_feature_index": CARRYING_FEATURE_INDEX,
        "map_cols": MAP_COLS,
        "map_rows": MAP_ROWS,
        "home_x": float(BLUE_HOME[0]),
        "home_y": float(BLUE_HOME[1]),
        "max_flag_reconstruction_disagreement_between_agents": max_dis,
    }


def _bootstrap_proportion(mask: np.ndarray, row_seed: np.ndarray, weights: np.ndarray,
                          seeds: list[int], rng: np.random.Generator) -> dict[str, float]:
    success = np.asarray([weights[(row_seed == s) & mask].sum() for s in seeds])
    total = np.asarray([weights[row_seed == s].sum() for s in seeds])
    draws = rng.multinomial(len(seeds), np.full(len(seeds), 1 / len(seeds)), size=SAMPLES)
    denom = draws @ total
    vals = (draws @ success) / np.maximum(denom, 1e-12)
    point = float((weights * mask).sum() / max(float(weights.sum()), 1e-12))
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
        raw_prop = _bootstrap_proportion(
            cats[name], row_seed, raw_weights, PHASE0_SEEDS, rng)
        weighted_prop = _bootstrap_proportion(
            cats[name], row_seed, objective_weights, PHASE0_SEEDS, rng)
        reports[name] = {
            # PRIMARY — frozen D1A question
            "raw_fit_rows": {
                "count": int(cats[name].sum()),
                "total": int(len(vec)),
                "proportion": raw_prop,
            },
            "by_pole_raw_rows": {
                label: {
                    "count": int((cats[name] & (pole == pi)).sum()),
                    "total": int((pole == pi).sum()),
                    "proportion": _bootstrap_proportion(
                        cats[name] & (pole == pi), row_seed,
                        (pole == pi).astype(float), PHASE0_SEEDS, rng),
                }
                for pi, label in ((0, "A"), (1, "B"))
            },
            # SECONDARY — useful because Q_psi used episode-normalized loss weights;
            # must not redefine the off-support interpretation.
            "secondary_training_weight_diagnostic": {
                "note": ("Episode-normalized/loss weighting used in Q_psi fitting. "
                         "Effective influence may differ from raw row frequency. "
                         "Secondary only; not the frozen D1A primary quantity."),
                "effective_count": float(objective_weights[cats[name]].sum()),
                "effective_total": float(objective_weights.sum()),
                "proportion": weighted_prop,
            },
        }
    return {
        "train_seed_block": "6500001..6500160",
        "n_train_seeds": 160,
        "held_out_shards_opened": 0,
        "n_plain_fit_rows": int(len(split.p_vec)),
        "n_branch_fit_rows_including_teacher_action_pairs": int(len(split.b_vec)),
        "n_total_fit_rows": int(len(vec)),
        "data_sha256": split.data_sha256,
        "category_reconstruction": reconstruction,
        "categories": reports,
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
    point = float(correct[point_mask].mean()) if point_mask.any() else float("nan")
    order = np.argsort(margin, kind="stable")
    sorted_margin = margin[order]
    rng = np.random.default_rng(RNG_SEED)
    vals = np.empty(SAMPLES)
    cuts = np.empty(SAMPLES)
    for i in range(SAMPLES):
        counts = rng.multinomial(len(D0_SEEDS), np.full(len(D0_SEEDS), 1 / len(D0_SEEDS)))
        weights = counts[seed_idx].astype(float)
        sw = weights[order]
        target = 0.25 * sw.sum()
        ci = min(int(np.searchsorted(np.cumsum(sw), target, side="left")), len(order) - 1)
        cut = float(sorted_margin[ci])
        cuts[i] = cut
        mask = margin <= cut
        vals[i] = np.dot(weights * mask, correct) / np.maximum((weights * mask).sum(), 1e-12)
    return {
        "n_rows": len(rows),
        "point_quartile_cutoff": point_cut,
        "bootstrap_cutoff_lcb95": float(np.quantile(cuts, ALPHA / 2)),
        "bootstrap_cutoff_ucb95": float(np.quantile(cuts, 1 - ALPHA / 2)),
        "qpsi_correct_rate": {
            "point": point,
            "lcb95": float(np.quantile(vals, ALPHA / 2)),
            "ucb95": float(np.quantile(vals, 1 - ALPHA / 2)),
        },
    }


def _verify_implementation_freeze() -> dict[str, Any]:
    """Fail closed unless the committed implementation freeze still matches disk."""
    if not IMPL_FREEZE.exists():
        raise SystemExit(f"REFUSING: missing committed implementation freeze: {IMPL_FREEZE}")
    freeze = json.loads(IMPL_FREEZE.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_D1_OUTPUTS":
        raise SystemExit("REFUSING: implementation freeze status drifted")
    checks = {
        "scientific_protocol_sha256": _sha(PROTOCOL),
        "runner_sha256": _sha(Path(__file__).resolve()),
        "D0_rows_sha256": _sha(D0_ROWS),
        "D0_uncertainty_sha256": _sha(D0_UNCERTAINTY),
    }
    expected = {
        "scientific_protocol_sha256": freeze["scientific_protocol_sha256"],
        "runner_sha256": freeze["runner_sha256"],
        "D0_rows_sha256": freeze["inputs"]["D0_rows_sha256"],
        "D0_uncertainty_sha256": freeze["inputs"]["D0_uncertainty_sha256"],
    }
    for key, got in checks.items():
        if got != expected[key]:
            raise SystemExit(
                f"REFUSING: {key} mismatch\n  expected {expected[key]}\n  got      {got}"
            )
    return freeze


def main() -> int:
    if OUT.exists():
        raise SystemExit(f"REFUSING: D1 already exists: {OUT}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if protocol.get("status") != "FROZEN_BEFORE_ANY_COMPUTATION":
        raise SystemExit("REFUSING: D1 protocol is not prospectively frozen")
    if not D0_UNCERTAINTY.exists():
        raise SystemExit("REFUSING: D0 seed-uncertainty report missing")

    impl = _verify_implementation_freeze()
    d1a = _phase0_audit()
    d0_rows = _read_d0()
    home_rows = [r for r in d0_rows if bool(r["own_flag_home"])]
    d1b = {
        "label": "POSTMORTEM FOLLOW-UP DIAGNOSTIC -- not independent confirmation; not a new gate",
        "stratum": "own_flag_home (== exclude own_flag_stolen by frozen complementary definitions)",
        "primary_excluding_own_flag_stolen": _worst_quartile_bootstrap(home_rows),
        "complementary_own_flag_home": "identical to primary_excluding_own_flag_stolen",
    }

    comparisons = {}
    d0_seed = np.asarray([r["seed"] for r in d0_rows])
    d0_rng = np.random.default_rng(RNG_SEED)
    for name in CATEGORIES:
        # PRIMARY comparison uses raw fit-row proportions, per frozen D1A.
        p0 = d1a["categories"][name]["raw_fit_rows"]["proportion"]
        d0_mask = np.asarray([
            (not bool(r["own_flag_home"]) if name == "own_flag_stolen"
             else bool(r["own_flag_home"]) if name == "own_flag_home"
             else bool(r["blue_carrying"]) if name == "carrying"
             else not bool(r["blue_carrying"]))
            for r in d0_rows
        ])
        d0_prop = _bootstrap_proportion(
            d0_mask, d0_seed, np.ones(len(d0_rows)), D0_SEEDS, d0_rng)
        comparisons[name] = {
            "phase0_raw_fit_row_proportion": p0,
            "D0_z1_B_raw_decision_proportion": d0_prop,
            "point_difference_phase0_minus_D0": p0["point"] - d0_prop["point"],
            "secondary_phase0_loss_weighted_proportion":
                d1a["categories"][name]["secondary_training_weight_diagnostic"]["proportion"],
        }

    result = {
        "record": "SPPPO D1 pole-B localization diagnostic",
        "status": "DIAGNOSTIC_ONLY_NOT_A_GATE",
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "protocol_sha256": impl["scientific_protocol_sha256"],
        "implementation_freeze": str(IMPL_FREEZE.relative_to(ROOT)),
        "implementation_freeze_sha256": _sha(IMPL_FREEZE),
        "runner_sha256": impl["runner_sha256"],
        "inputs": impl["inputs"],
        "bootstrap": {
            "unit": "seed",
            "samples": SAMPLES,
            "alpha": ALPHA,
            "rng_seed": RNG_SEED,
            "quartile_cutoff": "recomputed per replicate",
        },
        "D1A": d1a,
        "D1B": d1b,
        "phase0_vs_D0": comparisons,
        "branch_invariants": protocol["branch_invariants"],
        "interpretation": (
            "Descriptive mechanism localization only; no retrospective threshold "
            "or authorization. Loss-weighted coverage is secondary."
        ),
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"D1 complete -> {OUT}")
    print(f"  protocol_sha256={result['protocol_sha256']}")
    print(f"  runner_sha256={result['runner_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
