"""One-shot RASR-PPO four-regime scorer qualification on frozen DEV data."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.collect_rasr_dev_scorer_data import (  # noqa: E402
    COMPLETE,
    DEV_SEEDS,
    EXPECTED_TEACHER_SHA,
    OUT as DATA_DIR,
)
from experiments.phase0_fit_qpsi import TEACHERS  # noqa: E402
from experiments.phase0_scorer_common import (  # noqa: E402
    assert_teacher_query_valid,
    sha256_file,
    teacher_action_dists,
)
from experiments.run_rasrppo_ladder import require_dev_collection_gate  # noqa: E402
from rl.scorer.qpsi import QPsi, QPsiConfig, joint_action_index  # noqa: E402

RASR_DIR = ROOT / "artifacts" / "strategic_demand" / "rasrppo"
PROTOCOL = RASR_DIR / "RASR_PPO_CAUSAL_LADDER_PROTOCOL.json"
REGIME_RECORD = RASR_DIR / "RASR_REGIME_QPSI_FROZEN.json"
R1_WEIGHTS = RASR_DIR / "qpsi_regime_frozen.pt"
S0_WEIGHTS = (
    ROOT / "artifacts" / "strategic_demand" / "phase0_scorer_data" / "qpsi_frozen.pt"
)
OUTPUT = RASR_DIR / "RASR_SCORER_QUALIFICATION.json"
R1_SHA = "44c0680e037939de287ad4201fead6312bc92b6bcd1fd902f568868cb24b760a"
S0_SHA = "930051a725e55e4f14e05dfe178e5f1dc7bd8f3d7e3adeba01187958bb7417bf"
N_BOOT, ALPHA, RNG_SEED = 20_000, 0.05, 7
MIN_SUPPORT_SEEDS = 32


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def paired_seed_bootstrap_lcb(
    values: np.ndarray,
    seeds: np.ndarray,
    *,
    samples: int = N_BOOT,
    alpha: float = ALPHA,
    rng_seed: int = RNG_SEED,
) -> tuple[float, np.ndarray]:
    """Percentile LCB after resampling seed clusters and retaining all rows."""
    values = np.asarray(values, dtype=np.float64)
    seeds = np.asarray(seeds, dtype=np.int64)
    if values.ndim != 1 or seeds.shape != values.shape or not len(values):
        raise ValueError("values and seeds must be nonempty aligned vectors")
    unique = np.unique(seeds)
    by_seed = [values[seeds == seed] for seed in unique]
    rng = np.random.default_rng(rng_seed)
    indices = rng.integers(0, len(unique), size=(samples, len(unique)))
    draws = np.empty(samples, dtype=np.float64)
    for draw_index, selected in enumerate(indices):
        draws[draw_index] = np.concatenate(
            [by_seed[index] for index in selected]
        ).mean()
    return float(np.quantile(draws, alpha / 2)), draws


def support_counts(
    poles: np.ndarray, regimes: np.ndarray, seeds: np.ndarray
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for pole, pole_name in ((0, "A"), (1, "B")):
        for regime in range(4):
            selected = (poles == pole) & (regimes == regime)
            result[f"pole_{pole_name}_regime_{regime}"] = {
                "n_states": int(selected.sum()),
                "n_distinct_dev_seeds": int(np.unique(seeds[selected]).size),
            }
    return result


def _load_model(path: Path, device: str, expected_regimes: int) -> QPsi:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = QPsi(QPsiConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    if model.cfg.n_regimes != expected_regimes:
        raise RuntimeError(
            f"{path.name} has n_regimes={model.cfg.n_regimes}, expected {expected_regimes}"
        )
    model.eval()
    return model


def _load_dev_branch() -> SimpleNamespace:
    """Load only branch-state arrays. Plain grids are streamed later for MSE.

    Concatenating every plain decision-point grid across 96 seeds needs ~1.5 GiB
    and OOMs on this host; gates only need the 576 matched branch states.
    """
    complete = json.loads(COMPLETE.read_text(encoding="utf-8"))
    if complete.get("verdict") != "COLLECTION_COMPLETE":
        raise RuntimeError("REFUSING: DEV collection is not marked COMPLETE")
    if complete.get("seed_block") != [DEV_SEEDS[0], DEV_SEEDS[-1], len(DEV_SEEDS)]:
        raise RuntimeError("REFUSING: DEV collection seed block drifted")

    branch: dict[str, list[np.ndarray]] = {
        key: []
        for key in (
            "grid", "vec", "amask", "mask", "pole", "seed",
            "action_A", "action_B", "margin_A", "margin_B",
        )
    }
    opened: list[int] = []
    for seed in DEV_SEEDS:
        shard = DATA_DIR / "seed_shards" / f"seed_{seed}.npz"
        summary_path = DATA_DIR / "seed_summaries" / f"seed_{seed}.json"
        if not shard.is_file() or not summary_path.is_file():
            raise RuntimeError(f"REFUSING: missing DEV seed artifact for {seed}")
        opened.append(seed)
        with np.load(shard, allow_pickle=False) as data:
            shard_seed = (
                data["branch_seed"]
                if "branch_seed" in data.files
                else np.full(len(data["branch_pole"]), seed, dtype=np.int64)
            )
            if not np.all(shard_seed == seed) or not np.all(data["plain_seed"] == seed):
                raise RuntimeError(f"REFUSING: shard {shard.name} contains another seed")
            for source, target in (
                ("branch_obs_grid", "grid"),
                ("branch_obs_vec", "vec"),
                ("branch_obs_agent_mask", "amask"),
                ("branch_obs_mask", "mask"),
                ("branch_pole", "pole"),
            ):
                value = np.asarray(data[source])
                if source.startswith("branch_obs_"):
                    value = value[:, 0]
                branch[target].append(value)
            branch["seed"].append(np.asarray(shard_seed, dtype=np.int64))
            for tag, suffix in (("pi_A", "A"), ("pi_B", "B")):
                branch[f"action_{suffix}"].append(np.asarray(data[f"branch_{tag}_action"]))
                branch[f"margin_{suffix}"].append(
                    np.asarray(data[f"branch_{tag}_blue"], dtype=np.float32)
                    - np.asarray(data[f"branch_{tag}_red"], dtype=np.float32)
                )
    if opened != DEV_SEEDS:
        raise RuntimeError("REFUSING: opened DEV block does not match frozen order")
    return SimpleNamespace(
        **{f"b_{key}": np.concatenate(value) for key, value in branch.items()},
    )


def _teacher_validity_for(data: SimpleNamespace, *, teacher_idx: int, actions: np.ndarray):
    """Teacher validity on branch states without duplicating grids in RAM."""
    n = len(data.b_seed)
    split = SimpleNamespace(
        b_teacher=np.full(n, teacher_idx, dtype=np.int64),
        b_grid=data.b_grid,
        b_vec=data.b_vec,
        b_amask=data.b_amask,
        b_mask=data.b_mask,
        b_action=actions,
    )
    return split


def _expected_values(model, data, teachers, device: str) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for tag in ("pi_A", "pi_B"):
            chunks = []
            for start in range(0, len(data.b_seed), 512):
                rows = slice(start, start + 512)
                grid = torch.as_tensor(data.b_grid[rows], dtype=torch.float32, device=device)
                vec = torch.as_tensor(data.b_vec[rows], dtype=torch.float32, device=device)
                amask = torch.as_tensor(data.b_amask[rows], dtype=torch.float32, device=device)
                mask = torch.as_tensor(data.b_mask[rows], dtype=torch.float32, device=device)
                pole = torch.as_tensor(data.b_pole[rows], dtype=torch.long, device=device)
                p1, p2 = teacher_action_dists(
                    teachers[tag], grid, vec, amask, mask
                )
                chunks.append(
                    model.expected_value(grid, vec, amask, pole, p1, p2)
                    .cpu()
                    .numpy()
                )
            values[tag] = np.concatenate(chunks)
    return values


def _regression_report(model, device: str) -> dict:
    """Stream plain+branch regression rows seed-by-seed (secondary report only)."""
    totals = {
        str(regime): {
            "n_rows": 0,
            "weighted_sse": 0.0,
            "weight_sum": 0.0,
            "pred_wsum": 0.0,
            "target_wsum": 0.0,
            "prediction_min": float("inf"),
            "prediction_max": float("-inf"),
        }
        for regime in range(4)
    }
    with torch.no_grad():
        for seed in DEV_SEEDS:
            shard = DATA_DIR / "seed_shards" / f"seed_{seed}.npz"
            summary_path = DATA_DIR / "seed_summaries" / f"seed_{seed}.json"
            summaries = json.loads(summary_path.read_text(encoding="utf-8"))
            margins = {
                (row["policy"], row["pole"]): float(row["blue"] - row["red"])
                for row in summaries
            }
            with np.load(shard, allow_pickle=False) as data:
                plain_policy = np.asarray(data["plain_policy"], dtype=np.int64)
                plain_pole = np.asarray(data["plain_pole"], dtype=np.int64)
                plain_target = np.empty(len(plain_policy), dtype=np.float32)
                plain_weight = np.empty(len(plain_policy), dtype=np.float32)
                for policy_index, policy in ((0, "pi_A"), (1, "pi_B")):
                    for pole_index, pole_name in ((0, "A"), (1, "B")):
                        selected = (plain_policy == policy_index) & (
                            plain_pole == pole_index
                        )
                        count = int(selected.sum())
                        if count == 0:
                            continue
                        plain_target[selected] = margins[(policy, pole_name)]
                        plain_weight[selected] = 1.0 / count

                grids = [np.asarray(data["plain_obs_grid"])[:, 0]]
                vecs = [np.asarray(data["plain_obs_vec"])[:, 0]]
                amasks = [np.asarray(data["plain_obs_agent_mask"])[:, 0]]
                poles = [np.asarray(data["plain_pole"])]
                actions = [np.asarray(data["plain_action"])]
                targets = [plain_target]
                weights = [plain_weight]
                for tag in ("pi_A", "pi_B"):
                    grids.append(np.asarray(data["branch_obs_grid"])[:, 0])
                    vecs.append(np.asarray(data["branch_obs_vec"])[:, 0])
                    amasks.append(np.asarray(data["branch_obs_agent_mask"])[:, 0])
                    poles.append(np.asarray(data["branch_pole"]))
                    actions.append(np.asarray(data[f"branch_{tag}_action"]))
                    targets.append(
                        np.asarray(data[f"branch_{tag}_blue"], dtype=np.float32)
                        - np.asarray(data[f"branch_{tag}_red"], dtype=np.float32)
                    )
                    weights.append(
                        np.ones(len(data["branch_pole"]), dtype=np.float32)
                    )

                grid = np.concatenate(grids)
                vec = np.concatenate(vecs)
                amask = np.concatenate(amasks)
                pole = np.concatenate(poles)
                action = np.concatenate(actions)
                target = np.concatenate(targets)
                weight = np.concatenate(weights)

            regimes = (
                model.regime_from_vec(
                    torch.as_tensor(vec, dtype=torch.float32, device=device)
                )
                .cpu()
                .numpy()
            )
            a1, a2 = joint_action_index(torch.as_tensor(action, dtype=torch.long))
            predictions = []
            for start in range(0, len(target), 1024):
                rows = slice(start, start + 1024)
                predictions.append(
                    model(
                        torch.as_tensor(grid[rows], dtype=torch.float32, device=device),
                        torch.as_tensor(vec[rows], dtype=torch.float32, device=device),
                        torch.as_tensor(amask[rows], dtype=torch.float32, device=device),
                        torch.as_tensor(pole[rows], dtype=torch.long, device=device),
                        a1[rows].to(device),
                        a2[rows].to(device),
                    )
                    .cpu()
                    .numpy()
                )
            prediction = np.concatenate(predictions)
            for regime in range(4):
                selected = regimes == regime
                if not selected.any():
                    continue
                cell = totals[str(regime)]
                w = weight[selected]
                p = prediction[selected]
                y = target[selected]
                cell["n_rows"] += int(selected.sum())
                cell["weighted_sse"] += float(np.sum(w * (p - y) ** 2))
                cell["weight_sum"] += float(w.sum())
                cell["pred_wsum"] += float(np.sum(w * p))
                cell["target_wsum"] += float(np.sum(w * y))
                cell["prediction_min"] = min(cell["prediction_min"], float(p.min()))
                cell["prediction_max"] = max(cell["prediction_max"], float(p.max()))

    report = {}
    for regime, cell in totals.items():
        weight_sum = max(cell["weight_sum"], 1e-12)
        report[regime] = {
            "n_rows": cell["n_rows"],
            "weighted_mse": cell["weighted_sse"] / weight_sum,
            "prediction_min": cell["prediction_min"] if cell["n_rows"] else 0.0,
            "prediction_max": cell["prediction_max"] if cell["n_rows"] else 0.0,
            "prediction_mean": cell["pred_wsum"] / weight_sum,
            "target_mean": cell["target_wsum"] / weight_sum,
        }
    return report


def _contrast_result(values, seeds, selected) -> dict:
    lcb, draws = paired_seed_bootstrap_lcb(values[selected], seeds[selected])
    unique = np.unique(seeds[selected])
    per_seed = np.asarray(
        [values[selected & (seeds == seed)].mean() for seed in unique]
    )
    return {
        "mean": float(values[selected].mean()),
        "LCB95": lcb,
        "bootstrap_mean": float(draws.mean()),
        "n_states": int(selected.sum()),
        "n_distinct_dev_seeds": int(len(unique)),
        "per_seed_mean": float(per_seed.mean()),
        "per_seed_sd": float(per_seed.std(ddof=1)) if len(per_seed) > 1 else 0.0,
        "correct_rate": float((values[selected] > 0).mean()),
        "pass": bool(lcb > 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if OUTPUT.exists():
        raise SystemExit(f"REFUSING: {OUTPUT} already exists; qualification is one-shot")
    require_dev_collection_gate()
    if not COMPLETE.is_file():
        raise SystemExit(f"REFUSING: DEV collection is incomplete: {COMPLETE}")

    frozen = json.loads(REGIME_RECORD.read_text(encoding="utf-8"))
    declared = frozen["weights"]["sha256"]
    actual_r1 = sha256_file(R1_WEIGHTS)
    actual_s0 = sha256_file(S0_WEIGHTS)
    if declared != R1_SHA or actual_r1 != declared:
        raise SystemExit(
            f"REFUSING: regime Q_psi hash mismatch "
            f"(record={declared}, expected={R1_SHA}, actual={actual_r1})"
        )
    if actual_s0 != S0_SHA:
        raise SystemExit(f"REFUSING: S0 Q_psi hash mismatch: {actual_s0}")
    for name, path in TEACHERS.items():
        if sha256_file(path) != EXPECTED_TEACHER_SHA[name]:
            raise SystemExit(f"REFUSING: frozen {name} teacher hash mismatch")

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    r1 = _load_model(R1_WEIGHTS, device, expected_regimes=4)
    s0 = _load_model(S0_WEIGHTS, device, expected_regimes=1)
    data = _load_dev_branch()

    branch_vec = torch.as_tensor(data.b_vec, dtype=torch.float32, device=device)
    try:
        branch_regime = r1.regime_from_vec(branch_vec).cpu().numpy()
    except ValueError as exc:
        raise SystemExit(f"REFUSING: regime reconstruction failed closed: {exc}") from exc

    support = support_counts(data.b_pole, branch_regime, data.b_seed)
    invalid_cells = [
        name
        for name, count in support.items()
        if count["n_distinct_dev_seeds"] < MIN_SUPPORT_SEEDS
    ]
    base_record = {
        "record": "RASR-PPO DEV four-regime scorer qualification",
        "status": "FROZEN_RESULT",
        "utc": _now(),
        "one_shot": True,
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "data": {
            "path": str(DATA_DIR.relative_to(ROOT)),
            "seed_block": [DEV_SEEDS[0], DEV_SEEDS[-1], len(DEV_SEEDS)],
            "use": "qualification only; never fitting",
            "train_104_seeds_used_for_fitting": False,
            "final_106_seeds_touched": False,
        },
        "scorers": {
            "R1": {"path": str(R1_WEIGHTS.relative_to(ROOT)), "sha256": actual_r1},
            "S0": {"path": str(S0_WEIGHTS.relative_to(ROOT)), "sha256": actual_s0},
        },
        "support_validity": {
            "minimum_distinct_dev_seeds_per_pole_regime": MIN_SUPPORT_SEEDS,
            "cells": support,
            "invalid_cells": invalid_cells,
        },
        "bootstrap": {
            "unit": "DEV seed; retain every state belonging to a sampled seed",
            "samples": N_BOOT,
            "alpha": ALPHA,
            "rng_seed": RNG_SEED,
            "procedure": "paired percentile bootstrap",
        },
    }
    if invalid_cells:
        base_record.update(
            {
                "verdict": "INVALID",
                "primary_gate": {},
                "specific_D1_repair_gate": {},
                "secondary_reports_not_gates": {},
                "consequence": "STOP without data growth; support validity failed",
            }
        )
        OUTPUT.write_text(json.dumps(base_record, indent=2), encoding="utf-8")
        print(f"RASR scorer qualification INVALID -> {OUTPUT}")
        return 0

    from rl.custom_ppo import load_custom_ppo_policy
    import experiments.r2_learned_crossover as R2

    probe = R2.build_env(device, DEV_SEEDS[0])
    observation_space, action_space = probe.observation_space, probe.action_space
    probe.close()
    teachers = {
        name: load_custom_ppo_policy(
            str(path), observation_space, action_space, device=device
        )
        for name, path in TEACHERS.items()
    }
    teacher_validity = {
        "pi_A": assert_teacher_query_valid(
            teachers["pi_A"],
            _teacher_validity_for(data, teacher_idx=0, actions=data.b_action_A),
            0,
            device,
        ),
        "pi_B": assert_teacher_query_valid(
            teachers["pi_B"],
            _teacher_validity_for(data, teacher_idx=1, actions=data.b_action_B),
            1,
            device,
        ),
    }

    r1_values = _expected_values(r1, data, teachers, device)
    s0_values = _expected_values(s0, data, teachers, device)
    d_a_r1 = r1_values["pi_A"] - r1_values["pi_B"]
    d_b_r1 = r1_values["pi_B"] - r1_values["pi_A"]
    d_b_s0 = s0_values["pi_B"] - s0_values["pi_A"]

    primary = {}
    for regime in range(4):
        for pole, name, values in (
            (0, "d_A", d_a_r1),
            (1, "d_B", d_b_r1),
        ):
            selected = (data.b_pole == pole) & (branch_regime == regime)
            primary[f"{name}_regime_{regime}"] = _contrast_result(
                values, data.b_seed, selected
            )

    stolen = (data.b_pole == 1) & np.isin(branch_regime, (2, 3))
    repair_delta = d_b_r1 - d_b_s0
    repair = _contrast_result(repair_delta, data.b_seed, stolen)
    repair.update(
        {
            "definition": "d_B_stolen_R1 - d_B_stolen_S0",
            "stolen_regimes": [2, 3],
            "byte_identical_states_and_masks": True,
        }
    )

    primary_pass = all(result["pass"] for result in primary.values())
    passed = primary_pass and repair["pass"]
    base_record.update(
        {
            "verdict": "PASS" if passed else "FAIL",
            "expectation": "analytic over masked teacher action distributions",
            "teacher_query_validity": teacher_validity,
            "primary_gate": {
                "requirement": "LCB95(d_A_r)>0 and LCB95(d_B_r)>0 for r=0,1,2,3",
                "results": primary,
                "pass": primary_pass,
            },
            "specific_D1_repair_gate": repair,
            "secondary_reports_not_gates": {
                "correct_rate_by_pole_and_regime": {
                    key: value["correct_rate"] for key, value in primary.items()
                },
                "R1_weighted_mse_prediction_and_calibration_by_regime":
                    _regression_report(r1, device),
                "S0_weighted_mse_prediction_and_calibration_by_regime":
                    _regression_report(s0, device),
            },
            "consequence": (
                "Qualification PASS. Policy launch remains blocked until a separate "
                "explicit human step or freeze_rasr_policy_launch_gate.py updates "
                "policy_launch_authorized."
                if passed
                else "STOP before all policy arms; no replacement scorer or data growth"
            ),
        }
    )
    OUTPUT.write_text(json.dumps(base_record, indent=2), encoding="utf-8")
    print(f"RASR scorer qualification {base_record['verdict']} -> {OUTPUT}")
    if passed:
        print(
            "PASS does not authorize policy launch. A separate explicit human step "
            "or freeze_rasr_policy_launch_gate.py is required."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
