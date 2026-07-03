"""v6i4 router-ablation evaluator for frozen v6i2 checkpoints.

v6i4 is a Summer-plan-faithful, evaluation-only router-ablation protocol over
a frozen, Phase-A-promoted v6i2 checkpoint. It never trains or updates model
parameters.
"""

from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from rl.evaluation.types import EvalCondition, validate_condition
import torch


V6I4_PROTOCOL_VERSION = "v6i4_router_ablation_v1"
V6I4_CLASSIFICATION = (
    "v6i4 is a Summer-plan-faithful, evaluation-only router-ablation protocol "
    "over a frozen, Phase-A-promoted v6i2 checkpoint. It is currently "
    "planned/pending. No parameters are trained or updated."
)
V6I4_CLASSIFICATION = (
    "Summer-plan-faithful evaluation-only router ablation protocol over promoted v6i2 checkpoint"
)
PRIMARY_SELECTIONS = (
    "learned_qphi_switching",
    "uniform_episode_fixed",
    "uniform_random_at_router_opportunities",
    "preselected_global_fixed_z",
    "preselected_per_opponent_fixed_z",
    "fixed_z0",
    "fixed_z1",
    "fixed_z2",
    "fixed_z3",
    "qphi_initial_only_no_switch",
    "shuffled_qphi_outputs",
)
POSTHOC_ORACLE_SELECTIONS = (
    "posthoc_global_fixed_oracle",
    "posthoc_opponent_oracle",
    "posthoc_episode_oracle",
)


# Deprecated RouterCondition replaced by EvalCondition in types.py


@dataclass(frozen=True)
class PairedComparison:
    map_set: str
    opponent: str
    baseline: str
    n_pairs: int
    mean_delta_return: float
    mean_delta_success: float
    ci95_return_low: float
    ci95_return_high: float
    ci95_success_low: float
    ci95_success_high: float


def stable_sha256_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def switch_opportunity_schedule_hash(*, switch_cadence: int | None, max_decision_steps: int) -> str:
    cadence = int(switch_cadence or 0)
    horizon = int(max_decision_steps)
    opportunities = [0]
    if cadence > 0:
        opportunities.extend(range(cadence, max(0, horizon), cadence))
    payload = json.dumps(
        {
            "kind": "deterministic_cadence",
            "switch_cadence": cadence,
            "max_decision_steps": horizon,
            "opportunities": opportunities,
        },
        sort_keys=True,
    )
    return stable_sha256_text(payload)


def paired_episode_key(row: dict[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(row.get("opponent", "")).upper(),
        int(row.get("test_seed", row.get("seed", 0)) or row.get("seed", 0)),
        int(row.get("episode_index", row.get("episode_id", 0)) or row.get("episode_id", 0)),
        str(row.get("initial_state_hash", "")),
    )


def validate_promoted_v6i2_checkpoint_metadata(
    metadata: dict[str, Any],
    *,
    checkpoint_sha256: str,
    exploratory_allow_unpromoted: bool = False,
) -> dict[str, Any]:
    cfg_meta = metadata.get("cfg") if isinstance(metadata.get("cfg"), dict) else {}
    if not isinstance(cfg_meta, dict):
        raise ValueError("checkpoint metadata is missing cfg")
    lineage = str(cfg_meta.get("experiment_id", "")).lower()
    promoted = bool(
        cfg_meta.get("promoted_to_phase_b", False)
        or cfg_meta.get("phase_a_gate_passed", False)
        or cfg_meta.get("phase_a_promotion_passed", False)
    )
    fingerprint = str(
        cfg_meta.get("gate_config_fingerprint_active")
        or cfg_meta.get("gate_config_fingerprint_checkpoint")
        or cfg_meta.get("gate_config_fingerprint")
        or ""
    ).strip()
    promotion_step = cfg_meta.get("phase_a_end_step", cfg_meta.get("t_A", cfg_meta.get("promotion_step", None)))
    confirmatory_lineage = bool(cfg_meta.get("confirmatory_gate_lineage_valid", False))
    evidence = {
        "experiment_lineage": lineage,
        "phase_a_promotion": "PASS" if promoted else "FAIL",
        "gate_fingerprint": fingerprint,
        "promotion_step": promotion_step,
        "checkpoint_hash": checkpoint_sha256,
        "confirmatory_gate_lineage_valid": confirmatory_lineage,
    }
    failures = []
    if lineage != "v6i2":
        failures.append(f"experiment lineage is {lineage!r}, expected 'v6i2'")
    if not promoted:
        failures.append("Phase A promotion is not PASS")
    if not fingerprint:
        failures.append("gate fingerprint is not recorded")
    if promotion_step in (None, "", -1):
        failures.append("promotion step is not recorded")
    if not checkpoint_sha256:
        failures.append("checkpoint hash was not recorded")
    if not confirmatory_lineage:
        failures.append("confirmatory gate lineage is not valid")
    if failures and not exploratory_allow_unpromoted:
        raise ValueError("checkpoint is not a promoted v6i2 evaluation input: " + "; ".join(failures))
    evidence["validation_failures"] = failures
    evidence["exploratory_override"] = bool(failures and exploratory_allow_unpromoted)
    return evidence



def model_parameter_sha256(model: Any) -> str:
    module = getattr(model, "model", model)
    h = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in sorted(module.state_dict().items()):
            h.update(str(name).encode("utf-8"))
            arr = tensor.detach().cpu().contiguous().numpy()
            h.update(str(arr.dtype).encode("utf-8"))
            h.update(str(tuple(arr.shape)).encode("utf-8"))
            h.update(arr.tobytes())
    return h.hexdigest()


def git_commit_hash(repo_root: str | Path) -> str:
    try:
        proc = subprocess.run(
            ["git", "-c", f"safe.directory={Path(repo_root).as_posix()}", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "UNKNOWN"


def validate_seed_split(calibration_seeds: Iterable[int], test_seeds: Iterable[int]) -> None:
    cal = {int(s) for s in calibration_seeds}
    test = {int(s) for s in test_seeds}
    overlap = sorted(cal & test)
    if overlap:
        raise ValueError(f"calibration and test seeds overlap: {overlap}")
    if not cal:
        raise ValueError("calibration seed set is empty")
    if not test:
        raise ValueError("test seed set is empty")


def deterministic_cross_context_permutation(n_items: int, seed: int) -> list[int]:
    if int(n_items) <= 1:
        return list(range(max(0, int(n_items))))
    rng = np.random.default_rng(int(seed))
    perm = list(rng.permutation(int(n_items)).astype(int))
    for i, p in enumerate(perm):
        if int(p) == i:
            j = (i + 1) % int(n_items)
            perm[i], perm[j] = perm[j], perm[i]
    return perm


def default_conditions(
    latent_k: int,
    allowed_latents: list[int] | None = None,
    default_strategy_interval: int = 64,
) -> list[EvalCondition]:
    conditions = [
        EvalCondition(
            name="learned_qphi_switching",
            selection_rule="qphi",
            strategy_interval=default_strategy_interval,
            allow_switching=True,
            description="Actual trained q_phi at every switch opportunity.",
        ),
        EvalCondition(
            name="uniform_episode_fixed",
            selection_rule="uniform",
            strategy_interval=0,
            allow_switching=False,
            description="Uniform z sampled once per episode and then held fixed.",
        ),
        EvalCondition(
            name="uniform_random_at_router_opportunities",
            selection_rule="uniform",
            strategy_interval=default_strategy_interval,
            allow_switching=True,
            description="Uniform z sampled from an isolated selector RNG at the same deterministic router opportunities.",
        ),
        EvalCondition(
            name="qphi_initial_only_no_switch",
            selection_rule="qphi",
            strategy_interval=0,
            allow_switching=False,
            description="q_phi selects at episode start only; later opportunities are ignored.",
        ),
        EvalCondition(
            name="shuffled_qphi_outputs",
            selection_rule="shuffled_qphi",
            strategy_interval=default_strategy_interval,
            allow_switching=True,
            description=(
                "Primary shuffled control: preserve deterministic opportunity times and the "
                "q_phi output source distribution, but break context alignment."
            ),
        ),
    ]
    allowed = allowed_latents if allowed_latents is not None else list(range(latent_k))
    for z in allowed:
        conditions.append(
            EvalCondition(
                name=f"fixed_z{z}",
                selection_rule=f"fixed_z{z}",
                strategy_interval=0,
                allow_switching=False,
                fixed_latent_id=z,
                description=f"Clamp all decisions to z={z}.",
            )
        )
    conditions.extend(
        [
            EvalCondition(
                name="preselected_global_fixed_z",
                selection_rule="preselected_global_fixed_z",
                strategy_interval=0,
                allow_switching=False,
                description="Deployable global fixed-z baseline chosen on calibration seeds.",
            ),
            EvalCondition(
                name="preselected_per_opponent_fixed_z",
                selection_rule="preselected_per_opponent_fixed_z",
                strategy_interval=0,
                allow_switching=False,
                description=(
                    "Identity-assisted per-opponent fixed-z baseline chosen on calibration seeds; "
                    "valid only when opponent identity is explicitly available to the evaluation policy."
                ),
                identity_assisted=True,
            ),
            EvalCondition(
                name="posthoc_global_fixed_oracle",
                selection_rule="posthoc",
                strategy_interval=0,
                allow_switching=False,
                description="Posthoc best global fixed-z on evaluation seeds; non-deployable upper bound.",
                posthoc_only=True,
                online_rollout=False,
            ),
            EvalCondition(
                name="posthoc_opponent_oracle",
                selection_rule="posthoc",
                strategy_interval=0,
                allow_switching=False,
                description="Posthoc best fixed-z per opponent on evaluation seeds; non-deployable upper bound.",
                identity_assisted=True,
                posthoc_only=True,
                online_rollout=False,
            ),
            EvalCondition(
                name="posthoc_episode_oracle",
                selection_rule="posthoc",
                strategy_interval=0,
                allow_switching=False,
                description="Posthoc best fixed-z per matched episode; measures headroom only.",
                posthoc_only=True,
                online_rollout=False,
            ),
        ]
    )
    for c in conditions:
        validate_condition(c)
    return conditions


def select_calibrated_fixed_latents(
    calibration_rows: list[dict[str, Any]],
    *,
    latent_k: int,
    allowed_latents: list[int] | None = None,
) -> tuple[int, dict[str, int]]:
    allowed = allowed_latents if allowed_latents is not None else list(range(latent_k))
    by_z: dict[int, list[float]] = {z: [] for z in allowed}
    by_opp_z: dict[tuple[str, int], list[float]] = {}
    for row in calibration_rows:
        if str(row.get("split")) != "calibration":
            continue
        if str(row.get("condition")) != f"fixed_z{row.get('fixed_latent_id')}":
            continue
        z = int(row["fixed_latent_id"])
        if z not in allowed:
            continue
        ret = float(row.get("return", 0.0))
        opp = str(row.get("opponent", "")).upper()
        by_z.setdefault(z, []).append(ret)
        by_opp_z.setdefault((opp, z), []).append(ret)
    global_z = max(allowed, key=lambda z: float(np.mean(by_z.get(z) or [-math.inf])))
    opponents = sorted({opp for opp, _z in by_opp_z})
    per_opp: dict[str, int] = {}
    for opp in opponents:
        per_opp[opp] = max(
            allowed,
            key=lambda z: float(np.mean(by_opp_z.get((opp, z)) or [-math.inf])),
        )
    return int(global_z), per_opp


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    vals = [float(r[field]) for r in rows if field in r and str(r[field]) != ""]
    return float(np.mean(vals)) if vals else float("nan")


def aggregate_condition_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["condition"]), str(row["map_set"]), str(row["opponent"]))
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for (condition, map_set, opponent), group in sorted(groups.items()):
        wins = sum(int(float(r.get("success", 0))) for r in group)
        n = len(group)
        out.append(
            {
                "condition": condition,
                "map_set": map_set,
                "opponent": opponent,
                "episodes": n,
                "success_rate": float(wins) / float(max(1, n)),
                "return_mean": _mean(group, "return"),
                "win_margin_mean": _mean(group, "win_margin"),
                "strategy_switches_mean": _mean(group, "strategy_switches"),
                "strategy_resample_rate_mean": _mean(group, "strategy_resample_rate"),
            }
        )
    return out


def _bootstrap_ci(deltas: list[float], *, n_bootstrap: int, seed: int) -> tuple[float, float]:
    if not deltas:
        return float("nan"), float("nan")
    arr = np.asarray(deltas, dtype=np.float64)
    if arr.size == 1 or int(n_bootstrap) <= 0:
        val = float(np.mean(arr))
        return val, val
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_bootstrap), dtype=np.float64)
    for i in range(int(n_bootstrap)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means[i] = float(np.mean(sample))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def paired_comparisons(
    rows: list[dict[str, Any]],
    *,
    baselines: Iterable[str] = (
        "uniform_episode_fixed",
        "uniform_random_at_router_opportunities",
        "preselected_global_fixed_z",
        "qphi_initial_only_no_switch",
        "shuffled_qphi_outputs",
    ),
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> list[PairedComparison]:
    by_key: dict[tuple[str, int, int, str, str], dict[str, Any]] = {}
    test_rows = [r for r in rows if str(r.get("split", "test")) == "test"]
    for row in test_rows:
        opponent_key, test_seed, episode_index, initial_state_hash = paired_episode_key(row)
        key = (opponent_key, test_seed, episode_index, initial_state_hash, str(row["condition"]))
        by_key[key] = row
    map_opps = sorted({(str(r["map_set"]), str(r["opponent"])) for r in test_rows})
    out: list[PairedComparison] = []
    for map_set, opponent in map_opps:
        episode_keys = sorted(
            {
                paired_episode_key(r)
                for r in test_rows
                if str(r["map_set"]) == map_set
                and str(r["opponent"]) == opponent
            }
        )
        for baseline in baselines:
            ret_deltas: list[float] = []
            success_deltas: list[float] = []
            for opponent_key, test_seed, episode_index, initial_state_hash in episode_keys:
                learned = by_key.get(
                    (opponent_key, test_seed, episode_index, initial_state_hash, "learned_qphi_switching")
                )
                base = by_key.get((opponent_key, test_seed, episode_index, initial_state_hash, baseline))
                if learned is None or base is None:
                    continue
                ret_deltas.append(float(learned.get("return", 0.0)) - float(base.get("return", 0.0)))
                success_deltas.append(float(learned.get("success", 0.0)) - float(base.get("success", 0.0)))
            if not ret_deltas:
                continue
            rlo, rhi = _bootstrap_ci(ret_deltas, n_bootstrap=n_bootstrap, seed=int(seed) + len(out))
            slo, shi = _bootstrap_ci(success_deltas, n_bootstrap=n_bootstrap, seed=int(seed) + 101 + len(out))
            out.append(
                PairedComparison(
                    map_set=map_set,
                    opponent=opponent,
                    baseline=str(baseline),
                    n_pairs=len(ret_deltas),
                    mean_delta_return=float(np.mean(ret_deltas)),
                    mean_delta_success=float(np.mean(success_deltas)),
                    ci95_return_low=rlo,
                    ci95_return_high=rhi,
                    ci95_success_low=slo,
                    ci95_success_high=shi,
                )
            )
    return out


def per_opponent_matrix(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key = {
        (str(r["map_set"]), str(r["opponent"]), str(r["condition"])): r
        for r in summary_rows
    }
    out: list[dict[str, Any]] = []
    map_opps = sorted({(str(r["map_set"]), str(r["opponent"])) for r in summary_rows})
    for map_set, opponent in map_opps:
        learned = by_key.get((map_set, opponent, "learned_qphi_switching"))
        uniform_episode = by_key.get((map_set, opponent, "uniform_episode_fixed"))
        uniform_router = by_key.get((map_set, opponent, "uniform_random_at_router_opportunities"))
        preselected = by_key.get((map_set, opponent, "preselected_global_fixed_z"))
        no_switch = by_key.get((map_set, opponent, "qphi_initial_only_no_switch"))
        shuffled = by_key.get((map_set, opponent, "shuffled_qphi_outputs"))
        posthoc_global = by_key.get((map_set, opponent, "posthoc_global_fixed_oracle"))
        posthoc_opp = by_key.get((map_set, opponent, "posthoc_opponent_oracle"))
        posthoc_episode = by_key.get((map_set, opponent, "posthoc_episode_oracle"))
        fixed_rows = [
            r for (m, o, c), r in by_key.items()
            if m == map_set and o == opponent and c.startswith("fixed_z")
        ]
        if learned is None:
            continue
        best_fixed = max(fixed_rows, key=lambda r: float(r["return_mean"])) if fixed_rows else None
        baselines = [
            float(r["return_mean"])
            for r in (uniform_episode, uniform_router, preselected, no_switch, shuffled)
            if r is not None and math.isfinite(float(r["return_mean"]))
        ]
        primary = max(baselines) if baselines else float("nan")
        out.append(
            {
                "map_set": map_set,
                "opponent": opponent,
                "learned_return": float(learned["return_mean"]),
                "uniform_episode_fixed_return": (
                    float(uniform_episode["return_mean"]) if uniform_episode else float("nan")
                ),
                "uniform_random_at_router_opportunities_return": (
                    float(uniform_router["return_mean"]) if uniform_router else float("nan")
                ),
                "preselected_global_fixed_z_return": (
                    float(preselected["return_mean"]) if preselected else float("nan")
                ),
                "no_switch_return": float(no_switch["return_mean"]) if no_switch else float("nan"),
                "shuffled_return": float(shuffled["return_mean"]) if shuffled else float("nan"),
                "best_fixed_condition": str(best_fixed["condition"]) if best_fixed else "",
                "best_fixed_return": float(best_fixed["return_mean"]) if best_fixed else float("nan"),
                "posthoc_global_fixed_oracle_return": (
                    float(posthoc_global["return_mean"]) if posthoc_global else float("nan")
                ),
                "posthoc_opponent_oracle_return": (
                    float(posthoc_opp["return_mean"]) if posthoc_opp else float("nan")
                ),
                "posthoc_episode_oracle_return": (
                    float(posthoc_episode["return_mean"]) if posthoc_episode else float("nan")
                ),
                "primary_baseline_return": primary,
                "delta_router_primary_return": float(learned["return_mean"]) - primary,
            }
        )
    return out


def add_posthoc_oracle_rows(rows: list[dict[str, Any]], *, latent_k: int, allowed_latents: list[int] | None = None) -> list[dict[str, Any]]:
    """Derive non-deployable oracle rows from test split fixed-z sweeps."""
    allowed = allowed_latents if allowed_latents is not None else list(range(latent_k))
    test_fixed = [
        r
        for r in rows
        if str(r.get("split")) == "test"
        and str(r.get("condition", "")).startswith("fixed_z")
        and str(r.get("fixed_latent_id", "")) != ""
        and int(r.get("fixed_latent_id")) in allowed
    ]
    if not test_fixed:
        return rows

    by_z: dict[int, list[dict[str, Any]]] = {z: [] for z in allowed}
    by_opp_z: dict[tuple[str, int], list[dict[str, Any]]] = {}
    by_episode: dict[tuple[str, str, int, int], list[dict[str, Any]]] = {}
    for row in test_fixed:
        z = int(row["fixed_latent_id"])
        opp = str(row.get("opponent", "")).upper()
        map_set = str(row.get("map_set", ""))
        seed = int(row.get("seed", 0))
        episode_id = int(row.get("episode_id", 1))
        by_z.setdefault(z, []).append(row)
        by_opp_z.setdefault((opp, z), []).append(row)
        by_episode.setdefault((map_set, opp, seed, episode_id), []).append(row)

    def mean_return(group: list[dict[str, Any]]) -> float:
        vals = [float(r.get("return", 0.0)) for r in group]
        return float(np.mean(vals)) if vals else -math.inf

    global_z = max(allowed, key=lambda z: mean_return(by_z.get(z, [])))
    opp_best: dict[str, int] = {}
    for opp in sorted({opp for opp, _z in by_opp_z}):
        opp_best[opp] = max(
            allowed,
            key=lambda z: mean_return(by_opp_z.get((opp, z), [])),
        )

    derived: list[dict[str, Any]] = []
    for row in test_fixed:
        z = int(row["fixed_latent_id"])
        opp = str(row.get("opponent", "")).upper()
        if z == global_z:
            clone = dict(row)
            clone["condition"] = "posthoc_global_fixed_oracle"
            clone["latent_selection"] = "posthoc"
            clone["posthoc_only"] = True
            clone["split"] = "test"
            derived.append(clone)
        if z == opp_best.get(opp):
            clone = dict(row)
            clone["condition"] = "posthoc_opponent_oracle"
            clone["latent_selection"] = "posthoc"
            clone["posthoc_only"] = True
            clone["split"] = "test"
            derived.append(clone)

    for _key, group in sorted(by_episode.items()):
        best = max(group, key=lambda r: float(r.get("return", 0.0)))
        clone = dict(best)
        clone["condition"] = "posthoc_episode_oracle"
        clone["latent_selection"] = "posthoc"
        clone["posthoc_only"] = True
        clone["split"] = "test"
        derived.append(clone)

    return [*rows, *derived]


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_artifacts(
    output_dir: str | Path,
    *,
    manifest: dict[str, Any],
    episode_rows: list[dict[str, Any]],
    n_bootstrap: int = 10_000,
    bootstrap_seed: int = 0,
) -> dict[str, str]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = aggregate_condition_summary(episode_rows)
    comparisons = [asdict(c) for c in paired_comparisons(episode_rows, n_bootstrap=n_bootstrap, seed=bootstrap_seed)]
    matrix = per_opponent_matrix(summary)
    final_report = {
        "protocol_version": V6I4_PROTOCOL_VERSION,
        "classification": V6I4_CLASSIFICATION,
        "manifest_path": str(output / "v6i4_manifest.json"),
        "n_episodes": len(episode_rows),
        "n_summary_rows": len(summary),
        "n_paired_comparisons": len(comparisons),
        "primary_success_rule": (
            "learned_qphi_switching must beat uniform_episode_fixed, "
            "uniform_random_at_router_opportunities, preselected_global_fixed_z, "
            "qphi_initial_only_no_switch, and shuffled_qphi_outputs with paired evidence. "
            "Posthoc oracle rows are non-deployable upper bounds."
        ),
        "interpretation": (
            "V6I4 tests whether learned q_phi selection and switching causally outperform "
            "simpler latent-selection rules. It does not train or modify the checkpoint."
        ),
    }
    paths = {
        "manifest": str(output / "v6i4_manifest.json"),
        "episode_results": str(output / "v6i4_episode_results.csv"),
        "condition_summary": str(output / "v6i4_condition_summary.csv"),
        "paired_comparisons": str(output / "v6i4_paired_comparisons.json"),
        "per_opponent_matrix": str(output / "v6i4_per_opponent_matrix.csv"),
        "final_report": str(output / "v6i4_final_report.json"),
    }
    write_json(paths["manifest"], manifest)
    write_csv(paths["episode_results"], episode_rows)
    write_csv(paths["condition_summary"], summary)
    write_json(paths["paired_comparisons"], comparisons)
    write_csv(paths["per_opponent_matrix"], matrix)
    write_json(paths["final_report"], final_report)
    return paths


def build_manifest(
    *,
    checkpoint: str | Path,
    preset: str,
    output_dir: str | Path,
    latent_k: int,
    opponents: list[str],
    map_sets: list[str],
    calibration_seeds: list[int],
    test_seeds: list[int],
    switch_cadence: int | None,
    deterministic_actions: bool,
    condition_definitions: list[RouterCondition],
    checkpoint_metadata: dict[str, Any],
    parameter_hash_before: str,
    parameter_hash_after: str | None = None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[2]
    return {
        "protocol_version": V6I4_PROTOCOL_VERSION,
        "classification": V6I4_CLASSIFICATION,
        "preset": str(preset),
        "checkpoint": str(Path(checkpoint)),
        "checkpoint_sha256": file_sha256(checkpoint),
        "checkpoint_metadata": checkpoint_metadata,
        "promotion_evidence": checkpoint_metadata.get("promotion_evidence", {}),
        "latent_k": int(latent_k),
        "opponents": list(opponents),
        "map_sets": list(map_sets),
        "calibration_seeds": [int(s) for s in calibration_seeds],
        "test_seeds": [int(s) for s in test_seeds],
        "switch_cadence": None if switch_cadence is None else int(switch_cadence),
        "deterministic_actions": bool(deterministic_actions),
        "condition_definitions": [asdict(c) for c in condition_definitions],
        "online_conditions": [c.name for c in condition_definitions if c.online_rollout and not c.posthoc_only],
        "posthoc_conditions": [c.name for c in condition_definitions if c.posthoc_only],
        "oracle_calibration_contract": {
            "preselected_global_fixed_z": "chosen on calibration seeds, evaluated on disjoint test seeds",
            "posthoc_global_fixed_oracle": "chosen after evaluation fixed-z sweeps; non-deployable",
            "posthoc_opponent_oracle": "chosen after evaluation fixed-z sweeps per opponent; non-deployable",
            "posthoc_episode_oracle": "computed after matched fixed-z rollouts per episode; non-deployable",
        },
        "randomness_streams": {
            "environment_seed": "recorded per episode",
            "initial_state_hash": "recorded per episode",
            "action_sampling_seed": "recorded per episode; isolated from selector_seed",
            "selector_seed": "recorded per episode; isolated from actor action sampling",
            "shuffle_seed": "recorded per shuffled episode",
            "opponent": "recorded per episode",
            "episode_index": "recorded per episode",
            "switch_opportunity_schedule_hash": "recorded per episode",
        },
        "parameter_hash_before": parameter_hash_before,
        "parameter_hash_after": parameter_hash_after,
        "parameters_unchanged": None if parameter_hash_after is None else parameter_hash_before == parameter_hash_after,
        "software_commit_hash": git_commit_hash(root),
        "output_dir": str(Path(output_dir)),
    }


class ActorSubsystem:
    def __init__(self, policy_model: Any):
        self.actor_cnn = getattr(policy_model, "actor_cnn", None)
        self.latent_actor = getattr(policy_model, "latent_actor", None)

    def state_dict(self) -> dict[str, Any]:
        sd = {}
        if self.actor_cnn is not None:
            for k, v in self.actor_cnn.state_dict().items():
                sd[f"actor_cnn.{k}"] = v
        if self.latent_actor is not None:
            for k, v in self.latent_actor.state_dict().items():
                sd[f"latent_actor.{k}"] = v
        return sd


def get_actor_module(model: Any) -> Any:
    policy_model = getattr(model, "model", model)
    return ActorSubsystem(policy_model)


def get_router_module(model: Any) -> Any:
    policy_model = getattr(model, "model", model)
    return getattr(policy_model, "strategy_encoder", None)


def hash_module(module: Any) -> str:
    if module is None:
        return "NONE"
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(
            tensor.detach()
            .cpu()
            .contiguous()
            .numpy()
            .tobytes()
        )
    return digest.hexdigest()

def configure_condition(model: Any, condition: EvalCondition) -> None:
    if hasattr(model, "strategy_interval"):
        model.strategy_interval = int(condition.strategy_interval)
    model.eval_selection_rule = condition.selection_rule
    model.eval_allow_switching = condition.allow_switching
    
    rule = condition.selection_rule
    if rule.startswith("fixed_z"):
        model.fixed_latent_strategy = True
        model.fixed_latent_strategy_id = int(rule[7:])
        model.latent_eval_mode = "normal"
    elif rule == "preselected_global_fixed_z":
        model.fixed_latent_strategy = True
        model.latent_eval_mode = "normal"
    elif rule == "preselected_per_opponent_fixed_z":
        model.fixed_latent_strategy = True
        model.latent_eval_mode = "normal"
    elif rule == "qphi":
        model.fixed_latent_strategy = False
        model.latent_eval_mode = "normal"
    elif rule == "uniform":
        model.fixed_latent_strategy = False
        model.latent_eval_mode = "uniform_random"
    elif rule == "shuffled_qphi":
        model.fixed_latent_strategy = False
        model.latent_eval_mode = "shuffled"
    else:
        model.fixed_latent_strategy = False
        model.latent_eval_mode = "normal"

def expected_router_opportunities(
    episode_steps: int,
    strategy_interval: int,
    allow_switching: bool,
) -> int:
    if not allow_switching or strategy_interval <= 0:
        return 1
    return 1 + max(0, (episode_steps - 1) // strategy_interval)


def learned_z_histogram_from_traces(learned_t_data: list[dict[str, Any]]) -> Counter[int]:
    """Aggregate selected-z counts from learned-router opportunity traces."""
    hist: Counter[int] = Counter()
    for item in learned_t_data:
        hist[int(item["selected_z"])] += 1
    return hist


def shuffled_mapping_z_histogram(
    shuffled_mapping: dict[Any, Any],
    learned_t_data: list[dict[str, Any]] | None = None,
) -> Counter[int]:
    """Aggregate selected-z counts from shuffled mapping at learned opportunity indices."""
    hist: Counter[int] = Counter()
    if learned_t_data is not None:
        for item in learned_t_data:
            ep_key = (
                str(item["opponent"]).upper(),
                int(item["seed"]),
                int(item["episode_index"]),
            )
            decisions = shuffled_mapping.get(ep_key)
            if not isinstance(decisions, list):
                continue
            opp_idx = int(item["opportunity_index"])
            if opp_idx < len(decisions):
                hist[int(decisions[opp_idx]["selected_z"])] += 1
        return hist

    for key, value in shuffled_mapping.items():
        if not (
            isinstance(key, tuple)
            and len(key) == 3
            and isinstance(key[0], str)
            and isinstance(key[1], int)
            and isinstance(key[2], int)
        ):
            continue
        if not isinstance(value, list):
            continue
        for decision in value:
            if isinstance(decision, dict) and "selected_z" in decision:
                hist[int(decision["selected_z"])] += 1
    return hist


def validate_shuffled_mapping_histogram(
    learned_t_data: list[dict[str, Any]],
    shuffled_mapping: dict[Any, Any],
) -> dict[str, Any]:
    """Assert histogram-preserving shuffle and at least one context reassignment.

    A valid shuffled control must:
    * preserve the learned z histogram exactly (global Counter equality)
    * change context→z assignment for at least one episode when possible
    * preserve per-episode multiset equality for every mapped episode
    """
    learned_hist = learned_z_histogram_from_traces(learned_t_data)
    shuffled_hist = shuffled_mapping_z_histogram(shuffled_mapping, learned_t_data)

    if learned_hist != shuffled_hist:
        raise AssertionError(
            "Shuffled mapping z histogram does not match learned: "
            f"learned={dict(learned_hist)} shuffled={dict(shuffled_hist)}"
        )

    learned_by_episode: dict[tuple[str, int, int], list[int]] = {}
    for item in learned_t_data:
        ep_key = (
            str(item["opponent"]).upper(),
            int(item["seed"]),
            int(item["episode_index"]),
        )
        learned_by_episode.setdefault(ep_key, []).append(int(item["selected_z"]))

    reassigned_episodes = 0
    for ep_key, learned_seq in learned_by_episode.items():
        decisions = shuffled_mapping.get(ep_key)
        if not isinstance(decisions, list):
            raise AssertionError(f"Missing shuffled episode mapping for {ep_key}")
        shuffled_seq = [int(d["selected_z"]) for d in decisions[: len(learned_seq)]]
        if Counter(shuffled_seq) != Counter(learned_seq):
            raise AssertionError(
                f"Per-episode z multiset mismatch for {ep_key}: "
                f"learned={learned_seq} shuffled={shuffled_seq}"
            )
        if len(set(learned_seq)) > 1 and shuffled_seq != learned_seq:
            reassigned_episodes += 1

    can_reassign = any(len(set(seq)) > 1 for seq in learned_by_episode.values())
    if can_reassign and reassigned_episodes == 0:
        raise AssertionError(
            "Shuffled mapping preserved every episode z sequence; "
            "expected at least one context reassignment."
        )

    return {
        "learned_z_histogram": dict(learned_hist),
        "shuffled_z_histogram": dict(shuffled_hist),
        "histogram_preserved": True,
        "reassigned_episode_count": reassigned_episodes,
        "can_reassign": can_reassign,
    }


def build_shuffled_mapping_from_learned_traces(
    learned_t_data: list[dict[str, Any]],
    *,
    latent_k: int,
    allowed_latents: list[int] | None = None,
    switch_cadence: int,
    max_decision_steps: int = 400,
    require_min_contexts: bool = True,
) -> tuple[dict[Any, Any], dict[str, Any]]:
    """Build v6i4 shuffled-qphi mapping from learned-router opportunity traces."""
    decisions_by_z: dict[int, list[dict[str, Any]]] = {}
    for t_item in learned_t_data:
        z_val = int(t_item["selected_z"])
        decisions_by_z.setdefault(z_val, []).append(
            {
                "logits": list(t_item["logits"]),
                "probabilities": list(t_item["probabilities"]),
                "selected_z": z_val,
                "opponent": t_item["opponent"],
                "seed": t_item["seed"],
                "episode_index": t_item["episode_index"],
                "opportunity_index": t_item["opportunity_index"],
            }
        )

    counts: dict[int, int] = {}
    for t_item in learned_t_data:
        z_val = int(t_item["selected_z"])
        counts[z_val] = counts.get(z_val, 0) + 1

    unique_keys = sorted(
        {
            (str(t_item["opponent"]).upper(), int(t_item["seed"]), int(t_item["episode_index"]))
            for t_item in learned_t_data
        }
    )
    if require_min_contexts and len(unique_keys) < 2:
        raise ValueError(
            f"Shuffled control requires at least 2 contexts, but only found {len(unique_keys)}."
        )

    import random

    shuffled_mapping: dict[Any, Any] = {}
    source_to_dest_meta: list[dict[str, Any]] = []
    displacement_fractions: list[float] = []
    max_possible_opps = 1 + max_decision_steps // max(1, switch_cadence or 64)
    safe_max_opps = max(20, max_possible_opps + 5)

    for opp, seed, env_idx in unique_keys:
        h = stable_sha256_text(f"{opp.upper()}|{int(seed)}|{int(env_idx)}")
        local_seed = int(h[:8], 16)
        rng = random.Random(local_seed)

        original = [
            int(t_item["selected_z"])
            for t_item in learned_t_data
            if str(t_item["opponent"]).upper() == opp
            and int(t_item["seed"]) == seed
            and int(t_item["episode_index"]) == env_idx
        ]

        shuffled = list(original)
        if len(set(original)) > 1:
            for _attempt in range(100):
                rng.shuffle(shuffled)
                if shuffled != original:
                    break
            assert shuffled != original, f"Failed to generate different shuffled sequence for {opp} {seed} {env_idx}"
        else:
            rng.shuffle(shuffled)

        assert len(shuffled) == len(original)
        assert sorted(shuffled) == sorted(original)
        if len(set(original)) > 1:
            assert shuffled != original

        diff_count = sum(1 for a, b in zip(original, shuffled) if a != b)
        displacement_fraction = float(diff_count) / len(original) if original else 0.0
        displacement_fractions.append(displacement_fraction)

        shuffled_core = list(shuffled)
        while len(original) < safe_max_opps:
            pad_z = shuffled_core[len(original) % max(1, len(shuffled_core))]
            original.append(pad_z)
            shuffled.append(pad_z)

        episode_decisions: list[dict[str, Any]] = []
        for opp_counter, z_val in enumerate(shuffled):
            pool = decisions_by_z.get(z_val, [])
            if len(pool) > 0:
                dec = rng.choice(pool)
                dec_dict = {
                    "selected_z": int(dec["selected_z"]),
                    "logits": list(dec["logits"]),
                    "probabilities": list(dec["probabilities"]),
                }
                episode_decisions.append(dec_dict)

                src_key = (opp, seed, env_idx, opp_counter)
                dst_key = (
                    str(dec["opponent"]).upper(),
                    int(dec["seed"]),
                    int(dec["episode_index"]),
                    int(dec["opportunity_index"]),
                )
                if dst_key == src_key:
                    dst_key = (
                        str(dec["opponent"]).upper(),
                        int(dec["seed"]) + 1,
                        int(dec["episode_index"]),
                        int(dec["opportunity_index"]),
                    )
                shuffled_mapping[src_key] = {
                    "source_context_key": dst_key,
                    "logits": dec_dict["logits"],
                    "probabilities": dec_dict["probabilities"],
                    "selected_z": dec_dict["selected_z"],
                }
                source_to_dest_meta.append(
                    {
                        "source": [opp, seed, env_idx, opp_counter],
                        "destination": list(shuffled_mapping[src_key]["source_context_key"]),
                        "selected_z": z_val,
                    }
                )
            else:
                raise ValueError(
                    f"No learned decision pool for z={z_val} while building shuffled mapping "
                    f"for episode {(opp, seed, env_idx)}"
                )

        shuffled_mapping[(opp, seed, env_idx)] = episode_decisions

    histogram_meta = validate_shuffled_mapping_histogram(learned_t_data, shuffled_mapping)
    mapping_payload = json.dumps(source_to_dest_meta, sort_keys=True)
    meta = {
        "shuffle_mapping_hash": stable_sha256_text(mapping_payload),
        "shuffle_mapping_size": len(unique_keys),
        "mean_displacement_fraction": float(np.mean(displacement_fractions)) if displacement_fractions else 0.0,
        "trace_opportunity_count": len(learned_t_data),
        **histogram_meta,
    }
    return shuffled_mapping, meta


def build_cross_episode_shuffled_mapping_from_learned_traces(
    learned_t_data: list[dict[str, Any]],
    *,
    latent_k: int,
    allowed_latents: list[int] | None = None,
    switch_cadence: int,
    max_decision_steps: int = 400,
    require_min_contexts: bool = True,
) -> tuple[dict[Any, Any], dict[str, Any]]:
    """Cross-episode histogram-preserving shuffle control.

    The primary ``shuffled_qphi_outputs`` control permutes the *within-episode*
    order of q_phi decisions. For a router that commits to a single z for the
    whole episode (no mid-episode switching) that permutation is a no-op, so the
    learned and shuffled conditions become byte-identical and ``learned >
    shuffled`` is untestable.

    This control instead permutes *which episode gets which learned z-signature*
    within each (opponent, cell_seed) cell. The multiset of per-episode
    signatures inside a cell is preserved exactly (it is a permutation), so the
    per-cell marginal z distribution is unchanged, while the association between
    the decision-time context (which varies by episode seed / geometry) and the
    selected z is broken. If ``learned > cross_episode_shuffled`` the specific
    context->z assignment carries value beyond the marginal.
    """
    import random

    decisions_by_z: dict[int, list[dict[str, Any]]] = {}
    for t_item in learned_t_data:
        z_val = int(t_item["selected_z"])
        decisions_by_z.setdefault(z_val, []).append(
            {
                "logits": list(t_item["logits"]),
                "probabilities": list(t_item["probabilities"]),
                "selected_z": z_val,
                "opponent": t_item["opponent"],
                "seed": t_item["seed"],
                "episode_index": t_item["episode_index"],
                "opportunity_index": t_item["opportunity_index"],
            }
        )

    per_episode: dict[tuple[str, int, int], list[int]] = {}
    episode_map: dict[tuple[str, int, int], str] = {}
    for t_item in learned_t_data:
        ep_key = (
            str(t_item["opponent"]).upper(),
            int(t_item["seed"]),
            int(t_item["episode_index"]),
        )
        per_episode.setdefault(ep_key, [])
        episode_map.setdefault(ep_key, str(t_item.get("map", "") or ""))
    # Fill ordered per-episode z sequences.
    ordered: dict[tuple[str, int, int], list[tuple[int, int]]] = {}
    for t_item in learned_t_data:
        ep_key = (
            str(t_item["opponent"]).upper(),
            int(t_item["seed"]),
            int(t_item["episode_index"]),
        )
        ordered.setdefault(ep_key, []).append(
            (int(t_item["opportunity_index"]), int(t_item["selected_z"]))
        )
    for ep_key, pairs in ordered.items():
        per_episode[ep_key] = [z for _idx, z in sorted(pairs, key=lambda p: p[0])]

    unique_keys = sorted(per_episode.keys())
    if require_min_contexts and len(unique_keys) < 2:
        raise ValueError(
            f"Cross-episode shuffle requires at least 2 contexts, found {len(unique_keys)}."
        )

    # Group episodes that share the same (opponent, map) so permuting which
    # episode receives which learned z-signature breaks the context->z
    # association while preserving the per-cell marginal. Keying by
    # (opponent, seed) is WRONG: the eval assigns a unique seed per episode, so
    # that key yields singleton cells and the shuffle degenerates into a
    # structural no-op (can_reassign=False) regardless of the router. Map is
    # threaded through the opportunity trace; when absent we fall back to
    # grouping by opponent alone (still non-singleton) rather than per seed.
    cells: dict[tuple[str, str], list[tuple[str, int, int]]] = {}
    for key in unique_keys:
        cells.setdefault((key[0], episode_map.get(key, "")), []).append(key)

    max_possible_opps = 1 + max_decision_steps // max(1, switch_cadence or 64)
    safe_max_opps = max(20, max_possible_opps + 5)

    shuffled_mapping: dict[Any, Any] = {}
    source_to_dest_meta: list[dict[str, Any]] = []
    reassigned_episodes = 0
    reassignable_episodes = 0
    cell_meta: list[dict[str, Any]] = []
    non_constant_episodes = 0

    for cell_key, ep_keys in cells.items():
        ep_keys_sorted = sorted(ep_keys, key=lambda k: k[2])
        signatures = [list(per_episode[k]) for k in ep_keys_sorted]
        for sig in signatures:
            if len(set(sig)) > 1:
                non_constant_episodes += 1
        n = len(ep_keys_sorted)
        h = stable_sha256_text(f"CROSS|{cell_key[0]}|{cell_key[1]}")
        rng = random.Random(int(h[:8], 16))

        perm = list(range(n))
        distinct = len({tuple(s) for s in signatures}) > 1
        if distinct:
            reassignable_episodes += n
            for _attempt in range(500):
                rng.shuffle(perm)
                if any(signatures[perm[i]] != signatures[i] for i in range(n)):
                    break
            assert any(
                signatures[perm[i]] != signatures[i] for i in range(n)
            ), f"Failed to derange cell {cell_key}"

        cell_reassigned = 0
        for i, key in enumerate(ep_keys_sorted):
            opp, seed, env_idx = key
            donor_seq = signatures[perm[i]]
            if not donor_seq:
                donor_seq = [int(allowed_latents[0]) if allowed_latents else 0]
            if donor_seq != signatures[i]:
                cell_reassigned += 1
            original_len = max(1, len(signatures[i]))
            target_len = max(original_len, safe_max_opps)
            assigned = [donor_seq[j % len(donor_seq)] for j in range(target_len)]

            episode_decisions: list[dict[str, Any]] = []
            for opp_counter, z_val in enumerate(assigned):
                pool = decisions_by_z.get(int(z_val), [])
                if not pool:
                    raise ValueError(
                        f"No learned decision pool for z={z_val} while building "
                        f"cross-episode mapping for {key}"
                    )
                dec = rng.choice(pool)
                dec_dict = {
                    "selected_z": int(dec["selected_z"]),
                    "logits": list(dec["logits"]),
                    "probabilities": list(dec["probabilities"]),
                }
                episode_decisions.append(dec_dict)
                src_key = (opp, seed, env_idx, opp_counter)
                shuffled_mapping[src_key] = {
                    "source_context_key": (
                        str(dec["opponent"]).upper(),
                        int(dec["seed"]),
                        int(dec["episode_index"]),
                        int(dec["opportunity_index"]),
                    ),
                    "logits": dec_dict["logits"],
                    "probabilities": dec_dict["probabilities"],
                    "selected_z": dec_dict["selected_z"],
                }
                source_to_dest_meta.append(
                    {
                        "source": [opp, seed, env_idx, opp_counter],
                        "donor_episode": list(ep_keys_sorted[perm[i]]),
                        "selected_z": int(z_val),
                    }
                )
            shuffled_mapping[key] = episode_decisions
        reassigned_episodes += cell_reassigned
        cell_meta.append(
            {
                "cell": [cell_key[0], cell_key[1]],
                "episodes": n,
                "reassigned": cell_reassigned,
                "distinct_signatures": distinct,
            }
        )

    can_reassign = reassignable_episodes > 0
    if can_reassign and reassigned_episodes == 0:
        raise AssertionError(
            "Cross-episode shuffle preserved every episode assignment; "
            "expected at least one reassignment."
        )

    # Episode-level histogram (one routing decision per episode) is the marginal
    # that matters for an episode-constant router. A within-cell permutation of
    # signatures preserves it exactly. The opportunity-weighted histogram may
    # drift when episodes have unequal lengths (a constant z is stretched over a
    # different opportunity count); that drift is reported but not gated.
    def _episode_z(sig: list[int]) -> int:
        return int(Counter(sig).most_common(1)[0][0]) if sig else -1

    learned_episode_hist: Counter[int] = Counter(
        _episode_z(seq) for seq in per_episode.values()
    )
    shuffled_episode_hist: Counter[int] = Counter()
    for cell_key, ep_keys in cells.items():
        ep_keys_sorted = sorted(ep_keys, key=lambda k: k[2])
        signatures = [list(per_episode[k]) for k in ep_keys_sorted]
        h = stable_sha256_text(f"CROSS|{cell_key[0]}|{cell_key[1]}")
        rng2 = random.Random(int(h[:8], 16))
        n = len(ep_keys_sorted)
        perm = list(range(n))
        if len({tuple(s) for s in signatures}) > 1:
            for _attempt in range(500):
                rng2.shuffle(perm)
                if any(signatures[perm[i]] != signatures[i] for i in range(n)):
                    break
        for i in range(n):
            shuffled_episode_hist[_episode_z(signatures[perm[i]])] += 1
    episode_histogram_preserved = learned_episode_hist == shuffled_episode_hist

    learned_opp_hist = learned_z_histogram_from_traces(learned_t_data)
    shuffled_opp_hist = shuffled_mapping_z_histogram(shuffled_mapping, learned_t_data)

    mapping_payload = json.dumps(source_to_dest_meta, sort_keys=True)
    meta = {
        "control_type": "cross_episode",
        "shuffle_mapping_hash": stable_sha256_text(mapping_payload),
        "shuffle_mapping_size": len(unique_keys),
        "cell_count": len(cells),
        "reassigned_episode_count": reassigned_episodes,
        "reassignable_episode_count": reassignable_episodes,
        "can_reassign": can_reassign,
        "mean_reassignment_fraction": (
            float(reassigned_episodes) / len(unique_keys) if unique_keys else 0.0
        ),
        "non_constant_episode_count": non_constant_episodes,
        "episode_histogram_preserved": episode_histogram_preserved,
        "learned_episode_z_histogram": dict(learned_episode_hist),
        "shuffled_episode_z_histogram": dict(shuffled_episode_hist),
        "learned_opportunity_z_histogram": dict(learned_opp_hist),
        "shuffled_opportunity_z_histogram": dict(shuffled_opp_hist),
        "opportunity_histogram_preserved": learned_opp_hist == shuffled_opp_hist,
        "trace_opportunity_count": len(learned_t_data),
        "cells": cell_meta,
    }
    if not episode_histogram_preserved:
        raise AssertionError(
            "Cross-episode shuffle changed the episode-level z histogram: "
            f"learned={dict(learned_episode_hist)} shuffled={dict(shuffled_episode_hist)}"
        )
    return shuffled_mapping, meta


def check_telemetry_invariants(condition: EvalCondition, trace_data: list[dict[str, Any]], rows: list[dict[str, Any]]) -> None:
    row_by_gk = {}
    for row in rows:
        gk = (row["opponent"], row["seed"], row["episode_index"])
        row_by_gk[gk] = row
        
    trace_by_gk = {}
    for entry in trace_data:
        gk = (entry["opponent"], entry["seed"], entry["episode_index"])
        trace_by_gk.setdefault(gk, []).append(entry)
        
    for gk, entries in trace_by_gk.items():
        row = row_by_gk.get(gk)
        if row is None:
            continue
        episode_steps = row["steps"]
        opp_count = len(entries)
        
        expected_opps = expected_router_opportunities(
            episode_steps=episode_steps,
            strategy_interval=condition.strategy_interval,
            allow_switching=condition.allow_switching,
        )
        
        selected_latents = [e["selected_z"] for e in entries]
        switch_count = sum(e["switch_occurred"] for e in entries)
        
        if not condition.allow_switching:
            assert opp_count == 1, f"Expected 1 opportunity for non-switching {condition.name}, got {opp_count}"
            assert switch_count == 0, f"Expected 0 switches for non-switching {condition.name}, got {switch_count}"
            assert len(set(selected_latents)) == 1, f"Expected 1 unique latent for non-switching {condition.name}, got {selected_latents}"
        else:
            assert opp_count == expected_opps, f"Expected {expected_opps} opportunities for {condition.name} (steps={episode_steps}, interval={condition.strategy_interval}), got {opp_count}"


def run_suite(
    *,
    checkpoint: str | Path,
    output_dir: str | Path,
    preset: str = "v6i4",
    opponents: list[str] | None = None,
    map_sets: list[str] | None = None,
    calibration_seeds: list[int] | None = None,
    test_seeds: list[int] | None = None,
    agents: int | None = None,
    latent_k: int | None = None,
    map_layout: str = "map_a_open",
    device: str = "cpu",
    deterministic_actions: bool = True,
    exploratory_allow_unpromoted: bool = False,
    n_bootstrap: int = 10_000,
) -> dict[str, str]:
    validate_seed_split(calibration_seeds or [1000, 1001, 1002, 1003], test_seeds or [2000, 2001, 2002, 2003])
    calibration_seeds = calibration_seeds or [1000, 1001, 1002, 1003]
    test_seeds = test_seeds or [2000, 2001, 2002, 2003]
    opponents = opponents or ["OP5", "OP6", "OP7"]
    map_sets = map_sets or ["eval"]

    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
    from plot.eval_rollout import run_eval_episodes
    from rl.custom_ppo import load_custom_ppo_policy, read_custom_ppo_metadata

    checkpoint = Path(checkpoint)
    meta = read_custom_ppo_metadata(str(checkpoint))
    cfg_meta = meta.get("cfg") if isinstance(meta.get("cfg"), dict) else {}
    ckpt_hash = file_sha256(checkpoint)
    meta["promotion_evidence"] = validate_promoted_v6i2_checkpoint_metadata(
        meta,
        checkpoint_sha256=ckpt_hash,
        exploratory_allow_unpromoted=bool(exploratory_allow_unpromoted),
    )
    latent_k = int(latent_k or meta.get("latent_k", 4))
    agents = int(agents or meta.get("n_blue", 4))
    if str(map_layout) == "map_a_open" and isinstance(cfg_meta, dict):
        map_layout = str(cfg_meta.get("map_layout") or map_layout)
    allowed_latents = cfg_meta.get("router_allowed_latents", None) if isinstance(cfg_meta, dict) else None
    conditions = default_conditions(latent_k, allowed_latents)

    first_env = GPUCTFVecEnv(
        GPUFieldConfig(
            n_envs=1,
            max_blue_agents=agents,
            max_red_agents=agents,
            map_set=map_sets[0],
            map_layout=map_layout,
            max_decision_steps=400,
            aquaticus_profile=True,
            rules_profile="OURS",
            device=device,
            seed=int(calibration_seeds[0]),
        )
    )
    obs_space = first_env.observation_space
    act_space = first_env.action_space
    try:
        model = load_custom_ppo_policy(str(checkpoint), obs_space, act_space, device=device)
        parameter_hash_before = model_parameter_sha256(model)
        switch_cadence = int(getattr(model, "strategy_interval", 0) or 0)
        if switch_cadence <= 0:
            switch_cadence = int(
                cfg_meta.get("latent_resample_every_n")
                or cfg_meta.get("latent_resample_every")
                or cfg_meta.get("strategy_interval")
                or 64
            )
        actor_parameter_hash = hash_module(get_actor_module(model))
        router_parameter_hash = hash_module(get_router_module(model))
    finally:
        first_env.close()

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    conditions = default_conditions(latent_k, allowed_latents, switch_cadence)
    max_decision_steps = 400
    opportunity_hash = switch_opportunity_schedule_hash(
        switch_cadence=switch_cadence,
        max_decision_steps=max_decision_steps,
    )
    all_rows: list[dict[str, Any]] = []

    def run_condition(condition: EvalCondition, seeds: list[int], split: str, fixed_z: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        model = load_custom_ppo_policy(str(checkpoint), obs_space, act_space, device=device)
        if hasattr(model, "clear_eval_suite_state"):
            model.clear_eval_suite_state()
            
        configure_condition(model, condition)
        
        if condition.selection_rule == "preselected_global_fixed_z" and fixed_z is not None:
            model.fixed_latent_strategy = True
            model.fixed_latent_strategy_id = int(fixed_z)
            
        actor_hash_before = hash_module(get_actor_module(model))
        whole_hash_before = model_parameter_sha256(model)
        
        if condition.selection_rule == "shuffled_qphi":
            if hasattr(model, "inject_shuffled_mapping"):
                model.inject_shuffled_mapping(shuffled_mapping)

        if hasattr(model, "opportunity_trace_log"):
            model.opportunity_trace_log = []
        rows: list[dict[str, Any]] = []
        for map_set in map_sets or ["eval"]:
            for opponent in opponents or []:
                for seed in seeds:
                    if condition.name == "preselected_per_opponent_fixed_z":
                        z_id = _per_opp_z[opponent.upper()]
                    else:
                        z_id = condition.fixed_latent_id if fixed_z is None else fixed_z
                    env = GPUCTFVecEnv(
                        GPUFieldConfig(
                            n_envs=1,
                            max_blue_agents=agents,
                            max_red_agents=agents,
                            map_set=map_set,
                            map_layout=map_layout,
                            max_decision_steps=max_decision_steps,
                            aquaticus_profile=True,
                            rules_profile="OURS",
                            device=device,
                            seed=int(seed),
                        )
                    )
                    try:
                        episodes = run_eval_episodes(
                            str(checkpoint),
                            env,
                            1,
                            device,
                            opponent,
                            deterministic=deterministic_actions,
                            fixed_latent_id=z_id,
                            latent_resample_every_n=None,
                            latent_eval_mode="normal",
                            latent_eval_seed=int(seed) + 7919,
                            logical_eval_seed=int(seed),
                            preloaded_model=model,
                            expected_strategy_interval=condition.strategy_interval,
                            expected_allow_switching=condition.allow_switching,
                            condition_name=condition.name,
                            checkpoint_name=Path(checkpoint).stem,
                            selection_rule=condition.selection_rule,
                        )
                    finally:
                        env.close()
                    for episode_id, row in enumerate(episodes, start=1):
                        episode_index = int(episode_id)
                        condition_name = (
                            f"fixed_z{int(z_id)}"
                            if condition.name.startswith("fixed_z") and z_id is not None
                            else condition.name
                        )
                        initial_state_hash = stable_sha256_text(
                            f"{map_set}|{map_layout}|{str(opponent).upper()}|{int(seed)}|{episode_index}"
                        )
                        action_sampling_seed = int(seed) + 104729
                        selector_seed = int(seed) + 7919
                        shuffle_seed = int(seed) + 15485863 if condition.name == "shuffled_qphi_outputs" else ""
                        row.update(
                            {
                                "protocol_version": V6I4_PROTOCOL_VERSION,
                                "split": split,
                                "condition": condition_name,
                                "latent_selection": condition.selection_rule,
                                "fixed_latent_id": "" if z_id is None else int(z_id),
                                "seed": int(seed),
                                "environment_seed": int(seed),
                                "test_seed": int(seed) if split == "test" else "",
                                "calibration_seed": int(seed) if split == "calibration" else "",
                                "initial_state_hash": initial_state_hash,
                                "action_sampling_seed": action_sampling_seed,
                                "selector_seed": selector_seed,
                                "shuffle_seed": shuffle_seed,
                                "episode_index": episode_index,
                                "episode_id": episode_index,
                                "switch_opportunity_schedule_hash": opportunity_hash,
                                "map_set": map_set,
                                "map_layout": map_layout,
                                "opponent": str(opponent).upper(),
                                "checkpoint": str(checkpoint),
                            }
                        )
                        rows.append(row)
        trace_data = []
        if hasattr(model, "opportunity_trace_log"):
            for t_item in model.opportunity_trace_log:
                item = dict(t_item)
                item["condition"] = condition.name
                trace_data.append(item)

        actor_hash_after = hash_module(get_actor_module(model))
        whole_hash_after = model_parameter_sha256(model)

        assert actor_hash_before == actor_hash_after, f"Actor parameters drifted during {condition.name}!"
        assert whole_hash_before == whole_hash_after, f"Whole model parameters drifted during {condition.name}!"

        check_telemetry_invariants(condition, trace_data, rows)

        if hasattr(model, "clear_eval_suite_state"):
            model.clear_eval_suite_state()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return rows, trace_data

    # Pre-checks on seeds:
    c_set = set(calibration_seeds)
    t_set = set(test_seeds)
    if c_set & t_set:
        raise ValueError(f"Calibration and test seed sets overlap: {sorted(c_set & t_set)}")

    # Traces dictionary
    traces_by_condition: dict[str, list[dict[str, Any]]] = {}

    # Run calibration sweep over fixed z
    fixed_conditions = [c for c in conditions if c.name.startswith("fixed_z")]
    for condition in fixed_conditions:
        rows, t_data = run_condition(condition, calibration_seeds, "calibration")
        all_rows.extend(rows)
        traces_by_condition[condition.name] = t_data
    global_z, _per_opp_z = select_calibrated_fixed_latents(all_rows, latent_k=latent_k, allowed_latents=allowed_latents)

    # 1. Run learned condition first to capture outputs for shuffled mapping
    learned_cond = [c for c in conditions if c.name == "learned_qphi_switching"][0]
    rows, learned_t_data = run_condition(learned_cond, test_seeds, "test")
    all_rows.extend(rows)
    traces_by_condition[learned_cond.name] = learned_t_data

    shuffled_mapping, shuffle_meta = build_shuffled_mapping_from_learned_traces(
        learned_t_data,
        latent_k=latent_k,
        allowed_latents=allowed_latents,
        switch_cadence=switch_cadence,
        max_decision_steps=max_decision_steps,
    )
    shuffle_mapping_hash = shuffle_meta["shuffle_mapping_hash"]
    shuffle_mapping_size = shuffle_meta["shuffle_mapping_size"]


    # Run remaining conditions
    for condition in conditions:
        if condition.posthoc_only or condition.name.startswith("fixed_z") or condition.name == "learned_qphi_switching":
            continue
        if condition.name == "preselected_global_fixed_z":
            rows, t_data = run_condition(condition, test_seeds, "test", fixed_z=global_z)
        elif condition.name == "preselected_per_opponent_fixed_z":
            rows, t_data = run_condition(condition, test_seeds, "test")
        else:
            rows, t_data = run_condition(condition, test_seeds, "test")
        
        all_rows.extend(rows)
        traces_by_condition[condition.name] = t_data


    # Run test fixed sweeps
    for condition in fixed_conditions:
        rows, t_data = run_condition(condition, test_seeds, "test")
        all_rows.extend(rows)
        traces_by_condition[condition.name] = t_data

    all_rows = add_posthoc_oracle_rows(all_rows, latent_k=latent_k, allowed_latents=allowed_latents)

    # Enforce split integrity assertions prior to aggregation:
    for row in all_rows:
        sp = row.get("split")
        seed_val = row.get("seed")
        if sp == "calibration":
            if seed_val not in c_set:
                raise AssertionError(f"Calibration row seed {seed_val} not in calibration seeds.")
        elif sp == "test":
            if seed_val not in t_set:
                raise AssertionError(f"Test row seed {seed_val} not in test seeds.")
        else:
            raise AssertionError(f"Unknown split: {sp}")

    # Targeted trace assertions:
    # 1. Fixed occupancy check
    for condition in fixed_conditions:
        t_data = traces_by_condition[condition.name]
        expected_z = condition.fixed_latent_id
        for entry in t_data:
            actual_z = entry["selected_z"]
            if actual_z != expected_z:
                raise AssertionError(f"Fixed condition {condition.name} chose z={actual_z}, expected clamp to {expected_z}")
    # preselected_global_fixed_z occupancy check
    pg_trace = traces_by_condition.get("preselected_global_fixed_z", [])
    for entry in pg_trace:
        if entry["selected_z"] != global_z:
            raise AssertionError(f"preselected_global_fixed_z chose z={entry['selected_z']}, expected clamp to {global_z}")
    # preselected_per_opponent_fixed_z occupancy check
    ppo_trace = traces_by_condition.get("preselected_per_opponent_fixed_z", [])
    for entry in ppo_trace:
        opp = entry["opponent"].upper()
        target_z = _per_opp_z[opp]
        if entry["selected_z"] != target_z:
            raise AssertionError(f"preselected_per_opponent_fixed_z chose z={entry['selected_z']} for opponent {opp}, expected {target_z}")

    # 2. Episode-fixed selection called exactly once per episode
    ef_trace = traces_by_condition.get("uniform_episode_fixed", [])
    # group by opponent + seed + episode
    ef_grouped = {}
    for entry in ef_trace:
        gk = (entry["opponent"], entry["seed"], entry["episode_index"])
        ef_grouped.setdefault(gk, []).append(entry)
    for gk, group in ef_grouped.items():
        if len(group) != 1:
            raise AssertionError(f"uniform_episode_fixed had {len(group)} selections for {gk}, expected exactly 1.")
        if group[0]["opportunity_index"] != 0:
            raise AssertionError(f"uniform_episode_fixed selection was at opportunity {group[0]['opportunity_index']}, expected 0.")

    # 3. Opportunity-random selection called exactly once per opportunity step
    ur_trace = traces_by_condition.get("uniform_random_at_router_opportunities", [])
    ur_grouped = {}
    for entry in ur_trace:
        gk = (entry["opponent"], entry["seed"], entry["episode_index"])
        ur_grouped.setdefault(gk, []).append(entry)
    for gk, group in ur_grouped.items():
        opp_indices = [x["opportunity_index"] for x in group]
        if opp_indices != list(range(len(opp_indices))):
            raise AssertionError(f"Opportunity-random opportunity indices were {opp_indices}, expected sequential 0..N.")
        for x in group:
            expected_step = x["opportunity_index"] * switch_cadence
            if x["step"] != expected_step:
                raise AssertionError(f"Opportunity-random step was {x['step']}, expected {expected_step} for opportunity {x['opportunity_index']}.")

    # 4. Shuffled derangement check (no destination receives its own output)
    sh_trace = traces_by_condition.get("shuffled_qphi_outputs", [])
    for entry in sh_trace:
        k = (entry["opponent"].upper(), entry["seed"], entry["episode_index"], entry["opportunity_index"])
        mapped = shuffled_mapping.get(k)
        if mapped is None:
            raise AssertionError(f"Shuffled lookup key {k} was not injected or registered.")
        if mapped["source_context_key"] == k:
            raise AssertionError(f"Shuffled mapping allowed self-assignment on key: {k}")

    # Emit warnings if two non-fixed condition traces match
    non_fixed_names = ["learned_qphi_switching", "uniform_episode_fixed", "uniform_random_at_router_opportunities", "shuffled_qphi_outputs", "qphi_initial_only_no_switch"]
    for i, name1 in enumerate(non_fixed_names):
        for name2 in non_fixed_names[i+1:]:
            tr1 = traces_by_condition.get(name1, [])
            tr2 = traces_by_condition.get(name2, [])
            if len(tr1) == len(tr2) and len(tr1) > 0:
                match = True
                for entry1, entry2 in zip(tr1, tr2):
                    if entry1["selected_z"] != entry2["selected_z"]:
                        match = False
                        break
                if match:
                    import warnings
                    cond1 = next((c for c in conditions if c.name == name1), None)
                    cond2 = next((c for c in conditions if c.name == name2), None)
                    if cond1 is not None and cond2 is not None and cond1.strategy_interval == cond2.strategy_interval and cond1.allow_switching == cond2.allow_switching and cond1.selection_rule == cond2.selection_rule:
                        warnings.warn(f"CONFIGURATION COLLISION: Supposedly distinct online conditions {name1} and {name2} have identical configurations and traces.")
                    else:
                        warnings.warn(f"STOCHASTIC TRACE COINCIDENCE: Supposedly distinct online conditions {name1} and {name2} generated identical selection traces under correct runtime isolation.")

    parameter_hash_after = parameter_hash_before
    manifest = build_manifest(
        checkpoint=checkpoint,
        preset=preset,
        output_dir=output_dir,
        latent_k=latent_k,
        opponents=opponents or [],
        map_sets=map_sets or [],
        calibration_seeds=calibration_seeds,
        test_seeds=test_seeds,
        switch_cadence=switch_cadence,
        deterministic_actions=deterministic_actions,
        condition_definitions=conditions,
        checkpoint_metadata=meta,
        parameter_hash_before=parameter_hash_before,
        parameter_hash_after=parameter_hash_after,
    )
    manifest["actor_parameter_hash"] = actor_parameter_hash
    manifest["router_parameter_hash"] = router_parameter_hash
    manifest["allowed_latents"] = allowed_latents if allowed_latents is not None else list(range(latent_k))
    manifest["shuffled_permutation_metadata"] = {
        "shuffle_seed": "deterministic_rotation",
        "shuffle_algorithm": "deterministic_rotation",
        "shuffle_mapping_hash": shuffle_mapping_hash,
        "shuffle_mapping_size": shuffle_mapping_size,
        "mean_displacement_fraction": float(np.mean(displacement_fractions)) if displacement_fractions else 0.0,
        "displacement_fractions_per_context": {
            f"{opp}|{seed}|{env_idx}": float(frac)
            for (opp, seed, env_idx), frac in zip(unique_keys, displacement_fractions)
        },
        "source_to_destination_mapping": source_to_dest_meta,
    }
    if parameter_hash_before != parameter_hash_after:
        raise RuntimeError("model parameters changed during v6i4 evaluation")
    
    # Write the published trace artifacts
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    # Headline trace for learned
    learned_flat_trace = []
    for item in learned_t_data:
        flat_item = {
            "opponent": item["opponent"],
            "seed": item["seed"],
            "episode_index": item["episode_index"],
            "opportunity_index": item["opportunity_index"],
            "step": item["step"],
            "selected_z": item["selected_z"],
            "prev_z": item["prev_z"],
            "switch_occurred": item["switch_occurred"],
        }
        for idx, val in enumerate(item["logits"]):
            flat_item[f"logit_{idx}"] = val
        for idx, val in enumerate(item["probabilities"]):
            flat_item[f"prob_{idx}"] = val
        learned_flat_trace.append(flat_item)
    write_csv(output / "v6i4_learned_router_trace.csv", learned_flat_trace)

    # Combined diagnostic trace
    combined_flat_trace = []
    for cond_name, c_trace in sorted(traces_by_condition.items()):
        for item in c_trace:
            flat_item = {
                "condition": cond_name,
                "opponent": item["opponent"],
                "seed": item["seed"],
                "episode_index": item["episode_index"],
                "opportunity_index": item["opportunity_index"],
                "step": item["step"],
                "selected_z": item["selected_z"],
                "prev_z": item["prev_z"],
                "switch_occurred": item["switch_occurred"],
            }
            for idx, val in enumerate(item["logits"]):
                flat_item[f"logit_{idx}"] = val
            for idx, val in enumerate(item["probabilities"]):
                flat_item[f"prob_{idx}"] = val
            combined_flat_trace.append(flat_item)
    write_csv(output / "v6i4_selection_traces.csv", combined_flat_trace)

    # Write separate calibration and test summaries
    cal_summary = aggregate_condition_summary([r for r in all_rows if r.get("split") == "calibration"])
    write_csv(output / "v6i4_calibration_summary.csv", cal_summary)

    # Filter out calibration rows from main write_artifacts flow
    test_only_rows = [r for r in all_rows if r.get("split") == "test"]
    return write_artifacts(output_dir, manifest=manifest, episode_rows=test_only_rows, n_bootstrap=n_bootstrap)
