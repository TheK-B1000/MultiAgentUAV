"""v6i4 router-ablation evaluator for frozen v6i2 checkpoints.

v6i4 is a Summer-plan-faithful, evaluation-only router-ablation protocol over
a frozen, Phase-A-promoted v6i2 checkpoint. It never trains or updates model
parameters.
"""

from __future__ import annotations

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


@dataclass(frozen=True)
class RouterCondition:
    name: str
    latent_selection: str
    description: str
    fixed_latent_id: int | None = None
    latent_eval_mode: str = "normal"
    latent_resample_every: int | None = None
    online_rollout: bool = True
    identity_assisted: bool = False
    posthoc_only: bool = False


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


def default_conditions(latent_k: int) -> list[RouterCondition]:
    conditions = [
        RouterCondition(
            name="learned_qphi_switching",
            latent_selection="learned_qphi_switching",
            description="Actual trained q_phi at every switch opportunity.",
        ),
        RouterCondition(
            name="uniform_episode_fixed",
            latent_selection="uniform_episode_fixed",
            latent_eval_mode="uniform_random",
            latent_resample_every=0,
            description="Uniform z sampled once per episode and then held fixed.",
        ),
        RouterCondition(
            name="uniform_random_at_router_opportunities",
            latent_selection="uniform_random_at_router_opportunities",
            latent_eval_mode="uniform_random",
            description="Uniform z sampled from an isolated selector RNG at the same deterministic router opportunities.",
        ),
        RouterCondition(
            name="qphi_initial_only_no_switch",
            latent_selection="qphi_initial_only_no_switch",
            latent_resample_every=0,
            description="q_phi selects at episode start only; later opportunities are ignored.",
        ),
        RouterCondition(
            name="shuffled_qphi_outputs",
            latent_selection="shuffled_qphi_outputs",
            latent_eval_mode="shuffled",
            description=(
                "Primary shuffled control: preserve deterministic opportunity times and the "
                "q_phi output source distribution, but break context alignment."
            ),
        ),
    ]
    for z in range(int(latent_k)):
        conditions.append(
            RouterCondition(
                name=f"fixed_z{z}",
                latent_selection="fixed",
                fixed_latent_id=z,
                description=f"Clamp all decisions to z={z}.",
            )
        )
    conditions.extend(
        [
            RouterCondition(
                name="preselected_global_fixed_z",
                latent_selection="fixed",
                description="Deployable global fixed-z baseline chosen on calibration seeds.",
            ),
            RouterCondition(
                name="preselected_per_opponent_fixed_z",
                latent_selection="fixed",
                description=(
                    "Identity-assisted per-opponent fixed-z baseline chosen on calibration seeds; "
                    "valid only when opponent identity is explicitly available to the evaluation policy."
                ),
                identity_assisted=True,
            ),
            RouterCondition(
                name="posthoc_global_fixed_oracle",
                latent_selection="posthoc",
                description="Posthoc best global fixed-z on evaluation seeds; non-deployable upper bound.",
                posthoc_only=True,
                online_rollout=False,
            ),
            RouterCondition(
                name="posthoc_opponent_oracle",
                latent_selection="posthoc",
                description="Posthoc best fixed-z per opponent on evaluation seeds; non-deployable upper bound.",
                identity_assisted=True,
                posthoc_only=True,
                online_rollout=False,
            ),
            RouterCondition(
                name="posthoc_episode_oracle",
                latent_selection="posthoc",
                description="Posthoc best fixed-z per matched episode; measures headroom only.",
                posthoc_only=True,
                online_rollout=False,
            ),
        ]
    )
    return conditions


def select_calibrated_fixed_latents(
    calibration_rows: list[dict[str, Any]],
    *,
    latent_k: int,
) -> tuple[int, dict[str, int]]:
    by_z: dict[int, list[float]] = {z: [] for z in range(int(latent_k))}
    by_opp_z: dict[tuple[str, int], list[float]] = {}
    for row in calibration_rows:
        if str(row.get("split")) != "calibration":
            continue
        if str(row.get("condition")) != f"fixed_z{row.get('fixed_latent_id')}":
            continue
        z = int(row["fixed_latent_id"])
        ret = float(row.get("return", 0.0))
        opp = str(row.get("opponent", "")).upper()
        by_z.setdefault(z, []).append(ret)
        by_opp_z.setdefault((opp, z), []).append(ret)
    global_z = max(range(int(latent_k)), key=lambda z: float(np.mean(by_z.get(z) or [-math.inf])))
    opponents = sorted({opp for opp, _z in by_opp_z})
    per_opp: dict[str, int] = {}
    for opp in opponents:
        per_opp[opp] = max(
            range(int(latent_k)),
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


def add_posthoc_oracle_rows(rows: list[dict[str, Any]], *, latent_k: int) -> list[dict[str, Any]]:
    """Derive non-deployable oracle rows from test split fixed-z sweeps."""
    test_fixed = [
        r
        for r in rows
        if str(r.get("split")) == "test"
        and str(r.get("condition", "")).startswith("fixed_z")
        and str(r.get("fixed_latent_id", "")) != ""
    ]
    if not test_fixed:
        return rows

    by_z: dict[int, list[dict[str, Any]]] = {z: [] for z in range(int(latent_k))}
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

    global_z = max(range(int(latent_k)), key=lambda z: mean_return(by_z.get(z, [])))
    opp_best: dict[str, int] = {}
    for opp in sorted({opp for opp, _z in by_opp_z}):
        opp_best[opp] = max(
            range(int(latent_k)),
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
    conditions = default_conditions(latent_k)

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
    try:
        model = load_custom_ppo_policy(str(checkpoint), first_env.observation_space, first_env.action_space, device=device)
        if hasattr(model, "clear_eval_suite_state"):
            model.clear_eval_suite_state()
    finally:
        first_env.close()

    parameter_hash_before = model_parameter_sha256(model)
    switch_cadence = int(getattr(model, "strategy_interval", 0) or 0)
    max_decision_steps = 400
    opportunity_hash = switch_opportunity_schedule_hash(
        switch_cadence=switch_cadence,
        max_decision_steps=max_decision_steps,
    )
    all_rows: list[dict[str, Any]] = []

    def run_condition(condition: RouterCondition, seeds: list[int], split: str, fixed_z: int | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
                            latent_resample_every_n=condition.latent_resample_every,
                            latent_eval_mode=condition.latent_eval_mode,
                            latent_eval_seed=int(seed) + 7919,
                            preloaded_model=model,
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
                                "latent_selection": condition.latent_selection,
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
    global_z, _per_opp_z = select_calibrated_fixed_latents(all_rows, latent_k=latent_k)

    # 1. Run learned condition first to capture outputs for shuffled mapping
    learned_cond = [c for c in conditions if c.name == "learned_qphi_switching"][0]
    rows, learned_t_data = run_condition(learned_cond, test_seeds, "test")
    all_rows.extend(rows)
    traces_by_condition[learned_cond.name] = learned_t_data

    # Build the shuffled mapping using a deterministic rotation derangement
    # Key format: (opponent, environment_seed, episode_index, opportunity_index)
    sorted_contexts = sorted({
        (str(t_item["opponent"]).upper(), int(t_item["seed"]), int(t_item["episode_index"]), int(t_item["opportunity_index"]))
        for t_item in learned_t_data
    })
    
    if len(sorted_contexts) < 2:
        raise ValueError(f"Shuffled control requires at least 2 contexts, but only found {len(sorted_contexts)}.")

    rotated_contexts = sorted_contexts[1:] + sorted_contexts[:1]
    
    # Store source key to learned output
    learned_outputs_map = {}
    for t_item in learned_t_data:
        k = (str(t_item["opponent"]).upper(), int(t_item["seed"]), int(t_item["episode_index"]), int(t_item["opportunity_index"]))
        learned_outputs_map[k] = {
            "source_context_key": k,
            "logits": list(t_item["logits"]),
            "probabilities": list(t_item["probabilities"]),
            "selected_z": int(t_item["selected_z"]),
        }

    # Build shuffled mapping dict: lookup key -> permuted whole output
    shuffled_mapping = {}
    source_to_dest_meta = []
    for src, dst in zip(sorted_contexts, rotated_contexts):
        shuffled_mapping[src] = learned_outputs_map[dst]
        source_to_dest_meta.append({
            "source": [src[0], src[1], src[2], src[3]],
            "destination": [dst[0], dst[1], dst[2], dst[3]]
        })

    mapping_payload = json.dumps(source_to_dest_meta, sort_keys=True)
    shuffle_mapping_hash = stable_sha256_text(mapping_payload)
    shuffle_mapping_size = len(sorted_contexts)

    # Inject mapping to the model
    if hasattr(model, "inject_shuffled_mapping"):
        model.inject_shuffled_mapping(shuffled_mapping)

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

    # Shuffled is now run, clear policy mapping afterward
    if hasattr(model, "clear_eval_suite_state"):
        model.clear_eval_suite_state()

    # Run test fixed sweeps
    for condition in fixed_conditions:
        rows, t_data = run_condition(condition, test_seeds, "test")
        all_rows.extend(rows)
        traces_by_condition[condition.name] = t_data

    all_rows = add_posthoc_oracle_rows(all_rows, latent_k=latent_k)

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
                    warnings.warn(f"Telemetry warning: Supposedly distinct online conditions {name1} and {name2} generated identical selection traces.")

    parameter_hash_after = model_parameter_sha256(model)
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
    manifest["shuffled_permutation_metadata"] = {
        "shuffle_seed": "deterministic_rotation",
        "shuffle_algorithm": "deterministic_rotation",
        "shuffle_mapping_hash": shuffle_mapping_hash,
        "shuffle_mapping_size": shuffle_mapping_size,
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
