"""Probe the 6 opponent-behavior summary features fed to q_phi.

Per-opponent feature distributions (mean + std) are computed by running the same
vectorized env / opponent-param pipeline the trainer uses, then reading the
global state through ``env.state()`` (which calls ``build_global_state_batch``).
This guarantees we measure exactly what q_phi sees -- no duplicate implementation.

Two modes:
* random blue (default): uses random ``action_space.sample()`` actions. Fast worst-case probe.
* trained blue (``--checkpoint``): loads a custom-PPO checkpoint and uses
  ``model.predict`` for blue actions. Reveals separation under coherent play.
  In trained mode envs is forced to 1 (matches the per-step single-obs inference contract).

Verdict heuristic:
* ``clear``  : max pairwise |Δ| > 0.15 AND OP5 vs OP6 |Δ| > 0.10 in >=2 features.
* ``weak``   : any pairwise |Δ| > 0.05 but does not meet ``clear``.
* ``none``   : all pairwise |Δ| <= 0.05.
* ``bug``    : any feature is constant (std<1e-4 across all opponents),
               all zero, or NaN anywhere.

Usage:
    python tools/red_behavior_feature_probe.py
    python tools/red_behavior_feature_probe.py --steps 500 --envs 8 --device cpu
    python tools/red_behavior_feature_probe.py \
        --checkpoint checkpoints/4v4/final_latent_sharp3_smoke_4v4_seed1_4v4.zip \
        --steps 2000
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig  # noqa: E402
from opponent_params import sample_batched_opponent_params  # noqa: E402
from rl.curriculum import phase_from_tag  # noqa: E402
from rl.global_state import GLOBAL_STATE_FIELD_NAMES, GLOBAL_STATE_DIM  # noqa: E402
from rl.stress_schedule import STRESS_BY_PHASE  # noqa: E402

OPPONENTS: Tuple[str, ...] = ("OP3", "OP5", "OP6", "OP7")

# Index range for the 6 new behavior-summary features (sharp3 append-only).
BEHAVIOR_FEATURE_NAMES: Tuple[str, ...] = (
    "red_attacker_fraction_recent",
    "red_role_switch_rate_recent",
    "red_mean_speed_recent",
    "red_midline_pressure_recent",
    "red_home_defender_fraction_recent",
    "red_min_to_blue_flag_window_min",
)


def _resolve_feature_indices() -> List[int]:
    indices = []
    for name in BEHAVIOR_FEATURE_NAMES:
        if name not in GLOBAL_STATE_FIELD_NAMES:
            raise RuntimeError(
                f"feature {name!r} not found in GLOBAL_STATE_FIELD_NAMES "
                f"(saw {GLOBAL_STATE_FIELD_NAMES!r})"
            )
        indices.append(GLOBAL_STATE_FIELD_NAMES.index(name))
    return indices


def _apply_opponent(env: GPUCTFVecEnv, gpu_cfg: GPUFieldConfig, tag: str) -> None:
    phase = str(phase_from_tag(tag)).upper()
    env.env_method("set_stress_schedule", STRESS_BY_PHASE)
    env.env_method("set_phase", phase)
    env.env_method("set_next_opponent", "SCRIPTED", tag)
    try:
        opp_params = sample_batched_opponent_params(
            kind="SCRIPTED",
            key=tag,
            phase=phase,
            n_agents=gpu_cfg.max_red_agents,
            batch_size=gpu_cfg.n_envs,
            device=gpu_cfg.device,
        )
        dyn_cfg = {
            k: v
            for k, v in opp_params.items()
            if k
            in {
                "deception_prob",
                "speed_mult",
                "attacker_style",
                "defender_style",
                "role_switch_prob",
                "coordinated_attack",
                "attack_sync_window",
            }
        }
        if dyn_cfg:
            env.env_method("set_dynamics_config", dyn_cfg)
    except Exception as exc:
        print(f"[probe] WARNING: opponent_params sampling failed for {tag}: {exc}")


def _random_actions(env: GPUCTFVecEnv, rng: np.random.Generator) -> np.ndarray:
    n_envs = env.num_envs
    sample = env.action_space.sample()
    sample = np.asarray(sample, dtype=np.int64)
    return np.tile(sample[None, :], (n_envs, 1)) if n_envs > 1 else sample[None, :]


def _trained_actions(model: Any, env: GPUCTFVecEnv, deterministic: bool) -> np.ndarray:
    """Pull one timestep's actions from a custom-PPO checkpoint (n_envs=1 contract)."""
    obs = env.core.get_obs()
    single = {
        k: v[0] if hasattr(v, "shape") and len(v.shape) > 1 and v.shape[0] == 1 else v
        for k, v in obs.items()
    }
    try:
        single["global_state"] = env.state()[0]
    except Exception:
        pass
    act, _ = model.predict(single, deterministic=deterministic)
    return np.asarray(act, dtype=np.int64)[None, :]


def _collect_for_opponent(
    tag: str,
    gpu_cfg: GPUFieldConfig,
    feature_idx: List[int],
    steps: int,
    seed: int,
    warmup_drop: int = 25,
    model: Optional[Any] = None,
    deterministic: bool = False,
) -> np.ndarray:
    env = GPUCTFVecEnv(gpu_cfg)
    try:
        env.seed(seed)
        _apply_opponent(env, gpu_cfg, tag)
        env.reset()
        if model is not None and hasattr(model, "reset_strategy"):
            model.reset_strategy()
        rng = np.random.default_rng(seed)
        rows: List[np.ndarray] = []
        for step in range(steps):
            if model is None:
                acts = _random_actions(env, rng)
            else:
                acts = _trained_actions(model, env, deterministic)
            env.step_async(acts)
            _, _, dones, _ = env.step_wait()
            if model is not None and bool(np.any(dones)) and hasattr(model, "reset_strategy"):
                model.reset_strategy()
            if step < warmup_drop:
                continue  # let the ring buffer fill before sampling
            gs = env.state()  # (B, 25) numpy
            rows.append(gs[:, feature_idx].copy())
        if not rows:
            raise RuntimeError(f"no rows collected for {tag}")
        return np.concatenate(rows, axis=0)  # (total_samples, 6)
    finally:
        env.close()


def _summarize(per_opp: Dict[str, np.ndarray]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for tag, mat in per_opp.items():
        feat_stats: Dict[str, Dict[str, float]] = {}
        for j, name in enumerate(BEHAVIOR_FEATURE_NAMES):
            col = mat[:, j]
            finite = col[np.isfinite(col)]
            if finite.size == 0:
                feat_stats[name] = {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan"), "n": 0}
                continue
            feat_stats[name] = {
                "mean": float(np.mean(finite)),
                "std": float(np.std(finite)),
                "min": float(np.min(finite)),
                "max": float(np.max(finite)),
                "n": int(finite.size),
            }
        summary[tag] = feat_stats
    return summary


def _pairwise_diff_table(summary: Dict[str, Dict[str, Dict[str, float]]]) -> Dict[Tuple[str, str], Dict[str, float]]:
    pairs: List[Tuple[str, str]] = [
        ("OP3", "OP5"),
        ("OP3", "OP6"),
        ("OP3", "OP7"),
        ("OP5", "OP6"),
        ("OP5", "OP7"),
        ("OP6", "OP7"),
    ]
    out: Dict[Tuple[str, str], Dict[str, float]] = {}
    for a, b in pairs:
        if a not in summary or b not in summary:
            continue
        row: Dict[str, float] = {}
        for name in BEHAVIOR_FEATURE_NAMES:
            ma = summary[a][name]["mean"]
            mb = summary[b][name]["mean"]
            row[name] = float(abs(ma - mb))
        out[(a, b)] = row
    return out


def _verdict(
    summary: Dict[str, Dict[str, Dict[str, float]]],
    diffs: Dict[Tuple[str, str], Dict[str, float]],
) -> Tuple[str, List[str]]:
    notes: List[str] = []

    # Bug checks first.
    for name in BEHAVIOR_FEATURE_NAMES:
        across_means = [summary[op][name]["mean"] for op in OPPONENTS if op in summary]
        across_stds = [summary[op][name]["std"] for op in OPPONENTS if op in summary]
        if any(math.isnan(m) for m in across_means):
            notes.append(f"BUG: {name} produced NaN means")
            return "bug", notes
        rng = max(across_means) - min(across_means) if across_means else 0.0
        avg_std = float(np.mean(across_stds)) if across_stds else 0.0
        if rng < 1e-4 and avg_std < 1e-4:
            notes.append(f"BUG: {name} is constant across opponents (mean range={rng:.2e}, avg std={avg_std:.2e})")
            return "bug", notes
        if all(abs(m) < 1e-6 for m in across_means):
            notes.append(f"BUG: {name} is all zeros across all opponents")
            return "bug", notes

    # Pairwise magnitude tally.
    max_pair = 0.0
    op56_features_above_010 = 0
    if ("OP5", "OP6") in diffs:
        for name, d in diffs[("OP5", "OP6")].items():
            if d > 0.10:
                op56_features_above_010 += 1
    for _, row in diffs.items():
        for _, d in row.items():
            if d > max_pair:
                max_pair = d

    notes.append(f"max pairwise |Δ| across all pairs/features = {max_pair:.4f}")
    notes.append(f"OP5 vs OP6 features with |Δ| > 0.10 = {op56_features_above_010}/{len(BEHAVIOR_FEATURE_NAMES)}")

    if max_pair > 0.15 and op56_features_above_010 >= 2:
        return "clear", notes
    if max_pair > 0.05:
        return "weak", notes
    return "none", notes


def _write_summary_csv(
    path: Path,
    summary: Dict[str, Dict[str, Dict[str, float]]],
    diffs: Dict[Tuple[str, str], Dict[str, float]],
    verdict: str,
    notes: List[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["# verdict", verdict])
        for n in notes:
            writer.writerow(["# note", n])
        writer.writerow([])
        writer.writerow(["section", "opponent_or_pair", "feature", "metric", "value"])
        for tag, feats in summary.items():
            for name, stats in feats.items():
                for metric, val in stats.items():
                    writer.writerow(["per_opponent", tag, name, metric, val])
        for (a, b), row in diffs.items():
            for name, val in row.items():
                writer.writerow(["pairwise_abs_diff", f"{a}_vs_{b}", name, "abs_mean_diff", val])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500, help="sim steps per opponent (after warmup)")
    parser.add_argument("--envs", type=int, default=8, help="parallel envs per opponent (forced to 1 if --checkpoint)")
    parser.add_argument("--agents", type=int, default=4, help="agents per team")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--seed", type=int, default=1, help="RNG seed")
    parser.add_argument("--warmup-drop", type=int, default=25, help="initial sim steps to discard (ring warmup)")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="optional custom-PPO checkpoint .zip; if set, blue plays this trained policy",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="argmax actions when using --checkpoint (default: stochastic)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "logs" / "diagnostics",
        help="output directory for the summary CSV",
    )
    args = parser.parse_args()

    feature_idx = _resolve_feature_indices()
    print(f"[probe] global_state_dim = {GLOBAL_STATE_DIM}")
    print(f"[probe] feature indices  = {feature_idx} -> {BEHAVIOR_FEATURE_NAMES}")

    use_trained = args.checkpoint is not None
    n_envs = 1 if use_trained else int(args.envs)
    if use_trained and int(args.envs) != 1:
        print(f"[probe] --checkpoint set: forcing n_envs=1 (was {int(args.envs)})")

    print(f"[probe] device={args.device}  envs={n_envs}  agents={args.agents}  "
          f"steps={args.steps} (+warmup {args.warmup_drop})  seed={args.seed}")
    if use_trained:
        print(f"[probe] blue policy = {args.checkpoint}  deterministic={bool(args.deterministic)}")
    else:
        print(f"[probe] blue policy = RANDOM action_space.sample()")

    gpu_cfg = GPUFieldConfig(
        n_envs=n_envs,
        n_agents_per_team=int(args.agents),
        map_set="train",
        max_decision_steps=400,
        aquaticus_profile=True,
        rules_profile="OURS",
        device=str(args.device),
        seed=int(args.seed),
    )

    # Load checkpoint once; pass to each per-opponent collection.
    model = None
    if use_trained:
        ckpt = args.checkpoint
        if not ckpt.is_absolute():
            ckpt = ROOT / ckpt
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        # Build a throwaway env just for obs/action spaces.
        tmp_env = GPUCTFVecEnv(gpu_cfg)
        try:
            from rl.custom_ppo import load_custom_ppo_policy
            model = load_custom_ppo_policy(
                str(ckpt),
                tmp_env.observation_space,
                tmp_env.action_space,
                device=str(args.device),
            )
            print(f"[probe] checkpoint loaded: latent_strategy={bool(getattr(getattr(model, 'model', None), 'uses_latent_strategy', False))}")
        finally:
            tmp_env.close()

    per_opp: Dict[str, np.ndarray] = {}
    for tag in OPPONENTS:
        print(f"[probe] collecting {tag} ...")
        mat = _collect_for_opponent(
            tag,
            gpu_cfg,
            feature_idx,
            steps=int(args.steps),
            seed=int(args.seed),
            warmup_drop=int(args.warmup_drop),
            model=model,
            deterministic=bool(args.deterministic),
        )
        per_opp[tag] = mat
        print(f"[probe]   {tag}: {mat.shape[0]} samples")

    summary = _summarize(per_opp)
    diffs = _pairwise_diff_table(summary)
    verdict, notes = _verdict(summary, diffs)

    print("\n========== per-opponent feature statistics ==========")
    header = f"{'feature':36s}  " + "  ".join(f"{op:>11s}" for op in OPPONENTS)
    print(header)
    print("-" * len(header))
    for name in BEHAVIOR_FEATURE_NAMES:
        row = [f"{name:36s}"]
        for op in OPPONENTS:
            stats = summary[op][name]
            row.append(f"{stats['mean']:5.3f}±{stats['std']:4.2f}")
        print("  ".join(row))

    print("\n========== pairwise absolute mean differences ==========")
    pair_keys = list(diffs.keys())
    header = f"{'feature':36s}  " + "  ".join(f"{a}-{b:>3s}" for a, b in pair_keys)
    print(header)
    print("-" * len(header))
    for name in BEHAVIOR_FEATURE_NAMES:
        row = [f"{name:36s}"]
        for key in pair_keys:
            row.append(f"{diffs[key][name]:7.4f}")
        print("  ".join(row))

    print("\n========== verdict ==========")
    print(f"  verdict: {verdict.upper()}")
    for n in notes:
        print(f"   - {n}")

    if verdict == "clear":
        print("\n[probe] Features differentiate opponents at the input level.")
        print("[probe] q_phi failing to route is now a TRAINING-TIME problem, not a feature problem.")
    elif verdict == "weak":
        print("\n[probe] Features separate opponents only weakly.")
        print("[probe] Consider adding more discriminative features or extending the window.")
    elif verdict == "none":
        print("\n[probe] Features do NOT separate opponents.")
        print("[probe] Either the features are too smoothed, or there is a feature-side bug.")
    else:
        print("\n[probe] BUG suspected -- inspect the feature computation in _step.py / global_state.py.")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = "trained" if use_trained else "random"
    out_path = Path(args.out_dir) / f"red_behavior_feature_probe_{tag}_{ts}.csv"
    _write_summary_csv(out_path, summary, diffs, verdict, notes)
    print(f"\n[probe] summary written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
