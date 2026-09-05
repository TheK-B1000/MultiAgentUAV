"""R1 learned-repertoire training under the frozen V3 benchmark.

This is a non-latent DIAGNOSTIC gate. It inherits the exact G0-V5 PPO
configuration and changes only the four PI-mandated axes recorded in
REPERTOIRE_LADDER_FROZEN.json: M1, opponent distribution, terminal budget,
and canonical seed. The live opponent overlay is installed and asserted by
the production orchestration seam before identity, artifacts, trainer
construction, or the first rollout step.

Launch all three sequentially on one GPU:

    python experiments/run_r1_repertoire_training.py --policy ALL
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.opponent_spec import (  # noqa: E402
    assert_live_opponent_batch,
    install_keyed_opponent_overlays,
    pole_A_genome,
    pole_B_genome,
)
from experiments.run_g0_v5_long import build_config as build_g0_v5_config  # noqa: E402
from rl.curriculum import phase_from_tag  # noqa: E402
from rl.training.orchestrator import orchestrate_training_run  # noqa: E402

R1_ROOT = PROJECT_ROOT / "artifacts" / "strategic_demand" / "r1_training"
LADDER = PROJECT_ROOT / "artifacts" / "strategic_demand" / "REPERTOIRE_LADDER_FROZEN.json"
AUDITS = PROJECT_ROOT / "artifacts" / "strategic_demand" / "R_LADDER_BLOCK_AUDITS.json"
G0_V5_CANONICAL_PARENT_SEED = 3_200_001

POLICIES: dict[str, dict[str, Any]] = {
    "A": {
        "label": "pi_A_specialist",
        "seed": 7_100_001,
        "steps": 1_000_000,
        "pool": ("OP6",),
        "weights": (),
    },
    "B": {
        "label": "pi_B_specialist",
        "seed": 7_200_001,
        "steps": 1_000_000,
        "pool": ("OP7",),
        "weights": (),
    },
    "G": {
        "label": "pi_G_generalist",
        "seed": 7_000_001,
        "steps": 2_000_000,
        "pool": ("OP6", "OP7"),
        "weights": (0.5, 0.5),
    },
}

# A resolved field-by-field diff against build_g0_v5_config(seed) must contain
# nothing else. Paths and run_tag are provenance, not scientific changes.
ALLOWED_CONFIG_DIFFS = {
    "checkpoint_dir",
    "episode_csv_path",
    "fixed_opponent_tag",
    "metrics_csv_path",
    "opponent_pool",
    "opponent_pool_weights",
    "own_flag_home_required_to_score",
    "run_tag",
    "seed",
    "total_timesteps",
}


def _stable_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _config_diff(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    return {
        key: {"g0_v5": parent.get(key), "r1": child.get(key)}
        for key in sorted(set(parent) | set(child))
        if parent.get(key) != child.get(key)
    }


def build_r1_config(policy: str):
    policy = str(policy).upper()
    if policy not in POLICIES:
        raise ValueError(f"Unknown R1 policy {policy!r}; expected one of {sorted(POLICIES)}")
    spec = POLICIES[policy]
    cfg = build_g0_v5_config(G0_V5_CANONICAL_PARENT_SEED)
    parent = dataclasses.asdict(cfg)

    run_tag = f"r1_{spec['label']}_seed{int(spec['seed'])}"
    art = R1_ROOT / run_tag
    cfg.run_tag = run_tag
    cfg.seed = int(spec["seed"])
    cfg.total_timesteps = int(spec["steps"])
    cfg.own_flag_home_required_to_score = True
    cfg.fixed_opponent_tag = str(spec["pool"][0])
    cfg.opponent_pool = tuple(spec["pool"])
    cfg.opponent_pool_weights = tuple(spec["weights"])
    cfg.checkpoint_dir = str(art / "ckpts")
    cfg.metrics_csv_path = str(art / "metrics.csv")
    cfg.episode_csv_path = str(art / "episode_rows.csv")

    child = dataclasses.asdict(cfg)
    diff = _config_diff(parent, child)
    unexpected = sorted(set(diff) - ALLOWED_CONFIG_DIFFS)
    if unexpected:
        raise RuntimeError(
            "R1 config drifted from exact G0-V5 outside the frozen axes: "
            + ", ".join(unexpected)
        )
    required = {
        "own_flag_home_required_to_score",
        "opponent_pool",
        "fixed_opponent_tag",
        "run_tag",
        "checkpoint_dir",
        "metrics_csv_path",
        "episode_csv_path",
        "seed",
    }
    if int(spec["steps"]) != 1_000_000:
        required.add("total_timesteps")
    if policy == "G":
        required.add("opponent_pool_weights")
    missing = sorted(required - set(diff))
    if missing:
        raise RuntimeError(f"R1 expected config deltas are missing: {missing}")
    return cfg, {
        "parent": "experiments.run_g0_v5_long.build_config",
        "parent_resolved_config_sha256": _stable_hash(parent),
        "resolved_config_diff": diff,
        "allowed_diff_fields": sorted(ALLOWED_CONFIG_DIFFS),
    }


def configure_r1_live_environment(
    env,
    cfg,
    *,
    policy: str,
    config_contract: dict[str, Any],
    expected_steps: int | None = None,
) -> dict[str, Any]:
    """Authoritative R0 seam: clear, set, overlay, assert, manifest.

    ``expected_steps`` lets a CONTINUATION declare its own budget while keeping
    the drift guard active. It defaults to R1's frozen budget, so R1 runs are
    unchanged. A continuation passes its cumulative target explicitly rather
    than the guard being bypassed -- the check still fires on any budget the
    caller did not intend.
    """
    policy = str(policy).upper()
    spec = POLICIES[policy]
    want_steps = int(spec["steps"] if expected_steps is None else expected_steps)
    if int(cfg.seed) != int(spec["seed"]) or int(cfg.total_timesteps) != want_steps:
        raise RuntimeError(
            f"R1 {policy} seed/budget drift: resolved seed={cfg.seed}, "
            f"steps={cfg.total_timesteps}, expected={spec['seed']}/{want_steps}"
        )
    core = env.core
    if not bool(core.cfg.own_flag_home_required_to_score):
        raise RuntimeError("R1 requested M1 but the live core has it disabled")
    if core.cfg.ruleset_id != "RULESET_V3_M1_OWN_FLAG_HOME":
        raise RuntimeError(
            f"R1 live ruleset identity is {core.cfg.ruleset_id!r}, expected M1"
        )

    core._bt_profile_override = None
    core._sds_opening_hold_steps = 0
    # SIZE_NORMALIZED_POLE_SEMANTICS_SPEC.json: the poles' defender gate is an absolute
    # alive-count, so it must be resolved at the LIVE team size or a scaled run would train
    # against the 2v2 pole while playing N agents. A NO-OP at 2v2 by construction:
    # pole_A_genome(2) reproduces the frozen record and pole_B_genome(2) adds no overlay,
    # so no OP7 key is installed and the certified 2v2 path is byte-for-byte unchanged.
    _n_agents = int(getattr(cfg, "max_blue_agents", 2))
    genomes = {"OP6": pole_A_genome(_n_agents)} if policy in {"A", "G"} else {}
    if _n_agents != 2:
        genomes["OP7"] = pole_B_genome(_n_agents)
    install_keyed_opponent_overlays(core, genomes)

    # The generalist batch is exactly balanced and STATIC: each env keeps its
    # assigned pole for the whole run. Measured, not assumed -- across 75
    # env-terminations in a direct probe the composition never changed, so the
    # inherited opponent-pool sampler does not re-sample here.
    #
    # This is sound for R1: every gradient batch is exactly 50/50 A/B (no
    # sampling noise), and the policy is feedforward and cannot observe env
    # index, so a static split is equivalent to a shuffled order. It also means
    # the A<->B switch never occurs during pi_G, so there is no switch path for
    # a stale overlay to contaminate. The keyed override is still used because
    # it resolves per env from the live opponent key.
    #
    # Covered by tests/test_r1_training_contract.py::
    #   test_generalist_switch_path_stays_clean_across_resampling
    if policy == "G":
        if int(core.B) % 2:
            raise RuntimeError(f"R1 generalist requires an even n_envs; got {int(core.B)}")
        initial_keys = ["OP6" if i % 2 == 0 else "OP7" for i in range(int(core.B))]
    else:
        initial_keys = [str(spec["pool"][0])] * int(core.B)
    for env_i, key in enumerate(initial_keys):
        env.env_method("set_phase", phase_from_tag(key), indices=[env_i])
        env.env_method("set_next_opponent", "SCRIPTED", key, indices=[env_i])
    env.reset()
    rows = assert_live_opponent_batch(
        core,
        genomes,
        allowed_keys=tuple(spec["pool"]),
        context=f"R1 {policy} production construction",
    )
    counts = {key: initial_keys.count(key) for key in sorted(set(initial_keys))}
    if policy == "G" and counts != {"OP6": int(core.B) // 2, "OP7": int(core.B) // 2}:
        raise RuntimeError(f"R1 generalist initial batch is not balanced: {counts}")

    return {
        "r1_protocol": {
            "classification": "DIAGNOSTIC (non-latent learned-repertoire gate)",
            "scientific_delta": (
                "Train G0-V5 PPO from scratch under frozen M1 against pole A, "
                "pole B, or the balanced A/B distribution; no latent strategy."
            ),
            "policy": policy,
            "policy_label": spec["label"],
            "seed": int(spec["seed"]),
            "terminal_budget_steps": int(spec["steps"]),
            "scored_checkpoint": f"final_{cfg.run_tag}.zip",
            "checkpoint_rule": (
                "Only the terminal-budget checkpoint is scored. Periodic "
                "checkpoints are diagnostics and may not be cherry-picked."
            ),
            "opponent_distribution": {
                "pool": list(spec["pool"]),
                "weights": list(spec["weights"] or ((1.0,) if len(spec["pool"]) == 1 else ())),
                "opponent_identity_visible_to_policy": False,
            },
            "initial_live_batch_counts": counts,
            "resolved_opponent_rows": rows,
            "overlay_authority": "core._bt_resolved_profile_tensors()",
            "ruleset_id": core.cfg.ruleset_id,
            "own_flag_home_required_to_score": True,
            "benchmark": str(LADDER.relative_to(PROJECT_ROOT)),
            "seed_audits": str(AUDITS.relative_to(PROJECT_ROOT)),
            **config_contract,
        }
    }


def run_policy(policy: str) -> None:
    cfg, contract = build_r1_config(policy)
    print(json.dumps({
        "r1_policy": policy,
        "run_tag": cfg.run_tag,
        "seed": cfg.seed,
        "total_timesteps": cfg.total_timesteps,
        "config_contract": contract,
    }, indent=2, default=str))
    orchestrate_training_run(
        cfg,
        pre_rollout_env_setup=partial(
            configure_r1_live_environment,
            policy=policy,
            config_contract=contract,
        ),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", choices=("A", "B", "G", "ALL"), required=True)
    ap.add_argument("--contract-only", action="store_true")
    args = ap.parse_args()
    order = ("A", "B", "G") if args.policy == "ALL" else (args.policy,)
    if args.contract_only:
        for policy in order:
            cfg, contract = build_r1_config(policy)
            print(json.dumps({
                "policy": policy,
                "run_tag": cfg.run_tag,
                "seed": cfg.seed,
                "total_timesteps": cfg.total_timesteps,
                **contract,
            }, indent=2, default=str))
        return 0
    for policy in order:
        run_policy(policy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
