"""Single-pass matched-seed forced-z episode collection."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from experiments.forced_z_eval.io import CellEpisodes
from experiments.forced_z_eval.protocol import ForcedZProtocol, audit_protocol_note


def _make_env(protocol: ForcedZProtocol, map_name: str, seed: int) -> Any:
    from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig
    from rl.custom_ppo.inference import read_custom_ppo_metadata

    meta = read_custom_ppo_metadata(protocol.checkpoint)
    agents = int(meta.get("n_blue", 2))
    cfg = GPUFieldConfig(
        n_envs=1,
        max_blue_agents=agents,
        max_red_agents=agents,
        map_layout=map_name,
        max_decision_steps=int(protocol.max_decision_steps),
        aquaticus_profile=True,
        rules_profile="OURS",
        device=protocol.device,
        seed=seed,
    )
    return GPUCTFVecEnv(cfg)


def _hard_reset_between_z_blocks(
    env: Any,
    model: Any,
    *,
    opponent: str,
    cell_seed: int,
    fixed_latent_id: int,
) -> None:
    """Reset hidden evaluator / env state before each forced-z block."""
    import random

    if hasattr(model, "reset_strategy"):
        model.reset_strategy()
    if hasattr(model, "fixed_latent_strategy"):
        model.fixed_latent_strategy = True
    if hasattr(model, "fixed_latent_strategy_id"):
        model.fixed_latent_strategy_id = int(fixed_latent_id)

    seed = int(cell_seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(env, "seed"):
        env.seed(seed)
    try:
        env.env_method("set_phase", opponent)
        env.env_method("set_next_opponent", "SCRIPTED", opponent)
    except Exception:
        pass


def run_forced_z_episodes(
    protocol: ForcedZProtocol,
    *,
    env_mode: str = "reuse_block",
    shared_model: Any | None = None,
    latent_order: tuple[int, ...] | None = None,
    quiet: bool = False,
) -> CellEpisodes:
    """Collect forced-z episodes.

    env_mode:
      ``reuse_block`` — one env per (opponent, map); reuse across z (fast path).
      ``fresh_per_z`` — new env per z; reference for equivalence checks.

    When ``shared_model`` is provided, both modes use the same loaded policy
    object so the comparison isolates environment reuse only.
    """
    order = tuple(latent_order if latent_order is not None else protocol.latents)
    if env_mode == "fresh_per_z":
        return _run_fresh_env_per_z(protocol, shared_model=shared_model, latent_order=order, quiet=quiet)
    if env_mode != "reuse_block":
        raise ValueError(f"Unknown env_mode: {env_mode!r}")
    return _run_reuse_env_block(protocol, shared_model=shared_model, latent_order=order, quiet=quiet)


def _run_reuse_env_block(
    protocol: ForcedZProtocol,
    *,
    shared_model: Any | None,
    latent_order: tuple[int, ...],
    quiet: bool,
) -> CellEpisodes:
    from plot.eval_rollout import run_eval_episodes
    from rl.custom_ppo import load_custom_ppo_policy

    if not quiet:
        print(audit_protocol_note())
        print(f"Checkpoint : {protocol.checkpoint}")
        print(f"Episodes   : {protocol.episodes_per_cell} per (opponent, z, map)")
        print(f"Device     : {protocol.device}")
        print(f"Latent order: {list(latent_order)}")
        print()

    cells: CellEpisodes = {}
    for opp_idx, opponent in enumerate(protocol.opponents):
        for map_idx, map_name in enumerate(protocol.maps):
            cell_seed = protocol.cell_seed(opp_idx, map_idx)
            env = _make_env(protocol, map_name, cell_seed)
            try:
                model = shared_model
                if model is None:
                    model = load_custom_ppo_policy(
                        protocol.checkpoint,
                        env.observation_space,
                        env.action_space,
                        device=protocol.device,
                    )
                for z in latent_order:
                    _hard_reset_between_z_blocks(
                        env, model, opponent=opponent, cell_seed=cell_seed, fixed_latent_id=int(z)
                    )
                    try:
                        eps = run_eval_episodes(
                            protocol.checkpoint,
                            env,
                            int(protocol.episodes_per_cell),
                            protocol.device,
                            opponent,
                            fixed_latent_id=int(z),
                            deterministic=bool(protocol.deterministic_actions),
                            latent_eval_seed=cell_seed,
                            preloaded_model=model,
                            collect_behavior_mean=bool(protocol.collect_behavior_mean),
                            progress_every=int(protocol.progress_every),
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"  ERROR {opponent} z={z} {map_name}: {exc}")
                        eps = []
                    cells[(opponent, z, map_name)] = eps
                    if not quiet:
                        if eps:
                            wr = sum(int(e.get("success", 0)) for e in eps) / len(eps)
                            print(f"  [reuse] {opponent} z={z} {map_name}: WR={wr:.1%} ({len(eps)} eps)")
                        else:
                            print(f"  [reuse] {opponent} z={z} {map_name}: WR=nan% (0 eps)")
            finally:
                env.close()
    return cells


def _run_fresh_env_per_z(
    protocol: ForcedZProtocol,
    *,
    shared_model: Any | None,
    latent_order: tuple[int, ...],
    quiet: bool,
) -> CellEpisodes:
    from plot.eval_rollout import run_eval_episodes
    from rl.custom_ppo import load_custom_ppo_policy

    cells: CellEpisodes = {}
    for opp_idx, opponent in enumerate(protocol.opponents):
        for map_idx, map_name in enumerate(protocol.maps):
            cell_seed = protocol.cell_seed(opp_idx, map_idx)
            for z in latent_order:
                env = _make_env(protocol, map_name, cell_seed)
                try:
                    model = shared_model
                    if model is None:
                        model = load_custom_ppo_policy(
                            protocol.checkpoint,
                            env.observation_space,
                            env.action_space,
                            device=protocol.device,
                        )
                    _hard_reset_between_z_blocks(
                        env, model, opponent=opponent, cell_seed=cell_seed, fixed_latent_id=int(z)
                    )
                    try:
                        eps = run_eval_episodes(
                            protocol.checkpoint,
                            env,
                            int(protocol.episodes_per_cell),
                            protocol.device,
                            opponent,
                            fixed_latent_id=int(z),
                            deterministic=bool(protocol.deterministic_actions),
                            latent_eval_seed=cell_seed,
                            preloaded_model=model,
                            collect_behavior_mean=bool(protocol.collect_behavior_mean),
                            progress_every=int(protocol.progress_every),
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(f"  ERROR {opponent} z={z} {map_name}: {exc}")
                        eps = []
                finally:
                    env.close()
                cells[(opponent, z, map_name)] = eps
                if not quiet:
                    if eps:
                        wr = sum(int(e.get("success", 0)) for e in eps) / len(eps)
                        print(f"  [fresh] {opponent} z={z} {map_name}: WR={wr:.1%} ({len(eps)} eps)")
                    else:
                        print(f"  [fresh] {opponent} z={z} {map_name}: WR=nan% (0 eps)")
    return cells


def load_shared_policy(protocol: ForcedZProtocol, *, map_name: str, cell_seed: int) -> Any:
    """Load one policy object for equivalence comparisons."""
    from rl.custom_ppo import load_custom_ppo_policy

    env = _make_env(protocol, map_name, cell_seed)
    try:
        return load_custom_ppo_policy(
            protocol.checkpoint,
            env.observation_space,
            env.action_space,
            device=protocol.device,
        )
    finally:
        env.close()


__all__ = ["load_shared_policy", "run_forced_z_episodes"]
