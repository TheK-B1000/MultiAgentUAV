#!/usr/bin/env python3
"""V6I26 logit control-authority probe for a trained LRO branch.

Compares an init checkpoint to a short forced-z pilot and measures how much
z*-specific parameter movement actually moves action logits on fixed OP9
observations.

Probes:
  1) actual init→trained Δθ replay
  2) scaled-delta sweep (0.5×…10×)
  3) residual-α forward sweep (no training)

Classification: DIAGNOSTIC.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# This script's diagnostic print statements use Unicode (arrows, theta, approx)
# for readability; Windows consoles default to cp1252, which can't encode them
# and crashes the process mid-print (after the JSON report is already written).
# UTF-8 stdout/stderr makes this robust regardless of console codepage.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from experiments.run_v6i24_donor_teacher_kl import _kl_cat  # noqa: E402
from experiments.run_v6i26_lro_oracle_round import _distribution_logits  # noqa: E402
from experiments.v6i26_lro_core import write_json  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LRO z-branch logit control-authority probe")
    p.add_argument(
        "--init-checkpoint",
        default="artifacts/v6i23_population_birth_5u_seed1/final_v6i23_population_birth_5u_seed1_2v2.zip",
    )
    p.add_argument(
        "--trained-checkpoint",
        default="artifacts/v6i26_margin_pilot_5u_seed1/final_v6i26_lro_z0_r1_5u_seed1.zip",
    )
    p.add_argument(
        "--output",
        default="artifacts/v6i26_margin_pilot_5u_seed1/logit_control_authority_probe.json",
    )
    p.add_argument("--branch", type=int, default=0)
    p.add_argument("--opponent", default="OP9_SPLIT_LANE_FEINT")
    p.add_argument("--map", default="map_b_split_lane")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-obs", type=int, default=128)
    p.add_argument("--max-decision-steps", type=int, default=240)
    return p.parse_args()


def _inner_model(pol) -> Any:
    """Unwrap CustomPPOInferencePolicy → SharedActorCentralizedCritic."""
    if hasattr(pol, "model") and getattr(pol, "model", None) is not None:
        return pol.model
    return getattr(pol, "policy", pol)


def _latent_actor(pol) -> Any:
    model = _inner_model(pol)
    la = getattr(model, "latent_actor", None)
    if la is None:
        raise RuntimeError("policy has no latent_actor")
    return la


def _get_logits(pol, obs: dict, z: int) -> torch.Tensor:
    model = _inner_model(pol)
    with torch.no_grad():
        z_t = torch.full(
            (int(obs["grid"].shape[0]),),
            int(z),
            device=obs["grid"].device,
            dtype=torch.long,
        )
        try:
            dist = model.get_distribution(obs, z_idx=z_t)
        except TypeError:
            if hasattr(pol, "fixed_latent_strategy"):
                pol.fixed_latent_strategy = True
                pol.fixed_latent_strategy_id = int(z)
            dist = pol.get_distribution(obs) if hasattr(pol, "get_distribution") else model.get_distribution(obs)
        logits = _distribution_logits(dist)
        if logits is None:
            raise RuntimeError("could not resolve distribution logits")
        return logits.detach()


def _policy_metrics(logits_a: torch.Tensor, logits_b: torch.Tensor) -> dict[str, float]:
    kl = float(_kl_cat(logits_a, logits_b).mean().item())
    d = (logits_a.float() - logits_b.float()).reshape(logits_a.shape[0], -1)
    logit_l2 = float(torch.linalg.vector_norm(d, ord=2, dim=-1).mean().item())
    logit_linf = float(d.abs().amax(dim=-1).mean().item())
    # MultiDiscrete: treat each row as one softmax for argmax disagreement proxy.
    argmax_disagree = float(
        (logits_a.argmax(dim=-1) != logits_b.argmax(dim=-1)).float().mean().item()
    )
    return {
        "mean_kl": kl,
        "mean_logit_l2": logit_l2,
        "mean_logit_linf": logit_linf,
        "argmax_disagree": argmax_disagree,
    }


def _param_groups(la: Any, branch: int) -> dict[str, list[tuple[str, torch.Tensor]]]:
    """Collect named parameters belonging to branch-specific modules."""
    b = int(branch)
    groups: dict[str, list[tuple[str, torch.Tensor]]] = {
        "z_embedding": [],
        "z_branch_trunk": [],
        "z_action_head": [],
        "z_adapter_residual": [],
        "combined_z_pathway": [],
    }
    # Embedding row is shared Parameter; track as a view for delta bookkeeping.
    if getattr(la, "strategy_embedding", None) is not None:
        w = la.strategy_embedding.weight
        groups["z_embedding"].append((f"strategy_embedding.weight[{b}]", w))
    trunks = getattr(la, "latent_branch_trunks", None)
    if trunks is not None:
        for name, p in trunks[b].named_parameters():
            groups["z_branch_trunk"].append((f"latent_branch_trunks.{b}.{name}", p))
    heads = getattr(la, "latent_action_heads", None)
    if heads is not None:
        for name, p in heads[b].named_parameters():
            groups["z_action_head"].append((f"latent_action_heads.{b}.{name}", p))
    adapters = getattr(la, "latent_adapters", None)
    if adapters is not None:
        for name, p in adapters[b].named_parameters():
            groups["z_adapter_residual"].append((f"latent_adapters.{b}.{name}", p))
    for g in ("z_embedding", "z_branch_trunk", "z_action_head", "z_adapter_residual"):
        groups["combined_z_pathway"].extend(groups[g])
    return groups


def _snapshot_group(
    group: list[tuple[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, p in group:
        if name.endswith("]") and "weight[" in name:
            # strategy_embedding.weight[b]
            idx = int(name.rsplit("[", 1)[1].rstrip("]"))
            out[name] = p.detach()[idx].reshape(-1).cpu().clone()
        else:
            out[name] = p.detach().reshape(-1).cpu().clone()
    return out


def _group_delta_norm(
    a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]
) -> float:
    keys = sorted(set(a) & set(b))
    if not keys:
        return 0.0
    diffs = [torch.linalg.vector_norm((b[k] - a[k]).float()).item() for k in keys]
    # Euclidean over concatenated params
    cat = torch.cat([(b[k] - a[k]).float().reshape(-1) for k in keys], dim=0)
    return float(torch.linalg.vector_norm(cat).item())


def _apply_group_delta(
    la: Any,
    group_names_and_params: list[tuple[str, torch.Tensor]],
    init_snap: dict[str, torch.Tensor],
    trained_snap: dict[str, torch.Tensor],
    scale: float,
) -> None:
    """Set params = init + scale * (trained - init) for the group."""
    with torch.no_grad():
        for name, p in group_names_and_params:
            if name not in init_snap or name not in trained_snap:
                continue
            delta = (trained_snap[name] - init_snap[name]).to(device=p.device, dtype=p.dtype)
            if name.endswith("]") and "weight[" in name:
                idx = int(name.rsplit("[", 1)[1].rstrip("]"))
                base = init_snap[name].to(device=p.device, dtype=p.dtype)
                p.data[idx].copy_((base + float(scale) * delta).view_as(p.data[idx]))
            else:
                base = init_snap[name].to(device=p.device, dtype=p.dtype).view_as(p.data)
                p.data.copy_(base + float(scale) * delta.view_as(p.data))


def _restore_group(
    group_names_and_params: list[tuple[str, torch.Tensor]],
    snap: dict[str, torch.Tensor],
) -> None:
    with torch.no_grad():
        for name, p in group_names_and_params:
            if name not in snap:
                continue
            if name.endswith("]") and "weight[" in name:
                idx = int(name.rsplit("[", 1)[1].rstrip("]"))
                p.data[idx].copy_(snap[name].to(device=p.device, dtype=p.dtype).view_as(p.data[idx]))
            else:
                p.data.copy_(snap[name].to(device=p.device, dtype=p.dtype).view_as(p.data))


def _identity_trunk_snapshot(trunk: Any, branch: int) -> dict[str, torch.Tensor]:
    """Snapshot matching sync_latent_branch_trunks_to_identity (W=I, b=0)."""
    out: dict[str, torch.Tensor] = {}
    for name, p in trunk.named_parameters():
        key = f"latent_branch_trunks.{branch}.{name}"
        if name.endswith("weight"):
            eye = torch.eye(p.shape[0], device="cpu", dtype=torch.float32)
            out[key] = eye.reshape(-1).clone()
        else:
            out[key] = torch.zeros(p.numel(), dtype=torch.float32)
    return out


def _baseline_snapshots(
    groups_tr: dict[str, list[tuple[str, torch.Tensor]]],
    groups_init: dict[str, list[tuple[str, torch.Tensor]]],
    la_tr: Any,
    branch: int,
) -> dict[str, dict[str, torch.Tensor]]:
    """Birth baseline: init-ckpt tensors where present; identity for LRO trunks."""
    baselines: dict[str, dict[str, torch.Tensor]] = {}
    for gname in ("z_embedding", "z_branch_trunk", "z_action_head", "z_adapter_residual"):
        group = groups_tr[gname]
        snap_tr = _snapshot_group(group)
        snap_init = _snapshot_group(groups_init.get(gname, []))
        if gname == "z_branch_trunk":
            trunks = getattr(la_tr, "latent_branch_trunks", None)
            if trunks is None:
                baselines[gname] = {}
            else:
                baselines[gname] = _identity_trunk_snapshot(trunks[branch], branch)
            continue
        base = {}
        for name in snap_tr:
            if name in snap_init:
                base[name] = snap_init[name].clone()
            else:
                base[name] = snap_tr[name].clone()
        baselines[gname] = base
    combined: dict[str, torch.Tensor] = {}
    for gname in ("z_embedding", "z_branch_trunk", "z_action_head", "z_adapter_residual"):
        combined.update(baselines[gname])
    baselines["combined_z_pathway"] = combined
    return baselines


def _set_group_to_snapshot(
    group_names_and_params: list[tuple[str, torch.Tensor]],
    snap: dict[str, torch.Tensor],
) -> None:
    _restore_group(group_names_and_params, snap)


def main() -> int:
    args = _parse_args()
    from experiments.run_v6i24_population_eval_gates import (
        _collect_shared_history,
        _load_policies,
        _make_env,
        _obs_batch,
    )

    init_ckpt = Path(args.init_checkpoint)
    trained_ckpt = Path(args.trained_checkpoint)
    if not init_ckpt.is_file() or not trained_ckpt.is_file():
        print("ERROR: missing checkpoint(s)")
        return 2

    device = str(args.device)
    branch = int(args.branch)
    env0 = _make_env(init_ckpt, args.map, int(args.seed), device, int(args.max_decision_steps))
    try:
        init_pol = _load_policies(
            [(0, "init", init_ckpt)], env0.observation_space, env0.action_space, device
        )[0]["policy"]
        trained_pol = _load_policies(
            [(1, "trained", trained_ckpt)],
            env0.observation_space,
            env0.action_space,
            device,
        )[0]["policy"]
    finally:
        env0.close()

    for pol in (init_pol, trained_pol):
        if hasattr(pol, "fixed_latent_strategy"):
            pol.fixed_latent_strategy = True
            pol.fixed_latent_strategy_id = branch

    # Fixed OP9 observation batch from the init policy.
    env = _make_env(init_ckpt, args.map, int(args.seed) + 7, device, int(args.max_decision_steps))
    try:
        snaps = _collect_shared_history(
            init_pol,
            env,
            opponent=str(args.opponent),
            n_steps=int(args.n_obs),
            seed=int(args.seed) + 11,
        )
        if not snaps:
            print("ERROR: no observations collected")
            return 3
        obs = _obs_batch(snaps, device)
    finally:
        env.close()

    logits_init_ckpt = _get_logits(init_pol, obs, branch)
    logits_trained = _get_logits(trained_pol, obs, branch)
    checkpoint_compare = _policy_metrics(logits_init_ckpt, logits_trained)

    la_init = _latent_actor(init_pol)
    la_tr = _latent_actor(trained_pol)
    groups_init = _param_groups(la_init, branch)
    groups_tr = _param_groups(la_tr, branch)
    baselines = _baseline_snapshots(groups_tr, groups_init, la_tr, branch)
    trained_snaps = {g: _snapshot_group(groups_tr[g]) for g in groups_tr}

    # Birth-equivalent logits on the trained graph (identity trunks + init z0 modules).
    for gname in ("z_embedding", "z_branch_trunk", "z_action_head", "z_adapter_residual"):
        _set_group_to_snapshot(groups_tr[gname], baselines[gname])
    logits_birth = _get_logits(trained_pol, obs, branch)
    # Restore trained weights.
    for gname in ("z_embedding", "z_branch_trunk", "z_action_head", "z_adapter_residual"):
        _set_group_to_snapshot(groups_tr[gname], trained_snaps[gname])

    birth_vs_trained = _policy_metrics(logits_birth, logits_trained)
    birth_vs_init_ckpt = _policy_metrics(logits_init_ckpt, logits_birth)

    module_reports: dict[str, Any] = {}
    for gname in (
        "z_embedding",
        "z_branch_trunk",
        "z_action_head",
        "z_adapter_residual",
        "combined_z_pathway",
    ):
        snap_b = baselines[gname]
        snap_t = trained_snaps[gname]
        dtheta = _group_delta_norm(snap_b, snap_t)

        # Isolation: hold all modules at birth baseline; apply this group's 1x delta.
        for g2 in ("z_embedding", "z_branch_trunk", "z_action_head", "z_adapter_residual"):
            _set_group_to_snapshot(groups_tr[g2], baselines[g2])
        if gname == "combined_z_pathway":
            _apply_group_delta(
                la_tr, groups_tr["combined_z_pathway"], baselines["combined_z_pathway"],
                trained_snaps["combined_z_pathway"], scale=1.0,
            )
        else:
            _apply_group_delta(la_tr, groups_tr[gname], snap_b, snap_t, scale=1.0)
        logits_replay = _get_logits(trained_pol, obs, branch)
        metrics = _policy_metrics(logits_birth, logits_replay)
        for g2 in ("z_embedding", "z_branch_trunk", "z_action_head", "z_adapter_residual"):
            _set_group_to_snapshot(groups_tr[g2], trained_snaps[g2])
        restore_kl = float(
            _kl_cat(logits_trained, _get_logits(trained_pol, obs, branch)).mean().item()
        )

        authority = float(metrics["mean_logit_l2"] / max(dtheta, 1e-12))
        module_reports[gname] = {
            "n_tensors": len(snap_t),
            "param_delta_l2": dtheta,
            "actual_update_replay": metrics,
            "logit_authority_ratio": authority,
            "directional_sensitivity_J_dtheta": authority,
            "restore_kl_after_probe": restore_kl,
            "param_names": sorted(snap_t.keys()),
            "baseline": "identity_trunk" if gname == "z_branch_trunk" else "init_checkpoint",
        }

    # Scaled-delta sweep on combined z0 pathway (trained graph).
    scales = [0.5, 1.0, 2.0, 5.0, 10.0]
    scaled_sweep = []
    for s in scales:
        for g2 in ("z_embedding", "z_branch_trunk", "z_action_head", "z_adapter_residual"):
            _set_group_to_snapshot(groups_tr[g2], baselines[g2])
        _apply_group_delta(
            la_tr,
            groups_tr["combined_z_pathway"],
            baselines["combined_z_pathway"],
            trained_snaps["combined_z_pathway"],
            scale=float(s),
        )
        logits_s = _get_logits(trained_pol, obs, branch)
        m = _policy_metrics(logits_birth, logits_s)
        m["scale"] = float(s)
        scaled_sweep.append(m)
    for g2 in ("z_embedding", "z_branch_trunk", "z_action_head", "z_adapter_residual"):
        _set_group_to_snapshot(groups_tr[g2], trained_snaps[g2])

    # Residual-alpha forward sweep on the TRAINED model (no param changes).
    alpha_vals = [0.0, 0.1, 0.25, 0.5, 1.0]
    alpha_sweep = []
    la = la_tr
    alpha0 = float(getattr(la, "_latent_z_alpha", 0.1))
    logits_ref = _get_logits(trained_pol, obs, branch)
    for a in alpha_vals:
        la._latent_z_alpha = float(a)
        logits_a = _get_logits(trained_pol, obs, branch)
        m_vs_default = (
            {
                "mean_kl": 0.0,
                "mean_logit_l2": 0.0,
                "mean_logit_linf": 0.0,
                "argmax_disagree": 0.0,
            }
            if abs(a - alpha0) <= 1e-12
            else _policy_metrics(logits_ref, logits_a)
        )
        alpha_sweep.append(
            {
                "alpha": float(a),
                "vs_trained_default_alpha": m_vs_default,
                "vs_init_ckpt_logits": _policy_metrics(logits_init_ckpt, logits_a),
                "vs_birth_graph_logits": _policy_metrics(logits_birth, logits_a),
            }
        )
    la._latent_z_alpha = alpha0

    # Interpretation heuristics
    combined = module_reports["combined_z_pathway"]
    trunk = module_reports["z_branch_trunk"]
    head = module_reports["z_action_head"]
    adapter = module_reports["z_adapter_residual"]
    kl_at_1x = next(x["mean_kl"] for x in scaled_sweep if abs(x["scale"] - 1.0) < 1e-9)
    kl_at_10x = next(x["mean_kl"] for x in scaled_sweep if abs(x["scale"] - 10.0) < 1e-9)
    alpha_span = float(
        max(r["vs_birth_graph_logits"]["mean_kl"] for r in alpha_sweep)
        - min(r["vs_birth_graph_logits"]["mean_kl"] for r in alpha_sweep)
    )

    if (
        trunk["n_tensors"] > 0
        and trunk["param_delta_l2"] > 1e-3
        and trunk["actual_update_replay"]["mean_kl"] < 1e-4
        and head["actual_update_replay"]["mean_kl"] > 5.0 * max(trunk["actual_update_replay"]["mean_kl"], 1e-12)
    ):
        reading = "LOW_TRUNK_SENSITIVITY_HEALTHY_HEAD"
    elif adapter["param_delta_l2"] > 0 and alpha_span < 1e-4:
        reading = "ALPHA_INSENSITIVE_DEEPER_PATH"
    elif (
        alpha_span >= 1e-3
        and abs(alpha0 - 0.1) < 1e-9
        and adapter["actual_update_replay"]["mean_kl"] < head["actual_update_replay"]["mean_kl"]
    ):
        reading = "RESIDUAL_ALPHA_LIKELY_THROTTLING"
    elif kl_at_10x > 5.0 * max(kl_at_1x, 1e-8) and kl_at_1x < 1e-3:
        reading = "VALID_DIRECTION_OPTIMIZER_STEP_TOO_SMALL"
    elif kl_at_10x < 1e-4:
        reading = "PATHWAY_INSENSITIVE_OR_BYPASSED"
    elif combined["actual_update_replay"]["mean_kl"] > 1e-3 and checkpoint_compare["mean_kl"] < 1e-4:
        reading = "KL_LOGGER_OR_CHECKPOINT_COMPARE_MISMATCH"
    else:
        reading = "MIXED_OR_NEEDS_MANUAL_INSPECTION"

    report = {
        "protocol": "v6i26_logit_control_authority_probe",
        "init_checkpoint": str(init_ckpt),
        "trained_checkpoint": str(trained_ckpt),
        "branch": branch,
        "opponent": str(args.opponent),
        "map": str(args.map),
        "n_obs": int(obs["grid"].shape[0]),
        "default_residual_alpha": alpha0,
        "checkpoint_compare_init_vs_trained": checkpoint_compare,
        "birth_graph_vs_trained": birth_vs_trained,
        "birth_graph_vs_init_ckpt": birth_vs_init_ckpt,
        "module_reports": module_reports,
        "scaled_delta_sweep_combined_z_pathway": scaled_sweep,
        "residual_alpha_forward_sweep": alpha_sweep,
        "reading": reading,
        "do_not_continue_to_10u": True,
        "notes": [
            "LRO deep trunks absent from init ckpt; trunk baseline = identity (post-load sync)",
            "All replays mutate the trained graph so trunks stay in the forward path",
            "Isolation: all z0 modules at birth baseline; apply one module's measured delta",
            "logit_authority_ratio = mean_logit_l2 / param_delta_l2 for 1x isolated replay",
            "scaled sweep: birth + scale*(trained-birth) on combined z0 pathway",
            "alpha sweep is forward-only on the trained weights",
        ],
    }
    out = Path(args.output)
    write_json(out, report)
    print("=" * 72)
    print("Logit control-authority probe")
    print("=" * 72)
    print(f"branch=z{branch} obs={report['n_obs']} opponent={args.opponent}|{args.map}")
    print(
        f"init_ckpt→trained KL={checkpoint_compare['mean_kl']:.3e} "
        f"logit_l2={checkpoint_compare['mean_logit_l2']:.3e} "
        f"argmax_disagree={checkpoint_compare['argmax_disagree']:.3f}"
    )
    print(
        f"birth_graph→trained KL={birth_vs_trained['mean_kl']:.3e} "
        f"(birth≈init_ckpt KL={birth_vs_init_ckpt['mean_kl']:.3e})"
    )
    for gname, rep in module_reports.items():
        print(
            f"  {gname}: dθ={rep['param_delta_l2']:.3e} "
            f"replay_KL={rep['actual_update_replay']['mean_kl']:.3e} "
            f"authority={rep['logit_authority_ratio']:.3e}"
        )
    print("scaled sweep KL:", [round(x["mean_kl"], 6) for x in scaled_sweep])
    print(
        "alpha sweep vs_birth KL:",
        [round(x["vs_birth_graph_logits"]["mean_kl"], 6) for x in alpha_sweep],
    )
    print(f"reading={reading}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
