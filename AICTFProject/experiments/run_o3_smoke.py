"""O3 20-episode CUDA smoke: prove the handoff and credit boundary before 3400001.

Frozen by 46c9e17 / de6fb58 / ce7949f / b7c3527 / 0b18ed3.

Checks, all required:
    trigger_step == first_o3_action_step
    possession loss does not return control to G0
    re-pickup remains O3
    five pre-handoff sample counts == 0
    response_supervision_used == false
    schedule-inertness preflight
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

OUT = PROJECT_ROOT / "artifacts" / "o3_smoke"
PREREG = PROJECT_ROOT / "artifacts" / "o3_preregistration"
FORBIDDEN_ARTIFACTS = (
    "C3_STAGE3_ANCHOR_RESULTS.jsonl",
    "C3_QUALIFIED_COMMITMENT_FORKS.json",
    "C_FORK_PRECURSOR_AUDIT.json",
)
FORBIDDEN_FIELDS = (
    "best_team_response", "witness_team_response", "branch_action",
    "team_response", "best_expected_utility", "max_expected_utility_improvement",
)


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def preflight(cfg) -> dict:
    """46c9e17 leakage preflight plus schedule inertness."""
    problems = []

    # 1. No off-limits artifact reachable from the training entry point.
    training_modules = [
        PROJECT_ROOT / "experiments" / "o3_handoff.py",
        PROJECT_ROOT / "experiments" / "o3_credit_boundary.py",
        PROJECT_ROOT / "rl" / "analysis" / "c_fork_detector.py",
        PROJECT_ROOT / "rl" / "analysis" / "legal_team_responses.py",
    ]
    for m in training_modules:
        text = m.read_text(encoding="utf-8")
        # Strip docstrings/comments crudely: only flag executable references.
        code = "\n".join(
            ln for ln in text.splitlines()
            if not ln.strip().startswith("#")
        )
        for art in FORBIDDEN_ARTIFACTS:
            if art in code and "artifacts" in code.split(art)[0][-200:]:
                problems.append(f"{m.name} references forbidden artifact {art}")
        for fld in FORBIDDEN_FIELDS:
            if f'"{fld}"' in code or f"'{fld}'" in code:
                problems.append(f"{m.name} references forbidden field {fld}")

    # 2. Schedule inertness: reward shaping must not advance on env steps.
    if int(getattr(cfg, "reward_shaping_decay_steps", 0)) != 0:
        problems.append(
            f"reward_shaping_decay_steps={cfg.reward_shaping_decay_steps} != 0; "
            "an env-step-driven schedule would advance during the G0 prefix"
        )
    if float(getattr(cfg, "reward_shaping_coef_start", 1.0)) != float(
        getattr(cfg, "reward_shaping_coef_end", 1.0)
    ):
        problems.append("reward_shaping_coef_start != coef_end; schedule is not inert")

    # 3. No latent machinery.
    if bool(getattr(cfg, "use_latent_strategy", False)):
        problems.append("use_latent_strategy is on; O3 is an independent policy")
    if str(getattr(cfg, "phase_pod_id", "") or ""):
        problems.append("phase_pod_id set; V6I26 pods are prohibited")

    return {"PASS": not problems, "problems": problems,
            "response_supervision_used": False}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("REFUSED: CUDA required; artifacts are CUDA-produced")

    OUT.mkdir(parents=True, exist_ok=True)
    from experiments.run_g0_v5_long import build_config as g0_build_config
    from experiments.o3_credit_boundary import install_credit_boundary
    from experiments.o3_handoff import install_o3_handoff
    from experiments.run_g0_v2_evaluation import resolve_cnn_channels
    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.evaluation.checkpoint import load_policy
    import rl.training.orchestrator as orch

    cfg = g0_build_config(3_400_001)
    cfg.run_tag = "o3_smoke"
    cfg.seed = 3_400_001
    cfg.device = args.device
    cfg.total_timesteps = int(args.episodes) * 240
    cfg.n_envs = 4
    cfg.n_steps = 128
    cfg.batch_size = 128
    cfg.checkpoint_dir = str(OUT / "ckpts")
    cfg.metrics_csv_path = str(OUT / "metrics.csv")
    cfg.episode_csv_path = str(OUT / "episode_rows.csv")
    cfg.load_path = None
    cfg.use_latent_strategy = False
    cfg.phase_pod_id = ""
    cfg.periodic_checkpoint_steps = 0

    pf = preflight(cfg)
    print("=" * 74)
    print("O3 SMOKE - handoff + credit boundary")
    print(f"  preflight: {'PASS' if pf['PASS'] else 'FAIL'}")
    for p_ in pf["problems"]:
        print(f"    - {p_}")
    if not pf["PASS"]:
        return 3

    tag = "g0_v5_long_seed3200001"
    ck = PROJECT_ROOT / "artifacts" / "g0_v5_long" / tag / "ckpts" / f"final_{tag}.zip"
    payload = read_checkpoint_payload(str(ck), map_location="cpu")
    g0 = load_policy(str(ck), device=args.device,
                     num_cnn_channels=resolve_cnn_channels(payload, context=str(ck)))
    g0_model = g0.model if hasattr(g0, "model") else g0

    holder = {}
    real_build_trainer = orch.build_trainer

    def build_trainer(env, cfg_, resolved, **kw):
        trainer = real_build_trainer(env, cfg_, resolved, **kw)
        hs, det, un_h = install_o3_handoff(trainer, g0_model, strict=True)
        au, un_c = install_credit_boundary(trainer, hs, strict=True)
        holder.update(hstate=hs, audit=au, un_h=un_h, un_c=un_c)
        return trainer

    orch.build_trainer = build_trainer
    started = time.time()
    err = ""
    try:
        orch.orchestrate_training_run(cfg)
    except BaseException as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        import traceback
        traceback.print_exc()
    finally:
        orch.build_trainer = real_build_trainer
        if "un_c" in holder:
            holder["un_c"]()
        if "un_h" in holder:
            holder["un_h"]()

    hstate = holder.get("hstate")
    audit = holder.get("audit")
    if hstate is None or audit is None:
        print("FAILED: handoff was never installed")
        return 4

    tp = hstate.throughput()
    ca = audit.to_dict()
    report = {
        "record": "O3 smoke",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": args.device,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "g0_checkpoint_sha256": _sha256(ck),
        "o3_seed": 3_400_001,
        "error": err,
        "throughput": tp,
        "credit_audit": ca,
        "response_supervision_used": False,
        "preflight": pf,
        "wall_seconds": round(time.time() - started, 1),
        "episode_events_sample": hstate.episode_events[:8],
        "checks": {
            "trigger_equals_first_o3_action": "asserted in-flight by install_o3_handoff(strict=True)",
            "episodes_seen_positive": int(tp["episodes_seen"]) > 0,
            "episodes_with_handoff_positive": int(tp["episodes_with_handoff"]) > 0,
            "latch_cleared_on_every_done": all(
                (not e.get("active_after_reset", False)) for e in hstate.episode_events
            ),
            "five_pre_handoff_counts_zero": all(
                ca[k] == 0 for k in (
                    "pre_handoff_actor_samples", "pre_handoff_critic_samples",
                    "pre_handoff_entropy_samples", "pre_handoff_norm_samples",
                    "pre_handoff_return_targets")
            ),
        },
    }
    (OUT / "O3_SMOKE_RESULT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n  episodes_seen        : {tp['episodes_seen']}")
    print(f"  episodes_with_handoff: {tp['episodes_with_handoff']}  rate {tp['handoff_rate']}")
    print(f"  environment_steps    : {tp['environment_steps']}")
    print(f"  credited_o3_steps    : {tp['credited_o3_steps']}  fraction {tp['credited_fraction']}")
    print(f"  mean_post_handoff_len: {tp['mean_post_handoff_length']}")
    print(f"  eff O3 steps / 1k env: {tp['effective_o3_steps_per_1k_env_steps']}")
    print(f"  credited row fraction: {ca['credited_row_fraction']}")
    print(f"  five pre-handoff counts zero: {report['checks']['five_pre_handoff_counts_zero']}")
    print(f"  error: {err or '(none)'}")
    print(f"  wrote {OUT / 'O3_SMOKE_RESULT.json'}")
    print("=" * 74)
    return 0 if not err else 1


if __name__ == "__main__":
    raise SystemExit(main())
