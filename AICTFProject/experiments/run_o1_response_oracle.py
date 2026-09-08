"""O1 — the first response oracle, trained on the confirmed C1 niche.

Protocol: docs/o1-response-oracle-preregistration.md
Constants: artifacts/o1_preregistration/O1_PREREGISTRATION.json (frozen 2026-08-04)

This script does not choose anything. Every constant it uses is read out of the
frozen preregistration JSON and checked against it at startup; if the two ever
disagree the run refuses to start rather than training something the protocol
did not declare.

WHAT DIFFERS FROM G0-V5
-----------------------
Exactly one thing: the reset distribution. Episodes start inside C1 via
``experiments.c1_context.apply_c1_scenario``. The reward is the locked Reward V5
with no role bonus, no z label and no C1-specific term -- PPO is told where to
play, never how. The configuration is inherited from
``run_g0_v5_long.build_config`` verbatim rather than retyped, so "identical
constants to G0-V5" is enforced by construction and then re-verified field by
field against G0_V5_LONG_RUN_SPEC.json.

READING THE HEALTH OUTPUT
-------------------------
Two panels are reported per checkpoint and they mean different things:

    SYSTEM_HEALTH   gating. Is PPO numerically alive?
    C1 panel        diagnostic. Is O1 learning anything on its own niche?
    normal panel    DIAGNOSTIC ONLY, EXPLICITLY NON-GATING.

The normal panel evaluates O1 on ordinary uninjected play, where a response
oracle is *permitted to be worse than G0* -- gate 2 only requires that O1 does
not exceed G0 there. A weak normal panel is not a stop condition and is not
evidence against O1. Stopping the run because O1 looks bad at ordinary play
would be discarding the specialist for being a specialist.

Run:  python experiments/run_o1_response_oracle.py --seed 3300001
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

from experiments.c1_context import attach_c1_injector, c1_active_mask  # noqa: E402
from experiments.run_formal_ppo_smoke_2400001 import (  # noqa: E402
    HealthLedger,
    _finalize_event_checks,
    _scan_stats,
    _scan_tensor_mapping,
)
from experiments.run_g0_v2_seed import (  # noqa: E402
    CANONICAL_MAP,
    OPPONENTS,
    RULESET_ID,
)

PREREG_PATH = PROJECT_ROOT / "artifacts" / "o1_preregistration" / "O1_PREREGISTRATION.json"
G0_SPEC_PATH = PROJECT_ROOT / "artifacts" / "g0_v5_long" / "G0_V5_LONG_RUN_SPEC.json"

# Panel sizes: diagnostics only, deliberately small so they do not dominate
# wall time. Neither panel gates anything.
C1_PANEL_EPISODES = 2
PANEL_OPPONENTS = ("OP6", "OP9", "OP12")

# Reward fields the frozen G0-V5 spec names, mapped to their PPOConfig field.
# Fields the spec lists that are not PPOConfig overrides (flag capture points,
# pickup, carry-home) come from the env defaults that build_config inherits.
REWARD_FIELD_MAP = {
    "sparse_tag_no_flag_points": "env_sparse_tag_no_flag_points",
    "sparse_tag_with_flag_points": "env_sparse_tag_with_flag_points",
    "enemy_mav_kill_reward": "env_enemy_mav_kill_reward",
    "sparse_mine_tag_points": "env_sparse_mine_tag_points",
    "action_failed_punishment": "env_action_failed_punishment",
    "sparse_opponent_oob_points": "env_sparse_opponent_oob_points",
    "sparse_own_oob_points": "env_sparse_own_oob_points",
}


def load_prereg() -> dict:
    if not PREREG_PATH.is_file():
        raise FileNotFoundError(
            f"frozen preregistration missing: {PREREG_PATH}. O1 must not run "
            "without the protocol it was declared under."
        )
    return json.loads(PREREG_PATH.read_text(encoding="utf-8"))


def run_tag_for(seed: int) -> str:
    return f"o1_response_oracle_seed{seed}"


def artifact_dir_for(seed: int) -> Path:
    return PROJECT_ROOT / "artifacts" / "o1_response_oracle" / run_tag_for(seed)


def build_config(seed: int, prereg: dict):
    """G0-V5's config verbatim, re-pointed at this run. Nothing else changes."""
    from experiments.run_g0_v5_long import build_config as g0_v5_build_config

    cfg = g0_v5_build_config(seed)

    art = artifact_dir_for(seed)
    cfg.run_tag = run_tag_for(seed)
    cfg.seed = int(seed)
    cfg.checkpoint_dir = str(art / "ckpts")
    cfg.metrics_csv_path = str(art / "metrics.csv")
    cfg.episode_csv_path = str(art / "episode_rows.csv")

    train = prereg["training"]
    cfg.total_timesteps = int(train["steps_per_seed"])
    # Fresh init: warm start is prohibited by the protocol.
    cfg.load_path = None
    cfg.additional_timesteps = 0
    cfg.load_weights_only = False
    # No latent machinery: O1 is an independent policy, not an adapter.
    cfg.use_latent_strategy = False
    cfg.phase_pod_id = ""
    return cfg


# --- protocol enforcement ---------------------------------------------------


def verify_against_prereg(cfg, seed: int, prereg: dict) -> list[str]:
    """Refuse to train anything the frozen protocol did not declare."""
    train = prereg["training"]
    problems: list[str] = []

    if int(seed) not in [int(s) for s in train["seeds"]]:
        problems.append(f"seed {seed} is not one of the declared O1 seeds {train['seeds']}")
    if int(cfg.total_timesteps) != int(train["steps_per_seed"]):
        problems.append(
            f"total_timesteps={cfg.total_timesteps} != declared {train['steps_per_seed']}"
        )
    if cfg.load_path:
        problems.append(f"warm start is FORBIDDEN but load_path={cfg.load_path!r}")
    if str(cfg.map_layout) != str(train["map"]):
        problems.append(f"map {cfg.map_layout!r} != declared {train['map']!r}")
    if int(cfg.max_decision_steps) != int(train["episode_horizon"]):
        problems.append(
            f"horizon {cfg.max_decision_steps} != declared {train['episode_horizon']}"
        )
    if bool(cfg.train_domain_randomization) != bool(train["domain_randomization"]):
        problems.append("domain randomization does not match the declared value")
    if tuple(cfg.opponent_pool) != tuple(OPPONENTS):
        problems.append(f"opponent pool {cfg.opponent_pool} != admitted mixture {OPPONENTS}")
    if getattr(cfg, "phase_pod_id", ""):
        problems.append(
            f"phase_pod_id={cfg.phase_pod_id!r}: V6I26 phase pods are prohibited"
        )
    if bool(getattr(cfg, "use_latent_strategy", False)):
        problems.append("use_latent_strategy is on; O1 is an independent policy")
    return problems


def verify_reward_matches_g0(cfg) -> tuple[list[str], dict]:
    """Field-by-field check against the frozen G0-V5 reward, not a claim of it."""
    spec = json.loads(G0_SPEC_PATH.read_text(encoding="utf-8"))
    locked = spec["locked_reward_v5"]
    problems: list[str] = []
    verified: dict = {}
    for spec_key, cfg_field in REWARD_FIELD_MAP.items():
        want = float(locked[spec_key])
        got = getattr(cfg, cfg_field, None)
        if got is None or not math.isclose(float(got), want, rel_tol=0, abs_tol=1e-12):
            problems.append(f"{cfg_field}={got!r} != frozen V5 {spec_key}={want}")
        else:
            verified[cfg_field] = want
    want_gamma = float(locked["gamma"])
    if not math.isclose(float(cfg.gamma), want_gamma, rel_tol=0, abs_tol=1e-12):
        problems.append(f"gamma={cfg.gamma} != frozen V5 gamma={want_gamma}")
    else:
        verified["gamma"] = want_gamma
    return problems, verified


def install_c1_injection() -> tuple:
    """Patch the orchestrator's env builder to inject C1 after every reset.

    Patched at the name the orchestrator actually calls rather than at the
    definition, because ``rl.training.orchestrator`` imports the symbol
    directly and would otherwise keep the original binding.

    ``attach_c1_injector`` chains onto any existing after-reset hook, so the
    opponent pool's own before-reset hook is untouched.
    """
    import rl.training.orchestrator as orch

    real_build = orch.build_training_env

    def build_with_c1(*a, **k):
        env = real_build(*a, **k)
        attach_c1_injector(env)
        # The env may already have been reset while being primed, before the
        # hook existed. Inject once directly so episode 0 of every env starts
        # in C1 too; later resets go through the hook.
        from experiments.c1_context import apply_c1_scenario

        apply_c1_scenario(env.core)
        return env

    orch.build_training_env = build_with_c1
    return lambda: setattr(orch, "build_training_env", real_build)


def preflight_c1_density(seed: int, device: str) -> dict:
    """Prove the injector fires before spending 1,000,000 steps on it.

    Builds the real training env through the patched path and checks that every
    env starts inside C1, both on the initial reset and after a forced re-reset.
    A silent injector would train O1 on ordinary play while the report claimed
    otherwise -- which is exactly how POD_DEFEND_LEAD's dead clock line went
    unnoticed.
    """
    import rl.training.orchestrator as orch

    from experiments.run_g0_v5_long import build_config as g0_v5_build_config

    cfg = g0_v5_build_config(seed)
    cfg.n_envs = 4
    cfg.device = device
    env = orch.build_training_env(cfg, initial_phase="PHASE1", initial_opponent_tag="OP6")
    try:
        at_build = float(c1_active_mask(env.core).float().mean().item())
        env.reset()
        at_reset = float(c1_active_mask(env.core).float().mean().item())
    finally:
        env.close()
    return {
        "c1_fraction_at_build": round(at_build, 4),
        "c1_fraction_after_reset": round(at_reset, 4),
        "PASS": bool(at_build >= 0.999 and at_reset >= 0.999),
    }


# --- diagnostic panels ------------------------------------------------------


def run_c1_panel(trainer, *, device: str) -> dict:
    """Injected-C1 play with the current weights. Diagnostic, never gating."""
    from experiments.o1_rollout import run_c1_episode
    from rl.custom_ppo.inference_policy import CustomPPOInferencePolicy

    model = trainer.model
    was_training = model.training
    model.eval()
    rows = []
    try:
        policy = CustomPPOInferencePolicy(model=model, device=torch.device(device))
        for opp in PANEL_OPPONENTS:
            for i in range(C1_PANEL_EPISODES):
                rows.append(
                    run_c1_episode(
                        policy, opponent=opp, seed=9_900_000 + i,
                        device=device, inject_c1=True,
                    )
                )
    finally:
        model.train(was_training)

    n = max(len(rows), 1)
    led = [r for r in rows if r["lead_seen"]]
    return {
        "episodes": len(rows),
        "win_rate": round(sum(r["win"] for r in rows) / n, 4),
        "mean_margin": round(sum(r["score_margin"] for r in rows) / n, 4),
        "lead_preserved_rate": (
            round(sum(r["lead_preserved"] for r in led) / len(led), 4) if led else None
        ),
        "captures_conceded": sum(r["captures_red"] for r in rows),
        "note": "injected C1 starts; DIAGNOSTIC ONLY, gates score natural episodes",
    }


def install_probes(ledger: HealthLedger, records: list, art: Path, device: str):
    """SYSTEM_HEALTH numerics plus the two diagnostic panels."""
    from rl.custom_ppo.trainer import CustomPPOTrainer
    from rl.training.task_health import evaluate_task_health

    real_update = CustomPPOTrainer.update
    real_collect = CustomPPOTrainer.collect_rollout
    real_save = CustomPPOTrainer.save
    state = {"opt_wrapped": False}

    def _wrap_optimizers(trainer):
        if state["opt_wrapped"]:
            return
        state["opt_wrapped"] = True
        for name in ("primary", "actor", "critic", "router", "actor_cf"):
            opt = getattr(trainer.optimizers, name, None)
            if opt is None:
                continue
            real_step = opt.step

            def make(real_step=real_step, opt=opt):
                def step(*a, **k):
                    tot, n = 0.0, 0
                    for g in opt.param_groups:
                        for p in g["params"]:
                            if p.grad is None:
                                continue
                            if not torch.isfinite(p.grad).all():
                                ledger.fail("non-finite gradient at optimizer step")
                            tot += float(p.grad.detach().double().pow(2).sum().item())
                            n += 1
                    if n:
                        norm = math.sqrt(tot)
                        ledger.grad_norms.append(norm)
                        if norm <= 0.0:
                            ledger.zero_grad_steps += 1
                    ledger.optimizer_steps += 1
                    return real_step(*a, **k)
                return step
            opt.step = make()

    def collect_rollout(self, *a, **k):
        _wrap_optimizers(self)
        buf = real_collect(self, *a, **k)
        ledger.rollouts += 1
        fields = getattr(buf, "fields", None)
        if isinstance(fields, dict):
            _scan_tensor_mapping("rollout", fields, ledger.nonfinite_buffer, ledger)
        core = getattr(self.env, "core", None)
        if core is not None:
            # Standing check: the injector must keep firing for the whole run,
            # not just at preflight.
            state["c1_frac_last"] = float(c1_active_mask(core).float().mean().item())
        return buf

    def update(self, buffer, *a, **k):
        stats = real_update(self, buffer, *a, **k)
        ledger.updates += 1
        _scan_stats(ledger, dict(stats))
        bad = [n for n, p in self.model.named_parameters()
               if p.is_floating_point() and not torch.isfinite(p).all()]
        if bad:
            ledger.nonfinite_params.extend(bad[:5])
            ledger.fail(f"non-finite model parameters: {bad[:5]}")
        return stats

    def save(self, path: str):
        real_save(self, path)
        step = int(getattr(self, "global_step", 0))
        system_ok = not (
            ledger.nonfinite_stats or ledger.nonfinite_buffer or ledger.nonfinite_params
            or ledger.nonfinite_optimizer or ledger.legality_violations
            or ledger.identity_violations or ledger.zero_grad_steps
        )
        try:
            c1_panel = run_c1_panel(self, device=device)
        except Exception as exc:
            c1_panel = {"error": f"{type(exc).__name__}: {exc}"}

        # Ordinary play, for the learning curve only. A response oracle is
        # allowed to be worse than G0 here; gate 2 only forbids it being better.
        try:
            from experiments.run_g0_v5_long import run_validation_panel

            normal = run_validation_panel(self, global_step=step, device=device).to_dict()
            normal["GATING"] = False
            normal["note"] = "ordinary play; NON-GATING for a response oracle"
        except Exception as exc:
            normal = {"verdict": "ERROR", "error": f"{type(exc).__name__}: {exc}",
                      "GATING": False}

        rec = {
            "global_step": step,
            "checkpoint": str(path),
            "SYSTEM_HEALTH": "PASS" if system_ok else "FAIL",
            "c1_panel": c1_panel,
            "normal_panel_non_gating": normal,
            "c1_fraction_at_last_rollout": state.get("c1_frac_last"),
            "grad_norm_mean_postclip": (
                sum(ledger.grad_norms) / len(ledger.grad_norms)
            ) if ledger.grad_norms else None,
            "zero_gradient_steps": ledger.zero_grad_steps,
        }
        records.append(rec)
        (art / "health_timeline.json").write_text(
            json.dumps(records, indent=2, default=str), encoding="utf-8")
        print(f"[O1] step={step} SYSTEM_HEALTH={rec['SYSTEM_HEALTH']} "
              f"c1_wr={c1_panel.get('win_rate')} "
              f"c1_lead_kept={c1_panel.get('lead_preserved_rate')} "
              f"c1_frac={rec['c1_fraction_at_last_rollout']}")

    CustomPPOTrainer.collect_rollout = collect_rollout
    CustomPPOTrainer.update = update
    CustomPPOTrainer.save = save
    return lambda: (
        setattr(CustomPPOTrainer, "collect_rollout", real_collect),
        setattr(CustomPPOTrainer, "update", real_update),
        setattr(CustomPPOTrainer, "save", real_save),
    )


def main() -> int:
    prereg = load_prereg()
    declared_seeds = [int(s) for s in prereg["training"]["seeds"]]

    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True, choices=declared_seeds)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--preflight-only", action="store_true",
                    help="verify the protocol and the injector, then exit")
    args = ap.parse_args()

    torch.set_num_threads(max(1, int(args.threads)))
    seed = int(args.seed)
    art = artifact_dir_for(seed)
    art.mkdir(parents=True, exist_ok=True)
    (art / "ckpts").mkdir(parents=True, exist_ok=True)

    device = "cuda"
    if not torch.cuda.is_available():
        print("FATAL: CUDA required.")
        return 2

    cfg = build_config(seed, prereg)

    problems = verify_against_prereg(cfg, seed, prereg)
    reward_problems, reward_verified = verify_reward_matches_g0(cfg)
    problems += reward_problems
    if problems:
        print("=" * 78)
        print("REFUSING TO RUN: configuration does not match the frozen protocol")
        for p in problems:
            print(f"  - {p}")
        print("=" * 78)
        return 3

    print("=" * 78)
    print(f"O1 RESPONSE ORACLE  seed={seed}  steps={cfg.total_timesteps:,}")
    print(f"protocol: {PREREG_PATH.relative_to(PROJECT_ROOT)}")
    print(f"map={CANONICAL_MAP} ruleset={RULESET_ID} opponents={OPPONENTS}")
    print(f"init=fresh (warm start forbidden)  reset distribution=100% injected C1")
    print(f"reward: Reward V5, {len(reward_verified)} fields verified against the frozen spec")
    print("no role bonus, no z label, no C1-specific reward term")
    print("=" * 78)

    restore_env = install_c1_injection()
    try:
        pre = preflight_c1_density(seed, device)
        print(f"[preflight] C1 at build={pre['c1_fraction_at_build']} "
              f"after reset={pre['c1_fraction_after_reset']} -> "
              f"{'PASS' if pre['PASS'] else 'FAIL'}")
        if not pre["PASS"]:
            print("REFUSING TO RUN: the C1 injector did not fire on the training env.")
            return 4
        if args.preflight_only:
            print("preflight-only: protocol and injector verified, exiting before training.")
            return 0

        from rl.training.orchestrator import orchestrate_training_run

        ledger = HealthLedger()
        records: list = []
        restore_probes = install_probes(ledger, records, art, device)
        started = time.time()
        try:
            orchestrate_training_run(cfg)
            completed, error = True, ""
        except BaseException as exc:
            completed, error = False, f"{type(exc).__name__}: {exc}"
            ledger.fail(f"training raised: {error}")
            traceback.print_exc()
        finally:
            restore_probes()
    finally:
        restore_env()

    _finalize_event_checks(ledger)

    report = {
        "run": "O1 response oracle",
        "protocol": str(PREREG_PATH.relative_to(PROJECT_ROOT)),
        "seed": seed,
        "run_id": run_tag_for(seed),
        "preregistered_primary_step": int(cfg.total_timesteps),
        "initialization": "fresh",
        "warm_start": "FORBIDDEN",
        "reset_distribution": "100% apply_c1_scenario (injected C1 starts)",
        "reward": "locked Reward V5, unchanged from G0-V5",
        "reward_fields_verified": reward_verified,
        "role_reward_or_z_label": "NONE",
        "preflight": pre,
        "training": {"completed": completed, "error": error,
                     "wall_seconds": round(time.time() - started, 2)},
        "health_timeline": records,
        "system_failures": list(ledger.failures),
        "panels_are_diagnostic": (
            "Neither the C1 panel nor the normal panel gates anything. Retention "
            "is decided only by run_o1_gates.py on natural, uninjected episodes."
        ),
    }
    (art / "o1_training_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"seed {seed}: completed={completed} system_failures={len(ledger.failures)}")
    print(f"report: {art / 'o1_training_report.json'}")
    print("NEXT: all three seeds, then experiments/run_o1_gates.py")
    print("=" * 78)
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
