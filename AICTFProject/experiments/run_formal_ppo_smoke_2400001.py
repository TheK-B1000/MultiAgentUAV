"""Formal PPO smoke -- seed 2400001, 50k steps, map_a / RULESET_V2, OP6-OP12.

Drives the REAL production entrypoint (``orchestrate_training_run``) and
instruments it from the outside: no production module is edited, the probes
only observe. Phases:

    A  training with runtime-health probes
    B  checkpoint save/reload round-trip against the live RunIdentity
    C  negative-load checks (must reject BEFORE state application)
    D  on-disk bundle validation -> smoke_verdict.json

Run:  python experiments/run_formal_ppo_smoke_2400001.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch  # noqa: E402

# --- smoke constants (the handoff contract) ---------------------------------

RUN_TAG = "formal_smoke_2400001"
SEED = 2_400_001
TOTAL_TIMESTEPS = 50_000
CHECKPOINT_INTERVAL = 10_000
DEVICE = "cuda"
CANONICAL_MAP = "map_a"
RESOLVED_MAP = "map_a_open"
RULESET_ID = "RULESET_V2_AQUATICUS_10S"
EPISODE_HORIZON = 240
AGENTS = 2
# Gate 1 admission record: artifacts/gate1_opponent_sanity_v2/gate1_result.json
OPPONENTS = ("OP6", "OP7", "OP8", "OP9", "OP10", "OP11", "OP12")

# Plumbing rehearsal only -- exercises all four phases on a tiny budget so a
# wiring bug surfaces in seconds instead of after the real run. Never used for
# the reported smoke.
DRY_RUN = os.environ.get("SMOKE_DRY_RUN") == "1"
if DRY_RUN:
    RUN_TAG = "formal_smoke_dryrun"
    TOTAL_TIMESTEPS = 4_096
    CHECKPOINT_INTERVAL = 2_048

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / RUN_TAG
CKPT_DIR = ARTIFACT_DIR / "ckpts"

KL_MEAN_TOL = 1e-8
KL_MAX_TOL = 1e-7

# Round-trip comparison runs on CPU float64 accumulation of a fixed batch.
FIXED_BATCH_SEED = 909_001


# --- runtime health ledger --------------------------------------------------


class HealthLedger:
    """Accumulates every runtime-health signal the smoke must report."""

    def __init__(self) -> None:
        self.failures: list[str] = []
        self.updates = 0
        self.rollouts = 0
        self.nonfinite_stats: list[str] = []
        self.nonfinite_buffer: list[str] = []
        self.nonfinite_params: list[str] = []
        self.nonfinite_optimizer: list[str] = []
        self.grad_norms: list[float] = []
        self.zero_grad_steps = 0
        self.optimizer_steps = 0
        self.param_change_events = 0
        self.episodes_completed = 0
        self.resets_observed = 0
        self.tag_success = 0
        self.tag_denied_cooldown = 0
        self.capture_events = 0
        self.legality_violations: dict[str, int] = defaultdict(int)
        self.identity_violations: dict[str, int] = defaultdict(int)
        self.inactive_diagnostics: set[str] = set()
        self.events_seen = 0
        # identity bookkeeping
        self._event_seq_prev = 0
        self._seen_event_seq: set[int] = set()
        self._per_env_last_episode: dict[int, int] = {}
        self._per_env_episode_keys: dict[int, set] = defaultdict(set)
        self._success_keys: set = set()
        self._denied_keys: set = set()

    def fail(self, msg: str) -> None:
        if msg not in self.failures:
            self.failures.append(msg)


def _finite(value) -> bool:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return True  # non-numeric stats are not a finiteness signal
    return math.isfinite(f)


# NaN here is a designated "this diagnostic did not run" sentinel, not a
# divergence signal: rl/custom_ppo/update/minibatch_updater.py seeds exactly
# these keys with float("nan") and only overwrites them when the sequential
# actor/counterfactual pathway is active. This baseline has no latent strategy,
# so they stay NaN by construction. They are reported separately rather than
# silently dropped -- and any OTHER NaN is still a hard failure.
NOT_COMPUTED_SENTINELS = frozenset({
    "actor_jsd_update_start", "actor_jsd_after_ppo", "actor_jsd_after_cf",
    "actor_jsd_after_first_substep", "actor_jsd_after_second_substep",
    "ppo_jsd_delta", "cf_jsd_delta", "cf_gain", "retained_cf_gain",
    "cf_retention_ratio", "actor_kl_after_ppo", "actor_kl_after_cf",
    "actor_kl_after_second_substep",
})


def _scan_stats(ledger: HealthLedger, stats: dict) -> None:
    for key, value in stats.items():
        if not isinstance(value, (int, float)) or _finite(value):
            continue
        if key in NOT_COMPUTED_SENTINELS:
            ledger.inactive_diagnostics.add(key)
            continue
        ledger.nonfinite_stats.append(f"{key}={value}")
        ledger.fail(f"non-finite training stat: {key}={value}")


def _scan_tensor_mapping(name: str, mapping, sink: list, ledger: HealthLedger) -> None:
    for key, tensor in (mapping or {}).items():
        if not torch.is_tensor(tensor) or not tensor.is_floating_point():
            continue
        if not torch.isfinite(tensor).all():
            sink.append(f"{name}.{key}")
            ledger.fail(f"non-finite {name}: {key}")


# --- tag legality (same hard checks as Gate 1) ------------------------------

SUCCESS_FIELDS = {
    "tagger_on_own_side", "target_on_tagger_side", "distance_at_decision",
    "target_was_tagged", "tagger_cooldown_before", "eligible_target_indices",
    "selected_nearest_target", "tagger_team", "target_team", "tagger_index",
    "target_index", "env_index", "simulation_time",
}
DENIED_FIELDS = {"reason", "cooldown_remaining", "tagger_team", "tagger_index", "env_index"}

IDENTITY_FIELDS = (
    "env_index", "episode_id", "reset_sequence", "simulation_step",
    "decision_step", "event_sequence",
)
# A reset marker is an episode BOUNDARY, not an in-episode event: the step
# counters are being zeroed as it is emitted, so it carries the ended/new
# episode ids instead (gpu_env/state/episode_state.py).
RESET_IDENTITY_FIELDS = (
    "env_index", "episode_id", "ended_episode_id", "reset_sequence",
    "event_sequence",
)


def _check_event(ledger: HealthLedger, e: dict, *, tag_range: float) -> None:
    ledger.events_seen += 1
    et = e.get("event_type")

    # --- integer event identity: unique, ordered, episode-correct ----------
    required = RESET_IDENTITY_FIELDS if et == "episode_reset" else IDENTITY_FIELDS
    missing_ident = [k for k in required if k not in e]
    if missing_ident:
        ledger.identity_violations["missing_identity_field"] += 1
        ledger.fail(f"event missing identity fields: {missing_ident}")
        return

    seq = int(e["event_sequence"])
    env_i = int(e["env_index"])
    ep_id = int(e["episode_id"])

    if seq in ledger._seen_event_seq:
        ledger.identity_violations["duplicate_event_sequence"] += 1
    ledger._seen_event_seq.add(seq)
    if seq <= ledger._event_seq_prev:
        ledger.identity_violations["event_sequence_not_increasing"] += 1
    ledger._event_seq_prev = seq

    # episode ids must never move backwards within one parallel env
    last_ep = ledger._per_env_last_episode.get(env_i)
    if last_ep is not None and ep_id < last_ep:
        ledger.identity_violations["episode_id_regressed"] += 1
    ledger._per_env_last_episode[env_i] = ep_id
    ledger._per_env_episode_keys[env_i].add(ep_id)

    if et == "episode_reset":
        ledger.resets_observed += 1
        # the boundary must be contiguous: the episode that ended is exactly
        # the one before the episode that begins
        if int(e["ended_episode_id"]) + 1 != ep_id:
            ledger.identity_violations["reset_boundary_not_contiguous"] += 1
        return
    if et == "capture_scored":
        ledger.capture_events += 1
        return

    if et == "tag_success":
        ledger.tag_success += 1
        if SUCCESS_FIELDS - set(e):
            ledger.legality_violations["schema_missing_field"] += 1
            return
        key = (env_i, ep_id, round(float(e["simulation_time"]), 6),
               e["tagger_team"], int(e["tagger_index"]), int(e["target_index"]))
        if key in ledger._success_keys:
            ledger.legality_violations["duplicate_event"] += 1
        ledger._success_keys.add(key)

        if not e["tagger_on_own_side"]:
            ledger.legality_violations["tagger_not_on_own_side"] += 1
        if not e["target_on_tagger_side"]:
            ledger.legality_violations["target_not_on_tagger_side"] += 1
        if float(e["distance_at_decision"]) > tag_range + 1e-6:
            ledger.legality_violations["tag_out_of_range"] += 1
        if e["target_was_tagged"]:
            ledger.legality_violations["retagged_already_tagged_target"] += 1
        if float(e["tagger_cooldown_before"]) > 1e-9:
            ledger.legality_violations["tag_during_cooldown"] += 1
        elig = list(e["eligible_target_indices"])
        if not elig:
            ledger.legality_violations["tag_with_no_eligible_target"] += 1
        elif e["selected_nearest_target"] not in elig:
            ledger.legality_violations["selected_target_not_eligible"] += 1
        if e["tagger_team"] == e["target_team"]:
            ledger.legality_violations["friendly_tag"] += 1
        return

    if et == "tag_denied":
        ledger.tag_denied_cooldown += 1
        if DENIED_FIELDS - set(e):
            ledger.legality_violations["schema_missing_field"] += 1
            return
        if e["reason"] != "cooldown":
            ledger.legality_violations["unexpected_denial_reason"] += 1
        if float(e["cooldown_remaining"]) <= 0.0:
            ledger.legality_violations["denial_without_cooldown"] += 1
        ledger._denied_keys.add(
            (env_i, ep_id, round(float(e["simulation_time"]), 6),
             e["tagger_team"], int(e["tagger_index"]))
        )
        return

    ledger.legality_violations["unknown_event_type"] += 1


def _finalize_event_checks(ledger: HealthLedger) -> None:
    """A denial and a success must never coexist for one tagger at one instant."""
    succ = {(k[0], k[1], k[2], k[3], k[4]) for k in ledger._success_keys}
    for k in ledger._denied_keys:
        if k in succ:
            ledger.legality_violations["denied_and_succeeded_same_instant"] += 1

    # Parallel environment identities must be disjoint: no event_sequence may be
    # shared, and each env keeps its own episode namespace.
    envs = sorted(ledger._per_env_episode_keys)
    for i, a in enumerate(envs):
        for b in envs[i + 1:]:
            # Episode ids may legitimately repeat across envs (each env counts
            # its own). Disjointness is enforced on the FULL identity, which
            # includes env_index -- so the check is that no event_sequence
            # (global, monotonic) was reused. Already validated above.
            pass

    if ledger.legality_violations:
        ledger.fail(f"tag legality violations: {dict(ledger.legality_violations)}")
    if ledger.identity_violations:
        ledger.fail(f"event identity violations: {dict(ledger.identity_violations)}")


# --- probe installation -----------------------------------------------------


def install_probes(ledger: HealthLedger):
    """Wrap trainer methods so the real production loop runs, observed."""
    from rl.custom_ppo.trainer import CustomPPOTrainer

    real_update = CustomPPOTrainer.update
    real_collect = CustomPPOTrainer.collect_rollout
    state = {"opt_wrapped": False, "tag_range": None}

    def _wrap_optimizers(trainer):
        if state["opt_wrapped"]:
            return
        state["opt_wrapped"] = True
        bundle = trainer.optimizers
        for name in ("primary", "actor", "critic", "router", "actor_cf"):
            opt = getattr(bundle, name, None)
            if opt is None:
                continue
            real_step = opt.step

            def make(real_step=real_step, opt=opt):
                def step(*args, **kwargs):
                    total_sq = 0.0
                    n_with_grad = 0
                    for group in opt.param_groups:
                        for p in group["params"]:
                            if p.grad is None:
                                continue
                            if not torch.isfinite(p.grad).all():
                                ledger.fail("non-finite gradient at optimizer step")
                            total_sq += float(p.grad.detach().double().pow(2).sum().item())
                            n_with_grad += 1
                    norm = math.sqrt(total_sq)
                    if n_with_grad:
                        ledger.grad_norms.append(norm)
                        if norm <= 0.0:
                            ledger.zero_grad_steps += 1
                    ledger.optimizer_steps += 1
                    # prove the step actually moves parameters
                    probe = None
                    for group in opt.param_groups:
                        for p in group["params"]:
                            if p.grad is not None and p.numel():
                                probe = (p, p.detach().clone())
                                break
                        if probe is not None:
                            break
                    out = real_step(*args, **kwargs)
                    if probe is not None and not torch.equal(probe[0].detach(), probe[1]):
                        ledger.param_change_events += 1
                    return out
                return step

            opt.step = make()

    def collect_rollout(self, *args, **kwargs):
        _wrap_optimizers(self)
        buffer = real_collect(self, *args, **kwargs)
        ledger.rollouts += 1

        fields = getattr(buffer, "fields", None)
        if isinstance(fields, dict):
            _scan_tensor_mapping("rollout", fields, ledger.nonfinite_buffer, ledger)

        core = getattr(self.env, "core", None)
        if core is not None:
            if state["tag_range"] is None:
                state["tag_range"] = float(
                    getattr(core.cfg, "tag_radius", None)
                    or getattr(core, "tag_radius", 0.0)
                    or 0.0
                )
            try:
                for e in core.drain_tag_events():
                    _check_event(ledger, e, tag_range=state["tag_range"] or 1e9)
            except Exception as exc:  # observation must never break the run
                ledger.fail(f"tag event drain failed: {exc}")
        return buffer

    def update(self, buffer, *args, **kwargs):
        stats = real_update(self, buffer, *args, **kwargs)
        ledger.updates += 1
        _scan_stats(ledger, dict(stats))

        bad_params = [n for n, p in self.model.named_parameters()
                      if p.is_floating_point() and not torch.isfinite(p).all()]
        if bad_params:
            ledger.nonfinite_params.extend(bad_params[:5])
            ledger.fail(f"non-finite model parameters: {bad_params[:5]}")

        for name in ("primary", "actor", "critic", "router", "actor_cf"):
            opt = getattr(self.optimizers, name, None)
            if opt is None:
                continue
            for pstate in opt.state.values():
                for k, v in pstate.items():
                    if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():
                        ledger.nonfinite_optimizer.append(f"{name}.{k}")
                        ledger.fail(f"non-finite optimizer state: {name}.{k}")
        return stats

    CustomPPOTrainer.collect_rollout = collect_rollout
    CustomPPOTrainer.update = update
    return lambda: (
        setattr(CustomPPOTrainer, "collect_rollout", real_collect),
        setattr(CustomPPOTrainer, "update", real_update),
    )


# --- phase A: training ------------------------------------------------------


def build_config():
    from rl.config.ppo_config import PPOConfig
    from rl.telemetry_mode import TrainingTelemetryMode

    cfg = PPOConfig()
    cfg.run_tag = RUN_TAG
    cfg.seed = SEED
    cfg.total_timesteps = TOTAL_TIMESTEPS
    cfg.periodic_checkpoint_steps = CHECKPOINT_INTERVAL
    cfg.device = DEVICE
    cfg.map_layout = CANONICAL_MAP
    cfg.max_decision_steps = EPISODE_HORIZON
    cfg.max_blue_agents = AGENTS

    cfg.mode = "OPPONENT_POOL"
    cfg.opponent_randomize = True
    cfg.opponent_pool = OPPONENTS
    cfg.opponent_pool_weights = ()          # uniform over the admitted mixture
    cfg.train_domain_randomization = False  # DR off

    cfg.n_envs = 16
    cfg.n_steps = 128                       # 2048 timesteps per update
    cfg.batch_size = 512
    cfg.n_epochs = 4

    cfg.gpu_native_env = True
    cfg.use_latent_strategy = False
    cfg.use_stable_marl_ppo = False

    # Production path: formal runs require tag telemetry through PPOConfig.
    # Do not monkeypatch GPUFieldConfig — that hid the missing plumbing.
    cfg.formal_run = True
    cfg.tag_telemetry_enabled = True

    # fresh initialization: no warm start of any kind, no formal override
    cfg.load_path = None
    cfg.additional_timesteps = 0
    cfg.load_weights_only = False

    cfg.checkpoint_dir = str(CKPT_DIR)
    cfg.metrics_csv_path = str(ARTIFACT_DIR / "metrics.csv")
    cfg.episode_csv_path = str(ARTIFACT_DIR / "episode_rows.csv")
    cfg.training_telemetry_mode = TrainingTelemetryMode.OFF
    cfg.enable_progress_bar = False
    cfg.verbose_training = True
    return cfg


def phase_a_training(ledger: HealthLedger) -> dict:
    from rl.training.orchestrator import orchestrate_training_run

    cfg = build_config()
    restore_probes = install_probes(ledger)
    started = time.time()
    try:
        orchestrate_training_run(cfg)
        ok = True
        error = ""
    except BaseException as exc:
        ok = False
        error = f"{type(exc).__name__}: {exc}"
        ledger.fail(f"training raised: {error}")
        traceback.print_exc()
    finally:
        restore_probes()

    _finalize_event_checks(ledger)
    return {
        "completed": ok,
        "error": error,
        "wall_seconds": round(time.time() - started, 2),
        "cfg": cfg,
    }


# --- phase B: checkpoint round-trip -----------------------------------------


def _fixed_observation_batch(env, device):
    """A deterministic observation batch drawn from the live env.

    Built exactly like ``rl.custom_ppo.rollout.action_selection`` builds it, so
    the tensors entering the policy are the ones the trainer would really see.
    """
    torch.manual_seed(FIXED_BATCH_SEED)
    obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    return {
        key: torch.as_tensor(obs[key], dtype=torch.float32, device=device)
        for key in ("grid", "vec", "agent_mask", "mask")
    }


def _head_logits(model, obs):
    """Split the flat MultiDiscrete logits into one tensor per action head."""
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            flat = model.policy_logits(obs, z_idx=None)
    finally:
        model.train(was_training)
    heads = []
    offset = 0
    for dim in model.action_dims:
        heads.append(flat[:, offset:offset + int(dim)])
        offset += int(dim)
    return heads


def _compare_heads(heads_a, heads_b) -> dict:
    """Per-head KL + argmax agreement over the fixed batch."""
    kls = []
    argmax_diff = 0
    for la, lb in zip(heads_a, heads_b):
        pa = torch.log_softmax(la.double(), dim=-1)
        pb = torch.log_softmax(lb.double(), dim=-1)
        kls.append((pa.exp() * (pa - pb)).sum(dim=-1).abs().flatten())
        argmax_diff += int((la.argmax(dim=-1) != lb.argmax(dim=-1)).sum().item())
    allkl = torch.cat(kls) if kls else torch.zeros(1, dtype=torch.float64)
    return {
        "mean_kl": float(allkl.mean().item()),
        "max_kl": float(allkl.max().item()),
        "argmax_differences": argmax_diff,
        "action_heads": len(heads_a),
        "batch_elements": int(allkl.numel()),
    }


def phase_b_round_trip(cfg, ledger: HealthLedger) -> dict:
    """Save + reload through production paths; compare a fixed obs batch."""
    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.ruleset_identity import (
        ARTIFACT_IDENTITY_KEY,
        build_formal_run_identity,
        verify_checkpoint_run_identity,
    )
    from rl.training.env_factory import build_training_env
    from rl.training.initialization import build_trainer
    from rl.training.resolved_config import resolve_training_config

    result: dict = {"ok": False}
    final_ckpt = CKPT_DIR / f"final_{RUN_TAG}.zip"
    result["checkpoint_path"] = str(final_ckpt)
    if not final_ckpt.is_file():
        result["error"] = f"final checkpoint missing: {final_ckpt}"
        ledger.fail(result["error"])
        return result

    env = None
    try:
        resolved = resolve_training_config(cfg)
        env = build_training_env(
            cfg,
            initial_phase=resolved.initial_phase,
            initial_opponent_tag=resolved.initial_opponent_tag,
        )
        live_identity = build_formal_run_identity(env, run_id=RUN_TAG)
        result["live_run_identity"] = live_identity.artifact_identity()

        # checkpoint verifies against the LIVE identity
        payload = read_checkpoint_payload(str(final_ckpt), map_location="cpu")
        verify_checkpoint_run_identity(
            payload, live_identity, operation="load", context=str(final_ckpt)
        )
        result["identity_verified_against_live_env"] = True
        ck_ai = payload.get(ARTIFACT_IDENTITY_KEY, {})
        result["checkpoint_artifact_identity"] = dict(ck_ai)

        trainer = build_trainer(env, cfg, resolved, run_identity=live_identity)
        obs = _fixed_observation_batch(env, trainer.device)

        # Restore twice from the same on-disk bytes and compare. A faithful
        # restore is one where the second load reproduces the first exactly --
        # parameters, optimizer state, and the policy's answer to a fixed batch.
        trainer.load(str(final_ckpt))
        params_a = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
        opt_a = trainer.optimizers.primary.state_dict()
        heads_a = _head_logits(trainer.model, obs)
        step_a = int(trainer.global_step)

        # perturb, so a no-op load could not masquerade as a correct restore
        with torch.no_grad():
            for p in trainer.model.parameters():
                if p.is_floating_point():
                    p.add_(torch.randn_like(p) * 0.05)
        heads_perturbed = _head_logits(trainer.model, obs)
        perturb_check = _compare_heads(heads_a, heads_perturbed)
        result["perturbation_moved_policy"] = perturb_check["max_kl"] > KL_MAX_TOL

        trainer.load(str(final_ckpt))
        params_b = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
        opt_b = trainer.optimizers.primary.state_dict()
        heads_b = _head_logits(trainer.model, obs)
        step_b = int(trainer.global_step)

        mismatched = [n for n in params_a
                      if n not in params_b or not torch.equal(params_a[n], params_b[n])]
        result["parameter_tensors_equal"] = not mismatched
        result["mismatched_parameters"] = mismatched[:10]

        result.update(_compare_heads(heads_a, heads_b))

        result["optimizer_state_restored"] = _optimizer_states_equal(opt_a, opt_b)
        result["global_step_restored"] = step_a == step_b
        result["global_step"] = step_a

        result["ok"] = bool(
            result.get("parameter_tensors_equal")
            and result.get("optimizer_state_restored")
            and result.get("global_step_restored")
            and result.get("perturbation_moved_policy")
            and result.get("argmax_differences", 1) == 0
            and result.get("mean_kl", 1.0) <= KL_MEAN_TOL
            and result.get("max_kl", 1.0) <= KL_MAX_TOL
        )
        if not result["ok"]:
            ledger.fail(f"checkpoint round-trip failed: {result}")
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        ledger.fail(f"round-trip raised: {result['error']}")
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    return result


def _optimizer_states_equal(a: dict, b: dict) -> bool:
    sa, sb = a.get("state", {}), b.get("state", {})
    if set(sa) != set(sb):
        return False
    for k in sa:
        for field in set(sa[k]) | set(sb[k]):
            va, vb = sa[k].get(field), sb[k].get(field)
            if torch.is_tensor(va) and torch.is_tensor(vb):
                if not torch.equal(va, vb):
                    return False
            elif va != vb:
                return False
    return True


# --- phase C: negative-load checks ------------------------------------------


def phase_c_negative_loads(cfg, ledger: HealthLedger) -> dict:
    """Every hostile checkpoint must be rejected BEFORE state application."""
    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload
    from rl.ruleset_identity import (
        ARTIFACT_IDENTITY_KEY,
        RunIdentityError,
        build_formal_run_identity,
        ruleset_fingerprint_hash,
    )
    from rl.training.env_factory import build_training_env
    from rl.training.initialization import build_trainer
    from rl.training.resolved_config import resolve_training_config

    final_ckpt = CKPT_DIR / f"final_{RUN_TAG}.zip"
    neg_dir = ARTIFACT_DIR / "negative_checkpoints"
    neg_dir.mkdir(parents=True, exist_ok=True)
    cases: dict = {}

    env = None
    try:
        base_payload = read_checkpoint_payload(str(final_ckpt), map_location="cpu")
        resolved = resolve_training_config(cfg)
        env = build_training_env(
            cfg,
            initial_phase=resolved.initial_phase,
            initial_opponent_tag=resolved.initial_opponent_tag,
        )
        live_identity = build_formal_run_identity(env, run_id=RUN_TAG)
        trainer = build_trainer(env, cfg, resolved, run_identity=live_identity)

        def mutate(name: str, fn) -> Path:
            payload = dict(base_payload)
            payload["ruleset"] = dict(base_payload.get("ruleset") or {})
            payload[ARTIFACT_IDENTITY_KEY] = dict(base_payload.get(ARTIFACT_IDENTITY_KEY) or {})
            fn(payload)
            path = neg_dir / f"{name}.zip"
            torch.save(payload, path)
            return path

        def v1(payload):
            rs = {
                "ruleset_id": "RULESET_V1_TWO_TAGGER",
                "taggers_required": 2,
                "tag_min_interval_seconds": 0.0,
                "tag_nearest_only": False,
                "tag_channel_seconds": 1.0,
                "suppression_attackers_required": 2,
            }
            payload["ruleset"] = rs
            payload[ARTIFACT_IDENTITY_KEY]["ruleset_id"] = rs["ruleset_id"]
            payload[ARTIFACT_IDENTITY_KEY]["ruleset_fingerprint"] = ruleset_fingerprint_hash(rs)

        def legacy(payload):
            payload.pop("ruleset", None)
            payload.pop(ARTIFACT_IDENTITY_KEY, None)

        def different_map(payload):
            payload[ARTIFACT_IDENTITY_KEY]["canonical_map"] = "map_b_split_lane"
            payload[ARTIFACT_IDENTITY_KEY]["resolved_map"] = "map_b_split_lane"

        def altered_field(payload):
            rs = dict(payload["ruleset"])
            rs["tag_min_interval_seconds"] = 30.0   # V2 label, different game
            payload["ruleset"] = rs
            payload[ARTIFACT_IDENTITY_KEY]["ruleset_fingerprint"] = ruleset_fingerprint_hash(rs)

        builders = {
            "v1_checkpoint": v1,
            "legacy_checkpoint": legacy,
            "different_map": different_map,
            "altered_ruleset_field": altered_field,
        }

        for name, fn in builders.items():
            path = mutate(name, fn)
            before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
            step_before = int(trainer.global_step)
            entry = {"rejected": False, "error_type": "", "message": ""}
            try:
                trainer.load(str(path))
                entry["message"] = "LOADED WITHOUT REJECTION"
                ledger.fail(f"negative load not rejected: {name}")
            except RunIdentityError as exc:
                entry.update(rejected=True, error_type="RunIdentityError",
                             message=str(exc)[:220])
            except Exception as exc:
                entry.update(rejected=True, error_type=type(exc).__name__,
                             message=str(exc)[:220])
                ledger.fail(f"negative load {name} raised {type(exc).__name__}, "
                            "expected RunIdentityError")
            # state must be untouched
            changed = [n for n, p in trainer.model.named_parameters()
                       if not torch.equal(p.detach(), before[n])]
            entry["state_unchanged"] = not changed and int(trainer.global_step) == step_before
            entry["changed_parameters"] = changed[:5]
            if not entry["state_unchanged"]:
                ledger.fail(f"negative load {name} mutated trainer state before rejection")
            cases[name] = entry
    except BaseException as exc:
        cases["_error"] = f"{type(exc).__name__}: {exc}"
        ledger.fail(f"negative-load phase raised: {exc}")
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass

    cases["all_rejected_before_state_application"] = all(
        isinstance(v, dict) and v.get("rejected") and v.get("state_unchanged")
        for k, v in cases.items() if k not in ("_error", "all_rejected_before_state_application")
    )
    return cases


# --- phase D: bundle validation ---------------------------------------------


def phase_d_bundle(ledger: HealthLedger) -> dict:
    from rl.ruleset_identity import RunIdentityError, validate_bundle

    out: dict = {}
    run_config = ARTIFACT_DIR / f"{RUN_TAG}_run_config.json"
    expected = {
        "run_config.json": run_config,
        "training_manifest.json": ARTIFACT_DIR / "training_manifest.json",
        "evaluation_manifest.json": ARTIFACT_DIR / "evaluation_manifest.json",
        "result_summary.json": ARTIFACT_DIR / "result_summary.json",
    }
    episode_csv = ARTIFACT_DIR / "episode_rows.csv"
    final_ckpt = CKPT_DIR / f"final_{RUN_TAG}.zip"

    present = {label: p.is_file() for label, p in expected.items()}
    present["episode_rows.csv"] = episode_csv.is_file()
    present["checkpoint"] = final_ckpt.is_file()
    out["artifacts_present"] = present
    missing = [k for k, v in present.items() if not v]
    if missing:
        out["missing"] = missing
        ledger.fail(f"bundle missing artifacts: {missing}")
        out["valid"] = False
        return out

    jsons = {label: json.loads(p.read_text(encoding="utf-8"))
             for label, p in expected.items()}
    with open(episode_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out["episode_row_count"] = len(rows)

    try:
        ref = validate_bundle(jsons, {"episode_rows.csv": rows}, require_formal=True)
        out["valid"] = True
        out["reference_identity"] = {
            k: ref.get(k) for k in
            ("run_id", "canonical_map", "resolved_map", "ruleset_id",
             "ruleset_fingerprint", "formal_result_eligible")
        }
        out["map_correct"] = ref.get("canonical_map") == CANONICAL_MAP
        out["resolved_map_correct"] = ref.get("resolved_map") == RESOLVED_MAP
        out["ruleset_correct"] = ref.get("ruleset_id") == RULESET_ID
        out["formal_result_eligible"] = bool(ref.get("formal_result_eligible"))
        # Telemetry must travel through PPOConfig into the on-disk travelers —
        # not only into the live env via a harness monkeypatch.
        run_cfg = jsons["run_config.json"]
        train_man = jsons["training_manifest.json"]
        out["run_config_tag_telemetry_enabled"] = bool(
            run_cfg.get("tag_telemetry_enabled")
        )
        out["training_manifest_tag_telemetry_enabled"] = bool(
            train_man.get("tag_telemetry_enabled")
        )
        if not out["run_config_tag_telemetry_enabled"]:
            ledger.fail("run_config.json does not record tag_telemetry_enabled=true")
        if not out["training_manifest_tag_telemetry_enabled"]:
            ledger.fail("training_manifest.json does not record tag_telemetry_enabled=true")
        if not out["formal_result_eligible"]:
            ledger.fail("bundle is not formal_result_eligible")
        for key in ("map_correct", "resolved_map_correct", "ruleset_correct"):
            if not out[key]:
                ledger.fail(f"bundle identity wrong: {key}")
    except RunIdentityError as exc:
        out["valid"] = False
        out["error"] = str(exc)[:300]
        ledger.fail(f"bundle validation failed: {exc}")
    return out


# --- main -------------------------------------------------------------------


def main() -> int:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ledger = HealthLedger()

    print("=" * 78)
    print(f"FORMAL PPO SMOKE  seed={SEED}  steps={TOTAL_TIMESTEPS}  device={DEVICE}")
    print(f"map={CANONICAL_MAP}->{RESOLVED_MAP}  ruleset={RULESET_ID}")
    print(f"opponents={OPPONENTS}  horizon={EPISODE_HORIZON}")
    print(f"artifact_dir={ARTIFACT_DIR}")
    print("=" * 78)

    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("FATAL: CUDA required but unavailable.")
        return 2

    print("\n--- PHASE A: training ---")
    phase_a = phase_a_training(ledger)
    cfg = phase_a.pop("cfg")

    print("\n--- PHASE B: checkpoint round-trip ---")
    phase_b = phase_b_round_trip(cfg, ledger) if phase_a["completed"] else {
        "ok": False, "error": "skipped: training did not complete"}

    print("\n--- PHASE C: negative-load checks ---")
    phase_c = phase_c_negative_loads(cfg, ledger) if phase_a["completed"] else {
        "_error": "skipped: training did not complete"}

    print("\n--- PHASE D: bundle validation ---")
    phase_d = phase_d_bundle(ledger)

    grad_norms = ledger.grad_norms
    runtime_health = {
        "updates": ledger.updates,
        "rollouts": ledger.rollouts,
        "optimizer_steps": ledger.optimizer_steps,
        "parameter_change_events": ledger.param_change_events,
        "zero_gradient_steps": ledger.zero_grad_steps,
        "grad_norm_min": min(grad_norms) if grad_norms else None,
        "grad_norm_max": max(grad_norms) if grad_norms else None,
        "grad_norm_mean": (sum(grad_norms) / len(grad_norms)) if grad_norms else None,
        "nonfinite_training_stats": ledger.nonfinite_stats[:10],
        "nonfinite_rollout_fields": ledger.nonfinite_buffer[:10],
        "nonfinite_parameters": ledger.nonfinite_params[:10],
        "nonfinite_optimizer_state": ledger.nonfinite_optimizer[:10],
        "inactive_diagnostics_not_computed": sorted(ledger.inactive_diagnostics),
        "events_observed": ledger.events_seen,
        "tag_success_events": ledger.tag_success,
        "cooldown_denial_events": ledger.tag_denied_cooldown,
        "capture_events": ledger.capture_events,
        "resets_observed": ledger.resets_observed,
        "hard_tag_legality_violations": dict(ledger.legality_violations),
        "event_identity_violations": dict(ledger.identity_violations),
        "parallel_envs": len(ledger._per_env_episode_keys),
    }

    # contract requirements that are not simple finiteness
    if phase_a["completed"]:
        if ledger.optimizer_steps == 0:
            ledger.fail("no optimizer steps were taken")
        if ledger.param_change_events == 0:
            ledger.fail("optimizer never changed a parameter")
        if not grad_norms or max(grad_norms) <= 0.0:
            ledger.fail("gradients were all zero")
        if ledger.tag_success == 0:
            ledger.fail("no tag_success events occurred")
        if ledger.tag_denied_cooldown == 0:
            ledger.fail("no cooldown-denial events occurred")

    failures = list(ledger.failures)
    verdict = "PASS" if (
        phase_a["completed"] and phase_b.get("ok")
        and phase_c.get("all_rejected_before_state_application")
        and phase_d.get("valid") and not failures
    ) else "FAIL"

    smoke = {
        "verdict": verdict,
        "run_tag": RUN_TAG,
        "artifact_dir": str(ARTIFACT_DIR),
        "configuration": {
            "seed": SEED,
            "total_timesteps": TOTAL_TIMESTEPS,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "device": DEVICE,
            "canonical_map": CANONICAL_MAP,
            "resolved_map": RESOLVED_MAP,
            "ruleset_id": RULESET_ID,
            "opponents": list(OPPONENTS),
            "episode_horizon": EPISODE_HORIZON,
            "domain_randomization": False,
            "initialization": "fresh",
            "v1_warm_start": "forbidden",
            "formal_override": "forbidden",
        },
        "training": phase_a,
        "runtime_health": runtime_health,
        "checkpoint_round_trip": phase_b,
        "negative_loads": phase_c,
        "bundle_validation": phase_d,
        "thresholds": {
            "mean_kl_max": KL_MEAN_TOL,
            "max_kl_max": KL_MAX_TOL,
            "argmax_differences_max": 0,
        },
        "failures": failures,
    }

    verdict_path = ARTIFACT_DIR / "smoke_verdict.json"
    verdict_path.write_text(json.dumps(smoke, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"SMOKE VERDICT: {verdict}")
    print(f"written: {verdict_path}")
    if failures:
        print("failures:")
        for f in failures:
            print(f"  - {f}")
    print("=" * 78)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
