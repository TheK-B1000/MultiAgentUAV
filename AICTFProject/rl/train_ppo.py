"""Train the CTF policy with the local PPO/MAPPO implementation."""

from __future__ import annotations

import atexit
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import numpy as np
import torch

from game_field_gpu import GPUCTFVecEnv, GPUFieldConfig, VEC_OBS_DIM
from opponent_params import sample_batched_opponent_params
from rl.custom_ppo import CustomPPOTrainer
from rl.curriculum import CurriculumState, jacob_paper_curriculum_state, phase_from_tag
from rl.global_state import GLOBAL_STATE_DIM
from rl.latent_marl import CONTEXT_STATE_DIM
from rl.stress_schedule import STRESS_BY_PHASE


def _find_git_root() -> str:
    """Walk upward from this file to find a directory containing ``.git``; else ``cwd``."""
    p = os.path.abspath(_SCRIPT_DIR)
    for _ in range(8):
        if os.path.isdir(os.path.join(p, ".git")):
            return p
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return os.getcwd()


def _git_metadata() -> dict[str, Optional[str]]:
    """Best-effort ``git rev-parse`` / ``git describe`` from the repo root."""
    root = _find_git_root()
    meta: dict[str, Optional[str]] = {
        "git_sha": None,
        "git_describe": None,
        "git_root": root,
        "git_error": None,
    }
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if sha.returncode == 0 and sha.stdout.strip():
            meta["git_sha"] = sha.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        meta["git_error"] = str(exc)
    try:
        desc = subprocess.run(
            ["git", "describe", "--tags", "--always", "--dirty"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if desc.returncode == 0 and desc.stdout.strip():
            meta["git_describe"] = desc.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        meta["git_error"] = meta["git_error"] or str(exc)
    return meta


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    return str(obj)


def _run_config_json_path(cfg: PPOConfig) -> str:
    base_dir = cfg.checkpoint_dir
    if getattr(cfg, "metrics_csv_path", None):
        d = os.path.dirname(str(cfg.metrics_csv_path))
        if d:
            base_dir = d
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{cfg.run_tag}_run_config.json")


def write_run_config_json(cfg: PPOConfig, argv: Optional[list[str]] = None) -> str:
    """Write reproducibility sidecar JSON next to metrics CSV (or under ``checkpoint_dir``)."""
    path = _run_config_json_path(cfg)
    argv_list = list(sys.argv) if argv is None else list(argv)
    git_meta = _git_metadata()
    payload: dict[str, Any] = {
        "utc_timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "argv": argv_list,
        "working_directory": os.getcwd(),
        "python_executable": sys.executable,
        **git_meta,
        "run_tag": str(cfg.run_tag),
        "checkpoint_dir": str(cfg.checkpoint_dir),
        "total_timesteps": int(cfg.total_timesteps),
        "metrics_csv_path": cfg.metrics_csv_path,
        "episode_csv_path": cfg.episode_csv_path,
        "strategy_experience_csv_path": cfg.strategy_experience_csv_path,
        "load_path": cfg.load_path,
        "cli_preset": getattr(cfg, "cli_preset", None),
        "resolved_ppo_config": _json_safe(asdict(cfg)),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


def _metrics_csv_nonempty(path: Optional[str]) -> bool:
    return bool(path and os.path.isfile(path) and os.path.getsize(path) > 0)


def _rotate_csv_aside(path: Optional[str], *, label: str) -> None:
    if not _metrics_csv_nonempty(path):
        return
    assert path is not None
    bak = f"{path}.bak.{int(time.time())}"
    os.replace(path, bak)
    print(f"[PPO] Rotated existing {label} CSV aside: {bak!r} (--fresh-metrics-csv).")


@dataclass
class _RunLock:
    path: str
    token: str
    released: bool = False

    def release(self) -> None:
        if self.released:
            return
        self.released = True
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if payload.get("token") != self.token:
                return
            os.unlink(self.path)
            print(f"[PPO] Run lock released: {self.path}")
        except FileNotFoundError:
            return
        except Exception as exc:
            print(f"[PPO] WARNING: failed to release run lock {self.path!r}: {exc}")


def _pid_is_running(pid: int) -> bool:
    pid = int(pid)
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            process_query_limited_information = 0x1000
            still_active = 259
            handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
            if not handle:
                return False
            exit_code = ctypes.c_ulong()
            ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
            kernel32.CloseHandle(handle)
            return bool(ok) and int(exit_code.value) == still_active
        except Exception:
            return True
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_run_lock(path: str) -> dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _acquire_run_lock(cfg: PPOConfig) -> _RunLock:
    """Prevent duplicate trainers from sharing checkpoint/CSV artifacts for one run tag."""
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    lock_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}.run.lock")
    token = f"{os.getpid()}-{time.time_ns()}"
    payload = {
        "pid": os.getpid(),
        "token": token,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run_tag": str(cfg.run_tag),
        "argv": sys.argv,
        "metrics_csv_path": str(getattr(cfg, "metrics_csv_path", "") or ""),
        "episode_csv_path": str(getattr(cfg, "episode_csv_path", "") or ""),
        "strategy_experience_csv_path": str(getattr(cfg, "strategy_experience_csv_path", "") or ""),
    }
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            existing = _read_run_lock(lock_path)
            pid = int(existing.get("pid", 0) or 0)
            if pid > 0 and _pid_is_running(pid):
                raise RuntimeError(
                    f"Active PPO run lock exists for run_tag={cfg.run_tag!r}: {lock_path!r} "
                    f"(pid={pid}). Stop that trainer or use a different --run-tag before starting another run."
                ) from exc
            stale_path = f"{lock_path}.stale.{int(time.time())}"
            os.replace(lock_path, stale_path)
            print(f"[PPO] Rotated stale run lock aside: {stale_path!r}")
            continue

        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        setattr(cfg, "run_id", token)
        setattr(cfg, "run_pid", os.getpid())
        lock = _RunLock(lock_path, token)
        atexit.register(lock.release)
        print(f"[PPO] Run lock acquired: {lock_path}")
        return lock


def _gpu_env_reward_kwargs(cfg: PPOConfig) -> dict[str, Any]:
    """Map optional ``PPOConfig`` reward knobs onto ``GPUFieldConfig`` / ``RewardConfig`` field names."""
    pairs = (
        ("win_team_reward", getattr(cfg, "env_win_team_reward", None)),
        ("draw_team_penalty", getattr(cfg, "env_draw_team_penalty", None)),
        ("lose_team_punish", getattr(cfg, "env_lose_team_punish", None)),
        ("action_failed_punishment", getattr(cfg, "env_action_failed_punishment", None)),
        ("dense_weight", getattr(cfg, "env_dense_weight", None)),
        ("sparse_weight", getattr(cfg, "env_sparse_weight", None)),
        ("reward_scale", getattr(cfg, "env_reward_scale", None)),
        ("reward_clip", getattr(cfg, "env_reward_clip", None)),
        ("stalemate_penalty", getattr(cfg, "env_stalemate_penalty", None)),
        ("stalemate_max_steps", getattr(cfg, "env_stalemate_max_steps", None)),
    )
    out: dict[str, Any] = {}
    for name, raw in pairs:
        if raw is None:
            continue
        if name == "stalemate_max_steps":
            out[name] = max(1, int(raw))
        else:
            out[name] = float(raw)
    return out


def _resolve_2v2_checkpoint(filename: str) -> Optional[str]:
    """Find ``checkpoints/2v2/<filename>`` whether cwd is repo root or ``AICTFProject``."""
    cwd = os.getcwd()
    candidates = (
        os.path.join(_PARENT_DIR, "checkpoints", "2v2", filename),
        os.path.join(cwd, "checkpoints", "2v2", filename),
        os.path.join(cwd, "AICTFProject", "checkpoints", "2v2", filename),
        os.path.join(os.path.dirname(_PARENT_DIR), "AICTFProject", "checkpoints", "2v2", filename),
    )
    for raw in candidates:
        path = os.path.normpath(raw)
        if os.path.isfile(path):
            return path
    return None


def set_global_seed(seed: int, torch_seed: bool = True, deterministic: bool = False) -> None:
    """Set Python, NumPy, and Torch seeds."""
    random.seed(int(seed))
    np.random.seed(int(seed))
    if torch_seed:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


class TrainMode(str, Enum):
    FIXED_OPPONENT = "FIXED_OPPONENT"
    # Uniform random scripted opponent each episode (same hook as opponent_randomize; explicit mode).
    OPPONENT_POOL = "OPPONENT_POOL"
    CURRICULUM = "CURRICULUM"
    # Backward-compatible alias for old configs/commands.
    CURRICULUM_NO_LEAGUE = "CURRICULUM"


# Scripted tags dropped from training pools unless ``allow_op4_in_training_pool`` (eval / zero-shot default).
EVAL_ONLY_TRAINING_OPPONENT_TAGS: frozenset[str] = frozenset({"OP4"})


def _strip_eval_only_opponents_from_training_pool(cfg: PPOConfig) -> None:
    """Remove eval-only scripted opponents from ``cfg.opponent_pool`` when training samples that pool."""
    if bool(getattr(cfg, "allow_op4_in_training_pool", False)):
        return
    pool = tuple(str(x).strip().upper() for x in getattr(cfg, "opponent_pool", ()) if str(x).strip())
    banned = EVAL_ONLY_TRAINING_OPPONENT_TAGS
    filt = tuple(x for x in pool if x not in banned)
    if not filt:
        raise ValueError(
            "opponent_pool is empty after removing eval-only scripted tags "
            f"{sorted(banned)}. Use OP1–OP3, OP5–OP7 (and aliases OP5, OP6_TURTLE, OP7_SWITCHER) for training, or pass "
            "--allow-op4-in-training-pool together with OP4 in --opponent-pool."
        )
    removed = sorted(set(pool) - set(filt))
    if removed:
        print(
            "[PPO] opponent_pool excludes "
            f"{removed} (eval-only by default). "
            "Pass --allow-op4-in-training-pool to include those tags in training."
        )
        cfg.opponent_pool = filt


@dataclass
class PPOConfig:
    seed: int = 42
    total_timesteps: int = 1_000_000
    n_envs: int = 8
    n_steps: int = 2048
    batch_size: int = 1024
    n_epochs: int = 6
    gamma: float = 0.995
    gae_lambda: float = 0.99
    clip_range: float = 0.25
    clip_range_vf: Optional[float] = 0.2
    vf_coef: float = 1.0
    normalize_returns: bool = False
    ent_coef: float = 0.01
    learning_rate: float = 5e-4
    lr_floor_frac: float = 0.1
    max_grad_norm: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_dir: str = "checkpoints"
    load_path: Optional[str] = None
    run_tag: str = "ppo_latent_2v2"
    enable_metrics_csv: bool = True
    metrics_csv_path: Optional[str] = None
    episode_csv_path: Optional[str] = None
    strategy_experience_csv_path: Optional[str] = None
    # If True before training, existing non-empty metrics/episode CSVs are rotated aside so a new run
    # does not append duplicate timesteps under the same --run-tag.
    fresh_metrics_csv: bool = False
    # Set from CLI ``--preset`` only (reproducibility / run_config.json); behavior is already merged into fields below.
    cli_preset: Optional[str] = None
    # E3: optional per-step CSV (z, H(q), argmax, switch, phase). See `rl.custom_ppo.E3_STEP_TELEMETRY_FIELDS`.
    e3_step_telemetry_path: Optional[str] = None
    # SB3-compatible: ``tqdm`` (prefer ``tqdm.rich``) during rollout, ``total=remaining`` timesteps, ``update(n_envs)`` / step.
    enable_progress_bar: bool = True
    verbose_training: bool = False
    # After this many *completed* episodes, print W/L/D and win rate (0 = disabled).
    episode_log_every: int = 1000

    max_decision_steps: int = 400
    map_set: str = "train"
    mode: str = TrainMode.FIXED_OPPONENT.value
    fixed_opponent_tag: str = "OP3"
    # Uniform random scripted opponent per episode: either mode=OPPONENT_POOL or FIXED_OPPONENT + True.
    # Uses GPUCTFVecEnv pre-reset hook so the next episode matches sampled opponents from opponent_pool.
    # Default excludes OP4 (reserved for zero-shot eval). Use ``--allow-op4-in-training-pool`` to train vs OP4.
    opponent_randomize: bool = False
    opponent_pool: tuple[str, ...] = field(default_factory=lambda: ("OP1", "OP2", "OP3"))
    allow_op4_in_training_pool: bool = False
    max_blue_agents: int = 2
    use_deterministic: bool = False
    # Not in *Summer Implementation Plan.docx*; when True, overrides several PPO fields below for a legacy "stable" profile. Default False so explicit config matches the spec numbers.
    use_stable_marl_ppo: bool = False
    target_kl: Optional[float] = 0.02
    actor_cnn_feature_dim: int = 128

    # Summer/ICRA latent team strategy is the default proposed algorithm.
    use_latent_strategy: bool = True
    latent_k: int = 4
    latent_z_embed_dim: int = 16
    latent_actor_conditioning: Literal["concat"] = "concat"
    latent_vf_hidden: int = 128
    latent_strategy_hidden: int = 128
    # Plan IMPLEMENTATION §6: typical λ_H ∈ [0.001, 0.01]; λ_p ∈ [0.01, 0.05] (see also §3.3 for a wider λ_p range).
    # ``maximize`` matches the plan (encourage exploratory / diverse q_phi). ``minimize`` adds +λ_H·H to the
    # minimized loss and sharpens q_phi (recommended when telemetry shows strategy_entropy≈ln K with no persistence grad).
    # ``none`` removes the H term (strategy_encoder receives no gradient from λ_H when λ_p/KL are also inactive).
    latent_entropy_objective: Literal["maximize", "minimize", "none"] = "maximize"
    latent_lam_h: float = 0.005
    latent_lam_p: float = 0.02
    # A1: clipped PPO/REINFORCE-style update for sampled z. Kept low because z operates at episode cadence.
    latent_strategy_ppo_coef: float = 0.1
    # Option A episode-start strategy credit: PPO update on the sampled z using full
    # completed-episode return. Pure task-return credit; no labels or semantic heads.
    latent_episode_strategy_ppo: bool = False
    # Default 0.0 keeps episode-credit OFF by default per the SUMMER plan: latent z is
    # learned end-to-end from task reward via the MARL loss + persistence regularizer,
    # with no auxiliary objectives. Presets that opt into episode-credit (e.g.
    # plan_faithful_latent_episode_credit) must set this explicitly.
    latent_episode_strategy_coef: float = 0.0
    latent_episode_strategy_clip_eps: float = 0.2
    latent_episode_strategy_value_coef: float = 0.5
    latent_episode_strategy_return_norm: bool = True
    # A2 (opt-in): auxiliary MSE on the shared q_phi trunk predicting per-z returns from the **sampled** z only.
    # Not a full Q(s,a,z) critic and not off-policy Q-learning; MAPPO value remains V_phi(s, a, z).
    latent_strategy_aux_return_head: bool = False
    latent_strategy_aux_return_coef: float = 1.0
    latent_strategy_aux_predict_phase_coef: float = 0.0
    latent_strategy_tau: float = 1.0
    # 0 = sample once at episode start (main paper default; plan Option A). N>=2 = sparse refresh (Option B).
    latent_resample_every_n: int = 0
    # Mid-episode z changes make V(s,z) discontinuous; optionally break GAE carry across z[t]!=z[t+1].
    latent_gae_reset_on_z_change: bool = True
    # Use argmax z from q_phi(s') when bootstrapping V(s') so peek matches no duplicate stochastic z draw.
    latent_bootstrap_z_deterministic: bool = True
    # Baseline: keep latent actor/critic plumbing, but clamp every rollout to one strategy ID.
    # This tests whether learned/multiple strategy selection matters beyond a single learned z embedding.
    fixed_latent_strategy: bool = False
    fixed_latent_strategy_id: int = 0
    # **Ablation / plan §12 only** — not combined with the main “episode-start z” story by default.
    # Use ``rl.config_presets.ablation_flag_resample_config`` for an explicit run.
    latent_resample_on_flag: bool = False
    # Optional §12: KL( q_\phi(s_t) || q_\phi(s_{t-1}) ) on consecutive time steps; 0 = off (ablation only).
    latent_kl_consecutive: float = 0.0
    latent_q_phi_option_advantage: bool = False

    # Episode-level domain randomization for sim robustness (sensor dropout/noise, blue speed jitter).
    # See ``GPUFieldConfig`` for numeric ranges; eval harnesses should keep this False.
    train_domain_randomization: bool = False
    dr_sensor_noise_sigma_max: float = 0.12
    dr_sensor_dropout_max: float = 0.08
    dr_blue_speed_jitter: float = 0.12

    # Optional overrides forwarded to ``GPUFieldConfig`` reward shaping (None = env defaults).
    # Useful for training-winrate recipes: stronger W/D contrast and less dense dilution of terminals.
    env_win_team_reward: Optional[float] = None
    env_draw_team_penalty: Optional[float] = None
    env_lose_team_punish: Optional[float] = None
    env_action_failed_punishment: Optional[float] = None
    env_dense_weight: Optional[float] = None
    env_sparse_weight: Optional[float] = None
    env_reward_scale: Optional[float] = None
    env_reward_clip: Optional[float] = None
    env_stalemate_penalty: Optional[float] = None
    env_stalemate_max_steps: Optional[int] = None
    # Optional trainer-side reward shaping decay: scales (offense+pbrs+team) contribution seen by PPO.
    reward_shaping_coef_start: float = 1.0
    reward_shaping_coef_end: float = 1.0
    reward_shaping_decay_steps: int = 0
    periodic_checkpoint_steps: int = 50_000


def _apply_training_preset(cfg: PPOConfig, preset: str) -> PPOConfig:
    """Apply named high-level presets for repeatable training recipes."""
    from rl.presets import apply_preset

    return apply_preset(cfg, preset)


# Default ``python rl/train_ppo.py`` recipe when ``--preset`` is omitted: plan-faithful
# latent with sparse persistence and entropy. Pass ``--preset none`` to skip.
DEFAULT_CLI_TRAINING_PRESET = "plan_faithful_latent_persist_entropy"


def train_ppo(cfg: Optional[PPOConfig] = None) -> None:
    """Run the default local PPO/MAPPO training path."""
    cfg = cfg or PPOConfig()

    cfg.mode = _normalize_train_mode(cfg.mode)
    if cfg.mode == TrainMode.OPPONENT_POOL.value:
        cfg.opponent_randomize = True
    supported_modes = {
        TrainMode.FIXED_OPPONENT.value,
        TrainMode.OPPONENT_POOL.value,
        TrainMode.CURRICULUM.value,
    }
    if cfg.mode not in supported_modes:
        raise ValueError(
            "The local PPO trainer currently supports FIXED_OPPONENT, OPPONENT_POOL, and "
            "CURRICULUM training."
        )
    if cfg.mode == TrainMode.CURRICULUM.value:
        if bool(getattr(cfg, "use_latent_strategy", False)) or bool(getattr(cfg, "fixed_latent_strategy", False)):
            print("[PPO] Curriculum mode is the Jacob paper baseline; forcing latent strategy OFF.")
        cfg.use_latent_strategy = False
        cfg.fixed_latent_strategy = False

    if bool(getattr(cfg, "opponent_randomize", False)):
        if cfg.mode == TrainMode.CURRICULUM.value:
            raise ValueError(
                "opponent_randomize=True is incompatible with CURRICULUM mode "
                "(curriculum already sequences scripted opponents). "
                "Use mode=FIXED_OPPONENT or OPPONENT_POOL with opponent_randomize, or turn opponent_randomize off."
            )
        pool = tuple(str(x).strip().upper() for x in getattr(cfg, "opponent_pool", ()) if str(x).strip())
        allowed = {
            "OP1",
            "OP2",
            "OP3",
            "OP4",
            "OP5_RUSHER",
            "OP5",
            "OP6",
            "OP6_TURTLE",
            "OP7",
            "OP7_SWITCHER",
        }
        pool = tuple(x for x in pool if x in allowed)
        if not pool:
            raise ValueError(f"opponent_pool must contain at least one of {sorted(allowed)}; got {getattr(cfg, 'opponent_pool', ())!r}.")
        cfg.opponent_pool = pool
        _strip_eval_only_opponents_from_training_pool(cfg)

    if bool(getattr(cfg, "use_latent_strategy", False)):
        k = int(getattr(cfg, "latent_k", 4))
        if k < 1:
            raise ValueError("latent_k must be >= 1.")
        if bool(getattr(cfg, "fixed_latent_strategy", False)):
            fixed_id = int(getattr(cfg, "fixed_latent_strategy_id", 0) or 0)
            if fixed_id < 0 or fixed_id >= k:
                raise ValueError(f"fixed_latent_strategy_id must be in [0, {k - 1}] when latent_k={k}.")
        res_n = int(getattr(cfg, "latent_resample_every_n", 0) or 0)
        if res_n == 1:
            raise ValueError(
                "latent_resample_every_n=1 is disallowed (do not resample z every decision step). "
                "Use 0 (sample at episode start) or N>=2 (sparse refresh)."
            )
    elif bool(getattr(cfg, "fixed_latent_strategy", False)):
        raise ValueError("fixed_latent_strategy requires use_latent_strategy=True.")

    set_global_seed(cfg.seed, torch_seed=True, deterministic=cfg.use_deterministic)
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    max_agents = max(1, int(getattr(cfg, "max_blue_agents", 2)))
    team_size = _agents_suffix(max_agents)
    curriculum: CurriculumState | None = None
    initial_opponent_tag = str(cfg.fixed_opponent_tag).upper()
    initial_phase = phase_from_tag(initial_opponent_tag)
    if cfg.mode == TrainMode.CURRICULUM.value:
        curriculum = jacob_paper_curriculum_state(max_agents)
        initial_opponent_tag = curriculum.phase
        initial_phase = curriculum.phase
    print(f"[PPO] Agents: {max_agents} per team ({team_size}) | mode={cfg.mode} | run_tag={cfg.run_tag!r}")
    print("[PPO] Algorithm backend: custom local PPO")
    print(f"[PPO] Total timesteps: {int(cfg.total_timesteps):,}")
    base_gs_dim = GLOBAL_STATE_DIM
    temp_ctx_dim = CONTEXT_STATE_DIM if bool(getattr(cfg, "use_latent_strategy", False)) else 0
    q_phi_dim = CONTEXT_STATE_DIM if bool(getattr(cfg, "use_latent_strategy", False)) else 0
    crit_dim = CONTEXT_STATE_DIM if bool(getattr(cfg, "use_latent_strategy", False)) else GLOBAL_STATE_DIM
    actor_cnn_feat = int(getattr(cfg, "actor_cnn_feature_dim", 128))
    z_embed = int(getattr(cfg, "latent_z_embed_dim", 16))
    act_dim = (actor_cnn_feat + 20 + z_embed) if bool(getattr(cfg, "use_latent_strategy", False)) else (actor_cnn_feat + 20)
    print(
        f"[PPO] Input dims: base_global_state_dim={base_gs_dim} "
        f"temporal_context_dim={temp_ctx_dim} "
        f"q_phi_input_dim={q_phi_dim} "
        f"critic_context_dim={crit_dim} "
        f"actor_input_dim={act_dim}"
    )
    print(f"[PPO] Actor CNN feature dim: {int(getattr(cfg, 'actor_cnn_feature_dim', 128))}")
    print(f"[PPO] Map set: {str(getattr(cfg, 'map_set', 'train')).lower()}")
    if bool(getattr(cfg, "train_domain_randomization", False)):
        print(
            "[PPO] Domain randomization: ON "
            f"(sensor_noise_sigma max={float(getattr(cfg, 'dr_sensor_noise_sigma_max', 0.0)):.3f}, "
            f"sensor_dropout max={float(getattr(cfg, 'dr_sensor_dropout_max', 0.0)):.3f}, "
            f"blue_speed_jitter={float(getattr(cfg, 'dr_blue_speed_jitter', 0.0)):.3f}; "
            "blue-policy side only, slowdown-only speed scale)"
        )
    if curriculum is not None:
        print("[PPO] Training profile: curriculum baseline")
    elif bool(getattr(cfg, "use_latent_strategy", False)):
        print("[PPO] Training profile: default latent (Summer implementation)")
    else:
        print("[PPO] Training profile: no-latent baseline")
    if bool(getattr(cfg, "normalize_returns", False)):
        print("[PPO] Return normalization: enabled for critic targets/predictions; GAE uses denormalized values.")
    decay_steps = max(0, int(getattr(cfg, "reward_shaping_decay_steps", 0) or 0))
    if decay_steps > 0:
        print(
            "[PPO] Reward shaping decay: "
            f"coef {float(getattr(cfg, 'reward_shaping_coef_start', 1.0)):.3f} -> "
            f"{float(getattr(cfg, 'reward_shaping_coef_end', 1.0)):.3f} "
            f"over {decay_steps:,} steps before RewardConfig weighting/scaling."
        )
    if cfg.mode == TrainMode.OPPONENT_POOL.value or bool(getattr(cfg, "opponent_randomize", False)):
        label = "OPPONENT_POOL mode" if cfg.mode == TrainMode.OPPONENT_POOL.value else "opponent_randomize flag"
        print(
            "[PPO] Opponent randomization: enabled "
            f"({label}; uniform per completed episode over pool={list(cfg.opponent_pool)}; "
            "pre-reset hook — opponent logged for each episode is the one played during that episode)."
        )
    if curriculum is not None:
        print(
            "[PPO] Jacob paper curriculum: enabled "
            "(SCRIPTED:OP1 -> SCRIPTED:OP2 -> SCRIPTED:OP3; scripted-only curriculum)."
        )
        print(
            "[PPO] Curriculum gates: "
            f"min_episodes={curriculum.config.min_episodes}, "
            f"min_winrate={curriculum.config.min_winrate}, "
            f"windows={curriculum.config.winrate_window_by_phase}."
        )
    if bool(cfg.use_latent_strategy):
        interval = int(getattr(cfg, "latent_resample_every_n", 0) or 0)
        fixed = bool(getattr(cfg, "fixed_latent_strategy", False))
        interval_label = "fixed" if fixed else ("episode start" if interval <= 0 else f"every {interval} decision steps")
        on_flag = bool(getattr(cfg, "latent_resample_on_flag", False)) and not fixed
        lam_kl = 0.0 if fixed else float(getattr(cfg, "latent_kl_consecutive", 0.0) or 0.0)
        fixed_label = f", fixed_z={int(getattr(cfg, 'fixed_latent_strategy_id', 0) or 0)}" if fixed else ""
        h_obj = getattr(cfg, "latent_entropy_objective", "maximize") or "maximize"
        aux_head = bool(getattr(cfg, "latent_strategy_aux_return_head", False))
        episode_credit = bool(getattr(cfg, "latent_episode_strategy_ppo", False))
        print(
            "[PPO] Latent team strategy: enabled "
            f"(K={int(cfg.latent_k)}, sample={interval_label}, on_flag={on_flag}, "
            f"lambda_p={float(cfg.latent_lam_p):.4f}, lambda_H={float(cfg.latent_lam_h):.4f} "
            f"(H:{h_obj}), "
            f"lambda_KL={lam_kl:.4f}, strategy_ppo_coef={float(cfg.latent_strategy_ppo_coef):.3f}, "
            f"episode_credit={episode_credit}, episode_coef={float(getattr(cfg, 'latent_episode_strategy_coef', 0.0)):.3f}, "
            f"aux_return_head={aux_head}, aux_return_coef={float(cfg.latent_strategy_aux_return_coef):.3f}, "
            f"tau={float(cfg.latent_strategy_tau):.3f}, "
            f"GAE_reset_on_z_change={bool(getattr(cfg, 'latent_gae_reset_on_z_change', True))}, "
            f"bootstrap_z_deterministic={bool(getattr(cfg, 'latent_bootstrap_z_deterministic', True))}"
            f"{fixed_label})"
        )
        if (not fixed) and interval <= 0 and float(getattr(cfg, "latent_lam_p", 0.0) or 0.0) > 0.0:
            print(
                "[PPO] NOTE: latent_lam_p is active only on sparse mid-episode resamples; "
                "with sample=episode start it has near-zero training effect."
            )
        if fixed:
            print("[PPO] Fixed-latent baseline: q_phi sampling/losses are bypassed; actor/critic receive one z ID.")
    else:
        print("[PPO] Latent team strategy: disabled (vanilla local PPO baseline).")
    print(f"[PPO] Checkpoint dir: {cfg.checkpoint_dir}")
    if bool(getattr(cfg, "enable_metrics_csv", True)):
        if not cfg.metrics_csv_path:
            cfg.metrics_csv_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_metrics.csv")
        if not cfg.episode_csv_path:
            cfg.episode_csv_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_episodes.csv")
        strategy_experience_enabled = bool(getattr(cfg, "use_latent_strategy", False)) and bool(
            getattr(cfg, "latent_episode_strategy_ppo", False)
        )
        if strategy_experience_enabled and not cfg.strategy_experience_csv_path:
            cfg.strategy_experience_csv_path = os.path.join(
                cfg.checkpoint_dir, f"{cfg.run_tag}_strategy_experience.csv"
            )
        elif not strategy_experience_enabled:
            cfg.strategy_experience_csv_path = None
        print(f"[PPO] Update metrics CSV: {cfg.metrics_csv_path}")
        print(f"[PPO] Episode metrics CSV: {cfg.episode_csv_path}")
        if strategy_experience_enabled and cfg.strategy_experience_csv_path:
            print(f"[PPO] Strategy experience CSV: {cfg.strategy_experience_csv_path}")
        _e3p = str(getattr(cfg, "e3_step_telemetry_path", "") or "").strip()
        if _e3p:
            print(f"[PPO] E3 step telemetry CSV (per-step z, team_phase, behavior telemetry, buckets, MI-related fields): {_e3p}")
        if (not cfg.fresh_metrics_csv) and (not cfg.load_path) and (
            _metrics_csv_nonempty(cfg.metrics_csv_path)
            or _metrics_csv_nonempty(cfg.episode_csv_path)
            or _metrics_csv_nonempty(cfg.strategy_experience_csv_path)
        ):
            print(
                "[PPO] WARNING: metrics/episode/strategy-experience CSV already exists; this run will APPEND. "
                "That duplicates `timestep`/update indices if you reused --run-tag. "
                "Use --fresh-metrics-csv (rotates old files aside) or a new --run-tag."
            )
    else:
        cfg.metrics_csv_path = None
        cfg.episode_csv_path = None
        cfg.strategy_experience_csv_path = None
        print("[PPO] Metrics CSV logging disabled.")
    run_lock = _acquire_run_lock(cfg)
    if bool(getattr(cfg, "enable_metrics_csv", True)) and cfg.fresh_metrics_csv:
        _rotate_csv_aside(cfg.metrics_csv_path, label="metrics")
        _rotate_csv_aside(cfg.episode_csv_path, label="episode")
        _rotate_csv_aside(cfg.strategy_experience_csv_path, label="strategy experience")
    try:
        rc_path = write_run_config_json(cfg)
        print(f"[PPO] Run config written: {rc_path}")
    except Exception as exc:
        print(f"[PPO] WARNING: could not write run config JSON: {exc}")
    elog = int(getattr(cfg, "episode_log_every", 0) or 0)
    if elog > 0:
        mode_label = "curriculum phase" if curriculum is not None else "scripted opponent tag"
        if curriculum is not None:
            tag_label = initial_opponent_tag
        elif cfg.mode == TrainMode.OPPONENT_POOL.value or bool(getattr(cfg, "opponent_randomize", False)):
            tag_label = f"randomized pool {list(cfg.opponent_pool)}"
        else:
            tag_label = str(cfg.fixed_opponent_tag).upper()
        print(
            f"[PPO] Episode stats: every {elog} completed episode(s) print W/L/D and WR "
            f"(mode={cfg.mode}, {mode_label}={tag_label})."
        )
    else:
        print("[PPO] Episode stats logging disabled (episode_log_every=0).")

    if max_agents == 6:
        cfg.n_envs = min(int(cfg.n_envs), 1)
        cfg.n_steps = min(int(cfg.n_steps), 512)
        cfg.max_decision_steps = min(int(cfg.max_decision_steps), 400)

    if str(cfg.device).lower().startswith("cuda"):
        try:
            torch.zeros(1, device=cfg.device)
        except RuntimeError as exc:
            print(f"[PPO] CUDA unavailable for this torch build ({exc}). Falling back to CPU.")
            cfg.device = "cpu"

    reward_kw = _gpu_env_reward_kwargs(cfg)
    if reward_kw:
        parts = [f"{k}={v}" for k, v in sorted(reward_kw.items())]
        print("[PPO] GPU env reward overrides: " + ", ".join(parts))
    gpu_cfg = GPUFieldConfig(
        n_envs=max(1, int(cfg.n_envs)),
        n_agents_per_team=max_agents,
        map_set=str(getattr(cfg, "map_set", "train")).lower(),
        max_decision_steps=max(1, int(cfg.max_decision_steps)),
        aquaticus_profile=True,
        rules_profile="OURS",
        device=str(cfg.device),
        seed=int(cfg.seed),
        train_domain_randomization=bool(getattr(cfg, "train_domain_randomization", False)),
        dr_sensor_noise_sigma_max=float(getattr(cfg, "dr_sensor_noise_sigma_max", 0.12)),
        dr_sensor_dropout_max=float(getattr(cfg, "dr_sensor_dropout_max", 0.08)),
        dr_blue_speed_jitter=float(getattr(cfg, "dr_blue_speed_jitter", 0.12)),
        **reward_kw,
    )
    print(
        "[PPO] Trainer reward target mirrors GPU RewardConfig: "
        f"dense_weight={float(gpu_cfg.dense_weight):.3f}, "
        f"reward_scale={float(gpu_cfg.reward_scale):.3f}, "
        f"reward_clip={float(gpu_cfg.reward_clip):.3f}, "
        f"stalemate_penalty={float(gpu_cfg.stalemate_penalty):.3f}."
    )
    env = GPUCTFVecEnv(gpu_cfg)
    try:
        env.env_method("set_stress_schedule", STRESS_BY_PHASE)
        env.env_method("set_dynamics_config", {"rules_profile": "OURS"})
        env.env_method("set_phase", initial_phase)
        env.env_method("set_next_opponent", "SCRIPTED", initial_opponent_tag)
        _apply_initial_opponent_params(env, cfg, gpu_cfg, opponent_tag=initial_opponent_tag, phase=initial_phase)

        learning_rate = float(cfg.learning_rate)
        ent_coef = float(cfg.ent_coef)
        clip_range = float(cfg.clip_range)
        n_epochs = int(cfg.n_epochs)
        batch_size = int(cfg.batch_size)

        if bool(getattr(cfg, "use_stable_marl_ppo", False)):
            learning_rate = 1.5e-4
            ent_coef = 0.005
            clip_range = 0.10
            n_epochs = 2
            batch_size = 1024
            print(
                "[PPO] Optional stable-MARL override (use_stable_marl_ppo=True; not in Word spec): "
                "lr=1.5e-4, n_epochs=2, clip=0.10, ent=0.005, batch_size=1024."
            )
        if max_agents > 2:
            learning_rate *= 0.75
            print(f"[PPO] {team_size}: using lr={learning_rate:.2e} for stability.")

        rollout_size = max(1, int(cfg.n_steps) * max(1, int(cfg.n_envs)))
        if batch_size > rollout_size:
            batch_size = rollout_size
            print(f"[PPO] Adjusting batch_size to rollout size: {batch_size}.")

        trainer = CustomPPOTrainer(
            env,
            cfg,
            learning_rate=learning_rate,
            clip_range=clip_range,
            ent_coef=ent_coef,
            n_epochs=n_epochs,
            batch_size=batch_size,
            value_clip_range=getattr(cfg, "clip_range_vf", clip_range),
            curriculum=curriculum,
        )
        trainer.log_input_dim_contract()
        if cfg.load_path and os.path.isfile(cfg.load_path):
            print(f"[PPO] Resuming checkpoint: {cfg.load_path}")
            trainer.load(cfg.load_path)
        try:
            stats = trainer.learn(total_timesteps=int(cfg.total_timesteps))
        except KeyboardInterrupt:
            interrupt_path = os.path.join(
                cfg.checkpoint_dir,
                f"interrupt_{cfg.run_tag}_{int(getattr(trainer, 'global_step', 0))}.zip",
            )
            trainer.save(interrupt_path)
            print(f"[PPO] KeyboardInterrupt: emergency checkpoint saved to: {interrupt_path}")
            raise
        final_path = os.path.join(cfg.checkpoint_dir, f"final_{cfg.run_tag}.zip")
        trainer.save(final_path)
        if stats:
            print(
                "[PPO] Final stats: "
                f"policy_loss={stats.get('policy_loss', 0.0):.4f}, "
                f"value_loss={stats.get('value_loss', 0.0):.4f}, "
                f"approx_kl={stats.get('approx_kl', 0.0):.5f}"
            )
        print(f"[PPO] Training complete. Final checkpoint saved to: {final_path}")
    finally:
        env.close()
        run_lock.release()


def _apply_initial_opponent_params(
    env: GPUCTFVecEnv,
    cfg: PPOConfig,
    gpu_cfg: GPUFieldConfig,
    *,
    opponent_tag: str | None = None,
    phase: str | None = None,
) -> None:
    try:
        key = str(opponent_tag or cfg.fixed_opponent_tag).upper()
        phase_key = str(phase or phase_from_tag(key)).upper()
        opp_params = sample_batched_opponent_params(
            kind="SCRIPTED",
            key=key,
            phase=phase_key,
            n_agents=gpu_cfg.max_red_agents,
            batch_size=gpu_cfg.n_envs,
            device=gpu_cfg.device,
        )
        dyn_cfg: dict[str, object] = {
            key: value
            for key, value in opp_params.items()
            if key
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
        print(f"[PPO] opponent_params sampling failed; using defaults: {exc}")


def run_verify_4v4(num_episodes: int = 10) -> None:
    """Run random-action verification episodes at 4v4."""
    set_global_seed(42)
    cfg = GPUFieldConfig(n_envs=1, n_agents_per_team=4, max_decision_steps=400, device="cpu", seed=42)
    env = GPUCTFVecEnv(cfg)
    try:
        for ep in range(num_episodes):
            env.reset()
            done = False
            steps = 0
            while not done and steps < 800:
                env.step_async(np.asarray(env.action_space.sample(), dtype=np.int64)[None, :])
                _, _, done_arr, _ = env.step_wait()
                done = bool(done_arr[0])
                steps += 1
            print(f"[Verify-4v4] episode {ep + 1}/{num_episodes} steps={steps} done={done}")
    finally:
        env.close()


def run_test_vec_schema() -> None:
    """Verify GPU core observation and global-state schemas."""
    cfg = GPUFieldConfig(n_envs=1, n_agents_per_team=2, device="cpu", seed=42)
    env = GPUCTFVecEnv(cfg)
    try:
        obs = env.reset()
        vec = obs["vec"]
        state = env.state()
        assert vec.dtype == np.float32, f"vec.dtype {vec.dtype}, expected float32"
        assert vec.ndim == 3 and vec.shape[2] == VEC_OBS_DIM, (
            f"vec.shape {vec.shape}, expected (B,N,{VEC_OBS_DIM})"
        )
        assert np.all(np.isfinite(vec)), "vec has non-finite values"
        assert state.shape == (1, GLOBAL_STATE_DIM), f"state.shape {state.shape}"
        print("[test-vec-schema] obs vec and global state schemas OK.")
    finally:
        env.close()


def _agents_suffix(n_agents: int) -> str:
    n = max(1, min(int(n_agents), 16))
    return f"{n}v{n}"


def _ensure_run_tag_has_agent_suffix(run_tag: str, n_agents: int) -> str:
    suffix = _agents_suffix(n_agents)
    tag_suffix = f"_{suffix}"
    for existing in ("_2v2", "_4v4", "_6v6", "_8v8"):
        if run_tag.endswith(existing):
            run_tag = run_tag[: -len(existing)]
            break
    if not run_tag.endswith(tag_suffix):
        run_tag = run_tag.rstrip("_") + tag_suffix
    return run_tag


def _normalize_train_mode(mode: str) -> str:
    raw = str(mode).upper().strip()
    aliases = {
        "FIXED": TrainMode.FIXED_OPPONENT.value,
        "FIXED_OPPONENT": TrainMode.FIXED_OPPONENT.value,
        "OPPONENT_POOL": TrainMode.OPPONENT_POOL.value,
        "POOL": TrainMode.OPPONENT_POOL.value,
        "OPPONENT_RANDOM": TrainMode.OPPONENT_POOL.value,
        "RANDOM_OPPONENT": TrainMode.OPPONENT_POOL.value,
        "PAPER": TrainMode.CURRICULUM.value,
        "NO_LEAGUE": TrainMode.CURRICULUM.value,
        "CURRICULUM": TrainMode.CURRICULUM.value,
        "CURRICULUM_NO_LEAGUE": TrainMode.CURRICULUM.value,
        "JACOB": TrainMode.CURRICULUM.value,
    }
    removed = {"LEAGUE", "CURRICULUM_LEAGUE", "SELF_PLAY"}
    if raw in removed:
        print(f"[PPO] Train mode {raw!r} is not in the local PPO audit path; using FIXED_OPPONENT.")
        return TrainMode.FIXED_OPPONENT.value
    return aliases.get(raw, raw)


def _default_run_tag_for_mode(
    mode: str,
    fixed_opponent_tag: str = "OP3",
    n_agents: int = 2,
    *,
    latent: bool = True,
) -> str:
    suffix = _agents_suffix(n_agents)
    family = "ppo_latent" if latent else "ppo_custom"
    if _normalize_train_mode(mode) == TrainMode.CURRICULUM.value:
        return f"{family}_curriculum_{suffix}"
    if _normalize_train_mode(mode) == TrainMode.OPPONENT_POOL.value:
        return f"{family}_opp_pool_{suffix}"
    if _normalize_train_mode(mode) == TrainMode.FIXED_OPPONENT.value:
        return f"{family}_fixed_{fixed_opponent_tag.lower()}_{suffix}"
    return f"{family}_{suffix}"


if __name__ == "__main__":
    import argparse

    if "--verify-4v4" in sys.argv:
        run_verify_4v4(num_episodes=10)
    elif "--test-vec-schema" in sys.argv:
        run_test_vec_schema()
    else:
        parser = argparse.ArgumentParser(description="Train custom PPO/MAPPO for CTF")
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument(
            "--preset",
            type=str,
            default=DEFAULT_CLI_TRAINING_PRESET,
            help=(
                f"Apply a named training preset before other CLI overrides (default: {DEFAULT_CLI_TRAINING_PRESET!r}). "
                "Use 'none' or '' to skip presets and use PPOConfig + CLI fields only. "
                "Examples: plan_faithful_latent_persist_entropy (recommended), "
                "latent_op3_wrmax_1m (drift wrmax), latent_a1_plan_faithful (legacy A1)."
            ),
        )
        parser.add_argument(
            "--mode",
            type=str,
            default=None,
            help="Training mode: FIXED_OPPONENT, OPPONENT_POOL (uniform --opponent-pool per episode), or CURRICULUM.",
        )
        parser.add_argument("--run-tag", type=str, default=None)
        parser.add_argument("--total-steps", type=int, default=None)
        parser.add_argument("--checkpoint-dir", type=str, default=None)
        parser.add_argument("--metrics-csv", type=str, default=None, help="Path for per-update training metrics CSV.")
        parser.add_argument("--episode-csv", type=str, default=None, help="Path for per-episode training outcome CSV.")
        parser.add_argument(
            "--strategy-experience-csv",
            type=str,
            default=None,
            help="Path for latent bucket/z return diagnostics CSV.",
        )
        parser.add_argument("--no-metrics-csv", action="store_true", help="Disable training CSV telemetry.")
        parser.add_argument(
            "--fresh-metrics-csv",
            action="store_true",
            help="Rotate aside existing metrics/episode CSVs for this run_tag so telemetry is not appended.",
        )
        parser.add_argument("--load", type=str, default=None)
        parser.add_argument("--learning-rate", type=float, default=None, help="PPO learning rate.")
        parser.add_argument(
            "--lr-floor-frac",
            type=float,
            default=None,
            help="Minimum fraction of the base learning rate used by the linear schedule.",
        )
        parser.add_argument("--target-kl", type=float, default=None, help="PPO target KL; negative disables early stopping.")
        parser.add_argument("--n-epochs", type=int, default=None, help="Number of PPO optimization epochs per rollout.")
        parser.add_argument(
            "--n-envs",
            type=int,
            default=None,
            help="Number of vectorized GPU env instances (default: PPOConfig.n_envs=8). Increase to keep the GPU busy.",
        )
        parser.add_argument(
            "--n-steps",
            type=int,
            default=None,
            help="Rollout length per env between PPO updates (default: PPOConfig.n_steps=2048).",
        )
        parser.add_argument(
            "--clip-range-vf",
            type=float,
            default=None,
            help="Value-function clip range; negative disables value clipping.",
        )
        parser.add_argument("--vf-coef", type=float, default=None, help="Value-function loss coefficient.")
        parser.add_argument(
            "--return-normalization",
            action="store_true",
            help="Train critic on normalized returns while denormalizing values for GAE/advantages.",
        )
        parser.add_argument("--fixed-opponent", type=str, default="OP3")
        parser.add_argument(
            "--opponent-randomize",
            action="store_true",
            help="Each finished episode samples next scripted opponent uniformly from --opponent-pool (FIXED_OPPONENT or OPPONENT_POOL).",
        )
        parser.add_argument(
            "--opponent-pool",
            nargs="+",
            default=None,
            metavar="TAG",
            help=(
                "Scripted tags when --opponent-randomize or mode=OPPONENT_POOL (config default: OP1 OP2 OP3). "
                "OP4 is removed from training pools unless --allow-op4-in-training-pool is set. "
                "OP5_RUSHER (alias OP5) is a fast-rush stress test; OP6 / OP6_TURTLE is a defensive turtle; "
                "OP7 / OP7_SWITCHER is a trainable deceptive multi-profile scripted opponent."
            ),
        )
        parser.add_argument(
            "--allow-op4-in-training-pool",
            action="store_true",
            help="Allow OP4 in opponent_pool during training (default: OP4 is eval-only and stripped).",
        )
        parser.add_argument(
            "--e3-step-telemetry",
            action="store_true",
            help=(
                "Latent only: write per-decision-step CSV (z, entropy, argmax, switched, game_phase) next to metrics "
                "for E2 / strategy-switch analysis (path: <checkpoint-dir>/<run-tag>_e3_steps.csv)."
            ),
        )
        parser.add_argument("--map-set", type=str, choices=["train", "eval"], default=None)
        parser.add_argument("--agents", type=int, choices=[2, 4, 6, 8], default=None)
        parser.add_argument("--max-blue-agents", type=int, default=None)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument("--deterministic", action="store_true")
        parser.add_argument("--verbose-training", action="store_true")
        parser.add_argument(
            "--stable-marl",
            action="store_true",
            help="Enable stable-MARL PPO hyperparameter override (not defined in Summer Implementation Plan.docx).",
        )
        parser.add_argument(
            "--latent-strategy",
            action="store_true",
            help="Enable latent team strategy training (default in PPOConfig).",
        )
        parser.add_argument(
            "--no-latent-strategy",
            action="store_true",
            help="Disable latent team strategy for vanilla local PPO ablations.",
        )
        parser.add_argument("--latent-k", type=int, default=None, help="Number of discrete team strategies.")
        parser.add_argument(
            "--latent-resample-every",
            type=int,
            default=None,
            help="Sparse strategy refresh interval in decision steps; 0 samples at episode start only.",
        )
        parser.add_argument(
            "--fixed-latent-strategy",
            action="store_true",
            help="Baseline: keep latent actor/critic inputs but clamp all rollouts to one fixed z ID.",
        )
        parser.add_argument(
            "--fixed-latent-id",
            type=int,
            default=None,
            help="Strategy ID used by --fixed-latent-strategy (default: 0). Supplying this implies fixed latent.",
        )
        parser.add_argument("--latent-lam-p", type=float, default=None, help="Strategy persistence penalty weight.")
        parser.add_argument("--latent-lam-h", type=float, default=None, help="Strategy entropy weight (see --latent-entropy-objective).")
        parser.add_argument(
            "--latent-strategy-ppo-coef",
            type=float,
            default=None,
            help="Coefficient for the sampled-z clipped PPO strategy loss.",
        )
        parser.add_argument(
            "--latent-episode-strategy-ppo",
            action="store_true",
            help="Enable Option A episode-level PPO credit for q_phi's sampled episode-start z.",
        )
        parser.add_argument(
            "--latent-episode-strategy-coef",
            type=float,
            default=None,
            help="Weight for episode-level q_phi PPO credit (recommended sweep: 0.25, 0.5, 1.0).",
        )
        parser.add_argument(
            "--latent-episode-strategy-clip-eps",
            type=float,
            default=None,
            help="Clip epsilon for episode-level q_phi PPO credit.",
        )
        parser.add_argument(
            "--latent-episode-strategy-value-coef",
            type=float,
            default=None,
            help="Value baseline coefficient for episode-level q_phi PPO credit.",
        )
        parser.add_argument(
            "--no-latent-episode-strategy-return-norm",
            action="store_true",
            help="Disable rollout-normalized episode-level q_phi advantages.",
        )
        parser.add_argument(
            "--latent-strategy-aux-return-head",
            action="store_true",
            help="Enable A2 auxiliary per-z return regression on the shared q_phi trunk (sampled z only; not Q-learning).",
        )
        parser.add_argument(
            "--latent-strategy-q-head",
            action="store_true",
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--latent-strategy-aux-return-coef",
            type=float,
            default=None,
            help="Weight for the A2 auxiliary return MSE on resampled-z minibatches.",
        )
        parser.add_argument(
            "--latent-strategy-q-coef",
            type=float,
            default=None,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--latent-strategy-aux-predict-phase-coef",
            type=float,
            default=None,
            help="Weight for the optional phase prediction auxiliary loss.",
        )
        parser.add_argument(
            "--latent-strategy-tau",
            type=float,
            default=None,
            help="Softmax temperature for strategy logits when the auxiliary return head shares the trunk.",
        )
        parser.add_argument(
            "--latent-entropy-objective",
            type=str,
            choices=("maximize", "minimize", "none"),
            default=None,
            help="How λ_H shapes H(q_phi): maximize=paper bonus on entropy; minimize=penalty (sharper z); none=off.",
        )
        parser.add_argument(
            "--latent-resample-on-flag",
            action="store_true",
            help="Resample z when the global-state flag/territory features change (optional plan §12).",
        )
        parser.add_argument(
            "--latent-kl-consecutive",
            type=float,
            default=None,
            help="Weight for consecutive-step KL on q_phi logits (0=off; optional plan §12).",
        )
        parser.add_argument(
            "--no-latent-gae-z-reset",
            action="store_true",
            help="Keep legacy GAE: carry λ-returns across z switches (can smear credit when V(s,z) jumps).",
        )
        parser.add_argument(
            "--latent-bootstrap-z-stochastic",
            action="store_true",
            help="Sample z for V(s') bootstrap; default argmax to avoid RNG mismatch with the next step.",
        )
        parser.add_argument(
            "--domain-randomization",
            action="store_true",
            help="Enable episode-level domain randomization (obs noise/dropout, blue speed jitter).",
        )
        parser.add_argument(
            "--dr-sensor-noise-max",
            type=float,
            default=None,
            help="Max enemy obs position noise (map cells) per episode when domain randomization is on.",
        )
        parser.add_argument(
            "--dr-sensor-dropout-max",
            type=float,
            default=None,
            help="Max in-range enemy dropout probability per episode when domain randomization is on.",
        )
        parser.add_argument(
            "--dr-blue-speed-jitter",
            type=float,
            default=None,
            help="Blue max-speed scale draws from U(1-j,1) per episode (slowdown-only; marine speed cap unchanged).",
        )
        parser.add_argument(
            "--latent-z-embed-dim",
            type=int,
            default=None,
            help="Strategy embedding dimension used by the shared actor.",
        )
        parser.add_argument(
            "--latent-vf-hidden",
            type=int,
            default=None,
            help="Hidden width for the centralized latent critic.",
        )
        parser.add_argument(
            "--episode-log-every",
            type=int,
            default=None,
            metavar="N",
            help="Log W/L/D every N completed episodes (0=off; default from PPOConfig).",
        )
        parser.add_argument(
            "--env-win-reward",
            type=float,
            default=None,
            help="Override GPU env terminal win_team_reward (training-only shaping).",
        )
        parser.add_argument(
            "--env-draw-penalty",
            type=float,
            default=None,
            help="Override GPU env terminal draw_team_penalty.",
        )
        parser.add_argument(
            "--env-lose-penalty",
            type=float,
            default=None,
            help="Override GPU env terminal lose_team_punish.",
        )
        parser.add_argument(
            "--env-action-failed-penalty",
            type=float,
            default=None,
            help="Override GPU env action_failed_punishment; useful when macro failure noise dominates outcomes.",
        )
        parser.add_argument(
            "--env-dense-weight",
            type=float,
            default=None,
            help="Scale dense PBRS/team shaping vs sparse/terminal (GPU RewardConfig.dense_weight).",
        )
        parser.add_argument(
            "--env-sparse-weight",
            type=float,
            default=None,
            help="Sparse event weight before /100 normalization (GPU RewardConfig.sparse_weight).",
        )
        parser.add_argument(
            "--env-reward-scale",
            type=float,
            default=None,
            help="Denominator inside tanh(raw/scale) before reward_clip (GPU RewardConfig.reward_scale).",
        )
        parser.add_argument(
            "--env-reward-clip",
            type=float,
            default=None,
            help="Clamp bound on scaled per-step reward (GPU RewardConfig.reward_clip).",
        )
        parser.add_argument(
            "--env-stalemate-penalty",
            type=float,
            default=None,
            help="Extra penalty when stalemate truncation fires.",
        )
        parser.add_argument(
            "--env-stalemate-max-steps",
            type=int,
            default=None,
            help="Consecutive low-progress steps before stalemate truncation (per env).",
        )
        parser.add_argument(
            "--reward-shaping-coef-start",
            type=float,
            default=None,
            help="Initial multiplier applied to (offense+pbrs+team) in PPO training rewards.",
        )
        parser.add_argument(
            "--reward-shaping-coef-end",
            type=float,
            default=None,
            help="Final multiplier after --reward-shaping-decay-steps.",
        )
        parser.add_argument(
            "--reward-shaping-decay-steps",
            type=int,
            default=None,
            help="Linear decay horizon for reward shaping coefficient; 0 disables.",
        )
        parser.add_argument(
            "--periodic-checkpoint-steps",
            type=int,
            default=None,
            help="Save checkpoint every N env steps during training (0 disables).",
        )
        parser.add_argument(
            "--no-progress-bar",
            action="store_true",
            help="Disable the SB3-style tqdm rollout bar (default: on; uses tqdm.rich if installed).",
        )
        args = parser.parse_args()

        preset_key = str(args.preset or "").strip()
        if preset_key.lower() in {"", "none"}:
            preset_key = ""

        cfg = PPOConfig()
        if preset_key:
            cfg = _apply_training_preset(cfg, preset_key)
        if args.mode is not None:
            cfg.mode = _normalize_train_mode(args.mode)
        if args.seed is not None:
            cfg.seed = int(args.seed)
        if args.max_blue_agents is not None:
            cfg.max_blue_agents = max(1, min(int(args.max_blue_agents), 16))
        elif args.agents is not None:
            cfg.max_blue_agents = int(args.agents)
        cfg.fixed_opponent_tag = str(args.fixed_opponent).upper()
        if args.opponent_randomize:
            cfg.opponent_randomize = True
        if getattr(args, "opponent_pool", None):
            cfg.opponent_pool = tuple(str(x).strip().upper() for x in args.opponent_pool if str(x).strip())
        if getattr(args, "allow_op4_in_training_pool", False):
            cfg.allow_op4_in_training_pool = True
        if args.map_set is not None:
            cfg.map_set = str(args.map_set).lower()
        if args.latent_strategy:
            cfg.use_latent_strategy = True
        if args.no_latent_strategy:
            cfg.use_latent_strategy = False
        if args.latent_k is not None:
            cfg.latent_k = max(1, int(args.latent_k))
        if args.latent_resample_every is not None:
            cfg.latent_resample_every_n = max(0, int(args.latent_resample_every))
        if args.fixed_latent_strategy:
            cfg.fixed_latent_strategy = True
        if args.fixed_latent_id is not None:
            cfg.fixed_latent_strategy = True
            cfg.fixed_latent_strategy_id = max(0, int(args.fixed_latent_id))
        if args.latent_lam_p is not None:
            cfg.latent_lam_p = max(0.0, float(args.latent_lam_p))
        if args.latent_lam_h is not None:
            cfg.latent_lam_h = max(0.0, float(args.latent_lam_h))
        if args.latent_strategy_ppo_coef is not None:
            cfg.latent_strategy_ppo_coef = max(0.0, float(args.latent_strategy_ppo_coef))
        if args.latent_episode_strategy_ppo:
            cfg.latent_episode_strategy_ppo = True
        if args.latent_episode_strategy_coef is not None:
            cfg.latent_episode_strategy_coef = max(0.0, float(args.latent_episode_strategy_coef))
        if args.latent_episode_strategy_clip_eps is not None:
            cfg.latent_episode_strategy_clip_eps = max(1e-6, float(args.latent_episode_strategy_clip_eps))
        if args.latent_episode_strategy_value_coef is not None:
            cfg.latent_episode_strategy_value_coef = max(0.0, float(args.latent_episode_strategy_value_coef))
        if args.no_latent_episode_strategy_return_norm:
            cfg.latent_episode_strategy_return_norm = False
        if args.latent_strategy_aux_return_head or bool(getattr(args, "latent_strategy_q_head", False)):
            cfg.latent_strategy_aux_return_head = True
        aux_coef = getattr(args, "latent_strategy_aux_return_coef", None)
        if aux_coef is None and getattr(args, "latent_strategy_q_coef", None) is not None:
            aux_coef = getattr(args, "latent_strategy_q_coef", None)
        if aux_coef is not None:
            cfg.latent_strategy_aux_return_coef = max(0.0, float(aux_coef))
        if args.latent_strategy_tau is not None:
            cfg.latent_strategy_tau = max(1e-3, float(args.latent_strategy_tau))
        if getattr(args, "latent_strategy_aux_predict_phase_coef", None) is not None:
            cfg.latent_strategy_aux_predict_phase_coef = max(0.0, float(args.latent_strategy_aux_predict_phase_coef))
        if args.latent_entropy_objective is not None:
            cfg.latent_entropy_objective = args.latent_entropy_objective  # type: ignore[assignment]
        if args.latent_resample_on_flag:
            cfg.latent_resample_on_flag = True
        if args.latent_kl_consecutive is not None:
            cfg.latent_kl_consecutive = max(0.0, float(args.latent_kl_consecutive))
        if args.no_latent_gae_z_reset:
            cfg.latent_gae_reset_on_z_change = False
        if args.latent_bootstrap_z_stochastic:
            cfg.latent_bootstrap_z_deterministic = False
        if args.domain_randomization:
            cfg.train_domain_randomization = True
        if args.dr_sensor_noise_max is not None:
            cfg.dr_sensor_noise_sigma_max = max(0.0, float(args.dr_sensor_noise_max))
        if args.dr_sensor_dropout_max is not None:
            cfg.dr_sensor_dropout_max = max(0.0, min(1.0, float(args.dr_sensor_dropout_max)))
        if args.dr_blue_speed_jitter is not None:
            cfg.dr_blue_speed_jitter = max(0.0, min(0.75, float(args.dr_blue_speed_jitter)))
        if args.latent_z_embed_dim is not None:
            cfg.latent_z_embed_dim = max(1, int(args.latent_z_embed_dim))
        if args.latent_vf_hidden is not None:
            cfg.latent_vf_hidden = max(1, int(args.latent_vf_hidden))
        # Presets set ``cfg.run_tag``; only overwrite when user supplies --run-tag or no preset was applied.
        if args.run_tag is not None:
            cfg.run_tag = args.run_tag
        elif not preset_key:
            cfg.run_tag = _default_run_tag_for_mode(
                cfg.mode,
                cfg.fixed_opponent_tag,
                cfg.max_blue_agents,
                latent=bool(cfg.use_latent_strategy),
            )
        cfg.run_tag = _ensure_run_tag_has_agent_suffix(cfg.run_tag, cfg.max_blue_agents)
        cfg.checkpoint_dir = args.checkpoint_dir or os.path.join("checkpoints", _agents_suffix(cfg.max_blue_agents))
        if getattr(args, "e3_step_telemetry", False):
            if not cfg.use_latent_strategy:
                print("[PPO] WARNING: --e3-step-telemetry ignored (requires latent strategy).")
            else:
                os.makedirs(cfg.checkpoint_dir, exist_ok=True)
                cfg.e3_step_telemetry_path = os.path.join(cfg.checkpoint_dir, f"{cfg.run_tag}_e3_steps.csv")
        if args.fresh_metrics_csv:
            cfg.fresh_metrics_csv = True
        if args.no_metrics_csv:
            cfg.enable_metrics_csv = False
        if args.metrics_csv is not None:
            cfg.metrics_csv_path = args.metrics_csv
        if args.episode_csv is not None:
            cfg.episode_csv_path = args.episode_csv
        if args.strategy_experience_csv is not None:
            cfg.strategy_experience_csv_path = args.strategy_experience_csv
        if args.total_steps is not None:
            cfg.total_timesteps = int(args.total_steps)
        if args.load is not None:
            cfg.load_path = args.load
        if args.learning_rate is not None:
            cfg.learning_rate = max(0.0, float(args.learning_rate))
        if args.lr_floor_frac is not None:
            cfg.lr_floor_frac = max(0.0, min(float(args.lr_floor_frac), 1.0))
        if args.target_kl is not None:
            cfg.target_kl = None if float(args.target_kl) < 0.0 else max(0.0, float(args.target_kl))
        if args.n_epochs is not None:
            cfg.n_epochs = max(1, int(args.n_epochs))
        if args.n_envs is not None:
            cfg.n_envs = max(1, int(args.n_envs))
        if args.n_steps is not None:
            cfg.n_steps = max(1, int(args.n_steps))
        if args.clip_range_vf is not None:
            cfg.clip_range_vf = None if float(args.clip_range_vf) < 0.0 else max(0.0, float(args.clip_range_vf))
        if args.vf_coef is not None:
            cfg.vf_coef = max(0.0, float(args.vf_coef))
        if args.return_normalization:
            cfg.normalize_returns = True
        if args.device is not None:
            cfg.device = str(args.device).strip().lower()
        if args.deterministic:
            cfg.use_deterministic = True
        if args.verbose_training:
            cfg.verbose_training = True
        if args.stable_marl:
            cfg.use_stable_marl_ppo = True
        if args.episode_log_every is not None:
            cfg.episode_log_every = max(0, int(args.episode_log_every))
        if args.env_win_reward is not None:
            cfg.env_win_team_reward = float(args.env_win_reward)
        if args.env_draw_penalty is not None:
            cfg.env_draw_team_penalty = float(args.env_draw_penalty)
        if args.env_lose_penalty is not None:
            cfg.env_lose_team_punish = float(args.env_lose_penalty)
        if args.env_action_failed_penalty is not None:
            cfg.env_action_failed_punishment = float(args.env_action_failed_penalty)
        if args.env_dense_weight is not None:
            cfg.env_dense_weight = max(0.0, float(args.env_dense_weight))
        if args.env_sparse_weight is not None:
            cfg.env_sparse_weight = max(0.0, float(args.env_sparse_weight))
        if args.env_reward_scale is not None:
            cfg.env_reward_scale = max(1e-6, float(args.env_reward_scale))
        if args.env_reward_clip is not None:
            cfg.env_reward_clip = max(1e-6, float(args.env_reward_clip))
        if args.env_stalemate_penalty is not None:
            cfg.env_stalemate_penalty = float(args.env_stalemate_penalty)
        if args.env_stalemate_max_steps is not None:
            cfg.env_stalemate_max_steps = max(1, int(args.env_stalemate_max_steps))
        if args.reward_shaping_coef_start is not None:
            cfg.reward_shaping_coef_start = float(args.reward_shaping_coef_start)
        if args.reward_shaping_coef_end is not None:
            cfg.reward_shaping_coef_end = float(args.reward_shaping_coef_end)
        if args.reward_shaping_decay_steps is not None:
            cfg.reward_shaping_decay_steps = max(0, int(args.reward_shaping_decay_steps))
        if args.periodic_checkpoint_steps is not None:
            cfg.periodic_checkpoint_steps = max(0, int(args.periodic_checkpoint_steps))
        if args.no_progress_bar:
            cfg.enable_progress_bar = False
        if preset_key:
            cfg.cli_preset = preset_key
            print(f"[PPO] Training preset: {cfg.cli_preset!r}")
            if cfg.load_path:
                print(f"[PPO] Warm-start checkpoint: {cfg.load_path}")
        train_ppo(cfg)
