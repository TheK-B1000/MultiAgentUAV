"""v4i3: Periodic Return-Ranked Router Distillation hook.

This hook is invoked by the trainer **after** a periodic checkpoint save.
When enabled, it spawns two pre-existing offline tools as subprocesses
against the just-saved checkpoint and hot-swaps the resulting distilled
``strategy_encoder.*`` (q_phi) weights into the running model:

  1. ``tools/q_probe.py``                  -- matched-start return contrast +
                                              context capture (``--save-contexts``).
  2. ``tools/router_distill_from_qprobe.py`` -- offline KL distillation of
                                              q_phi from the q_probe returns.

The whole hook is **best-effort**: any subprocess failure or hot-swap
failure is logged and PPO training continues with the pre-distill
weights. v4i3 is scoped to make q_phi catch up to the proven latent
modes from v4i1 -- it does not touch the actor, critic, reward,
opponent pool, maps, arc-credit math, entropy schedule, or the PPO
loop. See ``rl/config/ppo_config.py`` for the ``latent_router_distill_*``
knobs.

Lifecycle (per trigger at global_step >= self._next_step):

  trainer.save(ckpt) -- already happened
  ----
  q_probe subprocess               --> <step_dir>/<run_tag>_qprobe.csv
                                      <step_dir>/<run_tag>_qprobe_contexts.npz
                                      <step_dir>/<run_tag>_qprobe_report.md
  router_distill subprocess        --> <step_dir>/distilled.zip
                                      <step_dir>/distilled_distill_metrics.csv
                                      <step_dir>/distilled_distill_report.md
  load distilled strategy_encoder.* into trainer.model
  reset Adam moments for those params in trainer.optimizer and
    (if present) trainer.latent_router_optimizer
  ----
  PPO update continues normally
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import torch


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_THIS_FILE = Path(__file__).resolve()
# rl/custom_ppo/router_distill_hook.py  -> AICTFProject/
_PROJECT_ROOT = _THIS_FILE.parent.parent.parent
_Q_PROBE_SCRIPT = _PROJECT_ROOT / "tools" / "q_probe.py"
_ROUTER_DISTILL_SCRIPT = _PROJECT_ROOT / "tools" / "router_distill_from_qprobe.py"


# ---------------------------------------------------------------------------
# Hook
# ---------------------------------------------------------------------------

class PeriodicRouterDistillHook:
    """Owns the v4i3 cadence counter and orchestrates one distill round.

    Construction reads the ``latent_router_distill_*`` knobs from ``cfg``
    once. ``maybe_run`` is intended to be called from the trainer right
    after each ``self.save(ckpt_path)`` in the periodic checkpoint loop.
    """

    def __init__(
        self,
        *,
        cfg: Any,
        run_tag: str,
        checkpoint_dir: Path | str,
        first_distill_step: int | None = None,
        python_executable: str | None = None,
    ) -> None:
        self.enabled: bool = bool(getattr(cfg, "latent_router_distill_enabled", False))
        self.every_n_steps: int = max(
            1, int(getattr(cfg, "latent_router_distill_every_n_steps", 250_000))
        )
        self.n_seeds: int = max(
            1, int(getattr(cfg, "latent_router_distill_n_seeds", 8))
        )
        self.base_seed: int = int(getattr(cfg, "latent_router_distill_base_seed", 1000))
        opps = tuple(
            str(o).strip().upper()
            for o in (
                getattr(cfg, "latent_router_distill_opponents", ("OP5", "OP6", "OP7"))
                or ("OP5", "OP6", "OP7")
            )
            if str(o).strip()
        )
        self.opponents: tuple[str, ...] = opps or ("OP5", "OP6", "OP7")
        self.epochs: int = max(1, int(getattr(cfg, "latent_router_distill_epochs", 100)))
        self.lr: float = float(getattr(cfg, "latent_router_distill_lr", 1e-4) or 1e-4)
        self.temperature: float = float(
            getattr(cfg, "latent_router_distill_temperature", 1.0) or 1.0
        )
        self.weight_decay: float = float(
            getattr(cfg, "latent_router_distill_weight_decay", 0.0) or 0.0
        )
        self.distill_device: str = str(
            getattr(cfg, "latent_router_distill_device", "cpu") or "cpu"
        )
        self.artifacts_subdir: str = str(
            getattr(
                cfg, "latent_router_distill_artifacts_subdir", "v4i3_router_distill"
            )
            or "v4i3_router_distill"
        )

        self.run_tag: str = str(run_tag or "v4i3")
        self.checkpoint_dir: Path = Path(checkpoint_dir)
        self.artifacts_root: Path = self.checkpoint_dir / self.artifacts_subdir
        self.python_executable: str = str(python_executable or sys.executable)

        # Cadence pointer: distill at the smallest multiple of
        # ``every_n_steps`` that is >= first_distill_step (defaults to
        # ``every_n_steps`` so the first round happens at step
        # ``every_n_steps``).
        self._next_step: int = (
            int(first_distill_step)
            if first_distill_step is not None
            else int(self.every_n_steps)
        )
        # Counters for logging only.
        self._rounds_completed: int = 0
        self._rounds_failed: int = 0

        if self.enabled:
            self.artifacts_root.mkdir(parents=True, exist_ok=True)
            print(
                "[v4i3] Periodic router distillation ENABLED.\n"
                f"[v4i3]   cadence: every {self.every_n_steps} steps\n"
                f"[v4i3]   probe: n_seeds={self.n_seeds} base_seed={self.base_seed} "
                f"opps={list(self.opponents)} device={self.distill_device}\n"
                f"[v4i3]   distill: epochs={self.epochs} lr={self.lr} "
                f"temperature={self.temperature} weight_decay={self.weight_decay}\n"
                f"[v4i3]   artifacts: {self.artifacts_root}"
            )

    # ------------------------------------------------------------------ public

    def maybe_run(
        self,
        trainer: Any,
        ckpt_path: str | Path,
        global_step: int,
    ) -> None:
        """Entry point invoked by the trainer after each periodic save.

        Triggers exactly when ``global_step >= self._next_step``; the
        cadence pointer is advanced regardless of whether the round
        succeeded so a flaky subprocess does not block the next attempt.
        """
        if not self.enabled:
            return
        if int(global_step) < int(self._next_step):
            return
        triggered_at = int(self._next_step)
        self._next_step += int(self.every_n_steps)
        try:
            self._run_one(trainer, Path(ckpt_path), int(global_step), triggered_at)
            self._rounds_completed += 1
        except Exception as exc:  # noqa: BLE001 -- intentionally broad: best-effort
            self._rounds_failed += 1
            print(
                f"[v4i3] WARNING: router-distill round @ step {global_step} "
                f"(trigger={triggered_at}) failed: {exc!r}. PPO training continues "
                "with pre-distill q_phi weights."
            )

    # ---------------------------------------------------------------- internals

    def _step_dir(self, triggered_at: int) -> Path:
        d = self.artifacts_root / f"step_{int(triggered_at)}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _qprobe_paths(self, step_dir: Path) -> tuple[Path, Path]:
        """Return (csv_path, contexts_npz_path) that q_probe will write."""
        csv = step_dir / f"{self.run_tag}_qprobe.csv"
        npz = step_dir / f"{self.run_tag}_qprobe_contexts.npz"
        return csv, npz

    def _distilled_ckpt_path(self, step_dir: Path) -> Path:
        return step_dir / "distilled.zip"

    def _run_one(
        self,
        trainer: Any,
        ckpt_path: Path,
        global_step: int,
        triggered_at: int,
    ) -> None:
        step_dir = self._step_dir(triggered_at)
        csv_path, npz_path = self._qprobe_paths(step_dir)
        distilled_path = self._distilled_ckpt_path(step_dir)
        log_path = step_dir / "subprocess.log"

        t0 = time.time()
        print(
            f"[v4i3] round start @ step={global_step} (trigger={triggered_at}); "
            f"artifacts -> {step_dir}"
        )

        # --- q_probe ----------------------------------------------------
        # Single-checkpoint mode against the just-saved file; isolates the
        # output filenames inside ``step_dir`` so each round is a clean
        # dataset with no cross-step CSV-resume artifacts.
        self._run_qprobe(
            ckpt_path=ckpt_path,
            out_dir=step_dir,
            log_path=log_path,
        )
        if not csv_path.exists() or not npz_path.exists():
            raise RuntimeError(
                f"q_probe produced no CSV/NPZ. Expected:\n  {csv_path}\n  {npz_path}"
            )
        t_probe = time.time()

        # --- distill ----------------------------------------------------
        self._run_distill(
            ckpt_path=ckpt_path,
            csv_path=csv_path,
            npz_path=npz_path,
            distilled_path=distilled_path,
            log_path=log_path,
        )
        if not distilled_path.exists():
            raise RuntimeError(
                f"router distillation produced no checkpoint: {distilled_path}"
            )
        t_distill = time.time()

        # --- hot swap ---------------------------------------------------
        swap_info = self._hot_swap_strategy_encoder(
            trainer=trainer,
            distilled_path=distilled_path,
        )
        t_swap = time.time()

        print(
            f"[v4i3] round done @ step={global_step}: "
            f"probe={t_probe - t0:.1f}s distill={t_distill - t_probe:.1f}s "
            f"swap={t_swap - t_distill:.2f}s "
            f"(strategy_encoder tensors changed: {swap_info['n_changed']} / "
            f"{swap_info['n_total']}; Adam moments reset for "
            f"{swap_info['n_optimizer_states_reset']} param-tensors)."
        )

    # ----------------------------------------------------------- subprocesses

    def _run_qprobe(
        self,
        *,
        ckpt_path: Path,
        out_dir: Path,
        log_path: Path,
    ) -> None:
        if not _Q_PROBE_SCRIPT.exists():
            raise RuntimeError(f"q_probe script missing: {_Q_PROBE_SCRIPT}")
        cmd: list[str] = [
            self.python_executable,
            str(_Q_PROBE_SCRIPT),
            "--checkpoint", str(ckpt_path),
            "--run-tag", str(self.run_tag),
            "--opponents", *list(self.opponents),
            "--n-seeds", str(int(self.n_seeds)),
            "--base-seed", str(int(self.base_seed)),
            "--device", str(self.distill_device),
            "--output-dir", str(out_dir),
            "--save-contexts",
        ]
        self._run_subprocess(cmd, log_path=log_path, label="qprobe")

    def _run_distill(
        self,
        *,
        ckpt_path: Path,
        csv_path: Path,
        npz_path: Path,
        distilled_path: Path,
        log_path: Path,
    ) -> None:
        if not _ROUTER_DISTILL_SCRIPT.exists():
            raise RuntimeError(
                f"router-distill script missing: {_ROUTER_DISTILL_SCRIPT}"
            )
        cmd: list[str] = [
            self.python_executable,
            str(_ROUTER_DISTILL_SCRIPT),
            "--checkpoint", str(ckpt_path),
            "--qprobe-csv", str(csv_path),
            "--contexts", str(npz_path),
            "--out", str(distilled_path),
            "--temperature", str(float(self.temperature)),
            "--epochs", str(int(self.epochs)),
            "--lr", str(float(self.lr)),
            "--weight-decay", str(float(self.weight_decay)),
            "--device", str(self.distill_device),
        ]
        self._run_subprocess(cmd, log_path=log_path, label="distill")

    def _run_subprocess(
        self,
        cmd: Sequence[str],
        *,
        log_path: Path,
        label: str,
    ) -> None:
        # Use a clean PYTHONPATH that prepends the project root so the
        # subprocesses import ``rl.*`` and ``macro_actions`` the same way
        # the trainer does.
        env = os.environ.copy()
        prev_pp = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{_PROJECT_ROOT}{os.pathsep}{prev_pp}" if prev_pp else str(_PROJECT_ROOT)
        )
        header = (
            f"\n========== [v4i3] {label} subprocess @ {time.strftime('%Y-%m-%dT%H:%M:%S')} ==========\n"
            f"$ {' '.join(shlex.quote(c) for c in cmd)}\n"
        )
        with log_path.open("a", encoding="utf-8") as log_fh:
            log_fh.write(header)
            log_fh.flush()
            proc = subprocess.run(
                list(cmd),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=str(_PROJECT_ROOT),
                env=env,
                check=False,
            )
        if proc.returncode != 0:
            tail = self._tail_log(log_path, n_lines=40)
            raise RuntimeError(
                f"{label} subprocess exited with code {proc.returncode}. "
                f"Tail of {log_path}:\n{tail}"
            )

    @staticmethod
    def _tail_log(path: Path, n_lines: int) -> str:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                lines = fh.readlines()
        except OSError:
            return "(could not read log)"
        return "".join(lines[-int(max(1, n_lines)) :])

    # --------------------------------------------------------------- hot swap

    def _hot_swap_strategy_encoder(
        self,
        *,
        trainer: Any,
        distilled_path: Path,
    ) -> dict[str, int]:
        """Load ``strategy_encoder.*`` from the distilled checkpoint into the
        running model, leaving every other parameter byte-identical.

        Adam moments for the swapped params are dropped from
        ``trainer.optimizer`` and (if present) ``trainer.latent_router_optimizer``
        so the next PPO update does not apply stale momentum to the new
        weights.

        Returns a small diagnostic dict.
        """
        model: torch.nn.Module = trainer.model
        try:
            payload = torch.load(str(distilled_path), map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(str(distilled_path), map_location="cpu")
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise RuntimeError(
                f"Distilled checkpoint malformed (no model_state_dict): {distilled_path}"
            )
        distilled_sd = payload["model_state_dict"]
        strategy_sd: dict[str, torch.Tensor] = {
            k: v for k, v in distilled_sd.items() if k.startswith("strategy_encoder.")
        }
        if not strategy_sd:
            raise RuntimeError(
                "Distilled checkpoint has no ``strategy_encoder.*`` keys; "
                "nothing to hot-swap."
            )

        # Locate the live strategy_encoder params for tensor-equality diffing
        # before/after the partial load.
        live_sd_before = {
            k: v.detach().cpu().clone()
            for k, v in model.state_dict().items()
            if k.startswith("strategy_encoder.")
        }

        # Cast each tensor to the live model's dtype/device, then partial-load.
        live_device = next(model.parameters()).device
        recast: dict[str, torch.Tensor] = {}
        for k, v in strategy_sd.items():
            live_v = live_sd_before.get(k)
            if live_v is None:
                # Distilled key not present in live model (shouldn't happen
                # since the distill was on this exact checkpoint).
                continue
            try:
                recast[k] = v.to(dtype=live_v.dtype, device=live_device)
            except Exception:
                recast[k] = v.to(live_device)
        result = model.load_state_dict(recast, strict=False)
        unexpected = [k for k in getattr(result, "unexpected_keys", []) if k.startswith("strategy_encoder.")]
        if unexpected:
            raise RuntimeError(
                f"Hot-swap rejected unexpected strategy_encoder keys: {unexpected!r}"
            )

        # Count how many tensors actually changed (sanity check + diagnostic).
        live_sd_after = {
            k: v.detach().cpu()
            for k, v in model.state_dict().items()
            if k.startswith("strategy_encoder.")
        }
        n_changed = 0
        for k, before_t in live_sd_before.items():
            after_t = live_sd_after.get(k)
            if after_t is None:
                continue
            if not torch.equal(before_t, after_t):
                n_changed += 1

        # Reset Adam moments for these params on both optimizers.
        n_reset = 0
        for opt_name in ("optimizer", "latent_router_optimizer"):
            opt = getattr(trainer, opt_name, None)
            if opt is None:
                continue
            for group in opt.param_groups:
                for p in group["params"]:
                    # Identify a parameter belonging to strategy_encoder by
                    # checking pointer identity against the live module's
                    # parameters.
                    if _param_is_under_module(
                        p, getattr(model, "strategy_encoder", None)
                    ):
                        if p in opt.state:
                            del opt.state[p]
                            n_reset += 1

        return {
            "n_total": len(live_sd_before),
            "n_changed": int(n_changed),
            "n_optimizer_states_reset": int(n_reset),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _param_is_under_module(
    param: torch.nn.Parameter, module: Optional[torch.nn.Module]
) -> bool:
    if module is None:
        return False
    for p in module.parameters():
        if p is param:
            return True
    return False


__all__ = ["PeriodicRouterDistillHook"]
