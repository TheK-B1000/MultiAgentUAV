"""Long-session progress: tqdm bars + durable log/JSON heartbeat.

Use this in multi-hour experiment runners so progress remains visible even when
Cursor/terminal capture is empty or buffered.

Typical usage::

    from experiments.long_session_progress import LongSessionProgress, configure_stdio

    configure_stdio()
    prog = LongSessionProgress(out_dir, name="c2_confirmation")
    prog.log("starting")
    for i in prog.bar(range(n), desc="episodes", unit="ep"):
        ...
        prog.heartbeat(done=i + 1, total=n, phase="EPISODES")

Watch:
  - <out_dir>/<name>_PROGRESS.log
  - <out_dir>/<name>_PROGRESS.json
  - live tqdm bar on stderr (prefer tqdm.rich when installed)
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, TypeVar

T = TypeVar("T")


def configure_stdio() -> None:
    """Force line-buffered stdout/stderr; set PYTHONUNBUFFERED if unset."""
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass


def _tqdm_cls() -> Any:
    try:
        from tqdm import TqdmExperimentalWarning

        warnings.filterwarnings("ignore", category=TqdmExperimentalWarning)
    except Exception:
        pass
    try:
        from tqdm.rich import tqdm  # type: ignore[import-not-found]
    except ImportError:
        from tqdm import tqdm  # type: ignore[import-not-found]
    return tqdm


def _now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


class LongSessionProgress:
    """Stdout + stderr tqdm + durable progress log/JSON."""

    def __init__(
        self,
        out_dir: Path,
        *,
        name: str = "session",
        pid: Optional[int] = None,
        enable_bar: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.log_path = self.out_dir / f"{name}_PROGRESS.log"
        self.json_path = self.out_dir / f"{name}_PROGRESS.json"
        self.pid = int(pid if pid is not None else os.getpid())
        self.enable_bar = bool(enable_bar)
        self.t0 = time.time()
        self.phase = "STARTED"
        self.detail = ""
        self.done = 0
        self.total: Optional[int] = None
        self._log_banner()

    def _log_banner(self) -> None:
        self.log(
            f"[PROGRESS] session={self.name} pid={self.pid} "
            f"log={self.log_path} json={self.json_path}"
        )

    def log(self, msg: str) -> None:
        line = f"{_now_utc()} {msg}"
        print(line, flush=True)
        try:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass

    def heartbeat(
        self,
        *,
        done: Optional[int] = None,
        total: Optional[int] = None,
        phase: Optional[str] = None,
        detail: str = "",
        **extra: Any,
    ) -> None:
        if done is not None:
            self.done = int(done)
        if total is not None:
            self.total = int(total)
        if phase is not None:
            self.phase = phase
        if detail:
            self.detail = detail
        elapsed = time.time() - self.t0
        eta = None
        frac = None
        if self.total and self.done > 0 and self.done <= self.total:
            frac = round(self.done / self.total, 4)
            rate = self.done / max(elapsed, 1e-6)
            if rate > 0:
                eta = round((self.total - self.done) / rate, 1)
        payload = {
            "updated_utc": _now_utc(),
            "pid": self.pid,
            "session": self.name,
            "phase": self.phase,
            "detail": self.detail,
            "done": self.done,
            "total": self.total,
            "frac": frac,
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": eta,
            "progress_log": str(self.log_path),
        }
        payload.update(extra)
        try:
            _atomic_write_json(self.json_path, payload)
        except Exception:
            pass

    def set_phase(self, phase: str, detail: str = "") -> None:
        self.phase = phase
        if detail:
            self.detail = detail
        msg = f"[PHASE] {phase}"
        if detail:
            msg = f"{msg}: {detail}"
        self.log(msg)
        self.heartbeat(phase=phase, detail=detail)

    def bar(
        self,
        iterable: Optional[Iterable[T]] = None,
        *,
        total: Optional[int] = None,
        desc: str = "",
        unit: str = "it",
        leave: bool = True,
    ) -> Iterator[T]:
        """Iterate with a live tqdm bar; also heartbeats durable JSON periodically."""
        items: Iterable[T]
        n_total = total
        if iterable is None:
            if n_total is None:
                raise ValueError("bar() requires iterable or total=")
            items = range(int(n_total))  # type: ignore[assignment]
            n_total = int(n_total)
        else:
            items = iterable
            if n_total is None:
                try:
                    n_total = len(items)  # type: ignore[arg-type]
                except TypeError:
                    n_total = None

        self.total = n_total
        self.heartbeat(done=0, total=n_total, phase=self.phase or desc or "BAR")

        use_tqdm = self.enable_bar
        tqdm = None
        if use_tqdm:
            try:
                tqdm = _tqdm_cls()
            except ImportError:
                self.log("[PROGRESS] tqdm not installed; continuing without bar (pip install tqdm rich)")
                use_tqdm = False

        last_hb = time.time()
        done = 0

        def _tick() -> None:
            nonlocal last_hb, done
            now = time.time()
            # Heartbeat at least every 5s so disk watchers stay fresh during long iters.
            if done == 1 or done == n_total or (now - last_hb) >= 5.0:
                self.heartbeat(done=done, total=n_total, detail=desc or self.detail)
                last_hb = now

        if use_tqdm and tqdm is not None:
            bar_it = tqdm(
                items,
                total=n_total,
                desc=desc or self.name,
                unit=unit,
                dynamic_ncols=True,
                file=sys.stderr,
                mininterval=0.25,
                leave=leave,
            )
            for item in bar_it:
                yield item
                done += 1
                _tick()
        else:
            for item in items:
                yield item
                done += 1
                _tick()
                if n_total and (done == 1 or done == n_total or done % max(1, n_total // 20) == 0):
                    pct = 100.0 * done / n_total
                    self.log(f"[PROGRESS] {desc or self.name} {done}/{n_total} ({pct:.1f}%)")

        self.heartbeat(done=done, total=n_total, detail=f"{desc or self.name} complete")
