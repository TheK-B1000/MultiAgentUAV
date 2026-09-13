r"""Rule 13 -- a lock that can tell a dead owner from a live one.

Motivating failure (2026-09-13): a broken PROJECTED_TEACHER_ORACLE run looked
dead, so its lock file was deleted by hand and a second run was launched. The
first run was NOT dead. Both processes wrote the same rows.csv concurrently,
producing null bytes, a missing header, and rows starting mid-cell -- a
corrupted artifact that gave no error, just quietly wrong data. The fix
that day was "kill by PID, verify, then clean up" done manually; this module
makes that the only way to do it.

THE CONTRACT
    A lock file records enough to answer "is the owner still running" without
    trusting a human's memory: pid, run_id, hostname, start time, and the
    command line. On startup:

        lock exists
            |
            is the recorded PID alive AND does its command line match?
                yes -> REFUSE to launch (a real owner exists)
                no  -> the lock is STALE: archive it (never silently delete),
                       then acquire a fresh one

    Releasing is symmetric: only the process that holds the lock (matching pid)
    may release it during its own normal exit. An externally-initiated stop
    (`stop_and_release`) must send the terminate signal, POLL until the OS
    confirms the process is actually gone, and only then archive the lock --
    never "pkill and hope."
"""

from __future__ import annotations

import json
import os
import platform
import socket
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil


class LockHeld(RuntimeError):
    """Raised when an acquire is refused because a live owner holds the lock."""


class NotOwner(RuntimeError):
    """Raised when release() is called by a process that does not hold the lock."""


class StillAlive(RuntimeError):
    """Raised when stop_and_release() cannot confirm the owner has exited."""


@dataclass
class LockInfo:
    pid: int
    run_id: str
    hostname: str
    start_utc: str
    cmdline: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, s: str) -> "LockInfo":
        return cls(**json.loads(s))


def _now() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cmdline_here() -> str:
    return " ".join(sys.argv)


def _is_same_owner(pid: int, cmdline: str) -> bool:
    """True iff `pid` is alive AND is plausibly running `cmdline` -- not merely
    a live PID that got reused by an unrelated process since the lock was
    written. A reused PID with a completely different command line is treated
    as a dead owner, not a live one."""
    if not psutil.pid_exists(pid):
        return False
    try:
        proc = psutil.Process(pid)
        actual = " ".join(proc.cmdline())
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False
    # loose containment check: enough to catch a recycled PID running something
    # else entirely, without being brittle to quoting/path differences
    key_tokens = [t for t in cmdline.split() if len(t) > 6][:3]
    return all(t in actual for t in key_tokens) if key_tokens else True


class RunLock:
    """One lock file at `path`. Use as a context manager for the common case:

        with RunLock(path, run_id="OPP_ABLATION_6V6") as lock:
            ...  # do the run; lock is released on normal exit, held on crash
    """

    def __init__(self, path: Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self._acquired = False

    def _read(self) -> LockInfo | None:
        if not self.path.is_file():
            return None
        try:
            return LockInfo.from_json(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, TypeError, KeyError):
            return None            # unreadable lock is treated as stale, not fatal

    def acquire(self) -> "RunLock":
        existing = self._read()
        if existing is not None and _is_same_owner(existing.pid, existing.cmdline):
            raise LockHeld(
                f"{self.path.name}: held by a LIVE process -- pid={existing.pid} "
                f"run_id={existing.run_id!r} started {existing.start_utc} on "
                f"{existing.hostname}. Refusing to launch. If you are certain "
                f"that process is not actually running this experiment, use "
                f"stop_and_release() rather than deleting this file by hand.")
        if existing is not None:
            self._archive_stale(existing)
        info = LockInfo(pid=os.getpid(), run_id=self.run_id,
                        hostname=socket.gethostname(), start_utc=_now(),
                        cmdline=_cmdline_here())
        self.path.write_text(info.to_json(), encoding="utf-8")
        self._acquired = True
        return self

    def _archive_stale(self, info: LockInfo) -> None:
        stale = self.path.with_suffix(f".stale-{int(time.time())}.json")
        self.path.rename(stale)
        print(f"[run_lock] STALE lock archived -> {stale.name} "
             f"(recorded pid={info.pid} is not a live match; not the same as "
             f"deleting -- the record is kept)")

    def release(self) -> None:
        """Normal-exit release: only the recorded owner may do this."""
        existing = self._read()
        if existing is None:
            return                 # already gone; release is idempotent
        if existing.pid != os.getpid():
            raise NotOwner(
                f"{self.path.name}: recorded pid={existing.pid} does not match "
                f"this process pid={os.getpid()}. Refusing to remove a lock this "
                f"process does not own.")
        self.path.unlink(missing_ok=True)
        self._acquired = False

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, exc_type, exc, tb) -> None:
        # Release ONLY on a normal exit. A crash must leave the lock in place
        # so a human investigates rather than a second process quietly
        # starting against a run that died mid-write -- releasing unconditionally
        # here would silently defeat the whole point of this module.
        if self._acquired and exc_type is None:
            self.release()


def stop_and_release(path: Path, timeout_s: float = 15.0, poll_s: float = 0.5) -> LockInfo:
    """Operator-facing stop: terminate the recorded owner, POLL until the OS
    confirms it is gone, THEN archive the lock. Never removes a lock for a
    process that has not been verified dead.

    This is the replacement for "pkill and hope, then delete the lock file."
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no lock at {path}")
    info = LockInfo.from_json(path.read_text(encoding="utf-8"))
    if psutil.pid_exists(info.pid):
        try:
            psutil.Process(info.pid).terminate()
        except psutil.NoSuchProcess:
            pass
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if not psutil.pid_exists(info.pid):
            break
        time.sleep(poll_s)
    else:
        raise StillAlive(
            f"pid={info.pid} did not exit within {timeout_s}s of being asked to "
            f"terminate. The lock was NOT removed. Escalate manually (kill -9 / "
            f"taskkill /F), confirm the process is gone, then call this again.")
    archived = path.with_suffix(f".stopped-{int(time.time())}.json")
    path.rename(archived)
    print(f"[run_lock] confirmed pid={info.pid} exited; lock archived -> {archived.name}")
    return info
