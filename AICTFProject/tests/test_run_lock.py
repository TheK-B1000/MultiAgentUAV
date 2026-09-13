"""Rule 13 self-test: the lock must REFUSE a live owner and require CONFIRMED
death before releasing a stopped one -- never delete-and-hope.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import psutil
import pytest

from experiments.run_lock import (LockHeld, LockInfo, NotOwner, RunLock, StillAlive,
                                  stop_and_release)


def _spawn_sleeper(seconds: float = 30.0) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", f"import time; time.sleep({seconds})"])


@pytest.fixture
def lockfile(tmp_path):
    return tmp_path / "TEST.run.lock"


def test_acquire_writes_full_identity(lockfile):
    lock = RunLock(lockfile, run_id="R1").acquire()
    info = LockInfo.from_json(lockfile.read_text(encoding="utf-8"))
    assert info.pid == pytest.importorskip("os").getpid()
    assert info.run_id == "R1"
    assert info.hostname
    assert info.start_utc
    lock.release()


def test_acquire_refuses_when_a_live_process_holds_it(lockfile):
    """The exact failure this rule exists to prevent: a second launch while the
    first is genuinely still running."""
    proc = _spawn_sleeper()
    try:
        lockfile.write_text(LockInfo(
            pid=proc.pid, run_id="OWNER", hostname="h", start_utc="t",
            cmdline=" ".join([sys.executable, "-c", "import time; time.sleep(30.0)"]),
        ).to_json(), encoding="utf-8")
        with pytest.raises(LockHeld, match="held by a LIVE process"):
            RunLock(lockfile, run_id="INTRUDER").acquire()
    finally:
        proc.terminate(); proc.wait(timeout=5)


def test_acquire_archives_a_genuinely_stale_lock_and_proceeds(lockfile):
    """A recorded pid that is verifiably dead: acquire must succeed, and the
    old record must be ARCHIVED, never silently deleted."""
    proc = _spawn_sleeper(0.1)
    proc.wait(timeout=5)
    assert not psutil.pid_exists(proc.pid)
    lockfile.write_text(LockInfo(
        pid=proc.pid, run_id="DEAD_OWNER", hostname="h", start_utc="t",
        cmdline="something that no longer exists",
    ).to_json(), encoding="utf-8")

    lock = RunLock(lockfile, run_id="NEW_OWNER").acquire()
    assert lockfile.is_file()
    info = LockInfo.from_json(lockfile.read_text(encoding="utf-8"))
    assert info.run_id == "NEW_OWNER"
    stale = list(lockfile.parent.glob("TEST.run.stale-*.json"))
    assert len(stale) == 1, "the dead owner's record must be archived, not deleted"
    archived = LockInfo.from_json(stale[0].read_text(encoding="utf-8"))
    assert archived.run_id == "DEAD_OWNER"
    lock.release()


def test_recycled_pid_running_something_else_is_treated_as_dead_owner(lockfile):
    """A live PID is not automatically a live OWNER: if the recorded command
    line does not match what that pid is actually running, the lock is stale.
    Guards against a PID being reused by an unrelated process."""
    lockfile.write_text(LockInfo(
        pid=__import__("os").getpid(),          # this pytest process: genuinely alive
        run_id="IMPOSTER", hostname="h", start_utc="t",
        cmdline="totally_unrelated_program --with-args-nobody-runs-here",
    ).to_json(), encoding="utf-8")
    lock = RunLock(lockfile, run_id="REAL").acquire()      # must NOT raise LockHeld
    assert LockInfo.from_json(lockfile.read_text(encoding="utf-8")).run_id == "REAL"
    lock.release()


def test_release_refuses_to_remove_a_lock_it_does_not_own(lockfile):
    lockfile.write_text(LockInfo(
        pid=999999, run_id="SOMEONE_ELSE", hostname="h", start_utc="t", cmdline="x",
    ).to_json(), encoding="utf-8")
    lock = RunLock(lockfile, run_id="ME")
    lock._acquired = True                       # simulate holding without acquiring
    with pytest.raises(NotOwner):
        lock.release()
    assert lockfile.is_file(), "a non-owned lock must survive a refused release"


def test_release_is_idempotent_when_lock_already_gone(lockfile):
    lock = RunLock(lockfile, run_id="R").acquire()
    lockfile.unlink()
    lock.release()                              # must not raise


def test_context_manager_releases_on_normal_exit(lockfile):
    with RunLock(lockfile, run_id="CTX"):
        assert lockfile.is_file()
    assert not lockfile.is_file()


def test_context_manager_leaves_lock_on_crash(lockfile):
    """A crash must NOT silently release the lock -- that would let a second
    process start against a run that died mid-write."""
    with pytest.raises(ValueError):
        with RunLock(lockfile, run_id="CRASHY"):
            assert lockfile.is_file()
            raise ValueError("simulated crash")
    assert lockfile.is_file(), "lock must survive a crash so a human investigates"


def test_stop_and_release_confirms_death_before_archiving(lockfile):
    """The replacement for 'pkill and hope': terminate, POLL until confirmed
    dead, THEN archive. Never remove the lock speculatively."""
    proc = _spawn_sleeper(30.0)
    lockfile.write_text(LockInfo(
        pid=proc.pid, run_id="TO_STOP", hostname="h", start_utc="t",
        cmdline=" ".join([sys.executable, "-c", "import time; time.sleep(30.0)"]),
    ).to_json(), encoding="utf-8")

    info = stop_and_release(lockfile, timeout_s=10.0, poll_s=0.1)
    assert info.run_id == "TO_STOP"
    assert not psutil.pid_exists(proc.pid), "stop_and_release must have actually killed it"
    assert not lockfile.is_file(), "the ORIGINAL path must no longer exist"
    stopped = list(lockfile.parent.glob("TEST.run.stopped-*.json"))
    assert len(stopped) == 1, "the stopped lock must be archived, not deleted outright"


def test_stop_and_release_raises_rather_than_giving_up_silently(lockfile, monkeypatch):
    """If the process refuses to die within the timeout, the function must
    raise -- not archive the lock as if the stop had worked."""
    proc = _spawn_sleeper(30.0)
    lockfile.write_text(LockInfo(
        pid=proc.pid, run_id="STUBBORN", hostname="h", start_utc="t",
        cmdline=" ".join([sys.executable, "-c", "import time; time.sleep(30.0)"]),
    ).to_json(), encoding="utf-8")
    # terminate() is mocked to a no-op so the process outlives the short timeout
    monkeypatch.setattr(psutil.Process, "terminate", lambda self: None)
    try:
        with pytest.raises(StillAlive):
            stop_and_release(lockfile, timeout_s=1.0, poll_s=0.1)
        assert lockfile.is_file(), "must NOT archive when death is unconfirmed"
    finally:
        proc.kill(); proc.wait(timeout=5)


def test_stop_and_release_missing_lock_raises(lockfile):
    with pytest.raises(FileNotFoundError):
        stop_and_release(lockfile)
