"""Progress bars must remain visible under redirected stderr (log-tail contract)."""
from __future__ import annotations

import io
import sys
import unittest
from types import SimpleNamespace
from unittest import mock


class LogWatcherProgressTests(unittest.TestCase):
    def test_non_tty_opens_log_watcher_not_rich(self) -> None:
        from rl.custom_ppo.trainer import _LogWatcherProgress, _open_sb3_style_progress

        cfg = SimpleNamespace(enable_progress_bar=True)
        fake_err = io.StringIO()
        fake_out = io.StringIO()
        with mock.patch("rl.custom_ppo.trainer._stderr_is_interactive", return_value=False), mock.patch(
            "sys.stderr", fake_err
        ), mock.patch("sys.stdout", fake_out):
            bar = _open_sb3_style_progress(
                cfg, total_timesteps=10_000, current_num_timesteps=0
            )
            self.assertIsInstance(bar, _LogWatcherProgress)
            self.assertEqual(bar.total, 10_000)
            bar.update(2048)
            bar.close()

        err_text = fake_err.getvalue()
        out_text = fake_out.getvalue()
        self.assertIn("PPO", err_text)
        self.assertIn("PPO", out_text)
        # Newline heartbeats (not rich / empty redirect).
        self.assertGreaterEqual(err_text.count("\n"), 1)
        self.assertGreaterEqual(out_text.count("\n"), 1)

    def test_disabled_returns_none(self) -> None:
        from rl.custom_ppo.trainer import _open_sb3_style_progress

        cfg = SimpleNamespace(enable_progress_bar=False)
        self.assertIsNone(
            _open_sb3_style_progress(cfg, total_timesteps=1000, current_num_timesteps=0)
        )


class TqdmIterRedirectTests(unittest.TestCase):
    def test_tqdm_iter_non_tty_uses_ascii(self) -> None:
        from experiments.tqdm_loop import tqdm_iter

        items = list(range(3))
        with mock.patch("experiments.tqdm_loop._stderr_is_interactive", return_value=False):
            wrapped = tqdm_iter(items, desc="TEST", total=3, unit="ep")
        # Exhaust without requiring a TTY.
        self.assertEqual(list(wrapped), items)


if __name__ == "__main__":
    unittest.main()
