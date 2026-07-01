"""V6I3 Slice 1: local communication transport unit tests."""

from __future__ import annotations

import unittest

import torch

from rl.custom_ppo.communication import CommConfig, LocalCommTransport


def _transport(*, dropout: float = 0.0) -> LocalCommTransport:
    cfg = CommConfig(dropout_probability=float(dropout), delivery_delay_steps=1)
    transport = LocalCommTransport(cfg)
    transport.reset(batch_size=1, num_agents=4, device=torch.device("cpu"))
    return transport


class CommTransportTests(unittest.TestCase):
    def test_no_self_message(self) -> None:
        transport = _transport()
        alive = torch.ones((1, 4), dtype=torch.bool)
        x = torch.tensor([[0.0, 3.0, 3.1, 20.0]], dtype=torch.float32)
        y = torch.tensor([[10.0, 10.0, 10.1, 10.0]], dtype=torch.float32)
        symbols = torch.tensor([[1, -1, -1, -1]], dtype=torch.long)
        transport.submit_outbound(symbols=symbols, sender_x=x, sender_y=y, alive=alive, apply_dropout=False)
        transport.advance_step(
            alive=alive, sender_x=x, sender_y=y, receiver_x=x, receiver_y=y, cols=20.0, rows=20.0
        )
        self.assertEqual(int(transport.active_signal[0, 0, 0].item()), -1)

    def test_only_nearby_teammates_receive(self) -> None:
        transport = _transport()
        alive = torch.ones((1, 4), dtype=torch.bool)
        x = torch.tensor([[0.0, 2.0, 12.0, 13.0]], dtype=torch.float32)
        y = torch.tensor([[10.0, 10.0, 10.0, 10.0]], dtype=torch.float32)
        symbols = torch.tensor([[2, -1, -1, -1]], dtype=torch.long)
        transport.submit_outbound(symbols=symbols, sender_x=x, sender_y=y, alive=alive, apply_dropout=False)
        transport.advance_step(
            alive=alive, sender_x=x, sender_y=y, receiver_x=x, receiver_y=y, cols=20.0, rows=20.0
        )
        self.assertEqual(int(transport.active_signal[0, 1, 0].item()), 2)
        self.assertEqual(int(transport.active_signal[0, 2, 0].item()), -1)

    def test_one_step_delay_is_exact(self) -> None:
        transport = _transport()
        alive = torch.ones((1, 4), dtype=torch.bool)
        x = torch.tensor([[0.0, 2.0, 2.1, 2.2]], dtype=torch.float32)
        y = torch.tensor([[10.0, 10.0, 10.0, 10.0]], dtype=torch.float32)
        symbols = torch.tensor([[3, -1, -1, -1]], dtype=torch.long)
        transport.submit_outbound(symbols=symbols, sender_x=x, sender_y=y, alive=alive, apply_dropout=False)
        transport.advance_step(
            alive=alive, sender_x=x, sender_y=y, receiver_x=x, receiver_y=y, cols=20.0, rows=20.0
        )
        self.assertEqual(int(transport.active_signal[0, 1, 0].item()), 3)

    def test_dead_sender_cancels_delivery(self) -> None:
        transport = _transport()
        alive = torch.ones((1, 4), dtype=torch.bool)
        x = torch.tensor([[0.0, 2.0, 2.1, 2.2]], dtype=torch.float32)
        y = torch.tensor([[10.0, 10.0, 10.0, 10.0]], dtype=torch.float32)
        symbols = torch.tensor([[1, -1, -1, -1]], dtype=torch.long)
        transport.submit_outbound(symbols=symbols, sender_x=x, sender_y=y, alive=alive, apply_dropout=False)
        alive_dead = alive.clone()
        alive_dead[0, 0] = False
        transport.advance_step(
            alive=alive_dead,
            sender_x=x,
            sender_y=y,
            receiver_x=x,
            receiver_y=y,
            cols=20.0,
            rows=20.0,
        )
        self.assertEqual(int(transport.active_signal[0, 1, 0].item()), -1)

    def test_dropout_affects_only_intended_delivery(self) -> None:
        transport = _transport(dropout=1.0)
        alive = torch.ones((1, 4), dtype=torch.bool)
        x = torch.tensor([[0.0, 2.0, 2.1, 2.2]], dtype=torch.float32)
        y = torch.tensor([[10.0, 10.0, 10.0, 10.0]], dtype=torch.float32)
        symbols = torch.tensor([[0, -1, -1, -1]], dtype=torch.long)
        transport.submit_outbound(
            symbols=symbols,
            sender_x=x,
            sender_y=y,
            alive=alive,
            apply_dropout=True,
            rng=torch.Generator().manual_seed(0),
        )
        transport.advance_step(
            alive=alive, sender_x=x, sender_y=y, receiver_x=x, receiver_y=y, cols=20.0, rows=20.0
        )
        self.assertEqual(int(transport.active_signal[0, 1, 0].item()), -1)
        self.assertGreater(int(transport.telemetry.dropout_count), 0)

    def test_reset_clears_state(self) -> None:
        transport = _transport()
        alive = torch.ones((1, 4), dtype=torch.bool)
        x = torch.ones((1, 4), dtype=torch.float32) * 5.0
        y = torch.ones((1, 4), dtype=torch.float32) * 5.0
        symbols = torch.tensor([[1, -1, -1, -1]], dtype=torch.long)
        transport.submit_outbound(symbols=symbols, sender_x=x, sender_y=y, alive=alive, apply_dropout=False)
        transport.reset(batch_size=1, num_agents=4, device=torch.device("cpu"))
        self.assertEqual(int(transport.global_step), 0)
        self.assertTrue(bool((transport.active_signal == -1).all()))

    def test_message_channel_marks_symbol_grid(self) -> None:
        transport = _transport()
        alive = torch.ones((1, 4), dtype=torch.bool)
        x = torch.tensor([[0.0, 2.0, 12.0, 13.0]], dtype=torch.float32)
        y = torch.tensor([[10.0, 10.0, 10.0, 10.0]], dtype=torch.float32)
        symbols = torch.tensor([[2, -1, -1, -1]], dtype=torch.long)
        transport.submit_outbound(symbols=symbols, sender_x=x, sender_y=y, alive=alive, apply_dropout=False)
        channels = transport.advance_step(
            alive=alive, sender_x=x, sender_y=y, receiver_x=x, receiver_y=y, cols=20.0, rows=20.0
        )
        assert channels is not None
        self.assertGreater(float(channels[0, 1, 2].sum().item()), 0.0)
        self.assertEqual(float(channels[0, 2, 2].sum().item()), 0.0)


if __name__ == "__main__":
    unittest.main()
