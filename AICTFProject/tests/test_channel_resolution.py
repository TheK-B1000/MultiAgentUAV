"""A formal evaluation must know its architecture, not guess it.

The original loader used ``cfg.get("cnn_channels", 0) or 8`` followed by a
try/except that retried with 7. Neither key existed on any checkpoint, so the
``or 8`` invented a width and the loader zero-expanded a 7-channel conv to 8.
It was behaviourally equivalent -- the added channel's weights were zero and the
obstacle plane is zero on map_a_open -- but the evaluation log then printed
``channels=8`` as though the checkpoint had asserted it, and the real width was
only ever discovered by catching an exception.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from experiments.run_g0_v2_evaluation import (  # noqa: E402
    ChannelResolutionError,
    resolve_cnn_channels,
)


def _weights(in_channels: int):
    return {"actor_cnn.conv.0.weight": torch.zeros(4, in_channels, 3, 3)}


def test_resolves_from_weights_when_no_metadata():
    assert resolve_cnn_channels({"model_state_dict": _weights(7)}) == 7
    assert resolve_cnn_channels({"model_state_dict": _weights(8)}) == 8


def test_explicit_metadata_takes_precedence_over_weights():
    payload = {"cnn_channels": 11, "model_state_dict": _weights(7)}
    assert resolve_cnn_channels(payload) == 11


def test_metadata_accepted_from_cfg_block():
    payload = {"cfg": {"num_cnn_channels": 9}, "model_state_dict": _weights(7)}
    assert resolve_cnn_channels(payload) == 9


def test_fails_closed_when_width_cannot_be_established():
    with pytest.raises(ChannelResolutionError, match="Refusing to guess"):
        resolve_cnn_channels({"model_state_dict": {}}, context="ckpt.zip")


def test_fails_closed_rather_than_defaulting_to_eight():
    """The specific regression: an absent key must never silently become 8."""
    with pytest.raises(ChannelResolutionError):
        resolve_cnn_channels({"cfg": {}}, context="ckpt.zip")


def test_zero_or_missing_metadata_falls_through_to_weights():
    """A falsy metadata value is 'unset', not 'zero channels'."""
    payload = {"cfg": {"cnn_channels": 0}, "model_state_dict": _weights(7)}
    assert resolve_cnn_channels(payload) == 7


def test_prefixed_state_dict_keys_still_resolve():
    payload = {"model_state_dict": {"module.actor_cnn.conv.0.weight": torch.zeros(4, 7, 3, 3)}}
    assert resolve_cnn_channels(payload) == 7
