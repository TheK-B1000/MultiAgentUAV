"""One-shot shrink of latent_strategy_state.py into modular layout."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MONO = ROOT / "rl" / "custom_ppo" / "latent_strategy_state.py"
LATENT = ROOT / "rl" / "custom_ppo" / "latent"
lines = MONO.read_text(encoding="utf-8").splitlines(keepends=True)


def slice_lines(start: int, end: int) -> str:
    return "".join(lines[start - 1 : end])


def write_preferences() -> None:
    header = '''"""v3i3 preference targets and router teacher helpers."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

'''
    body = slice_lines(89, 437)
    body = body.replace("def _", "def ")
    body = body.replace("v3i3_target_from_items(", "v3i3_target_from_items(")
    (LATENT / "preferences.py").write_text(header + body, encoding="utf-8")


def write_tensor_state() -> None:
    init_body = slice_lines(700, 934)
    init_body = init_body.replace("self.", "host.")
    indented = "".join(
        ("    " + line if line.strip() else line) for line in init_body.splitlines(keepends=True)
    )
    text = f'''"""Tensor and scalar field allocation for :class:`LatentStrategyStateCore`."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import torch

from rl.behavior_telemetry import N_TELEMETRY
from rl.custom_ppo.csv_writers import SCRIPTED_OPPONENT_MI_COUNT
from rl.custom_ppo.latent.lifecycle import EpisodeLifecycleState
from rl.custom_ppo.latent.records import EpisodeStrategyRecorder
from rl.custom_ppo.latent.selector_memory import SelectorMemory

if TYPE_CHECKING:
    from rl.custom_ppo.trainer import CustomPPOTrainer


def allocate_latent_state_fields(host: Any, trainer: "CustomPPOTrainer") -> None:
    """Attach rollout tensors, buffers, and lifecycle owners to ``host``."""
{indented}'''
    (LATENT / "tensor_state.py").write_text(text, encoding="utf-8")


def write_slim_core() -> None:
    header = '''"""Core latent z-machine state (resets, telemetry, competence, q_phi param helpers)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from rl.behavior_telemetry import N_TELEMETRY
from rl.custom_ppo.csv_writers import SCRIPTED_OPPONENT_MI_COUNT, _opponent_id_int_from_info
from rl.custom_ppo.latent.optimization.specialist_router import SpecialistRouterManager
from rl.custom_ppo.latent.tensor_state import allocate_latent_state_fields
from rl.custom_ppo.latent.records import stack_selector_hidden_records

if TYPE_CHECKING:
    from rl.custom_ppo.trainer import CustomPPOTrainer

# Backward-compatible re-exports (tests and presets import these names).
from rl.custom_ppo.latent.context_buckets import (  # noqa: F401
    carrier_progress_bucket_ids as _carrier_progress_bucket_ids,
    episode_bucket_baseline_keys as _episode_bucket_baseline_keys,
    flag_state_bucket_ids as _flag_state_bucket_ids,
    role_phase_specialist_context_keys as _role_phase_specialist_context_keys,
    score_pressure_bucket_ids as _score_pressure_bucket_ids,
    specialist_context_keys_for_mode as _specialist_context_keys_for_mode,
    strategy_experience_bucket_ids as _strategy_experience_bucket_ids,
    tactical_local_context_keys as _tactical_local_context_keys,
    tactical_specialist_context_keys as _tactical_specialist_context_keys,
    team_phase_bucket_ids as _team_phase_bucket_ids,
)
from rl.custom_ppo.latent.preferences import (  # noqa: F401
    advantage_weighted_target_from_records as _advantage_weighted_target_from_records,
    router_specialist_coef_scale as _router_specialist_coef_scale,
    router_specialist_loss as _router_specialist_loss,
    v3i3_resolve_target as _v3i3_resolve_target,
    v3i3_target_from_items as _v3i3_target_from_items,
    warmup_ramp_coef_scale as _warmup_ramp_coef_scale,
)

_stack_selector_hidden_records = stack_selector_hidden_records


'''
    chunks = [
        slice_lines(689, 697),
        "    def __init__(self, trainer: \"CustomPPOTrainer\") -> None:\n"
        "        self.trainer = trainer\n"
        "        allocate_latent_state_fields(self, trainer)\n"
        "        self._specialist_router = SpecialistRouterManager(self)\n\n",
        slice_lines(936, 1080),
        slice_lines(1296, 1459),
        slice_lines(1705, 1762),
        slice_lines(2636, 2707),
        slice_lines(2899, 2917),
        "    def apply_rollout_specialist_router(self, buffer: Any) -> dict[str, float]:\n"
        "        return self._specialist_router.apply_rollout_specialist_router(buffer)\n\n",
        slice_lines(4326, 4343),
        "\n\ndef __getattr__(name: str):\n"
        '    if name == "LatentStrategyState":\n'
        "        from rl.custom_ppo.latent.state import LatentStrategyState as _LS\n"
        "        return _LS\n"
        '    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")\n',
    ]
    MONO.write_text(header + "".join(chunks), encoding="utf-8")


def append_record_episode_outcome() -> None:
    path = LATENT / "credit" / "episode_credit.py"
    text = path.read_text(encoding="utf-8")
    if "def record_episode_strategy_outcome" in text:
        return
    body = slice_lines(2778, 2897)
    body = body.replace("self.", "self.host.")
    body = body.replace("_opponent_id_int_from_info", "opponent_id_int_from_info")
    extra = (
        "\n    def record_episode_strategy_outcome(\n"
        + body.split("def record_episode_strategy_outcome(\n", 1)[1]
    )
    if "from rl.custom_ppo.csv_writers import SCRIPTED_OPPONENT_MI_COUNT" not in text:
        text = text.replace(
            "from rl.custom_ppo.csv_writers import SCRIPTED_OPPONENT_MI_COUNT",
            "from rl.custom_ppo.csv_writers import SCRIPTED_OPPONENT_MI_COUNT, _opponent_id_int_from_info as opponent_id_int_from_info",
        )
    path.write_text(text.rstrip() + extra, encoding="utf-8")


if __name__ == "__main__":
    write_preferences()
    write_tensor_state()
    append_record_episode_outcome()
    write_slim_core()
    print("preferences, tensor_state, slim core written")
