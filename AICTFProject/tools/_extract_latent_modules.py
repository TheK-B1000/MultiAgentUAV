"""Extract latent_strategy_state methods into modular files."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MONO = ROOT / "rl" / "custom_ppo" / "latent_strategy_state.py"
LATENT = ROOT / "rl" / "custom_ppo" / "latent"
lines = MONO.read_text(encoding="utf-8").splitlines(keepends=True)


def slice_methods(start_name: str) -> str:
    start = None
    end = len(lines)
    for i, line in enumerate(lines):
        if start is None and re.match(rf"    def {re.escape(start_name)}\(", line):
            if i > 0 and lines[i - 1].strip() == "@staticmethod":
                start = i - 1
            else:
                start = i
            continue
        if start is not None and re.match(r"    def \w+\(", line):
            end = i
            break
        if start is not None and re.match(r"    @staticmethod", line) and i > start + 1:
            end = i
            break
    if start is None:
        raise ValueError(f"method not found: {start_name}")
    return "".join(lines[start:end])


def to_host_method(body: str, *, class_name: str, indent: str = "    ") -> str:
    body = re.sub(r"\bself\.(?!host\b)", "self.host.", body)
    body = body.replace("_stack_selector_hidden_records", "stack_selector_hidden_records")
    body = body.replace("_flag_state_bucket_ids", "flag_state_bucket_ids")
    body = body.replace("_team_phase_bucket_ids", "team_phase_bucket_ids")
    body = body.replace("_score_pressure_bucket_ids", "score_pressure_bucket_ids")
    body = body.replace("_strategy_experience_bucket_ids", "strategy_experience_bucket_ids")
    body = body.replace("_carrier_progress_bucket_ids", "carrier_progress_bucket_ids")
    body = body.replace("_tactical_local_context_keys", "tactical_local_context_keys")
    body = body.replace("_opponent_id_int_from_info", "opponent_id_int_from_info")
    body = body.replace("_episode_bucket_baseline_keys", "episode_bucket_baseline_keys")
    body = body.replace("_warmup_ramp_coef_scale", "warmup_ramp_coef_scale")
    body = body.replace("_v3i3_target_from_items", "v3i3_target_from_items")
    body = body.replace("_v3i3_refresh_target_probs", "v3i3_refresh_target_probs")
    # method defs: keep as methods on manager class
    return body.replace(f"class {class_name}", f"class {class_name}")


def write_macro_credit() -> None:
    parts = [
        '"""Macro-segment router credit (V6I1 Phase B/C)."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import TYPE_CHECKING, Any",
        "",
        "import torch",
        "from torch.distributions import Categorical",
        "",
        "from rl.custom_ppo.latent.optimization.router_ppo import RouterPPOEngine",
        "from rl.custom_ppo.latent.optimization.router_registry import LatentOptimizerRegistry",
        "from rl.custom_ppo.latent.records import stack_selector_hidden_records",
        "from rl.custom_ppo.latent.types import RouterPPOBatch, RouterPPOConfig",
        "",
        "if TYPE_CHECKING:",
        "    from rl.custom_ppo.latent.state import LatentStrategyState",
        "",
        "",
        "class MacroCreditManager:",
        '    """Macro boundary accumulate / finalize / PPO via RouterPPOEngine."""',
        "",
        "    def __init__(self, host: LatentStrategyState) -> None:",
        "        self.host = host",
        "        self._engine: RouterPPOEngine | None = None",
        "",
        "    def _engine_for_host(self) -> RouterPPOEngine:",
        "        if self._engine is None:",
        "            registry = LatentOptimizerRegistry.from_trainer(self.host.trainer)",
        "            self._engine = RouterPPOEngine(trainer=self.host.trainer, registry=registry)",
        "        return self._engine",
        "",
    ]
    for name in (
        "reset_macro_rollout_state",
        "macro_accumulate_step",
        "macro_finalize",
        "macro_open",
    ):
        parts.append(to_host_method(slice_methods(name), class_name="MacroCreditManager"))
    parts.append(to_host_method(slice_methods("empty_macro_strategy_stats"), class_name="MacroCreditManager"))
    parts.append(to_host_method(slice_methods("apply_macro_strategy_ppo"), class_name="MacroCreditManager"))
    (LATENT / "credit" / "macro_credit.py").write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_arc_credit() -> None:
    parts = [
        '"""Arc-level router consequence credit (v3i19)."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import TYPE_CHECKING, Any",
        "",
        "import torch",
        "from torch.distributions import Categorical",
        "",
        "from rl.ppo_core import ppo_policy_loss",
        "from rl.custom_ppo.latent.optimization.router_ppo import RouterPPOEngine",
        "from rl.custom_ppo.latent.optimization.router_registry import LatentOptimizerRegistry",
        "from rl.custom_ppo.latent.records import stack_selector_hidden_records",
        "from rl.custom_ppo.latent.types import RouterPPOBatch, RouterPPOConfig",
        "",
        "if TYPE_CHECKING:",
        "    from rl.custom_ppo.latent.state import LatentStrategyState",
        "",
        "",
        "class ArcCreditManager:",
        '    """Per-arc accumulate / finalize / PPO."""',
        "",
        "    def __init__(self, host: LatentStrategyState) -> None:",
        "        self.host = host",
        "        self._engine: RouterPPOEngine | None = None",
        "",
        "    def _engine_for_host(self) -> RouterPPOEngine:",
        "        if self._engine is None:",
        "            registry = LatentOptimizerRegistry.from_trainer(self.host.trainer)",
        "            self._engine = RouterPPOEngine(trainer=self.host.trainer, registry=registry)",
        "        return self._engine",
        "",
    ]
    for name in (
        "reset_arc_credit_rollout_state",
        "arc_accumulate_step",
        "arc_finalize",
        "arc_open",
    ):
        try:
            parts.append(to_host_method(slice_methods(name), class_name="ArcCreditManager"))
        except ValueError:
            pass
    parts.append(to_host_method(slice_methods("empty_arc_strategy_stats"), class_name="ArcCreditManager"))
    parts.append(to_host_method(slice_methods("apply_arc_strategy_ppo"), class_name="ArcCreditManager"))
    (LATENT / "credit" / "arc_credit.py").write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_refresh_credit() -> None:
    parts = [
        '"""v3i3 event refresh record finalization."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import TYPE_CHECKING, Any",
        "",
        "from rl.custom_ppo.csv_writers import opponent_id_int_from_info as _opponent_id_int_from_info",
        "",
        "if TYPE_CHECKING:",
        "    from rl.custom_ppo.latent.state import LatentStrategyState",
        "",
        "",
        "class RefreshCreditManager:",
        "    def __init__(self, host: LatentStrategyState) -> None:",
        "        self.host = host",
        "",
    ]
    parts.append(to_host_method(slice_methods("finalize_v3i3_refresh_records"), class_name="RefreshCreditManager"))
    (LATENT / "credit" / "refresh_credit.py").write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_episode_credit() -> None:
    parts = [
        '"""Episode-boundary router credit and q_phi PPO update."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import TYPE_CHECKING, Any, Optional",
        "",
        "import numpy as np",
        "import torch",
        "from torch.distributions import Categorical",
        "",
        "from rl.ppo_core import ppo_policy_loss",
        "from rl.custom_ppo.latent.records import stack_selector_hidden_records",
        "from rl.custom_ppo.latent_strategy_state import (",
        "    episode_bucket_baseline_keys as _episode_bucket_baseline_keys,",
        "    v3i3_refresh_target_probs as _v3i3_refresh_target_probs,",
        "    v3i3_target_from_items as _v3i3_target_from_items,",
        "    warmup_ramp_coef_scale as _warmup_ramp_coef_scale,",
        ")",
        "",
        "if TYPE_CHECKING:",
        "    from rl.custom_ppo.latent.state import LatentStrategyState",
        "",
        "",
        "class EpisodeCreditManager:",
        "    def __init__(self, host: LatentStrategyState) -> None:",
        "        self.host = host",
        "",
    ]
    for name in (
        "store_episode_strategy_start",
        "empty_episode_strategy_stats",
        "episode_strategy_training_batch",
    ):
        try:
            parts.append(to_host_method(slice_methods(name), class_name="EpisodeCreditManager"))
        except ValueError:
            pass
    parts.append(to_host_method(slice_methods("apply_episode_strategy_ppo"), class_name="EpisodeCreditManager"))
    (LATENT / "credit" / "episode_credit.py").write_text("\n".join(parts) + "\n", encoding="utf-8")


def append_router_helpers() -> None:
    path = LATENT / "router_sampling.py"
    text = path.read_text(encoding="utf-8")
    marker = "    def record_tactical_context_step"
    if marker in text:
        # strip previously appended helpers and rebuild tail
        text = text[: text.index(marker)].rstrip() + "\n"
    extra = [
        "",
        "    def record_tactical_context_step(self, global_state: torch.Tensor) -> None:",
        '        """Accumulate detached tactical occupancy for each active episode."""',
        "        from rl.custom_ppo.latent.context_buckets import tactical_local_context_keys",
        "",
        "        if global_state.dim() != 2:",
        "            return",
        "        keys = tactical_local_context_keys(global_state).detach().long()",
        "        env_ids = torch.arange(int(keys.shape[0]), dtype=torch.long, device=keys.device)",
        "        self.host.episode_tactical_bucket_counts[env_ids, keys] += 1",
        "",
    ]
    extra.append(to_host_method(slice_methods("mark_strategy_step_done"), class_name="RouterSamplingState"))
    extra.append(to_host_method(slice_methods("representative_tactical_bucket"), class_name="RouterSamplingState"))
    path.write_text(text.rstrip() + "\n" + "\n".join(extra) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_macro_credit()
    write_arc_credit()
    write_refresh_credit()
    write_episode_credit()
    append_router_helpers()
    print("extracted credit + router helpers")
