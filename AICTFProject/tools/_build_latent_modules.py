"""One-shot builder for latent module extraction (dev utility)."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LATENT = ROOT / "rl" / "custom_ppo" / "latent"

BUCKET_HEADER = '''"""Post-hoc context bucketing for latent strategy diagnostics and router losses."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from rl.global_state import GLOBAL_STATE_DIM


'''

bucket_body = (LATENT / "_extract_buckets.py").read_text(encoding="utf-8")
bucket_body = bucket_body.replace("def _", "def ")
# keep internal calls consistent with public function names
for old, new in [
    ("_team_phase_bucket_ids", "team_phase_bucket_ids"),
    ("_carrier_progress_bucket_ids", "carrier_progress_bucket_ids"),
    ("_score_pressure_bucket_ids", "score_pressure_bucket_ids"),
    ("_tactical_local_context_keys", "tactical_local_context_keys"),
    ("_role_phase_specialist_context_keys", "role_phase_specialist_context_keys"),
    ("_tactical_specialist_context_keys", "tactical_specialist_context_keys"),
]:
    bucket_body = bucket_body.replace(old, new)
(LATENT / "context_buckets.py").write_text(BUCKET_HEADER + bucket_body, encoding="utf-8")

ROUTER_HEADER = '''"""Per-step router sampling: sparse z resample, forced-z, refresh, behavior log-probs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.distributions import Categorical

from rl.global_state import GLOBAL_STATE_DIM
from rl.custom_ppo.latent.behavior_policy import (
    behavior_log_prob_from_probs,
    epsilon_behavior_probs,
)
from rl.custom_ppo.latent.context_buckets import (
    carrier_progress_bucket_ids,
    flag_state_bucket_ids,
    score_pressure_bucket_ids,
    strategy_experience_bucket_ids,
    team_phase_bucket_ids,
)
from rl.custom_ppo.schedules import resolve_latent_forced_z_frac

if TYPE_CHECKING:
    from rl.custom_ppo.latent.state import LatentStrategyState


class RouterSamplingState:
    def __init__(self, host: "LatentStrategyState") -> None:
        self.host = host

    @property
    def trainer(self):
        return self.host.trainer

'''

method_lines = (LATENT / "_extract_strategy_for_step.py").read_text(encoding="utf-8").splitlines()
# find body after docstring close of strategy_for_step
start = 0
for i, line in enumerate(method_lines):
    if '"""Return current sparse strategy' in line or "trainer = self.trainer" in line:
        start = i if "trainer = self.trainer" in line else i + 1
        break
out = [
    ROUTER_HEADER.rstrip(),
    "",
    "    def strategy_for_step(",
    "        self,",
    "        global_state: torch.Tensor,",
    "    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], dict[str, torch.Tensor]]:",
    '        """Return current sparse strategy and sampling metadata for one rollout step."""',
]
for line in method_lines[start:]:
    line = re.sub(r"\bself\.(?!host\b)", "self.host.", line)
    for old, new in [
        ("_flag_state_bucket_ids", "flag_state_bucket_ids"),
        ("_team_phase_bucket_ids", "team_phase_bucket_ids"),
        ("_score_pressure_bucket_ids", "score_pressure_bucket_ids"),
        ("_strategy_experience_bucket_ids", "strategy_experience_bucket_ids"),
        ("_carrier_progress_bucket_ids", "carrier_progress_bucket_ids"),
    ]:
        line = line.replace(old, new)
    out.append(line)
(LATENT / "router_sampling.py").write_text("\n".join(out) + "\n", encoding="utf-8")
print("built context_buckets and router_sampling")
