"""Maritime CTF configuration space, train/held-out splits, and eval seed blocks.

A *configuration* is what is drawn at the start of an episode and then held fixed
for its duration. Following the revised formalization (docs/paper/formalization.md):

    c = (opponent_kind, opponent_key, current_profile, team_size, episode_seed)

Everything defined here is restricted to factors that actually exist in this
branch's environment (``game_field_gpu.BatchedCTFCore`` plus ``rl/curriculum.py``).
The map is fixed and the ruleset is fixed -- they are recorded as constants so the
protocol is explicit, not varied:

  * opponents        SCRIPTED OP1..OP4, SPECIES RUSHER/CAMPER/BALANCED
  * current profile  ``current_strength_cps`` and ``drift_sigma_cells``, the only
                     two stress keys ``BatchedCTFCore._apply_profile_runtime``
                     actually applies at runtime
  * team size        symmetric NvN

The seen/held-out split is *derived from what training samples*, not from an
arbitrary cardinality: ``rl/curriculum.py``'s ``VALID_PHASES`` is
``("OP1", "OP2", "OP3")`` and ``rl/league.py`` samples SCRIPTED OP1-3 / SPECIES /
SNAPSHOT, so OP4 and any current profile outside ``STRESS_BY_PHASE`` are never
seen during training.

Team size is deliberately NOT a generalization axis. Observation and action
spaces are built by ``game_field_gpu._make_obs_action_spaces(n_agents, ...)``
with shapes ``(n_agents, ...)`` and ``MultiDiscrete([n_macros, n_targets] *
n_agents)``, so a 2v2 checkpoint cannot even be loaded at 3v3. Independently
trained 2v2/3v3/4v4 policies measure *scalability*; see
``assert_team_size_compatible``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Sequence, Tuple

from rl.curriculum import STRESS_BY_PHASE

# Fixed protocol constants -- recorded so the paper can state them, not varied.
FIXED_MAP = "map_a"
FIXED_RULES_PROFILE = "OURS"

SCRIPTED_OPPONENTS: Tuple[str, ...] = ("OP1", "OP2", "OP3", "OP4")
SPECIES_OPPONENTS: Tuple[str, ...] = ("RUSHER", "CAMPER", "BALANCED")

# Opponents the training distribution actually samples (curriculum phases +
# league SPECIES category). SNAPSHOT opponents are self-generated and therefore
# not part of a fixed, shareable evaluation set.
TRAIN_SCRIPTED: Tuple[str, ...] = ("OP1", "OP2", "OP3")
TRAIN_SPECIES: Tuple[str, ...] = SPECIES_OPPONENTS

# Held out: never sampled during training by any method under comparison.
HELDOUT_SCRIPTED: Tuple[str, ...] = ("OP4",)

# Named current profiles. Only current_strength_cps / drift_sigma_cells are
# honoured by BatchedCTFCore._apply_profile_runtime, so those are the only keys
# used -- claiming sensor-noise variation here would be false advertising.
_OP2 = STRESS_BY_PHASE["OP2"]
_OP3 = STRESS_BY_PHASE["OP3"]

CURRENT_PROFILES: Dict[str, Dict[str, float]] = {
    # Seen in training: OP2's and OP3's stress entries.
    "calm": {
        "current_strength_cps": float(_OP2.get("current_strength_cps", 0.02)),
        "drift_sigma_cells": float(_OP2.get("drift_sigma_cells", 0.0)),
    },
    "nominal": {
        "current_strength_cps": float(_OP3["current_strength_cps"]),
        "drift_sigma_cells": float(_OP3["drift_sigma_cells"]),
    },
    # Held out: stronger set and drift than anything the stress schedule emits.
    "strong": {
        "current_strength_cps": 0.20,
        "drift_sigma_cells": 0.06,
    },
    "severe": {
        "current_strength_cps": 0.28,
        "drift_sigma_cells": 0.10,
    },
}

TRAIN_CURRENT_PROFILES: Tuple[str, ...] = ("calm", "nominal")
HELDOUT_CURRENT_PROFILES: Tuple[str, ...] = ("strong", "severe")


class TeamSizeMismatchError(ValueError):
    """Raised when a checkpoint is asked to act at a team size it cannot represent."""


@dataclass(frozen=True)
class Configuration:
    """One point in the maritime configuration space (episode seed excluded)."""

    opponent_kind: str
    opponent_key: str
    current_profile: str
    team_size: int

    def __post_init__(self) -> None:
        kind = str(self.opponent_kind).upper()
        if kind not in ("SCRIPTED", "SPECIES"):
            raise ValueError(
                f"opponent_kind must be SCRIPTED or SPECIES (SNAPSHOT opponents are "
                f"run-specific and cannot be part of a shared eval set); got {self.opponent_kind!r}"
            )
        key = str(self.opponent_key).upper()
        valid = SCRIPTED_OPPONENTS if kind == "SCRIPTED" else SPECIES_OPPONENTS
        if key not in valid:
            raise ValueError(f"unknown {kind} opponent {self.opponent_key!r}; expected one of {valid}")
        if str(self.current_profile) not in CURRENT_PROFILES:
            raise ValueError(
                f"unknown current profile {self.current_profile!r}; "
                f"expected one of {tuple(CURRENT_PROFILES)}"
            )
        if int(self.team_size) < 1:
            raise ValueError(f"team_size must be >= 1, got {self.team_size}")

    @property
    def key(self) -> str:
        """Stable string ID used for CSV rows, seed blocks and dict keys."""
        return (
            f"{str(self.opponent_kind).upper()}:{str(self.opponent_key).upper()}"
            f"|cur={self.current_profile}|{int(self.team_size)}v{int(self.team_size)}"
        )

    @property
    def setting(self) -> str:
        return f"{int(self.team_size)}v{int(self.team_size)}"

    def stress_schedule(self, phase_name: str) -> Dict[str, Dict[str, float]]:
        """Stress schedule dict to hand to ``core.set_stress_schedule``.

        ``BatchedCTFCore._apply_profile_runtime`` looks the schedule up by the
        env's current phase string, so the profile is registered under the exact
        phase name the evaluator will set. Registering it under one key only is
        what makes the current profile a *controlled* factor rather than
        something silently inherited from ``STRESS_BY_PHASE``.
        """
        return {str(phase_name).upper(): dict(CURRENT_PROFILES[self.current_profile])}


def is_seen(config: Configuration) -> bool:
    """Whether ``config`` lies inside the training distribution.

    Membership is decided by the opponent and the current profile only; team size
    is a separate axis (each team size has its own independently trained policy).
    """
    kind = str(config.opponent_kind).upper()
    key = str(config.opponent_key).upper()
    if config.current_profile not in TRAIN_CURRENT_PROFILES:
        return False
    if kind == "SCRIPTED":
        return key in TRAIN_SCRIPTED
    return key in TRAIN_SPECIES


def seen_configurations(team_size: int) -> List[Configuration]:
    """C_seen: configurations every compared method encountered during training."""
    out: List[Configuration] = []
    for profile in TRAIN_CURRENT_PROFILES:
        for key in TRAIN_SCRIPTED:
            out.append(Configuration("SCRIPTED", key, profile, int(team_size)))
        for key in TRAIN_SPECIES:
            out.append(Configuration("SPECIES", key, profile, int(team_size)))
    return out


def heldout_configurations(team_size: int) -> List[Configuration]:
    """C_heldout: configurations no compared method encountered during training.

    Disjoint from ``seen_configurations`` by construction -- every element varies
    the opponent, the current profile, or both, away from the training set.
    """
    out: List[Configuration] = []
    # Unseen opponent, seen current profiles.
    for profile in TRAIN_CURRENT_PROFILES:
        for key in HELDOUT_SCRIPTED:
            out.append(Configuration("SCRIPTED", key, profile, int(team_size)))
    # Seen opponents, unseen current profiles.
    for profile in HELDOUT_CURRENT_PROFILES:
        for key in TRAIN_SCRIPTED:
            out.append(Configuration("SCRIPTED", key, profile, int(team_size)))
        for key in TRAIN_SPECIES:
            out.append(Configuration("SPECIES", key, profile, int(team_size)))
    # Unseen opponent and unseen current profile.
    for profile in HELDOUT_CURRENT_PROFILES:
        for key in HELDOUT_SCRIPTED:
            out.append(Configuration("SCRIPTED", key, profile, int(team_size)))
    return out


def split(team_size: int) -> Dict[str, List[Configuration]]:
    """Both evaluation sets for one team size, as ``{"seen": [...], "heldout": [...]}``."""
    return {
        "seen": seen_configurations(team_size),
        "heldout": heldout_configurations(team_size),
    }


def iter_all(team_sizes: Sequence[int]) -> Iterator[Tuple[str, Configuration]]:
    """Yield ``(split_name, config)`` across team sizes, seen split first."""
    for team_size in team_sizes:
        parts = split(int(team_size))
        for split_name in ("seen", "heldout"):
            for config in parts[split_name]:
                yield split_name, config


def config_seed_block(config: Configuration, *, block_size: int = 100_000) -> int:
    """Deterministic, collision-free seed-block offset for one configuration.

    Generalizes ``plot/eval_rollout.shared_episode_seeds``'s OP3/OP4 block trick to
    the whole configuration space: every configuration gets its own contiguous
    block, so two different configurations never reuse an episode seed while every
    *method* sees identical episodes within a configuration (common random
    numbers). Derived from the config key rather than enumeration order so adding
    a configuration never renumbers the existing ones.
    """
    digest = 0
    for ch in config.key:
        digest = (digest * 131 + ord(ch)) & 0xFFFFFFFF
    return (digest % 20_000) * int(block_size)


def episode_seeds(
    config: Configuration,
    n_episodes: int,
    *,
    seed_base: int = 0,
    block_size: int = 100_000,
) -> List[int]:
    """Shared per-episode seed list for ``config`` (common random numbers).

    Every checkpoint compared on this configuration is handed this exact list, so
    differences between methods are not differences in the episodes they faced.
    """
    if int(n_episodes) > int(block_size):
        raise ValueError(
            f"n_episodes={n_episodes} exceeds block_size={block_size}; seed blocks would overlap"
        )
    base = int(seed_base) + config_seed_block(config, block_size=block_size)
    return [base + i for i in range(int(n_episodes))]


def assert_team_size_compatible(checkpoint_team_size: int, eval_team_size: int) -> None:
    """Refuse to evaluate a checkpoint at a team size it cannot represent.

    ``_make_obs_action_spaces`` builds ``grid``/``vec``/``agent_mask`` with a
    leading ``n_agents`` dimension and a ``MultiDiscrete([n_macros, n_targets] *
    n_agents)`` action space, so weights trained at one team size are structurally
    incompatible with another. Zero-shot team-size transfer would require a
    variable-team (e.g. tokenized/attention) policy, which this branch does not
    have -- so the 2v2/3v3/4v4 results are a *scalability* study, not evidence of
    team-size generalization.
    """
    if int(checkpoint_team_size) != int(eval_team_size):
        raise TeamSizeMismatchError(
            f"checkpoint was trained at {int(checkpoint_team_size)}v{int(checkpoint_team_size)} but "
            f"evaluation asks for {int(eval_team_size)}v{int(eval_team_size)}. Observation and action "
            "spaces are team-size dependent (see game_field_gpu._make_obs_action_spaces), so this "
            "transfer is impossible with a fixed-team policy. Independently trained per-size policies "
            "measure SCALABILITY; report them as such, not as team-size generalization."
        )


def describe_split(team_size: int) -> str:
    """Human-readable protocol summary for logs and the paper's appendix."""
    parts = split(int(team_size))
    lines = [
        f"Configuration space @ {int(team_size)}v{int(team_size)} "
        f"(map={FIXED_MAP}, rules={FIXED_RULES_PROFILE})",
        f"  seen    ({len(parts['seen']):2d}): opponents {TRAIN_SCRIPTED + TRAIN_SPECIES}, "
        f"currents {TRAIN_CURRENT_PROFILES}",
        f"  heldout ({len(parts['heldout']):2d}): opponents {HELDOUT_SCRIPTED} and/or "
        f"currents {HELDOUT_CURRENT_PROFILES}",
    ]
    for name in ("seen", "heldout"):
        for config in parts[name]:
            lines.append(f"    [{name:7s}] {config.key}")
    return "\n".join(lines)


def _current_profile_table() -> List[Dict[str, Any]]:
    """Rows for the paper's configuration-space table."""
    return [
        {
            "profile": name,
            "current_strength_cps": vals["current_strength_cps"],
            "drift_sigma_cells": vals["drift_sigma_cells"],
            "split": "seen" if name in TRAIN_CURRENT_PROFILES else "heldout",
        }
        for name, vals in CURRENT_PROFILES.items()
    ]


if __name__ == "__main__":  # pragma: no cover - manual protocol inspection
    for size in (2, 3, 4):
        print(describe_split(size))
        print()
    for row in _current_profile_table():
        print(row)
