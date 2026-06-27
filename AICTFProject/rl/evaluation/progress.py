"""Progress reporting protocol for the map-awareness evaluation.

Dependency-inject a ``ProgressReporter`` into components that produce output so
that tests can use ``NullProgressReporter`` without capturing stdout.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

try:
    from typing import Protocol, runtime_checkable
except ImportError:
    from typing_extensions import Protocol, runtime_checkable  # type: ignore[assignment]

if TYPE_CHECKING:
    from rl.evaluation.models import EpisodeResult


@runtime_checkable
class ProgressReporter(Protocol):
    """Callback interface for evaluation progress events."""

    def on_episode_started(
        self,
        *,
        policy_name: str,
        map_name: str,
        opponent: str,
        seed: int,
        completed: int,
        total: int,
    ) -> None: ...

    def on_episode_finished(
        self,
        result: "EpisodeResult",
        *,
        completed: int,
        total: int,
    ) -> None: ...

    def on_probe_finished(self, probe_name: str, status: str) -> None: ...

    def on_evaluation_finished(self, artifact_dir: str) -> None: ...


class NullProgressReporter:
    """No-op reporter for tests and batch runs that suppress output."""

    def on_episode_started(self, **_: object) -> None:
        pass

    def on_episode_finished(self, result: object, **_: object) -> None:
        pass

    def on_probe_finished(self, probe_name: str, status: str) -> None:
        pass

    def on_evaluation_finished(self, artifact_dir: str) -> None:
        pass


class ConsoleProgressReporter:
    """Prints one line per episode to stdout, matching the original script format."""

    def on_episode_started(
        self,
        *,
        policy_name: str,
        map_name: str,
        opponent: str,
        seed: int,
        completed: int,
        total: int,
    ) -> None:
        pass  # we print on_episode_finished instead

    def on_episode_finished(
        self,
        result: "EpisodeResult",
        *,
        completed: int,
        total: int,
    ) -> None:
        print(
            f"[eval] {completed:>4}/{total} "
            f"policy={result.condition.policy_name:9s} "
            f"map={result.condition.map_name:24s} "
            f"requested={result.condition.requested_opponent} "
            f"resolved={result.resolved_opponent} "
            f"seed={result.condition.seed} "
            f"score={result.blue_score:.0f}:{result.red_score:.0f}"
        )

    def on_probe_finished(self, probe_name: str, status: str) -> None:
        pass  # probes print their own output

    def on_evaluation_finished(self, artifact_dir: str) -> None:
        print(f"\nArtifacts written to: {artifact_dir}")
