from __future__ import annotations

"""Compare reward-breakdown columns between two PPO policy versions.

The custom PPO trainer writes update metrics after each optimizer update, but the
rollout statistics in that row were collected immediately before the update. In
practice, CSV row ``update=1`` describes policy update 0, row ``update=2``
describes policy update 1, and so on.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


COMPARISON_COLUMNS: tuple[str, ...] = (
    "reward_offense_mean",
    "reward_pbrs_mean",
    "reward_team_mean",
    "reward_sparse_mean",
    "reward_failure_mean",
    "rollout_blue_score_mean",
    "rollout_red_score_mean",
)


@dataclass(frozen=True)
class UpdateComparison:
    column: str
    before: float
    after: float

    @property
    def delta(self) -> float:
        return self.after - self.before


def _read_rows(metrics_csv: Path) -> list[dict[str, str]]:
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{metrics_csv} has no metric rows.")
    if "update" not in rows[0]:
        raise ValueError(f"{metrics_csv} is missing the required 'update' column.")
    return rows


def _row_by_trainer_update(rows: Iterable[dict[str, str]], trainer_update: int) -> dict[str, str]:
    target = str(int(trainer_update))
    for row in rows:
        if str(row.get("update", "")).strip() == target:
            return row
    raise ValueError(f"metrics CSV is missing trainer update row {trainer_update}.")


def _as_float(row: dict[str, str], column: str, trainer_update: int) -> float:
    if column not in row:
        raise ValueError(f"metrics CSV is missing comparison column {column!r}.")
    value = row[column]
    if value is None or str(value).strip() == "":
        raise ValueError(f"column {column!r} is empty at trainer update row {trainer_update}.")
    return float(value)


def compare_policy_updates(
    metrics_csv: str | Path,
    *,
    before_policy_update: int = 0,
    after_policy_update: int = 1,
    columns: Sequence[str] = COMPARISON_COLUMNS,
) -> list[UpdateComparison]:
    """Return metric deltas for policy update ``before`` vs ``after``.

    Policy update N maps to trainer metrics row update N+1 because reward/score
    summaries are measured on the rollout collected before optimizer update N+1.
    """

    metrics_path = Path(metrics_csv)
    rows = _read_rows(metrics_path)
    before_trainer_update = int(before_policy_update) + 1
    after_trainer_update = int(after_policy_update) + 1
    before_row = _row_by_trainer_update(rows, before_trainer_update)
    after_row = _row_by_trainer_update(rows, after_trainer_update)
    return [
        UpdateComparison(
            column=column,
            before=_as_float(before_row, column, before_trainer_update),
            after=_as_float(after_row, column, after_trainer_update),
        )
        for column in columns
    ]


def format_markdown_table(
    comparisons: Sequence[UpdateComparison],
    *,
    before_label: str = "policy_update_0",
    after_label: str = "policy_update_1",
) -> str:
    lines = [
        f"| Metric | {before_label} | {after_label} | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for item in comparisons:
        lines.append(f"| {item.column} | {item.before:.6g} | {item.after:.6g} | {item.delta:+.6g} |")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics_csv", type=Path, help="Update metrics CSV written by custom PPO.")
    parser.add_argument("--before-policy-update", type=int, default=0)
    parser.add_argument("--after-policy-update", type=int, default=1)
    args = parser.parse_args(argv)

    comparisons = compare_policy_updates(
        args.metrics_csv,
        before_policy_update=args.before_policy_update,
        after_policy_update=args.after_policy_update,
    )
    print(
        format_markdown_table(
            comparisons,
            before_label=f"policy_update_{args.before_policy_update}",
            after_label=f"policy_update_{args.after_policy_update}",
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
