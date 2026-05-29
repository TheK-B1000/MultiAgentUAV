from __future__ import annotations

"""Compare reward-breakdown columns between two PPO policy versions.

The custom PPO trainer writes update metrics after each optimizer update, but the
rollout statistics in that row were collected immediately before the update. In
practice, CSV row ``update=1`` describes policy update 0, row ``update=2``
describes policy update 1, and so on.

By default this script filters the metrics CSV to the latest ``run_id`` only,
matching ``tools/diagnose_run.py`` — appended/rotated rows from earlier processes
are silently ignored so ``update=1`` always picks the most recent run's row.
Pass ``--all-runs`` to disable the filter (legacy behavior).
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


OPTIONAL_COMPARISON_COLUMNS: tuple[str, ...] = (
    "reward_outcome_mean",
    "reward_shaping_mean",
    "reward_failure_to_outcome_abs",
)

COMPARISON_COLUMNS: tuple[str, ...] = (
    "reward_offense_mean",
    "reward_pbrs_mean",
    "reward_team_mean",
    "reward_sparse_mean",
    "reward_failure_mean",
    *OPTIONAL_COMPARISON_COLUMNS,
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


def _latest_run_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """When metrics include run IDs, ignore older interleaved/rotated rows from prior processes."""
    if not rows:
        return rows
    latest_run_id = rows[-1].get("run_id", "")
    if latest_run_id == "":
        return rows
    selected = [row for row in rows if row.get("run_id", "") == latest_run_id]
    return selected or rows


def _read_rows(metrics_csv: Path, *, latest_run_only: bool = True) -> list[dict[str, str]]:
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{metrics_csv} has no metric rows.")
    if "update" not in rows[0]:
        raise ValueError(f"{metrics_csv} is missing the required 'update' column.")
    if latest_run_only:
        rows = _latest_run_rows(rows)
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
    latest_run_only: bool = True,
) -> list[UpdateComparison]:
    """Return metric deltas for policy update ``before`` vs ``after``.

    Policy update N maps to trainer metrics row update N+1 because reward/score
    summaries are measured on the rollout collected before optimizer update N+1.

    When ``latest_run_only`` is True (default), only rows from the most recent
    ``run_id`` are considered; older appended/rotated rows are silently dropped.
    """

    metrics_path = Path(metrics_csv)
    rows = _read_rows(metrics_path, latest_run_only=latest_run_only)
    before_trainer_update = int(before_policy_update) + 1
    after_trainer_update = int(after_policy_update) + 1
    before_row = _row_by_trainer_update(rows, before_trainer_update)
    after_row = _row_by_trainer_update(rows, after_trainer_update)
    if tuple(columns) == COMPARISON_COLUMNS:
        columns = tuple(
            column
            for column in columns
            if (column in before_row and column in after_row)
            or column not in OPTIONAL_COMPARISON_COLUMNS
        )
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
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help=(
            "Include rows from every run_id in the metrics CSV. "
            "Default behaviour mirrors tools/diagnose_run.py: only the latest run_id is kept, "
            "which avoids duplicate-update artifacts when --run-tag was reused or training was resumed."
        ),
    )
    args = parser.parse_args(argv)

    comparisons = compare_policy_updates(
        args.metrics_csv,
        before_policy_update=args.before_policy_update,
        after_policy_update=args.after_policy_update,
        latest_run_only=not args.all_runs,
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
