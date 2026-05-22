"""Build router CE labels from fixed-z eval aggregate CSVs.

This is the C1-facing name for ``derive_best_z_labels.py``. It keeps the
trainer label schema unchanged while making the staged-router workflow explicit:

    fixed-z eval -> router labels JSON -> frozen q_phi CE training
"""

from __future__ import annotations

from typing import Optional, Sequence

try:
    from .derive_best_z_labels import main as _derive_main
except ImportError:  # pragma: no cover - direct script execution path
    from derive_best_z_labels import main as _derive_main


def main(argv: Optional[Sequence[str]] = None) -> int:
    return int(_derive_main(argv))


if __name__ == "__main__":
    raise SystemExit(main())
