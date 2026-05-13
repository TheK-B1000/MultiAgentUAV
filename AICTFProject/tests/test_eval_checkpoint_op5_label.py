"""eval_checkpoint OP5 label suffix matches opponent_params tuning tag."""

from __future__ import annotations

import unittest

from opponent_params import OP5_RUSHER_TUNING_TAG
from plot.eval_checkpoint import _label_append_op5_tuning_tag


class EvalCheckpointOp5LabelTests(unittest.TestCase):
    def test_suffix_when_op5_in_pool(self) -> None:
        out = _label_append_op5_tuning_tag(
            "preflight_flat_op35_wr_2v2_bite_v1",
            ["OP3", "OP5_RUSHER"],
            no_suffix=False,
        )
        self.assertTrue(out.endswith(f"_op5_{OP5_RUSHER_TUNING_TAG}"), msg=out)

    def test_no_suffix_when_op5_absent(self) -> None:
        out = _label_append_op5_tuning_tag("x", ["OP3"], no_suffix=False)
        self.assertEqual(out, "x")

    def test_no_double_suffix(self) -> None:
        tail = f"_op5_{OP5_RUSHER_TUNING_TAG}"
        base = f"run{tail}"
        out = _label_append_op5_tuning_tag(base, ["OP5"], no_suffix=False)
        self.assertEqual(out, base)

    def test_respects_no_suffix_flag(self) -> None:
        out = _label_append_op5_tuning_tag("y", ["OP5_RUSHER"], no_suffix=True)
        self.assertEqual(out, "y")


if __name__ == "__main__":
    unittest.main()
