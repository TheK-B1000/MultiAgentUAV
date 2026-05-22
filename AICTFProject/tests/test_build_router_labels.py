from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tools.build_router_labels import main as build_router_labels_main


class BuildRouterLabelsTests(unittest.TestCase):
    def test_build_router_labels_writes_trainer_ce_schema(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for z, wr in enumerate((90.0, 75.0, 50.0, 25.0)):
                path = root / f"eval_unit_fix_z{z}_op5_bite_v3_4v4_aggregate.csv"
                path.write_text(
                    "\n".join(
                        [
                            "label,setting,map_set,opponent,episodes,success_rate",
                            f"unit_fix_z{z},4v4,eval,OP3,10,{wr}",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
            out = root / "router_labels.json"

            rc = build_router_labels_main(
                [
                    "--aggregate-dir",
                    str(root),
                    "--tag",
                    "op5_bite_v3_4v4",
                    "--map-set",
                    "eval",
                    "--temperature",
                    "0.05",
                    "--include-opponents",
                    "OP3",
                    "--out",
                    str(out),
                ]
            )

            self.assertEqual(rc, 0)
            bundle = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(bundle["k"], 4)
            self.assertIn("OP3", bundle["opponents"])
            self.assertEqual(bundle["opponents"]["OP3"]["opponent_id"], 2)
            self.assertEqual(bundle["opponents"]["OP3"]["hard_z"], 0)
            self.assertEqual(len(bundle["opponents"]["OP3"]["soft"]), 4)


if __name__ == "__main__":
    unittest.main()
