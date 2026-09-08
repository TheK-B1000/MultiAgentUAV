"""Frozen EXP2C terminal evaluation on untouched seeds 8900001..8900192.

All scoring and gate logic is inherited unchanged from the EXP2 evaluator.
Only experiment identity, frozen artifacts, output path, and reserved seeds vary.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import eval_exp2_k2_terminal as evaluator


evaluator.PROTOCOL = evaluator.SD / "EXP2C_MODE_SPECIFIC_ACTOR_COMPRESSION_PROTOCOL.json"
evaluator.PROTOCOL_ID = "EXP2C_MODE_SPECIFIC_ACTOR_COMPRESSION_V1"
evaluator.EXPERIMENT_ID = "EXP2C_MODE_SPECIFIC_ACTOR_COMPRESSION"
evaluator.TRAIN_DIR = (
    evaluator.SD
    / "exp2c_mode_specific_actor_compression"
    / "exp2c_mode_specific_actor_seed8700001_2m"
)
evaluator.STUDENT = (
    evaluator.TRAIN_DIR
    / "ckpts"
    / "final_exp2c_mode_specific_actor_seed8700001_2m.zip"
)
evaluator.OUT = evaluator.SD / "exp2c_k2_terminal_evaluation"
evaluator.EXPECTED_HASHES = {
    "student": "0a763759d49e6ba0ef1d78b25aba04cbc99a4cbcf50df8c19b201e941f646c07",
    "pi_A": "5bd5f54f5ce206b139626bded8ca1f296d82d47c0d4c21db4ed561297a2d411d",
    "pi_B": "8e4fb58be11465c24a258da3ac94648e669c0f65ab98a64b42a7b4c8b6a6c8fc",
}
evaluator.SEED_BASE = 8_900_001
evaluator.SEED_BLOCK = "8900001..8900192"
evaluator.DEVELOPMENT_SEED = 8_800_001


if __name__ == "__main__":
    raise SystemExit(evaluator.main())
