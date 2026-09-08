"""Frozen EXP2B terminal evaluation on untouched seeds 8600001..8600192.

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


evaluator.PROTOCOL = evaluator.SD / "EXP2B_SPECIALIZATION_PRESERVING_LATENT_COMPRESSION_PROTOCOL.json"
evaluator.PROTOCOL_ID = "EXP2B_SPECIALIZATION_PRESERVING_LATENT_COMPRESSION_V1"
evaluator.EXPERIMENT_ID = "EXP2B_SPECIALIZATION_PRESERVING_LATENT_COMPRESSION"
evaluator.TRAIN_DIR = (
    evaluator.SD
    / "exp2b_specialization_preserving_compression"
    / "exp2b_specialization_preserving_seed8400001_2m"
)
evaluator.STUDENT = (
    evaluator.TRAIN_DIR
    / "ckpts"
    / "final_exp2b_specialization_preserving_seed8400001_2m.zip"
)
evaluator.OUT = evaluator.SD / "exp2b_k2_terminal_evaluation"
evaluator.EXPECTED_HASHES = {
    "student": "4b0a0e08051be6abba8b34c11d34c5b7dd82616903fa87f39f859e65a7115a0b",
    "pi_A": "5bd5f54f5ce206b139626bded8ca1f296d82d47c0d4c21db4ed561297a2d411d",
    "pi_B": "8e4fb58be11465c24a258da3ac94648e669c0f65ab98a64b42a7b4c8b6a6c8fc",
}
evaluator.SEED_BASE = 8_600_001
evaluator.SEED_BLOCK = "8600001..8600192"
evaluator.DEVELOPMENT_SEED = 8_500_001


if __name__ == "__main__":
    raise SystemExit(evaluator.main())
