"""Frozen SPPPO V1 terminal evaluation on untouched seeds 10300001..10300192.

All scoring and gate logic is INHERITED UNCHANGED from the EXP2 evaluator, the
same one EXP2B and EXP2C used. Only experiment identity, frozen artifacts,
output path and reserved seeds vary. Nothing about crossover, retention,
behavioural identity, the paired bootstrap, the RNG seed, the resample count or
the confidence-bound convention is redefined here.

Q_psi is ABSENT from this path by construction: the inherited evaluator contains
zero references to qpsi/scorer/ranking. The final question is whether the trained
policy exhibits crossover, retention and behavioural identity in the real
environment -- not whether the scorer likes it. Actual environment payoff is
authoritative.

Both latent modes are scored on BOTH poles regardless of their assigned training
pole (student cells z0|A, z0|B, z1|A, z1|B), alongside the frozen SAPPO teacher
references (pi_A|A, pi_A|B, pi_B|A, pi_B|B).

Run:  python experiments/eval_sppo_v1_terminal.py                    # contract only
      python experiments/eval_sppo_v1_terminal.py --development-smoke # no 103... rows
      python experiments/eval_sppo_v1_terminal.py --launch            # spends the block
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import eval_exp2_k2_terminal as evaluator


evaluator.PROTOCOL = evaluator.SD / "sppo" / "SPPPO_V1_PROTOCOL.json"
evaluator.PROTOCOL_ID = "SPPPO_V1"
evaluator.EXPERIMENT_ID = "SPPPO_V1_STRATEGIC_PAYOFF_PRESERVING_PPO"
evaluator.TRAIN_DIR = (
    evaluator.SD / "sppo" / "production" / "sppo_v1_production_1M_seed10100001"
)
evaluator.STUDENT = (
    evaluator.TRAIN_DIR / "ckpts" / "final_sppo_v1_production_1M_seed10100001.zip"
)
evaluator.OUT = evaluator.SD / "sppo" / "sppo_v1_terminal_evaluation"
evaluator.EXPECTED_HASHES = {
    # the frozen 1M production terminal
    "student": "1260e420b85ad01b1aecf433d65688adf153538195d30511e26a3585b7285f1f",
    # the SAME frozen SAPPO references EXP2/EXP2B/EXP2C used for retention
    "pi_A": "5bd5f54f5ce206b139626bded8ca1f296d82d47c0d4c21db4ed561297a2d411d",
    "pi_B": "8e4fb58be11465c24a258da3ac94648e669c0f65ab98a64b42a7b4c8b6a6c8fc",
}
evaluator.SEED_BASE = 10_300_001
evaluator.SEED_BLOCK = "10300001..10300192"
evaluator.DEVELOPMENT_SEED = 10_200_001


# The wrapper pattern rebinds attributes on a SHARED module. If any other
# wrapper (EXP2, EXP2B, EXP2C) is imported into the same process afterwards, its
# bindings silently overwrite these and the evaluation would run against a
# different checkpoint and a different seed block while reporting itself as
# SPPPO. Verified reproducible: importing eval_exp2c_k2_terminal after this
# module moves SEED_BASE from 10300001 to 8900001.
#
# So the bindings are re-verified immediately before scoring. This asserts
# IDENTITY only -- no gate, threshold, bootstrap or cell definition is
# introduced here; all of that stays inherited.
_BINDINGS = {
    "PROTOCOL_ID": "SPPPO_V1",
    "SEED_BASE": 10_300_001,
    "SEED_BLOCK": "10300001..10300192",
    "DEVELOPMENT_SEED": 10_200_001,
}


def _assert_bindings_intact() -> None:
    for name, expected in _BINDINGS.items():
        actual = getattr(evaluator, name)
        if actual != expected:
            raise RuntimeError(
                f"SPPPO evaluator binding clobbered: {name} is {actual!r}, expected "
                f"{expected!r}. Another terminal-evaluation wrapper was imported into "
                "this process and redirected the evaluator. Refusing to score.")
    if evaluator.EXPECTED_HASHES["student"] != (
            "1260e420b85ad01b1aecf433d65688adf153538195d30511e26a3585b7285f1f"):
        raise RuntimeError("SPPPO evaluator student hash clobbered; refusing to score")


if __name__ == "__main__":
    _assert_bindings_intact()
    raise SystemExit(evaluator.main())
