import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from tests.test_tactical_opponents import TestOP9LateGamePressure as T

t = T("test_early_game_trailing_no_pressure")
print("enemy flag x", t._enemy_flag_x())
for label, kwargs in [
    ("early trailing", dict(step=50, red_score=0, blue_score=1)),
    ("late trailing", dict(step=320, red_score=0, blue_score=1)),
    ("late leading", dict(step=320, red_score=1, blue_score=0)),
    ("early leading", dict(step=50, red_score=1, blue_score=0)),
]:
    print(label, t._striker_target_x(**kwargs))
