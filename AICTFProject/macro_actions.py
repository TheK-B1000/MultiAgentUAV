# macro_actions.py
from enum import IntEnum

class MacroAction(IntEnum):
    # Core macro-actions (existing)
    GO_TO              = 0
    GRAB_MINE          = 1
    GET_FLAG           = 2
    PLACE_MINE         = 3
    GO_HOME            = 4
    INTERCEPT_CARRIER  = 5
    SUPPRESS_CARRIER   = 6
    DEFEND_ZONE        = 7

    # NEW strategic macro-actions (for more “tactical” moves)
    PATROL_ENEMY_BASE_LEFT   = 8
    PATROL_ENEMY_BASE_RIGHT  = 9
    PATROL_MID_TOP           = 10
    PATROL_MID_BOTTOM        = 11
    DEFEND_OWN_BASE_LEFT     = 12
    DEFEND_OWN_BASE_RIGHT    = 13

__all__ = ["MacroAction"]
