from enum import IntEnum

class MacroAction(IntEnum):
    GO_TO = 0
    GRAB_MINE = 1
    GET_FLAG = 2
    PLACE_MINE = 3
    GO_HOME = 4
    INTERCEPT_CARRIER = 5
    SUPPRESS_CARRIER = 6
    DEFEND_ZONE = 7
