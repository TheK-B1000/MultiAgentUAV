from enum import IntEnum

class MacroAction(IntEnum):
    GO_TO = 0
    GRAB_MINE = 1
    GET_FLAG = 2
    PLACE_MINE = 3
    GO_HOME = 4
    # Reserved for DEFEND_PRIMITIVE_AND_HOME_LEGALITY_V1: state-computed DEFEND
    # targets, resolved in gpu_env/_core/_rules.py::_build_targets_from_action
    # exactly like GET_FLAG/GO_HOME. No production n_macros config raises above
    # 5, so no policy action space can ever select these; they are reachable
    # only by a caller that builds a macro-id tensor directly (as the
    # Pyquaticus port contract does), never via the discrete action head.
    DEFEND_FLAG = 5
    DEFEND_OUTWARD = 6
    # DEFEND_SEMANTIC_COMMITMENT_V2: one committed macro whose inward/outward
    # target is re-derived from live state each tick, mirroring the frozen
    # Pyquaticus easy-mode defender exactly (including its standoff radius).
    # This is a Pyquaticus REFERENCE-CONTROLLER adapter, scoped to reproducing
    # that specific external behavior for the port contract. It is NOT a
    # proposal to add a fixed-threshold DEFEND action to the general learned
    # PPO action vocabulary -- a learned policy should be free to choose its
    # own defensive standoff distance, which this macro does not allow. See
    # artifacts/strategic_demand/sppo/DEFEND_SEMANTIC_COMMITMENT_V2_SPEC.json.
    DEFEND = 7
