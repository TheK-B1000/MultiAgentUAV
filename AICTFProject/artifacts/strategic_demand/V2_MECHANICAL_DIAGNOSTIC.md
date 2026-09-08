# V2 MECHANICAL DIAGNOSTIC

Generated 2026-08-17T13:47:20Z | scope: 2v2 only, map_a, stock RULESET_V2

## Q1_allocation_opportunity_cost

```json
{
  "answer": "CONTEXT_DEPENDENT \u2014 no global label",
  "OP6_FAST_RAID": {
    "defensive_benefit": "PASS (+0.6875, CI95 [+0.3438, +1.0312])",
    "offensive_cost": "FAIL (-0.1875, CI95 [-0.4062, +0.0625])",
    "gate_A": "FAIL_OFFENSE_COST",
    "reading": "holding a defender pays defensively and costs no demonstrated offense"
  },
  "OP7_FORTRESS": {
    "defensive_benefit": "FAIL (+0.1875, CI95 [-0.0938, +0.5000])",
    "offensive_cost": "PASS (+0.9062, CI95 [+0.5938, +1.2188])",
    "gate_A": "FAIL_DEFENSE_BENEFIT",
    "reading": "holding a defender costs ~0.9 captures and buys no demonstrated defense"
  },
  "CONTEXT_DEPENDENT_OFFENSIVE_VALUE": "SUPPORTED",
  "gate_A_overall": "INCOMPLETE \u2014 neither context passes both legs, but the contexts show OPPOSITE structure rather than a shared null"
}
```

## Q2_two_agent_offense_and_mechanism

```json
{
  "answer": "YES against FORTRESS; mechanism is THROUGHPUT, not suppression",
  "suppression_hypothesis": "FALSIFIED",
  "suppression_evidence": "mean_suppressions_of_red = 0.000 in ALL 16 arms (256 episodes) at every range 2.0/2.5/2.75/3.00",
  "conversion": {
    "ONE_DEFENDER_vs_OP7": {
      "breach": 0.75,
      "capture": 0.0625,
      "pickups": 1.3125
    },
    "BOTH_ATTACK_vs_OP7": {
      "breach": 1.0,
      "capture": 0.625,
      "pickups": 6.0625
    }
  },
  "rate_limit_evidence": {
    "median_inter_tag_gap_ONE_DEFENDER": 16.335,
    "median_inter_tag_gap_BOTH_ATTACK": 10.39500000000001,
    "cooldown_floor_seconds": 10.0,
    "post_tag_other_agent_pickup_ONE": 0.0,
    "post_tag_other_agent_pickup_BOTH": 0.28434504792332266,
    "tag_ended_attempt_ONE": 0.905940594059406,
    "tag_ended_attempt_BOTH": 0.7124600638977636
  },
  "reading": "the second attacker does not remove the defence; it outpaces a rate-limited defence. Inter-tag gap compresses toward the 10s floor, the second attacker converts post-tag windows, and a single tag stops the whole attempt far less often.",
  "metric_withdrawn": "cooldown 'at_floor' fraction is vacuous by construction (a tag can only succeed at zero cooldown) and is not used"
}
```

## Q3_commitment_reversibility

```json
{
  "answer": "UNRESOLVED",
  "7D_OP6": "INVALID_TREATMENT_INSTANTIATION",
  "why": "mean t_intent 7.2 preceded mean t_commit 7.3, so the RECOVERY arm approximated ONE_DEFENDER from episode start and the assay degenerated into a Gate 2B re-run",
  "valid_side_finding": "OP6_NO_PRECOMMITMENT_UNCERTAINTY_WINDOW = SUPPORTED under the frozen definitions",
  "caveat": "both t_commit and t_intent are midline-crossing events, so their near-simultaneity is partly structural given symmetric geometry",
  "not_established": "whether a deeper commitment threshold would precede intent"
}
```

## Q4_carrier_self_sufficiency

```json
{
  "answer": "NOT_SUPPORTED",
  "mean_path_efficiency": 4.304825811314066,
  "mean_escort_fraction": 0.5565723666009613,
  "mean_teammate_distance": 5.969448409550035,
  "arms": {
    "OP6": {
      "episodes": 20,
      "possessions": 81,
      "possessions_per_episode": 4.05,
      "total_captures": 58,
      "conversion_per_possession": 0.7160493827160493,
      "mean_duration_steps": 11.25925925925926,
      "mean_path_len": 13.060804009641119,
      "mean_straight_line": 14.2598592896057,
      "mean_path_efficiency": 2.6584830351176136,
      "mean_heading_reversals": 0.07407407407407407,
      "mean_teammate_dist": 4.858587514093161,
      "mean_teammate_min_dist": 2.198540599140542,
      "mean_escort_fraction": 0.6624502887574782,
      "mean_opponent_encounters": 3.5555555555555554,
      "mean_tags_during_possession": 0.4074074074074074,
      "escort_radius_cells": 6.0
    },
    "OP7": {
      "episodes": 20,
      "possessions": 96,
      "possessions_per_episode": 4.8,
      "total_captures": 16,
      "conversion_per_possession": 0.16666666666666666,
      "mean_duration_steps": 4.385416666666667,
      "mean_path_len": 5.048723540810329,
      "mean_straight_line": 14.281738097680217,
      "mean_path_efficiency": 5.9511685875105185,
      "mean_heading_reversals": 0.041666666666666664,
      "mean_teammate_dist": 7.0803093050069075,
      "mean_teammate_min_dist": 5.7040761874344605,
      "mean_escort_fraction": 0.45069444444444445,
      "mean_opponent_encounters": 4.020833333333333,
      "mean_tags_during_possession": 1.03125,
      "escort_radius_cells": 6.0
    }
  },
  "rule": "SUPPORTED when the teammate is near the carrier for a minority of the return; this is descriptive and not a frozen gate"
}
```

## Negative findings preserved

- suppression is dormant in 2v2: 0 events / 256 episodes / all ranges
- suppression_range is therefore removed from all future rule levers
- 7B_ORIGINAL (truncated 1-agent OP7) = INVALID_OPPONENT_INSTANTIATION
- OP7 pre-amendment offense_cost +0.6988 permanently non-gating

**Frozen 2x2:** Gate A INCOMPLETE (context-dependent) x commitment UNRESOLVED -- the frozen 2x2 cannot be applied because neither axis resolved to a single label
