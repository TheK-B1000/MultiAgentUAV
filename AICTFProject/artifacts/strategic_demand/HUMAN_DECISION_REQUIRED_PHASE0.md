# HUMAN_DECISION_REQUIRED — Phase 0 complete

V2 is diagnosed and frozen. Nothing further ran.

## Answers

- **Q1 opportunity cost:** CONTEXT_DEPENDENT — no global label
- **Q2 two-agent offense:** YES against FORTRESS; mechanism is THROUGHPUT, not suppression
- **Q3 commitment:** UNRESOLVED
- **Q4 carrier self-sufficiency:** NOT_SUPPORTED

## Proposed V3 intervention

`own_flag_home_required_to_score = True` — single boolean, NOT implemented.

`suppression_range` is removed from all future levers on evidence
(0 events / 256 episodes / all four ranges).

## Artifacts

```
K:\MultiAgentUAV\AICTFProject\artifacts\strategic_demand\V2_MECHANICAL_DIAGNOSTIC.md
K:\MultiAgentUAV\AICTFProject\artifacts\strategic_demand\V2_MECHANICAL_DIAGNOSTIC.json
K:\MultiAgentUAV\AICTFProject\artifacts\strategic_demand\V3_RECOMMENDATION.md
K:\MultiAgentUAV\AICTFProject\artifacts\phase7\tag_saturation.json
K:\MultiAgentUAV\AICTFProject\artifacts\phase7\carrier_return.json
K:\MultiAgentUAV\AICTFProject\artifacts\phase7\interaction_assay.json
K:\MultiAgentUAV\AICTFProject\artifacts\phase7\commitment_assay.json
```

## Not started, awaiting your decision

The strategic-demand searcher (Phases 1-12) was deliberately NOT auto-started.
Its ruleset search space depends on this diagnosis — R5 carrier vulnerability is
explicitly conditional on 7E — so selecting a search space before reading these
results would be choosing the experiment from unread data.

No ruleset change, no PPO training, no specialists, no FP/DO, no latent work.
