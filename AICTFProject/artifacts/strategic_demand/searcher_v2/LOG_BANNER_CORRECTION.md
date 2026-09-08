# Correction — stale banner in this run's log

`v2.log` and `launch.log` for this run open with:

```text
J/thresholds imported from V1 unchanged. Not Gate B. No PPO. Block 2500001 untouched.
```

**Both claims in that line are wrong for this run.** The banner text predates the
Observability V2 rewiring and was not updated before launch. The line is left in
the log rather than edited, because the log is a record of what ran.

The correct statements are:

| banner says | actually true for this run |
|---|---|
| "J/thresholds imported from V1 unchanged" | **False.** `J_v2 = p_C − degeneracy_penalty`, payoff enforced as a constraint. V1's `development_eligible` and sentinel gap gate are retired and not imported. |
| "Block 2500001 untouched" | **Stale.** 2500001 is permanently disqualified. The reserved block is `6000001..6000064` and it is untouched. |

What IS imported from V1 unchanged is the *treatment semantics* — `run_episode`,
and `degeneracy_penalty` — which is the property that matters: the search cannot
drift from the environment it is searching.

The source banner has been corrected for future runs. The running process was
not disturbed; it had already loaded the module, so the edit affects only
subsequent launches.

Authority for what this run actually optimized:
- `artifacts/strategic_demand/SEARCH_OBJECTIVE_V2_FROZEN.json`
- `artifacts/strategic_demand/OBSERVABILITY_V2_FROZEN.json`
- `artifacts/strategic_demand/SEARCH_RUN_SDS2_SCREENING_FROZEN.json`
