# Phase 6.1 Performance Investigation

Status: **PARTIAL_INVESTIGATION_COMPLETE**

## Findings
- System Python benchmark config and checkpoint hash match between baseline and current; uv environments do not match because baseline uv resolves to Python 3.11 without torch.
- OFF mode was not fully dormant: GPU monitor factory/start and emit_training_started perf_counter executed in OFF. Both were repaired.
- Focused OFF hot-path tests now pass and scientific telemetry invariance tests still pass.
- Component benchmark no longer shows a large rollout/optimization regression; checkpoint_load is slower in current, but complete_rollout and complete_optimization_phase are neutral or faster in the isolated 3-sample run.
- Short repaired 16/64 CUDA smoke is still not a final gate substitute because it uses 3 measured samples; final 10-sample 16/64/256 matrix remains required.

## Next Actions
- Run CUDA_LAUNCH_BLOCKING 256-env OFF repro in a fresh process.
- Run full 10-sample 16/64/256 matrix after 256-env root cause is known.
- If final 10-sample OFF still regresses, proceed to bisect with the stable benchmark harness.