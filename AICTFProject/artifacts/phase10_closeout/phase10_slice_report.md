# Phase 10 Slice Report

Status: PARTIAL

Implemented in this slice:

- Rewired `experiments/eval_v6i9_map_awareness.py` to build typed config via `rl.evaluation.config.config_from_namespace`.
- Rewired checkpoint dimension reads and native policy loading through `rl.evaluation.policy_loader` / `rl.evaluation.checkpoint`.
- Rewired distribution preflight through `rl.evaluation.preflight`.
- Preserved local compatibility helper names in the monolith for existing tests and later extraction slices.
- Preserved positive fallback semantics for nonpositive `n_macros` / `n_targets` metadata in the shared checkpoint reader.

Equivalence evidence:

- Baseline: `artifacts/phase10_baseline`
- Rewired output: `artifacts/phase10_equivalence_config_loading_preflight`
- Report: `artifacts/phase10_equivalence_config_loading_preflight/equivalence_report.json`
- Result: PASS

Exact matches:

- `episode_results.csv`
- `condition_summary.csv`
- `per_episode.csv`
- `per_condition.csv`
- `final_report.json`
- `summary.json`
- `obstacle_probe.json`
- stable `evaluation_manifest.json` fields, excluding run provenance fields

Validation:

`uv run python -m unittest tests.test_phase10_evaluation_slice tests.test_inference_distribution_contract tests.test_v6i9_map_aware`

Result: PASS, 24 tests.

Remaining Phase 10 work:

- Extract obstacle probes.
- Extract episode runner and matched-seed execution.
- Extract aggregation and gates.
- Extract manifest and artifact writing.
- Add final-verdict and full artifact equivalence tests for each slice.
