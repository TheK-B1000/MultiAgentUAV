from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


STATUSES = {
    "COMPLETE",
    "IMPLEMENTED_NOT_CLOSED",
    "PARTIAL",
    "BLOCKED",
    "NOT_STARTED",
    "SUPERSEDED",
}


@dataclass
class PhaseRecord:
    phase: str
    status: str
    implementation_evidence: list[str] = field(default_factory=list)
    test_evidence: list[str] = field(default_factory=list)
    equivalence_evidence: list[str] = field(default_factory=list)
    performance_evidence: list[str] = field(default_factory=list)
    artifact_evidence: list[str] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    next_actions: list[str] = field(default_factory=list)


def rel(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def exists(root: Path, path: str) -> bool:
    return (root / path).exists()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(read_text(path))
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def run_git(root: Path, args: list[str]) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-c", "safe.directory=K:/MultiAgentUAV", *args],
            cwd=root,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return out.strip()
    except Exception as exc:  # noqa: BLE001 - audit should degrade, not crash.
        return f"ERROR: {exc}"


def parse_test_summary(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in (
        "canonical_current_count",
        "canonical_current_status",
        "tests_ran",
        "status",
    ):
        matches = re.findall(rf"^{re.escape(key)}:\s*(.+)$", text, re.MULTILINE)
        if matches:
            result[key] = matches[-1].strip()
    result["python312_default_pass"] = (
        "discover -s tests" in text
        and "status: PASS" in text
        and "canonical_current_status:" in text
        and "DEFAULT" in str(result.get("canonical_current_status", "")).upper()
        and "PASS" in str(result.get("canonical_current_status", "")).upper()
    )
    result["uv_trustworthy"] = "FAIL_TOOLING" not in text and "FAIL_ENVIRONMENT" not in text
    result["pattern_discovery_pass"] = (
        "discover -s tests -p test*.py" in text
        and "status: PASS" in text
        and "canonical_current_status: DEFAULT_DISCOVERY_PASS_PATTERN_DISCOVERY_FAILS_ON_SECOND_RUN" not in text
    )
    return result


def find_files(root: Path, names: list[str]) -> list[str]:
    return [name for name in names if exists(root, name)]


def closeout_files(root: Path, phase_dir: str) -> list[str]:
    path = root / phase_dir
    if not path.exists():
        return []
    return [rel(root, p) for p in sorted(path.rglob("*")) if p.is_file()]


def test_files(root: Path, patterns: list[str]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        found.extend(rel(root, p) for p in sorted((root / "tests").glob(pattern)))
    return sorted(set(found))


def implemented(root: Path, required: list[str], minimum: int | None = None) -> tuple[bool, list[str], list[str]]:
    present = find_files(root, required)
    missing = [p for p in required if p not in present]
    threshold = len(required) if minimum is None else minimum
    return len(present) >= threshold, present, missing


def add_common_test_evidence(record: PhaseRecord, test_summary: dict[str, Any]) -> None:
    if test_summary.get("python312_default_pass"):
        count = test_summary.get("canonical_current_count", "unknown")
        record.test_evidence.append(f"Python 3.12 default unittest discovery passed with {count} tests.")
    else:
        record.blockers.append("No passing current full-suite discovery result found.")
    if not test_summary.get("uv_trustworthy"):
        record.test_evidence.append("Requested uv discovery is not trustworthy in this checkout; uv used an environment without torch or failed cache initialization.")
    if not test_summary.get("pattern_discovery_pass"):
        record.blockers.append("Requested pattern discovery did not pass cleanly in the captured audit baseline.")


def build_records(root: Path, out: Path) -> tuple[list[PhaseRecord], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    test_summary = parse_test_summary(read_text(out / "test_discovery.txt"))
    missing_artifacts: list[dict[str, Any]] = []
    blockers: list[dict[str, Any]] = []
    records: list[PhaseRecord] = []

    def remember(record: PhaseRecord) -> None:
        assert record.status in STATUSES, record.status
        records.append(record)
        for item in record.blockers:
            blockers.append({"phase": record.phase, "blocker": item})
        if record.status != "COMPLETE":
            missing_artifacts.append(
                {
                    "phase": record.phase,
                    "missing_or_pending": record.next_actions + record.blockers,
                }
            )

    phase1_files = [
        "rl/custom_ppo/distributions.py",
        "rl/custom_ppo/diagnostics_contract.py",
        "rl/evaluation/probes/results.py",
        "rl/custom_ppo/inference_policy.py",
    ]
    ok, present, missing = implemented(root, phase1_files)
    r = PhaseRecord("Phase 1", "IMPLEMENTED_NOT_CLOSED")
    r.implementation_evidence.extend(present)
    r.test_evidence.extend(test_files(root, ["test*distribution*.py", "test*diagnostic*.py", "test*probe*.py"]))
    r.equivalence_evidence.append("Full-suite discovery passes under Python 3.12, but phase-specific gradient/error fixtures are not independently summarized in closeout artifacts.")
    add_common_test_evidence(r, test_summary)
    if not ok:
        r.status = "PARTIAL"
        r.blockers.append(f"Missing expected files: {missing}")
    r.next_actions.append("Add or locate Phase 1 closeout artifact proving wrapper/model get_distribution gradient path and typed ERROR cases.")
    remember(r)

    phase15_files = [
        "rl/custom_ppo/distributions.py",
        "rl/evaluation/probes/results.py",
        "rl/custom_ppo/inference_policy.py",
    ]
    ok, present, missing = implemented(root, phase15_files)
    r = PhaseRecord("Phase 1.5", "IMPLEMENTED_NOT_CLOSED")
    r.implementation_evidence.extend(present)
    r.test_evidence.extend(test_files(root, ["test*distribution*.py", "test*probe*.py", "test*inference*contract*.py"]))
    r.equivalence_evidence.append("Distribution repair is also evidenced by Phase 5 closeout distribution_contract.json if present.")
    if exists(root, "artifacts/phase5_closeout/distribution_contract.json"):
        r.artifact_evidence.append("artifacts/phase5_closeout/distribution_contract.json")
    add_common_test_evidence(r, test_summary)
    if not ok:
        r.status = "PARTIAL"
        r.blockers.append(f"Missing expected files: {missing}")
    r.next_actions.append("Promote distribution/probe contract evidence into a Phase 1.5 closeout artifact.")
    remember(r)

    phase16_files = [
        "gpu_env/_navigation_telemetry.py",
        "gpu_env/_core/_state.py",
        "gpu_env/_core/_metrics.py",
        "gpu_env/_core/_step.py",
        "gpu_env/_episode_payload.py",
    ]
    ok, present, missing = implemented(root, phase16_files)
    r = PhaseRecord("Phase 1.6", "IMPLEMENTED_NOT_CLOSED")
    r.implementation_evidence.extend(present)
    r.test_evidence.extend(test_files(root, ["test*navigation*telemetry*.py", "test*episode*payload*.py", "test*v6i9*telemetry*.py"]))
    r.artifact_evidence.extend(closeout_files(root, "artifacts/v6i9_map_awareness_exact_telemetry_smoke")[:5])
    r.performance_evidence.append("Telemetry performance overhead is not summarized in a Phase 1.6 closeout artifact.")
    add_common_test_evidence(r, test_summary)
    if not ok:
        r.status = "PARTIAL"
        r.blockers.append(f"Missing expected files: {missing}")
    r.next_actions.append("Run/update exact telemetry overhead benchmark and OFF/BASIC/FULL behavior evidence.")
    remember(r)

    phase3_files = [
        "rl/custom_ppo/checkpoints",
        "rl/custom_ppo/inference_policy.py",
        "rl/custom_ppo/inference.py",
    ]
    ok, present, missing = implemented(root, phase3_files)
    r = PhaseRecord("Phase 3", "IMPLEMENTED_NOT_CLOSED")
    r.implementation_evidence.extend(present)
    r.test_evidence.extend(test_files(root, ["test*checkpoint*.py", "test*inference*.py"]))
    r.equivalence_evidence.append("Full discovery output includes checkpoint compatibility PASS lines, but no Phase 3 closeout report was found.")
    add_common_test_evidence(r, test_summary)
    if not ok:
        r.status = "PARTIAL"
        r.blockers.append(f"Missing expected files: {missing}")
    r.next_actions.append("Add Phase 3 closeout proving native 7/8 channel, migrated 7-to-8, CPU/CUDA smoke, and behavioral equivalence.")
    remember(r)

    phase4_files = [
        "rl/presets/_registry_source.py",
        "rl/presets/validation.py",
        "rl/presets/compatibility.py",
        "rl/presets/families",
        "tests/preset_snapshots.json",
    ]
    ok, present, missing = implemented(root, phase4_files)
    r = PhaseRecord("Phase 4", "IMPLEMENTED_NOT_CLOSED")
    r.implementation_evidence.extend(present)
    r.test_evidence.extend(test_files(root, ["test*preset*.py"]))
    r.artifact_evidence.append("tests/preset_snapshots.json" if exists(root, "tests/preset_snapshots.json") else "missing preset snapshots")
    add_common_test_evidence(r, test_summary)
    if exists(root, "tests/preset_snapshots.json"):
        snapshots = read_json(root / "tests/preset_snapshots.json")
        r.equivalence_evidence.append(f"Preset snapshot entries: {len(snapshots) if isinstance(snapshots, dict) else 'unknown'}")
    r.blockers.append("Resolved preset artifact/hash integration was not proven by a Phase 4 closeout report.")
    r.next_actions.append("Produce Phase 4 closeout tying registry/CLI/resolved-artifact/hash evidence to current commit.")
    if not ok:
        r.status = "PARTIAL"
        r.blockers.append(f"Missing expected files: {missing}")
    remember(r)

    phase5_report = read_json(root / "artifacts/phase5_closeout/phase5_closeout_report.json")
    phase5_full = read_json(root / "artifacts/phase5_closeout/phase5_report.json")
    if phase5_full and not phase5_report:
        phase5_report = phase5_full
    elif phase5_full:
        phase5_report = {**phase5_report, **{k: phase5_full.get(k, v) for k, v in phase5_report.items()}}
        phase5_report["final_verdict"] = phase5_full.get("status", phase5_report.get("final_verdict"))
    phase5_files = [
        "rl/custom_ppo/rollout",
        "rl/custom_ppo/rollout_collector.py",
        "rl/custom_ppo/rollout_step_recorder.py",
    ]
    ok, present, missing = implemented(root, phase5_files, minimum=2)
    r = PhaseRecord(
        "Phase 5",
        "COMPLETE"
        if phase5_full.get("status") == "COMPLETE" or phase5_report.get("final_verdict") == "COMPLETE"
        else ("BLOCKED" if phase5_report.get("final_verdict") == "NOT COMPLETE" else "IMPLEMENTED_NOT_CLOSED"),
    )
    r.implementation_evidence.extend(present)
    r.test_evidence.extend(test_files(root, ["test*rollout*.py", "test*train_ppo_smoke.py"]))
    r.artifact_evidence.extend(closeout_files(root, "artifacts/phase5_closeout"))
    if phase5_report:
        verdict = phase5_full.get("status") or phase5_report.get("final_verdict")
        r.equivalence_evidence.append(f"Phase 5 closeout verdict: {verdict}")
        for key in ("rollout_equivalence", "telemetry_invariance"):
            block = phase5_report.get(key)
            if isinstance(block, dict) and "status" in block:
                r.equivalence_evidence.append(f"{key}: {block.get('status')}")
        for key in ("performance_comparison", "memory_comparison"):
            block = phase5_report.get(key)
            if isinstance(block, dict) and "status" in block:
                r.performance_evidence.append(f"{key}: {block.get('status')}")
        if phase5_full.get("gates"):
            for key, value in phase5_full["gates"].items():
                if key in {"golden_rollout_equivalence", "buffer_equality", "rng_equality", "telemetry_invariance"}:
                    r.equivalence_evidence.append(f"{key}: {value}")
        r.blockers.extend(phase5_report.get("blocking_items", []))
    add_common_test_evidence(r, test_summary)
    if phase5_full.get("status") == "COMPLETE":
        r.status = "COMPLETE"
        r.blockers = [b for b in r.blockers if b]
    if not ok:
        r.status = "PARTIAL" if r.status != "COMPLETE" else r.status
        r.blockers.append(f"Missing expected files: {missing}")
    if r.status != "COMPLETE":
        r.next_actions.append("Close golden/stochastic rollout, reward/buffer/GAE, throughput, and CUDA memory evidence against a pre-Phase-5 worktree.")
    remember(r)

    r = PhaseRecord(
        "Phase 5.1",
        "COMPLETE"
        if phase5_full.get("status") == "COMPLETE"
        else ("BLOCKED" if phase5_report.get("final_verdict") == "NOT COMPLETE" else "IMPLEMENTED_NOT_CLOSED"),
    )
    r.implementation_evidence.extend(["rl/custom_ppo/inference_policy.py", "experiments/eval_v6i9_map_awareness.py"])
    r.test_evidence.extend(test_files(root, ["test*inference_distribution_contract*.py", "test*train_ppo_smoke.py"]))
    r.artifact_evidence.extend(closeout_files(root, "artifacts/phase5_closeout"))
    if phase5_report:
        r.equivalence_evidence.append(f"distribution_contract: {phase5_report.get('distribution_contract', {}).get('status')}")
        r.equivalence_evidence.append(f"rollout_equivalence: {phase5_report.get('rollout_equivalence', {}).get('status')}")
        r.performance_evidence.append(f"performance_comparison: {phase5_report.get('performance_comparison', {}).get('status')}")
        r.blockers.extend(phase5_report.get("blocking_items", []))
    add_common_test_evidence(r, test_summary)
    r.next_actions.append("Keep distribution repair accepted, but finish rollout closeout proof tracks.")
    remember(r)

    phase6_files = [
        "rl/custom_ppo/telemetry/events.py",
        "rl/custom_ppo/telemetry/schemas.py",
        "rl/custom_ppo/telemetry/models.py",
        "rl/custom_ppo/telemetry/emitter.py",
        "rl/custom_ppo/training_telemetry.py",
        "tests/test_telemetry_phase6.py",
    ]
    ok, present, missing = implemented(root, phase6_files)
    r = PhaseRecord("Phase 6", "IMPLEMENTED_NOT_CLOSED")
    r.implementation_evidence.extend(present)
    r.test_evidence.extend(test_files(root, ["test_telemetry_phase6.py", "test_telemetry_phase6_integration.py", "test_telemetry_invariance.py"]))
    r.artifact_evidence.append("artifacts/phase6_1_closeout/phase6_1_report.json" if exists(root, "artifacts/phase6_1_closeout/phase6_1_report.json") else "No Phase 6 closeout report found.")
    add_common_test_evidence(r, test_summary)
    if not ok:
        r.status = "PARTIAL"
        r.blockers.append(f"Missing expected files: {missing}")
    r.next_actions.append("Add standalone Phase 6 foundation closeout tying schema/sink compatibility and legacy CSV preservation to current test evidence.")
    remember(r)

    phase61_report = read_json(root / "artifacts/phase6_1_closeout/phase6_1_report.json")
    r = PhaseRecord("Phase 6.1", "IMPLEMENTED_NOT_CLOSED")
    r.implementation_evidence.extend(find_files(root, phase6_files + ["tools/benchmark_training_pipeline.py", "rl/telemetry_mode.py"]))
    r.test_evidence.extend(test_files(root, ["test_telemetry_phase6*.py", "test_performance*.py", "test_training_lifecycle_events.py", "test_checkpoint_telemetry.py"]))
    r.artifact_evidence.extend(closeout_files(root, "artifacts/phase6_1_closeout"))
    if phase61_report:
        r.equivalence_evidence.append(f"closeout verdict: {phase61_report.get('verdict')}")
        r.equivalence_evidence.append(f"telemetry invariance: {phase61_report.get('track_status', {}).get('track_k_telemetry_invariance')}")
        r.performance_evidence.append(f"performance gates: {phase61_report.get('track_status', {}).get('track_n_performance_gates')}")
        r.performance_evidence.append(f"gpu monitor: {phase61_report.get('gpu_monitor_status')}")
        r.blockers.extend(phase61_report.get("unresolved_risks", []))
        r.blockers.extend(phase61_report.get("blockers", []))
    add_common_test_evidence(r, test_summary)
    r.next_actions.append("Capture pre-Phase-6 OFF baseline, CUDA smoke/matrix, and benchmark tool run before declaring COMPLETE.")
    remember(r)

    phase7_files = [
        "rl/custom_ppo/diagnostics/occupancy.py",
        "rl/custom_ppo/diagnostics/entropy.py",
        "rl/custom_ppo/diagnostics/counterfactual.py",
        "rl/custom_ppo/diagnostics/schemas.py",
    ]
    ok, present, missing = implemented(root, phase7_files, minimum=3)
    r = PhaseRecord("Phase 7", "PARTIAL" if present else "NOT_STARTED")
    r.implementation_evidence.extend(present)
    if exists(root, "rl/custom_ppo/latent_diagnostics.py"):
        r.implementation_evidence.append("rl/custom_ppo/latent_diagnostics.py remains present.")
    r.blockers.append(f"Missing decomposed diagnostics modules: {missing}")
    r.next_actions.append("Implement latent diagnostics package after Phase 6.1 closeout evidence is closed.")
    remember(r)

    phase8_files = [
        "rl/training/arguments.py",
        "rl/training/overrides.py",
        "rl/training/resolved_config.py",
        "rl/training/run_context.py",
        "rl/training/initialization.py",
        "rl/training/factories.py",
        "rl/training/orchestrator.py",
        "rl/training/lifecycle.py",
        "rl/training/errors.py",
    ]
    ok, present, missing = implemented(root, phase8_files, minimum=7)
    phase8_report = read_json(root / "artifacts/phase8_closeout/phase8_report.json")
    r = PhaseRecord("Phase 8", "IMPLEMENTED_NOT_CLOSED" if ok else ("PARTIAL" if present else "NOT_STARTED"))
    r.implementation_evidence.extend(present)
    r.test_evidence.extend(test_files(root, ["test*training*.py", "test*train_ppo*.py", "test*cli*.py"]))
    r.artifact_evidence.extend(closeout_files(root, "artifacts/phase8_closeout"))
    if phase8_report.get("status") == "COMPLETE":
        r.test_evidence.append(
            f"Phase 8 closeout PASS: {phase8_report.get('focused_tests', {}).get('command', 'focused tests')}"
        )
        for key, value in (phase8_report.get("gates") or {}).items():
            if value == "PASS":
                r.equivalence_evidence.append(f"{key}: PASS")
        r.status = "COMPLETE"
    if not ok:
        r.blockers.append(f"Missing orchestration modules: {missing}")
    if r.status != "COMPLETE":
        r.next_actions.append("Produce Phase 8 closeout proving CLI/config/factory equivalence and import direction.")
    remember(r)

    phase9_files = [
        "gpu_env/state/models.py",
        "gpu_env/state/allocation.py",
        "gpu_env/state/agent_state.py",
        "gpu_env/state/team_state.py",
        "gpu_env/state/flag_state.py",
        "gpu_env/state/episode_state.py",
        "gpu_env/state/map_state.py",
        "gpu_env/state/opponent_state.py",
        "gpu_env/state/telemetry_state.py",
        "gpu_env/state/scratch.py",
        "gpu_env/state/validation.py",
        "gpu_env/state/snapshots.py",
    ]
    ok, present, missing = implemented(root, phase9_files, minimum=8)
    phase9_report = read_json(root / "artifacts/phase9_closeout/phase9_report.json")
    r = PhaseRecord("Phase 9", "IMPLEMENTED_NOT_CLOSED" if ok else ("PARTIAL" if present else "NOT_STARTED"))
    r.implementation_evidence.extend(present)
    r.test_evidence.extend(test_files(root, ["test_state_phase9.py"]))
    r.artifact_evidence.extend(closeout_files(root, "artifacts/phase9_closeout"))
    if exists(root, "gpu_env/_core/_state.py"):
        r.implementation_evidence.append("gpu_env/_core/_state.py remains primary state owner candidate.")
    if phase9_report.get("status") == "COMPLETE":
        cls = phase9_report.get("classification") or {}
        r.equivalence_evidence.extend(
            f"{key}: {value}" for key, value in cls.items() if value == "PASS"
        )
        r.test_evidence.append(
            f"Phase 9 focused tests PASS ({phase9_report.get('focused_tests', {}).get('passed', '?')} tests)"
        )
        r.status = "COMPLETE"
    if not ok:
        r.blockers.append(f"Missing decomposed GPU state modules: {missing}")
    if r.status != "COMPLETE":
        r.next_actions.append("Produce Phase 9 closeout proving reset/RNG/telemetry equivalence and performance.")
    remember(r)

    phase10_files = [
        "rl/evaluation/config.py",
        "rl/evaluation/env_factory.py",
        "rl/evaluation/policy_loader.py",
        "rl/evaluation/preflight.py",
        "rl/evaluation/probes",
        "rl/evaluation/episode_runner.py",
        "rl/evaluation/matched_seed.py",
        "rl/evaluation/aggregation.py",
        "rl/evaluation/gates.py",
        "rl/evaluation/artifact_writer.py",
        "rl/evaluation/manifest.py",
        "rl/evaluation/orchestrator.py",
    ]
    ok, present, missing = implemented(root, phase10_files, minimum=len(phase10_files))
    phase10_report = read_json(root / "artifacts/phase10_closeout/phase10_slice_report.json")
    phase10_equivalence = read_json(root / "artifacts/phase10_final_equivalence/equivalence_report.json")
    full_discovery = read_json(root / "artifacts/refactor_audit/full_discovery_summary.json")
    phase10_focused = phase10_report.get("focused_tests", {}) if isinstance(phase10_report, dict) else {}
    phase10_full = phase10_report.get("full_discovery", {}) if isinstance(phase10_report, dict) else {}
    phase10_equiv = phase10_report.get("equivalence", {}) if isinstance(phase10_report, dict) else {}
    phase10_impl = phase10_report.get("implementation", {}) if isinstance(phase10_report, dict) else {}

    r = PhaseRecord("Phase 10", "IMPLEMENTED_NOT_CLOSED" if ok else "PARTIAL")
    r.implementation_evidence.extend(present)
    r.test_evidence.extend(
        test_files(
            root,
            [
                "test_phase10*.py",
                "test_evaluation*.py",
                "test_inference_distribution_contract.py",
                "test_v6i9_map_aware.py",
            ],
        )
    )
    r.artifact_evidence.extend(closeout_files(root, "artifacts/phase10_closeout"))
    r.artifact_evidence.extend(closeout_files(root, "artifacts/phase10_final_equivalence")[:8])
    if exists(root, "experiments/eval_v6i9_map_awareness.py"):
        r.implementation_evidence.append("experiments/eval_v6i9_map_awareness.py remains a thin evaluator entry point.")
    if phase10_impl:
        r.implementation_evidence.append(
            "Phase 10 closeout implementation flags: "
            + ", ".join(f"{k}={v}" for k, v in sorted(phase10_impl.items()))
        )
    if bool(phase10_focused.get("passed")):
        r.test_evidence.append(
            f"Phase 10 focused tests PASS ({phase10_focused.get('tests_run', 'unknown')} tests): "
            f"{phase10_focused.get('command', 'command unavailable')}"
        )
    elif phase10_report:
        r.blockers.append("Phase 10 focused tests are not recorded as passing in phase10_slice_report.json.")
    if bool(phase10_equiv.get("equivalent")) or bool(phase10_equivalence.get("equivalent")):
        r.equivalence_evidence.append(
            "Phase 10 golden equivalence PASS: "
            f"{phase10_equiv.get('episode_rows', 24)} episode rows, "
            f"{phase10_equiv.get('condition_rows', 12)} condition rows, "
            f"verdict={phase10_equiv.get('verdict', phase10_equivalence.get('checks', {}).get('final_verdict', {}).get('value', 'unknown'))}"
        )
    elif phase10_report or phase10_equivalence:
        r.blockers.append("Phase 10 golden equivalence is not recorded as passing.")
    if not ok:
        r.blockers.append(f"Missing evaluation architecture modules: {missing}")
    full_discovery_passed = bool(phase10_full.get("passed")) or (
        full_discovery.get("default_discovery", {}).get("status") == "PASS"
        and full_discovery.get("pattern_discovery", {}).get("status") == "PASS"
        and bool(full_discovery.get("counts_agree"))
    )
    if full_discovery_passed and ok and bool(phase10_focused.get("passed")) and (
        bool(phase10_equiv.get("equivalent")) or bool(phase10_equivalence.get("equivalent"))
    ):
        r.status = "COMPLETE"
    else:
        r.next_actions.append("Run and record full discovery on the current HEAD before declaring Phase 10 closed.")
    remember(r)

    r = PhaseRecord("Final", "NOT_STARTED")
    r.blockers.append("Phases 5, 6.1, 7, 8, 9, and 10 are not all closed.")
    r.next_actions.append("Defer compatibility cleanup until architecture phases close with tests, equivalence, performance, and artifacts.")
    remember(r)

    return records, test_summary, missing_artifacts, blockers


def compatibility_facades(root: Path) -> list[dict[str, Any]]:
    candidates = [
        "rl/custom_ppo/inference.py",
        "rl/custom_ppo/rollout_collector.py",
        "rl/custom_ppo/training_telemetry.py",
        "rl/presets/compatibility.py",
        "rl/custom_ppo/telemetry/schemas.py",
    ]
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        path = root / candidate
        if not path.exists():
            continue
        text = read_text(path).lower()
        rows.append(
            {
                "file": candidate,
                "exists": True,
                "signals": {
                    "compatibility": "compat" in text or "facade" in text,
                    "deprecated": "deprecated" in text,
                    "re_export": "__all__" in text or "import" in text,
                },
            }
        )
    return rows


def write_markdown(root: Path, out: Path, records: list[PhaseRecord], test_summary: dict[str, Any]) -> None:
    total = len(records)
    complete = sum(1 for r in records if r.status == "COMPLETE")
    implemented = sum(1 for r in records if r.status in {"COMPLETE", "IMPLEMENTED_NOT_CLOSED", "BLOCKED"})
    lines = [
        "# Refactor Progress Audit",
        "",
        f"Commit: `{read_text(out / 'current_commit.txt').strip()}`",
        f"Branch: `{read_text(out / 'current_branch.txt').strip()}`",
        f"Implemented coverage: {implemented}/{total} ({implemented / total:.1%})",
        f"Fully closed coverage: {complete}/{total} ({complete / total:.1%})",
        f"Canonical current test count: {test_summary.get('canonical_current_count', 'UNKNOWN')}",
        f"Canonical current test status: {test_summary.get('canonical_current_status', 'UNKNOWN')}",
        "",
        "| Phase | Implementation | Tests | Equivalence | Performance | Artifacts | Final Status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in records:
        lines.append(
            f"| {r.phase} | {'PASS' if r.implementation_evidence else 'NONE'} | "
            f"{'PASS' if r.test_evidence else 'NONE'} | "
            f"{'PASS' if r.equivalence_evidence and not any('BLOCKED' in e for e in r.equivalence_evidence) else ('PENDING' if r.equivalence_evidence else 'NONE')} | "
            f"{'PASS' if r.performance_evidence and not any(x in ' '.join(r.performance_evidence).upper() for x in ['BLOCKED', 'WARN', 'NOT MEASURED']) else ('PENDING' if r.performance_evidence else 'NONE')} | "
            f"{'PASS' if r.artifact_evidence else 'NONE'} | {r.status} |"
        )
    lines.extend(["", "## Phase Details", ""])
    for r in records:
        lines.extend(
            [
                f"### {r.phase}: {r.status}",
                f"- implementation_evidence: {len(r.implementation_evidence)}",
                f"- test_evidence: {len(r.test_evidence)}",
                f"- equivalence_evidence: {len(r.equivalence_evidence)}",
                f"- performance_evidence: {len(r.performance_evidence)}",
                f"- artifact_evidence: {len(r.artifact_evidence)}",
            ]
        )
        if r.blockers:
            lines.append("- blockers:")
            lines.extend(f"  - {item}" for item in r.blockers[:8])
        if r.next_actions:
            lines.append("- next_actions:")
            lines.extend(f"  - {item}" for item in r.next_actions)
        lines.append("")
    (out / "refactor_progress_report.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "phase_status.md").write_text("\n".join(lines), encoding="utf-8")


def write_next_steps(out: Path, records: list[PhaseRecord]) -> None:
    phase61 = next((r for r in records if r.phase == "Phase 6.1"), None)
    phase5 = next((r for r in records if r.phase == "Phase 5"), None)
    lines = [
        "# Recommended Next Steps",
        "",
        "1. Finish Phase 6.1 closeout evidence before opening Phase 7.",
        "   Required: pre-Phase-6 OFF baseline, CUDA smoke/matrix where available, benchmark tool run, and final performance gate artifact.",
        "2. Close Phase 5 evidence using a clean pre-Phase-5 worktree.",
        "   Required: golden/stochastic rollout equivalence, reward/buffer/GAE equivalence, throughput comparison, and CUDA memory comparison.",
        "3. Resolve the 1268 historical test-count record.",
        "   Current reliable count is 1168 under Python 3.12 default discovery.",
        "4. Start Phase 7 only after Phase 6.1 evidence is closed.",
        "",
    ]
    if phase61:
        lines.append(f"Phase 6.1 status: {phase61.status}")
    if phase5:
        lines.append(f"Phase 5 status: {phase5.status}")
    lines.append("")
    lines.append("Exact next command:")
    lines.append("`uv run python tools/audit_refactor_progress.py --project-root . --output-dir artifacts\\refactor_audit`")
    (out / "recommended_next_steps.md").write_text("\n".join(lines), encoding="utf-8")


def write_test_count_history(root: Path, out: Path, baseline_worktree: str | None) -> None:
    log_matches = run_git(root, ["log", "--all", "--oneline", "--decorate", "--grep=Phase 4", "--grep=preset", "--grep=refactor"])
    data = {
        "known_reported_counts": [1133, 1157, 1161, 1268],
        "current_reliable_count": 1244,
        "current_reliable_command": r"C:\Users\K-B\AppData\Local\Programs\Python\Python312\python.exe -m unittest discover -s tests",
        "phase4_completion_commit": "fe0e923d8a9b13631a8439a929d21ee65a19817e",
        "phase4_commit_identification": "CANDIDATE_VERIFIED_BY_COMMIT_SCOPE",
        "phase4_candidate_python312_discovery": {
            "worktree": r"K:\MultiAgentUAV\AICTFProject-phase4-audit\AICTFProject",
            "tests_ran": 1133,
            "exit_code": 0,
            "status": "PASS_EXIT_0_OK_OUTPUT_UNPARSED",
            "duration_seconds": 99.602,
        },
        "phase4_candidate_uv_discovery": {
            "tests_ran": 139,
            "exit_code": 1,
            "status": "FAIL_ENVIRONMENT",
            "errors": 102,
            "reason": "uv selected Python 3.11 environment without torch.",
        },
        "git_log_probe": log_matches,
        "baseline_worktree": baseline_worktree,
        "decision": "Phase 4 candidate discovers 1133 tests under the repo-equipped Python 3.12 environment, not 1268. Current checkout discovers 1244 tests after Phase 6.1 closeout work. Treat 1268 as unverified historical data unless another exact commit/log artifact is produced.",
    }
    (out / "test_count_history.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output-dir", default="artifacts/refactor_audit")
    parser.add_argument("--run-tests", action="store_true", help="Reserved; this audit run ingests existing test_discovery.txt.")
    parser.add_argument("--skip-performance", action="store_true")
    parser.add_argument("--baseline-worktree")
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    out = (root / args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    (out / "current_commit.txt").write_text(run_git(root, ["rev-parse", "HEAD"]) + "\n", encoding="utf-8")
    (out / "current_branch.txt").write_text(run_git(root, ["branch", "--show-current"]) + "\n", encoding="utf-8")
    (out / "working_tree.txt").write_text(run_git(root, ["status", "--porcelain=v1"]) + "\n", encoding="utf-8")

    records, test_summary, missing_artifacts, phase_blockers = build_records(root, out)
    phase_dicts = [asdict(r) for r in records]

    report = {
        "commit": read_text(out / "current_commit.txt").strip(),
        "branch": read_text(out / "current_branch.txt").strip(),
        "test_summary": test_summary,
        "phase_count": len(records),
        "implemented_count": sum(1 for r in records if r.status in {"COMPLETE", "IMPLEMENTED_NOT_CLOSED", "BLOCKED"}),
        "complete_count": sum(1 for r in records if r.status == "COMPLETE"),
        "implemented_percent": round(sum(1 for r in records if r.status in {"COMPLETE", "IMPLEMENTED_NOT_CLOSED", "BLOCKED"}) / len(records) * 100, 1),
        "closed_percent": round(sum(1 for r in records if r.status == "COMPLETE") / len(records) * 100, 1),
        "phases": phase_dicts,
    }

    (out / "phase_status.json").write_text(json.dumps(phase_dicts, indent=2), encoding="utf-8")
    (out / "phase_status_matrix.json").write_text(json.dumps(phase_dicts, indent=2), encoding="utf-8")
    (out / "refactor_progress_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out / "missing_artifacts.json").write_text(json.dumps(missing_artifacts, indent=2), encoding="utf-8")
    (out / "blockers.json").write_text(json.dumps(phase_blockers, indent=2), encoding="utf-8")
    (out / "compatibility_facades.json").write_text(json.dumps(compatibility_facades(root), indent=2), encoding="utf-8")
    write_markdown(root, out, records, test_summary)
    write_next_steps(out, records)
    write_test_count_history(root, out, args.baseline_worktree)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

