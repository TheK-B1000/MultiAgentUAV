"""R0 — the authoritative opponent-setting path for the frozen V3 benchmark.

Why this exists
---------------
The A pole is OP6 plus a genome overlay (min_alive_for_defender=2). That
overlay is applied by experiments/sds_genome.py::apply_genome_to_core and is
NOT applied anywhere in rl/training/env_factory.py, which only does set_phase /
set_next_opponent / _apply_initial_opponent_params.

So a training run that asks for "pole A" without explicit plumbing would train
against plain OP6 (min_alive_for_defender=3), a genuinely different opponent,
while every log and graph looked perfectly healthy. The inverse is nastier: an
A override left in place while switching to OP7 silently contaminates the B
pole with OP6-derived parameters.

Both failure modes are silent, which is why this module does:

    clear previous override  ->  set base opponent  ->  apply intended overlay
                             ->  ASSERT the resolved profile

and refuses to return if the assertion fails.

What "resolved" means
---------------------
The authority is the live runtime object, not the override attribute:
core._bt_resolved_profile_tensors() is what the behaviour tree actually reads
(gpu_env/_core/_bt_red.py). It applies the override if present, otherwise the
canonical profile for core._opponent_key. Asserting on that tensor dict is the
only check that cannot be fooled by a stale or unapplied override.

Discriminators (why a weak assertion would not catch contamination):

    OP6 canonical : min_alive_for_defender=3, defender_zone_frac=0.35, threat_radius=0.0
    pole A        : min_alive_for_defender=2, defender_zone_frac=0.35, threat_radius=0.0
    OP7 canonical : min_alive_for_defender=2, defender_zone_frac=0.05, threat_radius=12.0

min_alive_for_defender alone does NOT separate pole A from OP7 (both 2), so the
assertion compares every BTProfile field the resolved tensors expose.
"""
from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]

from gpu_env._core._bt_profiles import BTProfile, profile_for_opponent_key  # noqa: E402
from experiments.sds_genome import SDSGenome, apply_genome_to_core           # noqa: E402

CANDIDATE_A = ROOT / "artifacts/strategic_demand/CANDIDATE_A2_SDS2_INIT_3_FROZEN.json"


class OpponentSpecError(RuntimeError):
    """Raised when the live environment does not resolve the requested pole.

    Deliberately fatal. A run that says 'A' but resolves plain OP6 must crash
    rather than produce a beautiful graph of the wrong experiment.
    """


def pole_A_genome() -> SDSGenome:
    """The frozen A pole, read from its freeze record rather than hardcoded."""
    d = json.loads(CANDIDATE_A.read_text(encoding="utf-8"))["candidate_genome"]
    return SDSGenome.from_dict(d)


def pole_B_genome() -> SDSGenome:
    """The frozen B pole: canonical OP7, no overlay."""
    from experiments.sds_genome import canonical_parent
    return canonical_parent("OP7")


def expected_profile(base_key: str, genome: Optional[SDSGenome]) -> BTProfile:
    parent = profile_for_opponent_key(base_key.upper())
    if genome is None or not genome.overlay:
        return parent
    allowed = {f.name for f in fields(BTProfile)}
    kw = {k: v for k, v in genome.overlay.items() if k in allowed}
    return replace(parent, **kw)


def _live_opponent_keys(core) -> set[str]:
    """core._opponent_key is per-env (a list), not a scalar.

    Returned as a set so a mixed-opponent vec env is visible rather than
    stringified into something that trivially fails comparison.
    """
    raw = getattr(core, "_opponent_key", None)
    if raw is None:
        return {"?"}
    if isinstance(raw, (str, bytes)):
        return {str(raw).upper()}
    try:
        return {str(k).upper() for k in raw}
    except TypeError:
        return {str(raw).upper()}


def resolved_profile_scalars(core) -> dict[str, Any]:
    """Read what the behaviour tree will actually use, from the live core."""
    t = core._bt_resolved_profile_tensors()
    out: dict[str, Any] = {}
    for k, v in t.items():
        try:
            out[k] = v.reshape(-1)[0].item()
        except Exception:
            continue
    return out


def assert_opponent_resolved(core, base_key: str,
                             genome: Optional[SDSGenome] = None,
                             *, context: str = "") -> dict[str, Any]:
    """Fatal check that the live env resolves the intended pole.

    Compares every BTProfile field the resolved tensors expose, so a stale
    override from the other pole cannot slip through on a single matching field.
    """
    exp = expected_profile(base_key, genome)
    got = resolved_profile_scalars(core)
    key_live = _live_opponent_keys(core)

    mismatches = []
    for f in fields(BTProfile):
        if f.name not in got:
            continue
        want = getattr(exp, f.name)
        have = got[f.name]
        if isinstance(want, bool):
            ok = bool(have) == bool(want)
        elif isinstance(want, (int,)):
            ok = int(round(float(have))) == int(want)
        elif isinstance(want, float):
            ok = abs(float(have) - float(want)) <= 1e-6
        else:
            continue
        if not ok:
            mismatches.append(f"{f.name}: expected {want!r}, live {have!r}")

    if key_live != {base_key.upper()}:
        mismatches.append(f"_opponent_key: expected all envs {base_key.upper()!r}, "
                          f"live {sorted(key_live)!r}")

    if mismatches:
        raise OpponentSpecError(
            f"OPPONENT SPEC MISMATCH{' (' + context + ')' if context else ''}: "
            f"requested base={base_key} overlay="
            f"{(genome.overlay if genome else {})!r}\n  "
            + "\n  ".join(mismatches)
            + "\nRefusing to continue. A run that reports one opponent and "
              "resolves another invalidates every downstream number.")
    return got


def set_opponent_spec(env, core, base_key: str,
                      genome: Optional[SDSGenome] = None,
                      *, phase: Optional[str] = None,
                      context: str = "") -> dict[str, Any]:
    """clear -> set base -> apply intended overlay -> assert.

    The clear step is what prevents an A override from contaminating a later B
    episode. Because the override survives env.reset() and auto-reset (verified
    on a live core), it does NOT need reapplying every episode -- only whenever
    the opponent specification changes.
    """
    core._bt_profile_override = None
    core._sds_opening_hold_steps = 0

    key = str(base_key).upper()
    env.env_method("set_phase", str(phase or key).upper())
    env.env_method("set_next_opponent", "SCRIPTED", key)

    if genome is not None and genome.overlay:
        apply_genome_to_core(core, genome)
    elif genome is not None:
        core._sds_opening_hold_steps = int(genome.opening_hold_steps)

    return assert_opponent_resolved(core, key, genome, context=context)


def manifest_entry(core, base_key: str,
                   genome: Optional[SDSGenome] = None) -> dict[str, Any]:
    """Resolved-opponent record for the run manifest.

    Written so a future reader can tell which opponent a run ACTUALLY faced,
    rather than which one it claimed to face.
    """
    got = resolved_profile_scalars(core)
    exp = expected_profile(base_key, genome)
    watch = ("min_alive_for_defender", "defender_zone_frac", "threat_radius",
             "enable_defender", "enable_intercept", "lane_amplitude_frac")
    return {
        "requested_base": base_key.upper(),
        "requested_overlay": dict(genome.overlay) if genome else {},
        "genome_id": genome.genome_id if genome else None,
        "live_opponent_key": sorted(_live_opponent_keys(core)),
        "resolved_watch_fields": {k: got.get(k) for k in watch if k in got},
        "expected_watch_fields": {k: getattr(exp, k) for k in watch
                                  if hasattr(exp, k)},
        "assertion": "passed at construction via assert_opponent_resolved",
    }
