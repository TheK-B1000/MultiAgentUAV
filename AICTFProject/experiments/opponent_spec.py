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


def _with_full_team_defender_gate(g: SDSGenome, n_agents: int) -> SDSGenome:
    """Apply SIZE_NORMALIZED_POLE_SEMANTICS_SPEC.json's defender-gate normalization.

    ``min_alive_for_defender`` is compared against an absolute ``alive_count`` in
    ``gpu_env/_core/_bt_red.py``. Pole A's frozen genome records its intent explicitly --
    "the defender only deploys while at least 2 RED agents are alive" -- which at 2v2 IS
    the whole red team. Carrying the literal 2 to 6v6 would leave the defender active
    until four teammates had died, silently destroying the pole's conditional character.
    So the invariant meaning is ``min_alive_for_defender = N``.

    At ``n_agents == 2`` this is a NO-OP by construction (Pole A's overlay is already 2 and
    canonical OP7 is natively 2), and the genome is returned untouched so the certified 2v2
    path cannot drift.
    """
    n = int(n_agents)
    if n == 2:
        return g
    overlay = dict(g.overlay or {})
    overlay["min_alive_for_defender"] = n
    return replace(g, overlay=overlay)


def pole_A_genome(n_agents: int = 2) -> SDSGenome:
    """The frozen A pole, read from its freeze record rather than hardcoded.

    The default ``n_agents=2`` reproduces the freeze record exactly; larger sizes carry the
    size-normalized defender gate.
    """
    d = json.loads(CANDIDATE_A.read_text(encoding="utf-8"))["candidate_genome"]
    return _with_full_team_defender_gate(SDSGenome.from_dict(d), n_agents)


def pole_B_genome(n_agents: int = 2) -> SDSGenome:
    """The frozen B pole: canonical OP7.

    At 2v2 this is the canonical parent with no overlay, exactly as before. At larger sizes
    an overlay is required because canonical OP7 carries min_alive_for_defender=2 natively;
    that structural change is recorded in SIZE_NORMALIZED_POLE_SEMANTICS_SPEC.json. The
    canonical OP6-OP12 registry is never overwritten.
    """
    from experiments.sds_genome import canonical_parent
    return _with_full_team_defender_gate(canonical_parent("OP7"), n_agents)


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


def install_keyed_opponent_overlays(core, overlays: dict[str, SDSGenome]) -> None:
    """Install overlays keyed by the live opponent tag.

    The behaviour tree resolves this mapping on every call from
    ``core._opponent_key``. It therefore stays synchronized with PPO's
    per-environment opponent sampler across auto-resets without a second
    callback or a stale per-env list.
    """
    resolved: dict[str, BTProfile] = {}
    opening_holds = set()
    for raw_key, genome in overlays.items():
        key = str(raw_key).upper()
        if str(genome.base_opponent).upper() != key:
            raise OpponentSpecError(
                f"Overlay {genome.genome_id!r} is based on {genome.base_opponent!r}, "
                f"not mapping key {key!r}."
            )
        if genome.overlay:
            resolved[key] = expected_profile(key, genome)
        opening_holds.add(int(genome.opening_hold_steps))
    if opening_holds - {0}:
        raise OpponentSpecError(
            "Keyed R1 opponent overlays do not support per-opponent opening holds; "
            f"received {sorted(opening_holds)!r}."
        )
    core._bt_profile_override = resolved or None
    core._sds_opening_hold_steps = 0


def resolved_profile_rows(core) -> list[dict[str, Any]]:
    """Return the resolved BT profile for every live sub-environment."""
    tensors = core._bt_resolved_profile_tensors()
    rows: list[dict[str, Any]] = []
    for env_i in range(int(core.B)):
        row: dict[str, Any] = {}
        for key, value in tensors.items():
            try:
                flat = value.reshape(int(core.B), -1)
                row[key] = flat[env_i, 0].item()
            except Exception:
                continue
        rows.append(row)
    return rows


def assert_live_opponent_batch(
    core,
    genomes_by_key: dict[str, SDSGenome],
    *,
    allowed_keys: tuple[str, ...],
    context: str = "",
) -> list[dict[str, Any]]:
    """Assert every live row against its requested canonical/overlaid profile."""
    keys = [str(k).upper() for k in getattr(core, "_opponent_key", [])]
    rows = resolved_profile_rows(core)
    allowed = {str(k).upper() for k in allowed_keys}
    if len(keys) != int(core.B) or len(rows) != int(core.B):
        raise OpponentSpecError(
            f"OPPONENT BATCH SHAPE MISMATCH ({context}): keys={len(keys)}, "
            f"profiles={len(rows)}, B={int(core.B)}"
        )
    mismatches = []
    manifest_rows = []
    for env_i, (key, got) in enumerate(zip(keys, rows)):
        if key not in allowed:
            mismatches.append(f"env {env_i}: live key {key!r} not in {sorted(allowed)!r}")
            continue
        genome = genomes_by_key.get(key)
        exp = expected_profile(key, genome)
        for f in fields(BTProfile):
            if f.name not in got:
                continue
            want = getattr(exp, f.name)
            have = got[f.name]
            if isinstance(want, bool):
                ok = bool(have) == bool(want)
            elif isinstance(want, int):
                ok = int(round(float(have))) == int(want)
            elif isinstance(want, float):
                ok = abs(float(have) - float(want)) <= 1e-6
            else:
                continue
            if not ok:
                mismatches.append(
                    f"env {env_i} {key} {f.name}: expected {want!r}, live {have!r}"
                )
        manifest_rows.append({
            "env_index": env_i,
            "live_opponent_key": key,
            "genome_id": genome.genome_id if genome else None,
            "requested_overlay": dict(genome.overlay) if genome else {},
            "resolved_profile": got,
        })
    if mismatches:
        raise OpponentSpecError(
            f"OPPONENT BATCH MISMATCH{' (' + context + ')' if context else ''}:\n  "
            + "\n  ".join(mismatches)
            + "\nRefusing to consume a training step."
        )
    return manifest_rows
