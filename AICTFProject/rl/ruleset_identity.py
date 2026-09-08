"""Ruleset identity: stamping and mismatch rejection.

A RULESET_V1 policy entering a RULESET_V2 result is a silent, unrecoverable
contamination -- V1 made a lone defender unable to tag at all, so a V1 policy
learned a different game. The old G0 family
(``checkpoints/k2v2_piR/*_1000000.zip``) is exactly such a policy and is sitting
in the tree, so this check is not hypothetical.

Comparison is on the FULL field dictionary, never the friendly ``ruleset_id``
alone: two configs could share a label while differing in a field that changes
play.

Policy:

    checkpoint ruleset == environment ruleset   -> load
    mismatch                                    -> hard error
    missing metadata                            -> LEGACY_UNKNOWN, rejected
                                                   for formal runs

An explicit diagnostic override exists, but it warns loudly and stamps the
result ``formal_result_eligible = false``.
"""
from __future__ import annotations

import warnings
from typing import Any, Mapping

RULESET_FIELDS = (
    "ruleset_id",
    "taggers_required",
    "tag_min_interval_seconds",
    "tag_nearest_only",
    "tag_channel_seconds",
    "suppression_attackers_required",
)

LEGACY_UNKNOWN = "LEGACY_UNKNOWN"


class RulesetFingerprintError(RuntimeError):
    """Raised when a checkpoint would be written without a ruleset fingerprint."""


class RulesetMismatchError(RuntimeError):
    """Raised when a checkpoint's ruleset does not match the environment's."""


def fingerprint(cfg: Any) -> dict:
    """Extract the ruleset fingerprint from a config-like object or mapping."""
    if isinstance(cfg, Mapping):
        get = cfg.get
    else:
        def get(k, default=None):
            return getattr(cfg, k, default)

    if hasattr(cfg, "ruleset_fields") and callable(cfg.ruleset_fields):
        return dict(cfg.ruleset_fields())

    out = {}
    for k in RULESET_FIELDS:
        v = get(k, None)
        if v is None:
            continue
        if k in ("taggers_required", "suppression_attackers_required"):
            out[k] = int(v)
        elif k in ("tag_min_interval_seconds", "tag_channel_seconds"):
            out[k] = float(v)
        elif k == "tag_nearest_only":
            out[k] = bool(v)
        else:
            out[k] = str(v)
    return out


def is_complete(fp: Mapping) -> bool:
    return all(k in fp for k in RULESET_FIELDS)


def classify(fp: Mapping | None) -> str:
    if not fp or not is_complete(fp):
        return LEGACY_UNKNOWN
    return str(fp.get("ruleset_id", LEGACY_UNKNOWN))


def compare(checkpoint_fp: Mapping | None, env_fp: Mapping) -> dict:
    """-> {match, checkpoint_ruleset, env_ruleset, differing_fields, reason}."""
    ck_id = classify(checkpoint_fp)
    env_id = classify(env_fp)
    if ck_id == LEGACY_UNKNOWN:
        return {"match": False, "checkpoint_ruleset": LEGACY_UNKNOWN,
                "env_ruleset": env_id, "differing_fields": sorted(RULESET_FIELDS),
                "reason": "checkpoint has no (or incomplete) ruleset metadata"}
    diff = [k for k in RULESET_FIELDS if checkpoint_fp.get(k) != env_fp.get(k)]
    return {"match": not diff, "checkpoint_ruleset": ck_id, "env_ruleset": env_id,
            "differing_fields": diff,
            "reason": "" if not diff else f"fields differ: {diff}"}


def enforce(checkpoint_fp: Mapping | None, env_fp: Mapping, *,
            allow_mismatch: bool = False, context: str = "") -> dict:
    """Raise on mismatch unless explicitly overridden.

    Returns a result dict carrying ``formal_result_eligible`` for stamping into
    the run's provenance.
    """
    res = compare(checkpoint_fp, env_fp)
    where = f" ({context})" if context else ""
    if res["match"]:
        res["formal_result_eligible"] = True
        res["ruleset_mismatch_override"] = False
        return res

    msg = (f"Ruleset mismatch{where}: checkpoint={res['checkpoint_ruleset']} "
           f"env={res['env_ruleset']}; {res['reason']}")
    if not allow_mismatch:
        raise RulesetMismatchError(
            msg + ". Refusing to load. A V1 policy learned a different game "
                  "(a lone defender could not tag), so mixing rulesets silently "
                  "corrupts the result. Pass allow_mismatch=True only for "
                  "diagnostics; the run will be marked formal_result_eligible=False."
        )
    warnings.warn("RULESET MISMATCH OVERRIDE IN EFFECT -- " + msg +
                  ". This result is NOT eligible as a formal result.",
                  RuntimeWarning, stacklevel=2)
    res["formal_result_eligible"] = False
    res["ruleset_mismatch_override"] = True
    return res


def stamp(target: dict, cfg: Any) -> dict:
    """Write the ruleset fingerprint into a run config / manifest dict."""
    target.update(fingerprint(cfg))
    return target


def stamp_artifact(target: dict, cfg: Any, *, formal_result_eligible: bool = True) -> dict:
    """Stamp the full V2 fingerprint into any run artifact.

    Use for run_config.json, the training manifest, evaluation manifests, and
    episode CSV rows. Every artifact must be able to answer "which game produced
    this number?" -- the friendly ruleset_id alone is not enough, because two
    configs can share a label while differing in a field that changes play.
    """
    target.update(fingerprint(cfg))
    target["formal_result_eligible"] = bool(formal_result_eligible)
    return target


def artifact_row_fields(cfg: Any, *, formal_result_eligible: bool = True) -> dict:
    """Flat dict suitable for CSV columns."""
    return stamp_artifact({}, cfg, formal_result_eligible=formal_result_eligible)


# ======================================================================
# Formal run identity -- resolved ONCE from the live environment
# ======================================================================
#
# Every writer receives the SAME resolved object. Writers must never
# reconstruct identity from configuration defaults: that is how five artifacts
# end up carrying five subtly different versions of "V2".

import hashlib as _hashlib
import json as _json
import uuid as _uuid
from pathlib import Path as _Path
from dataclasses import asdict as _asdict, dataclass as _dataclass, field as _field

ARTIFACT_IDENTITY_KEY = "artifact_identity"

# Scalar identity repeated on every CSV row, so a single mismatched or
# accidentally concatenated row is detectable without the manifest.
CSV_IDENTITY_FIELDS = (
    "run_id", "canonical_map", "resolved_map", "ruleset_id",
    "ruleset_fingerprint", "formal_result_eligible",
)


class RunIdentityError(RuntimeError):
    """Raised when a formal run identity cannot be resolved or is violated."""


def ruleset_fingerprint_hash(fields: Mapping) -> str:
    """Stable digest of the ruleset fields, comparable byte-for-byte."""
    canon = {k: fields[k] for k in RULESET_FIELDS if k in fields}
    return _hashlib.sha256(
        _json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@_dataclass(frozen=True)
class RunIdentity:
    run_id: str
    canonical_map: str
    resolved_map: str
    ruleset_id: str
    ruleset_fingerprint: str
    ruleset: dict = _field(default_factory=dict)
    formal_result_eligible: bool = True
    identity_override_used: bool = False

    def artifact_identity(self) -> dict:
        """Top-level block embedded in every JSON artifact."""
        return {
            "run_id": self.run_id,
            "canonical_map": self.canonical_map,
            "resolved_map": self.resolved_map,
            "ruleset_id": self.ruleset_id,
            "ruleset_fingerprint": self.ruleset_fingerprint,
            "formal_result_eligible": self.formal_result_eligible,
            "identity_override_used": self.identity_override_used,
        }

    def csv_fields(self) -> dict:
        return {k: getattr(self, k) for k in CSV_IDENTITY_FIELDS}

    def as_dict(self) -> dict:
        return _asdict(self)


def _resolve_env_cfg(env):
    for attr in ("core", "vec"):
        obj = getattr(env, attr, None)
        if obj is None:
            continue
        core = obj if attr == "core" else getattr(obj, "core", None)
        cfg = getattr(core, "cfg", None) if core is not None else None
        if cfg is not None:
            return cfg
    return getattr(env, "cfg", None)


def build_formal_run_identity(env, *, run_id: str | None = None,
                              canonical_map: str | None = None,
                              allow_override: bool = False) -> RunIdentity:
    """Resolve the run's identity ONCE, from the live environment.

    Fails closed: a formal run refuses to proceed when the environment cannot
    supply a complete ruleset, rather than silently falling back to config
    defaults and producing artifacts that disagree.
    """
    cfg = _resolve_env_cfg(env)
    if cfg is None:
        raise RunIdentityError(
            "Cannot resolve the live environment config; refusing to build a "
            "formal run identity from defaults.")
    fp = fingerprint(cfg)
    if not is_complete(fp):
        if not allow_override:
            raise RunIdentityError(
                f"Environment ruleset is incomplete ({sorted(fp)}); refusing to "
                "stamp a formal run. Pass allow_override=True only for "
                "diagnostics -- the run will be marked ineligible.")
        warnings.warn("Formal run identity resolved under OVERRIDE; artifacts are "
                      "NOT eligible as formal results.", RuntimeWarning, stacklevel=2)

    resolved = str(getattr(cfg, "map_layout", "") or "")
    canon = canonical_map or ("map_a" if resolved == "map_a_open" else resolved)
    return RunIdentity(
        run_id=run_id or _uuid.uuid4().hex,
        canonical_map=canon,
        resolved_map=resolved,
        ruleset_id=str(fp.get("ruleset_id", LEGACY_UNKNOWN)),
        ruleset_fingerprint=ruleset_fingerprint_hash(fp),
        ruleset=dict(fp),
        formal_result_eligible=bool(is_complete(fp)) and not allow_override,
        identity_override_used=bool(allow_override),
    )


def stamp_json_artifact(target: dict, identity: RunIdentity) -> dict:
    """Embed the shared identity block in a JSON artifact."""
    if not isinstance(identity, RunIdentity):
        raise RunIdentityError(
            "stamp_json_artifact requires the run's resolved RunIdentity, not a "
            "config -- writers must not rebuild identity independently.")
    target[ARTIFACT_IDENTITY_KEY] = identity.artifact_identity()
    return target


def stamp_csv_row(row: dict, identity: RunIdentity) -> dict:
    """Repeat scalar identity on a CSV row so contamination is detectable."""
    if not isinstance(identity, RunIdentity):
        raise RunIdentityError("stamp_csv_row requires the resolved RunIdentity.")
    row.update(identity.csv_fields())
    return row


def assert_ruleset_matches_identity(ruleset: Mapping | None, identity: RunIdentity,
                                    *, context: str = "") -> None:
    """Checkpoint ruleset payload and RunIdentity must describe the same game."""
    if not isinstance(identity, RunIdentity):
        raise RunIdentityError("assert_ruleset_matches_identity requires RunIdentity.")
    fp = dict(ruleset or {})
    if not is_complete(fp):
        raise RunIdentityError(
            f"{context or 'checkpoint'}: ruleset payload is incomplete or missing "
            f"(legacy/V1-unstamped); refusing to treat it as the same identity."
        )
    if str(fp.get("ruleset_id")) != identity.ruleset_id:
        raise RunIdentityError(
            f"{context or 'checkpoint'}: ruleset_id mismatch vs run identity: "
            f"{fp.get('ruleset_id')!r} != {identity.ruleset_id!r}"
        )
    # Full field dictionary — never the friendly label alone.
    for key in RULESET_FIELDS:
        if key == "ruleset_id":
            continue
        if key not in fp:
            raise RunIdentityError(
                f"{context or 'checkpoint'}: ruleset missing field {key!r}"
            )
        live = identity.ruleset.get(key)
        if live is not None and fp[key] != live:
            raise RunIdentityError(
                f"{context or 'checkpoint'}: ruleset field {key!r} differs from "
                f"live evaluation identity: {fp[key]!r} != {live!r}"
            )
    digest = ruleset_fingerprint_hash(fp)
    if digest != identity.ruleset_fingerprint:
        raise RunIdentityError(
            f"{context or 'checkpoint'}: ruleset fingerprint mismatch vs run "
            f"identity: {digest[:12]}… != {identity.ruleset_fingerprint[:12]}…"
        )


def map_identity_aliases(canonical_map: str, resolved_map: str) -> set[str]:
    """Equivalent map spellings (e.g. map_a ↔ map_a_open)."""
    forms = {str(canonical_map or ""), str(resolved_map or "")}
    forms.discard("")
    if forms & {"map_a", "map_a_open"}:
        forms.update({"map_a", "map_a_open"})
    return forms


def maps_compatible(
    left_canonical: str,
    left_resolved: str,
    right_canonical: str,
    right_resolved: str,
) -> bool:
    return bool(
        map_identity_aliases(left_canonical, left_resolved)
        & map_identity_aliases(right_canonical, right_resolved)
    )


def read_checkpoint_identity_payload(source_checkpoint) -> dict:
    """Load ruleset + artifact_identity from a checkpoint before model execution.

    Missing / legacy / incomplete identity fails closed.
    """
    from pathlib import Path

    from rl.custom_ppo.checkpoints.loader import read_checkpoint_payload

    path = str(Path(source_checkpoint))
    payload = read_checkpoint_payload(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RunIdentityError(
            f"Checkpoint {path!r} is not a mapping payload; refusing evaluation."
        )
    ruleset = payload.get("ruleset")
    if not isinstance(ruleset, dict) or not is_complete(ruleset):
        raise RunIdentityError(
            f"Checkpoint {path!r} has no complete ruleset identity "
            "(legacy/missing); refusing evaluation before model execution."
        )
    ai = payload.get(ARTIFACT_IDENTITY_KEY)
    if not isinstance(ai, Mapping):
        raise RunIdentityError(
            f"Checkpoint {path!r} is missing artifact_identity; refusing "
            "evaluation before model execution."
        )
    for key in ("run_id", "canonical_map", "resolved_map", "ruleset_id",
                "ruleset_fingerprint"):
        if not ai.get(key):
            raise RunIdentityError(
                f"Checkpoint {path!r} artifact_identity is missing {key!r}."
            )
    return {
        "path": path,
        "ruleset": dict(ruleset),
        "artifact_identity": dict(ai),
        "file_fingerprint": _sha256_file(path),
    }


def _sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def assert_checkpoint_compatible_with_evaluation_identity(
    evaluation_identity: RunIdentity,
    source_checkpoint,
    *,
    allow_override: bool = False,
    context: str = "",
) -> dict:
    """Require exact map + ruleset agreement before any model forward pass."""
    ckpt = read_checkpoint_identity_payload(source_checkpoint)
    ai = ckpt["artifact_identity"]
    label = context or ckpt["path"]

    map_ok = maps_compatible(
        evaluation_identity.canonical_map,
        evaluation_identity.resolved_map,
        str(ai["canonical_map"]),
        str(ai["resolved_map"]),
    )
    try:
        assert_ruleset_matches_identity(ckpt["ruleset"], evaluation_identity, context=label)
        if str(ai["ruleset_id"]) != evaluation_identity.ruleset_id:
            raise RunIdentityError(
                f"{label}: artifact_identity.ruleset_id mismatch: "
                f"{ai['ruleset_id']!r} != {evaluation_identity.ruleset_id!r}"
            )
        if str(ai["ruleset_fingerprint"]) != evaluation_identity.ruleset_fingerprint:
            raise RunIdentityError(
                f"{label}: artifact_identity.ruleset_fingerprint mismatch vs live eval."
            )
        if not map_ok:
            raise RunIdentityError(
                f"{label}: map mismatch vs live evaluation environment: "
                f"checkpoint=({ai['canonical_map']!r},{ai['resolved_map']!r}) "
                f"live=({evaluation_identity.canonical_map!r},"
                f"{evaluation_identity.resolved_map!r})"
            )
        if ai.get("formal_result_eligible") is False and not allow_override:
            raise RunIdentityError(
                f"{label}: source checkpoint is not formal_result_eligible."
            )
    except RunIdentityError:
        if not allow_override:
            raise
        warnings.warn(
            f"EVALUATION IDENTITY OVERRIDE for {label}; artifacts are NOT "
            "eligible as formal results.",
            RuntimeWarning,
            stacklevel=2,
        )
        return {**ckpt, "override_used": True}
    return {**ckpt, "override_used": False}


def build_evaluation_run_identity(
    env,
    *,
    evaluation_run_id: str,
    source_checkpoint=None,
    source_training_run_id: str | None = None,
    source_checkpoint_fingerprint: str | None = None,
    source_checkpoint_ruleset_fingerprint: str | None = None,
    allow_override: bool = False,
) -> tuple[RunIdentity, dict]:
    """Identity for a standalone evaluation run against a live eval environment.

    Preferred call shape::

        build_evaluation_run_identity(
            eval_env,
            evaluation_run_id=...,
            source_checkpoint=checkpoint_path,
        )

    Map and ruleset identity come from the LIVE evaluation environment.
    The checkpoint is checked for compatibility; it is not the identity source.
    """
    identity = build_formal_run_identity(
        env, run_id=evaluation_run_id, allow_override=allow_override
    )

    if source_checkpoint is not None:
        ckpt = assert_checkpoint_compatible_with_evaluation_identity(
            identity,
            source_checkpoint,
            allow_override=allow_override,
            context=str(source_checkpoint),
        )
        ai = ckpt["artifact_identity"]
        if ckpt.get("override_used"):
            identity = RunIdentity(
                run_id=identity.run_id,
                canonical_map=identity.canonical_map,
                resolved_map=identity.resolved_map,
                ruleset_id=identity.ruleset_id,
                ruleset_fingerprint=identity.ruleset_fingerprint,
                ruleset=dict(identity.ruleset),
                formal_result_eligible=False,
                identity_override_used=True,
            )
        lineage = {
            "evaluation_run_id": evaluation_run_id,
            "source_training_run_id": str(
                source_training_run_id or ai.get("run_id") or ""
            ),
            "source_checkpoint_id": ckpt["file_fingerprint"],
            "source_checkpoint_fingerprint": ckpt["file_fingerprint"],
            "source_checkpoint_ruleset_fingerprint": str(
                ai.get("ruleset_fingerprint") or ruleset_fingerprint_hash(ckpt["ruleset"])
            ),
        }
        return identity, lineage

    # Legacy fingerprint-only path (tests / transitional callers).
    if (
        source_checkpoint_ruleset_fingerprint
        and source_checkpoint_ruleset_fingerprint != identity.ruleset_fingerprint
        and not allow_override
    ):
        raise RunIdentityError(
            "Evaluation environment ruleset fingerprint does not match the "
            "source checkpoint; refusing formal evaluation. Pass "
            "allow_override=True only for ineligible diagnostics."
        )
    if allow_override and (
        source_checkpoint_ruleset_fingerprint
        and source_checkpoint_ruleset_fingerprint != identity.ruleset_fingerprint
    ):
        identity = RunIdentity(
            run_id=identity.run_id,
            canonical_map=identity.canonical_map,
            resolved_map=identity.resolved_map,
            ruleset_id=identity.ruleset_id,
            ruleset_fingerprint=identity.ruleset_fingerprint,
            ruleset=dict(identity.ruleset),
            formal_result_eligible=False,
            identity_override_used=True,
        )
    lineage = {
        "evaluation_run_id": evaluation_run_id,
        "source_training_run_id": source_training_run_id or "",
        "source_checkpoint_id": source_checkpoint_fingerprint or "",
        "source_checkpoint_fingerprint": source_checkpoint_fingerprint or "",
        "source_checkpoint_ruleset_fingerprint": (
            source_checkpoint_ruleset_fingerprint or ""
        ),
    }
    return identity, lineage


def read_artifact_identity(obj: Mapping) -> dict:
    ai = obj.get(ARTIFACT_IDENTITY_KEY)
    if not isinstance(ai, Mapping):
        raise RunIdentityError(f"artifact is missing '{ARTIFACT_IDENTITY_KEY}'")
    for k in ("run_id", "canonical_map", "resolved_map", "ruleset_id",
              "ruleset_fingerprint"):
        if not ai.get(k):
            raise RunIdentityError(f"artifact identity is missing {k!r}")
    return dict(ai)


def validate_bundle(json_artifacts: Mapping[str, Mapping],
                    csv_rows: Mapping[str, list] | None = None,
                    *, require_formal: bool = True) -> dict:
    """Verify every artifact in a run carries the SAME passport.

    ``json_artifacts`` maps a label to a parsed JSON artifact.
    ``csv_rows`` maps a label to a list of row dicts.
    Raises RunIdentityError on the first violation.
    """
    if not json_artifacts:
        raise RunIdentityError("no artifacts supplied to validate")

    seen: dict[str, dict] = {}
    for label, obj in json_artifacts.items():
        seen[label] = read_artifact_identity(obj)

    ref_label, ref = next(iter(seen.items()))
    for label, ai in seen.items():
        for k in ("run_id", "canonical_map", "resolved_map", "ruleset_id",
                  "ruleset_fingerprint"):
            if ai[k] != ref[k]:
                raise RunIdentityError(
                    f"{k} differs between {ref_label!r} and {label!r}: "
                    f"{ref[k]!r} != {ai[k]!r}")
        if require_formal and not ai.get("formal_result_eligible", False):
            raise RunIdentityError(f"{label!r} is not formal_result_eligible")
        if require_formal and ai.get("identity_override_used", False):
            raise RunIdentityError(f"{label!r} used the diagnostic identity override")

    for label, rows in (csv_rows or {}).items():
        if not rows:
            raise RunIdentityError(f"{label!r} has no rows")
        for i, r in enumerate(rows):
            for k in ("run_id", "canonical_map", "resolved_map", "ruleset_id",
                      "ruleset_fingerprint"):
                if k not in r:
                    raise RunIdentityError(f"{label!r} row {i} missing {k}")
                if str(r[k]) != str(ref[k]):
                    raise RunIdentityError(
                        f"{label!r} row {i} {k} mismatch: {r[k]!r} != {ref[k]!r}")
    return dict(ref)


# ======================================================================
# Verified checkpoint lineage
# ======================================================================
#
# A formal evaluation manifest must prove the CHECKPOINT and the live
# evaluation environment agree. The earlier writer defaulted the lineage
# fields to the evaluation run's own identity, so an omitted value produced a
# manifest asserting the checkpoint matched itself -- a border check comparing
# a passport to its own reflection.
#
# Lineage is therefore a validated object, constructible only by the
# compatibility verifier, rather than three loose strings a caller can forget.


@_dataclass(frozen=True)
class VerifiedCheckpointLineage:
    source_training_run_id: str
    source_checkpoint_fingerprint: str
    source_checkpoint_ruleset_fingerprint: str

    def as_dict(self) -> dict:
        return {
            "source_training_run_id": self.source_training_run_id,
            "source_checkpoint_fingerprint": self.source_checkpoint_fingerprint,
            "source_checkpoint_ruleset_fingerprint":
                self.source_checkpoint_ruleset_fingerprint,
            "checkpoint_lineage_complete": True,
        }

    @classmethod
    def for_in_training_evaluation(cls, identity: "RunIdentity",
                                   checkpoint_fingerprint: str
                                   ) -> "VerifiedCheckpointLineage":
        """Lineage for evaluation performed INSIDE the training run.

        Here the evaluation and the checkpoint genuinely share a run, so
        ``source_training_run_id == evaluation_run_id`` is a fact rather than a
        self-fallback. It is an explicit, named constructor precisely so that
        equality is a declared claim at the call site, not something a writer
        quietly filled in for an absent argument.
        """
        if not identity.formal_result_eligible:
            raise RunIdentityError(
                "for_in_training_evaluation is for formal runs; a diagnostic run "
                "must record VerifiedCheckpointLineage.missing().")
        if not (checkpoint_fingerprint or "").strip():
            raise RunIdentityError(
                "in-training lineage still requires a real checkpoint fingerprint.")
        return cls(
            source_training_run_id=identity.run_id,
            source_checkpoint_fingerprint=str(checkpoint_fingerprint),
            source_checkpoint_ruleset_fingerprint=identity.ruleset_fingerprint,
        )

    @staticmethod
    def missing() -> dict:
        """Honest record of absent lineage for a diagnostic evaluation.

        Null is honest; self-fallback is camouflage.
        """
        return {
            "source_training_run_id": None,
            "source_checkpoint_fingerprint": None,
            "source_checkpoint_ruleset_fingerprint": None,
            "checkpoint_lineage_complete": False,
        }


def verify_checkpoint_lineage(*, checkpoint_path: str, identity: RunIdentity,
                              checkpoint_ruleset: Mapping | None,
                              source_training_run_id: str | None,
                              checkpoint_file_fingerprint: str | None = None,
                              ) -> VerifiedCheckpointLineage:
    """Construct lineage ONLY after checking the checkpoint against the env.

    Called before model weights are executed. Raises rather than returning a
    partially-populated object, so a caller cannot proceed on a half-check.
    """
    if not identity.formal_result_eligible:
        raise RunIdentityError(
            "verify_checkpoint_lineage is for formal evaluations; a diagnostic "
            "run must record VerifiedCheckpointLineage.missing() instead.")

    ck_fp = classify(checkpoint_ruleset)
    if ck_fp == LEGACY_UNKNOWN:
        raise RunIdentityError(
            f"Checkpoint {checkpoint_path!r} has no complete ruleset metadata "
            "(LEGACY_UNKNOWN); refusing to evaluate it formally.")

    ck_hash = ruleset_fingerprint_hash(checkpoint_ruleset)
    if ck_hash != identity.ruleset_fingerprint:
        raise RunIdentityError(
            f"Checkpoint ruleset fingerprint does not match the live evaluation "
            f"environment: checkpoint={ck_hash[:12]} env="
            f"{identity.ruleset_fingerprint[:12]} "
            f"(checkpoint ruleset_id={ck_fp}, env={identity.ruleset_id}).")

    if not (source_training_run_id or "").strip():
        raise RunIdentityError(
            "source_training_run_id is required for a formal evaluation and "
            "must come from the checkpoint, not the evaluation run.")

    file_fp = checkpoint_file_fingerprint
    if not (file_fp or "").strip():
        p = _Path(checkpoint_path)
        if not p.exists():
            raise RunIdentityError(
                f"Cannot fingerprint checkpoint {checkpoint_path!r}: file not found.")
        h = _hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        file_fp = h.hexdigest()

    return VerifiedCheckpointLineage(
        source_training_run_id=str(source_training_run_id),
        source_checkpoint_fingerprint=str(file_fp),
        source_checkpoint_ruleset_fingerprint=ck_hash,
    )


# ======================================================================
# Checkpoint <-> RunIdentity join
# ======================================================================
#
# A checkpoint historically carried its own homemade passport: a "ruleset"
# payload unrelated to the run-level identity block. Two parallel systems can
# drift. This is the single boundary both save and load pass through, so the
# checkpoint's ruleset, its artifact_identity, and the live RunIdentity are
# three representations of ONE fact rather than three opinions.

CHECKPOINT_RULESET_KEY = "ruleset"

_IDENTITY_COMPARE_FIELDS = (
    "canonical_map", "resolved_map", "ruleset_id", "ruleset_fingerprint",
    "formal_result_eligible", "identity_override_used",
)


def verify_checkpoint_run_identity(checkpoint_payload: Mapping,
                                   live_identity: RunIdentity,
                                   *,
                                   operation: str,
                                   allow_different_run_id: bool = False,
                                   context: str = "") -> dict:
    """Compare checkpoint ruleset, artifact_identity, and live RunIdentity.

    ``operation`` is ``"save"`` or ``"load"``. On load this MUST be called
    before ``load_state_dict`` and before any forward pass.

    ``allow_different_run_id`` is for standalone evaluation, where a later run
    legitimately evaluates another run's checkpoint. Map and ruleset must still
    match exactly; only the run id may differ. It is an explicit argument rather
    than a permissive comparison so the intent is visible at the call site.
    """
    if operation not in ("save", "load"):
        raise RunIdentityError(f"operation must be 'save' or 'load', got {operation!r}")
    where = f" ({context})" if context else ""

    ai = checkpoint_payload.get(ARTIFACT_IDENTITY_KEY)
    if not isinstance(ai, Mapping):
        raise RunIdentityError(
            f"Checkpoint{where} has no {ARTIFACT_IDENTITY_KEY!r} block; refusing to "
            f"{operation}. Legacy/unstamped checkpoints are not eligible.")

    ck_rules = checkpoint_payload.get(CHECKPOINT_RULESET_KEY)
    if not isinstance(ck_rules, Mapping) or not is_complete(ck_rules):
        raise RunIdentityError(
            f"Checkpoint{where} has no complete {CHECKPOINT_RULESET_KEY!r} payload "
            f"(classified {classify(ck_rules)}); refusing to {operation}.")

    # (1) the checkpoint's two internal representations must agree
    ck_hash = ruleset_fingerprint_hash(ck_rules)
    if ai.get("ruleset_fingerprint") != ck_hash:
        raise RunIdentityError(
            f"Checkpoint{where} is internally inconsistent: artifact_identity "
            f"fingerprint {str(ai.get('ruleset_fingerprint'))[:12]} != ruleset "
            f"payload fingerprint {ck_hash[:12]}.")
    if str(ai.get("ruleset_id")) != str(ck_rules.get("ruleset_id")):
        raise RunIdentityError(
            f"Checkpoint{where} ruleset_id disagrees between artifact_identity "
            f"({ai.get('ruleset_id')!r}) and ruleset payload "
            f"({ck_rules.get('ruleset_id')!r}).")

    # (2) the checkpoint must agree with the live environment
    live = live_identity.artifact_identity()
    diffs = [k for k in _IDENTITY_COMPARE_FIELDS if ai.get(k) != live.get(k)]
    if diffs:
        raise RunIdentityError(
            f"Checkpoint{where} identity does not match the live environment on "
            f"{diffs}: checkpoint={{{', '.join(f'{k}={ai.get(k)!r}' for k in diffs)}}} "
            f"live={{{', '.join(f'{k}={live.get(k)!r}' for k in diffs)}}}. "
            "A matching ruleset_id alone is never sufficient.")

    # (3) field-level equality, so a doctored fingerprint cannot slip through
    field_diffs = [k for k in RULESET_FIELDS
                   if ck_rules.get(k) != live_identity.ruleset.get(k)]
    if field_diffs:
        raise RunIdentityError(
            f"Checkpoint{where} ruleset fields differ from the live environment: "
            f"{field_diffs}.")

    # (4) run id
    if not allow_different_run_id and ai.get("run_id") != live.get("run_id"):
        raise RunIdentityError(
            f"Checkpoint{where} run_id {ai.get('run_id')!r} != live run_id "
            f"{live.get('run_id')!r}. Pass allow_different_run_id=True only for "
            "standalone evaluation, and record the relationship in a "
            "VerifiedCheckpointLineage.")

    return {"operation": operation, "match": True,
            "checkpoint_run_id": ai.get("run_id"), "live_run_id": live.get("run_id"),
            "ruleset_id": live_identity.ruleset_id,
            "ruleset_fingerprint": live_identity.ruleset_fingerprint,
            "different_run_id_allowed": bool(allow_different_run_id)}


def build_checkpoint_identity_payload(live_identity: RunIdentity) -> dict:
    """The two identity representations a formal checkpoint must carry."""
    if not isinstance(live_identity, RunIdentity):
        raise RunIdentityError(
            "build_checkpoint_identity_payload requires the run's resolved "
            "RunIdentity; a checkpoint must not infer identity from trainer "
            "config or environment defaults.")
    return {
        CHECKPOINT_RULESET_KEY: dict(live_identity.ruleset),
        ARTIFACT_IDENTITY_KEY: live_identity.artifact_identity(),
    }
