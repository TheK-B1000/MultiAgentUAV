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
