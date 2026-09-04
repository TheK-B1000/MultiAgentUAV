"""Rung 0 of the sharing ladder: two complete specialist actors dispatched by z.

Implements SHARING_LADDER_SPEC.json#PROGRESSION_OF_RUNGS.rung_0. Zero shared parameters --
z0 routes the ENTIRE forward pass to pi_A, z1 to pi_B. This is deliberately not compression;
it is the structural positive control that asks whether the dispatch-and-evaluate path
preserves crossover the specialists already demonstrated.

It cannot be expressed as a single latent checkpoint (the current architecture shares a CNN and
body between latents by construction), so Rung 0 is a POLICY WRAPPER presenting the same
interface the sealed evaluators use: ``fixed_latent_strategy`` / ``fixed_latent_strategy_id`` /
``reset_strategy()`` / ``predict(obs, deterministic=...)``. The evaluator therefore exercises
the identical code path it uses for every other arm; only what sits behind ``predict`` differs.

Exactness is verified, never assumed -- see verify_equivalence(), which the preflight requires
to be bit-exact on masked logits, argmax actions, and both the macro and target heads.
"""
from __future__ import annotations

from typing import Any

import torch


class Rung0DispatchPolicy:
    """Dispatches the whole forward pass to a frozen specialist according to the forced z."""

    def __init__(self, pi_A: Any, pi_B: Any):
        self._by_z = {0: pi_A, 1: pi_B}
        self.pi_A = pi_A
        self.pi_B = pi_B
        # Interface parity with CustomPPOInferencePolicy as the evaluators use it.
        self.fixed_latent_strategy = True
        self.fixed_latent_strategy_id = 0
        for p in (pi_A, pi_B):
            if getattr(p, "fixed_latent_strategy", None) is not None:
                p.fixed_latent_strategy = False   # specialists are single-strategy; no z path
        self.latent_k = 2
        self.uses_latent_strategy = True

    # -- the active specialist ------------------------------------------------
    @property
    def active(self) -> Any:
        z = int(self.fixed_latent_strategy_id)
        if z not in self._by_z:
            raise ValueError(f"Rung 0 dispatch received z={z}; only 0 and 1 exist")
        return self._by_z[z]

    @property
    def model(self) -> Any:
        """The evaluators inspect .model.latent_k / .uses_latent_strategy. Report the wrapper's
        own K=2 identity while delegating everything else to the active specialist."""
        return _WrapperModelView(self)

    # -- evaluator-facing API -------------------------------------------------
    def reset_strategy(self) -> None:
        for p in self._by_z.values():
            p.reset_strategy()

    def predict(self, obs, deterministic: bool = True):
        return self.active.predict(obs, deterministic=deterministic)

    def __getattr__(self, name):
        # Anything not defined here falls through to the active specialist, so the wrapper
        # cannot silently diverge from it on an unmodelled attribute.
        return getattr(self.__dict__["_by_z"][int(self.__dict__["fixed_latent_strategy_id"])], name)


class _WrapperModelView:
    """Presents latent_k=2 / uses_latent_strategy=True to the evaluator's architecture check
    while forwarding real model attributes to the active specialist's model."""

    def __init__(self, wrapper: Rung0DispatchPolicy):
        self._w = wrapper
        self.latent_k = 2
        self.uses_latent_strategy = True

    def __getattr__(self, name):
        return getattr(self._w.active.model, name)


@torch.no_grad()
def verify_equivalence(wrapper: Rung0DispatchPolicy, obs: dict, *, device: str) -> dict:
    """Bit-exactness of wrapper(z) against the corresponding specialist.

    Checks masked logits, argmax actions, and the macro / target heads separately, because the
    sub-macro finding (SUBMACRO_LEVERAGE_DIAGNOSTIC_SPEC.json) showed the target head carries
    4.7-7.0x more disagreement than the macro head -- a wrapper bug there would be invisible if
    only macros were compared.
    """
    from rl.teacher_distillation import head_logits, masked_heads

    report = {}
    for z, spec_name, spec in ((0, "pi_A", wrapper.pi_A), (1, "pi_B", wrapper.pi_B)):
        wrapper.fixed_latent_strategy_id = z
        wrapper.reset_strategy()
        zt = torch.full((int(next(iter(obs.values())).shape[0]),), z, dtype=torch.long, device=device)

        lw = head_logits(wrapper.model, obs, z_idx=zt)
        ls = head_logits(spec.model, obs, z_idx=None)
        max_logit_delta = max(float((a - b).abs().max()) for a, b in zip(lw, ls))

        hw = masked_heads(wrapper.model, obs, z_idx=zt)
        hs = masked_heads(spec.model, obs, z_idx=None)
        per_head_argmax_equal = [bool((a.logits.argmax(-1) == b.logits.argmax(-1)).all())
                                 for a, b in zip(hw, hs)]
        # MultiDiscrete([5, 50, 5, 50]): heads 0,2 are MACRO; heads 1,3 are TARGET.
        macro_ok = all(per_head_argmax_equal[i] for i in (0, 2) if i < len(per_head_argmax_equal))
        target_ok = all(per_head_argmax_equal[i] for i in (1, 3) if i < len(per_head_argmax_equal))

        report[f"z{z}_vs_{spec_name}"] = {
            "max_abs_logit_delta": max_logit_delta,
            "logits_bit_exact": max_logit_delta == 0.0,
            "argmax_equal_per_head": per_head_argmax_equal,
            "macro_heads_equal": macro_ok,
            "target_heads_equal": target_ok,
            "n_heads": len(per_head_argmax_equal),
        }
    report["ALL_EXACT"] = all(
        v["logits_bit_exact"] and v["macro_heads_equal"] and v["target_heads_equal"]
        for k, v in report.items() if k.startswith("z")
    )
    return report
