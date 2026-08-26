"""Shared loading / teacher-query code for the Phase 0 scorer.

The fitter and Gate 0B both import from here so they cannot drift apart in how
they read shards, join targets, or query teacher distributions. A divergence
between "how the scorer was trained" and "how the gate evaluates it" would be
invisible in the verdict and wrong in the science.

Two things in here are load-bearing:

1. SPLIT ENFORCEMENT. ``load_split`` refuses to open a shard outside the seed
   set it was asked for, and reports how many held-out shards it touched. The
   96 held-out seeds are the scientific gate, not a development set, so the
   guard is an assertion rather than a convention.

2. MASKED teacher distributions. ``teacher_action_dists`` routes through
   ``strategy_anchor._masked_heads``, which reproduces the masking
   ``evaluate_actions()`` applies. ``get_distribution()`` does NOT mask. This
   exact discrepancy already produced one silent defect in this project (the
   SAPPO anchor scored unmasked logits; predict() and get_distribution() agreed
   on only 43%/66% of argmax actions), so the scorer reuses the fixed path
   rather than re-deriving it.
"""
from __future__ import annotations

import glob
import hashlib
from dataclasses import dataclass
from pathlib import Path

import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SD = ROOT / "artifacts/strategic_demand"
COLL = SD / "phase0_scorer_data/full_collection_rebuild_per_branch"

SEED_BASE = 6_500_001
N_SEEDS, N_TRAIN = 256, 160

TRAIN_SEEDS = list(range(SEED_BASE, SEED_BASE + N_TRAIN))                 # 160
HELDOUT_SEEDS = list(range(SEED_BASE + N_TRAIN, SEED_BASE + N_SEEDS))     # 96

# Prospective train-side development split, frozen before fitting. Model
# selection (early stopping) uses INNER_VAL only. It is carved out of the 160
# training seeds so the held-out 96 never act as a development set.
INNER_FIT_SEEDS = list(range(SEED_BASE, SEED_BASE + 128))                 # 128
INNER_VAL_SEEDS = list(range(SEED_BASE + 128, SEED_BASE + N_TRAIN))       # 32

POLE_NAME = {0: "A", 1: "B"}
POLICY_NAME = {0: "pi_A", 1: "pi_B"}


@dataclass
class ScorerSplit:
    """Plain (episode-labelled) rows and branch (counterfactual) rows."""
    # plain
    p_grid: np.ndarray; p_vec: np.ndarray; p_amask: np.ndarray
    p_pole: np.ndarray; p_action: np.ndarray; p_margin: np.ndarray
    p_seed: np.ndarray; p_weight: np.ndarray
    # branch: one row per (state, teacher)
    b_grid: np.ndarray; b_vec: np.ndarray; b_amask: np.ndarray; b_mask: np.ndarray
    b_pole: np.ndarray; b_action: np.ndarray; b_margin: np.ndarray
    b_seed: np.ndarray; b_teacher: np.ndarray          # 0 = pi_A, 1 = pi_B
    b_state_id: np.ndarray                             # pairs the two teachers
    seeds_opened: list
    heldout_opened: int
    data_sha256: str


def _episode_margins(seed: int) -> dict:
    """(policy, pole) -> terminal blue-red margin, from the seed summary."""
    rows = json.loads((COLL / "seed_summaries" / f"seed_{seed}.json").read_text())
    return {(r["policy"], r["pole"]): r["blue"] - r["red"] for r in rows}


def load_split(seeds, *, want_plain: bool = True) -> ScorerSplit:
    """Load exactly the requested seeds. Refuses anything outside the set.

    Target is TERMINAL WIN MARGIN (blue - red), per
    PHASE0_SCORER_TARGET_AMENDMENT.json. Monte-Carlo return is deliberately not
    read: it is excluded from scorer fitting.
    """
    want = set(seeds)
    P: dict[str, list] = {k: [] for k in
                          ("grid", "vec", "amask", "pole", "action", "margin", "seed", "weight")}
    B: dict[str, list] = {k: [] for k in
                          ("grid", "vec", "amask", "mask", "pole", "action",
                           "margin", "seed", "teacher", "state")}
    opened, heldout_opened, h = [], 0, hashlib.sha256()
    heldout = set(HELDOUT_SEEDS)

    for path in sorted(glob.glob(str(COLL / "seed_shards" / "*.npz"))):
        seed = int(Path(path).stem.split("seed_")[-1])
        if seed not in want:
            continue
        if seed in heldout:
            heldout_opened += 1
        opened.append(seed)
        h.update(Path(path).read_bytes())
        z = np.load(path, allow_pickle=True)

        if want_plain:
            margins = _episode_margins(seed)
            pol, pole = z["plain_policy"], z["plain_pole"]
            # episode-level label: every decision point in an episode shares the
            # single terminal margin. Weight by 1/len(episode) so each EPISODE
            # contributes one unit, honouring the frozen effective-sample-size
            # caveat rather than treating states as independent observations.
            marg = np.empty(len(pol), dtype=np.float32)
            wt = np.empty(len(pol), dtype=np.float32)
            for pi in (0, 1):
                for pl in (0, 1):
                    m = (pol == pi) & (pole == pl)
                    if not m.any():
                        continue
                    marg[m] = margins[(POLICY_NAME[pi], POLE_NAME[pl])]
                    wt[m] = 1.0 / int(m.sum())
            P["grid"].append(z["plain_obs_grid"][:, 0]); P["vec"].append(z["plain_obs_vec"][:, 0])
            P["amask"].append(z["plain_obs_agent_mask"][:, 0])
            P["pole"].append(pole.astype(np.int64)); P["action"].append(z["plain_action"])
            P["margin"].append(marg); P["weight"].append(wt)
            P["seed"].append(np.full(len(pol), seed, dtype=np.int64))

        n_b = z["branch_step"].shape[0]
        state_ids = np.arange(n_b, dtype=np.int64) + seed * 100
        for ti, tag in enumerate(("pi_A", "pi_B")):
            B["grid"].append(z["branch_obs_grid"][:, 0]); B["vec"].append(z["branch_obs_vec"][:, 0])
            B["amask"].append(z["branch_obs_agent_mask"][:, 0])
            B["mask"].append(z["branch_obs_mask"][:, 0])
            B["pole"].append(z["branch_pole"].astype(np.int64))
            B["action"].append(z[f"branch_{tag}_action"])
            B["margin"].append((z[f"branch_{tag}_blue"].astype(np.int32)
                                - z[f"branch_{tag}_red"].astype(np.int32)).astype(np.float32))
            B["seed"].append(np.full(n_b, seed, dtype=np.int64))
            B["teacher"].append(np.full(n_b, ti, dtype=np.int64))
            B["state"].append(state_ids)

    if set(opened) != want:
        raise SystemExit(f"REFUSING: opened {len(opened)} shards, requested {len(want)}")

    cat = lambda d, k, dt=None: (np.concatenate(d[k]).astype(dt) if d[k] else np.zeros(0))
    return ScorerSplit(
        p_grid=cat(P, "grid", np.float32), p_vec=cat(P, "vec", np.float32),
        p_amask=cat(P, "amask", np.float32), p_pole=cat(P, "pole", np.int64),
        p_action=cat(P, "action", np.int64), p_margin=cat(P, "margin", np.float32),
        p_seed=cat(P, "seed", np.int64), p_weight=cat(P, "weight", np.float32),
        b_grid=cat(B, "grid", np.float32), b_vec=cat(B, "vec", np.float32),
        b_amask=cat(B, "amask", np.float32), b_mask=cat(B, "mask", np.float32),
        b_pole=cat(B, "pole", np.int64), b_action=cat(B, "action", np.int64),
        b_margin=cat(B, "margin", np.float32), b_seed=cat(B, "seed", np.int64),
        b_teacher=cat(B, "teacher", np.int64), b_state_id=cat(B, "state", np.int64),
        seeds_opened=opened, heldout_opened=heldout_opened, data_sha256=h.hexdigest(),
    )


def _inner_model(model):
    """Unwrap CustomPPOInferencePolicy -> SharedActorCentralizedCritic.

    ``load_custom_ppo_policy`` returns the INFERENCE WRAPPER, which exposes
    neither ``policy_logits`` nor ``_mask_logits`` -- those live on the
    training-side model it wraps. Handing the wrapper to ``_masked_heads``
    silently selects its unmasked fallback (the one intended for test stubs)
    and yields distributions that do not match the policy's own behaviour.
    Measured consequence when this happened here: argmax of the returned
    distribution matched the recorded deterministic action on only 10-17% of
    branch states. The wrapper's own ``entropy()`` shows the correct path,
    calling ``self.model._mask_logits(self.model.policy_logits(...))``.
    """
    inner = getattr(model, "model", model)
    if not (hasattr(inner, "_mask_logits") and hasattr(inner, "policy_logits")):
        raise RuntimeError(
            f"{type(inner).__name__} exposes no masking path; refusing to fall "
            "back to unmasked logits for a scorer query"
        )
    if getattr(inner, "uses_latent_strategy", False):
        raise RuntimeError(
            "teacher uses latent strategy selection; z_idx would have to be "
            "resolved through the wrapper's state rather than passed as None"
        )
    return inner


def teacher_action_dists(model, grid, vec, amask, mask):
    """Per-agent action distributions over the 250-way joint macro x waypoint.

    Returns (p1, p2), each (N, 250), with a_i = macro_i * 50 + waypoint_i.
    Masking matches ``evaluate_actions()`` -- see the module docstring.
    """
    from rl.custom_ppo.strategy_anchor import _masked_heads

    inner = _inner_model(model)
    obs = {"grid": grid, "vec": vec, "agent_mask": amask, "mask": mask}
    heads = _masked_heads(inner, obs)          # agent-major: m1, w1, m2, w2
    if len(heads) != 4:
        raise RuntimeError(f"expected 4 action heads, got {len(heads)}")
    pm1, pw1, pm2, pw2 = (h.probabilities for h in heads)
    p1 = (pm1[:, :, None] * pw1[:, None, :]).reshape(pm1.shape[0], -1)
    p2 = (pm2[:, :, None] * pw2[:, None, :]).reshape(pm2.shape[0], -1)
    return p1, p2


def assert_teacher_query_valid(model, split, teacher_idx, device, *, min_agreement=0.99):
    """Refuse to proceed unless the queried distribution reproduces behaviour.

    Branch actions were recorded with ``predict(deterministic=True)``, so the
    argmax of the queried distribution MUST match the recorded action almost
    everywhere. This is the check that caught the unmasked-fallback defect;
    it runs before any value is computed from a teacher distribution.
    """
    import torch
    from rl.scorer.qpsi import joint_action_index

    m = split.b_teacher == teacher_idx
    if not m.any():
        raise RuntimeError("no branch rows for that teacher")
    t = lambda a: torch.as_tensor(a, dtype=torch.float32).to(device)
    with torch.no_grad():
        p1, p2 = teacher_action_dists(model, t(split.b_grid[m]), t(split.b_vec[m]),
                                      t(split.b_amask[m]), t(split.b_mask[m]))
    a1, a2 = joint_action_index(torch.as_tensor(split.b_action[m], dtype=torch.long))
    ag1 = float((p1.argmax(-1).cpu() == a1).float().mean())
    ag2 = float((p2.argmax(-1).cpu() == a2).float().mean())
    if min(ag1, ag2) < min_agreement:
        raise RuntimeError(
            f"teacher query does not reproduce recorded deterministic actions "
            f"(agent1 {ag1:.3f}, agent2 {ag2:.3f} < {min_agreement}). The "
            f"distribution being scored is not the policy's own behaviour."
        )
    return {"agent1_argmax_agreement": round(ag1, 6),
            "agent2_argmax_agreement": round(ag2, 6), "n_states": int(m.sum())}


def sha256_file(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
