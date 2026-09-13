r"""Production entity-tensor extraction for the 4v4 observation repair.

Feature definition matches experiments/eval_bc_representation_smoke.py::collect
EXACTLY, per the frozen spec's requirement -- same reference frame, same six
features, same self-exclusion rule, no normalization added:

    [dx, dy, dist, alive, carrying, tagged]

dx, dy, dist are RAW cell units (world coordinates, unmirrored, unnormalized --
identical to the smoke test's `bx[j]-bx[i]`, never divided by cols/rows). This
module does NOT touch grid/vec/agent_mask/mask; it is a pure addition alongside
``core.get_obs_tensors``, called separately.

BATCHED, not per-episode: core.blue_x/y, red_x/y, *_alive/carrying/tagged are
already (B, N) tensors, so extraction is direct broadcast + gather, matching
the project's existing GPU-vectorized convention (no Python loop over agents).

VALIDITY SEMANTICS (deliberately not alive-filtering)
    For a FIXED team size, every teammate/enemy INDEX slot always represents a
    real roster member -- validity is about whether the SLOT exists (relevant
    for future variable-N support), not whether that member is currently alive.
    ``alive`` is a FEATURE, exactly as in the smoke test, which never excluded
    dead agents from the entity list. So teammates_valid / enemies_valid are
    all-True for the current fixed-N training regime; the parameter exists so a
    future variable-roster case has a real mask to flip, not a mask to invent.
"""

from __future__ import annotations

import torch

ENTITY_FEATURES = 6          # dx, dy, dist, alive, carrying, tagged


def _pairwise_relative(self_pos: torch.Tensor, other_pos: torch.Tensor) -> torch.Tensor:
    """other[j] - self[i] for every (i, j). self_pos: (B, N), other_pos: (B, M).
    Returns (B, N, M)."""
    return other_pos.unsqueeze(1) - self_pos.unsqueeze(2)


def _exclude_self_index(n: int, device) -> torch.Tensor:
    """(N, N-1) static index table: row i lists the N-1 column indices != i,
    in ascending order. Built once per N; used to gather teammate columns
    without ever including the querying agent itself."""
    all_idx = torch.arange(n, device=device)
    rows = [all_idx[all_idx != i] for i in range(n)]
    return torch.stack(rows, dim=0)


def build_entity_tensors(core, side: str = "blue") -> dict[str, torch.Tensor]:
    """Batched entity tensors for every agent on ``side``, matching the smoke
    test's feature definition exactly.

    Returns a dict with:
        teammates        (B, N, N-1, 6) float32
        teammates_valid  (B, N, N-1)    bool  (all True; see module docstring)
        enemies          (B, N, N,   6) float32
        enemies_valid    (B, N, N)      bool  (all True)

    Only ``side="blue"`` is exercised by this project's training/eval (all
    specialists play blue against scripted red); a red-side call is not
    exercised or tested and is intentionally out of scope here.
    """
    if side != "blue":
        raise NotImplementedError(
            "build_entity_tensors only supports side='blue' -- the only side "
            "this project trains or evaluates as. Extend and re-anchor before "
            "using with side='red'.")

    own_x, own_y = core.blue_x, core.blue_y
    own_alive = core.blue_alive.to(torch.float32)
    own_carry = core.blue_carrying.to(torch.float32)
    own_tag = core.blue_tagged.to(torch.float32)
    enemy_x, enemy_y = core.red_x, core.red_y
    enemy_alive = core.red_alive.to(torch.float32)
    enemy_carry = core.red_carrying.to(torch.float32)
    enemy_tag = core.red_tagged.to(torch.float32)

    B, N = own_x.shape
    device = own_x.device

    # ---- teammates: exclude self via a static gather index -----------------
    dx_full = _pairwise_relative(own_x, own_x)            # (B, N, N)
    dy_full = _pairwise_relative(own_y, own_y)
    idx = _exclude_self_index(N, device)                  # (N, N-1)
    idx_b = idx.unsqueeze(0).expand(B, -1, -1)             # (B, N, N-1)
    dx_t = torch.gather(dx_full, 2, idx_b)
    dy_t = torch.gather(dy_full, 2, idx_b)
    dist_t = torch.sqrt(dx_t * dx_t + dy_t * dy_t + 1e-8)

    def _other_feature_teammate(f: torch.Tensor) -> torch.Tensor:
        # f: (B, N) feature of the CANDIDATE teammate -> gather same as dx/dy
        f_full = f.unsqueeze(1).expand(B, N, N)            # (B, N, N): col j = f[j]
        return torch.gather(f_full, 2, idx_b)

    alive_t = _other_feature_teammate(own_alive)
    carry_t = _other_feature_teammate(own_carry)
    tag_t = _other_feature_teammate(own_tag)
    teammates = torch.stack([dx_t, dy_t, dist_t, alive_t, carry_t, tag_t], dim=-1)
    teammates_valid = torch.ones(B, N, N - 1, dtype=torch.bool, device=device)

    # ---- enemies: no self-exclusion (blue never appears in red's roster) ---
    dx_e = _pairwise_relative(own_x, enemy_x)              # (B, N, N)
    dy_e = _pairwise_relative(own_y, enemy_y)
    dist_e = torch.sqrt(dx_e * dx_e + dy_e * dy_e + 1e-8)
    alive_e = enemy_alive.unsqueeze(1).expand(B, N, N)
    carry_e = enemy_carry.unsqueeze(1).expand(B, N, N)
    tag_e = enemy_tag.unsqueeze(1).expand(B, N, N)
    enemies = torch.stack([dx_e, dy_e, dist_e, alive_e, carry_e, tag_e], dim=-1)
    enemies_valid = torch.ones(B, N, N, dtype=torch.bool, device=device)

    return {"teammates": teammates, "teammates_valid": teammates_valid,
           "enemies": enemies, "enemies_valid": enemies_valid}


def augment_obs_with_entities(obs: dict, core, side: str = "blue") -> dict:
    """Rollout-side helper: numpy obs dict (as produced by the vectorized env)
    -> the SAME dict with 'teammates'/'teammates_valid'/'enemies'/'enemies_valid'
    added, as numpy arrays matching the existing keys' dtype convention.

    Kept UNFLATTENED at (B, N, K, F) / (B, N, K), the same convention as
    obs["grid"]'s (B, N, C, H, W) -- the (B*N) flatten happens inside
    ``SharedActorCentralizedCritic._encode_local_obs``, mirroring exactly how
    grid/vec are already flattened there, not by the caller.

    Returns a NEW dict (shallow: existing array values are not copied) so the
    caller's original obs object is never mutated in place.
    """
    d = build_entity_tensors(core, side)
    return {
        **obs,
        "teammates": d["teammates"].detach().cpu().numpy(),
        "teammates_valid": d["teammates_valid"].detach().cpu().numpy(),
        "enemies": d["enemies"].detach().cpu().numpy(),
        "enemies_valid": d["enemies_valid"].detach().cpu().numpy(),
    }


def flatten_for_policy(entity_dict: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """(B, N, K, F) -> (B*N, K, F), matching the exact flattening order
    ``policy._encode_local_obs`` uses for local_in (batch outer, agent inner:
    ``local_obs.reshape(batch * n_agents, -1)`` on a (batch, n_agents, F)
    tensor keeps agent index fastest within each batch block)."""
    tm, tm_v = entity_dict["teammates"], entity_dict["teammates_valid"]
    en, en_v = entity_dict["enemies"], entity_dict["enemies_valid"]
    B, N, Kt, F = tm.shape
    Ke = en.shape[2]
    return (tm.reshape(B * N, Kt, F), tm_v.reshape(B * N, Kt),
           en.reshape(B * N, Ke, F), en_v.reshape(B * N, Ke))
