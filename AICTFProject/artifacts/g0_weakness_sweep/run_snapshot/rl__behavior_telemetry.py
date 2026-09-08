"""Post-hoc team behavior scalars and buckets for latent strategy diagnostics (Summer Plan).

These features describe *what blue is doing geometrically and tactically* at a decision step.
They are not supervised labels for ``z``; they support MI(z; bucket) and phase-conditioned
occupancy without expanding :data:`rl.latent_phase_labels.TEAM_PHASES`.

**Role buckets** (explicit allocation, team-size–aware):

- **2v2** (ids 0–4): ``both_attack``, ``split_attack_defend``, ``both_defend``,
  ``escort_carrier``, ``intercept_enemy_carrier``.
- **4v4** (ids 0–6): ``all_push``, ``three_attack_one_defend``, ``two_attack_two_defend``,
  ``one_attack_three_defend``, ``escort_pair``, ``intercept_pair``, ``turtle_defense``.

Other team sizes fall back to a coarse 2v2-style partition (same id semantics where possible).

MI and rollout summaries use ``N_ROLE_BUCKET_MI = 7`` columns (2v2 leaves ids 5–6 unused).
"""

from __future__ import annotations

import numpy as np
import torch

from macro_actions import MacroAction

# Order matches ``BEHAVIOR_TELEMETRY_NAMES`` / rollout tensor slice.
BEHAVIOR_TELEMETRY_NAMES: tuple[str, ...] = (
    "team_spread",
    "num_attackers",
    "num_defenders",
    "num_go_to",
    "carrier_escort_count",
    "nearest_blue_to_carrier",
    "nearest_blue_to_enemy_carrier",
    "n_intercept_near_enemy_carrier",
    "avg_blue_to_enemy_flag",
    "avg_blue_to_own_flag",
    "intercept_pressure",
    "defense_pressure",
    "attack_defense_ratio",
)

N_TELEMETRY: int = len(BEHAVIOR_TELEMETRY_NAMES)

# spread_bucket
SPREAD_COMPACT = 0
SPREAD_NORMAL = 1
SPREAD_DISPERSED = 2

# pressure_bucket (defensive threat to blue + offensive pressure from blue)
PRESSURE_LOW = 0
PRESSURE_MEDIUM = 1
PRESSURE_HIGH = 2

# --- 2v2 role ids (0..4) ---
ROLE2_BOTH_ATTACK = 0
ROLE2_SPLIT = 1
ROLE2_BOTH_DEFEND = 2
ROLE2_ESCORT_CARRIER = 3
ROLE2_INTERCEPT_ENEMY_CARRIER = 4

# --- 4v4 role ids (0..6); MI matrix always 7 wide ---
ROLE4_ALL_PUSH = 0
ROLE4_THREE_ONE = 1
ROLE4_TWO_TWO = 2
ROLE4_ONE_THREE = 3
ROLE4_ESCORT_PAIR = 4
ROLE4_INTERCEPT_PAIR = 5
ROLE4_TURTLE = 6

N_ROLE_BUCKET_MI: int = 7

# attack_defense_ratio_bucket (discrete MI on z)
ADR_DEFEND_HEAVY = 0
ADR_MIXED = 1
ADR_ATTACK_HEAVY = 2
N_ATTACK_DEFENSE_RATIO_BUCKET: int = 3


def _diag(core: object) -> float:
    d = float(getattr(core, "max_dist", 1.0))
    return d if d > 1e-6 else 1.0


def compute_behavior_telemetry_batch(core: object, actions: torch.Tensor) -> torch.Tensor:
    """Return ``(B, N_TELEMETRY)`` float32 on ``core.device`` (pre-env-step core state).

    ``actions`` shape ``(B, Nb * 2)`` long — macro at even indices (policy sample for this step).
    """
    dev = core.device
    B = int(core.B)
    Nb = int(core.Nb)
    Nr = int(core.Nr)
    diag = _diag(core)

    bx, by = core.blue_x, core.blue_y
    ba = core.blue_alive
    bc = core.blue_carrying
    rx, ry = core.red_x, core.red_y
    ra = core.red_alive
    rc = core.red_carrying

    rfl = core.red_flag_pos
    bfl = core.blue_flag_pos

    w = ba.to(torch.float32)
    nba = torch.clamp(w.sum(dim=1), min=1.0)

    # --- spread: RMS of per-agent offset from team mean, / diag ---
    cnt = nba
    mx = (bx * w).sum(dim=1) / cnt
    my = (by * w).sum(dim=1) / cnt
    dx = bx - mx[:, None]
    dy = by - my[:, None]
    spread = torch.sqrt(torch.clamp(((dx * dx + dy * dy) * w).sum(dim=1) / cnt, min=0.0) + 1e-8) / diag

    macros = actions[:, 0::2].long()
    alive_m = ba.long()
    n_attack = ((macros == int(MacroAction.GET_FLAG)) & alive_m).sum(dim=1).to(torch.float32)
    n_defend = ((macros == int(MacroAction.GO_HOME)) & alive_m).sum(dim=1).to(torch.float32)
    n_goto = ((macros == int(MacroAction.GO_TO)) & alive_m).sum(dim=1).to(torch.float32)

    rflx = rfl[:, 0:1].expand(B, Nb)
    rfly = rfl[:, 1:2].expand(B, Nb)
    db_rf = torch.sqrt(torch.clamp((bx - rflx) ** 2 + (by - rfly) ** 2, min=0.0) + 1e-8)
    db_rf = torch.where(ba, db_rf, torch.full_like(db_rf, float("inf")))
    avg_b_rf = torch.where(
        ba.any(dim=1),
        torch.where(torch.isfinite(db_rf), db_rf, torch.zeros_like(db_rf)).sum(dim=1) / nba,
        torch.zeros(B, device=dev, dtype=torch.float32),
    ) / diag

    bflx = bfl[:, 0:1].expand(B, Nb)
    bfly = bfl[:, 1:2].expand(B, Nb)
    db_bf = torch.sqrt(torch.clamp((bx - bflx) ** 2 + (by - bfly) ** 2, min=0.0) + 1e-8)
    db_bf = torch.where(ba, db_bf, torch.full_like(db_bf, float("inf")))
    avg_b_bf = torch.where(
        ba.any(dim=1),
        torch.where(torch.isfinite(db_bf), db_bf, torch.zeros_like(db_bf)).sum(dim=1) / nba,
        torch.zeros(B, device=dev, dtype=torch.float32),
    ) / diag

    has_blue_carrier = bc.any(dim=1)
    carr_b = torch.argmax(bc.to(torch.int64), dim=1)
    cx = bx[torch.arange(B, device=dev), carr_b]
    cy = by[torch.arange(B, device=dev), carr_b]
    d_to_c = torch.sqrt(torch.clamp((bx - cx[:, None]) ** 2 + (by - cy[:, None]) ** 2, min=0.0) + 1e-8)
    escort_mask = ba & (~bc) & has_blue_carrier[:, None]
    escort_close = escort_mask & (d_to_c < 6.0)
    escort_cnt = escort_close.sum(dim=1).to(torch.float32)
    d_to_c = torch.where(escort_mask, d_to_c, torch.full_like(d_to_c, float("inf")))
    nb_to_carrier = d_to_c.min(dim=1).values / diag
    nb_to_carrier = torch.where(
        has_blue_carrier,
        torch.where(torch.isfinite(nb_to_carrier), nb_to_carrier, torch.zeros_like(nb_to_carrier)),
        torch.ones(B, device=dev) * 1.5,
    )

    has_red_carrier = rc.any(dim=1)
    carr_r = torch.argmax(rc.to(torch.int64), dim=1)
    ex = rx[torch.arange(B, device=dev), carr_r]
    ey = ry[torch.arange(B, device=dev), carr_r]
    d_to_e = torch.sqrt(torch.clamp((bx - ex[:, None]) ** 2 + (by - ey[:, None]) ** 2, min=0.0) + 1e-8)
    d_to_e = torch.where(ba, d_to_e, torch.full_like(d_to_e, float("inf")))
    n_intercept = (has_red_carrier[:, None] & ba & (d_to_e < 8.0)).sum(dim=1).to(torch.float32)
    nb_to_ec = d_to_e.min(dim=1).values / diag
    nb_to_ec = torch.where(
        has_red_carrier,
        torch.where(torch.isfinite(nb_to_ec), nb_to_ec, torch.zeros_like(nb_to_ec)),
        torch.ones(B, device=dev) * 1.5,
    )

    min_b_rf = db_rf.min(dim=1).values / diag
    min_b_rf = torch.where(torch.isfinite(min_b_rf), min_b_rf, torch.zeros_like(min_b_rf))
    bfx = bfl[:, 0:1].expand(B, Nr)
    bfy = bfl[:, 1:2].expand(B, Nr)
    dr_bf = torch.sqrt(torch.clamp((rx - bfx) ** 2 + (ry - bfy) ** 2, min=0.0) + 1e-8)
    dr_bf = torch.where(ra, dr_bf, torch.full_like(dr_bf, float("inf")))
    min_r_bf = dr_bf.min(dim=1).values / diag
    min_r_bf = torch.where(torch.isfinite(min_r_bf), min_r_bf, torch.zeros_like(min_r_bf))

    intercept_p = torch.clamp(1.0 - min_b_rf, 0.0, 1.0)
    defense_p = torch.clamp(1.0 - min_r_bf, 0.0, 1.0)

    att_w = n_attack + 0.5 * n_goto
    def_w = n_defend + 0.5 * n_goto
    denom_ad = torch.clamp(att_w + def_w, min=1e-3)
    ad_ratio = att_w / denom_ad

    out = torch.stack(
        [
            spread,
            n_attack,
            n_defend,
            n_goto,
            escort_cnt,
            nb_to_carrier,
            nb_to_ec,
            n_intercept,
            avg_b_rf,
            avg_b_bf,
            intercept_p,
            defense_p,
            ad_ratio,
        ],
        dim=1,
    ).to(dtype=torch.float32, device=dev)
    assert out.shape == (B, N_TELEMETRY), out.shape
    return out


def attack_defense_ratio_bucket_id(ratio: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Tertiles on ``[0, 1]``: 0 defend-heavy, 1 mixed, 2 attack-heavy."""
    if isinstance(ratio, torch.Tensor):
        r = ratio.to(dtype=torch.float32)
        out = torch.full_like(r, ADR_MIXED, dtype=torch.long)
        out = torch.where(r < 0.36, torch.full_like(out, ADR_DEFEND_HEAVY), out)
        out = torch.where(r > 0.64, torch.full_like(out, ADR_ATTACK_HEAVY), out)
        return out
    x = np.asarray(ratio, dtype=np.float64)
    out = np.full(x.shape, ADR_MIXED, dtype=np.int64)
    out[x < 0.36] = ADR_DEFEND_HEAVY
    out[x > 0.64] = ADR_ATTACK_HEAVY
    return out


def role_bucket_detailed_id(
    nb: int,
    n_attack: torch.Tensor,
    n_defend: torch.Tensor,
    n_intercept: torch.Tensor,
    escort_cnt: torch.Tensor,
    nb_to_ec: torch.Tensor,
    has_blue_carrier: torch.Tensor,
    has_red_carrier: torch.Tensor,
    n_alive: torch.Tensor,
) -> torch.Tensor:
    """Team-size-aware role bucket ids in ``0..N_ROLE_BUCKET_MI-1`` (torch, same device as inputs)."""
    na = n_attack.to(dtype=torch.float32)
    nd = n_defend.to(dtype=torch.float32)
    ni = n_intercept.to(dtype=torch.float32)
    ec = escort_cnt.to(dtype=torch.float32)
    hbc = has_blue_carrier.to(torch.bool)
    hrc = has_red_carrier.to(torch.bool)
    nlive = torch.clamp(n_alive.to(dtype=torch.float32), min=1.0)

    if nb == 2:
        esc_c = hbc & (ec >= 1.0)
        ic = hrc & ((ni >= 1.0) | (nb_to_ec < 0.22))
        both_a = na >= 2.0
        both_d = nd >= 2.0
        split = (na >= 1.0) & (nd >= 1.0) & ~both_a & ~both_d
        rid = torch.full_like(na, ROLE2_SPLIT, dtype=torch.long)
        rid = torch.where(esc_c, torch.full_like(rid, ROLE2_ESCORT_CARRIER), rid)
        rid = torch.where(~esc_c & ic, torch.full_like(rid, ROLE2_INTERCEPT_ENEMY_CARRIER), rid)
        rid = torch.where(~esc_c & ~ic & both_a, torch.full_like(rid, ROLE2_BOTH_ATTACK), rid)
        rid = torch.where(~esc_c & ~ic & ~both_a & both_d, torch.full_like(rid, ROLE2_BOTH_DEFEND), rid)
        rid = torch.where(~esc_c & ~ic & ~both_a & ~both_d & split, torch.full_like(rid, ROLE2_SPLIT), rid)
        return rid

    if nb == 4:
        esc2 = hbc & (ec >= 2.0)
        ic2 = hrc & (ni >= 2.0)
        ex31 = (na >= 3.0) & (nd >= 1.0) & (nd < 2.0)
        ex22 = (na >= 2.0) & (nd >= 2.0) & (na < 3.0) & (nd < 3.0)
        ex13 = (na >= 1.0) & (na < 2.0) & (nd >= 3.0)
        rid = torch.full_like(na, ROLE4_TWO_TWO, dtype=torch.long)
        rid = torch.where(esc2, torch.full_like(rid, ROLE4_ESCORT_PAIR), rid)
        rid = torch.where(~esc2 & ic2, torch.full_like(rid, ROLE4_INTERCEPT_PAIR), rid)
        rid = torch.where(~esc2 & ~ic2 & ex31, torch.full_like(rid, ROLE4_THREE_ONE), rid)
        rid = torch.where(~esc2 & ~ic2 & ~ex31 & ex22, torch.full_like(rid, ROLE4_TWO_TWO), rid)
        rid = torch.where(~esc2 & ~ic2 & ~ex31 & ~ex22 & ex13, torch.full_like(rid, ROLE4_ONE_THREE), rid)
        push = (~esc2) & (~ic2) & (~ex31) & (~ex22) & (~ex13) & (na / nlive >= 0.75)
        tur = (~esc2) & (~ic2) & (~ex31) & (~ex22) & (~ex13) & (~push) & (nd / nlive >= 0.75)
        rid = torch.where(push, torch.full_like(rid, ROLE4_ALL_PUSH), rid)
        rid = torch.where(tur, torch.full_like(rid, ROLE4_TURTLE), rid)
        return rid

    # Fallback (3v3, 6v6, 8v8): coarse escort / intercept / push / turtle / split
    esc_thr = torch.maximum(torch.ones_like(nlive), nlive * 0.4)
    base_esc = hbc & (ec >= esc_thr)
    base_ic = (~base_esc) & hrc & ((ni >= 1.0) | (nb_to_ec < 0.25))
    rid = torch.full_like(na, ROLE2_SPLIT, dtype=torch.long)
    rid = torch.where(base_esc, torch.full_like(rid, ROLE2_ESCORT_CARRIER), rid)
    rid = torch.where(base_ic, torch.full_like(rid, ROLE2_INTERCEPT_ENEMY_CARRIER), rid)
    push = ~base_esc & ~base_ic & (na / nlive >= 0.75)
    tur = ~base_esc & ~base_ic & ~push & (nd / nlive >= 0.75)
    rid = torch.where(push, torch.full_like(rid, ROLE4_ALL_PUSH), rid)
    rid = torch.where(tur, torch.full_like(rid, ROLE4_TURTLE), rid)
    return torch.clamp(rid, 0, N_ROLE_BUCKET_MI - 1)


def spread_bucket_id(team_spread: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """0=compact, 1=normal, 2=dispersed (normalized spread)."""
    if isinstance(team_spread, torch.Tensor):
        t = team_spread.to(dtype=torch.float32)
        b = torch.full_like(t, SPREAD_NORMAL, dtype=torch.long)
        b = torch.where(t < 0.05, torch.full_like(b, SPREAD_COMPACT), b)
        b = torch.where(t > 0.12, torch.full_like(b, SPREAD_DISPERSED), b)
        return b
    x = np.asarray(team_spread, dtype=np.float64)
    out = np.full(x.shape, SPREAD_NORMAL, dtype=np.int64)
    out[x < 0.05] = SPREAD_COMPACT
    out[x > 0.12] = SPREAD_DISPERSED
    return out


def pressure_bucket_id(intercept_pressure: torch.Tensor | np.ndarray, defense_pressure: torch.Tensor | np.ndarray):
    """0=low, 1=medium, 2=high from combined pressure in ``[0, 1]`` (both columns already high=hot)."""
    if isinstance(intercept_pressure, torch.Tensor):
        a = intercept_pressure.to(dtype=torch.float32)
        d = defense_pressure.to(dtype=torch.float32)
        threat = torch.clamp(0.5 * (a + d), 0.0, 1.0)
        out = torch.full_like(a, PRESSURE_MEDIUM, dtype=torch.long)
        out = torch.where(threat < 0.33, torch.full_like(out, PRESSURE_LOW), out)
        out = torch.where(threat > 0.66, torch.full_like(out, PRESSURE_HIGH), out)
        return out
    a = np.asarray(intercept_pressure, dtype=np.float64)
    d = np.asarray(defense_pressure, dtype=np.float64)
    threat = np.clip(0.5 * (a + d), 0.0, 1.0)
    out = np.full(a.shape, PRESSURE_MEDIUM, dtype=np.int64)
    out[threat < 0.33] = PRESSURE_LOW
    out[threat > 0.66] = PRESSURE_HIGH
    return out


def bucket_ids_from_telemetry(
    telemetry: torch.Tensor,
    actions: torch.Tensor,
    core: object,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(spread_bucket, role_bucket, pressure_bucket, attack_defense_ratio_bucket)``."""
    spread = telemetry[:, 0]
    n_att = telemetry[:, 1]
    n_def = telemetry[:, 2]
    esc = telemetry[:, 4]
    n_int = telemetry[:, 7]
    nb_to_ec = telemetry[:, 6]
    ad_ratio = telemetry[:, 12]
    ip = telemetry[:, 10]
    dp = telemetry[:, 11]
    sb = spread_bucket_id(spread)
    Nb = int(actions.shape[1] // 2)
    n_alive = core.blue_alive.sum(dim=1).to(dtype=torch.float32)
    rb = role_bucket_detailed_id(
        Nb,
        n_att,
        n_def,
        n_int,
        esc,
        nb_to_ec,
        core.blue_carrying.any(dim=1),
        core.red_carrying.any(dim=1),
        n_alive,
    )
    pb = pressure_bucket_id(ip, dp)
    adb = attack_defense_ratio_bucket_id(ad_ratio)
    return sb, rb, pb, adb
