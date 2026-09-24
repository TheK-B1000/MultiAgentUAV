"""SCAFFOLD_TO_NATIVE_REPRESENTABILITY_AUDIT_4V4_V1_SPEC.json.

READ-ONLY audit. Could native pi_A's existing 4v4 action interface have produced the defender targets the scaffold
injected into the SEALED A' episodes, without the injection? TARGET / LEGALITY / DIRECTION / TRAJECTORY, reusing the
frozen G6 sub-gate definitions and thresholds. No PPO, no training, no new outcome seed, pi_B untouched.

  contracts   Structural + parity contracts (replays 4 sealed cells to check they reproduce their sealed rows).
  replay      One shard of the 256 sealed A' cells: deterministic replay that RECORDS the forced defenders' per-tick
              state and legal-action mask. Resumable append-only partial file per shard. Prints counts only.
  analyze     Refuses unless all 256 cells are present. Runs the four sub-gates on the recorded states, writes the
              per-cell CSV, re-derives the gate rates from that CSV, and writes the labeled result.

Everything the analysis needs from the simulator is obtained by CALLING the real core objects (`_build_targets_from_action`,
`_defend_outward_target`, `_integrate_side`, `_build_action_mask`, `_macro_commit_ticks`); the few vectorized re-statements
are pinned by parity contracts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.probe_learned_composition as P  # noqa: E402
import experiments.run_learned_composition_probe as L  # noqa: E402
import experiments.run_scaffolded_a_crossover_bridge_4v4 as S  # noqa: E402  (imported, never modified)
from experiments.tqdm_loop import tqdm_iter  # noqa: E402

SD = S.SD
STEM = "SCAFFOLD_TO_NATIVE_REPRESENTABILITY_AUDIT_4V4"
SPEC_PATH = SD / f"{STEM}_V1_SPEC.json"
CONTRACT_PATH = SD / f"{STEM}_CONTRACT_RESULT.json"
RESULT_PATH = SD / f"{STEM}_RESULT.json"
CELLS_PATH = SD / f"{STEM}_CELLS.csv"
SHARD_GLOB = f"{STEM}_REPLAY_SHARD*_PARTIAL.jsonl"

N = S.N
POLES = S.POLES
SEEDS = S.SEEDS
DEVICE = S.DEVICE
HORIZON = 16
COS_MIN = 0.99
RADIUS_TOL = 2.5          # TARGET threshold and TRAJECTORY RMSE threshold (G6, unchanged)
STRIDE = 2                # window start stride (frozen)
COVER = 0.90              # coverage level (frozen, this audit's own choice)
N_BOOT, BOOT_SEED = 20000, 7
PROGRESS_TOL = 1e-5
ONE_STEP_TOL, ONE_STEP_MIN_FRAC = 1e-3, 0.95

STATE_FIELDS = ("alive", "tagged", "carrying", "mines", "x", "y", "h", "v", "Fx", "Fy", "Ex", "Ey", "cm", "ct", "cl", "dmin")
IX = {n: i for i, n in enumerate(STATE_FIELDS)}
GO_TO, GRAB, GET_FLAG, PLACE, GO_HOME = 0, 1, 2, 3, 4
RT_DEFAULTS = {"rt_blue_speed_scale": 1.0, "rt_current_strength_cps": 0.0, "rt_drift_sigma_cells": 0.0,
               "rt_sensor_dropout_prob": 0.0, "rt_sensor_noise_sigma_cells": 0.0}
GUARD = ("A statement about what the native 4v4 action interface can express relative to the scaffold's injected defender "
         "behavior on the sealed A' episodes. HIGH_COVERAGE_REPRESENTABLE_NATIVE means the episode-clustered LCB95 of the pass "
         "rate reached the pre-declared 0.90 on every gate and pole using a GO_TO-only, greedy, future-seeing path-oracle; it is "
         "constructive evidence of high-coverage representability, NOT a claim that the interface reproduces DEFEND at every "
         "state. A NOT_REPRESENTABLE_* label means the frozen coverage was not reached by that greedy oracle, not that no "
         "action sequence exists. Representability is not learnability, not a win-rate claim, and not evidence about pi_B or 6v6.")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def _sha(p: Path) -> str:
    return S._sha_file(Path(p))


def _cells() -> list[tuple[str, int]]:
    return [(pole, seed) for seed in SEEDS for pole in POLES]


def _sealed_rows() -> dict[tuple[str, int], tuple[int, int, int]]:
    out = {}
    with S.ROWS_PATH.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["arm"] == "A_prime":
                out[(r["pole"], int(r["seed"]))] = (int(r["steps"]), int(r["blue"]), int(r["red"]))
    return out


# ============================================================================================ frozen decision logic

def classify(lcb: float, ucb: float, cover: float = COVER) -> str:
    if lcb >= cover:
        return "PASS"
    if ucb < cover:
        return "FAIL"
    return "BORDERLINE"


def terminal_label(g: dict[str, dict[str, str]], invalid: bool = False) -> str:
    """g[gate][pole] in {PASS, FAIL, BORDERLINE}. Precedence: AUDIT_INVALID > vocabulary fail > borderline > trajectory."""
    if invalid:
        return "AUDIT_INVALID"
    vocab = [g["TARGET"][p] for p in POLES] + [g["DIRECTION"][p] for p in POLES]
    if "FAIL" in vocab:
        return "NOT_REPRESENTABLE_VOCABULARY"
    if "BORDERLINE" in vocab:
        return "INCONCLUSIVE_BORDERLINE"
    tn = [g["TRAJ_NATIVE"][p] for p in POLES]
    if "FAIL" not in tn and "BORDERLINE" in tn:
        return "INCONCLUSIVE_BORDERLINE"
    if all(s == "PASS" for s in tn):
        return "HIGH_COVERAGE_REPRESENTABLE_NATIVE"
    ti = [g["TRAJ_INTR"][p] for p in POLES]
    if all(s == "PASS" for s in ti):
        return "NOT_REPRESENTABLE_COMMITMENT_BOUND"
    if "FAIL" in ti:
        return "NOT_REPRESENTABLE_OTHER"
    return "INCONCLUSIVE_BORDERLINE"


def direction_gate(err: np.ndarray, cosv: np.ndarray) -> tuple[bool, bool, bool]:
    """DIRECTION as EXISTENCE over the legal witness set (never 'the target-nearest candidate only'). Returns
    (gate, joint, any). joint: some TARGET-satisfying candidate (err <= 2.5) is aimed (cos >= 0.99). any: some candidate is
    aimed. gate = joint when the TARGET-satisfying set is non-empty, else any -- so a TARGET miss does not double-count as
    a DIRECTION miss, and an unlucky nearest-waypoint tie cannot manufacture a vocabulary failure."""
    near = err <= RADIUS_TOL
    anyd = bool(cosv.max() >= COS_MIN)
    joint = bool(near.any() and cosv[near].max() >= COS_MIN)
    return (joint if bool(near.any()) else anyd), joint, anyd


def cluster_rate_ci(num, den, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    num, den = np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)
    if den.sum() <= 0:
        return float("nan"), float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(num), size=(n_boot, len(num)))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1e-12)
    return float(num.sum() / den.sum()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


# ============================================================================================ recorder (replay)

def _payload_only_mask(core):
    """The REAL action mask with the commit lock removed, restoring the commit state exactly."""
    saved = core.blue_commit_ticks_left.clone()
    core.blue_commit_ticks_left.zero_()
    try:
        return core._build_action_mask(side="blue")
    finally:
        core.blue_commit_ticks_left.copy_(saved)


def _snap(core, forced, nM: int, nT: int):
    import torch
    idx = torch.as_tensor(list(forced), device=core.blue_x.device, dtype=torch.long)
    nb = int(core.blue_x.shape[1])
    F, E = core.blue_flag_pos[0], core.red_flag_pos[0]
    # distance from each blue agent to the nearest OTHER live agent (blue or red): the only thing the live step's
    # avoid-collision shove depends on, and the only thing the isolated single-agent rollout leaves out
    bx, by, ba = core.blue_x[0], core.blue_y[0], core.blue_alive[0]
    rx, ry, ra = core.red_x[0], core.red_y[0], core.red_alive[0]
    big = torch.full((), 1e3, dtype=bx.dtype, device=bx.device)
    eye = torch.eye(nb, dtype=torch.bool, device=bx.device)
    d_bb = torch.where(ba[None, :] & ~eye, torch.hypot(bx[None, :] - bx[:, None], by[None, :] - by[:, None]), big)
    d_br = torch.where(ra[None, :], torch.hypot(rx[None, :] - bx[:, None], ry[None, :] - by[:, None]), big)
    dmin = torch.minimum(d_bb.min(dim=1).values, d_br.min(dim=1).values)
    per = torch.stack([
        core.blue_alive[0].float(), core.blue_tagged[0].float(), core.blue_carrying[0].float(),
        core.blue_mine_charges[0].float(), core.blue_x[0].float(), core.blue_y[0].float(),
        core.blue_heading[0].float(), core.blue_speed[0].float(),
        F[0].float().expand(nb), F[1].float().expand(nb), E[0].float().expand(nb), E[1].float().expand(nb),
        core.blue_commit_macro[0].float(), core.blue_commit_target[0].float(), core.blue_commit_ticks_left[0].float(),
        dmin.float(),
    ], dim=1)[idx]
    m = (_payload_only_mask(core).reshape(nb, nM + nT)[idx] > 0).cpu().numpy().astype(np.int64)
    bits = m @ (np.int64(1) << np.arange(nM + nT, dtype=np.int64))
    return per.cpu().numpy().astype(np.float64), bits


def replay_cell(setup: dict, policy, pole: str, seed: int, device: str) -> dict[str, Any]:
    """The sealed A' episode, replayed action for action (S.run_episode), recording the forced defenders each tick."""
    import torch
    from experiments.opponent_spec import assert_live_opponent_batch
    from gpu_env._core._entity_obs import augment_obs_with_entities
    R2 = setup["R2"]
    env = R2.build_env(device, seed)
    core = env.core
    forced = S.pair_for_seed(seed)
    nM, nT = int(core.cfg.n_macros), int(core.cfg.n_targets)
    try:
        policy.reset_strategy()
        gen, key = S._open_opponent(env, core, setup["genomes"], pole, "representability audit")
        obs = env.reset()
        obs["global_state"] = env.state()
        obs = augment_obs_with_entities(obs, core, side="blue")
        assert_live_opponent_batch(core, gen, allowed_keys=(key,), context=f"repr audit {pole} seed {seed}")
        got = core._bt_resolved_profile_tensors().get("min_alive_for_defender")
        got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
        if got_val != N:
            raise SystemExit(f"FAIL-CLOSED: live pole {pole} resolves min_alive_for_defender={got_val}, expected {N}")
        rt = {k: float(v.reshape(-1)[0]) for k, v in core.__dict__.items()
              if k.startswith("rt_") and torch.is_tensor(v) and v.numel() == 1}
        home = core.blue_flag_home[0].detach().cpu().numpy().astype(np.float64).tolist()
        for i in forced:
            P.install_forced_defend_target(core, int(i))
        ticks, bits_l, terminal, steps = [], [], None, 0
        for _ in range(R2.MAX_STEPS):
            st, bt = _snap(core, forced, nM, nT)
            ticks.append(st)
            bits_l.append(bt)
            action, _ = policy.predict(obs, deterministic=True)
            env.step_async(action)
            obs, _r, done, info = env.step_wait()
            steps += 1
            obs["global_state"] = env.state()
            obs = augment_obs_with_entities(obs, core, side="blue")
            if bool(np.asarray(done).any()):
                i0 = info[0] if isinstance(info, (list, tuple)) else info
                res = (i0 or {}).get("episode_result") or {}
                terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                break
        if terminal is None:
            terminal = (int(core.blue_score[0]), int(core.red_score[0]))
        meta = {"pole": pole, "seed": int(seed), "forced": list(map(int, forced)), "steps": steps,
                "blue": terminal[0], "red": terminal[1], "rt": rt, "dt": float(core.dt), "home": home}
        return {"meta": meta,
                "state": np.transpose(np.stack(ticks), (1, 0, 2)).tolist(),      # (2, T, 15)
                "legal": np.stack(bits_l).T.tolist()}                            # (2, T)
    finally:
        env.close()


def _shard_path(i: int, k: int) -> Path:
    return SD / f"{STEM}_REPLAY_SHARD{i}OF{k}_PARTIAL.jsonl"


def _load_partial(path: Path) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                out[tuple(o["key"])] = o["cell"]
    return out


def _parity_ok(meta: dict, sealed: dict) -> bool:
    return sealed.get((meta["pole"], meta["seed"])) == (meta["steps"], meta["blue"], meta["red"])


def _pi_a_path_and_pin() -> tuple[Path, str]:
    spec = _load_json(SPEC_PATH)
    return ROOT / spec["INTEGRITY_FROZEN"]["pi_A_checkpoint"]["path"], spec["INTEGRITY_FROZEN"]["pi_A_checkpoint"]["sha256"]


def _sealed_inputs_ok() -> tuple[bool, str]:
    pins = _load_json(SPEC_PATH)["INTEGRITY_FROZEN"]["sealed_inputs_sha256"]
    now = {"scaffold_spec": _sha(S.SPEC_PATH), "scaffold_result": _sha(S.RESULT_PATH), "scaffold_rows": _sha(S.ROWS_PATH),
           "scaffold_runner": _sha(Path(S.__file__)), "mechanism_source": _sha(S.MECHANISM_FILE)}
    bad = [k for k in pins if pins[k] != now[k]]
    return (not bad, "all five sealed inputs equal their pins" if not bad else f"DIFFER: {bad}")


def run_shard(shard: int, n_shards: int) -> int:
    if _load_json(CONTRACT_PATH).get("DECISION") != "CONTRACTS_PASS":
        raise SystemExit("REFUSING: contracts did not pass")
    if RESULT_PATH.exists():
        raise SystemExit(f"REFUSING: result already exists: {RESULT_PATH}")
    ok, msg = _sealed_inputs_ok()
    if not ok:
        raise SystemExit(f"REFUSING: {msg}")
    ck, pin = _pi_a_path_and_pin()
    if _sha(ck) != pin:
        raise SystemExit("REFUSING: pi_A checkpoint differs from its pin")
    mine = [c for i, c in enumerate(_cells()) if i % n_shards == shard]
    path = _shard_path(shard, n_shards)
    done = _load_partial(path)
    pending = [c for c in mine if c not in done]
    print(f"  shard {shard}/{n_shards}: {len(mine)} cells, {len(done)} recorded, {len(pending)} to replay", flush=True)
    if not pending:
        return 0
    import torch
    sealed = _sealed_rows()
    setup = S._setup_poles()
    pols = S._load_policies(setup["R2"], {"pi_A": ck}, DEVICE)
    before = L._param_digest(pols["pi_A"])
    with path.open("a", encoding="utf-8") as fh, torch.no_grad():
        for pole, seed in tqdm_iter(pending, desc=f"repr-audit replay {shard}/{n_shards}", total=len(pending), unit="ep"):
            cell = replay_cell(setup, pols["pi_A"], pole, seed, DEVICE)
            if not _parity_ok(cell["meta"], sealed):
                raise SystemExit(f"ABORT: replay of ({pole},{seed}) does not reproduce its sealed row "
                                 f"(got {(cell['meta']['steps'], cell['meta']['blue'], cell['meta']['red'])}, "
                                 f"sealed {sealed.get((pole, seed))})")
            fh.write(json.dumps({"key": [pole, seed], "cell": cell}) + "\n")
            fh.flush()
    if before != L._param_digest(pols["pi_A"]):
        raise SystemExit("ABORT: pi_A parameters changed during a read-only run")
    print(f"  shard {shard}/{n_shards} complete (parity held; parameter digest unchanged)", flush=True)
    return 0


# ============================================================================================ analysis engine

class Engine:
    """The real core objects on CPU, plus thin vectorized helpers whose equality with the live functions is contracted."""

    def __init__(self) -> None:
        import torch
        import experiments.r2_learned_crossover as R2
        from gpu_env._core._rules import _pyquaticus_defender_radius_cells
        R2.AGENTS = N
        self.T = torch
        self.env = R2.build_env("cpu", 1)
        self.core = self.env.core
        self.env.reset()
        c, cfg = self.core, self.core.cfg
        self.nM, self.nT = int(cfg.n_macros), int(cfg.n_targets)
        self.wp = c._macro_targets.detach().cpu().numpy().astype(np.float64)
        self.home = c.blue_flag_home[0].detach().cpu().numpy().astype(np.float64)
        self.max_speed = float(cfg.max_speed_cps)
        self.scale = float(c.rt_blue_speed_scale.reshape(-1)[0])
        self.radius = float(_pyquaticus_defender_radius_cells(float(cfg.tag_range_cells)))
        self.commit = {m: int(c._macro_commit_ticks(torch.tensor([[m]], dtype=torch.int64))[0, 0]) for m in range(self.nM)}
        self.arrival = float(cfg.macro_arrival_radius_cells)
        # The live step ends with a tangential repulsion shove (0.5 cells) between agents closer than avoid_collision_radius.
        # A shove at tick t needs some agent within  radius + (max relative displacement in one tick) + (two 0.5-cell shove
        # stages)  of the defender at the START of tick t. Outside that reach the isolated rollout must equal the live step.
        self.avoid_r = float(cfg.avoid_collision_radius_cells)
        self.reach = self.avoid_r + 2.0 * 2.2 * float(c.dt) + 1.0
        self._cand_cache: dict[tuple[int, bool], tuple[np.ndarray, np.ndarray]] = {}

    # ---- live-function wrappers -------------------------------------------------------------------------------
    def defend_t(self, xt, yt, ht, flag):
        """DEFEND target for (1,K) state tensors and a (1,2) flag tensor: calls the real _defend_outward_target and mirrors
        the six-line inward/outward selection of _build_targets_from_action (parity-contracted, C5)."""
        T = self.T
        otx, oty = self.core._defend_outward_target({"own_x": xt, "own_y": yt, "own_heading": ht}, flag)
        ax, ay = xt - flag[:, None, 0], yt - flag[:, None, 1]
        inward = T.sqrt(ax * ax + ay * ay) > self.radius
        tx = T.where(inward, flag[:, None, 0].expand_as(otx), otx)
        ty = T.where(inward, flag[:, None, 1].expand_as(oty), oty)
        return tx, ty, inward

    def defend(self, x: float, y: float, h: float, F) -> tuple[float, float, bool]:
        T = self.T
        f32 = T.float32
        tx, ty, inward = self.defend_t(T.tensor([[x]], dtype=f32), T.tensor([[y]], dtype=f32), T.tensor([[h]], dtype=f32),
                                       T.tensor([[F[0], F[1]]], dtype=f32))
        return float(tx), float(ty), bool(inward)

    def step_t(self, X, Y, H, V, TX, TY):
        T = self.T
        cap = T.full_like(V, self.max_speed) * self.scale
        return self.core._integrate_side(X, Y, H, V, T.ones_like(X, dtype=T.bool), TX, TY, speed_cap=cap)[:4]

    def step_one(self, x, y, h, v, tx, ty):
        T = self.T
        f = lambda a: T.tensor([[a]], dtype=T.float32)  # noqa: E731
        nx, ny, nh, nv = self.step_t(f(x), f(y), f(h), f(v), f(tx), f(ty))
        return float(nx), float(ny), float(nh), float(nv)

    def resolve_native(self, macro: np.ndarray, idx: np.ndarray, E, home) -> tuple[np.ndarray, np.ndarray]:
        tx, ty = self.wp[idx, 0].copy(), self.wp[idx, 1].copy()
        tx = np.where(macro == GET_FLAG, E[0], np.where(macro == GO_HOME, home[0], tx))
        ty = np.where(macro == GET_FLAG, E[1], np.where(macro == GO_HOME, home[1], ty))
        return tx, ty

    # ---- candidate sets --------------------------------------------------------------------------------------------
    def cands(self, bits: int, unrestricted: bool = False, motion_only: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """motion_only=True is the WITNESS set used by every gate: GO_TO waypoints only. GO_TO's whole semantics is
        'move toward the waypoint, commit ends on arrival or after 4 ticks' -- fully modeled in the isolated rollout --
        whereas GRAB/PLACE/GET_FLAG/GO_HOME carry event-dependent side effects the rollout does not model."""
        key = (int(bits), unrestricted, motion_only)
        if key not in self._cand_cache:
            mac, idx = [], []
            for m in range(self.nM):
                if motion_only and m != GO_TO:
                    continue
                if not (unrestricted or (bits >> m) & 1):
                    continue
                if m in (GET_FLAG, GO_HOME):
                    mac.append(m)
                    idx.append(0)
                else:
                    for j in range(self.nT):
                        if unrestricted or (bits >> (self.nM + j)) & 1:
                            mac.append(m)
                            idx.append(j)
            self._cand_cache[key] = (np.asarray(mac, dtype=np.int64), np.asarray(idx, dtype=np.int64))
        return self._cand_cache[key]

    def is_legal(self, bits: int, m: int, j: int) -> bool:
        return bool((bits >> m) & 1) and (m in (GET_FLAG, GO_HOME) or bool((bits >> (self.nM + j)) & 1))

    # ---- L1: per ACTIVE tick, free choice ---------------------------------------------------------------------------------
    @staticmethod
    def cos_vec(px, py, rx, ry, tx, ty) -> np.ndarray:
        """Vectorized replica of gpu_env.pyquaticus_port.direction_cosine (identical zero-vector conventions);
        equality with the unchanged function is contracted (C14)."""
        lx, ly, gx, gy = rx - px, ry - py, tx - px, ty - py
        ln, gn = np.hypot(lx, ly), np.hypot(gx, gy)
        z_l, z_g = ln <= 1e-8, gn <= 1e-8
        val = np.clip((lx * gx + ly * gy) / np.maximum(ln * gn, 1e-300), -1.0, 1.0)
        return np.where(z_l & z_g, 1.0, np.where(z_l | z_g, -1.0, val))

    def l1_tick(self, s: np.ndarray, bits: int) -> dict[str, Any]:
        """Per ACTIVE tick. Witness set W = legal GO_TO waypoints. TARGET: some w in W within 2.5 of T*. DIRECTION: EXISTENCE
        over W -- among the TARGET-satisfying members when there are any, otherwise over all of W -- of a candidate whose
        direction cosine >= 0.99 (never just the target-nearest candidate)."""
        x, y, h = s[IX["x"]], s[IX["y"]], s[IX["h"]]
        F, E = (s[IX["Fx"]], s[IX["Fy"]]), (s[IX["Ex"]], s[IX["Ey"]])
        tx, ty, inward = self.defend(x, y, h, F)
        wm, wi = self.cands(bits, motion_only=True)
        out: dict[str, Any] = {"inward": inward, "gohome_legal": bool((bits >> GO_HOME) & 1), "w_empty": len(wm) == 0}
        umac, uidx = self.cands(0, unrestricted=True)
        ux, uy = self.resolve_native(umac, uidx, E, self.home)
        uerr = np.hypot(ux - tx, uy - ty)
        ub = int(np.lexsort((uidx, umac, uerr))[0])
        out.update(exact_illegal=not self.is_legal(bits, int(umac[ub]), int(uidx[ub])), uerr=float(uerr[ub]))
        am, ai = self.cands(bits)                                  # diagnostic only: every legal macro
        ax, ay = self.resolve_native(am, ai, E, self.home)
        out["err_all_macros"] = float(np.min(np.hypot(ax - tx, ay - ty))) if len(am) else float("inf")
        if out["w_empty"]:
            return out | {"err": float("inf"), "dir_gate": False, "dir_joint": False, "dir_any": False, "legal_ok": False}
        rx, ry = self.wp[wi, 0], self.wp[wi, 1]
        err = np.hypot(rx - tx, ry - ty)
        gate, joint, anyd = direction_gate(err, self.cos_vec(x, y, rx, ry, tx, ty))
        return out | {"err": float(err.min()), "dir_gate": gate, "dir_joint": joint, "dir_any": anyd, "legal_ok": True}

    # ---- L2: 16-tick isolated rollouts ------------------------------------------------------------------------------------
    def rollout_ref(self, st_win: np.ndarray):
        s = st_win[0]
        x, y, h, v = s[IX["x"]], s[IX["y"]], s[IX["h"]], s[IX["v"]]
        pos, tstar, pre = np.zeros((HORIZON, 2)), np.zeros((HORIZON, 2)), []
        for k in range(HORIZON):
            tx, ty, _ = self.defend(x, y, h, (st_win[k, IX["Fx"]], st_win[k, IX["Fy"]]))
            pre.append((x, y))
            tstar[k] = (tx, ty)
            x, y, h, v = self.step_one(x, y, h, v, tx, ty)
            pos[k] = (x, y)
        return pos, tstar, pre

    def path_oracle(self, st_win: np.ndarray, bits: int, ref_pos: np.ndarray, interruptible: bool):
        T = self.T
        f32 = T.float32
        mac, idx = self.cands(bits, motion_only=True)          # witness = legal GO_TO waypoints only (see cands)
        K = len(mac)
        if K == 0:
            raise RuntimeError("empty witness set at a window start: GO_TO is masked for an ACTIVE agent")
        Lc = np.ones(K, dtype=np.int64) if interruptible else np.asarray([self.commit[int(m)] for m in mac], dtype=np.int64)
        wx, wy = self.wp[idx, 0], self.wp[idx, 1]
        m2 = m4 = np.zeros(K, dtype=bool)                       # no flag-derived targets in a GO_TO-only witness
        is_goto = mac == GO_TO
        s = st_win[0]
        x, y, h, v = s[IX["x"]], s[IX["y"]], s[IX["h"]], s[IX["v"]]
        pos, tgt, pre, chosen, t = np.zeros((HORIZON, 2)), np.zeros((HORIZON, 2)), [], [], 0
        while t < HORIZON:
            Lmax = min(int(Lc.max()), HORIZON - t)
            X, Y, Hh, V = (T.full((1, K), a, dtype=f32) for a in (x, y, h, v))
            errs, cnt, ended, hist = np.zeros(K), np.zeros(K, dtype=np.int64), np.zeros(K, dtype=bool), []
            for j in range(Lmax):
                tx, ty = wx, wy
                X, Y, Hh, V = self.step_t(X, Y, Hh, V, T.tensor(tx[None], dtype=f32), T.tensor(ty[None], dtype=f32))
                xn, yn = X[0].numpy().astype(np.float64), Y[0].numpy().astype(np.float64)
                committed = (j < Lc) & ~ended
                errs += np.where(committed, (xn - ref_pos[t + j, 0]) ** 2 + (yn - ref_pos[t + j, 1]) ** 2, 0.0)
                cnt += committed
                ended |= committed & is_goto & (np.hypot(xn - wx, yn - wy) <= self.arrival)
                hist.append((xn, yn, Hh[0].numpy().astype(np.float64), V[0].numpy().astype(np.float64), tx, ty))
            kb = int(np.lexsort((idx, mac, errs / np.maximum(cnt, 1)))[0])
            n = int(cnt[kb])
            for j in range(n):
                pre.append((x, y, h, v) if j == 0 else tuple(hist[j - 1][q][kb] for q in range(4)))
                pos[t + j] = (hist[j][0][kb], hist[j][1][kb])
                tgt[t + j] = (hist[j][4][kb], hist[j][5][kb])
            x, y, h, v = (hist[n - 1][q][kb] for q in range(4))
            chosen.append((int(mac[kb]), int(idx[kb])))
            t += n
        return pos, tgt, pre, chosen

    def window(self, st: np.ndarray, bits: int, t0: int) -> dict[str, Any]:
        from gpu_env.pyquaticus_port import direction_cosine
        T = self.T
        st_win = st[t0:t0 + HORIZON]
        ref_pos, ref_tstar, ref_pre = self.rollout_ref(st_win)
        s0 = np.asarray([st_win[0, IX["x"]], st_win[0, IX["y"]]])
        nat_pos, nat_tgt, nat_pre, nat_chosen = self.path_oracle(st_win, bits, ref_pos, interruptible=False)
        itr_pos, _t, _p, itr_chosen = self.path_oracle(st_win, bits, ref_pos, interruptible=True)
        rmse = lambda p: float(np.sqrt(np.mean(np.sum((p - ref_pos) ** 2, axis=1))))  # noqa: E731
        legal_bad = sum(0 if self.is_legal(bits, m, j) else 1 for m, j in nat_chosen + itr_chosen)
        # G6-hard-check comparator on the native-commit path (diagnostic only)
        tgt_max, radial_ok, cos_first = 0.0, True, 1.0
        ref_prev = s0
        for k in range(HORIZON):
            px, py, ph, _pv = nat_pre[k]
            tx, ty, _ = self.defend(px, py, ph, (st_win[k, IX["Fx"]], st_win[k, IX["Fy"]]))
            tgt_max = max(tgt_max, float(np.hypot(nat_tgt[k, 0] - tx, nat_tgt[k, 1] - ty)))
            if k == 0:
                f = lambda a, c: T.tensor([a, c], dtype=T.float64)  # noqa: E731
                cos_first = direction_cosine(f(px, py), f(float(nat_tgt[0, 0]), float(nat_tgt[0, 1])), f(tx, ty))
            tp = float(np.dot(ref_pos[k] - ref_prev, ref_tstar[k] - ref_prev))
            pp = float(np.dot(nat_pos[k] - np.asarray([px, py]), np.asarray([tx, ty]) - np.asarray([px, py])))
            radial_ok &= (tp >= -PROGRESS_TOL) == (pp >= -PROGRESS_TOL)
            ref_prev = ref_pos[k]
        live = st[t0 + 1:t0 + HORIZON, [IX["x"], IX["y"]]]
        fid = float(np.sqrt(np.mean(np.sum((ref_pos[:len(live)] - live) ** 2, axis=1)))) if len(live) else 0.0
        r_nat, r_itr = rmse(nat_pos), rmse(itr_pos)
        return {"inward0": bool(np.hypot(*(s0 - np.asarray([st_win[0, IX["Fx"]], st_win[0, IX["Fy"]]]))) > self.radius),
                "nat": r_nat, "itr": r_itr, "g6_tgt": tgt_max <= RADIUS_TOL, "g6_dir": (cos_first >= COS_MIN) and radial_ok,
                "g6_all": (tgt_max <= RADIUS_TOL) and (cos_first >= COS_MIN) and radial_ok and (r_nat <= RADIUS_TOL),
                "fid": fid, "legal_bad": legal_bad}


_ENGINE: Engine | None = None


def _engine() -> Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = Engine()
    return _ENGINE


CELL_FIELDS = ("pole", "seed", "pair", "steps", "n_ticks", "n_active", "n_tagged", "n_carrying", "n_dead", "n_in", "n_out",
               "tgt", "tgt_in", "tgt_out", "dir", "dir_in", "dir_out", "tgt_all", "dir_joint", "dir_any", "gohome_legal", "w_empty",
               "exact_ill", "exact_ill_in", "exact_ill_out", "sub_cost",
               "n_win", "n_win_in", "n_win_out", "nat", "nat_in", "nat_out", "itr", "itr_in", "itr_out", "n_win_free", "nat_free", "itr_free",
               "g6_tgt", "g6_dir", "g6_all",
               "fid_sum", "legal_bad", "rt_mismatch")


def analyze_cell(cell: dict) -> dict[str, Any]:
    eng = _engine()
    meta = cell["meta"]
    row = {k: 0 for k in CELL_FIELDS}
    row.update(pole=meta["pole"], seed=meta["seed"], pair="-".join(map(str, meta["forced"])), steps=meta["steps"])
    row["rt_mismatch"] = int(any(abs(meta["rt"].get(k, float("nan")) - v) > 0 or k not in meta["rt"] for k, v in RT_DEFAULTS.items())
                             or abs(meta["dt"] - eng.core.dt) > 1e-9 or not np.allclose(meta["home"], eng.home))
    for a in range(2):
        st = np.asarray(cell["state"][a], dtype=np.float64)
        lg = cell["legal"][a]
        T = len(st)
        alive, tagged, carrying = st[:, IX["alive"]] > 0.5, st[:, IX["tagged"]] > 0.5, st[:, IX["carrying"]] > 0.5
        active = alive & ~tagged & ~carrying
        row["n_ticks"] += T
        row["n_active"] += int(active.sum())
        row["n_tagged"] += int((alive & tagged).sum())
        row["n_carrying"] += int((alive & carrying & ~tagged).sum())
        row["n_dead"] += int((~alive).sum())
        for t in np.flatnonzero(active):
            r = eng.l1_tick(st[t], int(lg[t]))
            b = "in" if r["inward"] else "out"
            row["n_" + b] += 1
            if r["w_empty"]:                       # GO_TO masked for an ACTIVE agent: an error state, never a default
                row["w_empty"] += 1
                row["legal_bad"] += 1
                continue
            tp, dp, ei = int(r["err"] <= RADIUS_TOL), int(r["dir_gate"]), int(r["exact_illegal"])
            row["tgt"] += tp; row["tgt_" + b] += tp
            row["dir"] += dp; row["dir_" + b] += dp
            row["exact_ill"] += ei; row["exact_ill_" + b] += ei
            row["tgt_all"] += int(r["err_all_macros"] <= RADIUS_TOL)
            row["dir_joint"] += int(r["dir_joint"]); row["dir_any"] += int(r["dir_any"])
            row["gohome_legal"] += int(r["gohome_legal"])
            row["sub_cost"] += r["err"] - r["uerr"]
            row["legal_bad"] += int(not r["legal_ok"])
        for t0 in range(0, T - HORIZON + 1, STRIDE):
            if not active[t0:t0 + HORIZON].all():
                continue
            if len(eng.cands(int(lg[t0]), motion_only=True)[0]) == 0:
                row["w_empty"] += 1
                row["legal_bad"] += 1
                continue
            w = eng.window(st, int(lg[t0]), t0)
            b = "in" if w["inward0"] else "out"
            npass, ipass = int(w["nat"] <= RADIUS_TOL), int(w["itr"] <= RADIUS_TOL)
            row["n_win"] += 1; row["n_win_" + b] += 1
            row["nat"] += npass; row["nat_" + b] += npass
            row["itr"] += ipass; row["itr_" + b] += ipass
            if bool((st[t0:t0 + HORIZON, IX["dmin"]] >= eng.reach).all()):     # no agent within shove reach for the whole window
                row["n_win_free"] += 1; row["nat_free"] += npass; row["itr_free"] += ipass
            row["g6_tgt"] += int(w["g6_tgt"]); row["g6_dir"] += int(w["g6_dir"]); row["g6_all"] += int(w["g6_all"])
            row["fid_sum"] += w["fid"]
            row["legal_bad"] += w["legal_bad"]
    return row


def _analyze_cell_star(cell: dict) -> dict[str, Any]:
    return analyze_cell(cell)


# ============================================================================================ derive gates from the CSV

def _rows_of(src) -> list[dict]:
    if isinstance(src, list):
        return src
    with Path(src).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def derive_gates(src) -> dict[str, Any]:
    """src: the per-cell CSV path (production) or a list of row dicts (contract self-test)."""
    rows = _rows_of(src)
    out: dict[str, Any] = {"gates": {}, "diag": {}, "status": {}}
    for pole in POLES:
        rp = [r for r in rows if r["pole"] == pole]
        col = lambda k: np.asarray([float(r[k]) for r in rp])  # noqa: E731
        for gate, num, den in (("TARGET", "tgt", "n_active"), ("DIRECTION", "dir", "n_active"),
                               ("TRAJ_NATIVE", "nat", "n_win"), ("TRAJ_INTR", "itr", "n_win")):
            rate, lo, hi = cluster_rate_ci(col(num), col(den))
            out["gates"].setdefault(gate, {})[pole] = {"rate": rate, "lcb95": lo, "ucb95": hi, "n_units": int(col(den).sum()),
                                                       "n_episodes": len(rp)}
            out["status"].setdefault(gate, {})[pole] = classify(lo, hi)
        ratio = lambda a, b: (float(col(a).sum() / col(b).sum()) if col(b).sum() > 0 else None)  # noqa: E731
        out["diag"][pole] = {
            "ticks": {k: int(col(k).sum()) for k in ("n_ticks", "n_active", "n_tagged", "n_carrying", "n_dead", "n_in", "n_out")},
            "target_rate_inward": ratio("tgt_in", "n_in"), "target_rate_outward": ratio("tgt_out", "n_out"),
            "direction_rate_inward": ratio("dir_in", "n_in"), "direction_rate_outward": ratio("dir_out", "n_out"),
            "direction_components": {"joint_target_satisfying_and_aimed": ratio("dir_joint", "n_active"), "any_legal_waypoint_aimed": ratio("dir_any", "n_active")},
            "size_of_the_semantic_macro_exclusion": {
                "target_rate_if_every_legal_macro_were_admitted": ratio("tgt_all", "n_active"),
                "active_ticks_on_which_GO_HOME_is_legal": int(col("gohome_legal").sum()),
                "gate_target_rate_witness_GO_TO_only": ratio("tgt", "n_active")},
            "exact_semantic_macro_masked_rate": ratio("exact_ill", "n_active"),
            "exact_semantic_macro_masked_inward": ratio("exact_ill_in", "n_in"),
            "exact_semantic_macro_masked_outward": ratio("exact_ill_out", "n_out"),
            "mean_error_paid_for_legal_substitution_cells": (float(col("sub_cost").sum() / col("n_active").sum()) if col("n_active").sum() else None),
            "windows": int(col("n_win").sum()), "windows_inward_start": int(col("n_win_in").sum()), "windows_outward_start": int(col("n_win_out").sum()),
            "traj_native_rate_inward_start": ratio("nat_in", "n_win_in"), "traj_native_rate_outward_start": ratio("nat_out", "n_win_out"),
            "traj_interruptible_rate_inward_start": ratio("itr_in", "n_win_in"), "traj_interruptible_rate_outward_start": ratio("itr_out", "n_win_out"),
            "interaction_free_windows_robustness": {
                "windows": int(col("n_win_free").sum()), "share_of_all_windows": ratio("n_win_free", "n_win"),
                "traj_native_rate": ratio("nat_free", "n_win_free"), "traj_interruptible_rate": ratio("itr_free", "n_win_free"),
                "definition": "no other live agent within the shove reach at the start of any of the 16 ticks, so the isolated rollout equals the live step (contract C9)"},
            "g6_comparator_native_commit": {"committed_target_le_2_5": ratio("g6_tgt", "n_win"), "first_cos_and_radial": ratio("g6_dir", "n_win"),
                                            "all_four_jointly": ratio("g6_all", "n_win")},
            "isolated_vs_live_position_rmse_mean": (float(col("fid_sum").sum() / col("n_win").sum()) if col("n_win").sum() else None),
            "legality_violations": int(col("legal_bad").sum()), "rt_mismatch_cells": int(col("rt_mismatch").sum()),
        }
    return out


def _rederive_check(src, derived: dict) -> list[str]:
    """Independent pure-python recomputation of the point rates and the CSV's internal invariants."""
    problems = []
    rows = _rows_of(src)
    for pole in POLES:
        for gate, num, den in (("TARGET", "tgt", "n_active"), ("DIRECTION", "dir", "n_active"),
                               ("TRAJ_NATIVE", "nat", "n_win"), ("TRAJ_INTR", "itr", "n_win")):
            n = d = 0
            for r in rows:
                if r["pole"] == pole:
                    n += int(r[num]); d += int(r[den])
            rate = n / d if d else float("nan")
            if abs(rate - derived["gates"][gate][pole]["rate"]) > 1e-12:
                problems.append(f"{gate}/{pole}: csv rate {rate} != derived {derived['gates'][gate][pole]['rate']}")
    for r in rows:
        if int(r["n_active"]) + int(r["n_tagged"]) + int(r["n_carrying"]) + int(r["n_dead"]) != int(r["n_ticks"]):
            problems.append(f"tick partition broken for ({r['pole']},{r['seed']})")
        if int(r["n_in"]) + int(r["n_out"]) != int(r["n_active"]):
            problems.append(f"branch partition broken for ({r['pole']},{r['seed']})")
        if int(r["tgt"]) > int(r["n_active"]) or int(r["nat"]) > int(r["n_win"]) or int(r["itr"]) > int(r["n_win"]):
            problems.append(f"pass count exceeds units for ({r['pole']},{r['seed']})")
    return problems


def analyze() -> int:
    from concurrent.futures import ProcessPoolExecutor
    if _load_json(CONTRACT_PATH).get("DECISION") != "CONTRACTS_PASS":
        raise SystemExit("REFUSING: contracts did not pass")
    if RESULT_PATH.exists():
        raise SystemExit(f"REFUSING: result already exists: {RESULT_PATH}")
    ok, msg = _sealed_inputs_ok()
    if not ok:
        raise SystemExit(f"REFUSING: {msg}")
    merged: dict[tuple, dict] = {}
    for path in sorted(SD.glob(SHARD_GLOB)):
        for key, cell in _load_partial(path).items():
            merged[key] = cell
    want = set(_cells())
    missing, extra = sorted(want - set(merged)), sorted(set(merged) - want)
    if missing or extra:
        raise SystemExit(f"ABORT: {len(missing)} cell(s) missing (e.g. {missing[:2]}), {len(extra)} unexpected. Run/resume the shards first.")
    sealed = _sealed_rows()
    bad = [k for k, c in merged.items() if not _parity_ok(c["meta"], sealed)]
    if bad:
        raise SystemExit(f"ABORT: {len(bad)} cell(s) do not reproduce their sealed row, e.g. {bad[:2]}")
    ordered = [merged[c] for c in _cells()]
    print(f"analyzing {len(ordered)} cells with 6 workers ...", flush=True)
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows = list(tqdm_iter(pool.map(_analyze_cell_star, ordered, chunksize=2), desc="repr-audit analyze", total=len(ordered), unit="cell"))
    with CELLS_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(CELL_FIELDS))
        w.writeheader()
        w.writerows(rows)
    derived = derive_gates(CELLS_PATH)
    problems = _rederive_check(CELLS_PATH, derived)
    invalid = bool(problems) or any(derived["diag"][p]["legality_violations"] for p in POLES) or any(derived["diag"][p]["rt_mismatch_cells"] for p in POLES)
    label = terminal_label(derived["status"], invalid=invalid)
    payload = {
        "record_id": f"{STEM}_RESULT", "implements": SPEC_PATH.name, "utc": _now(),
        "TERMINAL_LABEL": label,
        "gate_status": derived["status"], "gates": derived["gates"], "diagnostics_non_gating": derived["diag"],
        "audit": {"rederivation_problems": problems, "csv_rows": len(rows), "csv_sha256": _sha(CELLS_PATH)},
        "read_only_reuse_of_spent_block": {"block": "21100001-128", "class": "sealed_confirmatory", "status_unchanged": True,
                                            "outcome_inference_drawn": False},
        "provenance": {"spec_sha256": _sha(SPEC_PATH), "script_sha256": _sha(Path(__file__)),
                       "sealed_inputs": _load_json(SPEC_PATH)["INTEGRITY_FROZEN"]["sealed_inputs_sha256"]},
        "guard": GUARD, "claim_boundary": _load_json(SPEC_PATH)["claim_boundary"],
        "no_label_authorizes_training": True,
    }
    RESULT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"TERMINAL_LABEL": label, "gate_status": derived["status"],
                      "rates": {g: {p: round(derived["gates"][g][p]["rate"], 4) for p in POLES} for g in derived["gates"]}}, indent=2))
    return 0 if not invalid else 2


# ============================================================================================ contracts

def contracts() -> dict:
    import torch
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    spec = _load_json(SPEC_PATH)
    add("C0_SPEC_FROZEN", spec.get("status") == "FROZEN_BEFORE_ANY_ARM_CONDITIONED_TELEMETRY_IS_READ", f"spec status = {spec.get('status')!r}")
    ok, msg = _sealed_inputs_ok()
    add("C1_SEALED_INPUTS_EQUAL_THEIR_PINS", ok, msg)
    ck, pin = _pi_a_path_and_pin()
    add("C2_PI_A_CHECKPOINT_EQUALS_ITS_PIN", ck.is_file() and _sha(ck) == pin, f"{ck.name} sha256 {'==' if ck.is_file() and _sha(ck) == pin else '!='} pin")
    reg_before = json.dumps(next(b for b in json.loads((ROOT / "artifacts" / "SEED_REGISTRY.json").read_text(encoding="utf-8"))["blocks"]
                                 if b["experiment_id"] == S.EXP_ID), sort_keys=True)
    setup = S._setup_poles()
    add("C3_BOTH_POLES_ATTESTED_AND_POLE_B_IS_THE_CERTIFIED_B3_3_GENOME", setup["pole_b"] is not None and set(setup["att"]) == set(POLES),
        f"attested {sorted(setup['att'])}; Pole B genome file {S.B33_GENOME.name}")
    from experiments import run_pyquaticus_port_contracts as G6
    g6 = {"HORIZON": G6.HORIZON, "DIRECTION_COSINE_MIN": G6.DIRECTION_COSINE_MIN, "INTERACTION_RADIUS_CELLS": G6.INTERACTION_RADIUS_CELLS}
    th = spec["GATES"]["thresholds_inherited_unchanged"]
    add("C4_G6_THRESHOLDS_EQUAL_THIS_SPECS", g6 == {"HORIZON": HORIZON, "DIRECTION_COSINE_MIN": COS_MIN, "INTERACTION_RADIUS_CELLS": RADIUS_TOL}
        and th["horizon_decision_ticks"] == HORIZON and th["first_direction_cosine"] == COS_MIN and th["target_error_cells"] == RADIUS_TOL == th["trajectory_rmse_cells"],
        f"G6 module {g6}")

    eng = Engine()
    core, T = eng.core, torch
    rng = np.random.default_rng(11)
    # C5: vector resolvers vs the LIVE _build_targets_from_action
    worst_d = worst_n = 0.0
    from macro_actions import MacroAction
    for _ in range(600):
        x, y = rng.uniform(0, 19, 2)
        h = rng.uniform(-np.pi, np.pi)
        F = (2.0, 10.0) if rng.random() < 0.5 else tuple(rng.uniform(0.5, 18.5, 2))
        E = tuple(rng.uniform(0.5, 18.5, 2))
        f32 = lambda a: T.tensor(a, dtype=T.float32)  # noqa: E731
        core.blue_x[0, 0], core.blue_y[0, 0], core.blue_heading[0, 0] = float(x), float(y), float(h)
        core.blue_carrying[0, 0] = False
        core.blue_flag_pos[0] = f32(list(F))
        core.red_flag_pos[0] = f32(list(E))
        macros = T.zeros((1, N), dtype=T.int64)
        targs = T.zeros((1, N), dtype=T.int64)
        macros[0, 0] = int(MacroAction.DEFEND)
        lx, ly = core._build_targets_from_action(macros, targs, side="blue")
        vx, vy, _ = eng.defend(float(core.blue_x[0, 0]), float(core.blue_y[0, 0]), float(core.blue_heading[0, 0]), (float(core.blue_flag_pos[0, 0]), float(core.blue_flag_pos[0, 1])))
        worst_d = max(worst_d, abs(float(lx[0, 0]) - vx), abs(float(ly[0, 0]) - vy))
        m, j = int(rng.integers(0, 5)), int(rng.integers(0, 50))
        macros[0, 0], targs[0, 0] = m, j
        lx, ly = core._build_targets_from_action(macros, targs, side="blue")
        nx, ny = eng.resolve_native(np.asarray([m]), np.asarray([j]), (float(core.red_flag_pos[0, 0]), float(core.red_flag_pos[0, 1])), eng.home)
        worst_n = max(worst_n, abs(float(lx[0, 0]) - float(nx[0])), abs(float(ly[0, 0]) - float(ny[0])))
    add("C5_VECTOR_RESOLVERS_EQUAL_THE_LIVE_RESOLVER", worst_d <= 1e-5 and worst_n <= 1e-5, f"600 random states: DEFEND max|diff| {worst_d:.2e}, native max|diff| {worst_n:.2e}")
    # C6: candidate-vectorized integration vs per-candidate
    K = 51
    a = [rng.uniform(0, 19, K), rng.uniform(0, 19, K), rng.uniform(-3, 3, K), rng.uniform(0, 2.2, K), rng.uniform(0, 19, K), rng.uniform(0, 19, K)]
    f = lambda arr: T.tensor(arr[None], dtype=T.float32)  # noqa: E731
    vx, vy, vh, vv = eng.step_t(*(f(z) for z in a))
    loop = np.asarray([eng.step_one(*(float(z[i]) for z in a)) for i in range(K)])
    d6 = float(np.max(np.abs(np.stack([vx[0].numpy(), vy[0].numpy(), vh[0].numpy(), vv[0].numpy()], 1) - loop)))
    add("C6_CANDIDATE_VECTORIZED_INTEGRATION_EQUALS_PER_CANDIDATE", d6 <= 1e-6, f"K={K}: max|diff| {d6:.2e}")
    # C7: mask facts
    obs = eng.env.reset()
    nM, nT = eng.nM, eng.nT
    m_obs = np.asarray(obs["mask"]).reshape(1, N, nM + nT)
    m_core = core._build_action_mask(side="blue").detach().cpu().numpy().reshape(1, N, nM + nT)
    nopay = m_obs[0, 0, :nM].astype(int).tolist()
    core.blue_commit_macro[0, 0], core.blue_commit_target[0, 0], core.blue_commit_ticks_left[0, 0] = 2, 7, 3
    saved = (core.blue_commit_macro.clone(), core.blue_commit_target.clone(), core.blue_commit_ticks_left.clone())
    real = core._build_action_mask(side="blue").detach().cpu().numpy().reshape(1, N, nM + nT)[0, 0]
    payload = _payload_only_mask(core).detach().cpu().numpy().reshape(1, N, nM + nT)[0, 0]
    restored = all(bool(T.equal(a_, b_)) for a_, b_ in zip(saved, (core.blue_commit_macro, core.blue_commit_target, core.blue_commit_ticks_left)))
    add("C7_MASK_FACTS", bool(np.array_equal(m_obs, m_core)) and nopay == [1, 0, 1, 0, 0] and restored
        and real[:nM].astype(int).tolist() == [0, 0, 1, 0, 0] and payload[:nM].astype(int).tolist() == [1, 0, 1, 0, 0],
        f"obs mask == core mask; no-payload macros {nopay} (GO_HOME masked); commit lock one-hot {real[:nM].astype(int).tolist()} vs payload-only {payload[:nM].astype(int).tolist()}; commit state restored={restored}")
    # C11: structural pins
    d_own = np.hypot(eng.wp[:, 0] - eng.home[0], eng.wp[:, 1] - eng.home[1])
    j5 = int(np.argmin(d_own))
    add("C11_STRUCTURAL_PINS_REPRODUCED", j5 == 5 and abs(float(d_own[j5]) - 2.076) < 1e-3 and int((d_own <= 2.5).sum()) == 1
        and eng.commit == {0: 4, 1: 3, 2: 4, 3: 2, 4: 4} and abs(eng.radius - 3.5) < 1e-9,
        f"nearest waypoint to own flag idx {j5} at {d_own[j5]:.3f}; {int((d_own <= 2.5).sum())} within 2.5; commit {eng.commit}; R {eng.radius}")
    # C8 / C9: replay parity + one-step fidelity on four sealed cells
    ck_path, _ = _pi_a_path_and_pin()
    pols = S._load_policies(setup["R2"], {"pi_A": ck_path}, DEVICE)
    before = L._param_digest(pols["pi_A"])
    sealed = _sealed_rows()
    probe_cells = [("A", SEEDS[0]), ("B", SEEDS[1]), ("A", SEEDS[2]), ("B", SEEDS[3])]
    cells = {}
    with torch.no_grad():
        for pole, seed in probe_cells:
            cells[(pole, seed)] = replay_cell(setup, pols["pi_A"], pole, seed, DEVICE)
    par = {k: _parity_ok(c["meta"], sealed) for k, c in cells.items()}
    finite = all(np.isfinite(np.asarray(c["state"])).all() and len(c["state"][0]) == c["meta"]["steps"] for c in cells.values())
    rt_ok = all(all(abs(c["meta"]["rt"].get(k, 9e9) - v) == 0 for k, v in RT_DEFAULTS.items()) for c in cells.values())
    add("C8_REPLAY_REPRODUCES_SEALED_ROWS_EXACTLY", all(par.values()) and finite and rt_ok and before == L._param_digest(pols["pi_A"]),
        f"{sum(par.values())}/{len(par)} cells equal their sealed (steps, blue, red); arrays finite & length==steps: {finite}; motion params at defaults: {rt_ok}; parameter digest unchanged: {before == L._param_digest(pols['pi_A'])}")
    n_ok = n_all = 0
    n_tight = 0
    bad: list[tuple[float, bool, bool, float, float]] = []  # (error, inward, status changed at t+1, |dist-to-ring|, neighbour distance)
    n_in_reach = 0
    witness_empty = witness_nonmotion = witness_n = 0
    for c in cells.values():
        for a_ in range(2):
            st = np.asarray(c["state"][a_], dtype=np.float64)
            lg_ = c["legal"][a_]
            act = (st[:, IX["alive"]] > 0.5) & (st[:, IX["tagged"]] < 0.5) & (st[:, IX["carrying"]] < 0.5)
            for t in np.flatnonzero(act):
                wm_, wi_ = eng.cands(int(lg_[t]), motion_only=True)
                witness_empty += int(len(wm_) == 0)
                witness_nonmotion += int(bool((wm_ != GO_TO).any()))
                witness_n = max(witness_n, len(wm_))
            for t in np.flatnonzero(act[:-1]):
                s = st[t]
                tx, ty, inw = eng.defend(s[IX["x"]], s[IX["y"]], s[IX["h"]], (s[IX["Fx"]], s[IX["Fy"]]))
                nx, ny, _, _ = eng.step_one(s[IX["x"]], s[IX["y"]], s[IX["h"]], s[IX["v"]], tx, ty)
                e = float(np.hypot(nx - st[t + 1, IX["x"]], ny - st[t + 1, IX["y"]]))
                n_all += 1
                n_ok += int(e <= ONE_STEP_TOL)
                n_tight += int(e <= 1e-5)
                if s[IX["dmin"]] < eng.reach:
                    n_in_reach += 1
                if e > ONE_STEP_TOL:
                    changed = bool((st[t + 1, [IX["alive"], IX["tagged"], IX["carrying"]]] != st[t, [IX["alive"], IX["tagged"], IX["carrying"]]]).any())
                    bad.append((e, bool(inw), changed, abs(float(np.hypot(s[IX["x"]] - s[IX["Fx"]], s[IX["y"]] - s[IX["Fy"]])) - eng.radius), float(s[IX["dmin"]])))
    frac = n_ok / max(n_all, 1)
    b_err = np.asarray([b[0] for b in bad]) if bad else np.zeros(1)
    unexplained = sum(1 for b in bad if b[4] >= eng.reach)
    add("C9_ISOLATED_ONE_STEP_INTEGRATOR_REPRODUCES_LIVE_MOTION_AND_EVERY_MISS_HAS_A_NEIGHBOUR_IN_SHOVE_REACH", n_all > 0 and frac >= ONE_STEP_MIN_FRAC and unexplained == 0,
        f"{n_ok}/{n_all} ACTIVE ticks within {ONE_STEP_TOL} cells ({frac:.4f}; {n_tight} within 1e-5); threshold {ONE_STEP_MIN_FRAC}. "
        f"Non-matching ticks ({len(bad)}): status changed at t+1 in {sum(b[2] for b in bad)}; inward-branch {sum(b[1] for b in bad)} / outward {sum(not b[1] for b in bad)}; "
        f"within 1.5 cells of the ring {sum(b[3] <= 1.5 for b in bad)}; error median {float(np.median(b_err)):.3f} max {float(b_err.max()):.3f}. "
        f"Mechanism (live step ends with a {0.5}-cell tangential shove between agents closer than avoid_collision_radius={eng.avoid_r}): "
        f"shove reach = {eng.reach:.3f} cells; ticks with a live neighbour inside reach {n_in_reach}/{n_all}; non-matching ticks with NO neighbour inside reach: {unexplained} (must be 0)")
    # C14: vectorized cosine == the unchanged direction_cosine; DIRECTION is EXISTENCE over the witness set
    from gpu_env.pyquaticus_port import direction_cosine
    worst_c = 0.0
    for _ in range(400):
        p_, r_v, t_v = rng.uniform(0, 19, 2), rng.uniform(0, 19, 2), rng.uniform(0, 19, 2)
        mode = int(rng.integers(0, 4))
        if mode in (1, 3):
            r_v = p_.copy()
        if mode in (2, 3):
            t_v = p_.copy()
        va = float(Engine.cos_vec(p_[0], p_[1], r_v[0], r_v[1], t_v[0], t_v[1]))
        vb = direction_cosine(T.tensor(p_, dtype=T.float64), T.tensor(r_v, dtype=T.float64), T.tensor(t_v, dtype=T.float64))
        worst_c = max(worst_c, abs(va - vb))
    e1, c1 = np.array([1.0, 2.0]), np.array([0.5, 0.995])
    nearest_only = bool(c1[int(np.argmin(e1))] >= COS_MIN)
    dg = [direction_gate(e1, c1), direction_gate(np.array([3.0, 4.0]), np.array([0.995, 0.1])),
          direction_gate(np.array([3.0, 4.0]), np.array([0.5, 0.1])), direction_gate(np.array([1.0, 3.0]), np.array([0.5, 0.995]))]
    dg_ok = dg == [(True, True, True), (True, False, True), (False, False, False), (False, False, True)] and nearest_only is False
    add("C14_COSINE_REPLICA_EQUALS_THE_UNCHANGED_FUNCTION_AND_DIRECTION_IS_EXISTENCE", worst_c <= 1e-12 and dg_ok,
        f"400 random incl. zero-vector cases: max|diff| {worst_c:.2e}; existence self-tests {dg}; a nearest-only rule would have said {nearest_only} on the case where existence says True")
    add("C15_WITNESS_SET_IS_GO_TO_ONLY_AND_NEVER_EMPTY_ON_ACTIVE_TICKS", witness_empty == 0 and witness_nonmotion == 0 and witness_n > 0,
        f"across all ACTIVE ticks of the 4 probe cells: empty witness sets {witness_empty}, non-GO_TO members {witness_nonmotion}, largest witness set {witness_n} waypoints")
    # C16: planted-truth tests of the path-oracle + end-to-end pipeline invariants (gate rates are NOT printed or recorded)
    c0 = next(iter(cells.values()))
    st0 = np.asarray(c0["state"][0], dtype=np.float64)[:HORIZON]
    bits0 = int(c0["legal"][0][0])

    def planted(seq: list[int]) -> np.ndarray:
        x, y, h, v = st0[0, IX["x"]], st0[0, IX["y"]], st0[0, IX["h"]], st0[0, IX["v"]]
        out_ = np.zeros((HORIZON, 2))
        for k in range(HORIZON):
            x, y, h, v = eng.step_one(x, y, h, v, eng.wp[seq[k], 0], eng.wp[seq[k], 1])
            out_[k] = (x, y)
        return out_

    rm = lambda p, r: float(np.sqrt(np.mean(np.sum((p - r) ** 2, axis=1))))  # noqa: E731
    ref1 = planted([5] * HORIZON)
    ref2 = planted([5 if k % 2 == 0 else 45 for k in range(HORIZON)])
    p1_native = rm(eng.path_oracle(st0, bits0, ref1, interruptible=False)[0], ref1)
    p2_intr = rm(eng.path_oracle(st0, bits0, ref2, interruptible=True)[0], ref2)
    p2_native = rm(eng.path_oracle(st0, bits0, ref2, interruptible=False)[0], ref2)
    planted_ok = p1_native <= 1e-6 and p2_intr <= 1e-6 and p2_native > 1e-3
    globals()["_ENGINE"] = eng
    rows_a = [analyze_cell(c) for c in cells.values()]
    rows_b = [analyze_cell(c) for c in cells.values()]
    inv = []
    for r in rows_a:
        inv += [r["n_active"] + r["n_tagged"] + r["n_carrying"] + r["n_dead"] == r["n_ticks"], r["n_in"] + r["n_out"] == r["n_active"],
                r["tgt"] <= r["n_active"], r["dir"] <= r["n_active"], r["nat"] <= r["n_win"], r["itr"] <= r["n_win"],
                r["n_win_in"] + r["n_win_out"] == r["n_win"], r["legal_bad"] == 0, r["w_empty"] == 0, r["rt_mismatch"] == 0,
                all(np.isfinite(float(r[k])) for k in CELL_FIELDS if k not in ("pole", "pair"))]
    deterministic = rows_a == rows_b
    dg_pipe = derive_gates(rows_a)
    pipe_ok = (all(inv) and deterministic and sum(r["n_win"] for r in rows_a) > 0 and sum(r["n_active"] for r in rows_a) > 0
               and not _rederive_check(rows_a, dg_pipe) and all(dg_pipe["status"][g][p] in ("PASS", "FAIL", "BORDERLINE") for g in dg_pipe["status"] for p in POLES))
    add("C16_PLANTED_TRUTH_AND_PIPELINE_INVARIANTS", planted_ok and pipe_ok,
        f"path-oracle: planted GO_TO path recovered natively RMSE {p1_native:.2e}; planted per-tick alternating path recovered by the interruptible variant RMSE {p2_intr:.2e} "
        f"but not natively (RMSE {p2_native:.3f} > 1e-3, i.e. the commit lock binds); pipeline on the 4 probe cells: {sum(bool(i) for i in inv)}/{len(inv)} invariants hold, "
        f"deterministic={deterministic}, re-derivation clean. Gate rates deliberately not printed or recorded.")
    # C10: decision-rule self-tests
    P_, F_, B_ = "PASS", "FAIL", "BORDERLINE"
    def g(t, d, n, i):  # noqa: E306
        return {"TARGET": dict.fromkeys(POLES, t), "DIRECTION": dict.fromkeys(POLES, d), "TRAJ_NATIVE": dict.fromkeys(POLES, n), "TRAJ_INTR": dict.fromkeys(POLES, i)}
    cases = [(g(P_, P_, P_, P_), False, "HIGH_COVERAGE_REPRESENTABLE_NATIVE"), (g(F_, P_, P_, P_), False, "NOT_REPRESENTABLE_VOCABULARY"),
             (g(P_, F_, F_, P_), False, "NOT_REPRESENTABLE_VOCABULARY"), (g(P_, P_, F_, P_), False, "NOT_REPRESENTABLE_COMMITMENT_BOUND"),
             (g(P_, P_, F_, F_), False, "NOT_REPRESENTABLE_OTHER"), (g(B_, P_, P_, P_), False, "INCONCLUSIVE_BORDERLINE"),
             (g(P_, P_, B_, P_), False, "INCONCLUSIVE_BORDERLINE"), (g(P_, P_, F_, B_), False, "INCONCLUSIVE_BORDERLINE"),
             (g(P_, P_, P_, P_), True, "AUDIT_INVALID")]
    lab_ok = all(terminal_label(gg, inv) == want for gg, inv, want in cases)
    cov_ok = classify(0.95, 0.99) == "PASS" and classify(0.5, 0.8) == "FAIL" and classify(0.85, 0.95) == "BORDERLINE" and classify(0.90, 1.0) == "PASS"
    r_, lo_, hi_ = cluster_rate_ci([10] * 8, [10] * 8, n_boot=500)
    add("C10_DECISION_RULE_SELFTESTS", lab_ok and cov_ok and (r_, lo_, hi_) == (1.0, 1.0, 1.0), f"{len(cases)}/{len(cases)} label cases, coverage classes, cluster-CI on all-pass = {(r_, lo_, hi_)}")
    # C12: read-only fence
    src = Path(__file__).read_text(encoding="utf-8")
    banned = [".le" + "arn(", "optimizer" + ".step", ".back" + "ward(", "PP" + "O(", "zero_" + "grad"]
    hits = [b for b in banned if b in src]
    add("C12_READ_ONLY_FENCE", not hits, "no training/optimizer call in the audit source" if not hits else f"found {hits}")
    reg_after = json.dumps(next(b for b in json.loads((ROOT / "artifacts" / "SEED_REGISTRY.json").read_text(encoding="utf-8"))["blocks"]
                                if b["experiment_id"] == S.EXP_ID), sort_keys=True)
    blk = json.loads(reg_after)
    add("C13_SEED_REGISTRY_ENTRY_UNCHANGED", reg_before == reg_after and blk["status"] == "SPENT" and blk["seed_class"] == "sealed_confirmatory",
        f"{S.EXP_ID}: {blk['status']} / {blk['seed_class']}; entry byte-identical before/after")
    decision = "CONTRACTS_PASS" if all(c["pass"] for c in checks) else "CONTRACTS_FAIL"
    result = {"record_id": f"{STEM}_CONTRACT_RESULT", "implements": SPEC_PATH.name, "utc": _now(), "DECISION": decision,
              "spec_sha256": _sha(SPEC_PATH), "script_sha256": _sha(Path(__file__)), "checks": checks}
    CONTRACT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\n  AUDIT CONTRACTS: {decision}  ({sum(not c['pass'] for c in checks)}/{len(checks)} failed)", flush=True)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("contracts", "replay", "analyze"), required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1, dest="n_shards")
    a = ap.parse_args()
    if a.stage == "contracts":
        return 0 if contracts()["DECISION"] == "CONTRACTS_PASS" else 2
    if a.stage == "replay":
        if not 0 <= a.shard < a.n_shards:
            raise SystemExit(f"--shard must be in [0, {a.n_shards})")
        return run_shard(a.shard, a.n_shards)
    return analyze()


if __name__ == "__main__":
    raise SystemExit(main())
