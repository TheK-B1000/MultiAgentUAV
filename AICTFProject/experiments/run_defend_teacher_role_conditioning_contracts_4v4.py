"""DEFEND_TEACHER_ROLE_CONDITIONING_A_V1_SPEC.json.

Contracts-only runner for CONTRACTS_before_training (C0-C10). Spends no
seed, writes no checkpoint (C10). Training and evaluation reuse the
existing generic launchers (experiments/train_specialist_scale.py,
experiments/eval_specialist_crossover_scaled.py) once these contracts pass
and the training/evaluation seeds are reserved via seed_registry -- this
script does not itself launch either.

  contracts   The only stage this script implements: C0-C10 below.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

STEM = "DEFEND_TEACHER_ROLE_CONDITIONING_A_V1"
SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / f"{STEM}_SPEC.json"
CONTRACT_PATH = SD / f"{STEM}_CONTRACT_RESULT.json"
A_FREEZE_RULE_PATH = SD / "A_FREEZE_RULE.json"

PI_A_CKPT = ROOT / (
    "artifacts/scale_4v4_specialists/pi_A_specialist_4v4_b3_entity_repair/"
    "ckpts/final_pi_A_specialist_4v4_b3_entity_repair.zip"
)
PI_A_SHA256 = "94dde69d091a79344db3390d5464dbb4bcf51677a175df93969ab252b2e0f478"

PROBE_SCRIPT = ROOT / "experiments" / "probe_defend_teacher_role_warmstart_contract.py"
PROBE_RESULT_PATH = SD / "DEFEND_TEACHER_ROLE_WARMSTART_CONTRACT_PROBE_A.json"

N_CH, ROWS, COLS, VEC_DIM = 7, 20, 20, 20


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _spaces(n: int = 4):
    from gymnasium import spaces

    obs = spaces.Dict({
        "grid": spaces.Box(0.0, 1.0, shape=(n, N_CH, ROWS, COLS), dtype="float32"),
        "vec": spaces.Box(-1.0, 1.0, shape=(n, VEC_DIM), dtype="float32"),
        "agent_mask": spaces.Box(0.0, 1.0, shape=(n,), dtype="float32"),
        "mask": spaces.Box(0.0, 1.0, shape=(n * (5 + 50),), dtype="float32"),
    })
    act = spaces.MultiDiscrete([5, 50] * n)
    return obs, act


def _build_role_model():
    import torch

    from rl.custom_ppo.policy import SharedActorCentralizedCritic

    obs_s, act_s = _spaces()
    m = SharedActorCentralizedCritic(
        obs_s, act_s, strategy_encoder_enabled=False, latent_k=0, role_conditioning_enabled=True,
    )
    opt = torch.optim.Adam(m.parameters(), lr=1e-4)
    return m, opt


def run_contracts() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)

    # ---- C0: spec frozen -----------------------------------------------
    spec = _load_json(SPEC_PATH)
    add(
        "C0_SPEC_FROZEN",
        spec.get("status") == "FROZEN_BEFORE_ANY_TRAINING",
        f"spec status = {spec.get('status')!r}",
    )

    # ---- C1: pi_A checkpoint pin + A_FREEZE_RULE supersession record ---
    ckpt_ok = PI_A_CKPT.is_file()
    ckpt_sha = _sha256(PI_A_CKPT) if ckpt_ok else ""
    add(
        "C1a_PI_A_CHECKPOINT_EQUALS_PIN",
        ckpt_ok and ckpt_sha == PI_A_SHA256,
        f"exists={ckpt_ok} sha256_matches_pin={ckpt_sha == PI_A_SHA256}",
    )
    supersedes = spec.get("SUPERSEDES", {})
    supersedes_ok = (
        supersedes.get("record") == "A_FREEZE_RULE.json"
        and bool(supersedes.get("exception_fired"))
        and A_FREEZE_RULE_PATH.is_file()
    )
    add(
        "C1b_A_FREEZE_RULE_SUPERSESSION_RECORD_PRESENT_AND_DATED",
        supersedes_ok,
        f"SUPERSEDES.record={supersedes.get('record')!r} "
        f"A_FREEZE_RULE.json_exists={A_FREEZE_RULE_PATH.is_file()} "
        f"exception_fired_present={bool(supersedes.get('exception_fired'))}",
    )

    # ---- C2: fixed_for_episode role hold semantics ----------------------
    import torch

    from rl.custom_ppo.rule_role_assignment import RoleHoldState

    def _home(B=1, x=0.0, y=0.0):
        return torch.tensor([[x, y]] * B, dtype=torch.float32)

    home = _home(1)
    pos_x = torch.tensor([[0.0, 1.0, 5.0, 6.0]])
    pos_y = torch.zeros(1, 4)
    alive = torch.ones(1, 4, dtype=torch.bool)

    hold = RoleHoldState(1, 4, hold_ticks=8, fixed_for_episode=True, device="cpu")
    r0 = hold.update(pos_x, pos_y, home, alive)
    alive_dead = torch.tensor([[False, True, True, True]])
    r_death = hold.update(pos_x, pos_y, home, alive_dead)
    death_ignored = torch.equal(r_death, r0)

    hold.reset_envs(torch.tensor([True]))
    pos_x_flip = torch.tensor([[6.0, 5.0, 1.0, 0.0]])
    r_reset = hold.update(pos_x_flip, pos_y, home, alive)
    reset_triggers_once = not torch.equal(r_reset, r0)
    r_again = hold.update(pos_x_flip, pos_y, home, alive)
    no_repeat = torch.equal(r_again, r_reset)

    hold_ticks_inert = True
    for ht in (1, 8, 1000):
        h2 = RoleHoldState(1, 4, hold_ticks=ht, fixed_for_episode=True, device="cpu")
        rr0 = h2.update(pos_x, pos_y, home, alive)
        for _ in range(20):
            rr = h2.update(pos_x, pos_y, home, alive)
            hold_ticks_inert = hold_ticks_inert and torch.equal(rr, rr0)

    force_ignored_hold = RoleHoldState(1, 4, hold_ticks=8, fixed_for_episode=True, device="cpu")
    fr0 = force_ignored_hold.update(pos_x, pos_y, home, alive, force=True)
    fr1 = force_ignored_hold.update(pos_x_flip, pos_y, home, alive, force=True)
    force_kwarg_ignored = torch.equal(fr1, fr0)

    add(
        "C2_FIXED_FOR_EPISODE_HOLD_SEMANTICS",
        death_ignored and reset_triggers_once and no_repeat and hold_ticks_inert and force_kwarg_ignored,
        f"death_ignored={death_ignored} reset_triggers_exactly_once={reset_triggers_once} "
        f"no_repeat_until_next_reset={no_repeat} role_hold_ticks_provably_inert={hold_ticks_inert} "
        f"force_kwarg_ignored={force_kwarg_ignored}",
    )

    # ---- C3: teacher parity against the sealed N' controller ------------
    import numpy as np

    import experiments.audit_scaffold_to_native_representability_4v4 as A
    import experiments.run_goto_only_defend_substitution_4v4 as Nmod
    from rl.custom_ppo.defend_teacher import compute_defend_teacher_waypoints

    eng = A.Engine()
    core = eng.core
    n_agents = int(core.blue_x.shape[1])

    def _packed_bits(agent_idx: int) -> int:
        nM, nT = int(core.cfg.n_macros), int(core.cfg.n_targets)
        mask = core._build_action_mask(side="blue").view(1, -1, nM + nT)
        row = mask[0, agent_idx].detach().cpu().numpy() > 0
        return int(sum(int(row[i]) << i for i in range(len(row))))

    def _row_for(agent_idx: int) -> "np.ndarray":
        row = np.zeros(len(A.STATE_FIELDS))
        row[A.IX["x"]] = float(core.blue_x[0, agent_idx])
        row[A.IX["y"]] = float(core.blue_y[0, agent_idx])
        row[A.IX["h"]] = float(core.blue_heading[0, agent_idx])
        row[A.IX["v"]] = float(core.blue_speed[0, agent_idx])
        row[A.IX["Fx"]] = float(core.blue_flag_pos[0, 0])
        row[A.IX["Fy"]] = float(core.blue_flag_pos[0, 1])
        return row

    n_checked, n_match = 0, 0
    for tick in range(3):
        batched = compute_defend_teacher_waypoints(core)
        for agent_idx in range(n_agents):
            bits = _packed_bits(agent_idx)
            row = _row_for(agent_idx)
            ref_w = Nmod.controller_select(eng, row, bits)
            got_w = int(batched[0, agent_idx].item())
            n_checked += 1
            n_match += int(got_w == ref_w)
        nT = int(core.cfg.n_targets)
        actions = np.zeros((1, n_agents * 2), dtype=np.int64)
        for a in range(n_agents):
            actions[0, 2 * a] = 0
            actions[0, 2 * a + 1] = (tick * 11 + a * 7) % nT
        eng.env.step_async(actions)
        eng.env.step_wait()
    add(
        "C3_TEACHER_PARITY_BIT_IDENTICAL_TO_SEALED_N_PRIME",
        n_checked > 0 and n_match == n_checked,
        f"{n_match}/{n_checked} (agent,tick) samples bit-identical to the sealed controller_select",
    )

    # ---- C4/C9 shared setup: a synthetic role-conditioned model ---------
    from rl.custom_ppo.defend_teacher import DefendTeacherRunner, defend_teacher_loss

    m, opt = _build_role_model()
    rng = np.random.default_rng(0)
    n, batch = 4, 2

    def _rand_obs():
        return {
            "grid": torch.tensor(rng.random((batch, n, N_CH, ROWS, COLS)), dtype=torch.float32),
            "vec": torch.tensor(rng.uniform(-1, 1, (batch, n, VEC_DIM)), dtype=torch.float32),
            "agent_mask": torch.ones(batch, n),
            "mask": torch.ones(batch, n * 55),
        }

    obs = _rand_obs()
    obs["roles"] = torch.tensor([[0.0, 0.0, 1.0, 1.0]] * batch)
    m3 = obs["mask"].clone().view(batch, n, 55)
    m3[:, 1, :] = 0.0
    m3[:, 1, 1] = 1.0
    m3[:, 1, 5 + 3] = 1.0
    obs["mask"] = m3.view(batch, n * 55)
    waypoint_target = torch.from_numpy(rng.integers(0, 50, size=(batch, n))).long()
    loss, tel = defend_teacher_loss(m, obs, waypoint_target)
    gate_ok = int(tel["n_gated"]) == 1 * batch and loss.requires_grad and float(loss.detach()) > 0.0
    add(
        "C4_TEACHER_LOSS_GATING",
        gate_ok,
        f"n_gated={int(tel['n_gated'])} (expected {1 * batch}: only agent 0 is DEFEND+eligible+alive), "
        f"loss={float(loss.detach()):.4f}",
    )

    # ---- C5: lambda schedule boundary values + monotonicity -------------
    from rl.custom_ppo.schedules import resolve_defend_teacher_lambda

    sched_cfg = SimpleNamespace(
        defend_teacher_lambda=0.1, defend_teacher_lambda_end=0.0,
        defend_teacher_decay_start_step=50_000, defend_teacher_decay_end_step=150_000,
    )
    boundary = {0: 0.1, 49_999: 0.1, 50_000: 0.1, 100_000: 0.05, 150_000: 0.0, 200_000: 0.0}
    boundary_ok = all(
        abs(resolve_defend_teacher_lambda(sched_cfg, global_step=s) - v) < 1e-9
        for s, v in boundary.items()
    )
    v_149999 = resolve_defend_teacher_lambda(sched_cfg, global_step=149_999)
    just_above_zero_ok = 0.0 < v_149999 < 1e-3
    monotone_steps = list(range(50_000, 150_001, 5_000))
    monotone_vals = [resolve_defend_teacher_lambda(sched_cfg, global_step=s) for s in monotone_steps]
    monotone_ok = all(monotone_vals[i] >= monotone_vals[i + 1] - 1e-12 for i in range(len(monotone_vals) - 1))
    constant_before = all(
        resolve_defend_teacher_lambda(sched_cfg, global_step=s) == 0.1 for s in range(0, 50_000, 10_000)
    )
    constant_after = all(
        resolve_defend_teacher_lambda(sched_cfg, global_step=s) == 0.0 for s in range(150_000, 200_001, 10_000)
    )
    add(
        "C5_LAMBDA_SCHEDULE_EXACT_BOUNDARY_VALUES_AND_MONOTONIC",
        boundary_ok and just_above_zero_ok and monotone_ok and constant_before and constant_after,
        f"boundary_values_ok={boundary_ok} t149999_just_above_zero={just_above_zero_ok} "
        f"monotonic_on_decay_window={monotone_ok} constant_before={constant_before} constant_after={constant_after}",
    )

    # ---- C6/C7: warm-start t0 equivalence + fresh optimizer (real pi_A) -
    probe_script_ok = PROBE_SCRIPT.is_file()
    if PROBE_RESULT_PATH.is_file():
        probe_result = _load_json(PROBE_RESULT_PATH)
        probe_pass = bool(probe_result.get("pass"))
        probe_ckpt_ok = probe_result.get("parent_ckpt_sha256") == PI_A_SHA256
        probe_detail = (
            f"probe recorded pass={probe_pass} parent_ckpt_sha256_matches_pin={probe_ckpt_ok} "
            f"max_logit_diff={probe_result.get('max_logit_diff')} "
            f"argmax_diff_total={probe_result.get('argmax_diff_total')} "
            f"fresh_optimizer_survives_migration={probe_result.get('fresh_optimizer_survives_migration')}"
        )
        probe_ok = probe_pass and probe_ckpt_ok
    else:
        probe_ok = False
        probe_detail = (
            "probe result not found -- run "
            "experiments/probe_defend_teacher_role_warmstart_contract.py before spending the training seed"
        )
    add(
        "C6_C7_WARMSTART_T0_EQUIVALENCE_AND_FRESH_OPTIMIZER",
        probe_script_ok and probe_ok,
        f"probe_script_exists={probe_script_ok}; {probe_detail}",
    )

    # ---- C8: mutual exclusion guards -------------------------------------
    from rl.training.orchestrator import (
        _maybe_attach_defend_teacher,
        _maybe_attach_getflag_preservation,
        _maybe_attach_role_preservation,
        _maybe_attach_sibling_separation,
    )

    mutual_ok = True
    mutual_detail: list[str] = []
    for attr in (
        "sappo_anchor_runner", "exp2_teacher_compression_runner",
        "sibling_sep_runner", "role_pres_runner", "getflag_preserve_runner",
    ):
        cfg_mock = SimpleNamespace(defend_teacher_lambda=0.1, role_conditioning_enabled=True)
        trainer_mock = SimpleNamespace(**{attr: object()})
        try:
            _maybe_attach_defend_teacher(cfg_mock, trainer_mock)
            mutual_ok = False
            mutual_detail.append(f"_maybe_attach_defend_teacher missing guard against {attr}")
        except RuntimeError:
            pass

    reverse_cases = [
        (_maybe_attach_getflag_preservation, dict(
            getflag_preserve_lambda=0.1, getflag_preserve_ckpt="x",
            sibling_sep_lambda=0.0, role_pres_lambda=0.0, defend_teacher_lambda=0.1,
        )),
        (_maybe_attach_sibling_separation, dict(
            sibling_sep_lambda=0.1, sibling_sep_ckpt="x", sibling_sep_dataset="y",
            defend_teacher_lambda=0.1,
        )),
        (_maybe_attach_role_preservation, dict(
            role_pres_lambda=0.1, role_pres_targets="x", role_pres_style="GUARD",
            sibling_sep_lambda=0.0, getflag_preserve_lambda=0.0, defend_teacher_lambda=0.1,
        )),
    ]
    for fn, cfg_kwargs in reverse_cases:
        try:
            fn(SimpleNamespace(**cfg_kwargs), SimpleNamespace())
            mutual_ok = False
            mutual_detail.append(f"{fn.__name__} missing guard against defend_teacher_lambda")
        except RuntimeError:
            pass

    add(
        "C8_MUTUAL_EXCLUSION_GUARDS",
        mutual_ok,
        "; ".join(mutual_detail) if mutual_detail else "all 8 mutual-exclusion guards raise as expected",
    )

    # ---- C9: structurally absent ------------------------------------------
    ctor_rejects_zero = False
    try:
        DefendTeacherRunner(m, opt, lambda_teacher=0.0)
    except ValueError:
        ctor_rejects_zero = True

    ctor_requires_role_conditioning = False
    obs_s2, act_s2 = _spaces()
    from rl.custom_ppo.policy import SharedActorCentralizedCritic

    m_no_role = SharedActorCentralizedCritic(
        obs_s2, act_s2, strategy_encoder_enabled=False, latent_k=0, role_conditioning_enabled=False,
    )
    opt_no_role = torch.optim.Adam(m_no_role.parameters(), lr=1e-4)
    try:
        DefendTeacherRunner(m_no_role, opt_no_role, lambda_teacher=0.1)
    except ValueError:
        ctor_requires_role_conditioning = True

    runner = DefendTeacherRunner(m, opt, lambda_teacher=0.1, cadence=1)
    runner.lambda_teacher = 0.0
    batch_dict = {
        "obs_grid": obs["grid"],
        "obs_vec": obs["vec"],
        "obs_agent_mask": obs["agent_mask"],
        "obs_mask": torch.ones(batch, n * 55),
        "obs_roles": obs["roles"],
        "obs_defend_teacher_waypoint": waypoint_target,
    }
    params_before = [p.detach().clone() for p in m.parameters()]
    fired = runner.note_ppo_minibatch(batch_dict)
    params_unchanged = all(torch.equal(a, b) for a, b in zip(params_before, m.parameters()))
    add(
        "C9_STRUCTURALLY_ABSENT_AT_LAMBDA_LE_0",
        ctor_rejects_zero and ctor_requires_role_conditioning and (not fired) and params_unchanged,
        f"constructor_rejects_lambda<=0={ctor_rejects_zero} "
        f"constructor_requires_role_conditioning={ctor_requires_role_conditioning} "
        f"zero_lambda_step_fires={fired} params_unchanged_after_zero_lambda_step={params_unchanged}",
    )

    # ---- C10: contracts consume no training step, write no checkpoint ---
    src = Path(__file__).read_text(encoding="utf-8")
    forbidden = [".le" + "arn(", "optimizer" + ".step(", "torch.sa" + "ve(", "checkpoint_" + "dir"]
    hits = [t for t in forbidden if t in src]
    add(
        "C10_CONTRACTS_ARE_READ_ONLY",
        not hits,
        "no training/checkpoint-writing call in this script's own source"
        if not hits else f"found {hits}",
    )

    decision = "CONTRACTS_PASS" if all(c["pass"] for c in checks) else "CONTRACTS_FAIL"
    result = {
        "record_id": f"{STEM}_CONTRACT_RESULT",
        "implements": SPEC_PATH.name,
        "utc": _now(),
        "DECISION": decision,
        "spec_sha256": _sha256(SPEC_PATH),
        "script_sha256": _sha256(Path(__file__)),
        "checks": checks,
    }
    CONTRACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        f"\n  DEFEND_TEACHER_ROLE_CONDITIONING_A CONTRACTS: {decision}  "
        f"({sum(not c['pass'] for c in checks)}/{len(checks)} failed)",
        flush=True,
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("contracts",), default="contracts")
    ap.parse_args()
    result = run_contracts()
    return 0 if result["DECISION"] == "CONTRACTS_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
