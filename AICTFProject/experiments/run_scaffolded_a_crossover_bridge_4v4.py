"""SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_V1_SPEC.json.

CONFIRMATORY causal bridge on the FROZEN 4v4 learned specialists (no training, no parameter change):
does imposing 2A/2D on pi_A restore the A/B payoff separation the learned specialists failed to show?

Arms on both certified poles, the same 128 fresh paired seeds:
    pi_A        native pi_A3
    A_prime     pi_A3 + 2D: two of the four agents have their resolved target forced to DEFEND's
                live-state target (experiments.probe_learned_composition.install_forced_defend_target)
    pi_B        native corrected pi_B3
Primary gate: LCB95(WR(A',A) - WR(pi_B,A)) > 0  AND  LCB95(WR(pi_B,B) - WR(A',B)) > 0.
I_A, I_B (scaffold vs native pi_A) are mechanism diagnostics, never gates.

  contracts   Mechanism / attestation / parity / registry / decision-rule contracts. Spends no seed.
  run         One shard of the 768 episodes (resumable, append-only partial file per shard). Prints
              progress COUNTS only -- no outcomes -- per the frozen no-interim-look rule.
  analyze     Refuses to run unless all 768 cells are present; seals through the formal audit pipeline.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import experiments.probe_learned_composition as P  # noqa: E402  (mechanism + trace snapshot)
import experiments.run_learned_composition_probe as L  # noqa: E402  (param digest, sha helper)
from experiments.run_routed_composition_outcome import ALPHA, N_BOOT, RNG_SEED, _bootstrap  # noqa: E402
from experiments.tqdm_loop import tqdm_iter  # noqa: E402

SD = ROOT / "artifacts" / "strategic_demand" / "sppo"
SPEC_PATH = SD / "SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_V1_SPEC.json"
CONTRACT_PATH = SD / "SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_CONTRACT_RESULT.json"
RESULT_PATH = SD / "SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_RESULT.json"
ROWS_PATH = SD / "SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_ROWS.csv"
SHARD_GLOB = "SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_SHARD*_PARTIAL.jsonl"
SEALED_PARITY_ROWS = SD / "confirmatory_b3_entity_repair_corrected_specialist_crossover_eval_rows.csv"
B33_GENOME = SD / "pole_b2_candidates" / "B3-3_lockdef10_2v1.json"
MECHANISM_FILE = ROOT / "experiments" / "probe_learned_composition.py"

DEVICE = "cuda"
N = 4
EXP_ID = "SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4"
SEED_BASE, SEED_N, SEED_CLASS = 21_100_001, 128, "sealed_confirmatory"
SEEDS = list(range(SEED_BASE, SEED_BASE + SEED_N))
ARMS = ("pi_A", "A_prime", "pi_B")
POLES = ("A", "B")
ARM_POLICY = {"pi_A": "pi_A", "A_prime": "pi_A", "pi_B": "pi_B"}
BASE_KEY = {"A": "OP6", "B": "OP7"}
PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
LABEL_CONFIRMED = "SCAFFOLDED_CROSSOVER_CONFIRMED"
LABEL_NOT = "SCAFFOLDED_CROSSOVER_NOT_ESTABLISHED"
GUARD_SENTENCE = ("A' is pi_A with an externally imposed defender structure. A scaffolded crossover is a causal "
                  "bridge showing what the balanced structure does to the payoff separation; it is not a fully "
                  "learned crossover and must not be reported as one.")

# (contrast name) -> (minuend (arm, pole), subtrahend (arm, pole)); win-rate currency, paired by seed
CONTRASTS = {
    "Delta_A_prime": (("A_prime", "A"), ("pi_B", "A")),
    "Delta_B_prime": (("pi_B", "B"), ("A_prime", "B")),
    "I_A": (("A_prime", "A"), ("pi_A", "A")),
    "I_B": (("A_prime", "B"), ("pi_A", "B")),
    "native_Delta_A": (("pi_A", "A"), ("pi_B", "A")),
    "native_Delta_B": (("pi_B", "B"), ("pi_A", "B")),
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _sha_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pair_for_seed(seed: int) -> tuple[int, int]:
    return PAIRS[int(seed) % len(PAIRS)]


def _cells() -> list[tuple[str, str, int]]:
    return [(arm, pole, s) for s in SEEDS for arm in ARMS for pole in POLES]


def _registered() -> tuple[bool, str]:
    from experiments import seed_registry as sr
    ok, msg = sr.check_block(SEED_BASE, SEED_BASE + SEED_N - 1, SEED_CLASS, experiment_id=EXP_ID)
    entry = next((b for b in sr.load()["blocks"] if b["experiment_id"] == EXP_ID), None)
    good = bool(ok and entry is not None and entry["status"] == "RESERVED" and entry["lo"] == SEED_BASE
                and entry["hi"] == SEED_BASE + SEED_N - 1 and entry["seed_class"] == SEED_CLASS)
    return good, f"{msg}; registry status={entry and entry['status']} class={entry and entry['seed_class']}"


# ---------------------------------------------------------------- frozen decision rule

def terminal_label(lcb_a_prime: float, lcb_b_prime: float) -> str:
    """CONFIRMED iff both primary lower bounds are strictly above zero (audit failure is the sealed status)."""
    return LABEL_CONFIRMED if (lcb_a_prime > 0 and lcb_b_prime > 0) else LABEL_NOT


def _label_selftest() -> tuple[bool, str]:
    cases = [("both lower bounds positive", 0.05, 0.30, LABEL_CONFIRMED),
             ("A' side fails", -0.02, 0.30, LABEL_NOT), ("B' side fails", 0.05, -0.10, LABEL_NOT),
             ("lower bound of exactly zero is not above zero", 0.0, 0.30, LABEL_NOT),
             ("both fail", -0.1, -0.1, LABEL_NOT), ("tiny positive bounds still confirm", 1e-6, 1e-6, LABEL_CONFIRMED)]
    bad = [n for n, a, b, want in cases if terminal_label(a, b) != want]
    return not bad, (f"{len(cases)}/{len(cases)} synthetic cases return the frozen label" if not bad else f"WRONG on: {bad}")


# ---------------------------------------------------------------- environment / poles

def _setup_poles() -> dict[str, Any]:
    """Mirror eval_specialist_crossover_scaled.py: resolve BOTH poles at the live team size and attest them
    against the governing certification, with the B3-3 genome supplied so canonical OP7 can never be scored."""
    import experiments.r2_learned_crossover as R2
    from experiments.opponent_spec import pole_A_genome
    from experiments.pole_attestation import (assert_resolved_matches_certification, format_attestation_banner,
                                              governing_certification, resolve_pole_genome)
    R2.AGENTS = N
    verdict, cert_path = governing_certification(N)
    att = {}
    for pol in POLES:
        g = resolve_pole_genome(pol, N, str(B33_GENOME) if pol == "B" else None)
        att[pol] = assert_resolved_matches_certification(pol, N, cert_path, g, is_smoke=False)
    pole_b = resolve_pole_genome("B", N, str(B33_GENOME))
    return {"R2": R2, "verdict": verdict, "cert_path": cert_path, "att": att, "banner": format_attestation_banner,
            "genomes": {"A": {"OP6": pole_A_genome(N)}, "B": {"OP7": pole_b}}, "pole_b": pole_b}


def _load_policies(R2, paths: dict[str, Path], device: str) -> dict[str, Any]:
    from rl.custom_ppo import load_custom_ppo_policy
    probe = R2.build_env(device, SEED_BASE)
    obs_space, act_space = probe.observation_space, probe.action_space
    grid_agents = int(obs_space.spaces["grid"].shape[0])
    probe.close()
    if grid_agents != N:
        raise SystemExit(f"FAIL-CLOSED: env grid agent dim {grid_agents} != team size {N}")
    pols = {n: load_custom_ppo_policy(str(p), obs_space, act_space, device=device) for n, p in paths.items()}
    for n, pol in pols.items():
        m = pol.model
        if (getattr(m, "uses_latent_strategy", False) or getattr(m, "role_conditioning_enabled", False)
                or getattr(m, "assignment_conditioning_enabled", False)):
            raise SystemExit(f"REFUSING: {n} is latent-, role-, or assignment-conditioned; specialists must be plain")
    return pols


def _checkpoint_paths(spec: dict) -> dict[str, Path]:
    return {n: ROOT / spec["INTEGRITY_FROZEN"]["checkpoints"][n]["path"] for n in ("pi_A", "pi_B")}


def _open_opponent(env, core, genomes: dict, pole: str, context: str):
    from experiments.opponent_spec import assert_live_opponent_batch, install_keyed_opponent_overlays
    from rl.curriculum import phase_from_tag
    core._bt_profile_override = None
    core._sds_opening_hold_steps = 0
    gen = genomes[pole]
    install_keyed_opponent_overlays(core, gen)
    key = BASE_KEY[pole]
    env.env_method("set_phase", phase_from_tag(key))
    env.env_method("set_next_opponent", "SCRIPTED", key)
    return gen, key


def run_episode(setup: dict, policy, pole: str, seed: int, forced_ids: tuple[int, ...], device: str) -> dict[str, Any]:
    """One learned episode, mirroring eval_specialist_crossover_scaled.run_cell action for action. `forced_ids`
    are the agents whose resolved target is forced to DEFEND's (empty for the native arms)."""
    from experiments.opponent_spec import assert_live_opponent_batch
    from gpu_env._core._entity_obs import augment_obs_with_entities
    R2 = setup["R2"]
    env = R2.build_env(device, seed)
    core = env.core
    try:
        policy.reset_strategy()
        gen, key = _open_opponent(env, core, setup["genomes"], pole, "bridge")
        obs = env.reset()
        obs["global_state"] = env.state()
        obs = augment_obs_with_entities(obs, core, side="blue")
        assert_live_opponent_batch(core, gen, allowed_keys=(key,), context=f"scaffold bridge {pole} seed {seed}")
        got = core._bt_resolved_profile_tensors().get("min_alive_for_defender")
        got_val = int(got.flatten()[0].item()) if hasattr(got, "flatten") else int(got)
        if got_val != N:
            raise SystemExit(f"FAIL-CLOSED: live pole {pole} resolves min_alive_for_defender={got_val}, expected {N}")
        for i in forced_ids:
            P.install_forced_defend_target(core, int(i))
        terminal, steps = None, 0
        for _ in range(R2.MAX_STEPS):
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
        blue, red = terminal
        return {"steps": steps, "blue": blue, "red": red, "win": int(blue > red), "margin": blue - red}
    finally:
        env.close()


# ------------------------------------------------------ scripted fixture (contract only)

def _scripted_episode(setup: dict, pole: str, seed: int, roles: tuple[int, ...], device: str, *, n_macros: int,
                      override_ids: tuple[int, ...] = ()) -> dict[str, Any]:
    """A scripted team at 4v4. Reference: `roles` marks defenders, fed the raw DEFEND macro at n_macros=8.
    Injected: the `override_ids` agents are fed a non-DEFEND raw macro (GO_TO) but have their target forced."""
    from experiments.run_pyquaticus_6v6_role_composition_sweep import action_for_roles_n
    from gpu_env import GPUCTFVecEnv, GPUFieldConfig
    R2 = setup["R2"]
    cfg = GPUFieldConfig(n_envs=1, max_blue_agents=N, max_red_agents=N, map_set="train", map_layout=R2.MAP,
                         max_decision_steps=R2.MAX_STEPS, aquaticus_profile=True, rules_profile="OURS", device=device,
                         seed=int(seed), tag_telemetry_enabled=True, own_flag_home_required_to_score=True,
                         n_macros=int(n_macros), **R2.RULESET)
    env = GPUCTFVecEnv(cfg)
    core = env.core
    try:
        _open_opponent(env, core, setup["genomes"], pole, "fixture")
        env.reset()
        for i in override_ids:
            P.install_forced_defend_target(core, int(i))
        ticks, terminal = [], None
        for _ in range(R2.MAX_STEPS):
            action = action_for_roles_n(core, roles)
            for i in override_ids:
                action[0, i, 0], action[0, i, 1] = 0, 0
            ticks.append(P._snap(core))
            env.step_async(action)
            _o, _r, done, info = env.step_wait()
            if bool(np.asarray(done).any()):
                i0 = info[0] if isinstance(info, (list, tuple)) else info
                res = (i0 or {}).get("episode_result") or {}
                terminal = (int(res.get("blue_score", 0)), int(res.get("red_score", 0)))
                break
        if terminal is None:
            terminal = (int(core.blue_score[0]), int(core.red_score[0]))
        out = {k: np.stack([t[k] for t in ticks]) for k in ("pos", "alive", "tagged", "carrying")}
        out["outcome"] = terminal
        return out
    finally:
        env.close()


def _equivalence_cell(setup: dict, pole: str, pair: tuple[int, int], seed: int) -> tuple[bool, str]:
    roles = tuple(1 if i in pair else 0 for i in range(N))
    ref = _scripted_episode(setup, pole, seed, roles, DEVICE, n_macros=8)
    problems: list[str] = []
    for n_macros in (8, 5):
        inj = _scripted_episode(setup, pole, seed, roles, DEVICE, n_macros=n_macros, override_ids=pair)
        for k in ("pos", "alive", "tagged", "carrying"):
            if ref[k].shape != inj[k].shape or not np.array_equal(ref[k], inj[k]):
                problems.append(f"pole={pole} pair={pair} n_macros={n_macros} field={k} MISMATCH")
        if ref["outcome"] != inj["outcome"]:
            problems.append(f"pole={pole} pair={pair} n_macros={n_macros} outcome {inj['outcome']} != {ref['outcome']}")
    return (not problems, "; ".join(problems) if problems else f"pole={pole} pair={pair}: exact at n_macros 8 and 5")


# ------------------------------------------------------------------------------ contracts

def contracts() -> dict:
    checks: list[dict] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "gating": True, "passed": bool(passed), "detail": detail})

    spec = _load_json(SPEC_PATH)
    record("C0_SPEC_FROZEN", str(spec.get("status", "")).startswith("FROZEN"), f"spec status = {spec.get('status')!r}")

    want_pin = spec["MECHANISM"]["source_pin"]["sha256"]
    got_pin = _sha_file(MECHANISM_FILE)
    record("C1_MECHANISM_SOURCE_EQUALS_THE_PIN", got_pin == want_pin, f"{got_pin} vs pinned {want_pin}")

    paths = _checkpoint_paths(spec)
    pins = {n: spec["INTEGRITY_FROZEN"]["checkpoints"][n]["sha256"] for n in paths}
    got = {n: _sha_file(p) for n, p in paths.items() if p.is_file()}
    record("C2_CHECKPOINTS_EQUAL_THE_PINNED_HASHES", got == pins, f"{got} vs pinned {pins}")

    setup = _setup_poles()
    att, cert_path = setup["att"], setup["cert_path"]
    for pol in POLES:
        print(f"  POLE {pol} ATTESTATION vs {cert_path.name}:\n{setup['banner'](att[pol])}")
    want_cert = spec["INTEGRITY_FROZEN"]["pole_attestation"]
    lb = att["B"]
    lo = dict(lb["live_overlay"] or {})
    ok3 = (setup["verdict"] == "CERTIFIED" and cert_path.name == Path(want_cert["governing_certification"]).name
           and _sha_file(cert_path) == want_cert["certification_sha256"]
           and _sha_file(B33_GENOME) == want_cert["pole_b_genome_sha256"]
           and all(att[p]["hashes_match"] for p in POLES)
           and lb["live_genome_id"] == "SDS2_B3_LOCKDEF10_2V1" and setup["pole_b"].base_opponent == "OP7"
           and lo.get("lock_defender") == 10 and lo.get("enable_2v1") is True and lo.get("min_alive_for_defender") == N)
    record("C3_BOTH_POLES_ATTESTED_AND_POLE_B_IS_THE_CERTIFIED_B3_3_GENOME",
           ok3, f"certification {cert_path.name} verdict={setup['verdict']}; live Pole B {lb['live_genome_id']} overlay={lo}; "
                f"hashes_match A={att['A']['hashes_match']} B={att['B']['hashes_match']}")

    counts = {p: sum(1 for s in SEEDS if pair_for_seed(s) == p) for p in PAIRS}
    record("C4_ROTATION_COVERS_ALL_SIX_PAIRS_WITHIN_ONE_SEED_OF_UNIFORM",
           len(counts) == 6 and min(counts.values()) > 0 and max(counts.values()) - min(counts.values()) <= 1,
           f"pair counts: {counts}")

    ok5, msg5 = _registered()
    prior = [(16_700_001, 16_700_128), (17_800_001, 17_800_128), (18_300_001, 18_300_128), (18_500_001, 18_500_128),
             (20_900_001, 20_900_096), (21_000_001, 21_000_096)]
    disjoint = all(SEED_BASE + SEED_N - 1 < lo_ or SEED_BASE > hi_ for lo_, hi_ in prior)
    record("C5_BLOCK_RESERVED_AS_SEALED_CONFIRMATORY_UNSPENT_AND_DISJOINT", ok5 and disjoint,
           f"{msg5}; disjoint from {prior}: {disjoint}")

    mism = []
    for (pole, pair, seed) in (("A", (1, 3), 99_901_001), ("B", (0, 2), 99_901_002)):
        ok, detail = _equivalence_cell(setup, pole, pair, seed)
        if not ok:
            mism.append(detail)
    record("C6_TWO_DEFENDER_INJECTION_MATCHES_A_SCRIPTED_TWO_DEFENDER_TEAM_EXACTLY_AT_4V4", not mism,
           "; ".join(mism) if mism else "2/2 (pole, pair) cells exact at n_macros 8 and 5 "
                                       "(pos/alive/tagged/carrying for all 4 agents every tick, and the outcome)")

    pols = _load_policies(setup["R2"], paths, DEVICE)
    record("C7_LOADED_POLICIES_ARE_PLAIN_SPECIALISTS", True, "neither is latent-, role-, or assignment-conditioned")
    sealed = {}
    with SEALED_PARITY_ROWS.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sealed[(r["policy"], r["pole"], int(r["seed"]))] = (int(r["blue"]), int(r["red"]), int(r["win"]), int(r["margin"]))
    import torch
    before = {n: L._param_digest(p) for n, p in pols.items()}
    bad = []
    with torch.no_grad():
        for name in ("pi_A", "pi_B"):
            for pole in POLES:
                for seed in range(18_300_001, 18_300_005):
                    r = run_episode(setup, pols[name], pole, seed, (), DEVICE)
                    got_t = (r["blue"], r["red"], r["win"], r["margin"])
                    if got_t != sealed[(name, pole, seed)]:
                        bad.append(f"{name}/{pole}/{seed}: replayed {got_t} != sealed {sealed[(name, pole, seed)]}")
    record("C8_NATIVE_PATH_REPRODUCES_16_SEALED_EPISODES_EXACTLY", not bad,
           "; ".join(bad) if bad else "16/16 episodes of the sealed corrected crossover reproduced exactly "
                                     "(env config, entity observations, attestation and checkpoint loading all agree)")
    record("C9_POLICY_PARAMETERS_UNCHANGED_BY_THE_PARITY_RUN", before == {n: L._param_digest(p) for n, p in pols.items()},
           "parameter digests equal before/after")

    ok10, msg10 = _label_selftest()
    record("C10_TERMINAL_RULE_RETURNS_THE_FROZEN_LABEL_ON_SYNTHETIC_CASES", ok10, msg10)

    src = inspect.getsource(run_episode)
    record("C11_REAL_RUNNER_NEVER_WRITES_A_RAW_MACRO_OR_COMMIT_TENSOR", "action[" not in src and "commit_macro" not in src,
           "run_episode only calls install_forced_defend_target(); it touches no action array or commit_* tensor")

    n_failed = sum(1 for c in checks if not c["passed"])
    report = {"record_id": "SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_CONTRACT_RESULT", "utc": _now(), "implements": SPEC_PATH.name,
              "n_checks": len(checks), "n_gating": len(checks), "n_failed": n_failed, "checks": checks,
              "DECISION": "CONTRACTS_PASS" if n_failed == 0 else "CONTRACT_FAILURE",
              "claim_boundary": f"Contracts only. No seed from {SEED_BASE}-{SEED_BASE + SEED_N - 1} was spent; the parity replay "
                                "uses spent seeds 18300001-4 for verification only."}
    CONTRACT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\n{'=' * 66}\n  BRIDGE CONTRACTS: {report['DECISION']}  ({n_failed}/{len(checks)} gating failed)\n{'=' * 66}")
    for c in checks:
        print(f"  [{'PASS' if c['passed'] else 'FAIL'}] {c['name']}: {c['detail']}")
    return report


# ---------------------------------------------------------------------------------- run

def _shard_path(i: int, k: int) -> Path:
    return SD / f"SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_SHARD{i}OF{k}_PARTIAL.jsonl"


def _load_partial(path: Path) -> dict[tuple, dict]:
    out: dict[tuple, dict] = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                o = json.loads(line)
                out[tuple(o["key"])] = o["row"]
    return out


def run_shard(shard: int, n_shards: int) -> int:
    if _load_json(CONTRACT_PATH).get("DECISION") != "CONTRACTS_PASS":
        raise SystemExit("REFUSING: contracts did not pass")
    if RESULT_PATH.exists():
        raise SystemExit(f"REFUSING: result already exists: {RESULT_PATH}")
    ok, msg = _registered()
    if not ok:
        raise SystemExit(f"REFUSING: seed block not reserved as frozen: {msg}")
    spec = _load_json(SPEC_PATH)
    paths = _checkpoint_paths(spec)
    pins = {n: spec["INTEGRITY_FROZEN"]["checkpoints"][n]["sha256"] for n in paths}
    if {n: _sha_file(p) for n, p in paths.items()} != pins:
        raise SystemExit("REFUSING: checkpoint hashes differ from the frozen pins")

    mine = [c for i, c in enumerate(_cells()) if i % n_shards == shard]
    path = _shard_path(shard, n_shards)
    done = _load_partial(path)
    pending = [c for c in mine if c not in done]
    print(f"  shard {shard}/{n_shards}: {len(mine)} cells, {len(done)} already recorded, {len(pending)} to run", flush=True)
    if not pending:
        return 0
    import torch
    setup = _setup_poles()
    pols = _load_policies(setup["R2"], paths, DEVICE)
    before = {n: L._param_digest(p) for n, p in pols.items()}
    with path.open("a", encoding="utf-8") as fh, torch.no_grad():
        for key in tqdm_iter(pending, desc=f"bridge shard {shard}/{n_shards} (cuda)", total=len(pending), unit="ep"):
            arm, pole, seed = key
            forced = pair_for_seed(seed) if arm == "A_prime" else ()
            r = run_episode(setup, pols[ARM_POLICY[arm]], pole, seed, forced, DEVICE)
            row = {"arm": arm, "pole": pole, "seed": seed, "pair": "-".join(map(str, forced)), **r}
            fh.write(json.dumps({"key": list(key), "row": row}) + "\n"); fh.flush()
    if before != {n: L._param_digest(p) for n, p in pols.items()}:
        raise SystemExit("ABORT: policy parameters changed during a read-only run")
    print(f"  shard {shard}/{n_shards} complete (parameter digests unchanged)", flush=True)
    return 0


# ------------------------------------------------------------------------------- analyze

def analyze() -> int:
    from experiments import run_state as rs
    from experiments import seed_registry as sr

    if _load_json(CONTRACT_PATH).get("DECISION") != "CONTRACTS_PASS":
        raise SystemExit("REFUSING: contracts did not pass")
    if RESULT_PATH.exists():
        raise SystemExit(f"REFUSING: result already exists: {RESULT_PATH}")
    ok, msg = _registered()
    if not ok:
        raise SystemExit(f"REFUSING: seed block not reserved as frozen: {msg}")

    merged: dict[tuple, dict] = {}
    for path in sorted(SD.glob(SHARD_GLOB)):
        for key, row in _load_partial(path).items():
            if key in merged and merged[key] != row:
                raise SystemExit(f"ABORT: conflicting rows for {key} across shards")
            merged[key] = row
    want = set(_cells())
    missing, extra = sorted(want - set(merged)), sorted(set(merged) - want)
    if missing or extra:
        raise SystemExit(f"ABORT: {len(missing)} cell(s) missing (e.g. {missing[:2]}), {len(extra)} unexpected "
                         f"(e.g. {extra[:2]}). Run/resume the shards first.")
    bad_pair = [k for k, r in merged.items()
                if r["pair"] != ("-".join(map(str, pair_for_seed(k[2]))) if k[0] == "A_prime" else "")]
    if bad_pair:
        raise SystemExit(f"ABORT: {len(bad_pair)} row(s) whose pair does not follow the frozen rotation, e.g. {bad_pair[0]}")
    rows = [merged[c] for c in _cells()]
    with ROWS_PATH.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    spec = _load_json(SPEC_PATH)
    paths = _checkpoint_paths(spec)
    pins = {n: spec["INTEGRITY_FROZEN"]["checkpoints"][n]["sha256"] for n in paths}
    state = rs.RunState(SD, "SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4").begin(spec=SPEC_PATH.name, n_episodes=len(rows))

    val = {f: {(r["arm"], r["pole"], int(r["seed"])): float(r[f]) for r in rows} for f in ("win", "blue", "margin")}

    def diff(f: str, plus: tuple[str, str], minus: tuple[str, str]) -> np.ndarray:
        return np.asarray([val[f][(plus[0], plus[1], s)] - val[f][(minus[0], minus[1], s)] for s in SEEDS], dtype=np.float64)

    contrasts = {f: {name: _bootstrap(diff(f, a, b)) for name, (a, b) in CONTRASTS.items()} for f in ("win", "blue", "margin")}
    cell_means = {f"{arm}_pole{pole}": {f: round(float(np.mean([val[f][(arm, pole, s)] for s in SEEDS])), 6)
                                        for f in ("win", "blue", "margin")} for arm in ARMS for pole in POLES}
    win = contrasts["win"]
    label = terminal_label(win["Delta_A_prime"]["lcb95"], win["Delta_B_prime"]["lcb95"])
    sides = {"Delta_A_prime_lcb95_above_zero": bool(win["Delta_A_prime"]["lcb95"] > 0),
             "Delta_B_prime_lcb95_above_zero": bool(win["Delta_B_prime"]["lcb95"] > 0)}

    claims = [rs.Claim(name=f"{f}_{name}", recorded={k: contrasts[f][name][k] for k in ("mean", "lcb95", "ucb95")},
                       minuend={"arm": a[0], "pole": a[1]}, subtrahend={"arm": b[0], "pole": b[1]}, value_field=f)
              for f in ("win", "blue", "margin") for name, (a, b) in CONTRASTS.items()]

    payload = {
        "record_id": "SCAFFOLDED_A_CROSSOVER_BRIDGE_4V4_RESULT", "implements": SPEC_PATH.name, "utc": _now(), "device": DEVICE,
        "seed_block": {"base": SEED_BASE, "n": SEED_N, "class": SEED_CLASS},
        "PRIMARY_WIN_RATE_CONTRASTS": {n: {**win[n], "lcb95_above_zero": bool(win[n]["lcb95"] > 0)} for n in ("Delta_A_prime", "Delta_B_prime")},
        "PRIMARY_SIDES_CLEARED_DESCRIPTIVE": sides,
        "MECHANISM_DIAGNOSTICS_NOT_GATES_win_rate": {n: win[n] for n in ("I_A", "I_B")},
        "NATIVE_CROSSOVER_ON_THE_FRESH_BLOCK_DESCRIPTIVE_win_rate": {n: win[n] for n in ("native_Delta_A", "native_Delta_B")},
        "DESCRIPTIVE_BLUE_GOALS_CONTRASTS": contrasts["blue"], "DESCRIPTIVE_MARGIN_CONTRASTS": contrasts["margin"],
        "CELL_MEANS": cell_means,
        "sealed_baseline_for_reference": spec["WHY_THIS_LEVER"]["baseline_verified_from_the_sealed_rows"],
        "TERMINAL_OUTCOME": label,
        "interpretation_guard": GUARD_SENTENCE,
        "provenance": {"spec_sha256": _sha_file(SPEC_PATH), "script_sha256": _sha_file(Path(__file__)),
                       "mechanism_source_sha256": _sha_file(MECHANISM_FILE),
                       "certification_sha256": _sha_file(ROOT / spec["INTEGRITY_FROZEN"]["pole_attestation"]["governing_certification"]),
                       "pole_b_genome_sha256": _sha_file(B33_GENOME)},
        "claim_boundary": "Two frozen 4v4 learned specialists, the two certified 4v4 poles, a rollout-time scaffold on pi_A only, "
                          "fresh sealed_confirmatory seeds. TERMINAL_OUTCOME is certified ONLY if this record's own status is "
                          "SEALED; integrity/audit failure takes precedence over every label. " + GUARD_SENTENCE,
    }
    plan = rs.AuditPlan(
        rows_csv=ROWS_PATH, expected_rows=len(rows), expected_seeds=SEEDS, group_by=("arm", "pole"), seed_field="seed",
        int_fields=("seed", "steps", "blue", "red", "margin"), binary_fields=("win",),
        derived={"margin": rs.Derived("margin == blue - red", lambda r: r["blue"] - r["red"]),
                 "win": rs.Derived("win == (blue > red)", lambda r: int(r["blue"] > r["red"]))},
        checkpoints={n: (paths[n], pins[n]) for n in paths}, spec_path=SPEC_PATH, seed_class=SEED_CLASS,
        experiment_id=EXP_ID, n_boot=N_BOOT, alpha=ALPHA, rng_seed=RNG_SEED, claims=claims)
    rs.seal(out_path=RESULT_PATH, payload=payload, plan=plan, state=state, strict=False)
    sealed = _load_json(RESULT_PATH)
    sr.set_status(EXP_ID, "SPENT", note=f"sealed {sealed.get('status')}; {label}")
    print(json.dumps({"status": sealed.get("status"), "TERMINAL_OUTCOME": label, "sides_cleared": sides,
                      "win_rate_contrasts": {n: {k: win[n][k] for k in ("mean", "lcb95", "ucb95")} for n in CONTRASTS}}, indent=2))
    return 0 if sealed.get("status") == "SEALED" else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("contracts", "run", "analyze"), required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1, dest="n_shards")
    a = ap.parse_args()
    if a.stage == "contracts":
        return 0 if contracts()["DECISION"] == "CONTRACTS_PASS" else 2
    if a.stage == "run":
        if not 0 <= a.shard < a.n_shards:
            raise SystemExit(f"--shard must be in [0, {a.n_shards})")
        return run_shard(a.shard, a.n_shards)
    return analyze()


if __name__ == "__main__":
    raise SystemExit(main())
