r"""Verify a trained specialist pair against its frozen spec before the seeds are marked SPENT.

    python experiments/verify_specialist_foundation_pair.py \
        --spec artifacts/strategic_demand/sppo/STANDARDIZED_2V2_FOUNDATION_SPEC.json \
        --trainer-git-sha ca52a24bc19f9817675c1213de4a95007a7c519a [--write]

Every expectation comes from the frozen spec or the sealed certification record, never from
the run being checked. Per policy it checks the final checkpoint, the run manifest, the resolved
run config, the checkpoint's own cfg and parameter names, the launcher's watch log and the run
log. Across the pair it checks that both manifests record the SAME trainer git identity (the
provenance lock: the repo HEAD may move later, the runs' recorded code identity must not).
A missing field is a failure, never a default.

With --write the record is written next to the spec, and only if every check passed. This
script does not change the seed registry; marking the block SPENT is a separate, explicit step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MANIFEST_RECORD = "train_specialist_scale run manifest"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_text(p: Path) -> str:
    raw = p.read_bytes()
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16", errors="replace")
    return raw.decode("utf-8", errors="replace")


def _arg(cmd: str, flag: str) -> str:
    m = re.search(rf"{re.escape(flag)}\s+(\S+)", cmd)
    if not m:
        raise SystemExit(f"FAIL-CLOSED: spec launch command has no {flag}: {cmd}")
    return m.group(1)


def verify(spec_path: Path, trainer_git_sha: str) -> dict:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if not str(spec.get("status", "")).startswith("FROZEN"):
        raise SystemExit(f"FAIL-CLOSED: spec is not frozen: {spec.get('status')!r}")
    seeds, launch, recipe = spec["SEEDS"], spec["LAUNCH"], spec["RECIPE_identical_to_4v4_base"]
    n_expected = int(_arg(launch["A"], "--team-size"))
    budget = int(recipe["total_timesteps"])

    cert_path = ROOT / "artifacts" / "strategic_demand" / "sppo" / f"STRATEGIC_DEMAND_{n_expected}v{n_expected}_CERTIFICATION.json"
    cert = json.loads(cert_path.read_text(encoding="utf-8"))
    certified_hash = cert["POLE_HANDOFF"]["pole_config_hash"]

    checks: list[dict] = []

    def check(policy: str, name: str, ok: bool, detail) -> None:
        checks.append({"policy": policy, "check": name, "ok": bool(ok), "detail": detail})

    watch = ROOT / "artifacts" / f"scale_{n_expected}v{n_expected}_specialists" / "std_foundation_watch.log"
    watch_txt = _read_text(watch) if watch.is_file() else ""
    per_policy: dict[str, dict] = {}

    for pol in ("A", "B"):
        seed = int(seeds[pol])
        cmd = launch[pol]
        check(pol, "spec_command_seed_matches_SEEDS", int(_arg(cmd, "--seed")) == seed, _arg(cmd, "--seed"))
        final = ROOT / launch["expected_finals"][pol]
        run_dir = final.parent.parent
        info: dict = {"final": final.relative_to(ROOT).as_posix(), "seed": seed}

        # final checkpoint
        check(pol, "final_checkpoint_exists", final.is_file(), info["final"])
        if final.is_file():
            import torch
            info["final_sha256"] = _sha256(final)
            ck = torch.load(final, map_location="cpu", weights_only=False)
            step = int(ck.get("global_step", -1))
            info["global_step"] = step
            rollout = 16 * 128
            check(pol, "global_step_reached_budget", budget <= step < budget + rollout, step)
            ccfg = ck.get("cfg") or {}
            check(pol, "checkpoint_cfg_seed", ccfg.get("seed") == seed, ccfg.get("seed"))
            check(pol, "checkpoint_cfg_entity_repair_off", ccfg.get("entity_repair_enabled") is False,
                  ccfg.get("entity_repair_enabled"))
            check(pol, "checkpoint_cfg_anchor_dataset_empty", ccfg.get("sappo_anchor_dataset", None) == "",
                  ccfg.get("sappo_anchor_dataset", "<absent>"))
            check(pol, "checkpoint_cfg_no_load_path", not ccfg.get("load_path"), ccfg.get("load_path"))
            ent = [k for k in (ck.get("model_state_dict") or {}) if "entity" in k]
            check(pol, "checkpoint_has_no_entity_parameters", not ent and bool(ck.get("model_state_dict")), ent[:3])

        # run manifest
        mp = run_dir / "run_manifest.json"
        m = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}
        check(pol, "manifest_exists", bool(m), mp.relative_to(ROOT).as_posix())
        check(pol, "manifest_record", m.get("record") == MANIFEST_RECORD, m.get("record"))
        check(pol, "manifest_seed", m.get("seed") == seed, m.get("seed"))
        check(pol, "manifest_policy", m.get("policy") == pol, m.get("policy"))
        check(pol, "manifest_team_size", m.get("team_size") == n_expected, m.get("team_size"))
        check(pol, "manifest_not_smoke", m.get("smoke_non_scientific") is False, m.get("smoke_non_scientific"))
        check(pol, "manifest_budget", m.get("total_timesteps") == budget, m.get("total_timesteps"))
        check(pol, "manifest_trainer_git_sha", m.get("git_sha") == trainer_git_sha, m.get("git_sha"))
        check(pol, "manifest_certification_record", m.get("certification_record") == cert_path.name,
              m.get("certification_record"))
        check(pol, "manifest_certification_CERTIFIED", m.get("certification_verdict") == "CERTIFIED",
              m.get("certification_verdict"))
        check(pol, "manifest_live_pole_attestation_passed", m.get("live_pole_attestation_passed") is True,
              m.get("live_pole_attestation_passed"))
        check(pol, "manifest_certified_hash_is_sealed_handoff", m.get("certified_config_hash") == certified_hash[pol],
              m.get("certified_config_hash"))
        check(pol, "manifest_live_hash_equals_certified", m.get("live_config_hash") == certified_hash[pol],
              m.get("live_config_hash"))
        check(pol, "manifest_entity_repair_off", m.get("entity_repair_enabled") is False, m.get("entity_repair_enabled"))
        check(pol, "manifest_no_warm_start", m.get("warm_start_from", "<absent>") is None, m.get("warm_start_from", "<absent>"))
        check(pol, "manifest_no_resume", m.get("resume_from", "<absent>") is None, m.get("resume_from", "<absent>"))
        check(pol, "manifest_pole_b_canonical", m.get("pole_b_source") == "canonical_pole_B_genome", m.get("pole_b_source"))
        info["manifest_git_sha"] = m.get("git_sha")
        info["live_config_hash"] = m.get("live_config_hash")

        # resolved run config
        rcs = sorted(run_dir.glob("*_run_config.json"))
        rc = json.loads(rcs[0].read_text(encoding="utf-8")) if len(rcs) == 1 else {}
        check(pol, "run_config_exists_once", len(rcs) == 1, [p.name for p in rcs])
        c = rc.get("resolved_ppo_config") or {}
        argv = rc.get("argv") or []
        spec_argv = cmd.split()[2:]          # drop interpreter and script
        check(pol, "run_config_argv_equals_spec_launch", argv[1:] == spec_argv, argv[1:])
        check(pol, "run_config_load_path_none", rc.get("load_path", "<absent>") is None and c.get("load_path", "<absent>") is None,
              [rc.get("load_path", "<absent>"), c.get("load_path", "<absent>")])
        check(pol, "run_config_anchor_dataset_empty", c.get("sappo_anchor_dataset", None) == "",
              c.get("sappo_anchor_dataset", "<absent>"))
        check(pol, "run_config_entity_repair_off", c.get("entity_repair_enabled") is False, c.get("entity_repair_enabled"))
        check(pol, "run_config_seed", c.get("seed") == seed, c.get("seed"))
        check(pol, "run_config_team_size", c.get("max_blue_agents") == n_expected, c.get("max_blue_agents"))
        check(pol, "run_config_recipe_n_envs", c.get("n_envs") == recipe["n_envs"], c.get("n_envs"))
        check(pol, "run_config_recipe_n_steps", c.get("n_steps") == recipe["n_steps"], c.get("n_steps"))
        check(pol, "run_config_recipe_lr", c.get("learning_rate") == recipe["learning_rate"], c.get("learning_rate"))
        check(pol, "run_config_git_sha", rc.get("git_sha") == trainer_git_sha, rc.get("git_sha"))

        # termination
        check(pol, "launcher_logged_exit_0", f"pi_{pol} exited 0" in watch_txt, watch.name)
        log = ROOT / "artifacts" / f"scale_{n_expected}v{n_expected}_specialists" / f"std_foundation_{pol}.log"
        err = log.with_suffix(".log.err")
        txt = (_read_text(log) if log.is_file() else "") + (_read_text(err) if err.is_file() else "")
        check(pol, "run_log_present", log.is_file(), log.name)
        check(pol, "run_log_no_traceback", "Traceback" not in txt, "Traceback" in txt)
        check(pol, "run_log_training_returned", "training returned" in txt, "training returned" in txt)
        per_policy[pol] = info

    shas = {per_policy[p].get("manifest_git_sha") for p in ("A", "B")}
    check("pair", "same_trainer_git_identity_for_both", len(shas) == 1 and trainer_git_sha in shas, sorted(map(str, shas)))
    check("pair", "launcher_logged_pair_done", "foundation DONE" in watch_txt, watch.name)

    failed = [c for c in checks if not c["ok"]]
    return {
        "record_id": f"{spec['record_id']}_VERIFICATION",
        "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "spec": spec_path.name,
        "spec_sha256": _sha256(spec_path),
        "certification_record": cert_path.name,
        "trainer_git_sha": trainer_git_sha,
        "registry_experiment_id": seeds["registry_experiment_id"],
        "PASS": not failed,
        "n_checks": len(checks),
        "n_failed": len(failed),
        "policies": per_policy,
        "checks": checks,
        "registry_status_change": "NOT performed by this script; set SPENT explicitly after PASS",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", required=True)
    ap.add_argument("--trainer-git-sha", required=True,
                    help="the commit the pair was launched from (from the spec's own freeze commit)")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    spec_path = Path(a.spec)
    rec = verify(spec_path, a.trainer_git_sha)
    for c in rec["checks"]:
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {c['policy']:<4} {c['check']:<44} {c['detail']}")
    print(f"\n{rec['n_checks'] - rec['n_failed']}/{rec['n_checks']} checks passed -> "
          f"{'PASS' if rec['PASS'] else 'FAIL'}")
    if a.write:
        if not rec["PASS"]:
            print("not writing: verification failed")
            return 1
        out = spec_path.with_name(f"{rec['record_id']}.json")
        if out.exists():
            print(f"REFUSING to overwrite {out.name}")
            return 1
        out.write_text(json.dumps(rec, indent=2), encoding="utf-8")
        print(f"-> {out}")
    return 0 if rec["PASS"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
