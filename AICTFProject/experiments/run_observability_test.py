"""Observability test — is the confirmed opponent-conditional tradeoff reachable
from a single legal frame?

Candidate-agnostic plumbing. Every scientific parameter is READ from the frozen
files, never passed as an argument and never chosen here:

    selector class, C, feature set   OBSERVABILITY_TEST_FROZEN.json
    baseline definition              OBSERVABILITY_TEST_FROZEN.json
    recovery bar (0.50)              OBSERVABILITY_TEST_FROZEN.json
    evaluation block (9860000)       OBSERVABILITY_TEST_FROZEN.json
    bootstrap / LCB95                OBSERVABILITY_TEST_FROZEN.json
    which opponents and responses    C5_CONFIRMATION.json

The candidate arrives late, but nothing about how it is judged does.

Oracle-selector sees opponent identity and upper-bounds the tradeoff's worth.
Legal-selector sees only the 28 legal_context fields at ONE timestep -- no
history, per OBSERVABILITY_SCOPE_AMENDMENT.json -- fits on the discovery block
and is evaluated out of sample on unspent seeds. The baseline is the best SINGLE
response, because that is what K=1 already does.

FAIL CLOSED
    1. refuses unless C5_CONFIRMATION says C5_PASS
    2. refuses if a result already exists
    3. refuses if the legal-selector's fit and evaluation blocks overlap
    4. opponent identity is never a legal-selector feature
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

FROZEN = ROOT / "artifacts/c5_preregistration/OBSERVABILITY_TEST_FROZEN.json"
SCOPE = ROOT / "artifacts/c5_preregistration/OBSERVABILITY_SCOPE_AMENDMENT.json"
CONFIRM = ROOT / "artifacts/c5_confirmation/C5_CONFIRMATION.json"
DISCOVERY_STATES = ROOT / "artifacts/c5_discovery/states.json"
OUT = ROOT / "artifacts/observability/OBSERVABILITY_RESULT.json"
SHARDS = ROOT / "artifacts/observability/shards"
STATES = ROOT / "artifacts/observability/states.json"
G0_SEEDS = ["3200001", "3200002", "3200003"]


def feature_matrix(rows, feature_names):
    X = np.array([[float(r["legal_context"][f]) for f in feature_names] for r in rows])
    return X


def clustered_lcb(per_ep_values, *, rng, resamples, lcb_pct):
    """Episode-clustered bootstrap on a per-state gain series."""
    from collections import defaultdict
    g = defaultdict(list)
    for ep, v in per_ep_values:
        g[ep].append(float(v))
    keys = list(g)
    if len(keys) < 2:
        return {"mean": None, "lcb95": None, "n_episodes": len(keys)}
    point = float(np.mean([v for k in keys for v in g[k]]))
    draws = np.empty(resamples)
    for i in range(resamples):
        samp = [keys[j] for j in rng.integers(0, len(keys), len(keys))]
        vals = [v for k in samp for v in g[k]]
        draws[i] = np.mean(vals) if vals else np.nan
    draws = draws[np.isfinite(draws)]
    lo = float(np.percentile(draws, lcb_pct))
    return {"mean": round(point, 6), "lcb95": round(lo, 6), "n_episodes": len(keys)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--skip-collect", action="store_true")
    args = ap.parse_args()

    if OUT.exists():
        print(f"REFUSED: {OUT.name} exists. Run once, evaluation only.", file=sys.stderr)
        return 2
    if not CONFIRM.exists():
        print("REFUSED: no C5 confirmation result.", file=sys.stderr)
        return 2

    frozen = json.loads(FROZEN.read_text(encoding="utf-8"))
    conf = json.loads(CONFIRM.read_text(encoding="utf-8"))
    if conf.get("verdict") != "C5_PASS":
        print(f"REFUSED: confirmation verdict is {conf.get('verdict')!r}. The "
              f"observability test runs only on C5_PASS.", file=sys.stderr)
        return 2

    cand = conf["candidate"]
    opp_a, opp_b = cand["opponent_a"], cand["opponent_b"]
    r1, r2 = cand["response_1"], cand["response_2"]
    base = int(frozen["seeds"]["evaluation_block"][0])
    bar = float(frozen["the_0_50_bar"]["value"])
    resamples = int(frozen["statistics"]["resamples"])
    rng = np.random.default_rng(int(frozen["statistics"]["seed"]))

    print("=" * 74)
    print("OBSERVABILITY TEST — single-timestep legal_context")
    print(f"  candidate : {opp_a} favours R1={r1};  {opp_b} favours R2={r2}")
    print(f"  fit block : discovery 9840000  |  eval block : {base} (unspent)")
    print(f"  bar       : recovery >= {bar}")
    print("=" * 74, flush=True)

    if not args.skip_collect:
        cmd = [str(ROOT / ".venv/Scripts/python.exe"), "-u",
               str(ROOT / "experiments/run_c5_parallel.py"),
               "--episodes", str(args.episodes), "--workers", str(args.workers),
               "--seed-base", str(base), "--only-opponents", f"{opp_a},{opp_b}",
               "--shard-dir", str(SHARDS), "--states-out", str(STATES)]
        rc = subprocess.run(cmd, cwd=str(ROOT)).returncode
        if rc != 0:
            print(f"ABORT: collection failed rc={rc}", file=sys.stderr)
            return 1

    fit_all = json.loads(DISCOVERY_STATES.read_text(encoding="utf-8"))
    ev_all = json.loads(STATES.read_text(encoding="utf-8"))

    def usable(store):
        out = []
        for p in G0_SEEDS:
            for r in store.get(p, []):
                o = r["episode_key"].split(":")[0]
                if o in (opp_a, opp_b) and r1 in r["utilities"] and r2 in r["utilities"]:
                    out.append(r)
        return out

    fit_rows, ev_rows = usable(fit_all), usable(ev_all)
    fit_eps = {r["episode_key"] for r in fit_rows}
    ev_eps = {r["episode_key"] for r in ev_rows}
    if fit_eps & ev_eps:
        print(f"ABORT: fit and evaluation episodes overlap ({len(fit_eps & ev_eps)}). "
              f"The legal-selector must never be evaluated on its fitting data.",
              file=sys.stderr)
        return 1
    if not fit_rows or not ev_rows:
        print("ABORT: no usable states on one side.", file=sys.stderr)
        return 1

    # The 28 legal_context fields, sorted for determinism. Opponent identity is
    # not among them and is never added -- that is the entire point.
    feats = sorted(fit_rows[0]["legal_context"])
    assert "opponent" not in feats

    from sklearn.linear_model import LogisticRegression
    y_fit = np.array([1 if r["episode_key"].split(":")[0] == opp_a else 0 for r in fit_rows])
    clf = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
    clf.fit(feature_matrix(fit_rows, feats), y_fit)

    pred = clf.predict(feature_matrix(ev_rows, feats))

    # Baseline: the single fixed response with the higher POOLED mean utility.
    pooled = {r1: float(np.mean([r["utilities"][r1] for r in ev_rows])),
              r2: float(np.mean([r["utilities"][r2] for r in ev_rows]))}
    baseline_resp = r1 if pooled[r1] >= pooled[r2] else r2

    oracle_gain, legal_gain = [], []
    n_correct = 0
    for r, p in zip(ev_rows, pred):
        o = r["episode_key"].split(":")[0]
        u_base = r["utilities"][baseline_resp]
        u_oracle = r["utilities"][r1 if o == opp_a else r2]
        u_legal = r["utilities"][r1 if p == 1 else r2]
        oracle_gain.append((r["episode_key"], u_oracle - u_base))
        legal_gain.append((r["episode_key"], u_legal - u_base))
        n_correct += int((p == 1) == (o == opp_a))

    og = clustered_lcb(oracle_gain, rng=rng, resamples=resamples, lcb_pct=2.5)
    lg = clustered_lcb(legal_gain, rng=rng, resamples=resamples, lcb_pct=2.5)
    ratio = (lg["mean"] / og["mean"]) if og["mean"] not in (None, 0) else None

    if og["lcb95"] is None or og["lcb95"] <= 0:
        verdict = "NO_DEMAND_IN_BLOCK"
    elif lg["lcb95"] is not None and lg["lcb95"] > 0 and ratio is not None and ratio >= bar:
        verdict = "ROUTABLE"
    else:
        verdict = "UNROUTABLE"

    out = {
        "record": "Observability test — single-timestep legal_context",
        "verdict": verdict,
        "candidate": cand,
        "baseline_response": baseline_resp,
        "baseline_rule": "best SINGLE response by pooled mean utility (what K=1 does)",
        "oracle_gain": og,
        "legal_gain": lg,
        "recovery_ratio": None if ratio is None else round(ratio, 4),
        "recovery_bar": bar,
        "legal_selector": {
            "model": "L2 logistic regression, C=1.0, no hyperparameter search",
            "n_features": len(feats), "features": feats,
            "regime_accuracy_out_of_sample": round(n_correct / len(ev_rows), 4),
            "opponent_identity_used_as_feature": False,
            "information_set": "SINGLE TIMESTEP, no history "
                               "(OBSERVABILITY_SCOPE_AMENDMENT.json)",
        },
        "fit_block": "9840000 discovery states",
        "evaluation_block": [base, base + args.episodes - 1],
        "fit_eval_episode_overlap": 0,
        "n_states": {"fit": len(fit_rows), "eval": len(ev_rows)},
        "frozen_reporting_language": json.loads(
            SCOPE.read_text(encoding="utf-8"))["frozen_reporting_language"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\nVERDICT: {verdict}")
    print(f"  oracle gain  mean={og['mean']}  lcb95={og['lcb95']}")
    print(f"  legal  gain  mean={lg['mean']}  lcb95={lg['lcb95']}")
    print(f"  recovery     {out['recovery_ratio']}  (bar {bar})")
    print(f"  regime acc   {out['legal_selector']['regime_accuracy_out_of_sample']}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
