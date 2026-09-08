# The shared run log is CONTAMINATED — do not parse it as a result

`artifacts/strategic_demand/sappo_representability_run.log` contains
**interleaved output from two different probe invocations**:

* an obsolete 3-arm run (started 18:23), whose kill silently failed
* the corrected 4-arm run (started 18:35)

Concrete evidence of the interleaving, both present in the same file:

```text
pole A seed 7900002: TRUE=1 PROJ=0 OPP=0 PROJ_OPP=1     <- corrected 4-arm
pole A seed 7900007: TRUE=1 PROJ=1 OPP=1                <- stale 3-arm
```

The file is retained for provenance only. **Do not reconstruct results from it.**

## Authoritative source

`summary.json` in this directory, and only if it is attributable to the
corrected run. Attribution signals, all four of which must hold:

1. `PROJECTED_OPPOSITE` present in every pole's results — the 3-arm version
   structurally cannot emit this key
2. `n == 32` — the smoke test used 3
3. modification time after 18:35
4. it appeared in this directory *after* the directory was verified to contain
   no summary at all

`SMOKE_TEST_n3_NOT_A_RESULT.json` is a 3-seed CPU smoke test, quarantined under
that name precisely so it can never be mistaken for a result.

Later invocations additionally carry a `run_identity` block with the invocation
string and its hash, so attribution no longer depends on inference.

## Why this happened

Two probes were launched against the same output paths because a `taskkill`
reported success on a redirector stub while the real worker survived. Process
kills in this environment must be verified by re-listing, not trusted from the
kill command's own exit status.
