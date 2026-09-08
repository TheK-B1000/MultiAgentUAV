# Audit tool defects found and fixed — 2026-08-18

Found while auditing a wider block for a possible larger confirmation n. The
tool had **already certified two blocks** before these were caught, so both were
re-audited under the corrected version.

## Three defects

### 1. `.jsonl` and `.log` were never scanned — false NEGATIVE

`TEXT_EXT` omitted both. The repo contains 238 `.log` and 8 `.jsonl` files, and
that is exactly where run output lands. A seed recorded only in an episode-anchor
`.jsonl` or a run log would have been invisible to the working-tree scan.

This is the dangerous direction: it could hide a real collision.

### 2. `\b` matched digits inside decimals — false POSITIVE

`"carrier_dist_home": "0.6000145"` registered as the integer `6000145`, making
the unrelated commit `568a3152` (C3 Stage-4) look like it had used that seed.

Fixed to `(?<![\d.])(\d{6,12})(?!\d)`.

### 3. The history query used `--perl-regexp` and under-matched

`git log -G` rejects lookbehind outright:

```text
fatal: invalid regex: Invalid preceding regular expression
```

so the decimal-safe form had to be rewritten in ERE as
`(^|[^0-9.])(N|N|...)([^0-9]|$)`.

More importantly, the **original** `--perl-regexp` form was under-matching: on
the known-dirty positive control it returned **3** commits where the ERE form
returns **8**. Under-matching is the false-negative direction, so every earlier
"0 commits in history" result was weaker evidence than reported.

## What caught it

The **positive control**. When the lookbehind broke the query, the tool reported
`UNTRUSTED` rather than a false `CLEAN`, because the control block stopped
returning hits. A tool without a control would have silently reported the block
as clean.

The **four-check design** also worked as intended: the git-history check flagged
`568a3152`, which the working-tree scan had missed entirely because of defect 1.
Each check covered a blind spot in another.

## Re-verification of already-certified blocks

| block | status | result under corrected tool |
|---|---|---|
| `5000001..5000032` | **SPENT** on the SDS_G1_4 confirmation | Appears in 6 commits; the oldest is `45d81a52`, its own reservation. **Nothing predates it.** All working-tree hits are the confirmation's own `episode_rows.csv`, `run.log`, `summary.json`. Hygiene holds. |
| `6000001..6000064` | reserved | clean |
| `6000001..6000192` | reserved | **CLEAN** under the corrected tool, with the positive control firing (8 commits) |

No result is invalidated. The SDS_G1_4 confirmation's seed hygiene is confirmed
under a *stronger* query than the one that originally certified it.

## Also fixed

Directory exclusions were not globbing: `:(exclude,glob)path/to/dir/` matched
nothing, so a directory passed to `--exclude` was still searched in history.
Now expands to `dir/**` for directory prefixes, `pre*` for filename prefixes,
and exact otherwise.

## Standing lesson

An audit tool needs its own negative controls, not just a positive one. The
positive control proved the query could still detect a known hit; nothing proved
the *scan surface* was complete. The `.jsonl`/`.log` hole existed from the first
version and was found only by accident, when a decimal false positive in a
`.jsonl` forced an investigation into a file class the scanner never opened.
