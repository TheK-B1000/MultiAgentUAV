# 4v4 dual-branch: paused before Stage 4 (for PI review)

**Status:** PAUSED_FOR_PI_REVIEW. The 4v4 driver is suspended after matched-128 sealed; Stage 4 has not started. Nothing frozen was changed.

4v4 Stage 3 produced a negative result under the frozen symmetric protocol. The B side collapsed during training, yielding negative Pole-B crossover separation. Stage 4 was paused only to avoid spending compute on distillation from an invalid teacher pair while the PI reviews the failure mode. Nothing frozen was changed.

## Stage-3 matched-128 (post-hoc, historical block 21800001..128; sealed, 512 episodes)

| System | A@A | B@A | A@B | B@B | Δ_A [95% CI] | Δ_B [95% CI] |
|---|---|---|---|---|---|---|
| Dual-branch (k=2) | 0.578 | 0.383 | 0.109 | 0.023 | +0.195 [+0.070, +0.320] | **−0.086 [−0.148, −0.031]** |
| Old asymmetric Ours (same seeds) | 0.648 | 0.492 | 0.102 | 0.508 | +0.156 [+0.039, +0.273] | +0.406 [+0.297, +0.508] |

Paired dual-branch minus old: Δ_A +0.039 [−0.133, +0.211]; Δ_B −0.492 [−0.609, −0.375]. Score margin tells the same story (Δ_B −0.64 vs +1.39).
Integrity flag (Δ_B ≤ 0) → row-level audit **PASS** (512 rows, exact seed block, 0 consistency violations). B on its own pole: 3 wins / 128, scored in 11 / 128.

## When it failed (training, read-only; see TRAINING_DIAGNOSTIC.md)

- 4v4 B (vs OP7): own-pole win 0.62 (0-20k) -> 0.41 (20-40k) -> 0.10 (40-60k), then 0.01-0.08 through 200k
- the failure is mainly scoring: own goals per episode 1.18 -> 0.13 by 40-60k; goals conceded rise only modestly in training (0.10 -> ~0.2)
- collapse begins at 20-60k while the DEFEND-teacher weight is still at its peak (0.1, before decay starts at 50k); no recovery in the 50k steps after the weight reaches 0 (150k)
- 2v2 B: same collapse (win 0.62 -> 0.00 by 80-100k) and recovery only after the weight reached 0 (0.34 at 160-180k, 0.50 at 180-200k)
- A sides under the same scaffold schedule do not collapse at either scale (4v4 A ends 0.73, 2v2 A 0.62)
- DEFEND-teacher weight: 0.1 until 50k, linear to 0 at 150k. Not in the logs: per-update teacher loss, per-update DEFEND share (k = 2 fixed by construction).

## Freeze state

- Driver pids [32912, 10348] suspended; completed steps: phase0_preflight, phase1_smoke_A, phase1_smoke_B, phase1_train, phase1_export, phase1_technical_seal.
- at ~05:52Z the driver was accidentally un-suspended for a few seconds by a mistaken command and re-suspended at once; it was blocked waiting on the matched-128 subprocess, so it advanced no phase (driver log unchanged)
- a reboot ends the suspension; relaunching 4v4/run_dual_branch_4v4.ps1 would resume and proceed to Stage 4 -- do not relaunch until the PI decides
- If the PI says continue as frozen: `python artifacts/strategic_demand/sppo/dual_branch_v1/matched128_2v2/hold_4v4.py resume (same frozen teachers, same seeds)`

## Proposed next step (not authorized; needs PI approval and a frozen spec)

**R1:** k = max(1, round(N/3)) — 2v2 1→1, **4v4 2→1**, 6v6 2→2. Retrain both 4v4 A and B from the sealed 1M specialists, 200k, everything else identical; confirm on a fresh 128-seed block. R2 (longer symmetric recovery) and R3 (weaker scaffold for both) only if needed; adaptive allocation later.
