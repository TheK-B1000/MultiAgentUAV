# 6v6 — send this folder to your professor

**One place for everything 6v6 related to the current school-PC run.**

| What to send | Path |
|---|---|
| **Finished package (preferred)** | `6v6/dual_branch_6v6_results.zip` |
| Unpacked copy of the same | `6v6/FOR_PROFESSOR/` |
| While the run is still going | watch `6v6/dual_branch_OVERALL.log.err` |

Pack command (from `AICTFProject`):

```powershell
powershell -ExecutionPolicy Bypass -File experiments/pack_6v6_handoff.ps1
```

That writes a dated zip under `6v6/` from `FOR_PROFESSOR/` (or the dual-branch bundle if present).

---

## Current pipeline (what the school PC is running)

Launcher: `6v6/run_dual_branch_6v6.ps1`  
Spec: dual-branch ATTACK+DEFEND teachers → top-50 diagnostic → Stage-4 sharing ladder.

When the run finishes, `FOR_PROFESSOR/` contains:

```text
FOR_PROFESSOR/
  README.txt
  checkpoints/A/   dual-branch DEFEND + exported ATTACK (Pole A)
  checkpoints/B/   dual-branch DEFEND + exported ATTACK (Pole B)
  evaluation/      crossover RESULT / AUDIT / rows / STATE
  seals/           technical seal + Stage-4 teacher seal + deploy manifest
  stage4/          Share-Encoder, Fully Shared+z+r, Role-only students + dataset manifest
  progress/        overall ETA snapshot
```

---

## Where files live on disk (canonical vs handoff)

| Content | Canonical training location (large) | Copied into handoff |
|---|---|---|
| Dual-branch PPO runs | `artifacts/scale_6v6_specialists/pi_*_dual_branch_v1/` | checkpoints in `FOR_PROFESSOR/` |
| Stage-4 students | `artifacts/strategic_demand/sppo/suite_sharing_std/6v6_stage4/` | `FOR_PROFESSOR/stage4/` |
| Sealed eval JSON/CSV | `artifacts/strategic_demand/sppo/DUAL_BRANCH_*` and `TOP50_6V6_STAGE4*` | `FOR_PROFESSOR/evaluation/` |
| Suite progress / logs | `6v6/dual_branch_*.log(.err)`, `dual_branch_OVERALL*` | `FOR_PROFESSOR/progress/` + logs stay in `6v6/` |
| Historical split-k=1 repair handoff (Sept) | also under `6v6/models|results|seals/` | **old**; not the current dual-branch package |

Professor does **not** need the full `artifacts/` tree. Send the zip.

---

## Watch progress

```powershell
Get-Content 6v6\dual_branch_OVERALL.log.err -Wait -Tail 5
Get-Content 6v6\dual_branch_OVERALL_PROGRESS.json
Get-Content 6v6\dual_branch_<stage>.log.err -Wait -Tail 3   # e.g. train_A, eval, stage4_share_encoder_train
```

---

## Do not confuse with

- `run_symmetric_6v6.ps1` — older defender-only ablation (refuses unless explicitly allowed).
- `6v6/models/final_pi_*_repair.zip` — earlier school-PC split-k=1 handoff, not dual-branch V1.
