# School PC — 6v6 locked pipeline

**Same algorithmic recipe as 4v4, scale-specific config** (`N=6`, `k=1`).
Do **not** start extra exploratory branches on this machine.

Full spec: [`../artifacts/strategic_demand/sppo/SCHOOL_PC_6V6_LOCKED_PIPELINE.json`](../artifacts/strategic_demand/sppo/SCHOOL_PC_6V6_LOCKED_PIPELINE.json)

---

## What this machine does (unattended)

1. Fail-closed **foundation preflight**
2. **π_A** entity-repair 1M (warm-start from existing base specialist)
3. Seal / hash A
4. **π_B** entity-repair 1M
5. Seal / hash B
6. Fail-closed **split preflight** (teacher / k=1 / hashes)
7. **200k** split: frozen ATTACK π_A + π_D from π_A + **same N′ teacher as 4v4** + `CLOSEST_DEFENDS(k=1)`
8. Exploratory crossover (n=64)
9. **Stop.** Confirmatory (n=128) is manual only if exploratory PASSes.

ETA ≈ **6–7 days** on one GPU (A∥B on two GPUs cuts the foundation roughly in half).

---

## One-time setup

From the repo checkout:

```powershell
cd <path-to>\MultiAgentUAV\AICTFProject
# venv must exist with project deps (same as home machine)
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

Confirm base specialists exist (hashes must match the SPEC):

```text
artifacts/scale_6v6_specialists/pi_A_specialist_6v6/ckpts/final_pi_A_specialist_6v6.zip
artifacts/scale_6v6_specialists/pi_B_specialist_6v6/ckpts/final_pi_B_specialist_6v6.zip
```

Optional (recommended once before leaving):

```powershell
.\.venv\Scripts\python.exe experiments/run_school_pc_6v6_preflight.py --stage foundation
```

Must print `PREFLIGHT PASS`.

---

## Start and leave

```powershell
cd <path-to>\MultiAgentUAV\AICTFProject
powershell -ExecutionPolicy Bypass -File experiments/launch_school_pc_6v6_detached.ps1
```

You can close that window. The job keeps running.
Pid: `artifacts/strategic_demand/sppo/school_pc_6v6_OVERALL.pid`

---

## Watch progress (tqdm)

**Whole pipeline ETA:**

```powershell
Get-Content artifacts\strategic_demand\sppo\school_pc_6v6_OVERALL.err -Wait -Tail 5
```

**Heartbeat JSON:**

```powershell
Get-Content artifacts\strategic_demand\sppo\school_pc_6v6_OVERALL_PROGRESS.json
```

**Per-stage bars** (PPO / eval):

```powershell
Get-Content artifacts\strategic_demand\sppo\school_pc_6v6_repair_A.err -Wait -Tail 3
Get-Content artifacts\strategic_demand\sppo\school_pc_6v6_repair_B.err -Wait -Tail 3
Get-Content artifacts\strategic_demand\sppo\school_pc_6v6_split_k1.err -Wait -Tail 3
Get-Content artifacts\strategic_demand\sppo\school_pc_6v6_crossover_exploratory.err -Wait -Tail 3
```

---

## Where results are saved

| What | Path |
|------|------|
| Logs, seals, overall bar | `artifacts/strategic_demand/sppo/` |
| Repaired π_A | `artifacts/scale_6v6_specialists/pi_A_specialist_6v6_c2_entity_repair/ckpts/` |
| Repaired π_B | `artifacts/scale_6v6_specialists/pi_B_specialist_6v6_c2_entity_repair/ckpts/` |
| Split π_D | `artifacts/scale_6v6_specialists/pi_A_specialist_6v6_split_defend_k1_v1/ckpts/` |
| Exploratory result | `artifacts/strategic_demand/EXPLORATORY_6V6_SPLIT_K1_SPECIALIST_CROSSOVER_EVAL_RESULT.json` |
| Done marker | `artifacts/strategic_demand/sppo/SCHOOL_PC_6V6_PIPELINE_DONE.json` |

---

## After exploratory finishes

Read the exploratory result. **PASS** only if `Δ_A>0`, `Δ_B>0`, and **both LCB95>0**.

If PASS, run confirmatory **manually** (do not invent a new experiment):

```powershell
.\.venv\Scripts\python.exe experiments/eval_specialist_crossover_scaled.py `
  --team-size 6 `
  --spec artifacts/strategic_demand/sppo/SCHOOL_PC_6V6_LOCKED_PIPELINE.json `
  --pi-a-path artifacts/scale_6v6_specialists/pi_A_specialist_6v6_split_defend_k1_v1/ckpts/final_pi_A_specialist_6v6_split_defend_k1_v1.zip `
  --pi-b-path artifacts/scale_6v6_specialists/pi_B_specialist_6v6_c2_entity_repair/ckpts/final_pi_B_specialist_6v6_c2_entity_repair.zip `
  --frozen-attack-path artifacts/scale_6v6_specialists/pi_A_specialist_6v6_c2_entity_repair/ckpts/final_pi_A_specialist_6v6_c2_entity_repair.zip `
  --role-fixed-for-episode --role-k-defend 1 `
  --seed-base 22800001 --n-seeds 128 `
  --label CONFIRMATORY_6V6_SPLIT_K1 `
  --device cuda
```

If confirmatory PASSes → **6v6 = DONE**.

---

## Do not

- Start k=1 vs k=3 mixture training as the paper system
- Use the old pre-entity specialists as the split foundation
- Turn off / retune the N′ teacher schedule
- Spend confirmatory seeds before exploratory PASS
- Launch random ablations “while we wait”

---

## Resume helpers (only if a stage already finished)

```powershell
# Repair done; continue from split preflight + 200k + eval
powershell -ExecutionPolicy Bypass -File experiments/launch_school_pc_6v6_detached.ps1 -SkipRepair

# Split done; eval only
powershell -ExecutionPolicy Bypass -File experiments/launch_school_pc_6v6_detached.ps1 -SkipRepair -SkipSplit
```

Foreground (not detached):

```powershell
.\.venv\Scripts\python.exe experiments/run_school_pc_6v6_locked_pipeline.py
```
