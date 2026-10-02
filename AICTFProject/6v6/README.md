# 6v6 — professor handoff

## What your professor does

1. Open `AICTFProject\6v6\`
2. Right-click **`FOR_PROFESSOR`**
3. **Compress to ZIP file**
4. Send that ZIP

No PowerShell. No packing script. No `artifacts/` tree.

**Rule: if it is not inside `FOR_PROFESSOR/`, the professor does not need it.**

`FOR_PROFESSOR/` is created when `run_dual_branch_6v6.py` finishes. Look for `FOR_PROFESSOR/READY_TO_ZIP.txt`.

```text
6v6/
├── START_HERE.txt              <- you are here
├── FOR_PROFESSOR/              <- ZIP THIS FOLDER
│   ├── START_HERE.txt
│   ├── README.txt
│   ├── READY_TO_ZIP.txt
│   ├── SUMMARY/
│   │   └── SUMMARY.txt
│   ├── TEACHERS/               Ours-Teachers: dual-branch DEFEND + ATTACK (A and B)
│   ├── STAGE3_EVALUATION/      top-50 four-cell diagnostic
│   ├── STAGE4_SHARING/         Strategic Representation Under Parameter Sharing:
│   │                             Share-Encoder / Ours-Shared Fully Shared+z+r / Role-only
│   ├── SEALS/
│   └── PROVENANCE/
├── dual_branch_OVERALL.log.err <- watch while running
└── run_dual_branch_6v6.ps1     <- school-PC launcher
```

## Launch (school PC)

```powershell
cd AICTFProject
.\.venv\Scripts\python.exe 6v6\run_dual_branch_6v6.py --check
powershell -ExecutionPolicy Bypass -File 6v6\run_dual_branch_6v6.ps1
```

Watch:

```powershell
Get-Content 6v6\dual_branch_OVERALL.log.err -Wait -Tail 5
```

## Do not confuse with

- `6v6/models|results|seals` — older Sept split-k=1 handoff, not the current dual-branch package
- `run_symmetric_6v6.ps1` — defender-only ablation; not the main send package
