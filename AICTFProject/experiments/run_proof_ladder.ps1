<#
.SYNOPSIS
    Train + evaluate the plan-faithful proof ladder (A..E [+ optional F]) end to end.

.DESCRIPTION
    Locks the Summer Plan claim "discrete latent team strategy, learned without
    phase labels or handcrafted strategy supervision, improves coordination and
    win rate" with a five-row plan-faithful ladder. K=1 (row B) is the dagger
    test against K=4 (row C): if C > B fails, the latent is not strategy
    discovery, just extra conditioning capacity.

    Rows:
      A. plan_faithful_no_latent                   (baseline)
      B. plan_faithful_latent_k1                   (DAGGER: K=1 collapsed latent)
      C. plan_faithful_latent_persist_entropy      (PRIMARY: K=4 + persist + entropy)
      D. plan_faithful_latent_no_persistence       (ablation: lambda_p = 0)
      E. plan_faithful_latent_no_entropy           (ablation: lambda_H = 0, entropy off)
      F. plan_faithful_latent_option_a   (OPTIONAL, -IncludeOptionA): Fix D / Option A
                                                   episode-start z, no persistence,
                                                   no aux heads.

    For each row this script will:
      1. (preflight)  Verify the --preset resolves without error.
      2. (train)      Skip if final_<tag>.zip already exists unless -Force.
                      Otherwise train at 1M steps with --opponent-randomize over
                      --OpponentPool. Logs to logs\proof_ladder_<ts>\<tag>.train.log.
      3. (eval)       Multi-opponent zero-shot eval via experiments/eval_op4_zero_shot.py.
      4. (mi)         Per-row MI(z;opponent) heatmap via experiments/analyze_latent_mi.py.
      5. (proof)      Joint proof table via experiments/build_proof_table.py.

    Stages are individually skippable (-SkipTrain / -SkipEval / -SkipMI /
    -SkipProof) and per-row failures do not abort the chain - failed rows are
    reported in the final summary.

.PARAMETER Seeds
    One or more integer seeds. Default: single seed 0 (matches the existing
    hardpool runs in checkpoints\4v4\*_run_config.json).

.PARAMETER Steps
    --total-steps for each training run. Default 1_000_000 (~30-90 min per row on
    a single CUDA GPU at n-envs 32).

.PARAMETER Agents
    --agents (team size per side). Default 4 to match your hardpool_4v4 runs.

.PARAMETER NEnvs
    --n-envs. Default 32 (matches your prior 4v4 hardpool runs).

.PARAMETER NEpochs
    --n-epochs. Default 6 (matches your prior 4v4 hardpool runs).

.PARAMETER Device
    --device (cuda / cpu). Default cuda.

.PARAMETER OpponentPool
    --opponent-pool (training mix). Default OP3,OP5,OP6 (matches "hardpool").

.PARAMETER CheckpointDir
    --checkpoint-dir for training, --checkpoint-dir for eval / MI / proof.
    Default checkpoints\4v4.

.PARAMETER TagSuffix
    Suffix appended after the per-row tag stem. Default 'hardpool_1m_4v4' so
    new rows slot into your existing eval / proof pipeline. Set e.g.
    'hardpool_seed1_1m_4v4' to run a second seed without colliding with
    existing checkpoints.

.PARAMETER EvalEpisodes
    Per-opponent eval episode count. Default 200 (matches your terminal-5 chain).

.PARAMETER EvalOpponents
    Eval-time opponents. Default OP3,OP4,OP5_RUSHER,OP6_TURTLE.

.PARAMETER LatentEvalModes
    For each latent row, run eval once per mode in this list. Modes:
        normal           - q_phi(z|s)                   (trained behavior)
        uniform_random   - z ~ Uniform({0..K-1})         (destroys q_phi entirely)
        shuffled         - z ~ marginal P(z)             (preserves P(z), destroys state-cond)
    The marginal for 'shuffled' is auto-computed per row from that row's normal-mode OP3 CSV
    (via experiments/_z_marginal_from_csv.py), so 'shuffled' only runs if 'normal' also runs
    (or 'normal' was already cached from a prior chain invocation). No-latent rows are always
    evaluated in 'normal' only (destruction modes are no-ops on a no-latent policy).
    Default: normal,uniform_random,shuffled (the three modes the proof gates need).

.PARAMETER IncludeOptionA
    Add row F (plan_faithful_latent_option_a, a.k.a. Fix D).

.PARAMETER Force
    Re-train rows even if final_<tag>.zip already exists.

.PARAMETER SkipTrain / -SkipEval / -SkipMI / -SkipProof
    Skip the named stage. Useful for resumed / partial reruns.

.PARAMETER DryRun
    Print every command that would be run; execute nothing.

.EXAMPLE
    # All-day default: train rows B/C/D/E (row A already trained), eval everything,
    # MI on latent rows, then proof table. Skips A's training automatically.
    .\experiments\run_proof_ladder.ps1

.EXAMPLE
    # Add Fix D / Option A as a sixth row.
    .\experiments\run_proof_ladder.ps1 -IncludeOptionA

.EXAMPLE
    # Three-seed all-night run (no Option A).
    .\experiments\run_proof_ladder.ps1 -Seeds 0,1,2 -TagSuffix hardpool_1m_4v4_multiseed

.EXAMPLE
    # Re-run only the eval + MI + proof stages on existing checkpoints.
    .\experiments\run_proof_ladder.ps1 -SkipTrain
#>

[CmdletBinding()]
param(
    [int[]]$Seeds = @(0),
    [int]$Steps = 1000000,
    [int]$Agents = 4,
    [int]$NEnvs = 32,
    [int]$NEpochs = 6,
    [string]$Device = "cuda",
    [string[]]$OpponentPool = @("OP3", "OP5", "OP6"),
    [string]$CheckpointDir = "checkpoints\4v4",
    [string]$TagSuffix = "hardpool_1m_4v4",
    [int]$EvalEpisodes = 200,
    [string[]]$EvalOpponents = @("OP3", "OP4", "OP5_RUSHER", "OP6_TURTLE"),
    [string[]]$LatentEvalModes = @("normal", "uniform_random", "shuffled"),
    [switch]$IncludeOptionA,
    [switch]$Force,
    [switch]$SkipTrain,
    [switch]$SkipEval,
    [switch]$SkipMI,
    [switch]$SkipProof,
    [switch]$DryRun
)

$ErrorActionPreference = "Continue"

$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) { $python = "python" }

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir    = Join-Path $projectRoot ("logs\proof_ladder_" + $timestamp)
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$chainLog  = Join-Path $logDir "chain.log"
$timingCsv = Join-Path $logDir "stage_timings.csv"
"label,started_at,elapsed_seconds,status,log_path" | Set-Content -LiteralPath $timingCsv -Encoding utf8

# Global chain stopwatch (printed at every stage completion as a running total)
# and a tally of (label -> elapsed_seconds) for the final summary block.
$ChainStopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$StageTimings   = [System.Collections.Generic.List[psobject]]::new()

function Format-Elapsed {
    param([TimeSpan]$Span)
    if ($Span.TotalHours -ge 1) {
        return ("{0}h{1:00}m{2:00}s" -f [int]$Span.TotalHours, $Span.Minutes, $Span.Seconds)
    } elseif ($Span.TotalMinutes -ge 1) {
        return ("{0}m{1:00}s" -f [int]$Span.TotalMinutes, $Span.Seconds)
    } else {
        return ("{0:0.0}s" -f $Span.TotalSeconds)
    }
}

function Write-Chain {
    param([string]$Message, [string]$Color = "Gray")
    $stamp = Get-Date -Format "HH:mm:ss"
    $line  = "[" + $stamp + "] " + $Message
    Write-Host $line -ForegroundColor $Color
    Add-Content -LiteralPath $chainLog -Value $line -Encoding utf8
}

function Invoke-Stage {
    param(
        [string]$Label,
        [string]$LogPath,
        [string[]]$Cmd
    )
    Write-Chain ("[" + $Label + "] " + ($Cmd -join " "))
    if ($DryRun) {
        Write-Chain ("[" + $Label + "] DRY-RUN (skipped, 0s)") "DarkGray"
        $StageTimings.Add([pscustomobject]@{ Label = $Label; ElapsedSeconds = 0.0; Status = "dryrun"; LogPath = $LogPath })
        ("{0},{1},{2:0.00},{3},{4}" -f $Label, (Get-Date -Format "o"), 0.0, "dryrun", $LogPath) | Add-Content -LiteralPath $timingCsv -Encoding utf8
        return $true
    }
    $stageStartIso = Get-Date -Format "o"
    Write-Chain ("[" + $Label + "] started (chain elapsed: " + (Format-Elapsed $ChainStopwatch.Elapsed) + ")") "DarkCyan"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    # Pipe Tee-Object output to Out-Host so the function's pipeline return is JUST $ok,
    # not the merged stream. Tee still writes the full transcript to $LogPath.
    & $Cmd[0] @($Cmd[1..($Cmd.Length - 1)]) *>&1 | Tee-Object -FilePath $LogPath -Append | Out-Host
    $ok = ($LASTEXITCODE -eq 0)
    $sw.Stop()
    $elapsed = Format-Elapsed $sw.Elapsed
    $chainElapsed = Format-Elapsed $ChainStopwatch.Elapsed
    $status = if ($ok) { "ok" } else { "fail" }
    $StageTimings.Add([pscustomobject]@{
        Label          = $Label
        ElapsedSeconds = [Math]::Round($sw.Elapsed.TotalSeconds, 2)
        Status         = $status
        LogPath        = $LogPath
    })
    ("{0},{1},{2:0.00},{3},{4}" -f $Label, $stageStartIso, $sw.Elapsed.TotalSeconds, $status, $LogPath) | Add-Content -LiteralPath $timingCsv -Encoding utf8
    if (-not $ok) {
        Write-Chain ("[" + $Label + "] FAILED in " + $elapsed + " (exit=" + $LASTEXITCODE + "; chain elapsed: " + $chainElapsed + "); see " + $LogPath) "Red"
    } else {
        Write-Chain ("[" + $Label + "] OK in " + $elapsed + " (chain elapsed: " + $chainElapsed + ")") "Green"
    }
    return $ok
}

# Five-row ladder + optional Fix D. (preset -> stem of run_tag).
$rows = @(
    [pscustomobject]@{ Letter = "A"; Preset = "plan_faithful_no_latent";              TagStem = "plan_faithful_no_latent";              Latent = $false; Description = "no latent (baseline)" }
    [pscustomobject]@{ Letter = "B"; Preset = "plan_faithful_latent_k1";              TagStem = "plan_faithful_latent_k1";              Latent = $true;  Description = "K=1 collapsed latent (DAGGER)" }
    [pscustomobject]@{ Letter = "C"; Preset = "plan_faithful_latent_persist_entropy"; TagStem = "plan_faithful_latent_persist_entropy"; Latent = $true;  Description = "K=4 + persistence + entropy (PRIMARY)" }
    [pscustomobject]@{ Letter = "D"; Preset = "plan_faithful_latent_no_persistence";  TagStem = "plan_faithful_latent_no_persistence";  Latent = $true;  Description = "K=4, lambda_p = 0 (ablation)" }
    [pscustomobject]@{ Letter = "E"; Preset = "plan_faithful_latent_no_entropy";      TagStem = "plan_faithful_latent_no_entropy";      Latent = $true;  Description = "K=4, lambda_H = 0 / entropy off (ablation)" }
)
if ($IncludeOptionA) {
    $rows += [pscustomobject]@{
        Letter      = "F"
        Preset      = "plan_faithful_latent_option_a"
        TagStem     = "plan_faithful_latent_option_a"
        Latent      = $true
        Description = "K=4, episode-start z, lambda_p = 0, lambda_H = 0.001 (Fix D / Option A)"
    }
}

# Build seeded plan: one job per (row, seed). If only one seed and TagSuffix
# already implies hardpool_1m_4v4, we keep tags identical to existing files.
$plan = @()
foreach ($seed in $Seeds) {
    foreach ($row in $rows) {
        if ($Seeds.Count -le 1) {
            $runTag = $row.TagStem + "_" + $TagSuffix
        } else {
            $runTag = $row.TagStem + "_seed" + $seed + "_" + $TagSuffix
        }
        $finalCkpt = Join-Path $CheckpointDir ("final_" + $runTag + ".zip")
        $plan += [pscustomobject]@{
            Letter         = $row.Letter
            Preset         = $row.Preset
            RunTag         = $runTag
            Seed           = $seed
            Latent         = $row.Latent
            Description    = $row.Description
            FinalCheckpoint = $finalCkpt
            TrainOK        = $null
            EvalOK         = @{}   # mode -> bool ("normal"/"uniform_random"/"shuffled" for latent rows; "normal" only for no-latent)
            MiOK           = $null
        }
    }
}

# Normalize and validate the latent-mode list. "normal" is forced to the front because
# shuffled depends on the normal CSV existing first (we read its strategy_occupancy_* columns
# to estimate the marginal). uniform_random is independent and can be in any position.
$validModes = @("normal", "uniform_random", "shuffled")
$wantModesRaw = @()
foreach ($m in $LatentEvalModes) {
    $mLower = $m.ToLower().Trim()
    if ($validModes -notcontains $mLower) {
        Write-Host ("[fatal] -LatentEvalModes contains unknown mode '" + $m + "'; valid: " + ($validModes -join ",")) -ForegroundColor Red
        exit 2
    }
    if ($wantModesRaw -notcontains $mLower) { $wantModesRaw += $mLower }
}
if ($wantModesRaw.Count -eq 0) { $wantModesRaw = @("normal") }
# If shuffled is requested but normal is not, we still need normal first to produce the
# marginal source CSV. Insert it implicitly.
if (($wantModesRaw -contains "shuffled") -and ($wantModesRaw -notcontains "normal")) {
    $wantModesRaw = @("normal") + $wantModesRaw
}
# Reorder so normal is first.
$LatentEvalModesOrdered = @("normal") + ($wantModesRaw | Where-Object { $_ -ne "normal" })
$LatentEvalModesOrdered = $LatentEvalModesOrdered | Where-Object { $wantModesRaw -contains $_ }

# ----------------------------- Preflight --------------------------------------
Write-Chain ("project root: " + $projectRoot) "Cyan"
Write-Chain ("python:       " + $python) "Cyan"
Write-Chain ("log dir:      " + $logDir) "Cyan"
Write-Chain ("rows:         " + ($plan.Count) + "   seeds=" + ($Seeds -join ",") + "   include_option_a=" + $IncludeOptionA) "Cyan"
Write-Chain ("opp pool:     " + ($OpponentPool -join " ")) "Cyan"
Write-Chain ("eval opps:    " + ($EvalOpponents -join " ")) "Cyan"
Write-Chain ("latent modes: " + ($LatentEvalModesOrdered -join ",") + "   (no-latent rows always eval normal-only)") "Cyan"

Write-Chain "" "Gray"
Write-Chain "[preflight] verify every --preset resolves" "Yellow"
$presetCheckSrc = @"
import sys, os
sys.path.insert(0, os.getcwd())
from rl.train_ppo import PPOConfig, _apply_training_preset
for name in sys.argv[1:]:
    cfg = _apply_training_preset(PPOConfig(), name)
    print(f'OK {name:<48} K={cfg.latent_k} resample_every={cfg.latent_resample_every_n} lam_p={cfg.latent_lam_p} lam_h={cfg.latent_lam_h} aux_phase={cfg.latent_strategy_aux_predict_phase_coef} aux_ret={cfg.latent_strategy_aux_return_head}')
"@
$presetCheckPath = Join-Path $logDir "_preflight_resolve_presets.py"
Set-Content -LiteralPath $presetCheckPath -Value $presetCheckSrc -Encoding utf8
$uniquePresets = $plan | Select-Object -ExpandProperty Preset -Unique
$preflightCmd  = @($python, $presetCheckPath) + $uniquePresets
$preflightOK   = Invoke-Stage "preflight" (Join-Path $logDir "preflight.log") $preflightCmd
if (-not $preflightOK -and -not $DryRun) {
    Write-Chain "[preflight] aborting chain (one or more presets failed to resolve)" "Red"
    exit 1
}

# ----------------------------- Train ------------------------------------------
if (-not $SkipTrain) {
    foreach ($job in $plan) {
        Write-Chain "" "Gray"
        Write-Chain ("[" + $job.Letter + " train] " + $job.RunTag + "   (" + $job.Description + ")") "Yellow"
        if ((-not $Force) -and (Test-Path $job.FinalCheckpoint)) {
            Write-Chain ("[" + $job.Letter + " train] SKIP (final checkpoint exists: " + $job.FinalCheckpoint + ")") "DarkGray"
            $job.TrainOK = $true
            continue
        }
        $trainCmd = @(
            $python, "-u", "rl/train_ppo.py",
            "--preset", $job.Preset,
            "--agents", $Agents,
            "--seed", $job.Seed,
            "--total-steps", $Steps,
            "--device", $Device,
            "--n-envs", $NEnvs,
            "--n-epochs", $NEpochs,
            "--opponent-randomize",
            "--opponent-pool"
        ) + $OpponentPool + @(
            "--e3-step-telemetry",
            "--checkpoint-dir", $CheckpointDir,
            "--run-tag", $job.RunTag
        )
        $stageLog = Join-Path $logDir ($job.Letter + "_train_" + $job.RunTag + ".log")
        $job.TrainOK = Invoke-Stage ($job.Letter + " train") $stageLog $trainCmd
    }
} else {
    Write-Chain "[train] -SkipTrain set; checking which checkpoints already exist" "DarkGray"
    foreach ($job in $plan) {
        $job.TrainOK = (Test-Path $job.FinalCheckpoint)
        if (-not $job.TrainOK) {
            Write-Chain ("[" + $job.Letter + " train] MISSING checkpoint: " + $job.FinalCheckpoint) "Red"
        }
    }
}

# ----------------------------- Eval -------------------------------------------
#
# For each row, eval once per latent-mode in $LatentEvalModesOrdered. The mode is appended
# to the run_tag in op4_zero_shot_comparison.csv as "__<mode>" by eval_op4_zero_shot.py.
# 'shuffled' needs a marginal P(z) — we compute it from the row's own normal-mode OP3 CSV
# right after the normal-mode eval finishes (so this works in both fresh-run and resume modes).
$evalOutDir = Join-Path $CheckpointDir "eval_op4_zero_shot"
if (-not $SkipEval) {
    foreach ($job in $plan) {
        Write-Chain "" "Gray"
        Write-Chain ("[" + $job.Letter + " eval]  " + $job.RunTag) "Yellow"
        if (-not (Test-Path $job.FinalCheckpoint)) {
            Write-Chain ("[" + $job.Letter + " eval]  SKIP (no checkpoint at " + $job.FinalCheckpoint + ")") "DarkGray"
            foreach ($m in $LatentEvalModesOrdered) { $job.EvalOK[$m] = $false }
            continue
        }

        # No-latent rows: only ever run normal mode (destruction modes are no-ops there).
        $modesForRow = if ($job.Latent) { $LatentEvalModesOrdered } else { @("normal") }

        foreach ($mode in $modesForRow) {
            $modeLabel = if ($mode -eq "normal") { "eval " } else { "eval-" + $mode.Substring(0, [Math]::Min(4, $mode.Length)) }
            $extraArgs = @("--latent-mode", $mode)

            if ($mode -eq "shuffled") {
                # Estimate marginal P(z) from this row's normal-mode OP3 eval CSV. The
                # normal-mode pass above (or a prior chain invocation that left the CSV
                # behind) creates eval_<run_tag>_OP3_<N>ep.csv inside $evalOutDir.
                $marginalSrc = Join-Path $evalOutDir ("eval_" + $job.RunTag + "_OP3_" + $EvalEpisodes + "ep.csv")
                if (-not (Test-Path $marginalSrc)) {
                    Write-Chain ("[" + $job.Letter + " " + $modeLabel + "] SKIP (no normal-mode CSV for marginal at " + $marginalSrc + ")") "DarkGray"
                    $job.EvalOK[$mode] = $false
                    continue
                }
                $marginal = (& $python (Join-Path $projectRoot "experiments\_z_marginal_from_csv.py") $marginalSrc --k 4 2>$null | Select-Object -Last 1).Trim()
                if (-not $marginal -or $marginal -notmatch ',') {
                    Write-Chain ("[" + $job.Letter + " " + $modeLabel + "] SKIP (could not compute marginal from " + $marginalSrc + ")") "DarkGray"
                    $job.EvalOK[$mode] = $false
                    continue
                }
                Write-Chain ("[" + $job.Letter + " " + $modeLabel + "] P(z) = " + $marginal) "Gray"
                $extraArgs += @("--latent-marginal", $marginal)
            }

            $evalCmd = @(
                $python, "-u", "experiments/eval_op4_zero_shot.py",
                "--checkpoint-dir", $CheckpointDir,
                "--agents", $Agents,
                "--device", $Device,
                "--episodes", $EvalEpisodes,
                "--map-set", "eval",
                "--opponents"
            ) + $EvalOpponents + @(
                "--run-tags", $job.RunTag
            ) + $extraArgs
            $stageLog = Join-Path $logDir ($job.Letter + "_eval_" + $mode + "_" + $job.RunTag + ".log")
            $job.EvalOK[$mode] = Invoke-Stage ($job.Letter + " " + $modeLabel) $stageLog $evalCmd
        }
    }
} else {
    Write-Chain "[eval] -SkipEval set; skipping" "DarkGray"
}

# ----------------------------- MI ---------------------------------------------
if (-not $SkipMI) {
    $latentTags = @()
    foreach ($job in $plan) {
        if ($job.Latent -and (Test-Path $job.FinalCheckpoint)) { $latentTags += $job.RunTag }
    }
    if ($latentTags.Count -gt 0) {
        Write-Chain "" "Gray"
        Write-Chain ("[MI]   latent rows: " + ($latentTags -join " ")) "Yellow"
        $miCmd = @(
            $python, "-u", "experiments/analyze_latent_mi.py",
            "--checkpoint-dir", $CheckpointDir,
            "--run-tags"
        ) + $latentTags + @("--plots")
        $stageLog = Join-Path $logDir "mi.log"
        $miOK = Invoke-Stage "MI    " $stageLog $miCmd
        foreach ($job in $plan) {
            if ($job.Latent) { $job.MiOK = $miOK }
        }
    } else {
        Write-Chain "[MI] no latent checkpoints available; skipping" "DarkGray"
    }
} else {
    Write-Chain "[MI] -SkipMI set; skipping" "DarkGray"
}

# ----------------------------- Proof table ------------------------------------
$proofOK = $true
if (-not $SkipProof) {
    $allTags = @()
    foreach ($job in $plan) {
        if (Test-Path $job.FinalCheckpoint) { $allTags += $job.RunTag }
    }
    if ($allTags.Count -gt 0) {
        Write-Chain "" "Gray"
        Write-Chain ("[proof] joint proof table over: " + ($allTags -join " ")) "Yellow"
        $proofCmd = @(
            $python, "-u", "experiments/build_proof_table.py",
            "--checkpoint-dir", $CheckpointDir,
            "--run-tags"
        ) + $allTags
        $stageLog = Join-Path $logDir "proof.log"
        $proofOK = Invoke-Stage "proof " $stageLog $proofCmd
    } else {
        Write-Chain "[proof] no checkpoints available; skipping" "DarkGray"
        $proofOK = $false
    }
} else {
    Write-Chain "[proof] -SkipProof set; skipping" "DarkGray"
}

# ----------------------------- Summary ----------------------------------------
Write-Chain "" "Gray"
Write-Chain "============================== SUMMARY ==============================" "Cyan"
Write-Chain ("log dir: " + $logDir) "Cyan"
# Per-mode eval columns: normal / uniform_random / shuffled (only those actually requested)
$modeHeaders = $LatentEvalModesOrdered | ForEach-Object {
    switch ($_) {
        "normal"         { "ev:norm" }
        "uniform_random" { "ev:rand" }
        "shuffled"       { "ev:shuf" }
        default          { "ev:" + $_ }
    }
}
$modeHeaderCells = ($modeHeaders | ForEach-Object { "{0,-8}" -f $_ }) -join " "
$modeDashCells   = ($modeHeaders | ForEach-Object { "{0,-8}" -f "----" }) -join " "
$fmtHead   = "{0,-3} {1,-8} {2,-58} {3,-6}  {4} {5,-4}"
$fmtRow    = "{0,-3} {1,-8} {2,-58} {3,-6}  {4} {5,-4}"
Write-Chain ($fmtHead -f "row","seed","run_tag","train",$modeHeaderCells,"MI") "White"
Write-Chain ($fmtHead -f "---","----","-------","-----",$modeDashCells,"--") "White"
foreach ($job in $plan) {
    $t = if ($null -eq $job.TrainOK) { "-" } elseif ($job.TrainOK) { "OK" } else { "FAIL" }
    $m = if ($null -eq $job.MiOK)    { "-" } elseif ($job.MiOK)    { "OK" } else { "FAIL" }
    $modeCells = @()
    $anyEvalFail = $false
    foreach ($mode in $LatentEvalModesOrdered) {
        if (-not $job.Latent -and $mode -ne "normal") {
            $modeCells += "{0,-8}" -f "n/a"
        } elseif (-not $job.EvalOK.ContainsKey($mode)) {
            $modeCells += "{0,-8}" -f "-"
        } elseif ($job.EvalOK[$mode]) {
            $modeCells += "{0,-8}" -f "OK"
        } else {
            $modeCells += "{0,-8}" -f "FAIL"
            $anyEvalFail = $true
        }
    }
    $color = if (($t -eq "FAIL") -or $anyEvalFail -or ($m -eq "FAIL")) { "Red" } else { "Green" }
    Write-Chain ($fmtRow -f $job.Letter, $job.Seed, $job.RunTag, $t, ($modeCells -join " "), $m) $color
}
if ($SkipProof) {
    $proofStr = "skipped"
    $proofColor = "DarkGray"
} elseif ($proofOK) {
    $proofStr = "OK"
    $proofColor = "Green"
} else {
    $proofStr = "FAIL"
    $proofColor = "Red"
}
Write-Chain ("proof table: " + $proofStr) $proofColor
Write-Chain "=====================================================================" "Cyan"

# Per-stage wall-clock breakdown.
$ChainStopwatch.Stop()
Write-Chain "" "Gray"
Write-Chain "============================== TIMINGS ==============================" "Cyan"
Write-Chain ("{0,-46} {1,-6} {2,12}" -f "stage", "status", "elapsed") "White"
Write-Chain ("{0,-46} {1,-6} {2,12}" -f "-----", "------", "-------") "White"
$totalSecondsByStatus = @{ ok = 0.0; fail = 0.0; dryrun = 0.0 }
foreach ($t in $StageTimings) {
    $span = [TimeSpan]::FromSeconds($t.ElapsedSeconds)
    $color = if ($t.Status -eq "fail") { "Red" } elseif ($t.Status -eq "dryrun") { "DarkGray" } else { "Green" }
    Write-Chain ("{0,-46} {1,-6} {2,12}" -f $t.Label, $t.Status, (Format-Elapsed $span)) $color
    if ($totalSecondsByStatus.ContainsKey($t.Status)) {
        $totalSecondsByStatus[$t.Status] += [double]$t.ElapsedSeconds
    }
}
Write-Chain ("{0,-46} {1,-6} {2,12}" -f "-----", "------", "-------") "White"
$okSpan   = [TimeSpan]::FromSeconds($totalSecondsByStatus["ok"])
$failSpan = [TimeSpan]::FromSeconds($totalSecondsByStatus["fail"])
Write-Chain ("{0,-46} {1,-6} {2,12}" -f "stages OK total", "ok",   (Format-Elapsed $okSpan)) "Green"
if ($totalSecondsByStatus["fail"] -gt 0.0) {
    Write-Chain ("{0,-46} {1,-6} {2,12}" -f "stages FAILED total", "fail", (Format-Elapsed $failSpan)) "Red"
}
Write-Chain ("{0,-46} {1,-6} {2,12}" -f "chain wall-clock total", "-", (Format-Elapsed $ChainStopwatch.Elapsed)) "Cyan"
Write-Chain ("(per-stage CSV: " + $timingCsv + ")") "DarkGray"
Write-Chain "=====================================================================" "Cyan"

# Headline reminder of what to read
Write-Chain "" "Gray"
Write-Chain "READ FROM THE PROOF TABLE + COMPARISON CSV:" "Yellow"
Write-Chain ("  comparison: " + (Join-Path $CheckpointDir "eval_op4_zero_shot\op4_zero_shot_comparison.csv")) "Gray"
Write-Chain "" "Gray"
Write-Chain "BIG GATES (Summer abstract is in business if these all pass):" "Yellow"
Write-Chain "  WR(C) > WR(A)                            latent setup helped." "White"
Write-Chain "  WR(C) > WR(B)                            K=4 strategy discovery helped (DAGGER)." "White"
if ($LatentEvalModesOrdered -contains "uniform_random") {
    Write-Chain "  WR(C, normal) > WR(C, uniform_random)    q_phi beats uniform random z." "White"
}
if ($LatentEvalModesOrdered -contains "shuffled") {
    Write-Chain "  WR(C, normal) > WR(C, shuffled)          q_phi's state-conditioning matters (not just marginal P(z))." "White"
}
Write-Chain "" "Gray"
Write-Chain "ABLATION DETAILS:" "Yellow"
Write-Chain "  Compare D and E to C to attribute the gain to persistence vs entropy." "White"
if ($IncludeOptionA) {
    Write-Chain "  WR(F) vs WR(C): Option A (episode-start z) vs Option B (sparse refresh)." "White"
}
Write-Chain "" "Gray"
Write-Chain "Per-row table for the headline write-up:" "Yellow"
Write-Chain "  WR vs no-latent, WR vs K=1, OP4 zero-shot WR, OP5_RUSHER WR, MI(z;opp)," "White"
Write-Chain "  MI(z;phase), MI(z;flag_state), forced_z_macro_jsd_mean," "White"
Write-Chain "  latent_behavior_diversity_l2_mean, selected-z occupancy, normal/random/shuffled WR." "White"
