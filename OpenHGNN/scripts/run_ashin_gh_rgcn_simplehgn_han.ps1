# run_ashin_gh_rgcn_simplehgn_han.ps1
#
# 跑 ASHIN-G/H 的批量实验。
# 覆盖 RGCN、SimpleHGN、HAN，以及 ohgbn-acm / ohgbn-imdb。
# 流程是先补足 Optuna trial，再用 best_params.json 跑多 seed，最后汇总成 Excel。
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Python = "E:\Anaconda3\envs\openhgnn2025511\python.exe",
    [int]$Gpu = 0,
    [int]$Trials = 50,
    [string[]]$Seeds = @("0", "1", "2", "3", "4", "5"),
    [string[]]$Models = @("RGCN", "SimpleHGN", "HAN"),
    [string[]]$Datasets = @("ohgbn-acm", "ohgbn-imdb"),
    [string[]]$Versions = @("G", "H"),
    [int[]]$WaitForPids = @(),
    [switch]$StopOnError
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $Root "openhgnn\output\overnight\ashin_gh_rgcn_simplehgn_han_$Stamp"
$LogPath = Join-Path $RunRoot "ashin_gh_rgcn_simplehgn_han.log"
$ExcelPath = Join-Path $RunRoot "ashin_gh_rgcn_simplehgn_han_results.xlsx"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

function Write-StepLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $LogPath -Value $line -Encoding UTF8
}

function Wait-ExistingPython {
    param([int[]]$Pids)
    foreach ($pidValue in $Pids) {
        $process = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
        if ($null -ne $process -and $process.ProcessName -ne "python") {
            Write-StepLog "Skip waiting for PID=$pidValue because it is $($process.ProcessName), not python."
            continue
        }
        while (Get-Process -Id $pidValue -ErrorAction SilentlyContinue) {
            Write-StepLog "Waiting for existing python process PID=$pidValue to finish before starting ASHIN-G/H queue."
            Start-Sleep -Seconds 60
        }
    }
}

function Convert-SeedList {
    param([string[]]$RawSeeds)
    $items = @()
    foreach ($rawSeed in $RawSeeds) {
        foreach ($part in ($rawSeed -split ",")) {
            $trimmed = $part.Trim()
            if ($trimmed.Length -gt 0) {
                $items += [int]$trimmed
            }
        }
    }
    return $items
}

function Invoke-LoggedPython {
    param(
        [string]$Label,
        [string[]]$Arguments
    )
    Write-StepLog "START $Label"
    Write-StepLog ("COMMAND {0} {1}" -f $Python, ($Arguments -join " "))
    $oldErrorActionPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        & $Python @Arguments 2>&1 | ForEach-Object { $_.ToString() } | Tee-Object -FilePath $LogPath -Append
    }
    finally {
        $ErrorActionPreference = $oldErrorActionPreference
    }
    $code = $LASTEXITCODE
    Write-StepLog "END $Label exit_code=$code"
    if ($code -ne 0 -and $StopOnError) {
        throw "Command failed: $Label"
    }
}

function Get-CompletedTrials {
    param(
        [string]$Model,
        [string]$Dataset,
        [string]$Version
    )
    $code = @'
from pathlib import Path
import sys
import optuna

root = Path(sys.argv[1])
model = sys.argv[2]
dataset = sys.argv[3]
version = sys.argv[4]
study_name = f"ashin_{model}_{dataset}_{version}"
db = root / "openhgnn" / "output" / "optuna" / f"ashin_{model.lower()}_{dataset}.db"
if not db.exists():
    print(0)
    raise SystemExit(0)
try:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{db.as_posix()}")
except Exception:
    print(0)
    raise SystemExit(0)
print(sum(1 for trial in study.trials if trial.state.name == "COMPLETE"))
'@
    $output = $code | & $Python - $Root $Model $Dataset $Version
    if ($LASTEXITCODE -ne 0) {
        return 0
    }
    $lastLine = ($output | Select-Object -Last 1)
    return [int]$lastLine
}

$Seeds = Convert-SeedList -RawSeeds $Seeds
Write-StepLog "ASHIN-G/H RGCN/SimpleHGN/HAN run root: $RunRoot"
Write-StepLog "Target trials per study: $Trials; seeds: $($Seeds -join ',')"
if ($WaitForPids.Count -gt 0) {
    Wait-ExistingPython -Pids $WaitForPids
}

foreach ($model in $Models) {
    foreach ($dataset in $Datasets) {
        foreach ($version in $Versions) {
            $completed = Get-CompletedTrials -Model $model -Dataset $dataset -Version $version
            $remaining = $Trials - $completed
            if ($remaining -lt 0) {
                $remaining = 0
            }
            Write-StepLog "Study progress model=$model dataset=$dataset ashin=$version completed=$completed target=$Trials remaining=$remaining"
            Invoke-LoggedPython -Label "tune model=$model dataset=$dataset ashin=$version remaining_trials=$remaining" -Arguments @(
                "scripts\tune_ashin_hgnn.py",
                "--model", $model,
                "--dataset", $dataset,
                "--ashin_version", $version,
                "--n_trials", "$remaining",
                "--gpu", "$Gpu",
                "--seed", "0"
            )
        }
    }
}

foreach ($model in $Models) {
    foreach ($dataset in $Datasets) {
        foreach ($version in $Versions) {
            foreach ($seed in $Seeds) {
                $runName = "final_${model}_${dataset}_ashin${version}_seed${seed}"
                Invoke-LoggedPython -Label "eval model=$model dataset=$dataset ashin=$version seed=$seed" -Arguments @(
                    "main.py",
                    "-m", $model,
                    "-d", $dataset,
                    "-t", "node_classification",
                    "-g", "$Gpu",
                    "--use_ashin",
                    "--ashin_version", $version,
                    "--use_best_config",
                    "--seed", "$seed",
                    "--run_name", $runName
                )
            }
        }
    }
}

$summaryArgs = @(
    "scripts\summarize_ashin_results.py",
    "--output", $ExcelPath,
    "--datasets"
)
$summaryArgs += $Datasets
$summaryArgs += @("--versions")
$summaryArgs += $Versions
$summaryArgs += @("--models")
$summaryArgs += $Models
Invoke-LoggedPython -Label "summarize ASHIN-G/H RGCN/SimpleHGN/HAN results" -Arguments $summaryArgs

Write-StepLog "DONE. Excel: $ExcelPath"
