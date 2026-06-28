# run_ashin_d_rgcn_simplehgn.ps1
#
# 补跑 ASHIN-D 的 RGCN/SimpleHGN 实验。
# 覆盖 ohgbn-acm 和 ohgbn-imdb。
# 流程是先补足 Optuna trial，再用 best_params.json 跑多 seed，最后汇总成 Excel。
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Python = "E:\Anaconda3\envs\openhgnn2025511\python.exe",
    [int]$Gpu = 0,
    [int]$Trials = 50,
    [int[]]$Seeds = @(0, 1, 2, 3, 4, 5),
    [string[]]$Models = @("RGCN", "SimpleHGN"),
    [string[]]$Datasets = @("ohgbn-acm", "ohgbn-imdb"),
    [int[]]$WaitForPids = @(),
    [switch]$StopOnError
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $Root "openhgnn\output\overnight\ashin_d_rgcn_simplehgn_$Stamp"
$LogPath = Join-Path $RunRoot "ashin_d_rgcn_simplehgn.log"
$ExcelPath = Join-Path $RunRoot "ashin_d_rgcn_simplehgn_results.xlsx"
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
            Write-StepLog "Waiting for existing python process PID=$pidValue to finish before starting ASHIN-D queue."
            Start-Sleep -Seconds 60
        }
    }
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
        [string]$Dataset
    )
    $code = @'
from pathlib import Path
import sys
import optuna

root = Path(sys.argv[1])
model = sys.argv[2]
dataset = sys.argv[3]
study_name = f"ashin_{model}_{dataset}_D"
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
    $output = & $Python -c $code $Root $Model $Dataset
    if ($LASTEXITCODE -ne 0) {
        return 0
    }
    $lastLine = ($output | Select-Object -Last 1)
    return [int]$lastLine
}

Write-StepLog "ASHIN-D RGCN/SimpleHGN run root: $RunRoot"
Write-StepLog "Target trials per study: $Trials; seeds: $($Seeds -join ',')"
if ($WaitForPids.Count -gt 0) {
    Wait-ExistingPython -Pids $WaitForPids
}

foreach ($model in $Models) {
    foreach ($dataset in $Datasets) {
        $completed = Get-CompletedTrials -Model $model -Dataset $dataset
        $remaining = $Trials - $completed
        if ($remaining -lt 0) {
            $remaining = 0
        }
        Write-StepLog "Study progress model=$model dataset=$dataset ashin=D completed=$completed target=$Trials remaining=$remaining"
        Invoke-LoggedPython -Label "tune model=$model dataset=$dataset ashin=D remaining_trials=$remaining" -Arguments @(
            "scripts\tune_ashin_hgnn.py",
            "--model", $model,
            "--dataset", $dataset,
            "--ashin_version", "D",
            "--n_trials", "$remaining",
            "--gpu", "$Gpu",
            "--seed", "0"
        )
    }
}

foreach ($model in $Models) {
    foreach ($dataset in $Datasets) {
        foreach ($seed in $Seeds) {
            $runName = "final_${model}_${dataset}_ashinD_seed${seed}"
            Invoke-LoggedPython -Label "eval model=$model dataset=$dataset ashin=D seed=$seed" -Arguments @(
                "main.py",
                "-m", $model,
                "-d", $dataset,
                "-t", "node_classification",
                "-g", "$Gpu",
                "--use_ashin",
                "--ashin_version", "D",
                "--use_best_config",
                "--seed", "$seed",
                "--run_name", $runName
            )
        }
    }
}

$summaryArgs = @(
    "scripts\summarize_ashin_results.py",
    "--output", $ExcelPath,
    "--datasets"
)
$summaryArgs += $Datasets
$summaryArgs += @("--versions", "D", "--models")
$summaryArgs += $Models
Invoke-LoggedPython -Label "summarize ASHIN-D RGCN/SimpleHGN results" -Arguments $summaryArgs

Write-StepLog "DONE. Excel: $ExcelPath"
