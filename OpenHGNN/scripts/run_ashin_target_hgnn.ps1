# run_ashin_target_hgnn.ps1
#
# 批量跑 HAN/RSHN/HPN/HGT + ASHIN-B/C/D/E/F。
# 数据集是 ohgbn-acm 和 ohgbn-imdb；默认每组 80 个 Optuna trial。
# 另外补上 RGCN/SimpleHGN 的 ASHIN-D，最后写到同一个 Excel。

[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Python = "E:\Anaconda3\envs\openhgnn2025511\python.exe",
    [int]$Gpu = 0,
    [int]$Trials = 80,
    [int[]]$Seeds = @(0),
    [string[]]$Models = @("HAN", "RSHN", "HPN", "HGT"),
    [string[]]$ExtraDModels = @("RGCN", "SimpleHGN"),
    [string[]]$Datasets = @("ohgbn-acm", "ohgbn-imdb"),
    [string[]]$Versions = @("B", "C", "D", "E", "F"),
    [int[]]$WaitForPids = @(),
    [switch]$StopOnError
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $Root "openhgnn\output\overnight\ashin_target_hgnn_$Stamp"
$LogPath = Join-Path $RunRoot "target_hgnn.log"
$ExcelPath = Join-Path $RunRoot "ashin_target_hgnn_results.xlsx"
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
            Write-StepLog "Waiting for existing python process PID=$pidValue to finish before starting target HGNN queue."
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

Write-StepLog "ASHIN target HGNN run root: $RunRoot"
if ($WaitForPids.Count -gt 0) {
    Wait-ExistingPython -Pids $WaitForPids
}

foreach ($model in $Models) {
    foreach ($dataset in $Datasets) {
        foreach ($version in $Versions) {
            Invoke-LoggedPython -Label "tune model=$model dataset=$dataset ashin=$version trials=$Trials" -Arguments @(
                "scripts\tune_ashin_hgnn.py",
                "--model", $model,
                "--dataset", $dataset,
                "--ashin_version", $version,
                "--n_trials", "$Trials",
                "--gpu", "$Gpu",
                "--seed", "0"
            )
        }
    }
}

foreach ($model in $ExtraDModels) {
    foreach ($dataset in $Datasets) {
        Invoke-LoggedPython -Label "tune extra-D model=$model dataset=$dataset ashin=D trials=$Trials" -Arguments @(
            "scripts\tune_ashin_hgnn.py",
            "--model", $model,
            "--dataset", $dataset,
            "--ashin_version", "D",
            "--n_trials", "$Trials",
            "--gpu", "$Gpu",
            "--seed", "0"
        )
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

foreach ($model in $ExtraDModels) {
    foreach ($dataset in $Datasets) {
        foreach ($seed in $Seeds) {
            $runName = "final_${model}_${dataset}_ashinD_seed${seed}"
            Invoke-LoggedPython -Label "eval extra-D model=$model dataset=$dataset ashin=D seed=$seed" -Arguments @(
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
$summaryArgs += @("--versions")
$summaryArgs += $Versions
$summaryArgs += @("--models")
$summaryArgs += $Models
$summaryArgs += $ExtraDModels
Invoke-LoggedPython -Label "summarize target HGNN results" -Arguments $summaryArgs

Write-StepLog "DONE. Excel: $ExcelPath"
