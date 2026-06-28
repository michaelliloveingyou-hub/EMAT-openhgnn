# run_ashin_overnight.ps1
#
# 夜间队列脚本。
# 顺序跑 RGCN/SimpleHGN 在 ohgbn-acm 和 ohgbn-imdb 上的 ASHIN-B/C/D/E/F 搜索，
# 再用 best_params.json 跑多 seed，最后调用汇总脚本生成 Excel。

[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Python = "E:\Anaconda3\envs\openhgnn2025511\python.exe",
    [int]$Gpu = 0,
    [int]$Trials = 100,
    [int[]]$Seeds = @(0, 1, 2, 3, 4),
    [string[]]$Datasets = @("ohgbn-acm", "ohgbn-imdb"),
    [string[]]$Versions = @("E", "F", "B", "C", "D"),
    [int[]]$WaitForPids = @(),
    [switch]$StopOnError
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $Root "openhgnn\output\overnight\ashin_overnight_$Stamp"
$LogPath = Join-Path $RunRoot "overnight.log"
$ExcelPath = Join-Path $RunRoot "ashin_overnight_results.xlsx"
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
            Write-StepLog "Waiting for existing python process PID=$pidValue to finish before starting overnight queue."
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

Write-StepLog "ASHIN overnight run root: $RunRoot"
if ($WaitForPids.Count -gt 0) {
    Wait-ExistingPython -Pids $WaitForPids
}

$TuneScripts = @{
    "RGCN" = "scripts\tune_ashin_rgcn.py"
    "SimpleHGN" = "scripts\tune_ashin_simplehgn.py"
}

foreach ($model in @("RGCN", "SimpleHGN")) {
    foreach ($dataset in $Datasets) {
        foreach ($version in $Versions) {
            $label = "tune model=$model dataset=$dataset ashin=$version trials=$Trials"
            Invoke-LoggedPython -Label $label -Arguments @(
                $TuneScripts[$model],
                "--dataset", $dataset,
                "--ashin_version", $version,
                "--n_trials", "$Trials",
                "--gpu", "$Gpu",
                "--seed", "0"
            )
        }
    }
}

foreach ($model in @("RGCN", "SimpleHGN")) {
    foreach ($dataset in $Datasets) {
        foreach ($version in $Versions) {
            foreach ($seed in $Seeds) {
                $runName = "final_${model}_${dataset}_ashin${version}_seed${seed}"
                $label = "eval model=$model dataset=$dataset ashin=$version seed=$seed"
                Invoke-LoggedPython -Label $label -Arguments @(
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
$summaryArgs += @("--models", "RGCN", "SimpleHGN")
Invoke-LoggedPython -Label "summarize results" -Arguments $summaryArgs

Write-StepLog "DONE. Excel: $ExcelPath"
