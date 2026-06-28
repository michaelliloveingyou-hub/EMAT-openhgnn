# run_ashin_rgcn_simplehgn_seed05.ps1
#
# 用已有 best_params 跑 RGCN/SimpleHGN 的最终多 seed 测试。
# 默认覆盖 ohgbn-acm、ohgbn-imdb 和 ASHIN-B/C/D/E/F/G/H，seed 固定为 0..5。
# 不重新调参；如果某个 best_params.json 缺失，会写入日志并跳过该组合。
[CmdletBinding(PositionalBinding = $false)]
param(
    [string]$Python = "E:\Anaconda3\envs\openhgnn2025511\python.exe",
    [int]$Gpu = 0,
    [string[]]$Seeds = @("0", "1", "2", "3", "4", "5"),
    [string[]]$Models = @("RGCN", "SimpleHGN"),
    [string[]]$Datasets = @("ohgbn-acm", "ohgbn-imdb"),
    [string[]]$Versions = @("B", "C", "D", "E", "F", "G", "H"),
    [int[]]$WaitForPids = @(),
    [switch]$StopOnError,
    [switch]$OnlyMissing
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunRoot = Join-Path $Root "openhgnn\output\overnight\ashin_rgcn_simplehgn_seed05_$Stamp"
$LogPath = Join-Path $RunRoot "ashin_rgcn_simplehgn_seed05.log"
$ExcelPath = Join-Path $RunRoot "ashin_rgcn_simplehgn_seed05_results.xlsx"
New-Item -ItemType Directory -Force -Path $RunRoot | Out-Null

function Write-StepLog {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $LogPath -Value $line -Encoding UTF8
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

function Wait-ExistingPython {
    param([int[]]$Pids)
    foreach ($pidValue in $Pids) {
        $process = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
        if ($null -ne $process -and $process.ProcessName -ne "python") {
            Write-StepLog "Skip waiting for PID=$pidValue because it is $($process.ProcessName), not python."
            continue
        }
        while (Get-Process -Id $pidValue -ErrorAction SilentlyContinue) {
            Write-StepLog "Waiting for existing python process PID=$pidValue to finish before starting seed 0..5 queue."
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
    return $code
}

function Get-BestParamsPath {
    param(
        [string]$Model,
        [string]$Dataset,
        [string]$Version
    )
    return Join-Path $Root "openhgnn\output\optuna\ashin_${Model}_${Dataset}_${Version}\best_params.json"
}

function Test-SeedResultExists {
    param(
        [string]$Model,
        [string]$Dataset,
        [string]$Version,
        [int]$Seed
    )
    $pattern = "final_${Model}_${Dataset}_ashin${Version}_seed${Seed}_*"
    $matches = Get-ChildItem -Path (Join-Path $Root "openhgnn\output\ashin_logs") -Directory -Filter $pattern -ErrorAction SilentlyContinue |
        Where-Object { Test-Path (Join-Path $_.FullName "metrics.json") }
    return (($matches | Measure-Object).Count -gt 0)
}

$Seeds = Convert-SeedList -RawSeeds $Seeds
Write-StepLog "ASHIN RGCN/SimpleHGN seed 0..5 run root: $RunRoot"
Write-StepLog "Models: $($Models -join ','); datasets: $($Datasets -join ','); versions: $($Versions -join ','); seeds: $($Seeds -join ',')"
Write-StepLog "OnlyMissing: $OnlyMissing"

if ($WaitForPids.Count -gt 0) {
    Wait-ExistingPython -Pids $WaitForPids
}

$missingBest = @()
$skippedExisting = 0
$runCount = 0

foreach ($model in $Models) {
    foreach ($dataset in $Datasets) {
        foreach ($version in $Versions) {
            $bestPath = Get-BestParamsPath -Model $model -Dataset $dataset -Version $version
            if (-not (Test-Path $bestPath)) {
                $missingBest += [PSCustomObject]@{
                    model = $model
                    dataset = $dataset
                    ashin_version = $version
                    best_params_path = $bestPath
                }
                Write-StepLog "SKIP missing best_params model=$model dataset=$dataset ashin=$version path=$bestPath"
                continue
            }

            foreach ($seed in $Seeds) {
                if ($OnlyMissing -and (Test-SeedResultExists -Model $model -Dataset $dataset -Version $version -Seed $seed)) {
                    $skippedExisting += 1
                    Write-StepLog "SKIP existing metrics model=$model dataset=$dataset ashin=$version seed=$seed"
                    continue
                }

                $runName = "final_${model}_${dataset}_ashin${version}_seed${seed}"
                $runCount += 1
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
                ) | Out-Null
            }
        }
    }
}

if ($missingBest.Count -gt 0) {
    $missingPath = Join-Path $RunRoot "missing_best_params.csv"
    $missingBest | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $missingPath
    Write-StepLog "Missing best_params records written to: $missingPath"
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
Invoke-LoggedPython -Label "summarize RGCN/SimpleHGN ASHIN-B/C/D/E/F/G/H seed results" -Arguments $summaryArgs | Out-Null

Write-StepLog "DONE. Executed eval runs: $runCount; skipped existing: $skippedExisting; missing best_params combos: $($missingBest.Count)"
Write-StepLog "DONE. Excel: $ExcelPath"
