# 项目文件功能说明：
# 本文件用于在 Windows PowerShell 中创建并验证 Lisan_project 的 OpenHGNN Conda 环境。
# 它创建 lisan-openhgnn 环境，安装 environment-openhgnn.yml 中声明的依赖，
# 然后以 editable 模式安装项目内的 OpenHGNN 源码，方便直接在 E:\Lisan_project\OpenHGNN 下运行实验。

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$EnvFile = Join-Path $ProjectRoot "environment-openhgnn.yml"
$OpenHGNNRoot = Join-Path $ProjectRoot "OpenHGNN"
$EnvName = "lisan-openhgnn"

$EnvExists = $null -ne (conda env list | Select-String -Pattern "^\s*$EnvName\s")
if ($EnvExists) {
    Write-Host "Conda environment '$EnvName' already exists. Updating it from $EnvFile ..."
    conda env update -n $EnvName -f $EnvFile --prune
} else {
    Write-Host "Creating Conda environment '$EnvName' from $EnvFile ..."
    conda env create -f $EnvFile
}

Write-Host "Installing local OpenHGNN source in editable mode ..."
conda run -n $EnvName python -m pip install -e $OpenHGNNRoot

Write-Host "Verifying key imports ..."
conda run -n $EnvName python -c "import torch, dgl, openhgnn; print('torch', torch.__version__); print('dgl', dgl.__version__)"
