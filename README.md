# Lisan_project

本项目保存 Lisan 异构图数据集处理、OpenHGNN 接入代码和实验自动化脚本，并在项目内置的 `OpenHGNN/` 源码副本中注册 `lisan-acm`、`lisan-dblp`，用于节点分类、链接预测和超参数搜索实验。项目已按 GitHub + AutoDL 同步方式整理：Git 只管理代码、配置和文档，大型原始数据、处理后特征、训练输出和模型权重留在本地或 AutoDL 数据盘。

## 目录说明

- `Datasets/`: 原始数据集目录，包含 ACM、DBLP、IMDB、TEST 等子数据集的 `node.dat`、`link.dat`、`label.dat*`、`info.dat`、`meta.dat` 等文件；该目录含大文件，不纳入 Git。
- `Dataset_Emat/`: 已按基矩阵方法处理好的 ACM/DBLP EMat 特征，`EMat_<节点类型编号>.pt` 为 `基数量 × 节点数量`，转换为 OpenHGNN 节点特征时会转置为 `节点数量 × 基数量`；该目录不纳入 Git。
- `OpenHGNN/`: 项目内置 OpenHGNN 源码副本，包含 Lisan 数据集注册逻辑和实验入口；源码纳入 Git，下载或生成的 `.bin`、`.pt`、`.zip` 等数据产物不纳入 Git。
- `experiments_lisan/`: Lisan raw baseline 自动化实验代码，负责 Optuna 搜索、5 seed 复训、resume、日志、元数据和最终汇总。
- `scripts/`: 项目维护脚本。
  - `audit_lisan_datasets.py`: 只读扫描数据集并生成 Markdown 审计报告。
  - `convert_lisan_to_openhgnn.py`: 将 `Datasets/ACM`、`Datasets/DBLP` 转换为 OpenHGNN 可读取的 DGL 图文件。
  - `run_lisan_raw_experiments.py`: 一键运行 `lisan-acm`、`lisan-dblp` 的 raw 特征 baseline 实验。
  - `run_lisan_model_feature_sweep.py`: 一键运行 HGT、SimpleHGN、GCN、GAT 的 A-E 特征 sweep。
  - `autodl_run_model_feature_parallel.sh`: AutoDL 上两模型并行运行 HGT、SimpleHGN、GCN、GAT sweep 的启动脚本，预检通过后分两批训练，全部结束后关机。
- `docs/`: 项目文档。
  - `openhgnn_lisan_usage.md`: Lisan 数据集接入 OpenHGNN 的运行说明。
  - `work_targets/lisan_raw_experiments_goal.md`: raw baseline 自动化实验目标文件。
  - `reports/dataset_detailed_report.md`: 数据集审计报告。
- `experiments/`: 自动化实验输出目录，`lisan_raw_runs/` 保存本阶段 raw baseline 的 trial、best params、seed 结果和最终汇总表；该目录不纳入 Git。
- `data/archives/`: 数据归档文件，当前包含 `Datasets.zip`；该目录不纳入 Git。
- `outputs/`: 实验输出、日志和模型权重归档位置，当前包含已移动的 RGCN 冒烟测试输出。
- `environment-openhgnn.yml`: 用于创建 `lisan-openhgnn` Conda 环境的依赖清单。
- `setup_openhgnn_env.ps1`: 自动创建/更新 Conda 环境并安装项目内 OpenHGNN 源码的 PowerShell 脚本。

## 安装环境

在项目根目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_openhgnn_env.ps1
```

## GitHub 与 AutoDL 同步

本项目适合用 GitHub 同步代码和配置，用 AutoDL 数据盘或其他外部存储保存大数据与训练产物。不要把 `Datasets/`、`Dataset_Emat/`、`data/lisan_processed_features/`、`experiments/`、`outputs/`、模型权重和二进制图文件上传到 GitHub。

本地首次初始化和提交：

```powershell
git init
git status --short
git add --dry-run .
git add .
git commit -m "Initial project structure for AutoDL sync"
git branch -M main
git remote add origin <你的 GitHub 仓库地址>
git push -u origin main
```

AutoDL 上首次拉取：

```bash
git clone <你的 GitHub 仓库地址>
cd Lisan_project
conda env create -f environment-openhgnn.yml
conda activate lisan-openhgnn
```

AutoDL 上需要单独准备未进入 Git 的数据目录，例如 `Datasets/`、`Dataset_Emat/` 或处理后特征目录。可以用 AutoDL 数据盘、网盘、对象存储、Hugging Face Dataset、DVC 或 Git LFS 管理这些文件；当前项目默认不使用 Git LFS。

后续同步流程：

```powershell
# 本地修改代码后
git status --short
git add .
git commit -m "Update experiment scripts"
git push
```

```bash
# AutoDL 项目目录中
git pull
```

提交前建议检查是否有大文件将被纳入 Git：

```powershell
git status --short
git add --dry-run .
```

## 运行命令

生成或刷新数据集审计报告：

```powershell
python .\scripts\audit_lisan_datasets.py --datasets .\Datasets --output .\docs\reports\dataset_detailed_report.md
```

转换 ACM/DBLP 到 OpenHGNN 数据格式：

```powershell
python .\scripts\convert_lisan_to_openhgnn.py --datasets-root .\Datasets --openhgnn-root .\OpenHGNN --dataset all
```

额外生成 EMat 特征版本图文件，不覆盖 raw 版本。`--feature_mode` 支持 A-E：

- `A` / `raw`: 原始 `node.dat` 特征。
- `B` / `emat`: 直接使用 EMat 转置后的基坐标特征。
- `C` / `emat_svd_128`: 读取预处理产物，对每个节点类型的 EMat 单独做 TruncatedSVD 到 128 维，并做 z-score。
- `D` / `emat_tfidf_3025`: 读取预处理产物，对 EMat 做 log1p TF-IDF/IDF 加权，保留全基维度；DBLP 的全基维度是 14314，不是 3025。
- `E` / `raw_emat_sparse_encoder`: 可学习 SparseLinear/EmbeddingBag 风格 runtime encoder 模式；当前只生成 CSR 输入和 encoder 模块，不再生成错误的固定随机投影图文件，通用 RGCN sweep 会跳过该模式。

```powershell
python .\scripts\diagnose_emat_features.py --datasets lisan-acm lisan-dblp --output_dir .\data\lisan_processed_features
python .\scripts\preprocess_emat_svd.py --datasets lisan-acm lisan-dblp --svd_dims 64 128 256 --output_dir .\data\lisan_processed_features
python .\scripts\preprocess_emat_tfidf.py --datasets lisan-acm lisan-dblp --output_dir .\data\lisan_processed_features
python .\scripts\preprocess_emat_tfidf_svd.py --datasets lisan-acm lisan-dblp --svd_dims 64 128 256 --output_dir .\data\lisan_processed_features
python .\scripts\preprocess_emat_sparse_inputs.py --datasets lisan-acm lisan-dblp --output_dir .\data\lisan_processed_features
python .\scripts\convert_lisan_to_openhgnn.py --datasets-root .\Datasets --emat-root .\Dataset_Emat --openhgnn-root .\OpenHGNN --processed-feature-root .\data\lisan_processed_features --dataset all --feature_mode emat_svd_128
```

运行 OpenHGNN 实验示例：

```powershell
conda activate lisan-openhgnn
cd E:\Lisan_project\OpenHGNN
python .\main.py -m RGCN -d lisan-acm -t node_classification -g -1 --max_epoch 50 --patience 10
python .\main.py -m RGCN -d lisan-acm -t node_classification -g -1 --feature_mode C --max_epoch 50 --patience 10
python .\main.py -m RGCN -d lisan-acm -t node_classification -g -1 --feature_mode raw_emat_tfidf_svd_128 --max_epoch 50 --patience 10
```

更多节点分类、链接预测和 HPO 命令见 `docs/openhgnn_lisan_usage.md`。

运行 Lisan raw baseline 自动化实验：

```powershell
conda activate lisan-openhgnn
cd E:\Lisan_project
python .\scripts\run_lisan_raw_experiments.py `
  --datasets lisan-acm lisan-dblp `
  --tasks node_classification link_prediction `
  --models RGCN HAN HGT SimpleHGN `
  --n_trials 50 `
  --seeds 0 1 2 3 4 `
  --max_epoch 200 `
  --patience 20 `
  --gpu 0 `
  --resume
```

运行 EMat 静态特征版本时只增加 `--feature_mode A|B|C|D|raw_emat_svd_128|raw_emat_tfidf_svd_128`；搜索空间、seed 和数据划分逻辑与 raw 版本保持一致，非 raw 模式默认输出到 `experiments/lisan_emat_runs/`。`E/raw_emat_sparse_encoder` 需要 runtime encoder wrapper，当前通用 OpenHGNN/RGCN 自动脚本会跳过：

```powershell
python .\scripts\run_lisan_raw_experiments.py `
  --feature_mode C `
  --datasets lisan-acm lisan-dblp `
  --tasks node_classification link_prediction `
  --models RGCN HAN HGT SimpleHGN `
  --n_trials 50 `
  --seeds 0 1 2 3 4 `
  --max_epoch 200 `
  --patience 20 `
  --gpu 0 `
  --resume
```

运行 RGCN 的 A-E 五种特征自动 sweep。该脚本固定模型为 RGCN，默认遍历 `A B C D E × lisan-acm/lisan-dblp × node_classification/link_prediction`，每个组合运行一次 HPO 流程，默认输出到 `experiments/lisan_rgcn_feature_sweep/`：

```powershell
python .\scripts\run_lisan_rgcn_feature_sweep.py --resume
```

VSCode 中也可以直接运行任务 `Lisan RGCN A-E feature sweep`。该任务默认使用轻量配置 `--n_trials 1 --seeds 0 --max_epoch 50 --patience 10 --gpu -1`；正式长实验可在命令行改为 `--n_trials 50 --seeds 0 1 2 3 4 --gpu 0`。

运行 HGT、SimpleHGN、GCN、GAT 的 A-E 特征 sweep。该脚本默认遍历 `HGT/SimpleHGN/GCN/GAT × A/B/C/D/E × lisan-acm/lisan-dblp × node_classification/link_prediction`，每个组合先做 HPO，再用 5 个 seed 复训并汇总均值和样本标准差。GCN/GAT 使用 OpenHGNN 的 `homo_GNN` 后端，分别设置 `gnn_type=gcnconv/gatconv`；`E/raw_emat_sparse_encoder` 当前会被记录为 skipped：

```powershell
python .\scripts\run_lisan_model_feature_sweep.py `
  --n_trials 50 `
  --seeds 0 1 2 3 4 `
  --max_epoch 200 `
  --patience 20 `
  --gpu 0 `
  --resume
```

AutoDL 上同样先 `conda activate lisan-openhgnn`，再运行上面的命令；数据目录和预处理图文件需要在 AutoDL 数据盘中提前准备或重新生成，GitHub 只同步代码和文档。

AutoDL 上需要长时间运行并避免手动复制长命令时，可使用两模型并行启动脚本。该脚本默认先跑 `HGT + SimpleHGN`，再跑 `GCN + GAT`，每个模型写入独立日志，使用 `--resume` 续跑，预检失败时不会关机；预检通过并完成两批训练后会执行 `/usr/bin/shutdown`：

```bash
cd /root/autodl-tmp/EMAT-openhgnn
mkdir -p outputs/logs
nohup bash scripts/autodl_run_model_feature_parallel.sh > outputs/logs/autodl_model_parallel_stdout.log 2>&1 &
```

查看运行状态和日志：

```bash
ps -ef | grep run_lisan_model_feature_sweep | grep -v grep
tail -f outputs/logs/model_sweep_HGT.log
tail -f outputs/logs/model_feature_parallel_launcher_*.log
```

如需调试时不关机，可临时设置：

```bash
SHUTDOWN_AFTER=0 bash scripts/autodl_run_model_feature_parallel.sh
```

快速预览将要运行的组合：

```powershell
python .\scripts\run_lisan_raw_experiments.py --dry_run
```

只运行单个数据集和单个模型：

```powershell
python .\scripts\run_lisan_raw_experiments.py --datasets lisan-acm --tasks node_classification --models RGCN --n_trials 10 --seeds 0 1 2 3 4 --gpu -1 --resume
```

Windows CMD 示例：

```cmd
conda activate lisan-openhgnn
cd /d E:\Lisan_project
python scripts\run_lisan_raw_experiments.py --datasets lisan-acm lisan-dblp --tasks node_classification link_prediction --models RGCN HAN HGT SimpleHGN --n_trials 50 --seeds 0 1 2 3 4 --max_epoch 200 --patience 20 --gpu 0 --resume
```


