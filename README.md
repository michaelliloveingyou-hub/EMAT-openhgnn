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

## 输出位置

- 转换后的 Lisan 图文件仍位于 `OpenHGNN/openhgnn/dataset/lisan_hgb/`，因为当前 OpenHGNN loader 直接从该路径读取。
- raw 特征图文件为 `lisan-acm.bin`、`lisan-dblp.bin`；静态 EMat 特征图文件统一位于 `OpenHGNN/openhgnn/dataset/lisan_hgb/`，通过 `--feature_mode A|B|C|D|raw_emat_svd_128|raw_emat_tfidf_svd_128` 等选择。`E/raw_emat_sparse_encoder` 不是静态图文件。
- raw baseline 自动化实验默认输出到 `experiments/lisan_raw_runs/`，其中包含：
  - `final_summary_node_classification.csv`
  - `final_summary_link_prediction.csv`
  - 每个 `dataset/task/model` 组合的 `hpo_trials.csv`、`best_params.json`、`seed_results.csv`、`summary.json`、`metadata.json`、`run.log` 和失败时的 `error.log`
- 节点分类 HPO objective 使用验证集 `Macro_f1`；链接预测 HPO objective 使用当前 OpenHGNN 返回的验证集 `roc_auc`。链接预测最终汇总只填 OpenHGNN 当前路径提供的 `roc_auc/loss`，AP、MRR、Hits@10 列保留为空并在 notes 中说明。
- `--resume` 会跳过已完成组合和已完成 seed，继续运行缺失或失败的部分。
- EMat baseline 自动化实验默认输出到 `experiments/lisan_emat_runs/`，避免覆盖 raw 版本的 `best_params.json`、trial 和 seed 汇总。
- 除本次 raw baseline 明确使用 `experiments/lisan_raw_runs/` 外，其他临时实验输出建议统一放入 `outputs/`，不要混入项目根目录。
- `__pycache__/`、`*.pyc`、`openhgnn.egg-info/`、模型权重、TensorBoard 事件文件、日志、`experiments/lisan_raw_runs/`、`experiments/lisan_emat_runs/` 和派生的 `*-emat.bin` 不建议纳入 Git；如果后续初始化 Git，建议在根目录配置 `.gitignore`。

## 改动记录

### 改动 1

新增 `dataset_detail_reader.py`，用于详细读取 ACM、DBLP、IMDB、TEST 数据集的节点、边、标签、特征维度、划分重叠、端点一致性和链接预测候选关系。

### 改动 2

新增 `README.md`，说明项目数据文件、数据集读取脚本用途和运行方式。

### 改动 3

运行 `dataset_detail_reader.py` 生成 `dataset_detailed_report.md`，保存当前 ACM、DBLP、IMDB、TEST 数据集的详细统计报告，便于直接阅读和判断任务适配性。

### 改动 4

补充 `README.md` 的文件说明，加入 `dataset_detailed_report.md` 的用途说明。

### 改动 5

新增 `convert_to_openhgnn.py`，用于将 `Datasets/ACM` 和 `Datasets/DBLP` 转换成 OpenHGNN 可读取的 DGL 异构图二进制文件，并写入节点分类训练、验证、测试划分。

### 改动 6

新增 `OPENHGNN_LISAN_USAGE.md`，记录新数据集名称、转换方式、划分策略、节点分类命令、链接预测命令和超参数搜索命令。

### 改动 7

在 `E:\OpenHGNN-513\OpenHGNN-513` 源码中新增 `openhgnn/dataset/lisan_hgb_dataset.py`，并修改 `openhgnn/dataset/__init__.py`，注册 `lisan-acm`、`lisan-dblp` 两个数据集，支持节点分类和链接预测任务。

### 改动 8

在 `E:\OpenHGNN-513\OpenHGNN-513\scripts` 新增 `lisan_hpo.py`，提供 `lisan-acm`、`lisan-dblp` 的命令行超参数搜索入口；同时调整 OpenHGNN 的 HPO 逻辑，优先使用验证集指标选择超参数，避免使用测试集调参。

### 改动 9

复制 OpenHGNN 源码到项目内 `OpenHGNN/` 目录，保留已注册的 `lisan-acm`、`lisan-dblp` 数据集逻辑，后续可直接在本项目内运行 OpenHGNN。

### 改动 10

新增 `environment-openhgnn.yml` 和 `setup_openhgnn_env.ps1`，用于创建 `lisan-openhgnn` Conda 环境并安装 OpenHGNN 依赖。

### 改动 11

更新 `convert_to_openhgnn.py` 和 `OPENHGNN_LISAN_USAGE.md`，默认使用项目内 `OpenHGNN/` 路径，不再依赖外部 `E:\OpenHGNN-513` 路径。

### 改动 12

修正 `environment-openhgnn.yml` 的依赖组合，使用已验证可导入的 `torch==2.2.1`、`torchdata==0.7.1`、`numpy<2`、CPU 版 `dgl==2.2.1`，并补充 `pydantic` 以满足 DGL GraphBolt 的导入依赖。

### 改动 13

运行 `convert_to_openhgnn.py` 生成项目内 OpenHGNN 数据文件：`OpenHGNN/openhgnn/dataset/lisan_hgb/lisan-acm.bin`、`lisan-acm.manifest.json`、`lisan-dblp.bin`、`lisan-dblp.manifest.json`。

### 改动 14

使用 `lisan-openhgnn` Conda 环境完成 RGCN 冒烟测试：`lisan-acm`、`lisan-dblp` 的节点分类任务均可运行；`lisan-acm`、`lisan-dblp` 的链接预测任务也均可运行，并输出验证集和测试集指标。

### 改动 15

更新 `OPENHGNN_LISAN_USAGE.md` 的命令示例，补充 `conda activate lisan-openhgnn`，并移除 `main.py` 不支持的 `--score_fn` 参数；RGCN 链接预测使用 OpenHGNN 配置中的默认 `distmult` 打分函数。

### 改动 16

修正 `OpenHGNN/scripts/lisan_hpo.py`，补齐 HPO 运行时默认参数，默认使用 full-batch，并增加 `--max_epoch`、`--patience` 便于快速验证；链接预测 HPO 固定使用已验证可运行的 `distmult`。同步更新 `OPENHGNN_LISAN_USAGE.md` 的 HPO 快速检查命令。

### 改动 17

将 `lisan-openhgnn` 环境从 CPU 版依赖切换为 CUDA 12.1 版依赖，固定 `torch==2.2.1+cu121` 和 `dgl==2.2.1+cu121`；已用 `-g 0` 完成 RGCN GPU 冒烟测试，训练输出显示 `device: cuda:0`。

### 改动 18

整理项目目录结构：将数据审计脚本移动到 `scripts/audit_lisan_datasets.py`，将 OpenHGNN 转换脚本移动到 `scripts/convert_lisan_to_openhgnn.py`，将使用说明移动到 `docs/openhgnn_lisan_usage.md`，将审计报告移动到 `docs/reports/dataset_detailed_report.md`，将 `Datasets.zip` 移动到 `data/archives/`，并将已有 RGCN 冒烟测试输出移动到 `outputs/openhgnn/RGCN_2026-06-26_smoke/`。本次未删除缓存文件，未移动原始 `Datasets/`，未改变 OpenHGNN 的核心代码逻辑。

### 改动 19

新增 `.gitignore`、`docs/work_targets/lisan_raw_experiments_goal.md`、`experiments_lisan/` 和 `scripts/run_lisan_raw_experiments.py`，实现 raw 特征 baseline 自动化实验入口。该入口按 `dataset × task × model` 组合运行 Optuna 验证集调参，保存 best params，再用指定 seeds 复训并输出 mean/std 汇总；支持 `--dry_run`、`--resume`、失败隔离、组合级日志和环境元数据。同步小改 `OpenHGNN/openhgnn/trainerflow/link_prediction.py`，让链接预测最终返回 valid/test 两组 OpenHGNN 指标，确保 HPO objective 可以严格使用验证集 `roc_auc`。链接预测汇总限定为当前 OpenHGNN 路径提供的 `roc_auc/loss`，AP、MRR、Hits@10 目前保留为空并写入 notes。

### 改动 20

新增 EMat 特征图生成与训练切换流程：`scripts/convert_lisan_to_openhgnn.py` 支持 `--feature_mode raw|emat|both` 和 `--emat-root`，EMat 版本输出为 `lisan-acm-emat.bin`、`lisan-dblp-emat.bin`，不覆盖 raw 图文件；`OpenHGNN/main.py`、`OpenHGNN/scripts/lisan_hpo.py` 和 `scripts/run_lisan_raw_experiments.py` 支持 `--feature_mode raw|emat`，在保持原始节点分类 mask、链接预测边划分逻辑和 HPO 搜索空间不变的前提下选择 raw 或 EMat 特征。同步更新 `.gitignore`，避免将派生 EMat 大文件和 `experiments/lisan_emat_runs/` 纳入 Git。后续改动 21 已在此基础上扩展到 A-E 五种特征模式。

### 改动 21

扩展 A-E 五种特征模式：`A=raw`、`B=direct EMat`、`C=EMat-SVD-128`、`D=EMat-TFIDF`、`E=EMat-sparse-128`。`scripts/convert_lisan_to_openhgnn.py` 可以生成五套图文件，OpenHGNN 主入口、HPO 入口和批量实验脚本都支持 `--feature_mode A|B|C|D|E`。新增 `scripts/validate_lisan_feature_graphs.py` 用于校验 A-E 图文件的边、节点顺序、label 和 mask 一致性；已生成 `data/lisan_processed_features/feature_graph_validation.json` 验证报告。新增 `experiments_lisan/README_EMAT_FEATURES.md` 说明各特征模式、运行命令、输出位置和当前限制。

### 改动 22

新增 `scripts/run_lisan_rgcn_feature_sweep.py` 和 `.vscode/tasks.json`。该脚本/任务用于自动执行 RGCN 的 A-E 特征 sweep，覆盖 `lisan-acm`、`lisan-dblp` 两个数据集以及 `node_classification`、`link_prediction` 两个任务；每个组合独立保存 HPO trial、best params、seed results、summary 和日志，默认输出到 `experiments/lisan_rgcn_feature_sweep/`。本次只创建并 dry-run 验证任务，未启动完整实验。

### 改动 23

修正 EMat 后三种特征构建方式：新增 `scripts/diagnose_emat_features.py`、`scripts/preprocess_emat_svd.py`、`scripts/preprocess_emat_tfidf.py`、`scripts/preprocess_emat_tfidf_svd.py`、`scripts/preprocess_emat_sparse_inputs.py` 和 `experiments_lisan/models/emat_sparse_encoder.py`。SVD、TF-IDF、TF-IDF+SVD 现在先保存到 `data/lisan_processed_features/<dataset>/`，再由转换器读取产物生成静态图；`E/raw_emat_sparse_encoder` 不再使用错误的固定随机投影，当前只保存 CSR 输入和可学习 encoder 模块，通用 RGCN sweep 会跳过并记录原因。已重新生成并验证 `emat_svd_128`、`emat_tfidf_3025`、`emat_tfidf_svd_128`、`raw_emat_3025`、`raw_emat_svd_128`、`raw_emat_tfidf_3025`、`raw_emat_tfidf_svd_128` 的 ACM/DBLP 图文件。

### 改动 24

新增 `scripts/run_lisan_model_feature_sweep.py`，用于批量运行 HGT、SimpleHGN、GCN、GAT 在 A-E 特征、两个数据集、两个任务上的 HPO + 5 seed 评估。扩展 `experiments_lisan/search_spaces.py` 和 `experiments_lisan/runner.py`：GCN/GAT 通过 OpenHGNN `homo_GNN` 后端运行，分别映射为 `gcnconv` 和 `gatconv`，结果目录仍按 `GCN`、`GAT` 保存。`.vscode/tasks.json` 改用跨机器的 `python` 命令，并新增该 sweep 的 dry-run 和正式任务，便于 GitHub 同步到 AutoDL 后使用。

### 改动 25

整理 GitHub + AutoDL 同步边界：扩展 `.gitignore`，默认排除 `Datasets/`、`Dataset_Emat/`、`data/archives/`、`data/lisan_processed_features/`、`experiments/`、`outputs/`、OpenHGNN 生成的数据图文件、模型权重、日志、TensorBoard 文件和压缩包；新增 `.gitattributes` 统一 Windows/AutoDL Linux 之间的文本换行；更新 README 的项目简介、目录说明、GitHub/AutoDL 同步流程、数据外置说明和提交前验证命令。本次未删除、移动或压缩任何已有数据与训练结果。
