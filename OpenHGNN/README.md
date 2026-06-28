# OpenHGNN + ASHIN 实验代码

这个仓库是在 OpenHGNN 上做的 ASHIN 特征增强实验版本。原来的 OpenHGNN 训练入口、模型和数据集加载方式基本保留；这里主要加了一套 ASHIN 结构特征构造逻辑，用来给异质图神经网络补充结构信息，然后在节点分类任务上比较效果。

目前主要实验数据集是 `ohgbn-acm` 和 `ohgbn-imdb`，任务是 `node_classification`，常用评价指标是 `Macro-F1` 和 `Micro-F1`。

## 现在这份代码做什么

ASHIN 的思路是：不直接用标签，也不看测试集结果，只从异质图结构里构造额外特征，再把这些特征接到 HGNN 的输入侧。

当前保留了这些版本：

| 版本 | 大致含义 | 融合方式 |
|---|---|---|
| B | 给目标节点构造 signature one-hot 结构特征 | concat |
| C | 在 B 的基础上做 incidence / dense 压缩 | concat |
| D | 固定使用 C 的结构特征，并在目标节点侧做门控融合 | gated |
| E | 给所有节点类型分别构造 ASHIN-B/C 风格特征 | concat |
| F | 可选 B/C 作为基础结构特征，再做门控融合 | gated |
| G | 先增强属性节点，再把属性节点结构信息聚合回目标节点 | gated |
| H | 在目标节点 ASHIN-C 上加入 target-target 共性传播 | gated |

这里的 `concat` 是直接把原始特征和 ASHIN 特征拼起来；`gated` 是让模型学习一个门控权重，决定原始特征和 ASHIN 特征各占多少。

## 代码位置

ASHIN 相关代码主要在这里：

```text
openhgnn/ashin/
```

几个关键文件：

```text
openhgnn/ashin/builder.py        # ASHIN 总入口，按版本分发 B/C/D/E/F/G/H
openhgnn/ashin/version_b.py      # ASHIN-B
openhgnn/ashin/version_c.py      # ASHIN-C
openhgnn/ashin/version_g.py      # ASHIN-G
openhgnn/ashin/version_h.py      # ASHIN-H
openhgnn/ashin/best_config.py    # 读取 ASHIN 的 best_params.json
openhgnn/ashin/cache.py          # ASHIN 特征缓存
openhgnn/ashin/logger.py         # 实验日志、参数、环境信息记录
openhgnn/ashin/validator.py      # ASHIN 构造检查
```

ASHIN 接入 OpenHGNN 的位置：

```text
main.py
openhgnn/experiment.py
openhgnn/trainerflow/base_flow.py
openhgnn/layers/HeteroLinear.py
```

常用实验脚本：

```text
scripts/tune_ashin_common.py
scripts/tune_ashin_rgcn.py
scripts/tune_ashin_simplehgn.py
scripts/tune_ashin_hgnn.py
scripts/run_ashin_rgcn_simplehgn_seed05.ps1
scripts/run_ashin_gh_rgcn_simplehgn_han.ps1
scripts/summarize_ashin_results.py
scripts/check_data_leakage.py
```

更详细的技术说明在：

```text
docs/ASHIN_BCDEFGH_TECHNICAL_GUIDE_TYPORA.md
```

## 环境

当前代码是在 Windows + PowerShell 下跑的。已有实验使用的环境大致是：

```text
Python 3.10
PyTorch 2.2.1 + CUDA 11.8
DGL 2.2.1 + CUDA 11.8
Optuna
```

如果用本机已经配置好的环境，先进入项目目录：

```powershell
cd E:\OpenHGNN-513\OpenHGNN-513
conda activate openhgnn2025511
```

## 单次运行

不使用 ASHIN，跑原始模型：

```powershell
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0
```

使用 ASHIN-B：

```powershell
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_ashin --ashin_version B
```

使用已经搜索好的 ASHIN best config：

```powershell
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_ashin --ashin_version B --use_best_config --seed 0
```

`--use_best_config --use_ashin` 会去这里找最优参数：

```text
openhgnn/output/optuna/ashin_<Model>_<Dataset>_<Version>/best_params.json
```

比如：

```text
openhgnn/output/optuna/ashin_RGCN_ohgbn-acm_B/best_params.json
```

## 搜索超参数

RGCN：

```powershell
python scripts\tune_ashin_rgcn.py --dataset ohgbn-acm --ashin_version B --n_trials 50 --gpu 0 --seed 0
```

SimpleHGN：

```powershell
python scripts\tune_ashin_simplehgn.py --dataset ohgbn-acm --ashin_version B --n_trials 50 --gpu 0 --seed 0
```

通用入口，适合 HAN、RSHN、HPN、HGT，也可以跑 RGCN 和 SimpleHGN：

```powershell
python scripts\tune_ashin_hgnn.py --model HAN --dataset ohgbn-acm --ashin_version G --n_trials 50 --gpu 0 --seed 0
```

调参目标看验证集，不用测试集选参数。最终参数会保存为 `best_params.json`。

## 多 seed 测试

如果只想用已有 best config，把 RGCN 和 SimpleHGN 在两个数据集、B 到 H、seed 0 到 5 都跑一遍：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_ashin_rgcn_simplehgn_seed05.ps1 -Gpu 0 -Seeds 0,1,2,3,4,5
```

如果只想补缺失的 seed，不想重复跑已有结果：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\run_ashin_rgcn_simplehgn_seed05.ps1 -Gpu 0 -Seeds 0,1,2,3,4,5 -OnlyMissing
```

跑完后会在下面生成新的结果目录：

```text
openhgnn/output/overnight/ashin_rgcn_simplehgn_seed05_<时间戳>/
```

里面会有：

```text
ashin_rgcn_simplehgn_seed05.log
ashin_rgcn_simplehgn_seed05_results.xlsx
```

## 结果文件

最近整理好的结果表在：

```text
openhgnn/output/overnight/ashin_bcdefgh_unified_results_utf8_fixed.xlsx
openhgnn/output/overnight/ashin_bcdefgh_seed05_best_epoch_single_runs.xlsx
openhgnn/output/overnight/ashin_bcdefgh_seed05_best_epoch_top3_highlight.xlsx
```

其中：

- `ashin_bcdefgh_unified_results_utf8_fixed.xlsx`：把 B/C/D/E/F/G/H 的已有结果合到一个总表。
- `ashin_bcdefgh_seed05_best_epoch_single_runs.xlsx`：一行一个 seed，不求平均，记录 best epoch 对应的 Macro-F1 和 Micro-F1。
- `ashin_bcdefgh_seed05_best_epoch_top3_highlight.xlsx`：在每个模型、数据集、seed 下，对 B 到 H 的前三名做了颜色标注。

每一次单独训练的详细日志在：

```text
openhgnn/output/ashin_logs/
```

每个目录里通常有：

```text
train.log
metrics.json
args.json
ashin_metadata.json
graph_info.json
environment.txt
```

`metrics.json` 记录最终指标和 best epoch；`train.log` 记录每个 epoch 的训练过程。

## 数据集

当前主要使用：

```text
openhgnn/dataset/ohgbn-acm
openhgnn/dataset/ohgbn-imdb
```

`ohgbn-acm` 的目标节点是 `paper`，任务是论文类别预测。

`ohgbn-imdb` 的目标节点是 `movie`，任务是电影类别预测。

数据集说明可以看：

```text
openhgnn/dataset/ohgb.md
docs/ASHIN_BCDEFGH_TECHNICAL_GUIDE_TYPORA.md
```

## 打包时怎么处理大文件

如果只是发代码，可以不带这些运行产物：

```text
openhgnn/output/ashin_cache
openhgnn/output/ashin_logs
openhgnn/output/RGCN
openhgnn/output/SimpleHGN
openhgnn/output/HAN
openhgnn/output/HPN
openhgnn/output/HGT
openhgnn/output/RSHN
```

`openhgnn/output/optuna` 建议保留。这里面有 `best_params.json`，直接用 `--use_best_config` 复现实验会用到它。

`openhgnn/output/overnight` 里放了整理好的 Excel。要发结果表就保留；只发代码可以不带。

数据集目录不要整个删掉，因为里面有加载数据用的 `.py` 文件。可以不带不用的数据，例如：

```text
openhgnn/dataset/dblp4MAGNN
```

如果希望别人不用重新下载数据，就保留：

```text
openhgnn/dataset/ohgbn-acm
openhgnn/dataset/ohgbn-imdb
```

Python 缓存可以不带：

```text
__pycache__
```

## 常见提醒

- 不同 seed 的结果会有波动，论文表格里最好写 `mean ± std`，不要挑一个最好看的 seed。
- 调参时只看验证集，测试集只用于最后报告。
- 如果 `--use_best_config` 报找不到文件，先检查对应组合的 `best_params.json` 是否存在。
- ASHIN cache 很大，删掉不会影响代码，只是下次运行会重新生成。
