# Lisan ACM/DBLP 接入 OpenHGNN 使用说明

本文件说明如何把当前 `Datasets/ACM` 和 `Datasets/DBLP` 注册为 OpenHGNN 数据集，并用于节点分类、链接预测和超参数搜索。

## 数据集名称

- `lisan-acm`: 对应当前项目的 `Datasets/ACM`
- `lisan-dblp`: 对应当前项目的 `Datasets/DBLP`

这两个名字刻意避开 OpenHGNN 内置的 `HGBn-ACM`、`HGBl-ACM`、`HGBn-DBLP`、`HGBl-DBLP`。

## 转换数据

先确保运行环境已经安装 `torch` 和 `dgl`，然后在本项目根目录运行：

```powershell
python .\scripts\convert_lisan_to_openhgnn.py --datasets-root .\Datasets --openhgnn-root .\OpenHGNN --dataset all
```

生成 EMat 静态特征版本时，先运行诊断和预处理，再转换指定 feature mode：

```powershell
python .\scripts\diagnose_emat_features.py --datasets lisan-acm lisan-dblp --output_dir .\data\lisan_processed_features
python .\scripts\preprocess_emat_svd.py --datasets lisan-acm lisan-dblp --svd_dims 64 128 256 --output_dir .\data\lisan_processed_features
python .\scripts\preprocess_emat_tfidf.py --datasets lisan-acm lisan-dblp --output_dir .\data\lisan_processed_features
python .\scripts\preprocess_emat_tfidf_svd.py --datasets lisan-acm lisan-dblp --svd_dims 64 128 256 --output_dir .\data\lisan_processed_features
python .\scripts\preprocess_emat_sparse_inputs.py --datasets lisan-acm lisan-dblp --output_dir .\data\lisan_processed_features
python .\scripts\convert_lisan_to_openhgnn.py --datasets-root .\Datasets --emat-root .\Dataset_Emat --openhgnn-root .\OpenHGNN --processed-feature-root .\data\lisan_processed_features --dataset all --feature_mode emat_svd_128
```

输出位置：

```text
E:\Lisan_project\OpenHGNN\openhgnn\dataset\lisan_hgb\lisan-acm.bin
E:\Lisan_project\OpenHGNN\openhgnn\dataset\lisan_hgb\lisan-dblp.bin
E:\Lisan_project\OpenHGNN\openhgnn\dataset\lisan_hgb\lisan-acm-emat.bin
E:\Lisan_project\OpenHGNN\openhgnn\dataset\lisan_hgb\lisan-dblp-emat.bin
E:\Lisan_project\OpenHGNN\openhgnn\dataset\lisan_hgb\lisan-acm-emat-svd-128.bin
E:\Lisan_project\OpenHGNN\openhgnn\dataset\lisan_hgb\lisan-dblp-emat-svd-128.bin
E:\Lisan_project\OpenHGNN\openhgnn\dataset\lisan_hgb\lisan-acm-emat-tfidf-3025.bin
E:\Lisan_project\OpenHGNN\openhgnn\dataset\lisan_hgb\lisan-dblp-emat-tfidf-3025.bin
E:\Lisan_project\OpenHGNN\openhgnn\dataset\lisan_hgb\lisan-acm-raw-emat-tfidf-svd-128.bin
E:\Lisan_project\OpenHGNN\openhgnn\dataset\lisan_hgb\lisan-dblp-raw-emat-tfidf-svd-128.bin
```

`--feature_mode A|B|C|D|E` 的含义：

- `A` / `raw`: 原始 `node.dat` 特征。
- `B` / `emat`: 直接使用 EMat 转置后的基坐标特征。
- `C` / `emat_svd_128`: 读取预处理产物，对每个节点类型的 EMat 单独做 TruncatedSVD 到 128 维。
- `D` / `emat_tfidf_3025`: 读取预处理产物，对 EMat 做 log1p TF-IDF/IDF 加权，保留全基维度；DBLP 是 14314 维。
- `E` / `raw_emat_sparse_encoder`: 可学习 sparse encoder runtime 模式；当前只生成 CSR 输入和 encoder 模块，通用 RGCN sweep 会跳过。

## 划分策略

节点分类：

- `label.dat` 作为原始训练标签池。
- 从 `label.dat` 内按类别分层切出 20% 作为验证集。
- 剩余 80% 作为训练集。
- `label.dat.test` 作为测试集。
- `label.dat.test_full` 作为额外完整测试标签，保存在 `test_full_mask`，默认 OpenHGNN 流程不直接使用。

链接预测：

- `lisan-acm` 默认预测 `paper-ref-paper`，反向边是 `paper-cite-paper`。
- `lisan-dblp` 默认预测 `author-paper`，反向边是 `paper-author`。
- OpenHGNN 链接预测数据集会按 70/10/20 生成 train/valid/test 边划分，并同步处理反向边，避免反向边泄漏。

## 命令示例

节点分类训练和测试：

```powershell
conda activate lisan-openhgnn
cd E:\Lisan_project\OpenHGNN
python .\main.py -m RGCN -d lisan-acm -t node_classification -g -1 --max_epoch 50 --patience 10
python .\main.py -m HAN -d lisan-dblp -t node_classification -g -1 --max_epoch 50 --patience 10
python .\main.py -m RGCN -d lisan-acm -t node_classification -g -1 --feature_mode C --max_epoch 50 --patience 10
python .\main.py -m RGCN -d lisan-acm -t node_classification -g -1 --feature_mode raw_emat_tfidf_svd_128 --max_epoch 50 --patience 10
```

链接预测训练和测试：

```powershell
conda activate lisan-openhgnn
cd E:\Lisan_project\OpenHGNN
python .\main.py -m RGCN -d lisan-acm -t link_prediction -g -1 --max_epoch 50 --patience 10
python .\main.py -m RGCN -d lisan-dblp -t link_prediction -g -1 --max_epoch 50 --patience 10
python .\main.py -m RGCN -d lisan-dblp -t link_prediction -g -1 --feature_mode C --max_epoch 50 --patience 10
```

当前 OpenHGNN 的 `main.py` 没有暴露 `--score_fn` 命令行参数；RGCN 链接预测会从配置中使用默认 `distmult` 打分函数。

超参数搜索：

```powershell
conda activate lisan-openhgnn
cd E:\Lisan_project\OpenHGNN
python .\scripts\lisan_hpo.py --model RGCN --dataset lisan-acm --task node_classification --gpu -1 --trials 20
python .\scripts\lisan_hpo.py --model RGCN --dataset lisan-dblp --task link_prediction --gpu -1 --trials 20
python .\scripts\lisan_hpo.py --model RGCN --dataset lisan-acm --task node_classification --feature_mode C --gpu -1 --trials 20
```

快速检查 HPO 入口时可以临时限制 epoch：

```powershell
python .\scripts\lisan_hpo.py --model RGCN --dataset lisan-acm --task node_classification --gpu -1 --trials 1 --max_epoch 2 --patience 1
python .\scripts\lisan_hpo.py --model RGCN --dataset lisan-dblp --task link_prediction --gpu -1 --trials 1 --max_epoch 1 --patience 1
python .\scripts\lisan_hpo.py --model RGCN --dataset lisan-acm --task node_classification --feature_mode C --gpu -1 --trials 1 --max_epoch 2 --patience 1
```

`lisan_hpo.py` 的链接预测搜索固定使用已验证可运行的 `distmult`，避免 `dot-product` 在当前异构 RGCN 路径中缺少 `x` 字段导致失败。

## 改特征方式

后续如果要替换节点特征，建议改 `scripts/convert_lisan_to_openhgnn.py` 中写入 `graph.nodes[node_type].data["h"]` 的部分，重新生成 `.bin`。不要直接改原始 `node.dat`，这样能保持原始数据和实验数据分离。

当前静态 EMat 版本只替换节点特征 `h`：节点分类的 `label/train_mask/val_mask/test_mask/test_full_mask` 使用同一套代码生成；链接预测仍由 OpenHGNN 在相同边集合上按同一逻辑切分。命令行用 `--feature_mode A|B|C|D|raw_emat_svd_128|raw_emat_tfidf_svd_128` 等选择静态特征版本。`E/raw_emat_sparse_encoder` 不是静态图特征，需要后续 runtime wrapper 接入。可用 `scripts/validate_lisan_feature_graphs.py` 复检图文件一致性。
