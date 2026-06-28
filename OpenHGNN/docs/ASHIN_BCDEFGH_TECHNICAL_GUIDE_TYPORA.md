# ASHIN-BCDEFGH 各个版本技术说明与运行手册

[TOC]

<div style="page-break-after: always;"></div>

> 师兄，这个文档用于说明本项目中 ASHIN-B、ASHIN-C、ASHIN-D、ASHIN-E、ASHIN-F、ASHIN-G、ASHIN-H 七个版本的设计思想、数据处理流程、数值化示例、核心源码位置、实验任务、超参数搜索方式、best config 测试方式以及可直接执行的命令行。

## 1. 项目概览

本项目是在 OpenHGNN 框架上增加 ASHIN 结构特征增强器，用于异质图节点分类任务。ASHIN 不替代 RGCN、SimpleHGN、HAN、RSHN、HPN、HGT 等 HGNN 模型，而是在模型训练前先从异质图结构中构造额外的节点结构特征，再把这些结构特征拼接或门控融合到节点输入特征中。

当前主要实验数据集是：

| 数据集 | 任务 | 目标节点类型 | 典型非目标节点类型 | 评价指标 |
|---|---|---|---|---|
| `ohgbn-acm` | node classification | paper | author、subject 等 | Macro-F1、Micro-F1 |
| `ohgbn-imdb` | node classification | movie | actor、director 等 | Macro-F1、Micro-F1 |

### 1.1 数据集信息（来源于openhgnn）

| 数据集 | 目标节点 | 节点类型与数量 | 有向边类型与数量 | 类别数与类别语义 | 划分 | 特征信息 | 常用元路径 |
|---|---|---|---|---|---|---|---|
| `ohgbn-acm` | `paper` | `paper`: 3,025；`author`: 5,912；`subject`: 57；总节点数 8,994 | `paper-author`: 9,936；`author-paper`: 9,936；`paper-subject`: 3,025；`subject-paper`: 3,025；总有向边数 25,922 | 3 类：Database、Wireless Communication、Data Mining | train/valid/test = 600/300/2,125；训练集每类 200，验证集每类 100 | 所有节点类型都有 `h` 特征；`h` 维度为 1,902；同时包含 `pap_m2v_emb`、`psp_m2v_emb`、`pspap_m2v_emb` 等 64 维元路径特征 | `PAP`: paper-author-paper；`PSP`: paper-subject-paper |
| `ohgbn-imdb` | `movie` | `movie`: 4,661；`actor`: 5,841；`director`: 2,270；总节点数 12,772 | `movie-actor`: 13,983；`actor-movie`: 13,983；`movie-director`: 4,661；`director-movie`: 4,661；总有向边数 37,288 | 3 类：Action、Comedy、Drama | train/valid/test = 300/300/2,339；训练集每类 100，验证集每类 100；另有 1,722 个 `movie` 节点 label=-1，不参与监督评价 | 所有节点类型都有 `h` 特征；`h` 维度为 1,256；`actor`、`director` 为非目标节点但同样有特征 | `MAM`: movie-actor-movie；`MDM`: movie-director-movie |

补充说明：

- 两个数据集在本文实验中都是**单标签多分类节点分类**任务，不是多标签分类任务；使用 Macro-F1 和 Micro-F1 作为评价指标不代表任务一定是多标签分类。
- 两个数据集都**显式包含反向边**，因此消息传递模型可以同时利用正向和反向关系。
- 对 ASHIN 来说，`ohgbn-acm` 中的 `author`、`subject`，以及 `ohgbn-imdb` 中的 `actor`、`director` 都是重要的属性节点类型，ASHIN-G/H 的设计主要围绕这些**属性节点**与**目标节点**之间的结构关系展开（**基于师兄你说的改进一和改进二**）。

### 1.2 concat 融合与 gated 融合(后续几种方法主要的融合方式)

ASHIN 会先构造一组额外的结构特征，再把它和节点原始特征一起送入下游 HGNN。这里的“融合方式”指的是原始特征 `raw_h` 和 ASHIN 结构特征 `ashin_h` 如何合并。

| 融合方式 | 含义 | 数学形式 | 使用版本 | 直观理解 |
|---|---|---|---|---|
| `concat` | 直接拼接原始特征和 ASHIN 特征 | `h_new = [raw_h ; ashin_h]` | B、C、E | 把 ASHIN 特征当作额外输入维度，直接追加到原始特征后面 |
| `gated` | 先分别映射原始特征和 ASHIN 特征，再学习一个门控权重决定 ASHIN 特征贡献多少 | `out = raw_z + gate × ashin_z` | D、F、G、H | 模型自动判断每一维结构特征是否有用，避免 ASHIN 特征简单拼接后带来过多噪声 |

举例来说，如果一个节点的原始特征是：

```text
raw_h = [1.0, 0.2]
```

ASHIN 结构特征是：

```text
ashin_h = [0.8, 0.1]
```

`concat` 融合会直接得到：

```text
h_new = [1.0, 0.2, 0.8, 0.1]
```

`gated` 融合不会直接把拼接结果作为最终输入，而是先学习：

```text
raw_z   = Linear(raw_h)
ashin_z = Linear(ashin_h)
gate    = sigmoid(Linear([raw_h ; ashin_h]))
out     = raw_z + gate × ashin_z
```

因此，`concat` 更简单，保留全部 ASHIN 信息；`gated` 更灵活，允许模型降低无用或有噪声的 ASHIN 特征影响。

整体流程是：

1. 通过 `main.py` 读取模型、数据集、任务、GPU、seed、ASHIN 参数。
2. `Experiment` 构建训练配置，如果启用 `--use_ashin --use_best_config`，会从 ASHIN 的 Optuna 输出中加载 `best_params.json`。
3. `BaseFlow` 拿到 DGL 异质图后，如果 `args.use_ashin=True`，调用 `openhgnn/ashin/builder.py` 中的 `apply_ashin_features()`。
4. **ASHIN** 根据版本 **B/C/D/E/F/G/H** 构造结构特征。
5. 构造后的结构特征写回图的节点特征字段 `h` 或 `feat`。
6. 对 B/C/E 使用普通拼接输入；对 D/F/G/H 使用 `HeteroFeature` 中的可训练门控融合。
7. 原始 HGNN 模型继续按 OpenHGNN 训练流程训练、验证和测试。
8. 调参脚本通过 Optuna 搜索验证集最优参数，并保存 `best_params.json`。
9. 最终测试阶段用 `--use_best_config` 加载最优超参数，并在多个 seed 上重复运行，最后汇总成 Excel。

### 1.3. 对比基准（来源于openhgnn官方）

本项目节点分类任务报告 Macro-F1 和 Micro-F1：

| 指标     | 含义                                                         |
| -------- | ------------------------------------------------------------ |
| Macro-F1 | 先分别计算每个类别的 F1，再对类别取平均。它更关注少数类表现。 |
| Micro-F1 | 先累计所有类别的 TP/FP/FN，再整体计算 F1。它更受样本数多的类别影响。 |

在单标签多分类任务中也可以使用 Macro-F1 和 Micro-F1。它们不意味着任务一定是多标签分类。对 `ohgbn-acm` 和 `ohgbn-imdb`，当前任务是目标节点分类，评价同时看 Macro-F1 和 Micro-F1。

#### ohgbn-acm数据集

官方榜单：Evaluation metric is Macro-F1 and Micro-F1.

| `ohgbn-acm` | Macro-F1 | Micro-F1 |
| ----------- | -------- | -------- |
| MHNF        | 0.9259   | 0.9252   |
| RGCN        | 0.9242   | 0.9242   |
| HAN         | 0.9245   | 0.9233   |
| NARS        | 0.9241   | 0.9233   |
| RSHN        | 0.9226   | 0.9233   |
| HPN         | 0.9214   | 0.9200   |
| GTN         | 0.9203   | 0.9200   |
| SimpleHGN   | 0.9189   | 0.9181   |
| CompGCN     | 0.9128   | 0.9125   |
| HGT         | 0.8946   | 0.8941   |

### 我们的最优-acm

| `ohgbn-acm` | Macro_f1 | Micro_f1 |
| ----------- | -------- | -------- |
| RGCN        | 0.9321   | 0.9313   |
| SimpleHGN   | 0.9289   | 0.9400   |
| HAN         | 0.9129   | 0.9120   |

#### ohgbn-imdb数据集

Evaluation metric is Macro-F1 and Micro-F1.

| `ohgbn-imdb` | Macro_f1 | Micro_f1 |
| ------------ | -------- | -------- |
| RGCN         | 0.5757   | 0.6366   |
| RSHN         | 0.5914   | 0.6127   |
| MHNF         | 0.5913   | 0.6114   |
| CompGCN      | 0.5869   | 0.6148   |
| HAN          | 0.5863   | 0.6037   |
| GTN          | 0.5791   | 0.6003   |
| NARS         | 0.5470   | 0.6259   |
| HPN          | 0.5596   | 0.5703   |
| SimpleHGN    | 0.5521   | 0.5635   |
| HGT          | 0.5440   | 0.5519   |

### 我们的最优-imdb

| `ohgbn-acm` | Macro_f1     | Micro_f1 |
| ----------- | ------------ | -------- |
| RGCN        | 0.6159       | 0.6331   |
| SimpleHGN   | 0.6292       | 0.6520   |
| HAN         | 0.54550.5618 |          |

### 详细数据

Test  Macro-F1

![image-20260617182529039](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260617182529039.png)



Test  Micro-F1

![image-20260617182602725](C:\Users\Administrator\AppData\Roaming\Typora\typora-user-images\image-20260617182602725.png)



## 2. ASHIN 的基础（基于师兄你的论文）

ASHIN 的核心思想是把异质图局部结构转成可以输入 HGNN 的数值特征。它不使用 label、mask、test 指标构造特征，属于基于全图结构的无监督特征增强。

### 2.1 Unit 的定义

在最基础的 ASHIN-B/C 中，一个目标节点就是一个 unit core。例如在 `ohgbn-acm` 中：

- 目标节点类型是 `paper`。
- 每篇论文 p<sub>i</sub> 是一个 unit core。
- 该论文连接到的作者、主题、期刊等一跳邻居会被纳入该 unit 的结构描述。

在 `ohgbn-imdb` 中：

- 目标节点类型是 `movie`。
- 每部电影 m<sub>i</sub> 是一个 unit core。
- 该电影连接到的演员、导演等一跳邻居会被纳入该 unit 的结构描述。

### 2.2 Signature 的构造规则

对每个 unit core，ASHIN 会遍历排序后的 canonical edge types。对每一种边类型，记录一组结构统计量：

| 字段 | 含义 |
|---|---|
| `src_ntype` | 边的源节点类型 |
| `relation` | 边关系名 |
| `dst_ntype` | 边的目标节点类型 |
| `direction` | 目标节点在该边中是 `out`、`in` 还是 `none` |
| `core_degree` | unit core 在该关系下的度 |
| `neighbor_count` | 一跳邻居数量 |
| `neighbor_degree_sum` | 邻居节点在该关系下的度之和 |
| `neighbor_degree_mean_x1000` | 邻居度均值乘以 1000 后取整数 |
| `neighbor_degree_max` | 邻居度最大值 |

一个节点在所有关系上的统计量拼成一个 tuple，这个 tuple 就是该节点的 signature。

### 2.3 示例

为了让每个版本能用同一个例子解释，下面定义一个小型 ACM 风格图：

目标节点是论文：

```text
paper: p0, p1, p2
author: a0, a1
subject: s0, s1
```

论文-作者关系：

```text
p0 - a0
p0 - a1
p1 - a1
p2 - a0
```

论文-主题关系：

```text
p0 - s0
p1 - s0
p2 - s1
```

作者度：

```text
deg(a0)=2，因为连接 p0,p2
deg(a1)=2，因为连接 p0,p1
```

主题度：

```text
deg(s0)=2，因为连接 p0,p1
deg(s1)=1，因为连接 p2
```

如果只看 `paper-author` 和 `paper-subject` 两类关系，则三个论文节点的结构统计为：

| paper | author 统计 `(core_degree, count, degree_sum, mean_x1000, max)` | subject 统计 `(core_degree, count, degree_sum, mean_x1000, max)` |
|---|---|---|
| p<sub>0</sub> | `(2, 2, 4, 2000, 2)` | `(1, 1, 2, 2000, 2)` |
| p<sub>1</sub> | `(1, 1, 2, 2000, 2)` | `(1, 1, 2, 2000, 2)` |
| p<sub>2</sub> | `(1, 1, 2, 2000, 2)` | `(1, 1, 1, 1000, 1)` |

因此三个 paper 的 signature 都不同下面假设 signature id 映射为：

```text
signature(p0) -> 0
signature(p1) -> 1
signature(p2) -> 2
```

## 3. ASHIN-B：目标节点 直接Signature One-Hot

### 3.1 设计思想

ASHIN-B 是最基础版本。它只增强目标节点类型，例如 `ohgbn-acm` 只增强 `paper`，`ohgbn-imdb` 只增强 `movie`。

处理流程：

1. 每个目标节点作为一个 unit core。
2. 对每个目标节点构造 deterministic signature。
3. 收集所有不同 signature，形成 signature vocabulary。
4. 每个目标节点用一个 one-hot 向量表示它属于哪个 signature。
5. 默认不归一化，即保持严格 0/1 incidence 特征。
6. 把 ASHIN-B 特征拼接到目标节点原始特征后面。

### 3.2 数值示例

根据上面的示例图，三个 paper 的 signature 分别映射为 0、1、2，则 ASHIN-B 特征矩阵为：

```text
X_B =
       sig0 sig1 sig2
p0      1    0    0
p1      0    1    0
p2      0    0    1
```

如果原始 paper 特征是：

```text
X_raw =
p0 [1.0, 0.2]
p1 [0.4, 0.7]
p2 [0.3, 0.5]
```

拼接后输入 HGNN 的 paper 特征为：

```text
X_new =
p0 [1.0, 0.2, 1, 0, 0]
p1 [0.4, 0.7, 0, 1, 0]
p2 [0.3, 0.5, 0, 0, 1]
```

### 3.3 适用性

优点：

- 信息最直接，不损失 signature 区分度。
- 可解释性最强，每一列对应一种结构模式。

风险：

- signature 数量多时维度很高。
- one-hot 很稀疏，可能增加模型输入噪声。
- 小数据集上可能容易把训练结构记得过细，泛化不稳定。

## 4. ASHIN-C：ASHIN-B 的无监督压缩版

### 4.1 设计思想

ASHIN-C 不是重新定义结构特征，而是先构造 ASHIN-B 的 one-hot incidence 矩阵，再用无监督降维得到低维 dense 特征。

处理流程：

1. 先运行 ASHIN-B，得到 `X_B`。
2. 对 `X_B` 做无监督降维。
3. 优先使用 `sklearn.decomposition.TruncatedSVD`。
4. 如果 sklearn 不可用，则回退到 `torch.pca_lowrank`。
5. 输出维度由 `--ashin_dim` 控制。
6. 可通过 `--ashin_norm` 做 `none`、`zscore`、`log1p_zscore` 等归一化。
7. 把 ASHIN-C 特征拼接到目标节点原始特征后面。

### 4.2 数值示例

示例图中：

```text
X_B =
p0 [1, 0, 0]
p1 [0, 1, 0]
p2 [0, 0, 1]
```

假设降到 2 维时学到一个无监督投影矩阵：

```text
W =
sig0 [ 0.8,  0.1]
sig1 [ 0.1,  0.9]
sig2 [ 0.6, -0.2]
```

则：

```text
X_C = X_B × W

p0 [0.8,  0.1]
p1 [0.1,  0.9]
p2 [0.6, -0.2]
```

如果原始 paper 特征是：

```text
p0 [1.0, 0.2]
p1 [0.4, 0.7]
p2 [0.3, 0.5]
```

拼接后为：

```text
p0 [1.0, 0.2, 0.8,  0.1]
p1 [0.4, 0.7, 0.1,  0.9]
p2 [0.3, 0.5, 0.6, -0.2]
```

注意：上面的 `W` 是示例矩阵。实际代码中 `W` 由 TruncatedSVD 或 PCA 根据全图 `X_B` 无监督学习得到。

### 4.3 适用性

优点：

- 相比 B 维度更低。
- dense 特征更适合多数神经网络输入层。
- 保留 signature 的主要变化方向。

风险：

- 降维会损失一部分原始结构区分度。
- `ashin_dim` 过大可能仍然噪声较多，过小可能压缩过度。

## 5. ASHIN-D：ASHIN-C + 目标节点门控融合

### 5.1 设计思想

ASHIN-D 固定使用 ASHIN-C 作为结构特征，但不再简单拼接后直接线性变换，而是在 `HeteroFeature` 中对目标节点启用可训练门控融合。

处理流程：

1. 对目标节点构造 ASHIN-C 特征。
2. 将原始特征和 ASHIN-C 特征先拼接写回图。
3. 在 `HeteroFeature` 中把拼接后的特征拆成两段：
   - `raw_h`
   - `ashin_h`
4. 分别线性映射：
   - `raw_z = Linear(raw_h)`
   - `ashin_z = Linear(ashin_h)`
5. 根据完整输入学习门控：
   - `gate = sigmoid(Linear([raw_h, ashin_h]))`
6. 最终融合：
   - `out = raw_z + gate × ashin_z`
7. 输出再做 L2 normalize。

### 5.2 数值示例

假设某个 paper 节点：

```text
raw_h   = [1.0, 0.2]
ashin_h = [0.8, 0.1]
```

经过可训练线性层后：

```text
raw_z   = [0.40, 0.10]
ashin_z = [0.20, 0.50]
gate    = [0.60, 0.30]
```

则融合前输出：

```text
out = raw_z + gate × ashin_z
    = [0.40, 0.10] + [0.60 × 0.20, 0.30 × 0.50]
    = [0.40, 0.10] + [0.12, 0.15]
    = [0.52, 0.25]
```

L2 范数：

```text
√(0.52² + 0.25²) = 0.577
```

归一化后大约：

```text
[0.52/0.577, 0.25/0.577] = [0.901, 0.433]
```

### 5.3 和 C 的区别

| 版本 | 结构特征来源 | 融合方式 |
|---|---|---|
| C | ASHIN-B 降维得到 ASHIN-C | 直接 concat |
| D | ASHIN-C | 可训练 gate |

D 的核心创新点不是新的结构统计，而是让模型自己学习“原始特征”和“结构特征”之间的融合强度。

## 6. ASHIN-E：所有节点类型都增强

### 6.1 设计思想

ASHIN-E 不只增强目标节点，而是对图中所有节点类型都构造 ASHIN 特征。

处理流程：

1. 遍历所有 node type。
2. 对每一种 node type 都把它临时当作 target node type。
3. 根据 `--ashin_base_version B/C` 构造该节点类型自己的 ASHIN-B 或 ASHIN-C 特征。
4. 把每种节点类型的 ASHIN 特征拼接到该类型原始特征后。
5. 下游 HGNN 得到的是全图多类型节点都增强后的输入。

### 6.2 数值示例

示例图中有三类节点：

```text
paper:  p0,p1,p2
author: a0,a1
subject:s0,s1
```

如果 `--ashin_base_version C --ashin_dim 2`，则可能得到：

```text
paper ASHIN-C:
p0 [0.8, 0.1]
p1 [0.1, 0.9]
p2 [0.6,-0.2]

author ASHIN-C:
a0 [0.3, 0.7]
a1 [0.9, 0.2]

subject ASHIN-C:
s0 [0.5, 0.4]
s1 [0.2, 0.8]
```

如果原始特征是：

```text
p0 [1.0, 0.2]
a0 [0.6]
s0 [0.1]
```

增强后：

```text
p0 [1.0, 0.2, 0.8, 0.1]
a0 [0.6, 0.3, 0.7]
s0 [0.1, 0.5, 0.4]
```

### 6.3 适用性

优点：

- 不再只让目标节点携带结构信息。
- 对依赖中间节点表示传播的模型可能更友好。

风险：

- 所有节点类型都增强，计算和缓存更多。
- 非目标节点的结构特征如果噪声较强，可能影响消息传递。

## 7. ASHIN-F：可选 B/C 基底 + 目标节点门控融合

### 7.1 设计思想

ASHIN-F 和 D 一样只增强目标节点，并使用门控融合。区别是 F 的结构特征基底可以由 `--ashin_base_version` 控制：

- `--ashin_base_version B`：使用 ASHIN-B 作为结构特征。
- `--ashin_base_version C`：使用 ASHIN-C 作为结构特征。

处理流程：

1. 只处理目标节点类型。
2. 根据 `--ashin_base_version` 构造 B 或 C 特征。
3. 将原始特征和 ASHIN 特征拼接写回图。
4. `HeteroFeature` 中启用门控融合：
   - `out = raw_z + gate × ashin_z`

### 7.2 数值示例

如果选择：

```text
--ashin_base_version B
```

对 p<sub>0</sub>：

```text
raw_h   = [1.0, 0.2]
ashin_h = [1, 0, 0]
```

线性层可能得到：

```text
raw_z   = [0.40, 0.10]
ashin_z = [0.70, 0.05]
gate    = [0.20, 0.80]
```

融合：

```text
out = [0.40, 0.10] + [0.20 × 0.70, 0.80 × 0.05]
    = [0.54, 0.14]
```

如果选择：

```text
--ashin_base_version C
```

对 p<sub>0</sub>：

```text
raw_h   = [1.0, 0.2]
ashin_h = [0.8, 0.1]
```

它就类似 D，但 F 允许 Optuna 在 B/C 之间搜索。

### 7.3 和 D 的区别

| 版本 | 是否固定 C | 是否可选 B/C | 是否门控 |
|---|---:|---:|---:|
| D | 是 | 否 | 是 |
| F | 否 | 是 | 是 |

F 的价值是把“高维稀疏 one-hot 是否更有用”和“低维 dense 压缩是否更有用”交给验证集搜索。

## 8. ASHIN-G：属性节点中心增强（基于师兄你的想法改动1）

### 8.1 设计思想

ASHIN-G 来自“优先选择属性类型节点，而不是只看文章/电影自身”的思想。目标节点之间的共性可能主要藏在作者、主题、演员、导演等属性节点中。因此 G 不是先给目标节点构造 ASHIN-C，而是：

1. 找到目标节点相邻的非目标节点类型。
2. 对每一种属性节点类型构造 ASHIN-C。
3. 沿着目标-属性边，把属性节点 ASHIN-C 聚合回目标节点。
4. 不同属性类型的聚合结果拼接。
5. 得到目标节点的 ASHIN-G 特征。
6. 最后对目标节点使用门控融合。

在 `ohgbn-acm` 中，典型形式是：

```text
paper <- author 的结构特征
paper <- subject 的结构特征
```

在 `ohgbn-imdb` 中，典型形式是：

```text
movie <- actor 的结构特征
movie <- director 的结构特征
```

### 8.2 聚合方式

参数：

```text
--ashin_attr_agg mean|sum|max
```

含义：

| 参数 | 含义 |
|---|---|
| `mean` | 对相邻属性节点特征取均值 |
| `sum` | 对相邻属性节点特征求和 |
| `max` | 对相邻属性节点特征逐维取最大值 |

### 8.3 数值示例

假设作者节点经过 ASHIN-C 后：

```text
a0 [0.2, 0.4]
a1 [0.8, 0.1]
```

主题节点经过 ASHIN-C 后：

```text
s0 [0.6, 0.3]
s1 [0.1, 0.9]
```

对 p<sub>0</sub>：

```text
p0 连接 a0,a1
p0 连接 s0
```

如果 `--ashin_attr_agg mean`：

```text
author_agg(p0) = mean([a0, a1])
               = ([0.2,0.4] + [0.8,0.1]) / 2
               = [0.5, 0.25]

subject_agg(p0) = [0.6, 0.3]
```

拼接得到：

```text
ASHIN-G(p0) = [0.5, 0.25, 0.6, 0.3]
```

对 p<sub>1</sub>：

```text
p1 连接 a1 和 s0
ASHIN-G(p1) = [0.8, 0.1, 0.6, 0.3]
```

对 p<sub>2</sub>：

```text
p2 连接 a0 和 s1
ASHIN-G(p2) = [0.2, 0.4, 0.1, 0.9]
```

如果原始特征是：

```text
p0 raw = [1.0, 0.2]
```

写回图中后暂时是：

```text
p0 [1.0, 0.2, 0.5, 0.25, 0.6, 0.3]
```

进入 `HeteroFeature` 时会被拆成：

```text
raw_h   = [1.0, 0.2]
ashin_h = [0.5, 0.25, 0.6, 0.3]
```

再使用门控融合。

### 8.4 适用性

优点：

- 更符合许多异质图数据的语义：目标节点相似性经常由共享属性节点决定。
- 能让作者、主题、演员、导演等属性节点的结构模式直接回流到目标节点。

可能的风险：

- 如果属性节点本身结构噪声大，G 会把噪声聚合回目标节点。
- 多个属性类型拼接后维度可能变大，`ashin_dim` 和 `ashin_attr_agg` 需要调参。

## 9. ASHIN-H：目标节点共性修正（基于师兄你的想法二）

### 9.1 设计思想

它先给目标节点构造 ASHIN-C，再利用共享属性节点建立 target-target 共性矩阵，最后用这个共性矩阵传播和修正目标节点结构特征。

处理流程：

1. 对目标节点构造 ASHIN-C，得到 `X_base`。
2. 找到目标节点相邻的属性节点类型。
3. 如果两个目标节点共享同一个属性节点，就认为它们有共性。
4. 根据共享属性构造 target-target 共性权重。
5. 用共性权重对 `X_base` 做传播：
   - `X_H[i] = Σ_j weight(i,j) × X_base[j]`
6. 对 `X_H` 做可选归一化。
7. 最后对目标节点使用门控融合。

### 9.2 共性参数

| 参数 | 可选值 | 含义 |
|---|---|---|
| `--ashin_common_op` | `max`、`sum` | 多条共享属性路径如何合并分数 |
| `--ashin_common_norm` | `row`、`binary` | 共性传播时如何归一化权重 |
| `--ashin_common_topk` | `0`、`10`、`20`、`50` | 每个目标节点保留多少个共性邻居，0 表示全部保留 |

### 9.3 数值示例

先假设 ASHIN-C 得到：

```text
X_base(p0) = [0.8,  0.1]
X_base(p1) = [0.1,  0.9]
X_base(p2) = [0.6, -0.2]
```

从玩具图看：

```text
p0 和 p1 共享 a1
p0 和 p1 共享 s0
p0 和 p2 共享 a0
p1 和 p2 没有共享属性节点
```

如果：

```text
--ashin_common_op sum
--ashin_common_norm row
```

那么 p<sub>0</sub> 到其他 paper 的共性分数为：

```text
score(p0,p0) = 1，自连接
score(p0,p1) = 2，共享 a1 和 s0
score(p0,p2) = 1，共享 a0
```

行归一化：

```text
sum = 1 + 2 + 1 = 4
w(p0,p0) = 1/4 = 0.25
w(p0,p1) = 2/4 = 0.50
w(p0,p2) = 1/4 = 0.25
```

则 H 修正后的 p<sub>0</sub> 结构特征是：

```text
X_H(p0)
= 0.25 × X_base(p0) + 0.50 × X_base(p1) + 0.25 × X_base(p2)
= 0.25 × [0.8,0.1] + 0.50 × [0.1,0.9] + 0.25 × [0.6,-0.2]
= [0.20,0.025] + [0.05,0.45] + [0.15,-0.05]
= [0.40,0.425]
```

如果 `--ashin_common_norm binary`，则只看是否存在共性，不看共享次数。p<sub>0</sub> 的三个目标邻居 p<sub>0</sub>、p<sub>1</sub>、p<sub>2</sub> 权重都是 `1/3`。

### 9.4 适用性

优点：

- 直接建模目标节点之间由共享属性诱导出的共性。
- 对“共享作者、共享主题、共享演员、共享导演”这类模式更敏感。

风险：

- 共性矩阵可能较稠密，因此需要 `topk` 控制传播范围。
- 如果共享属性过于常见，例如一个超高频主题连接大量论文，可能引入过平滑。

## 10. 七个版本的对比表

| 版本 | 增强节点类型 | 结构特征来源 | 是否降维 | 融合方式 | 关键参数 |
|---|---|---|---:|---|---|
| B | 目标节点 | 目标节点 signature one-hot | 否 | concat | `ashin_norm` |
| C | 目标节点 | B 的 incidence 矩阵 | 是 | concat | `ashin_dim`, `ashin_norm` |
| D | 目标节点 | 固定 ASHIN-C | 是 | gated | `ashin_dim`, `ashin_norm` |
| E | 所有节点类型 | 每类节点各自构造 B/C | 可选 | concat | `ashin_base_version`, `ashin_dim`, `ashin_norm` |
| F | 目标节点 | 可选 B/C | 可选 | gated | `ashin_base_version`, `ashin_dim`, `ashin_norm` |
| G | 目标节点 | 属性节点 ASHIN-C 聚合回目标节点 | 是 | gated | `ashin_dim`, `ashin_norm`, `ashin_attr_agg` |
| H | 目标节点 | 目标 ASHIN-C + target-target 共性传播 | 是 | gated | `ashin_dim`, `ashin_norm`, `ashin_common_op`, `ashin_common_norm`, `ashin_common_topk` |

## 11. 核心源码文件

### 11.1 ASHIN 构造逻辑

| 文件 | 作用 |
|---|---|
| `openhgnn/ashin/signature.py` | 构造 ASHIN-B 的 node signature 和 one-hot incidence 矩阵 |
| `openhgnn/ashin/transform.py` | 提供归一化和 ASHIN-C 降维工具 |
| `openhgnn/ashin/version_b.py` | ASHIN-B 专属实现 |
| `openhgnn/ashin/version_c.py` | ASHIN-C 专属实现 |
| `openhgnn/ashin/version_g.py` | ASHIN-G 属性节点中心增强实现 |
| `openhgnn/ashin/version_h.py` | ASHIN-H 目标节点共性修正实现 |
| `openhgnn/ashin/builder.py` | ASHIN 主入口，决定 B/C/D/E/F/G/H 如何构造、缓存、写回图 |
| `openhgnn/ashin/cache.py` | ASHIN 特征缓存路径和读写 |
| `openhgnn/ashin/logger.py` | ASHIN 运行日志、参数、图信息、metrics 保存 |
| `openhgnn/ashin/best_config.py` | `--use_best_config` 时加载 ASHIN 的 `best_params.json` |
| `openhgnn/ashin/validator.py` | ASHIN 构造逻辑的校验工具 |

### 11.2 OpenHGNN 接入点

| 文件 | 作用 |
|---|---|
| `main.py` | 命令行入口，增加 ASHIN 参数 |
| `openhgnn/experiment.py` | 创建实验配置，加载 ASHIN best config，保存结果 |
| `openhgnn/trainerflow/base_flow.py` | 在训练流程开始前调用 `apply_ashin_features()` 写回图特征 |
| `openhgnn/layers/HeteroLinear.py` | `HeteroFeature` 中实现 D/F/G/H 的门控融合 |

### 11.3 调参和批量实验脚本

| 文件 | 作用 |
|---|---|
| `scripts/tune_ashin_common.py` | Optuna 调参公共入口，定义模型和 ASHIN 搜索空间 |
| `scripts/tune_ashin_rgcn.py` | RGCN + ASHIN 调参 wrapper |
| `scripts/tune_ashin_simplehgn.py` | SimpleHGN + ASHIN 调参 wrapper |
| `scripts/tune_ashin_hgnn.py` | HAN/RSHN/HPN/HGT 等通用 HGNN + ASHIN 调参 wrapper |
| `scripts/run_ashin_d_rgcn_simplehgn.ps1` | 批量跑 RGCN/SimpleHGN + ASHIN-D，并汇总 Excel |
| `scripts/run_ashin_gh_rgcn_simplehgn_han.ps1` | 批量跑 RGCN/SimpleHGN/HAN + ASHIN-G/H，并汇总 Excel |
| `scripts/summarize_ashin_results.py` | 扫描日志和 best params，生成 Excel 结果表 |
| `scripts/ashin_validate.py` | ASHIN 构造校验入口 |

## 12. 输出目录说明

| 路径 | 内容 |
|---|---|
| `openhgnn/output/ashin_cache` | ASHIN 特征缓存，避免重复构造 |
| `openhgnn/output/ashin_logs` | 每次训练的参数、日志、metrics |
| `openhgnn/output/optuna` | Optuna SQLite 数据库、best params、trial CSV |
| `openhgnn/output/overnight` | 批量实验日志和最终 Excel |

典型 best params 路径：

```text
openhgnn/output/optuna/ashin_RGCN_ohgbn-acm_G/best_params.json
openhgnn/output/optuna/ashin_SimpleHGN_ohgbn-imdb_H/best_params.json
```

典型训练结果路径：

```text
openhgnn/output/ashin_logs/final_RGCN_ohgbn-acm_ashinG_seed0_YYYYMMDD_HHMMSS/metrics.json
```

## 13. 重要命令行参数

### 13.1 通用 OpenHGNN 参数

| 参数 | 示例 | 含义 |
|---|---|---|
| `-m`, `--model` | `RGCN` | 模型名 |
| `-d`, `--dataset` | `ohgbn-acm` | 数据集名 |
| `-t`, `--task` | `node_classification` | 任务名 |
| `-g`, `--gpu` | `0` | GPU 编号，`-1` 表示 CPU |
| `--seed` | `0` | 随机种子 |
| `--use_best_config` | 无值 flag | 加载 best config |
| `--max_epoch` | `1` | 最大训练 epoch，常用于 smoke test |

### 13.2 ASHIN 参数

| 参数 | 示例 | 含义 |
|---|---|---|
| `--use_ashin` | 无值 flag | 启用 ASHIN |
| `--ashin_version` | `B`/`C`/`D`/`E`/`F`/`G`/`H` | 选择 ASHIN 版本 |
| `--ashin_base_version` | `B`/`C` | E/F 使用的基底版本 |
| `--ashin_dim` | `128` | C/D/E/F/G/H 中降维输出维度 |
| `--ashin_norm` | `none`/`zscore`/`log1p_zscore` | ASHIN 特征归一化方式 |
| `--ashin_attr_agg` | `mean`/`sum`/`max` | G 的属性节点聚合方式 |
| `--ashin_common_op` | `max`/`sum` | H 的共性分数合并方式 |
| `--ashin_common_norm` | `row`/`binary` | H 的共性传播归一化方式 |
| `--ashin_common_topk` | `0`/`10`/`20`/`50` | H 每个目标节点保留的共性邻居数 |
| `--ashin_cache_dir` | `./openhgnn/output/ashin_cache` | ASHIN 缓存目录 |
| `--ashin_rebuild` | 无值 flag | 强制重建 ASHIN 缓存 |
| `--run_name` | `final_RGCN_acm_seed0` | 本次运行名称 |

## 14. 环境配置

当前已验证环境：

```text
Python 3.10.20
torch 2.2.1+cu118
dgl 2.2.1+cu118
optuna 4.8.0
```

推荐在 Windows PowerShell 中执行：

```powershell
cd E:\OpenHGNN-513\OpenHGNN-513
conda create -n openhgnn2025511 python=3.10 -y
conda activate openhgnn2025511
```

安装 PyTorch CUDA 11.8 版本：

```powershell
pip install torch==2.2.1+cu118 --index-url https://download.pytorch.org/whl/cu118
```

安装 DGL CUDA 11.8 版本：

```powershell
pip install dgl==2.2.1 -f https://data.dgl.ai/wheels/torch-2.2/cu118/repo.html
```

安装项目依赖：

```powershell
pip install -r requirements.txt
pip install -e .
```

验证环境：

```powershell
python -c "import torch,dgl,optuna; print(torch.__version__); print(dgl.__version__); print(optuna.__version__)"
```

如果使用当前机器上已经存在的环境，可以直接：

```powershell
conda activate openhgnn2025511
cd E:\OpenHGNN-513\OpenHGNN-513
```

## 15. 单次运行命令

### 15.1 不使用 ASHIN 的原始 RGCN

```powershell
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --seed 0
```

运行内容：

- 使用 RGCN。
- 数据集是 `ohgbn-acm`。
- 任务是节点分类。
- 不启用 ASHIN。
- 结果可作为原始 HGNN baseline。

### 15.2 RGCN + ASHIN-B

```powershell
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_ashin --ashin_version B --ashin_norm none --seed 0
```

运行内容：

- 构造目标节点 ASHIN-B one-hot signature 特征。
- 拼接到目标节点原始特征。
- 用 RGCN 训练节点分类。

### 15.3 RGCN + ASHIN-C

```powershell
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_ashin --ashin_version C --ashin_dim 128 --ashin_norm log1p_zscore --seed 0
```

运行内容：

- 先构造 ASHIN-B incidence。
- 再压缩为 128 维 ASHIN-C。
- 拼接到目标节点原始特征。

### 15.4 RGCN + ASHIN-D

```powershell
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_ashin --ashin_version D --ashin_dim 128 --ashin_norm log1p_zscore --seed 0
```

运行内容：

- 固定使用 ASHIN-C。
- 目标节点使用 raw/ASHIN 门控融合。

### 15.5 RGCN + ASHIN-E

```powershell
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_ashin --ashin_version E --ashin_base_version C --ashin_dim 64 --ashin_norm zscore --seed 0
```

运行内容：

- 对所有节点类型构造 ASHIN 特征。
- 每个节点类型都拼接自己的结构增强特征。

### 15.6 RGCN + ASHIN-F

```powershell
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_ashin --ashin_version F --ashin_base_version C --ashin_dim 128 --ashin_norm none --seed 0
```

运行内容：

- 只增强目标节点。
- 基底可选 B/C。
- 使用 raw/ASHIN 门控融合。

### 15.7 RGCN + ASHIN-G

```powershell
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_ashin --ashin_version G --ashin_dim 128 --ashin_norm none --ashin_attr_agg mean --seed 0
```

运行内容：

- 为目标节点相邻的属性节点构造 ASHIN-C。
- 将属性节点结构特征聚合回目标节点。
- 使用门控融合。

### 15.8 RGCN + ASHIN-H

```powershell
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_ashin --ashin_version H --ashin_dim 128 --ashin_norm none --ashin_common_op sum --ashin_common_norm row --ashin_common_topk 20 --seed 0
```

运行内容：

- 构造目标节点 ASHIN-C。
- 通过共享属性节点建立 target-target 共性矩阵。
- 使用共性矩阵传播修正目标结构特征。
- 使用门控融合。

### 15.9 CPU 快速 smoke test

```powershell
python main.py -m RGCN -d ohgbn-acm -t node_classification -g -1 --use_ashin --ashin_version G --max_epoch 1 --seed 0
```

运行内容：

- 在 CPU 上快速跑 1 个 epoch。
- 主要用于检查代码能否跑通，不用于正式结果。

## 16. Optuna 超参数搜索命令

### 16.1 RGCN 搜索

```powershell
python scripts\tune_ashin_rgcn.py --dataset ohgbn-acm --ashin_version B --n_trials 50 --gpu 0 --seed 0
python scripts\tune_ashin_rgcn.py --dataset ohgbn-acm --ashin_version C --n_trials 50 --gpu 0 --seed 0
python scripts\tune_ashin_rgcn.py --dataset ohgbn-acm --ashin_version D --n_trials 50 --gpu 0 --seed 0
python scripts\tune_ashin_rgcn.py --dataset ohgbn-acm --ashin_version E --n_trials 50 --gpu 0 --seed 0
python scripts\tune_ashin_rgcn.py --dataset ohgbn-acm --ashin_version F --n_trials 50 --gpu 0 --seed 0
python scripts\tune_ashin_rgcn.py --dataset ohgbn-acm --ashin_version G --n_trials 50 --gpu 0 --seed 0
python scripts\tune_ashin_rgcn.py --dataset ohgbn-acm --ashin_version H --n_trials 50 --gpu 0 --seed 0
```

运行内容：

- 每条命令对一个 ASHIN 版本进行 Optuna 搜索。
- 搜索目标是验证集指标。
- 搜索结果写入 `openhgnn/output/optuna/ashin_RGCN_数据集_版本/`。

### 16.2 SimpleHGN 搜索

```powershell
python scripts\tune_ashin_simplehgn.py --dataset ohgbn-imdb --ashin_version B --n_trials 50 --gpu 0 --seed 0
python scripts\tune_ashin_simplehgn.py --dataset ohgbn-imdb --ashin_version C --n_trials 50 --gpu 0 --seed 0
python scripts\tune_ashin_simplehgn.py --dataset ohgbn-imdb --ashin_version D --n_trials 50 --gpu 0 --seed 0
python scripts\tune_ashin_simplehgn.py --dataset ohgbn-imdb --ashin_version E --n_trials 50 --gpu 0 --seed 0
python scripts\tune_ashin_simplehgn.py --dataset ohgbn-imdb --ashin_version F --n_trials 50 --gpu 0 --seed 0
python scripts\tune_ashin_simplehgn.py --dataset ohgbn-imdb --ashin_version G --n_trials 50 --gpu 0 --seed 0
python scripts\tune_ashin_simplehgn.py --dataset ohgbn-imdb --ashin_version H --n_trials 50 --gpu 0 --seed 0
```

### 16.3 通用 HGNN 搜索

`scripts\tune_ashin_hgnn.py` 支持：

```text
RGCN, SimpleHGN, HAN, RSHN, HPN, HGT
```

示例：

```powershell
python scripts\tune_ashin_hgnn.py --model HAN --dataset ohgbn-acm --ashin_version G --n_trials 50 --gpu 0 --seed 0
python scripts\tune_ashin_hgnn.py --model HGT --dataset ohgbn-imdb --ashin_version H --n_trials 50 --gpu 0 --seed 0
python scripts\tune_ashin_hgnn.py --model HPN --dataset ohgbn-acm --ashin_version D --n_trials 50 --gpu 0 --seed 0
```

运行内容：

- 使用统一搜索空间。
- 每个 trial 都通过 subprocess 调用 `main.py`。
- best params 保存为：

```text
openhgnn/output/optuna/ashin_{model}_{dataset}_{ashin_version}/best_params.json
```

## 17. 调参搜索空间

通用搜索空间：

| 参数 | 搜索范围 |
|---|---|
| `lr` | `1e-4` 到 `5e-2`，log scale |
| `weight_decay` | `1e-6` 到 `1e-2`，log scale |
| `dropout` | `0.0` 到 `0.7` |
| `hidden_dim` | `[32, 64, 128, 256]` |
| `patience` | `[30, 50, 100]` |

模型专属搜索空间：

| 模型 | 额外搜索参数 |
|---|---|
| RGCN | `num_layers=[1,2,3]`, `n_bases=[-1,4,8,16]` |
| SimpleHGN | `num_layers=[1,2,3]`, `num_heads=[2,4,8]`, `edge_dim=[16,32,64]`, `negative_slope=[0.01,0.05,0.2]`, `residual=[True,False]`, `beta=[0,1]` |
| HAN | `han_num_heads=[2,4,8]`, `num_layers=1` |
| RSHN | `rw_len=[2,4,6]`, `num_node_layer=[1,2,3]`, `num_edge_layer=[1,2,3]`, `batch_size=[512,1000,2048]` |
| HPN | `k_layer=[1,2,3,4]`, `alpha=[0.05,0.5]`, `edge_drop=[0.0,0.1,0.3,0.5]` |
| HGT | `num_heads=[2,4,8]`, `num_layers=[2,3]`, `batch_size=[1024,2048,5120]`, `fanout=[2,5,10]`, `norm=[True,False]` |

ASHIN 专属搜索空间：

| ASHIN 版本 | 搜索参数 |
|---|---|
| B | `ashin_norm=["none","zscore"]` |
| C | `ashin_dim=[32,64,128,256]`, `ashin_norm=["none","zscore","log1p_zscore"]` |
| D | `ashin_dim=[32,64,128,256]`, `ashin_norm=["none","zscore","log1p_zscore"]` |
| E | `ashin_base_version=["B","C"]`, C 时搜索 `ashin_dim=[32,64,128,256]`, `ashin_norm=["none","zscore","log1p_zscore"]` |
| F | `ashin_base_version=["B","C"]`, C 时搜索 `ashin_dim=[32,64,128,256]`, `ashin_norm=["none","zscore","log1p_zscore"]` |
| G | `ashin_dim=[32,64,128,256]`, `ashin_norm=["none","zscore","log1p_zscore"]`, `ashin_attr_agg=["mean","sum","max"]` |
| H | `ashin_dim=[32,64,128,256]`, `ashin_norm=["none","zscore","log1p_zscore"]`, `ashin_common_op=["max","sum"]`, `ashin_common_norm=["row","binary"]`, `ashin_common_topk=[0,10,20,50]` |

## 18. 使用 best params 做最终测试

调参完成后，使用：

```powershell
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_ashin --ashin_version G --use_best_config --seed 0
```

运行内容：

- 自动读取：

```text
openhgnn/output/optuna/ashin_RGCN_ohgbn-acm_G/best_params.json
```

- 用 best params 训练和测试。
- seed 控制模型初始化、dropout、采样等随机因素。

多 seed 推荐写法：

```powershell
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_ashin --ashin_version G --use_best_config --seed 0 --run_name final_RGCN_ohgbn-acm_ashinG_seed0
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_ashin --ashin_version G --use_best_config --seed 1 --run_name final_RGCN_ohgbn-acm_ashinG_seed1
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_ashin --ashin_version G --use_best_config --seed 2 --run_name final_RGCN_ohgbn-acm_ashinG_seed2
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_ashin --ashin_version G --use_best_config --seed 3 --run_name final_RGCN_ohgbn-acm_ashinG_seed3
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_ashin --ashin_version G --use_best_config --seed 4 --run_name final_RGCN_ohgbn-acm_ashinG_seed4
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_ashin --ashin_version G --use_best_config --seed 5 --run_name final_RGCN_ohgbn-acm_ashinG_seed5
```

注意：

- 超参数搜索阶段通常固定一个 seed，例如 seed 0。
- 最终报告阶段再使用多个 seed 复现实验，报告 `mean ± std`。
- 不应该从多个 seed 里挑最好的 test 结果作为最终论文结果。

## 23. 实验设计流程

公平实验流程：

1. 所有方法使用相同的数据集划分。
2. 每个方法先在验证集上搜索超参数。
3. 超参数搜索阶段固定一个 seed，例如 `seed=0`。
4. 确定 best params 后，再用相同 seed 列表重复最终测试，例如 `0,1,2,3,4,5`。
5. 对所有 baseline 和 ASHIN 方法使用同一组 seed。
6. 报告 `mean ± std`，不要挑单个 seed 的最好 test 结果。

示例：

```text
seed 0 test Macro-F1 = 0.920
seed 1 test Macro-F1 = 0.925
seed 2 test Macro-F1 = 0.918
seed 3 test Macro-F1 = 0.930
seed 4 test Macro-F1 = 0.922

mean = (0.920+0.925+0.918+0.930+0.922)/5 = 0.923
std 约为 0.0046

最终报告：0.923 ± 0.0046
```

## 24. 常见问题

### 24.1 `--use_best_config` 找不到 best_params.json

原因：

- 该模型、数据集、ASHIN 版本还没有完成 Optuna 搜索。
- 或者 best params 路径不符合默认命名规则。

解决：

```powershell
python scripts\tune_ashin_hgnn.py --model RGCN --dataset ohgbn-acm --ashin_version G --n_trials 50 --gpu 0 --seed 0
```

然后再运行：

```powershell
python main.py -m RGCN -d ohgbn-acm -t node_classification -g 0 --use_ashin --ashin_version G --use_best_config --seed 0
```

### 24.2 trial 数超过 n_trials

Optuna 的 `n_trials` 表示“本次再运行多少个 trial”，不是“总共最多多少个 trial”。批量脚本中已经通过读取 completed trial 数量来计算 remaining，避免重复追加过多 trial。

### 24.3 为什么同一个命令不同 seed 结果不同

seed 会影响：

- 模型参数初始化。
- dropout 随机失活。
- 采样或部分训练流程中的随机顺序。
- SVD/PCA 中可能存在的随机初始化。

所以最终论文结果应该多 seed 报告均值和标准差。
