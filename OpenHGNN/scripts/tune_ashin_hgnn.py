"""
tune_ashin_hgnn.py

通用 HGNN + ASHIN 调参入口。

默认模型是 HAN，也可以用 --model 覆盖成 RGCN、SimpleHGN、RSHN、HPN、HGT。
搜索空间和 objective 在 scripts/tune_ashin_common.py。
"""

from tune_ashin_common import main


if __name__ == "__main__":
    main(default_model="HAN")
