"""
ashin_validate.py

ASHIN 特征构造检查脚本。

这个入口现在主要检查 B/C：toy graph、真实数据集形状、NaN/Inf、B 的 0/1 incidence、
C 的输出维度和非零、signature 可复现、cache 命中、拼接后的维度。

脚本只做检查，不训练模型。ASHIN 构造仍只读图结构，不读 label 或测试指标。

例子：
python scripts/ashin_validate.py --dataset ohgbn-acm --version B
python scripts/ashin_validate.py --dataset ohgbn-acm --version C --ashin_dim 128
"""

import argparse
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openhgnn.ashin.validator import (  # noqa: E402
    assert_basic,
    default_args,
    run_cache_test,
    run_concat_test,
    run_repro_test,
    run_toy_test,
)
from openhgnn.ashin.builder import build_ashin_features  # noqa: E402
from openhgnn.dataset import build_dataset  # noqa: E402


class _Logger:
    # 给 build_dataset 提供一个最小 logger，避免为了验证脚本构造完整 OpenHGNN Logger。
    def dataset_info(self, msg):
        print("[Dataset]", msg)

    def info(self, msg):
        print(msg)


def load_dataset(name):
    # 复用 OpenHGNN 原始数据集构建逻辑，确保验证对象和训练对象一致。
    dummy_args = SimpleNamespace(model="RGCN", dataset=name, graphbolt=False, seed=0)
    dataset = build_dataset(name, "node_classification", logger=_Logger(), args=dummy_args)
    return dataset


def main():
    # 命令行入口：先跑 toy graph，再跑真实数据集检查。
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ohgbn-acm")
    parser.add_argument("--version", choices=["B", "C"], required=True)
    parser.add_argument("--ashin_dim", type=int, default=128)
    parser.add_argument("--cache_dir", default=None)
    args = parser.parse_args()

    run_toy_test()
    dataset = load_dataset(args.dataset)
    cache_dir = args.cache_dir or tempfile.mkdtemp(prefix="ashin_validate_cache_")
    ashin_args = default_args(args.dataset, args.version, args.ashin_dim, cache_dir=cache_dir)
    payload = build_ashin_features(dataset.g, dataset, ashin_args)
    assert_basic(payload, dataset.g, dataset.category, args.version, args.ashin_dim)
    print("[ASHIN validate] basic shape/value test passed.")

    run_repro_test(dataset.g, dataset, ashin_args)
    run_cache_test(dataset.g, dataset, ashin_args)
    run_concat_test(dataset.g, dataset, ashin_args)
    print("[ASHIN validate] all checks passed.")


if __name__ == "__main__":
    main()
