"""
cache.py

ASHIN 特征 cache。

signature 和 SVD 都不便宜，调参时重复算会浪费很多时间，所以按数据集、目标节点类型、
版本、norm、dim、G/H 的额外参数、图 schema 和 seed 生成文件名。

cache 只保存结构特征和构造元信息。别把 label、split、metrics 写进来，否则实验边界会变脏。
"""

import hashlib
import os
from pathlib import Path

import torch


def graph_schema_hash(g):
    # 用图结构摘要识别 cache 是否适用于当前图。
    # 这里不读取任何节点标签或 mask。
    parts = []
    for ntype in sorted(g.ntypes, key=str):
        parts.append(f"ntype:{ntype}:{g.num_nodes(ntype)}")
    for etype in sorted(g.canonical_etypes, key=lambda e: (str(e[0]), str(e[1]), str(e[2]))):
        parts.append(f"etype:{etype[0]}:{etype[1]}:{etype[2]}:{g.num_edges(etype=etype)}")
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]


def _safe_name(value):
    return str(value).replace(os.sep, "_").replace(" ", "_").replace(":", "_")


def cache_path(args, g, target_ntype, version, effective_norm):
    # 根据数据集、目标节点类型、ASHIN 设置和图结构生成稳定 cache 路径。
    # 文件名一定要覆盖会改变特征数值的参数，否则不同实验可能误用同一个 cache。
    dataset_name = getattr(args, "dataset_name", getattr(args, "dataset", "dataset"))
    seed = getattr(args, "seed", 0)
    schema = graph_schema_hash(g)
    root = Path(getattr(args, "ashin_cache_dir", "./openhgnn/output/ashin_cache"))
    folder = root / _safe_name(dataset_name) / f"target_{_safe_name(target_ntype)}"
    version_text = str(version)
    # D/E/F 会以 D_C、E_B、F_C 这样的 cache version 传进来。
    # 只要最终以 C 为基础，或者是 G/H，就需要把 ashin_dim 放进文件名。
    uses_dim = version_text.endswith("C") or version_text.startswith(("G", "H"))
    dim_part = f"_dim{getattr(args, 'ashin_dim', 128)}" if uses_dim else ""
    extra_part = ""
    if version_text.startswith("G"):
        extra_part = f"_agg{_safe_name(getattr(args, 'ashin_attr_agg', 'mean'))}"
    elif version_text.startswith("H"):
        extra_part = (
            f"_op{_safe_name(getattr(args, 'ashin_common_op', 'max'))}"
            f"_cnorm{_safe_name(getattr(args, 'ashin_common_norm', 'row'))}"
            f"_topk{getattr(args, 'ashin_common_topk', 0)}"
        )
    filename = (
        f"ashin{version}{dim_part}{extra_part}_norm{_safe_name(effective_norm)}_"
        f"schema{schema}_nodes{g.num_nodes(target_ntype)}_edges{g.num_edges()}_seed{seed}.pt"
    )
    return folder / filename


def load_cache(path):
    # 读取 CPU cache，避免因为训练设备不同导致加载失败。
    if not Path(path).exists():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def save_cache(path, payload):
    # 保存前自动创建目录；不会删除旧文件。
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
