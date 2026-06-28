"""
version_g.py

ASHIN-G。

G 先站在属性节点视角构造 ASHIN-C，再沿目标-属性边聚合回目标节点。
ACM 里通常是 author/subject 回到 paper；IMDB 里是 actor/director 回到 movie。
"""

from collections import OrderedDict

import torch

from .transform import normalize_features
from .version_c import build_ashin_c


def adjacent_attribute_ntypes(g, target_ntype):
    # G 的“属性节点”就是和目标节点一跳相连、且类型不同的节点。
    # 这里不写死 author/subject/actor/director，保证换数据集时还能按 schema 自动找。
    attrs = set()
    for src_ntype, _, dst_ntype in g.canonical_etypes:
        if src_ntype == target_ntype and dst_ntype != target_ntype:
            attrs.add(dst_ntype)
        elif dst_ntype == target_ntype and src_ntype != target_ntype:
            attrs.add(src_ntype)
    return sorted(attrs, key=str)


def target_attribute_pairs(g, target_ntype, attr_ntype):
    # 返回统一方向的 pair: [target_id, attr_id]。
    # 原图里可能同时有 target->attr 和 attr->target，先统一方向再去重，后面聚合才不会重复算。
    pairs = []
    for etype in g.canonical_etypes:
        src_ntype, _, dst_ntype = etype
        src, dst = g.edges(etype=etype)
        src = src.cpu().long()
        dst = dst.cpu().long()
        if src_ntype == target_ntype and dst_ntype == attr_ntype:
            pairs.append(torch.stack([src, dst], dim=1))
        elif src_ntype == attr_ntype and dst_ntype == target_ntype:
            pairs.append(torch.stack([dst, src], dim=1))
    if not pairs:
        return torch.empty((0, 2), dtype=torch.long)
    all_pairs = torch.cat(pairs, dim=0)
    return torch.unique(all_pairs, dim=0)


def aggregate_attr_to_target(num_target_nodes, pairs, attr_features, agg="mean"):
    # 把属性节点自己的 ASHIN-C 特征沿边聚合回目标节点。
    # mean/sum 用 index_add_，max 因为要逐维取最大值，直接按 pair 循环更清楚。
    out = torch.zeros((num_target_nodes, attr_features.shape[1]), dtype=torch.float32)
    if pairs.numel() == 0:
        return out

    target_ids = pairs[:, 0].long()
    attr_ids = pairs[:, 1].long()
    values = attr_features[attr_ids].float()

    if agg == "max":
        out.fill_(float("-inf"))
        for target_id, value in zip(target_ids.tolist(), values):
            out[target_id] = torch.maximum(out[target_id], value)
        out[out == float("-inf")] = 0.0
        return out

    out.index_add_(0, target_ids, values)
    if agg == "mean":
        counts = torch.zeros(num_target_nodes, dtype=torch.float32)
        counts.index_add_(0, target_ids, torch.ones_like(target_ids, dtype=torch.float32))
        out = out / counts.clamp(min=1.0).unsqueeze(1)
    elif agg != "sum":
        raise ValueError("ASHIN-G supports ashin_attr_agg in {mean,sum,max}.")
    return out


def build_ashin_g(g, target_ntype, norm="log1p_zscore", ashin_dim=128, seed=0, attr_agg="mean"):
    num_target_nodes = g.num_nodes(target_ntype)
    attr_ntypes = adjacent_attribute_ntypes(g, target_ntype)
    per_attr = OrderedDict()
    signature_to_id = {}
    signature_freq = []
    reduction = {"method": "attribute_centric_ashin_c", "attr_agg": attr_agg, "attributes": []}
    features = []

    for attr_ntype in attr_ntypes:
        # 先站在属性节点视角构造 ASHIN-C，再把它回传给目标节点。
        # 这是 G 和普通目标节点 ASHIN-C 最大的区别。
        attr_result = build_ashin_c(g, attr_ntype, norm="none", ashin_dim=ashin_dim, seed=seed)
        pairs = target_attribute_pairs(g, target_ntype, attr_ntype)
        aggregated = aggregate_attr_to_target(num_target_nodes, pairs, attr_result["x_ashin"], agg=attr_agg)
        features.append(aggregated)
        per_attr[str(attr_ntype)] = {
            "num_attr_nodes": int(g.num_nodes(attr_ntype)),
            "num_target_attr_pairs": int(pairs.shape[0]),
            "attr_signature_count": int(attr_result["num_signatures"]),
            "attr_reduction": attr_result["reduction"],
        }
        reduction["attributes"].append({str(attr_ntype): per_attr[str(attr_ntype)]})
        signature_to_id[str(attr_ntype)] = attr_result["signature_to_id"]
        signature_freq.append({"attr_ntype": str(attr_ntype), "top10": attr_result["signature_freq"][:10]})

    if features:
        # 每一种属性类型占一个 ashin_dim block。
        # 例如 ACM 里通常是 author block + subject block。
        x_raw = torch.cat(features, dim=1)
    else:
        x_raw = torch.zeros((num_target_nodes, ashin_dim), dtype=torch.float32)
    x_ashin = normalize_features(x_raw, norm)

    return {
        "x_ashin": x_ashin,
        "x_b_raw": x_raw,
        "signature_to_id": signature_to_id,
        "signature_freq": signature_freq,
        "num_signatures": int(sum(item["attr_signature_count"] for item in per_attr.values())),
        "reduction": reduction,
    }
