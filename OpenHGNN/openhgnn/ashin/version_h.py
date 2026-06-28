"""
version_h.py

ASHIN-H。

H 先构造目标节点 ASHIN-C，再用共享属性节点建 target-target 共性分数，
最后按这个分数对目标节点结构特征做一轮传播修正。
"""

from collections import defaultdict

import torch

from .transform import normalize_features
from .version_c import build_ashin_c
from .version_g import adjacent_attribute_ntypes, target_attribute_pairs


def _merge_score(old_score, new_score, op):
    if op == "max":
        return max(old_score, new_score)
    if op == "sum":
        return old_score + new_score
    raise ValueError("ASHIN-H supports ashin_common_op in {max,sum}.")


def build_commonality_rows(g, target_ntype, common_op="max"):
    # rows[i] 记录目标节点 i 和其他目标节点的共性分数。
    # 共性来自共享属性节点：两篇 paper 共享 author/subject，或两部 movie 共享 actor/director。
    num_target_nodes = g.num_nodes(target_ntype)
    rows = [defaultdict(float) for _ in range(num_target_nodes)]
    attr_stats = []

    for attr_ntype in adjacent_attribute_ntypes(g, target_ntype):
        pairs = target_attribute_pairs(g, target_ntype, attr_ntype)
        attr_to_targets = defaultdict(list)
        for target_id, attr_id in pairs.tolist():
            attr_to_targets[int(attr_id)].append(int(target_id))

        for targets in attr_to_targets.values():
            # 同一个属性节点连接到的一组 target 两两建立共性。
            # max 表示“共享过即可”，sum 表示“共享次数越多分数越高”。
            unique_targets = sorted(set(targets))
            for src in unique_targets:
                row = rows[src]
                for dst in unique_targets:
                    row[dst] = _merge_score(row.get(dst, 0.0), 1.0, common_op)

        attr_stats.append({
            "attr_ntype": str(attr_ntype),
            "num_target_attr_pairs": int(pairs.shape[0]),
            "num_attr_groups": int(len(attr_to_targets)),
        })

    for node_id in range(num_target_nodes):
        # 保留自连接，避免一个节点没有共享属性时完全丢掉自己的 ASHIN-C 表示。
        rows[node_id][node_id] = max(rows[node_id].get(node_id, 0.0), 1.0)
    return rows, attr_stats


def propagate_by_commonality(x_base, rows, common_norm="row", topk=0):
    # 用共性矩阵对目标节点 ASHIN-C 做一轮加权平均。
    # 这里没有训练参数，只是按共享属性诱导出的结构相似性做平滑/修正。
    out = torch.zeros_like(x_base, dtype=torch.float32)
    for node_id, score_dict in enumerate(rows):
        items = list(score_dict.items())
        if topk and topk > 0:
            # topk 只截断其他节点，自身项一定保留。
            # 这样能减少高频属性节点带来的过度平滑。
            self_item = [(dst, score) for dst, score in items if dst == node_id]
            other_items = [(dst, score) for dst, score in items if dst != node_id]
            other_items = sorted(other_items, key=lambda item: (-item[1], item[0]))[:topk]
            items = self_item + other_items

        if not items:
            out[node_id] = x_base[node_id]
            continue

        ids = torch.tensor([dst for dst, _ in items], dtype=torch.long)
        scores = torch.tensor([score for _, score in items], dtype=torch.float32)
        if common_norm == "binary":
            # binary 不看共享次数，只看是否有关联。
            weights = torch.ones_like(scores) / float(scores.numel())
        elif common_norm == "row":
            # row 保留共享次数或 max 分数的相对大小。
            weights = scores / scores.sum().clamp(min=1e-12)
        else:
            raise ValueError("ASHIN-H supports ashin_common_norm in {row,binary}.")
        out[node_id] = (x_base[ids] * weights.unsqueeze(1)).sum(dim=0)
    return out


def build_ashin_h(
    g,
    target_ntype,
    norm="log1p_zscore",
    ashin_dim=128,
    seed=0,
    common_op="max",
    common_norm="row",
    common_topk=0,
):
    base_result = build_ashin_c(g, target_ntype, norm="none", ashin_dim=ashin_dim, seed=seed)
    rows, attr_stats = build_commonality_rows(g, target_ntype, common_op=common_op)
    x_raw = propagate_by_commonality(
        base_result["x_ashin"].float(),
        rows,
        common_norm=common_norm,
        topk=int(common_topk),
    )
    x_ashin = normalize_features(x_raw, norm)
    nonzero_links = sum(len(row) for row in rows)

    reduction = {
        "method": "target_commonality_corrected_ashin_c",
        "base_reduction": base_result["reduction"],
        "common_op": common_op,
        "common_norm": common_norm,
        "common_topk": int(common_topk),
        "num_commonality_links_with_self": int(nonzero_links),
        "attribute_stats": attr_stats,
    }

    return {
        "x_ashin": x_ashin,
        "x_b_raw": x_raw,
        "signature_to_id": base_result["signature_to_id"],
        "signature_freq": base_result["signature_freq"],
        "num_signatures": base_result["num_signatures"],
        "reduction": reduction,
    }
