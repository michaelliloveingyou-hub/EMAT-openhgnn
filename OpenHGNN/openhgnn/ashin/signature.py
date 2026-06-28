"""
signature.py

ASHIN-B 的 signature 构造。

一个目标节点就是一个 unit core。代码会按固定顺序扫 canonical_etypes，
把每种关系下的邻居数、core degree、邻居 degree 统计拼成一个 tuple。
不同 tuple 是不同 signature，最后转成 node-signature one-hot incidence。
"""

from collections import Counter

import torch


def sorted_canonical_etypes(g):
    # DGL 返回的 canonical_etypes 顺序不一定适合作为实验可复现依据。
    # 这里统一按字符串排序，保证 signature 构造顺序稳定。
    return sorted(g.canonical_etypes, key=lambda e: (str(e[0]), str(e[1]), str(e[2])))


def sorted_ntypes(g):
    return sorted(g.ntypes, key=str)


def _degree_tensor(g, ntype, etype, direction):
    # 根据关系方向取出对应节点在该关系下的入度或出度。
    if direction == "out":
        return g.out_degrees(etype=etype).cpu().long()
    return g.in_degrees(etype=etype).cpu().long()


def _empty_neighbor_lists(num_nodes):
    return [[] for _ in range(num_nodes)]


def _edge_neighbor_lists(g, etype, target_ntype, mode, num_target_nodes):
    # 将某一种边类型下的邻居整理成“每个目标节点一个邻居列表”。
    # 排序后再追加，避免边遍历顺序影响 signature。
    src, dst = g.edges(etype=etype)
    src = src.cpu().long()
    dst = dst.cpu().long()
    neighbors = _empty_neighbor_lists(num_target_nodes)
    if mode == "out":
        order = torch.argsort(src * (dst.numel() + 1) + dst)
        for s, d in zip(src[order].tolist(), dst[order].tolist()):
            neighbors[s].append(d)
    else:
        order = torch.argsort(dst * (src.numel() + 1) + src)
        for s, d in zip(src[order].tolist(), dst[order].tolist()):
            neighbors[d].append(s)
    return neighbors


def build_node_signatures(g, target_ntype):
    """Build deterministic ASHIN-B signatures for every target node.
        遍历目标节点类型的每个节点
        2. 把每个目标节点作为 unit core
        3. 遍历排序后的 canonical_etypes
        4. 如果目标节点类型在边的 src 端，就收集 successors
        5. 如果目标节点类型在边的 dst 端，就收集 predecessors
        6. 对每种关系统计：
        - core degree
        - neighbor count
        - neighbor degree sum
        - neighbor degree mean
        - neighbor degree max
        7. 把所有关系的统计拼成 signature tuple
        8. 所有不同 tuple 组成 signature vocabulary
    """
    if target_ntype not in g.ntypes:
        raise ValueError(f"target_ntype {target_ntype!r} is not in graph ntypes {g.ntypes}.")

    num_target_nodes = g.num_nodes(target_ntype)
    etypes = sorted_canonical_etypes(g)
    per_etype = []
    for etype in etypes:
        # 对每一种 canonical edge type，判断目标节点位于源端还是目标端。
        # 位于源端时使用 successors，位于目标端时使用 predecessors。
        src_ntype, rel, dst_ntype = etype
        if src_ntype == target_ntype:
            core_degree = _degree_tensor(g, target_ntype, etype, "out")
            neighbor_degree = _degree_tensor(g, dst_ntype, etype, "in")
            neighbors = _edge_neighbor_lists(g, etype, target_ntype, "out", num_target_nodes)
            direction = "out"
        elif dst_ntype == target_ntype:
            core_degree = _degree_tensor(g, target_ntype, etype, "in")
            neighbor_degree = _degree_tensor(g, src_ntype, etype, "out")
            neighbors = _edge_neighbor_lists(g, etype, target_ntype, "in", num_target_nodes)
            direction = "in"
        else:
            core_degree = torch.zeros(num_target_nodes, dtype=torch.long)
            neighbor_degree = torch.zeros(0, dtype=torch.long)
            neighbors = _empty_neighbor_lists(num_target_nodes)
            direction = "none"
        per_etype.append((etype, direction, core_degree, neighbor_degree, neighbors))

    signatures = []
    for nid in range(num_target_nodes):
        # 每个目标节点都会得到一个完整 signature。
        # 即使某种关系下没有邻居，也会写入 0 统计值，保证 tuple 长度一致。
        parts = []
        for etype, direction, core_degree, neighbor_degree, neighbors in per_etype:
            neigh = neighbors[nid]
            if neigh:
                deg_values = neighbor_degree[torch.tensor(neigh, dtype=torch.long)]
                deg_sum = int(deg_values.sum().item())
                deg_max = int(deg_values.max().item())
                deg_mean_x1000 = int(round(float(deg_sum) * 1000.0 / len(neigh)))
            else:
                deg_sum = 0
                deg_max = 0
                deg_mean_x1000 = 0
            parts.append((
                str(etype[0]),
                str(etype[1]),
                str(etype[2]),
                direction,
                int(core_degree[nid].item()),
                int(len(neigh)),
                deg_sum,
                deg_mean_x1000,
                deg_max,
            ))
        signatures.append(tuple(parts))

    signature_freq = Counter(signatures)
    # signature id 由排序后的 signature 决定，不依赖字典插入顺序。
    signature_to_id = {sig: idx for idx, sig in enumerate(sorted(signature_freq.keys()))}
    return signatures, signature_to_id, signature_freq


def signatures_to_incidence(signatures, signature_to_id):
    # 将 signature 序列转成 one-hot incidence 矩阵。
    x = torch.zeros((len(signatures), len(signature_to_id)), dtype=torch.float32)
    for row, sig in enumerate(signatures):
        x[row, signature_to_id[sig]] = 1.0
    return x
