"""
version_b.py

ASHIN-B。

先给目标节点构造 deterministic signature，再把 signature vocabulary 转成 one-hot incidence。
默认保持 0/1，不做归一化。C/D/F 等版本会复用这里的 raw incidence。
"""

from .signature import build_node_signatures, signatures_to_incidence
from .transform import normalize_features


def _signature_frequency_list(signature_to_id, signature_freq):
    # 将 Counter 形式的 signature 频次转成可保存到 cache/json 的列表。
    # 排序规则：先按频次从高到低，再按 signature id 从小到大。
    return [
        {"signature_id": int(signature_to_id[sig]), "count": int(count), "signature": repr(sig)}
        for sig, count in sorted(signature_freq.items(), key=lambda item: (-item[1], signature_to_id[item[0]]))
    ]


def build_ashin_b(g, target_ntype, norm="none"):
    # 1. 构造每个目标节点的 deterministic signature。
    signatures, signature_to_id, signature_freq = build_node_signatures(g, target_ntype)

    # 2. 将 signature 映射成严格 one-hot incidence 矩阵。
    # x_b_raw 是未归一化的 ASHIN-B，Version C 也会复用它作为降维输入。
    x_b_raw = signatures_to_incidence(signatures, signature_to_id)

    if x_b_raw.shape[1] > 10000:
        print(f"ASHIN-B signature dimension is large: K={x_b_raw.shape[1]}")

    # 3. Version B 默认 norm=none；如果显式指定 zscore 等，则在这里处理。
    x_ashin = normalize_features(x_b_raw, norm)

    return {
        "x_ashin": x_ashin,
        "x_b_raw": x_b_raw,
        "signature_to_id": signature_to_id,
        "signature_freq": _signature_frequency_list(signature_to_id, signature_freq),
        "num_signatures": int(len(signature_to_id)),
        "reduction": None,
    }
