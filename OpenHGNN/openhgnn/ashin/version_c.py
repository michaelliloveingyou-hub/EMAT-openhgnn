"""
version_c.py

ASHIN-C。

C 就是 B 的压缩版：先拿 B 的 raw incidence，再做无监督降维，最后按 norm 参数处理。
不要在这里重新发明一套 degree feature。降维也不要碰 label/mask/test 指标。
"""

from .transform import normalize_features, reduce_incidence
from .version_b import build_ashin_b


def build_ashin_c(g, target_ntype, norm="log1p_zscore", ashin_dim=128, seed=0):
    # 1. C 必须先得到 B 的严格 incidence。
    # 这里强制 norm="none"，避免把归一化后的 B 输入 SVD/PCA。
    b_result = build_ashin_b(g, target_ntype, norm="none")
    x_b_raw = b_result["x_b_raw"]

    # 2. 对 B incidence 做无监督降维，得到低维 dense 结构特征。
    x_c, reduction_metadata = reduce_incidence(x_b_raw, ashin_dim, seed)

    # 3. 对降维后的 C 特征做可选归一化。
    x_ashin = normalize_features(x_c, norm)

    return {
        "x_ashin": x_ashin,
        "x_b_raw": x_b_raw,
        "signature_to_id": b_result["signature_to_id"],
        "signature_freq": b_result["signature_freq"],
        "num_signatures": b_result["num_signatures"],
        "reduction": reduction_metadata,
    }
