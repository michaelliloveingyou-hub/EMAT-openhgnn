"""
transform.py

ASHIN 特征变换工具。

normalize_features 只做几种简单归一化；reduce_incidence 把 B 的高维 one-hot 压成 C 的 dense 特征。
降维只看 incidence matrix，不看 label/split。sklearn 可用时用 TruncatedSVD，不可用再退到 torch.pca_lowrank。
"""

import torch


def normalize_features(x, norm):
    # 对 ASHIN 特征做可选归一化。
    # B 默认不归一化，保留 0/1 incidence；其他版本再按参数处理。
    if norm in (None, "none"):
        return x.float()
    out = x.float()
    if norm in ("log1p", "log1p_zscore"):
        out = torch.log1p(torch.clamp(out, min=0.0))
    if norm in ("zscore", "log1p_zscore"):
        mean = out.mean(dim=0, keepdim=True)
        std = out.std(dim=0, keepdim=True)
        std = torch.where(std < 1e-12, torch.ones_like(std), std)
        out = (out - mean) / std
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.float()


def reduce_incidence(x_b, out_dim, seed=0):
    # ASHIN-C 的唯一入口：输入必须是 ASHIN-B incidence 矩阵。
    out_dim = int(out_dim)
    input_dim = int(x_b.shape[1])
    if out_dim <= 0:
        raise ValueError("ashin_dim must be positive for ASHIN-C.")
    if input_dim == 0:
        return torch.zeros((x_b.shape[0], out_dim), dtype=torch.float32), {
            "method": "empty",
            "input_dim": 0,
            "output_dim": out_dim,
            "actual_rank": 0,
            "random_seed": int(seed),
            "explained_variance_ratio_sum": 0.0,
        }

    rank = min(input_dim, out_dim, max(1, x_b.shape[0] - 1))
    x_cpu = x_b.float().cpu()
    metadata = {
        "input_dim": input_dim,
        "output_dim": out_dim,
        "actual_rank": int(rank),
        "random_seed": int(seed),
    }
    try:
        # TruncatedSVD 适合稀疏/高维的 incidence 特征。
        from sklearn.decomposition import TruncatedSVD

        svd = TruncatedSVD(n_components=rank, random_state=int(seed))
        reduced_np = svd.fit_transform(x_cpu.numpy())
        reduced = torch.from_numpy(reduced_np).float()
        metadata["method"] = "sklearn.TruncatedSVD"
        metadata["explained_variance_ratio_sum"] = float(svd.explained_variance_ratio_.sum())
    except Exception as exc:
        # 如果 sklearn 不可用，使用 PyTorch 的 PCA 作为无监督降维后备方案。
        torch.manual_seed(int(seed))
        q = min(rank + 2, min(x_cpu.shape))
        u, s, v = torch.pca_lowrank(x_cpu, q=q, center=True)
        reduced = torch.matmul(x_cpu, v[:, :rank])
        total_var = torch.var(x_cpu, dim=0).sum()
        explained = (s[:rank] ** 2).sum() / (total_var * max(x_cpu.shape[0] - 1, 1) + 1e-12)
        metadata["method"] = "torch.pca_lowrank"
        metadata["fallback_reason"] = str(exc)
        metadata["explained_variance_ratio_sum"] = float(explained)

    if reduced.shape[1] < out_dim:
        # 当 signature 数量小于请求维度时，补零保持下游输入维度一致。
        pad = torch.zeros((reduced.shape[0], out_dim - reduced.shape[1]), dtype=reduced.dtype)
        reduced = torch.cat([reduced, pad], dim=1)
    return reduced[:, :out_dim].float(), metadata
