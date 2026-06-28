from __future__ import annotations

import torch
from torch import nn


class EMatSparseEncoder(nn.Module):
    def __init__(
        self,
        num_basis: int = 3025,
        out_dim: int = 128,
        aggregation: str = "weighted_mean",
        value_transform: str = "log1p",
        dropout: float = 0.2,
        use_layernorm: bool = True,
        use_no_basis_embedding: bool = True,
    ) -> None:
        super().__init__()
        if aggregation not in {"sum", "mean", "weighted_sum", "weighted_mean", "sqrt_norm"}:
            raise ValueError(f"Unsupported aggregation: {aggregation}")
        if value_transform not in {"raw", "binary", "log1p"}:
            raise ValueError(f"Unsupported value_transform: {value_transform}")
        self.num_basis = num_basis
        self.out_dim = out_dim
        self.aggregation = aggregation
        self.value_transform = value_transform
        self.use_no_basis_embedding = use_no_basis_embedding
        extra = 1 if use_no_basis_embedding else 0
        self.embedding = nn.Embedding(num_basis + extra, out_dim)
        self.dropout = nn.Dropout(dropout)
        self.layernorm = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()

    def forward(self, indices: torch.Tensor, values: torch.Tensor, indptr: torch.Tensor) -> torch.Tensor:
        if indptr.ndim != 1:
            raise ValueError("indptr must be a 1D tensor")
        num_nodes = indptr.numel() - 1
        device = self.embedding.weight.device
        indices = indices.to(device=device, dtype=torch.long)
        values = values.to(device=device, dtype=torch.float32)
        indptr = indptr.to(device=device, dtype=torch.long)
        output = self.embedding.weight.new_zeros((num_nodes, self.out_dim))
        if indices.numel() == 0:
            return self._fill_no_basis(output, indptr)

        transformed = self._transform_values(values)
        basis = self.embedding(indices)
        if self.aggregation in {"sum", "mean"}:
            messages = basis
        else:
            messages = basis * transformed.unsqueeze(-1)
        row_counts = indptr[1:] - indptr[:-1]
        node_ids = torch.repeat_interleave(torch.arange(num_nodes, device=device), row_counts)
        output.index_add_(0, node_ids, messages)

        if self.aggregation == "mean":
            denom = row_counts.clamp_min(1).to(dtype=output.dtype).unsqueeze(-1)
            output = output / denom
        elif self.aggregation == "weighted_mean":
            denom = torch.zeros(num_nodes, device=device, dtype=output.dtype)
            denom.index_add_(0, node_ids, transformed.abs())
            output = output / denom.clamp_min(1e-12).unsqueeze(-1)
        elif self.aggregation == "sqrt_norm":
            denom = row_counts.clamp_min(1).to(dtype=output.dtype).sqrt().unsqueeze(-1)
            output = output / denom

        output = self._fill_no_basis(output, indptr)
        return self.layernorm(self.dropout(output))

    def _transform_values(self, values: torch.Tensor) -> torch.Tensor:
        if self.value_transform == "binary":
            return torch.ones_like(values)
        if self.value_transform == "log1p":
            return torch.log1p(values.clamp_min(0))
        return values

    def _fill_no_basis(self, output: torch.Tensor, indptr: torch.Tensor) -> torch.Tensor:
        if not self.use_no_basis_embedding:
            return output
        empty = (indptr[1:] - indptr[:-1]) == 0
        if empty.any():
            output = output.clone()
            output[empty] = self.embedding.weight[self.num_basis]
        return output
