from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


DATASET_CONFIG = {
    "ACM": {
        "output_name": "lisan-acm",
        "node_type_names": {
            "0": "paper",
            "1": "author",
            "2": "subject",
            "3": "term",
        },
    },
    "DBLP": {
        "output_name": "lisan-dblp",
        "node_type_names": {
            "0": "author",
            "1": "paper",
            "2": "term",
            "3": "venue",
        },
    },
}

OUTPUT_NAME_TO_DATASET = {config["output_name"]: key for key, config in DATASET_CONFIG.items()}


def resolve_dataset_key(dataset: str) -> str:
    if dataset in DATASET_CONFIG:
        return dataset
    try:
        return OUTPUT_NAME_TO_DATASET[dataset]
    except KeyError as exc:
        valid = sorted(DATASET_CONFIG) + sorted(OUTPUT_NAME_TO_DATASET)
        raise ValueError(f"Unsupported dataset {dataset!r}. Valid values: {valid}") from exc


def output_name(dataset_key: str) -> str:
    return DATASET_CONFIG[resolve_dataset_key(dataset_key)]["output_name"]


def processed_dataset_dir(output_dir: Path, dataset: str) -> Path:
    return output_dir / output_name(resolve_dataset_key(dataset))


def read_nodes(dataset_dir: Path, dataset_key: str) -> dict[str, list[tuple[str, list[float]]]]:
    config = DATASET_CONFIG[resolve_dataset_key(dataset_key)]
    nodes_by_type: dict[str, list[tuple[str, list[float]]]] = defaultdict(list)
    node_path = dataset_dir / "node.dat"
    with node_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 3:
                raise ValueError(f"{node_path}:{line_number} has fewer than 3 columns")
            global_id, raw_type = parts[0], parts[2]
            node_type = config["node_type_names"].get(raw_type, raw_type)
            feature = _parse_feature(parts[3]) if len(parts) >= 4 else []
            nodes_by_type[node_type].append((global_id, feature))
    return dict(nodes_by_type)


def build_raw_feature_tensors(nodes_by_type: dict[str, list[tuple[str, list[float]]]]) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    dims = {}
    for node_type, rows in nodes_by_type.items():
        dims[node_type] = max((len(row[-1]) for row in rows), default=0)
    fallback_dim = max(dims.values(), default=1)
    features_by_type = {}
    feature_dims = {}
    for node_type, rows in nodes_by_type.items():
        dim = dims[node_type] or fallback_dim
        values = []
        for row in rows:
            feature = row[-1]
            if not feature:
                values.append([0.0] * dim)
            elif len(feature) == dim:
                values.append(feature)
            else:
                raise ValueError(f"{node_type} raw feature dimension mismatch: got {len(feature)}, expected {dim}")
        features_by_type[node_type] = torch.tensor(values, dtype=torch.float32)
        feature_dims[node_type] = dim
    return features_by_type, feature_dims


def load_emat_features(
    emat_root: Path,
    dataset: str,
    nodes_by_type: dict[str, Iterable],
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    tensors = load_emat_tensors(emat_root, dataset, nodes_by_type)
    features = {node_type: tensor.t().contiguous().to(dtype=torch.float32) for node_type, tensor in tensors.items()}
    dims = {node_type: int(tensor.shape[0]) for node_type, tensor in tensors.items()}
    return features, dims


def load_emat_tensors(emat_root: Path, dataset: str, nodes_by_type: dict[str, Iterable]) -> dict[str, torch.Tensor]:
    dataset_key = resolve_dataset_key(dataset)
    config = DATASET_CONFIG[dataset_key]
    emat_dir = emat_root / dataset_key
    if not emat_dir.exists():
        raise FileNotFoundError(f"Missing EMat feature directory: {emat_dir}")
    raw_type_by_name = {name: raw_type for raw_type, name in config["node_type_names"].items()}
    tensors = {}
    for node_type, rows in nodes_by_type.items():
        raw_type = raw_type_by_name[node_type]
        emat_path = emat_dir / f"EMat_{raw_type}.pt"
        if not emat_path.exists():
            raise FileNotFoundError(f"Missing EMat tensor for {dataset_key} {node_type}: {emat_path}")
        tensor = torch.load(emat_path, map_location="cpu")
        if not torch.is_tensor(tensor) or tensor.ndim != 2:
            raise ValueError(f"{emat_path} must be a 2D torch tensor")
        row_count = len(rows)
        if tensor.shape[1] != row_count:
            raise ValueError(
                f"{emat_path} column count {tensor.shape[1]} does not match "
                f"{dataset_key} {node_type} node count {row_count}"
            )
        tensors[node_type] = tensor.to(dtype=torch.float32)
    return tensors


def emat_tensor_to_csr(tensor: torch.Tensor) -> sparse.csr_matrix:
    coo = tensor.to_sparse_coo().coalesce()
    if coo._nnz() == 0:
        return sparse.csr_matrix((tensor.shape[1], tensor.shape[0]), dtype=np.float32)
    idx = coo.indices()
    values = coo.values().cpu().numpy().astype(np.float32, copy=False)
    rows = idx[1].cpu().numpy()
    cols = idx[0].cpu().numpy()
    return sparse.coo_matrix((values, (rows, cols)), shape=(tensor.shape[1], tensor.shape[0])).tocsr()


def normalize_dense(array: np.ndarray, mode: str) -> np.ndarray:
    array = array.astype(np.float32, copy=False)
    if mode == "none":
        return array
    if mode == "zscore":
        mean = array.mean(axis=0, keepdims=True)
        std = array.std(axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        return ((array - mean) / std).astype(np.float32, copy=False)
    if mode == "l2":
        norm = np.linalg.norm(array, axis=1, keepdims=True)
        norm[norm < 1e-12] = 1.0
        return (array / norm).astype(np.float32, copy=False)
    raise ValueError(f"Unsupported normalize mode: {mode}")


def apply_tfidf(matrix: sparse.csr_matrix, tf_mode: str = "log1p") -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    matrix = matrix.copy().astype(np.float32)
    if tf_mode == "binary":
        matrix.data[:] = 1.0
    elif tf_mode == "log1p":
        matrix.data = np.log1p(matrix.data).astype(np.float32, copy=False)
    elif tf_mode != "raw":
        raise ValueError(f"Unsupported tf_mode: {tf_mode}")
    df = np.asarray((matrix > 0).sum(axis=0)).ravel()
    n_nodes = matrix.shape[0]
    idf = (np.log((n_nodes + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
    return matrix.multiply(idf).tocsr().astype(np.float32), idf, df


def fit_svd_features(
    matrix: sparse.csr_matrix,
    requested_dim: int,
    normalize: str,
    seed: int,
) -> tuple[np.ndarray, dict]:
    nonzero_columns = int(np.asarray((matrix > 0).sum(axis=0)).ravel().astype(bool).sum())
    max_rank = min(matrix.shape[0] - 1, nonzero_columns - 1)
    actual_dim = max(0, min(requested_dim, max_rank))
    metadata = {
        "requested_dim": requested_dim,
        "actual_dim": actual_dim,
        "num_nodes": int(matrix.shape[0]),
        "original_dim": int(matrix.shape[1]),
        "num_nonzero_columns": nonzero_columns,
    }
    if actual_dim <= 0:
        return np.zeros((matrix.shape[0], requested_dim), dtype=np.float32), metadata
    svd = TruncatedSVD(n_components=actual_dim, random_state=seed)
    reduced = svd.fit_transform(matrix).astype(np.float32, copy=False)
    reduced = normalize_dense(reduced, normalize)
    if actual_dim < requested_dim:
        reduced = np.pad(reduced, ((0, 0), (0, requested_dim - actual_dim)), mode="constant")
    metadata["explained_variance_ratio_sum"] = float(svd.explained_variance_ratio_.sum())
    metadata["singular_values"] = svd.singular_values_.astype(float).tolist()
    return reduced.astype(np.float32, copy=False), metadata


def save_feature_payload(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_feature_payload(processed_root: Path, dataset: str, feature_mode: str) -> dict:
    dataset_dir = processed_dataset_dir(processed_root, dataset)
    filename = feature_payload_filename(feature_mode)
    path = dataset_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing processed EMat feature file: {path}. "
            "Run scripts/preprocess_emat_svd.py, scripts/preprocess_emat_tfidf.py, "
            "or scripts/preprocess_emat_tfidf_svd.py first."
        )
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or "features" not in payload:
        raise ValueError(f"{path} must contain a dict with a 'features' field")
    return payload


def feature_payload_filename(feature_mode: str) -> str:
    if feature_mode.startswith("raw_"):
        feature_mode = feature_mode[4:]
    if feature_mode in {"emat_tfidf", "emat_tfidf_3025"}:
        return "emat_tfidf.pt"
    if feature_mode == "emat":
        return "emat_3025.pt"
    return f"{feature_mode}.pt"


def sparse_payload_filename(raw_prefix: bool = False) -> str:
    return "raw_emat_sparse_encoder.pt" if raw_prefix else "emat_sparse_encoder.pt"


def diagnostics_for_matrix(matrix: sparse.csr_matrix) -> dict:
    nnz_per_node = np.diff(matrix.indptr)
    df = np.asarray((matrix > 0).sum(axis=0)).ravel()
    dense_values = matrix.data
    total = matrix.shape[0] * matrix.shape[1]
    unique_values = np.unique(dense_values) if dense_values.size else np.array([], dtype=np.float32)
    is_binary = bool(unique_values.size == 0 or np.all(np.isin(unique_values, [0.0, 1.0])))
    has_count_values = bool(dense_values.size > 0 and np.nanmax(dense_values) > 1.0)
    mean_square = np.asarray(matrix.power(2).mean(axis=0)).ravel()
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    variance = mean_square - np.square(mean)
    return {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "nnz": int(matrix.nnz),
        "nonzero_ratio": float(matrix.nnz / total) if total else 0.0,
        "zero_ratio": float(1.0 - matrix.nnz / total) if total else 1.0,
        "nnz_per_node": _stats(nnz_per_node),
        "zero_columns": int((df == 0).sum()),
        "df_lt_1": int((df < 1).sum()),
        "df_lt_5": int((df < 5).sum()),
        "df_lt_10": int((df < 10).sum()),
        "df_lt_20": int((df < 20).sum()),
        "variance": _stats(variance),
        "is_binary": is_binary,
        "has_count_values": has_count_values,
        "min": float(dense_values.min()) if dense_values.size else 0.0,
        "max": float(dense_values.max()) if dense_values.size else 0.0,
        "mean": float(matrix.mean()) if total else 0.0,
        "has_nan": bool(np.isnan(dense_values).any()) if dense_values.size else False,
        "has_inf": bool(np.isinf(dense_values).any()) if dense_values.size else False,
    }


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_feature(raw: str) -> list[float]:
    raw = raw.strip()
    if not raw:
        return []
    return [float(value) for value in raw.split(",") if value != ""]


def _stats(values: np.ndarray) -> dict:
    values = np.asarray(values)
    if values.size == 0:
        return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }
