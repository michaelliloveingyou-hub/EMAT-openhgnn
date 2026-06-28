# 项目文件功能说明：
# 本文件用于把 Lisan_project 当前的 ACM、DBLP 文本异构图数据集转换为 OpenHGNN 可读取的 DGL graph.bin。
# 它读取 Datasets/<ACM|DBLP>/node.dat、link.dat、label.dat、label.dat.test、label.dat.test_full、info.dat，
# 构造按节点类型局部编号的 dgl.heterograph，写入节点特征 h、节点分类标签 label、train/val/test/test_full mask，
# 并输出到 OpenHGNN 源码目录 openhgnn/dataset/lisan_hgb/，供新注册的数据集 lisan-acm、lisan-dblp 使用。

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_lisan.emat_preprocessing import (
    build_raw_feature_tensors,
    feature_payload_filename,
    load_emat_features as load_direct_emat_features,
    load_feature_payload,
)


DATASET_CONFIG = {
    "ACM": {
        "output_name": "lisan-acm",
        "category": "paper",
        "num_classes": 3,
        "node_type_names": {
            "0": "paper",
            "1": "author",
            "2": "subject",
            "3": "term",
        },
    },
    "DBLP": {
        "output_name": "lisan-dblp",
        "category": "author",
        "num_classes": 4,
        "node_type_names": {
            "0": "author",
            "1": "paper",
            "2": "term",
            "3": "venue",
        },
    },
}

FEATURE_MODE_ALIASES = {
    "A": "raw",
    "B": "emat",
    "C": "emat_svd_128",
    "D": "emat_tfidf_3025",
    "E": "raw_emat_sparse_encoder",
    "raw": "raw",
    "emat": "emat",
    "emat_3025": "emat",
    "raw_emat_3025": "raw_emat_3025",
    "emat_svd_64": "emat_svd_64",
    "emat_svd_128": "emat_svd_128",
    "emat_svd_256": "emat_svd_256",
    "raw_emat_svd_64": "raw_emat_svd_64",
    "raw_emat_svd_128": "raw_emat_svd_128",
    "raw_emat_svd_256": "raw_emat_svd_256",
    "emat_tfidf": "emat_tfidf_3025",
    "emat_tfidf_3025": "emat_tfidf_3025",
    "raw_emat_tfidf_3025": "raw_emat_tfidf_3025",
    "emat_tfidf_svd_64": "emat_tfidf_svd_64",
    "emat_tfidf_svd_128": "emat_tfidf_svd_128",
    "emat_tfidf_svd_256": "emat_tfidf_svd_256",
    "raw_emat_tfidf_svd_64": "raw_emat_tfidf_svd_64",
    "raw_emat_tfidf_svd_128": "raw_emat_tfidf_svd_128",
    "raw_emat_tfidf_svd_256": "raw_emat_tfidf_svd_256",
    "emat_sparse_encoder": "emat_sparse_encoder",
    "raw_emat_sparse_encoder": "raw_emat_sparse_encoder",
}
FEATURE_MODES = tuple(dict.fromkeys(FEATURE_MODE_ALIASES.values()))
SPARSE_ENCODER_MODES = {"emat_sparse_encoder", "raw_emat_sparse_encoder"}
MATERIALIZABLE_FEATURE_MODES = tuple(
    mode for mode in FEATURE_MODES if mode not in SPARSE_ENCODER_MODES and mode != "raw"
)


def normalize_feature_mode(feature_mode: str) -> str:
    try:
        return FEATURE_MODE_ALIASES[feature_mode]
    except KeyError as exc:
        valid = ", ".join(sorted(FEATURE_MODE_ALIASES))
        raise ValueError(f"Unsupported feature_mode: {feature_mode}. Valid values: {valid}") from exc


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_feature(raw: str) -> List[float]:
    raw = raw.strip()
    if not raw:
        return []
    return [float(value) for value in raw.split(",") if value != ""]


def read_nodes(dataset_dir: Path, config: dict) -> Tuple[dict, dict, dict, dict]:
    nodes_by_type: Dict[str, List[Tuple[str, str, List[float]]]] = defaultdict(list)
    global_to_type: Dict[str, str] = {}
    global_to_local: Dict[str, int] = {}
    feature_dims: Dict[str, int] = {}

    node_path = dataset_dir / "node.dat"
    with node_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 3:
                raise ValueError(f"{node_path}:{line_number} has fewer than 3 columns")
            global_id, name, raw_type = parts[0], parts[1], parts[2]
            node_type = config["node_type_names"].get(raw_type, raw_type)
            feature = parse_feature(parts[3]) if len(parts) >= 4 else []
            local_id = len(nodes_by_type[node_type])
            nodes_by_type[node_type].append((global_id, name, feature))
            global_to_type[global_id] = node_type
            global_to_local[global_id] = local_id
            if feature:
                feature_dims[node_type] = len(feature)

    fallback_dim = max(feature_dims.values()) if feature_dims else 1
    for node_type in nodes_by_type:
        feature_dims.setdefault(node_type, fallback_dim)
    return nodes_by_type, global_to_type, global_to_local, feature_dims


def edge_type_specs(info: dict, config: dict) -> Dict[str, Tuple[str, str, str]]:
    specs = {}
    link_types = info.get("link.dat", {}).get("link type", {})
    for raw_edge_type, spec in link_types.items():
        src_type = config["node_type_names"].get(str(spec["start"]), str(spec["start"]))
        dst_type = config["node_type_names"].get(str(spec["end"]), str(spec["end"]))
        edge_name = str(spec.get("meaning", raw_edge_type))
        specs[str(raw_edge_type)] = (src_type, edge_name, dst_type)
    return specs


def read_edges(dataset_dir: Path, specs: dict, global_to_local: dict) -> dict:
    edges = defaultdict(lambda: ([], []))
    link_path = dataset_dir / "link.dat"
    with link_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 3:
                raise ValueError(f"{link_path}:{line_number} has fewer than 3 columns")
            src_global, dst_global, raw_edge_type = parts[0], parts[1], parts[2]
            canonical_etype = specs[raw_edge_type]
            src_local = global_to_local[src_global]
            dst_local = global_to_local[dst_global]
            edges[canonical_etype][0].append(src_local)
            edges[canonical_etype][1].append(dst_local)
    return edges


def read_labels(path: Path, global_to_local: dict, category: str, global_to_type: dict) -> Dict[int, int]:
    labels = {}
    if not path.exists():
        return labels
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            parts = line.rstrip("\n\r").split("\t")
            if len(parts) < 4:
                raise ValueError(f"{path}:{line_number} has fewer than 4 columns")
            global_id, class_id = parts[0], int(parts[3])
            if global_to_type[global_id] != category:
                raise ValueError(f"{path}:{line_number} labels {global_to_type[global_id]}, expected {category}")
            labels[global_to_local[global_id]] = class_id
    return labels


def stratified_train_val_split(labels: Dict[int, int], val_ratio: float, seed: int) -> Tuple[set, set]:
    by_class = defaultdict(list)
    for local_id, class_id in labels.items():
        by_class[class_id].append(local_id)

    rng = random.Random(seed)
    train_ids, val_ids = set(), set()
    for _, ids in sorted(by_class.items()):
        rng.shuffle(ids)
        val_count = max(1, int(round(len(ids) * val_ratio))) if len(ids) > 1 else 0
        val_ids.update(ids[:val_count])
        train_ids.update(ids[val_count:])
    return train_ids, val_ids


def make_bool_mask(torch_module, size: int, ids: Iterable[int]):
    mask = torch_module.zeros(size, dtype=torch_module.bool)
    id_list = list(ids)
    if id_list:
        mask[torch_module.tensor(id_list, dtype=torch_module.long)] = True
    return mask


def load_emat_features(emat_root: Path, dataset_key: str, config: dict, nodes_by_type: dict) -> Tuple[dict, dict]:
    import torch

    emat_dir = emat_root / dataset_key
    if not emat_dir.exists():
        raise FileNotFoundError(f"Missing EMat feature directory: {emat_dir}")

    raw_type_by_name = {name: raw_type for raw_type, name in config["node_type_names"].items()}
    features_by_type = {}
    feature_dims = {}
    for node_type, items in nodes_by_type.items():
        raw_type = raw_type_by_name[node_type]
        emat_path = emat_dir / f"EMat_{raw_type}.pt"
        if not emat_path.exists():
            raise FileNotFoundError(f"Missing EMat tensor for {dataset_key} {node_type}: {emat_path}")
        tensor = torch.load(emat_path, map_location="cpu")
        if not torch.is_tensor(tensor) or tensor.ndim != 2:
            raise ValueError(f"{emat_path} must be a 2D torch tensor")
        if tensor.shape[1] != len(items):
            raise ValueError(
                f"{emat_path} column count {tensor.shape[1]} does not match "
                f"{dataset_key} {node_type} node count {len(items)}"
            )
        features_by_type[node_type] = tensor.t().contiguous().to(dtype=torch.float32)
        feature_dims[node_type] = int(tensor.shape[0])
    return features_by_type, feature_dims


def _emat_tensor_by_type(emat_root: Path, dataset_key: str, config: dict, nodes_by_type: dict) -> dict:
    import torch

    emat_dir = emat_root / dataset_key
    raw_type_by_name = {name: raw_type for raw_type, name in config["node_type_names"].items()}
    tensors = {}
    for node_type, items in nodes_by_type.items():
        raw_type = raw_type_by_name[node_type]
        emat_path = emat_dir / f"EMat_{raw_type}.pt"
        if not emat_path.exists():
            raise FileNotFoundError(f"Missing EMat tensor for {dataset_key} {node_type}: {emat_path}")
        tensor = torch.load(emat_path, map_location="cpu")
        if not torch.is_tensor(tensor) or tensor.ndim != 2:
            raise ValueError(f"{emat_path} must be a 2D torch tensor")
        if tensor.shape[1] != len(items):
            raise ValueError(
                f"{emat_path} column count {tensor.shape[1]} does not match "
                f"{dataset_key} {node_type} node count {len(items)}"
            )
        tensors[node_type] = tensor
    return tensors


def _tensor_to_csr_nodes_by_basis(tensor):
    import numpy as np
    from scipy import sparse

    coo = tensor.to_sparse_coo().coalesce()
    if coo._nnz() == 0:
        return sparse.csr_matrix((tensor.shape[1], tensor.shape[0]), dtype=np.float32)
    idx = coo.indices()
    values = coo.values().cpu().numpy().astype(np.float32, copy=False)
    # EMat is basis x nodes; scipy matrix must be nodes x basis.
    rows = idx[1].cpu().numpy()
    cols = idx[0].cpu().numpy()
    return sparse.coo_matrix((values, (rows, cols)), shape=(tensor.shape[1], tensor.shape[0])).tocsr()


def _zscore_numpy(array):
    import numpy as np

    mean = array.mean(axis=0, keepdims=True)
    std = array.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    return ((array - mean) / std).astype(np.float32, copy=False)


def build_emat_svd_features(emat_root: Path, dataset_key: str, config: dict, nodes_by_type: dict, dim: int = 128):
    import numpy as np
    import torch
    from sklearn.decomposition import TruncatedSVD

    tensors = _emat_tensor_by_type(emat_root, dataset_key, config, nodes_by_type)
    features_by_type = {}
    metadata = {"method": "TruncatedSVD", "requested_dim": dim, "normalize": "zscore", "actual_dims": {}}
    for node_type, tensor in tensors.items():
        matrix = _tensor_to_csr_nodes_by_basis(tensor)
        max_rank = max(1, min(matrix.shape) - 1)
        actual_dim = min(dim, max_rank)
        if actual_dim <= 0:
            reduced = np.zeros((matrix.shape[0], dim), dtype=np.float32)
        else:
            svd = TruncatedSVD(n_components=actual_dim, random_state=2026)
            reduced = svd.fit_transform(matrix).astype(np.float32, copy=False)
            reduced = _zscore_numpy(reduced)
            if actual_dim < dim:
                reduced = np.pad(reduced, ((0, 0), (0, dim - actual_dim)), mode="constant")
            metadata.setdefault("explained_variance_ratio_sum", {})[node_type] = float(
                svd.explained_variance_ratio_.sum()
            )
        metadata["actual_dims"][node_type] = int(actual_dim)
        features_by_type[node_type] = torch.tensor(reduced, dtype=torch.float32)
    return features_by_type, {ntype: dim for ntype in features_by_type}, metadata


def build_emat_tfidf_features(emat_root: Path, dataset_key: str, config: dict, nodes_by_type: dict, tf_mode: str = "log1p"):
    import numpy as np
    import torch

    tensors = _emat_tensor_by_type(emat_root, dataset_key, config, nodes_by_type)
    features_by_type = {}
    feature_dims = {}
    metadata = {"method": "TFIDF", "tf_mode": tf_mode, "idf_smooth": True, "df_stats": {}}
    for node_type, tensor in tensors.items():
        matrix = _tensor_to_csr_nodes_by_basis(tensor)
        if tf_mode == "binary":
            matrix.data[:] = 1.0
        elif tf_mode == "log1p":
            matrix.data = np.log1p(matrix.data).astype(np.float32, copy=False)
        elif tf_mode != "raw":
            raise ValueError(f"Unsupported tf_mode: {tf_mode}")

        df = np.asarray((matrix > 0).sum(axis=0)).ravel()
        n_nodes = matrix.shape[0]
        idf = (np.log((n_nodes + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
        weighted = matrix.multiply(idf).astype(np.float32)
        dense = weighted.toarray().astype(np.float32, copy=False)
        features_by_type[node_type] = torch.tensor(dense, dtype=torch.float32)
        feature_dims[node_type] = int(dense.shape[1])
        metadata["df_stats"][node_type] = {
            "min": int(df.min()) if df.size else 0,
            "max": int(df.max()) if df.size else 0,
            "mean": float(df.mean()) if df.size else 0.0,
            "zero_df": int((df == 0).sum()) if df.size else 0,
        }
    return features_by_type, feature_dims, metadata


def build_emat_sparse_projection_features(
    emat_root: Path,
    dataset_key: str,
    config: dict,
    nodes_by_type: dict,
    dim: int = 128,
    value_transform: str = "log1p",
):
    import numpy as np
    import torch

    tensors = _emat_tensor_by_type(emat_root, dataset_key, config, nodes_by_type)
    features_by_type = {}
    metadata = {
        "method": "FixedSparseEmbeddingBagProjection",
        "dim": dim,
        "aggregation": "weighted_mean",
        "value_transform": value_transform,
        "seed": 2026,
        "note": "Static deterministic projection used to materialize a graph feature file.",
    }
    for offset, (node_type, tensor) in enumerate(tensors.items()):
        matrix = _tensor_to_csr_nodes_by_basis(tensor)
        if value_transform == "binary":
            matrix.data[:] = 1.0
        elif value_transform == "log1p":
            matrix.data = np.log1p(matrix.data).astype(np.float32, copy=False)
        elif value_transform != "raw":
            raise ValueError(f"Unsupported value_transform: {value_transform}")

        rng = np.random.default_rng(2026 + offset)
        basis = rng.normal(0.0, 1.0 / max(1, dim) ** 0.5, size=(matrix.shape[1], dim)).astype(np.float32)
        projected = matrix @ basis
        denom = np.asarray(np.abs(matrix).sum(axis=1)).reshape(-1, 1).astype(np.float32)
        denom[denom < 1e-12] = 1.0
        projected = (projected / denom).astype(np.float32, copy=False)
        projected = _zscore_numpy(projected)
        features_by_type[node_type] = torch.tensor(projected, dtype=torch.float32)
    return features_by_type, {ntype: dim for ntype in features_by_type}, metadata


def _concat_feature_dicts(left: dict, right: dict) -> dict:
    import torch

    return {node_type: torch.cat([left[node_type], right[node_type]], dim=1) for node_type in left}


def _payload_features(processed_feature_root: Path, dataset_key: str, feature_mode: str) -> Tuple[dict, dict, dict]:
    import torch

    payload = load_feature_payload(processed_feature_root, dataset_key, feature_mode)
    features = {node_type: tensor.to(dtype=torch.float32) for node_type, tensor in payload["features"].items()}
    dims = {node_type: int(tensor.shape[1]) for node_type, tensor in features.items()}
    metadata = {
        "method": payload.get("method"),
        "source_file": str(processed_feature_root / DATASET_CONFIG[dataset_key]["output_name"] / feature_payload_filename(feature_mode)),
        "payload_metadata": _manifest_safe(payload.get("metadata", {})),
    }
    return features, dims, metadata


def _manifest_safe(value):
    import numpy as np
    import torch

    if torch.is_tensor(value):
        return {"tensor_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.ndarray):
        return {"array_shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _manifest_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_manifest_safe(item) for item in value]
    return value


def build_feature_tensors(
    feature_mode: str,
    emat_root: Path,
    processed_feature_root: Path,
    dataset_key: str,
    config: dict,
    nodes_by_type: dict,
    raw_features_by_type: dict,
    raw_feature_dims: dict,
):
    if feature_mode == "raw":
        return raw_features_by_type, raw_feature_dims, {"method": "RawNodeDat"}
    if feature_mode == "emat":
        features, dims = load_direct_emat_features(emat_root, dataset_key, nodes_by_type)
        return features, dims, {"method": "DirectEMat", "dim": next(iter(dims.values())) if dims else 0}
    if feature_mode == "raw_emat_3025":
        emat_features, emat_dims = load_direct_emat_features(emat_root, dataset_key, nodes_by_type)
        features = _concat_feature_dicts(raw_features_by_type, emat_features)
        dims = {node_type: raw_feature_dims[node_type] + emat_dims[node_type] for node_type in raw_feature_dims}
        return features, dims, {"method": "RawNodeDat+DirectEMat", "dim": dims}
    if feature_mode in SPARSE_ENCODER_MODES:
        raise NotImplementedError(
            f"{feature_mode} is a learnable runtime encoder mode. "
            "It cannot be materialized as a static graph h feature. "
            "Use scripts/preprocess_emat_sparse_inputs.py to build its CSR inputs, then run a model wrapper that "
            "instantiates experiments_lisan.models.EMatSparseEncoder."
        )

    raw_prefix = feature_mode.startswith("raw_")
    base_mode = feature_mode[4:] if raw_prefix else feature_mode
    processed_features, processed_dims, metadata = _payload_features(processed_feature_root, dataset_key, base_mode)
    if raw_prefix:
        features = _concat_feature_dicts(raw_features_by_type, processed_features)
        dims = {
            node_type: raw_feature_dims[node_type] + processed_dims[node_type]
            for node_type in raw_feature_dims
        }
        metadata = {"method": f"RawNodeDat+{metadata.get('method')}", "processed_feature": metadata}
        return features, dims, metadata
    return processed_features, processed_dims, metadata


def output_stem(output_name: str, feature_mode: str) -> str:
    suffix = "" if feature_mode == "raw" else feature_mode.replace("_", "-")
    return output_name if not suffix else f"{output_name}-{suffix}"


def convert_one(
    dataset_root: Path,
    output_dir: Path,
    dataset_key: str,
    val_ratio: float,
    seed: int,
    feature_mode: str,
    emat_root: Path,
    processed_feature_root: Path,
) -> dict:
    import dgl
    import torch

    feature_mode = normalize_feature_mode(feature_mode)

    config = DATASET_CONFIG[dataset_key]
    dataset_dir = dataset_root / dataset_key
    info = load_json(dataset_dir / "info.dat")
    nodes_by_type, global_to_type, global_to_local, feature_dims = read_nodes(dataset_dir, config)
    raw_features_by_type, raw_feature_dims = build_raw_feature_tensors(nodes_by_type)
    specs = edge_type_specs(info, config)
    edges = read_edges(dataset_dir, specs, global_to_local)

    num_nodes_dict = {node_type: len(items) for node_type, items in nodes_by_type.items()}
    graph_data = {
        canonical_etype: (
            torch.tensor(src_ids, dtype=torch.int64),
            torch.tensor(dst_ids, dtype=torch.int64),
        )
        for canonical_etype, (src_ids, dst_ids) in edges.items()
    }
    graph = dgl.heterograph(graph_data, num_nodes_dict=num_nodes_dict)

    processed_features_by_type, processed_feature_dims, feature_metadata = build_feature_tensors(
        feature_mode,
        emat_root,
        processed_feature_root,
        dataset_key,
        config,
        nodes_by_type,
        raw_features_by_type,
        raw_feature_dims,
    )
    feature_dims = processed_feature_dims

    for node_type, items in nodes_by_type.items():
        original_ids = []
        for global_id, _, feature in items:
            original_ids.append(int(global_id))
        graph.nodes[node_type].data["h"] = processed_features_by_type[node_type]
        graph.nodes[node_type].data["global_id"] = torch.tensor(original_ids, dtype=torch.int64)

    category = config["category"]
    category_node_count = num_nodes_dict[category]
    train_labels = read_labels(dataset_dir / "label.dat", global_to_local, category, global_to_type)
    test_labels = read_labels(dataset_dir / "label.dat.test", global_to_local, category, global_to_type)
    test_full_labels = read_labels(dataset_dir / "label.dat.test_full", global_to_local, category, global_to_type)

    all_labels = {}
    all_labels.update(test_full_labels)
    all_labels.update(test_labels)
    all_labels.update(train_labels)

    label_tensor = torch.full((category_node_count,), -1, dtype=torch.long)
    for local_id, class_id in all_labels.items():
        label_tensor[local_id] = class_id
    train_ids, val_ids = stratified_train_val_split(train_labels, val_ratio=val_ratio, seed=seed)
    test_ids = set(test_labels.keys())
    test_full_ids = set(test_full_labels.keys())

    graph.nodes[category].data["label"] = label_tensor
    graph.nodes[category].data["train_mask"] = make_bool_mask(torch, category_node_count, train_ids)
    graph.nodes[category].data["val_mask"] = make_bool_mask(torch, category_node_count, val_ids)
    graph.nodes[category].data["test_mask"] = make_bool_mask(torch, category_node_count, test_ids)
    graph.nodes[category].data["test_full_mask"] = make_bool_mask(torch, category_node_count, test_full_ids)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(config["output_name"], feature_mode)
    graph_path = output_dir / f"{stem}.bin"
    dgl.save_graphs(str(graph_path), [graph])

    manifest = {
        "dataset": dataset_key,
        "output_name": config["output_name"],
        "feature_mode": feature_mode,
        "graph_path": str(graph_path),
        "category": category,
        "num_classes": config["num_classes"],
        "node_counts": num_nodes_dict,
        "edge_counts": {str(k): len(v[0]) for k, v in edges.items()},
        "feature_dims": feature_dims,
        "feature_metadata": feature_metadata,
        "emat_root": str(emat_root) if feature_mode != "raw" else "",
        "processed_feature_root": str(processed_feature_root),
        "node_classification_split": {
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "test_full": len(test_full_ids),
            "val_ratio_from_label_dat": val_ratio,
            "seed": seed,
        },
        "label_distribution": {
            "label.dat": dict(Counter(train_labels.values())),
            "label.dat.test": dict(Counter(test_labels.values())),
            "label.dat.test_full": dict(Counter(test_full_labels.values())),
        },
    }
    (output_dir / f"{stem}.manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Lisan ACM/DBLP datasets to OpenHGNN graph.bin files.")
    parser.add_argument("--datasets-root", type=Path, default=Path("Datasets"))
    parser.add_argument(
        "--openhgnn-root",
        type=Path,
        default=Path("OpenHGNN"),
        help="Path to the OpenHGNN source root containing openhgnn/dataset.",
    )
    parser.add_argument("--dataset", choices=["ACM", "DBLP", "all"], default="all")
    parser.add_argument(
        "--feature-mode",
        "--feature_mode",
        dest="feature_mode",
        choices=sorted(FEATURE_MODE_ALIASES) + ["all", "both"],
        default="raw",
        help="Feature mode. A=raw, B=direct EMat, C=EMat-SVD-128, D=EMat-TFIDF-3025, E=raw EMat sparse encoder.",
    )
    parser.add_argument(
        "--emat-root",
        type=Path,
        default=Path("Dataset_Emat"),
        help="Root containing Dataset_Emat/<ACM|DBLP>/EMat_<type>.pt.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--processed-feature-root",
        type=Path,
        default=Path("data/lisan_processed_features"),
        help="Root containing preprocessed EMat feature payloads.",
    )
    args = parser.parse_args()

    output_dir = args.openhgnn_root / "openhgnn" / "dataset" / "lisan_hgb"
    keys = ["ACM", "DBLP"] if args.dataset == "all" else [args.dataset]
    if args.feature_mode == "all":
        feature_modes = MATERIALIZABLE_FEATURE_MODES
    elif args.feature_mode == "both":
        feature_modes = ("raw", "emat")
    else:
        feature_modes = (normalize_feature_mode(args.feature_mode),)
    manifests = [
        convert_one(
            args.datasets_root.resolve(),
            output_dir.resolve(),
            key,
            args.val_ratio,
            args.seed,
            feature_mode,
            args.emat_root.resolve(),
            args.processed_feature_root.resolve(),
        )
        for key in keys
        for feature_mode in feature_modes
    ]
    for manifest in manifests:
        split = manifest["node_classification_split"]
        print(
            f"{manifest['output_name']}[{manifest['feature_mode']}]: graph={manifest['graph_path']} "
            f"train={split['train']} val={split['val']} test={split['test']} test_full={split['test_full']}"
        )


if __name__ == "__main__":
    main()
