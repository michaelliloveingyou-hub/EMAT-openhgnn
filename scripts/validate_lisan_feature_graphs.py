from __future__ import annotations

import argparse
import json
from pathlib import Path


DATASETS = {
    "lisan-acm": {
        "raw": "lisan-acm.bin",
        "emat": "lisan-acm-emat.bin",
        "emat_svd_128": "lisan-acm-emat-svd-128.bin",
        "emat_tfidf_3025": "lisan-acm-emat-tfidf-3025.bin",
    },
    "lisan-dblp": {
        "raw": "lisan-dblp.bin",
        "emat": "lisan-dblp-emat.bin",
        "emat_svd_128": "lisan-dblp-emat-svd-128.bin",
        "emat_tfidf_3025": "lisan-dblp-emat-tfidf-3025.bin",
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
SPARSE_ENCODER_MODES = {"emat_sparse_encoder", "raw_emat_sparse_encoder"}


def normalize_modes(values: list[str]) -> list[str]:
    return [FEATURE_MODE_ALIASES[value] for value in values]


def tensor_stats(torch_module, tensor):
    finite = torch_module.isfinite(tensor)
    nnz = int((tensor != 0).sum().item())
    total = tensor.numel()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "finite": bool(finite.all().item()),
        "nonzero": nnz,
        "zero_ratio": 1.0 - (nnz / total if total else 0.0),
        "min": float(tensor.min().item()) if total else 0.0,
        "max": float(tensor.max().item()) if total else 0.0,
        "mean": float(tensor.float().mean().item()) if total else 0.0,
    }


def validate_dataset(dgl_module, torch_module, graph_dir: Path, dataset: str, feature_modes: list[str]) -> dict:
    files = DATASETS[dataset]
    raw_graph = dgl_module.load_graphs(str(graph_dir / files["raw"]))[0][0]
    dataset_report = {"dataset": dataset, "feature_modes": {}}

    for mode in feature_modes:
        if mode in SPARSE_ENCODER_MODES:
            dataset_report["feature_modes"][mode] = {
                "status": "skipped",
                "reason": "learnable runtime encoder mode; no static graph.bin is expected",
            }
            continue
        filename = files.get(mode)
        if filename is None:
            filename = f"{files['raw'][:-4]}-{mode.replace('_', '-')}.bin"
        path = graph_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing graph file for {dataset} {mode}: {path}")
        graph = dgl_module.load_graphs(str(path))[0][0]
        mode_report = {
            "path": str(path),
            "canonical_etypes_match": raw_graph.canonical_etypes == graph.canonical_etypes,
            "edges": {},
            "nodes": {},
        }
        if raw_graph.canonical_etypes != graph.canonical_etypes:
            raise AssertionError(f"{dataset} {mode}: canonical edge types differ from raw")

        for etype in raw_graph.canonical_etypes:
            raw_edges = raw_graph.edges(etype=etype)
            other_edges = graph.edges(etype=etype)
            count_match = raw_graph.num_edges(etype) == graph.num_edges(etype)
            order_match = all(torch_module.equal(a, b) for a, b in zip(raw_edges, other_edges))
            mode_report["edges"][str(etype)] = {
                "count": raw_graph.num_edges(etype),
                "count_match": count_match,
                "order_match": order_match,
            }
            if not count_match or not order_match:
                raise AssertionError(f"{dataset} {mode}: edge mismatch for {etype}")

        for ntype in sorted(raw_graph.ntypes):
            raw_data = raw_graph.nodes[ntype].data
            data = graph.nodes[ntype].data
            gid_match = torch_module.equal(raw_data["global_id"], data["global_id"])
            node_report = {
                "node_count": graph.num_nodes(ntype),
                "global_id_match": gid_match,
                "h": tensor_stats(torch_module, data["h"]),
            }
            if not gid_match:
                raise AssertionError(f"{dataset} {mode}: global_id mismatch for {ntype}")
            for key in ["label", "train_mask", "val_mask", "test_mask", "test_full_mask"]:
                if key in raw_data or key in data:
                    match = key in raw_data and key in data and torch_module.equal(raw_data[key], data[key])
                    node_report[f"{key}_match"] = match
                    if not match:
                        raise AssertionError(f"{dataset} {mode}: {key} mismatch for {ntype}")
            mode_report["nodes"][ntype] = node_report
        dataset_report["feature_modes"][mode] = mode_report
    return dataset_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Lisan raw and EMat feature graph variants.")
    parser.add_argument("--datasets", nargs="+", choices=sorted(DATASETS), default=sorted(DATASETS))
    parser.add_argument("--feature_modes", nargs="+", default=["A", "B", "C", "D", "E"])
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("OpenHGNN") / "openhgnn" / "dataset" / "lisan_hgb",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "lisan_processed_features" / "feature_graph_validation.json",
    )
    args = parser.parse_args()

    import dgl
    import torch

    feature_modes = normalize_modes(args.feature_modes)
    report = {
        "graph_dir": str(args.graph_dir.resolve()),
        "feature_modes": feature_modes,
        "datasets": [],
    }
    for dataset in args.datasets:
        report["datasets"].append(validate_dataset(dgl, torch, args.graph_dir.resolve(), dataset, feature_modes))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Validation passed. Report written to {args.output}")


if __name__ == "__main__":
    main()
