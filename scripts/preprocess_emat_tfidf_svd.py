from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_lisan.emat_preprocessing import (  # noqa: E402
    apply_tfidf,
    emat_tensor_to_csr,
    fit_svd_features,
    load_emat_tensors,
    processed_dataset_dir,
    read_nodes,
    resolve_dataset_key,
    save_feature_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Lisan EMat features with TF-IDF followed by TruncatedSVD.")
    parser.add_argument("--datasets", nargs="+", default=["lisan-acm", "lisan-dblp"])
    parser.add_argument("--datasets-root", type=Path, default=Path("Datasets"))
    parser.add_argument("--emat-root", type=Path, default=Path("Dataset_Emat"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/lisan_processed_features"))
    parser.add_argument("--svd_dims", nargs="+", type=int, default=[64, 128, 256])
    parser.add_argument("--tf_mode", choices=["binary", "raw", "log1p"], default="log1p")
    parser.add_argument("--normalize", choices=["none", "zscore", "l2"], default="zscore")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--per_ntype", action="store_true", default=True)
    return parser.parse_args()


def preprocess_dataset(args: argparse.Namespace, dataset: str) -> None:
    dataset_key = resolve_dataset_key(dataset)
    nodes_by_type = read_nodes(args.datasets_root / dataset_key, dataset_key)
    tensors = load_emat_tensors(args.emat_root, dataset_key, nodes_by_type)
    weighted_matrices = {}
    tfidf_metadata = {}
    for node_type, tensor in tensors.items():
        weighted, idf, df = apply_tfidf(emat_tensor_to_csr(tensor), args.tf_mode)
        weighted_matrices[node_type] = weighted
        tfidf_metadata[node_type] = {
            "idf_mean": float(idf.mean()) if idf.size else 0.0,
            "idf_max": float(idf.max()) if idf.size else 0.0,
            "df_min": int(df.min()) if df.size else 0,
            "df_max": int(df.max()) if df.size else 0,
            "df_zero": int((df == 0).sum()) if df.size else 0,
        }
    out_dir = processed_dataset_dir(args.output_dir, dataset_key)
    for dim in args.svd_dims:
        features = {}
        actual_dims = {}
        metadata_by_type = {}
        for node_type, matrix in weighted_matrices.items():
            reduced, metadata = fit_svd_features(matrix, dim, args.normalize, args.seed)
            features[node_type] = torch.tensor(reduced, dtype=torch.float32)
            actual_dims[node_type] = metadata["actual_dim"]
            metadata_by_type[node_type] = metadata
        payload = {
            "dataset": dataset_key,
            "method": "TFIDF+TruncatedSVD",
            "tf_mode": args.tf_mode,
            "dim": dim,
            "features": features,
            "metadata": {
                "per_ntype": True,
                "normalize": args.normalize,
                "fit_scope": "all_nodes",
                "actual_dims": actual_dims,
                "tfidf": tfidf_metadata,
                "by_node_type": metadata_by_type,
                "random_state": args.seed,
            },
        }
        save_feature_payload(out_dir / f"emat_tfidf_svd_{dim}.pt", payload)
        print(f"{dataset_key}: wrote emat_tfidf_svd_{dim}.pt")


def main() -> int:
    args = parse_args()
    for dataset in args.datasets:
        preprocess_dataset(args, dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
