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
    load_emat_tensors,
    processed_dataset_dir,
    read_nodes,
    resolve_dataset_key,
    save_feature_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Lisan EMat features with per-node-type TF-IDF weighting.")
    parser.add_argument("--datasets", nargs="+", default=["lisan-acm", "lisan-dblp"])
    parser.add_argument("--datasets-root", type=Path, default=Path("Datasets"))
    parser.add_argument("--emat-root", type=Path, default=Path("Dataset_Emat"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/lisan_processed_features"))
    parser.add_argument("--tf_mode", choices=["binary", "raw", "log1p"], default="log1p")
    parser.add_argument("--per_ntype", action="store_true", default=True)
    parser.add_argument("--idf_smooth", action="store_true", default=True)
    return parser.parse_args()


def preprocess_dataset(args: argparse.Namespace, dataset: str) -> None:
    dataset_key = resolve_dataset_key(dataset)
    nodes_by_type = read_nodes(args.datasets_root / dataset_key, dataset_key)
    tensors = load_emat_tensors(args.emat_root, dataset_key, nodes_by_type)
    features = {}
    metadata = {"per_ntype": True, "tf_mode": args.tf_mode, "fit_scope": "all_nodes", "df_stats": {}, "idf": {}}
    for node_type, tensor in tensors.items():
        matrix = emat_tensor_to_csr(tensor)
        weighted, idf, df = apply_tfidf(matrix, args.tf_mode)
        dense = weighted.toarray().astype("float32", copy=False)
        features[node_type] = torch.tensor(dense, dtype=torch.float32)
        metadata["idf"][node_type] = torch.tensor(idf, dtype=torch.float32)
        metadata["df_stats"][node_type] = {
            "min": int(df.min()) if df.size else 0,
            "max": int(df.max()) if df.size else 0,
            "mean": float(df.mean()) if df.size else 0.0,
            "zero_df": int((df == 0).sum()) if df.size else 0,
        }
    payload = {
        "dataset": dataset_key,
        "method": "TFIDF",
        "tf_mode": args.tf_mode,
        "features": features,
        "metadata": metadata,
    }
    out_dir = processed_dataset_dir(args.output_dir, dataset_key)
    save_feature_payload(out_dir / "emat_tfidf.pt", payload)
    print(f"{dataset_key}: wrote emat_tfidf.pt")


def main() -> int:
    args = parse_args()
    for dataset in args.datasets:
        preprocess_dataset(args, dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
