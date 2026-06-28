from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_lisan.emat_preprocessing import (  # noqa: E402
    emat_tensor_to_csr,
    load_emat_tensors,
    processed_dataset_dir,
    read_nodes,
    resolve_dataset_key,
    save_feature_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Save CSR-style sparse EMat inputs for learnable sparse encoders.")
    parser.add_argument("--datasets", nargs="+", default=["lisan-acm", "lisan-dblp"])
    parser.add_argument("--datasets-root", type=Path, default=Path("Datasets"))
    parser.add_argument("--emat-root", type=Path, default=Path("Dataset_Emat"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/lisan_processed_features"))
    return parser.parse_args()


def preprocess_dataset(args: argparse.Namespace, dataset: str) -> None:
    dataset_key = resolve_dataset_key(dataset)
    nodes_by_type = read_nodes(args.datasets_root / dataset_key, dataset_key)
    tensors = load_emat_tensors(args.emat_root, dataset_key, nodes_by_type)
    sparse_inputs = {}
    for node_type, tensor in tensors.items():
        matrix = emat_tensor_to_csr(tensor)
        sparse_inputs[node_type] = {
            "indices": torch.tensor(matrix.indices, dtype=torch.long),
            "values": torch.tensor(matrix.data, dtype=torch.float32),
            "indptr": torch.tensor(matrix.indptr, dtype=torch.long),
            "shape": tuple(int(v) for v in matrix.shape),
            "num_basis": int(matrix.shape[1]),
        }
    payload = {
        "dataset": dataset_key,
        "method": "CSR sparse EMat inputs for EMatSparseEncoder",
        "sparse_inputs": sparse_inputs,
        "metadata": {"fit_scope": "all_nodes", "leakage_risk_for_link_prediction": "Unknown"},
    }
    out_dir = processed_dataset_dir(args.output_dir, dataset_key)
    save_feature_payload(out_dir / "emat_sparse_encoder_inputs.pt", payload)
    print(f"{dataset_key}: wrote emat_sparse_encoder_inputs.pt")


def main() -> int:
    args = parse_args()
    for dataset in args.datasets:
        preprocess_dataset(args, dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
