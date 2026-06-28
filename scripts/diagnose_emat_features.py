from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_lisan.emat_preprocessing import (  # noqa: E402
    diagnostics_for_matrix,
    emat_tensor_to_csr,
    load_emat_tensors,
    processed_dataset_dir,
    read_nodes,
    resolve_dataset_key,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose Lisan EMat feature sparsity and value statistics.")
    parser.add_argument("--datasets", nargs="+", default=["lisan-acm", "lisan-dblp"])
    parser.add_argument("--datasets-root", type=Path, default=Path("Datasets"))
    parser.add_argument("--emat-root", type=Path, default=Path("Dataset_Emat"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/lisan_processed_features"))
    return parser.parse_args()


def diagnose_dataset(args: argparse.Namespace, dataset: str) -> dict:
    dataset_key = resolve_dataset_key(dataset)
    dataset_dir = args.datasets_root / dataset_key
    nodes_by_type = read_nodes(dataset_dir, dataset_key)
    tensors = load_emat_tensors(args.emat_root, dataset_key, nodes_by_type)
    report = {"dataset": dataset_key, "node_types": {}}
    rows = []
    for node_type, tensor in tensors.items():
        matrix = emat_tensor_to_csr(tensor)
        stats = diagnostics_for_matrix(matrix)
        report["node_types"][node_type] = stats
        rows.append(
            {
                "dataset": dataset_key,
                "node_type": node_type,
                "shape": "x".join(str(value) for value in stats["shape"]),
                "nnz": stats["nnz"],
                "nonzero_ratio": stats["nonzero_ratio"],
                "zero_ratio": stats["zero_ratio"],
                "nnz_per_node_mean": stats["nnz_per_node"]["mean"],
                "nnz_per_node_std": stats["nnz_per_node"]["std"],
                "nnz_per_node_median": stats["nnz_per_node"]["median"],
                "nnz_per_node_min": stats["nnz_per_node"]["min"],
                "nnz_per_node_max": stats["nnz_per_node"]["max"],
                "zero_columns": stats["zero_columns"],
                "df_lt_1": stats["df_lt_1"],
                "df_lt_5": stats["df_lt_5"],
                "df_lt_10": stats["df_lt_10"],
                "df_lt_20": stats["df_lt_20"],
                "variance_mean": stats["variance"]["mean"],
                "variance_max": stats["variance"]["max"],
                "is_binary": stats["is_binary"],
                "has_count_values": stats["has_count_values"],
                "min": stats["min"],
                "max": stats["max"],
                "mean": stats["mean"],
                "has_nan": stats["has_nan"],
                "has_inf": stats["has_inf"],
            }
        )

    out_dir = processed_dataset_dir(args.output_dir, dataset_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "emat_diagnostics.json", report)
    with (out_dir / "emat_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["dataset", "node_type"])
        writer.writeheader()
        writer.writerows(rows)
    return report


def main() -> int:
    args = parse_args()
    for dataset in args.datasets:
        report = diagnose_dataset(args, dataset)
        print(f"{report['dataset']}: wrote diagnostics for {len(report['node_types'])} node types")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
