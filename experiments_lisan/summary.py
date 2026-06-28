from pathlib import Path
from typing import Any

from .config import LINK_SUMMARY_COLUMNS, NODE_SUMMARY_COLUMNS
from .io_utils import write_csv_rows
from .metrics import summarize_values


def build_final_row(
    task: str,
    dataset: str,
    model: str,
    feature_mode: str,
    n_trials: int,
    seeds: list[int],
    best_params_path: Path,
    status: str,
    notes: str,
    seed_rows: list[dict[str, Any]],
    best_values: dict[str, Any],
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "task": task,
        "dataset": dataset,
        "model": model,
        "feature_mode": feature_mode,
        "n_trials": n_trials,
        "seeds": " ".join(str(seed) for seed in seeds),
        "best_params_path": str(best_params_path) if best_params_path else "",
        "status": status,
        "notes": notes,
    }
    if task == "node_classification":
        macro_mean, macro_std = summarize_values(seed_rows, "test_macro_f1")
        micro_mean, micro_std = summarize_values(seed_rows, "test_micro_f1")
        acc_mean, acc_std = summarize_values(seed_rows, "test_accuracy")
        row.update(
            {
                "best_val_macro_f1": best_values.get("best_val_macro_f1"),
                "test_macro_f1_mean": macro_mean,
                "test_macro_f1_std": macro_std,
                "test_micro_f1_mean": micro_mean,
                "test_micro_f1_std": micro_std,
                "test_accuracy_mean": acc_mean,
                "test_accuracy_std": acc_std,
            }
        )
    else:
        auc_mean, auc_std = summarize_values(seed_rows, "test_auc")
        row.update(
            {
                "best_val_auc": best_values.get("best_val_auc"),
                "best_val_ap": "",
                "test_auc_mean": auc_mean,
                "test_auc_std": auc_std,
                "test_ap_mean": "",
                "test_ap_std": "",
                "test_mrr_mean": "",
                "test_mrr_std": "",
                "test_hits10_mean": "",
                "test_hits10_std": "",
            }
        )
    return row


def write_final_summaries(output_root: Path, rows: list[dict[str, Any]]) -> None:
    node_rows = [row for row in rows if row.get("task") == "node_classification"]
    link_rows = [row for row in rows if row.get("task") == "link_prediction"]
    write_csv_rows(output_root / "final_summary_node_classification.csv", node_rows, NODE_SUMMARY_COLUMNS)
    write_csv_rows(output_root / "final_summary_link_prediction.csv", link_rows, LINK_SUMMARY_COLUMNS)
