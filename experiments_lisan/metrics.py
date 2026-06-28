from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Any


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        try:
            return float(value.item())
        except Exception:
            return None


def get_metric(metrics: dict[str, Any], *names: str) -> float | None:
    lowered = {str(k).lower(): v for k, v in metrics.items()}
    for name in names:
        value = lowered.get(name.lower())
        parsed = as_float(value)
        if parsed is not None:
            return parsed
    return None


def objective_from_result(task: str, result: dict[str, Any]) -> float:
    metrics = result.get("metric", {})
    valid = metrics.get("valid")
    if valid is None:
        raise KeyError("validation metrics are required for HPO objective")
    if task == "node_classification":
        score = get_metric(valid, "Macro_f1", "macro_f1")
    elif task == "link_prediction":
        score = get_metric(valid, "roc_auc", "AUC")
    else:
        score = None
    if score is None:
        raise KeyError(f"missing validation objective metric for {task}: {valid}")
    return score


def flatten_result_metrics(task: str, seed: int, result: dict[str, Any], status: str, notes: str = "") -> dict[str, Any]:
    metrics = result.get("metric", {}) if isinstance(result, dict) else {}
    valid = metrics.get("valid", {}) or {}
    test = metrics.get("test", {}) or {}
    row: dict[str, Any] = {"seed": seed, "status": status, "notes": notes}
    if task == "node_classification":
        valid_macro = get_metric(valid, "Macro_f1", "macro_f1")
        test_macro = get_metric(test, "Macro_f1", "macro_f1")
        test_micro = get_metric(test, "Micro_f1", "micro_f1", "Mirco_f1")
        row.update(
            {
                "valid_macro_f1": valid_macro,
                "valid_micro_f1": get_metric(valid, "Micro_f1", "micro_f1", "Mirco_f1"),
                "test_macro_f1": test_macro,
                "test_micro_f1": test_micro,
                "test_accuracy": test_micro,
            }
        )
    else:
        row.update(
            {
                "valid_auc": get_metric(valid, "roc_auc", "AUC"),
                "valid_loss": get_metric(valid, "loss"),
                "test_auc": get_metric(test, "roc_auc", "AUC"),
                "test_loss": get_metric(test, "loss"),
                "test_ap": None,
                "test_mrr": None,
                "test_hits10": None,
            }
        )
    return row


def summarize_values(rows: list[dict[str, Any]], key: str) -> tuple[float | None, float | None]:
    values = [as_float(row.get(key)) for row in rows if row.get("status") == "completed"]
    values = [v for v in values if v is not None and not math.isnan(v)]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return mean(values), stdev(values)

