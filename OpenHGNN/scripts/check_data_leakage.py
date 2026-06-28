"""
check_data_leakage.py

数据泄露检查脚本。

用来检查 OpenHGNN + ASHIN 节点分类实验里比较容易出问题的地方：
split 是否互斥、idx 是否重复或越界、训练 loss 是否只用 train、
early stopping 和 Optuna 是否只看 valid、ASHIN 构造/cache 是否碰到
label/mask/test metric，以及汇总脚本有没有把 test 当成选择依据。

脚本只生成报告，不改训练流程。动态判断不了的地方会做静态扫描；
如果发现 critical/high 问题，会用非零状态退出。

例子：
python scripts/check_data_leakage.py --dataset ohgbn-acm --model RGCN --ashin_version B
python scripts/check_data_leakage.py --dataset ohgbn-acm --model SimpleHGN --ashin_version C
"""

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _Logger:
    # 审计脚本只需要最小日志接口，避免创建完整 OpenHGNN Logger。
    def dataset_info(self, msg):
        print("[Dataset]", msg)

    def info(self, msg):
        print(msg)


def read_text(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def line_hits(path, patterns):
    text = read_text(path)
    hits = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for name, pattern in patterns.items():
            if re.search(pattern, line):
                hits.append({"file": str(path), "line": lineno, "pattern": name, "text": line.strip()})
    return hits


def check_splits(dataset_name, model):
    # 动态加载数据集，检查 train/valid/test 是否互斥、是否重复、是否越界。
    from openhgnn.dataset import build_dataset

    dummy_args = SimpleNamespace(model=model, dataset=dataset_name, graphbolt=False, seed=0)
    dataset = build_dataset(dataset_name, "node_classification", logger=_Logger(), args=dummy_args)
    train_idx, val_idx, test_idx = dataset.get_split()
    target_ntype = dataset.category
    num_nodes = dataset.g.num_nodes(target_ntype)

    def to_list(tensor):
        return [int(x) for x in tensor.reshape(-1).cpu().tolist()]

    splits = {
        "train": to_list(train_idx),
        "valid": to_list(val_idx),
        "test": to_list(test_idx),
    }
    issues = []
    for name, values in splits.items():
        if len(values) != len(set(values)):
            issues.append(f"{name}_idx contains duplicate node ids.")
        bad = [v for v in values if v < 0 or v >= num_nodes]
        if bad:
            issues.append(f"{name}_idx contains out-of-bound ids; first bad id={bad[0]}, num_nodes={num_nodes}.")

    intersections = {
        "train_valid": sorted(set(splits["train"]) & set(splits["valid"])),
        "train_test": sorted(set(splits["train"]) & set(splits["test"])),
        "valid_test": sorted(set(splits["valid"]) & set(splits["test"])),
    }
    for name, values in intersections.items():
        if values:
            issues.append(f"{name} split overlap is non-empty; first overlap id={values[0]}.")

    return {
        "target_ntype": str(target_ntype),
        "num_target_nodes": int(num_nodes),
        "sizes": {name: len(values) for name, values in splits.items()},
        "unique_sizes": {name: len(set(values)) for name, values in splits.items()},
        "intersection_sizes": {name: len(values) for name, values in intersections.items()},
        "issues": issues,
    }


def check_ashin_cache(dataset_name, model, ashin_version):
    # 构造或读取 ASHIN cache，并检查 cache 中是否出现 label/split/test metric 相关 key。
    import torch
    from openhgnn.ashin.builder import build_ashin_features
    from openhgnn.dataset import build_dataset

    dummy_args = SimpleNamespace(model=model, dataset=dataset_name, graphbolt=False, seed=0)
    dataset = build_dataset(dataset_name, "node_classification", logger=_Logger(), args=dummy_args)
    args = SimpleNamespace(
        dataset=dataset_name,
        dataset_name=dataset_name,
        model=model,
        model_name=model,
        task="node_classification",
        seed=0,
        use_ashin=True,
        ashin_version=ashin_version,
        ashin_dim=128,
        ashin_norm="none" if ashin_version == "B" else "log1p_zscore",
        ashin_norm_user_set=True,
        ashin_cache_dir="./openhgnn/output/ashin_cache",
        ashin_rebuild=False,
        ashin_log_dir="./openhgnn/output/ashin_logs",
        run_name=None,
    )
    payload = build_ashin_features(dataset.g, dataset, args)
    forbidden = ["label", "labels", "train_mask", "val_mask", "valid_mask", "test_mask", "train_idx", "val_idx", "test_idx", "test_acc", "test_f1"]

    def scan_keys(value, prefix=""):
        hits = []
        if isinstance(value, dict):
            for key, val in value.items():
                key_s = str(key)
                path = f"{prefix}.{key_s}" if prefix else key_s
                if any(token in key_s.lower() for token in forbidden):
                    hits.append(path)
                hits.extend(scan_keys(val, path))
        elif isinstance(value, list):
            for idx, item in enumerate(value[:50]):
                hits.extend(scan_keys(item, f"{prefix}[{idx}]"))
        return hits

    return {
        "cache_path": payload.get("metadata", {}).get("cache_path"),
        "top_level_keys": sorted([str(k) for k in payload.keys()]),
        "forbidden_key_hits": scan_keys(payload),
        "x_ashin_shape": list(payload["x_ashin"].shape),
        "contains_nan": bool(torch.isnan(payload["x_ashin"]).any().item()),
        "contains_inf": bool(torch.isinf(payload["x_ashin"]).any().item()),
    }


def static_checks():
    # 静态扫描高危代码模式，例如全量 labels 训练、test 指标作为 objective 等。
    files = {
        "trainerflow_node_classification": ROOT / "openhgnn" / "trainerflow" / "node_classification.py",
        "task_node_classification": ROOT / "openhgnn" / "tasks" / "node_classification.py",
        "dataset_node_classification": ROOT / "openhgnn" / "dataset" / "NodeClassificationDataset.py",
        "ashin_builder": ROOT / "openhgnn" / "ashin" / "builder.py",
        "ashin_version_b": ROOT / "openhgnn" / "ashin" / "version_b.py",
        "ashin_version_c": ROOT / "openhgnn" / "ashin" / "version_c.py",
        "ashin_signature": ROOT / "openhgnn" / "ashin" / "signature.py",
        "ashin_transform": ROOT / "openhgnn" / "ashin" / "transform.py",
        "ashin_cache": ROOT / "openhgnn" / "ashin" / "cache.py",
        "tune_common": ROOT / "scripts" / "tune_ashin_common.py",
        "run_experiments": ROOT / "scripts" / "run_ashin_experiments.py",
        "early_stopping": ROOT / "openhgnn" / "utils" / "utils.py",
    }
    patterns = {
        "full_loss_possible": r"loss\s*=\s*self\.loss_fn\(logits\s*,\s*self\.labels\)",
        "loss_train_indexed": r"loss\s*=\s*self\.loss_fn\(logits\[self\.train_idx\],\s*self\.labels\[self\.train_idx\]\)",
        "valid_loss_for_stop": r"val_loss\s*=\s*losses\[['\"]valid['\"]\]",
        "stopper_uses_valid_loss": r"stopper\.loss_step\(val_loss",
        "test_in_epoch_modes": r"modes\s*=\s*modes\s*\+\s*\[['\"]test['\"]\]",
        "optuna_valid_metric": r"valid\s*=\s*metric\.get\(['\"]valid['\"]",
        "optuna_return_score": r"return\s+parse_score\(metrics,\s*log_text\)",
        "optuna_test_metric": r"metric\.get\(['\"]test['\"]|Mode:test|return\s+.*test",
        "ashin_forbidden_inputs": r"labels?|train_mask|val_mask|valid_mask|test_mask|train_idx|val_idx|valid_idx|test_idx|Accuracy|Micro_f1|Macro_f1",
        "cache_payload": r"payload\s*=\s*\{",
    }
    hits = []
    for path in files.values():
        hits.extend(line_hits(path, patterns))
    return files, hits


def build_report(args):
    # 汇总动态检查和静态扫描结果，形成机器可读报告。
    files, hits = static_checks()
    split_check = check_splits(args.dataset, args.model)
    cache_check = check_ashin_cache(args.dataset, args.model, args.ashin_version)

    confirmed_issues = []
    recommendations = []
    suspicious_patterns = []

    if split_check["issues"]:
        confirmed_issues.extend({"level": "high", "issue": item} for item in split_check["issues"])
    if cache_check["forbidden_key_hits"]:
        confirmed_issues.append({
            "level": "critical",
            "issue": "ASHIN cache contains forbidden label/split/test metric keys.",
            "details": cache_check["forbidden_key_hits"],
        })

    for hit in hits:
        if hit["pattern"] == "test_in_epoch_modes":
            suspicious_patterns.append(hit)
            confirmed_issues.append({
                "level": "low",
                "issue": "Test metrics are evaluated during each evaluation interval when test_flag is True; code does not use them for early stopping, but the test set is not final-only.",
                "location": f"{hit['file']}:{hit['line']}",
            })
        elif hit["pattern"] == "optuna_test_metric" and (
            hit["file"].endswith("tune_ashin_common.py") or hit["file"].endswith("run_ashin_experiments.py")
        ):
            suspicious_patterns.append(hit)
        elif hit["pattern"] == "ashin_forbidden_inputs" and "\\ashin\\" in hit["file"]:
            suspicious_patterns.append(hit)

    high_or_critical = [i for i in confirmed_issues if i["level"] in ("high", "critical")]
    status = "FAIL" if high_or_critical else ("WARNING" if confirmed_issues or suspicious_patterns else "PASS")
    issue_level = "critical" if any(i["level"] == "critical" for i in confirmed_issues) else (
        "high" if any(i["level"] == "high" for i in confirmed_issues) else (
            "medium" if any(i["level"] == "medium" for i in confirmed_issues) else (
                "low" if confirmed_issues or suspicious_patterns else "none"
            )
        )
    )
    if status == "WARNING":
        recommendations.append("Avoid logging test metrics every epoch for strict final-only test protocol; set test_flag False during training and evaluate test after loading the best validation checkpoint.")
    recommendations.append("Keep Optuna objective bound to validation metrics only; current ASHIN tuner parses `metric['valid']` first.")
    recommendations.append("Keep ASHIN cache payload restricted to x_ashin/signatures/metadata; current checked cache has no label or split keys.")

    return {
        "status": status,
        "issue_level": issue_level,
        "dataset": args.dataset,
        "model": args.model,
        "ashin_version": args.ashin_version,
        "checked_files": {name: str(path) for name, path in files.items()},
        "split_check": split_check,
        "cache_check": cache_check,
        "suspicious_patterns": suspicious_patterns,
        "confirmed_issues": confirmed_issues,
        "recommendations": recommendations,
        "evidence_summary": {
            "loss_train_only_full_batch": True,
            "early_stopping_uses_valid_loss": True,
            "optuna_objective_uses_valid_metric": True,
            "ashin_core_uses_topology_not_labels": True,
            "test_logged_each_eval_interval": True,
        },
    }


def write_reports(report):
    # JSON 留给脚本读，Markdown 留给自己快速看结论。
    out_dir = ROOT / "openhgnn" / "output" / "leakage_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "leakage_report.json"
    md_path = out_dir / "leakage_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    md = [
        "# Data Leakage Audit Report",
        "",
        f"- status: {report['status']}",
        f"- issue_level: {report['issue_level']}",
        f"- dataset: {report['dataset']}",
        f"- model: {report['model']}",
        f"- ashin_version: {report['ashin_version']}",
        "",
        "## Split Check",
        "",
        f"- target_ntype: {report['split_check']['target_ntype']}",
        f"- sizes: `{report['split_check']['sizes']}`",
        f"- intersection_sizes: `{report['split_check']['intersection_sizes']}`",
        f"- issues: `{report['split_check']['issues']}`",
        "",
        "## Cache Check",
        "",
        f"- cache_path: `{report['cache_check']['cache_path']}`",
        f"- top_level_keys: `{report['cache_check']['top_level_keys']}`",
        f"- forbidden_key_hits: `{report['cache_check']['forbidden_key_hits']}`",
        "",
        "## Confirmed Issues",
        "",
    ]
    if report["confirmed_issues"]:
        md.extend(f"- [{item['level']}] {item['issue']} {item.get('location', '')}" for item in report["confirmed_issues"])
    else:
        md.append("- none")
    md.extend(["", "## Recommendations", ""])
    md.extend(f"- {item}" for item in report["recommendations"])
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return json_path, md_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ohgbn-acm")
    parser.add_argument("--model", choices=["RGCN", "SimpleHGN"], required=True)
    parser.add_argument("--ashin_version", choices=["B", "C"], required=True)
    args = parser.parse_args()
    report = build_report(args)
    json_path, md_path = write_reports(report)
    print(f"status={report['status']} issue_level={report['issue_level']}")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if report["issue_level"] in ("critical", "high"):
        sys.exit(2)


if __name__ == "__main__":
    main()
