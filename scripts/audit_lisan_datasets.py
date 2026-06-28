# 项目文件功能说明：
# 本文件是 Lisan_project 的数据集详细读取与审计脚本。
# 它只读取 Datasets 目录下各子数据集的 node.dat、link.dat、label.dat、info.dat、meta.dat 等文件，
# 汇总节点类型、特征维度、边类型、标签划分、端点一致性、孤立节点和链接预测候选关系，
# 并可输出为 Markdown 报告，方便后续判断节点分类与链接预测任务是否可直接使用。

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def read_json_file(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle), None
    except Exception as exc:  # noqa: BLE001 - report parser problems instead of failing the whole scan.
        return None, f"{type(exc).__name__}: {exc}"


def count_lines(path: Path) -> int:
    total = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for _ in handle:
            total += 1
    return total


def short_text(value: str, limit: int = 90) -> str:
    value = value.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def percent(part: int | float, total: int | float) -> str:
    if not total:
        return "0.00%"
    return f"{(part / total) * 100:.2f}%"


def mean_value(total: int | float, count: int) -> str:
    if count <= 0:
        return "0.00"
    return f"{total / count:.2f}"


def sorted_type_keys(values: Iterable[str]) -> List[str]:
    def key_func(value: str) -> Tuple[int, Any]:
        try:
            return (0, int(value))
        except ValueError:
            return (1, value)

    return sorted(values, key=key_func)


def ensure_counter_text(counter: Counter[str], empty: str = "none") -> str:
    if not counter:
        return empty
    return ", ".join(f"{key}: {counter[key]}" for key in sorted_type_keys(counter.keys()))


def get_info_node_names(info: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not info:
        return {}
    node_block = info.get("node.dat", {})
    node_types = node_block.get("node type", {}) if isinstance(node_block, dict) else {}
    return {str(key): str(value) for key, value in node_types.items()}


def get_info_attribute_dims(info: Optional[Dict[str, Any]]) -> Dict[str, str]:
    if not info:
        return {}
    node_block = info.get("node.dat", {})
    dims = node_block.get("Attribute Dimension", {}) if isinstance(node_block, dict) else {}
    return {str(key): str(value) for key, value in dims.items()}


def get_info_edge_specs(info: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    if not info:
        return {}
    link_block = info.get("link.dat", {})
    link_types = link_block.get("link type", {}) if isinstance(link_block, dict) else {}
    specs: Dict[str, Dict[str, str]] = {}
    for key, value in link_types.items():
        if not isinstance(value, dict):
            continue
        specs[str(key)] = {
            "start": str(value.get("start", "")),
            "end": str(value.get("end", "")),
            "meaning": str(value.get("meaning", "")),
        }
    return specs


def get_info_label_names(info: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, str]]:
    if not info:
        return {}
    label_block = info.get("label.dat", {})
    node_types = label_block.get("node type", {}) if isinstance(label_block, dict) else {}
    result: Dict[str, Dict[str, str]] = {}
    for node_type, class_map in node_types.items():
        if isinstance(class_map, dict):
            result[str(node_type)] = {str(key): str(value) for key, value in class_map.items()}
    return result


def scan_files(dataset_dir: Path) -> Dict[str, Dict[str, Any]]:
    files: Dict[str, Dict[str, Any]] = {}
    for path in sorted(dataset_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file():
            continue
        files[path.name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "lines": count_lines(path),
        }
    return files


def parse_node_file(path: Path) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "exists": path.exists(),
        "total": 0,
        "malformed": 0,
        "malformed_samples": [],
        "duplicate_ids": 0,
        "duplicate_samples": [],
        "id_min": None,
        "id_max": None,
        "node_types": Counter(),
        "samples_by_type": defaultdict(list),
        "feature_by_type": defaultdict(
            lambda: {
                "records": 0,
                "with_feature": 0,
                "dims": Counter(),
                "nonzero_total": 0,
                "nonzero_min": None,
                "nonzero_max": None,
                "binary_like_records": 0,
                "empty_feature_records": 0,
            }
        ),
        "node_id_to_type": {},
        "node_id_to_name": {},
    }
    if not path.exists():
        return stats

    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n\r")
            parts = line.split("\t")
            if len(parts) < 3:
                stats["malformed"] += 1
                if len(stats["malformed_samples"]) < 5:
                    stats["malformed_samples"].append({"line": line_number, "raw": short_text(line)})
                continue

            node_id, node_name, node_type = parts[0], parts[1], parts[2]
            stats["total"] += 1
            stats["node_types"][node_type] += 1
            stats["node_id_to_type"][node_id] = node_type
            stats["node_id_to_name"][node_id] = node_name

            if node_id in seen_ids:
                stats["duplicate_ids"] += 1
                if len(stats["duplicate_samples"]) < 5:
                    stats["duplicate_samples"].append({"line": line_number, "node_id": node_id})
            seen_ids.add(node_id)

            try:
                numeric_id = int(node_id)
                stats["id_min"] = numeric_id if stats["id_min"] is None else min(stats["id_min"], numeric_id)
                stats["id_max"] = numeric_id if stats["id_max"] is None else max(stats["id_max"], numeric_id)
            except ValueError:
                pass

            if len(stats["samples_by_type"][node_type]) < 3:
                stats["samples_by_type"][node_type].append({"id": node_id, "name": short_text(node_name)})

            feature_raw = parts[3].strip() if len(parts) >= 4 else ""
            feature_stats = stats["feature_by_type"][node_type]
            feature_stats["records"] += 1
            if not feature_raw:
                feature_stats["empty_feature_records"] += 1
                feature_stats["dims"]["0"] += 1
                continue

            values = [value.strip() for value in feature_raw.split(",")]
            dim = len(values)
            nonzero = sum(1 for value in values if value not in {"", "0", "0.0"})
            binary_like = all(value in {"", "0", "1", "0.0", "1.0"} for value in values)

            feature_stats["with_feature"] += 1
            feature_stats["dims"][str(dim)] += 1
            feature_stats["nonzero_total"] += nonzero
            feature_stats["nonzero_min"] = (
                nonzero
                if feature_stats["nonzero_min"] is None
                else min(feature_stats["nonzero_min"], nonzero)
            )
            feature_stats["nonzero_max"] = (
                nonzero
                if feature_stats["nonzero_max"] is None
                else max(feature_stats["nonzero_max"], nonzero)
            )
            if binary_like:
                feature_stats["binary_like_records"] += 1

    return stats


def parse_link_file(path: Path, node_id_to_type: Dict[str, str], edge_specs: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "exists": path.exists(),
        "total": 0,
        "malformed": 0,
        "malformed_samples": [],
        "edge_types": Counter(),
        "endpoint_pairs_by_edge_type": defaultdict(Counter),
        "expected_endpoint_mismatches": Counter(),
        "missing_source": 0,
        "missing_target": 0,
        "duplicate_edges": 0,
        "self_loops": 0,
        "weight_present": 0,
        "weight_missing": 0,
        "weight_non_numeric": 0,
        "weight_min": None,
        "weight_max": None,
        "weight_total": 0.0,
        "weight_numeric_count": 0,
        "out_degree": Counter(),
        "in_degree": Counter(),
        "degree": Counter(),
        "unique_edges_by_type": defaultdict(set),
        "edge_samples_by_type": defaultdict(list),
    }
    if not path.exists():
        return stats

    seen_edges: set[Tuple[str, str, str]] = set()
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n\r")
            parts = line.split("\t")
            if len(parts) < 3:
                stats["malformed"] += 1
                if len(stats["malformed_samples"]) < 5:
                    stats["malformed_samples"].append({"line": line_number, "raw": short_text(line)})
                continue

            source, target, edge_type = parts[0], parts[1], parts[2]
            weight = parts[3] if len(parts) >= 4 else ""
            stats["total"] += 1
            stats["edge_types"][edge_type] += 1
            stats["out_degree"][source] += 1
            stats["in_degree"][target] += 1
            stats["degree"][source] += 1
            stats["degree"][target] += 1
            stats["unique_edges_by_type"][edge_type].add((source, target))

            edge_key = (source, target, edge_type)
            if edge_key in seen_edges:
                stats["duplicate_edges"] += 1
            seen_edges.add(edge_key)

            if source == target:
                stats["self_loops"] += 1

            source_type = node_id_to_type.get(source)
            target_type = node_id_to_type.get(target)
            if source_type is None:
                stats["missing_source"] += 1
            if target_type is None:
                stats["missing_target"] += 1
            pair = f"{source_type if source_type is not None else 'missing'}->{target_type if target_type is not None else 'missing'}"
            stats["endpoint_pairs_by_edge_type"][edge_type][pair] += 1

            expected = edge_specs.get(edge_type)
            if expected and source_type is not None and target_type is not None:
                if source_type != expected.get("start") or target_type != expected.get("end"):
                    stats["expected_endpoint_mismatches"][edge_type] += 1

            if len(stats["edge_samples_by_type"][edge_type]) < 3:
                stats["edge_samples_by_type"][edge_type].append(
                    {"source": source, "target": target, "type": edge_type, "weight": weight}
                )

            if weight == "":
                stats["weight_missing"] += 1
            else:
                stats["weight_present"] += 1
                try:
                    numeric_weight = float(weight)
                    stats["weight_numeric_count"] += 1
                    stats["weight_total"] += numeric_weight
                    stats["weight_min"] = (
                        numeric_weight
                        if stats["weight_min"] is None
                        else min(stats["weight_min"], numeric_weight)
                    )
                    stats["weight_max"] = (
                        numeric_weight
                        if stats["weight_max"] is None
                        else max(stats["weight_max"], numeric_weight)
                    )
                except ValueError:
                    stats["weight_non_numeric"] += 1

    return stats


def parse_label_file(path: Path, node_id_to_type: Dict[str, str]) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "exists": path.exists(),
        "total": 0,
        "unique_node_ids": set(),
        "malformed": 0,
        "malformed_samples": [],
        "missing_node_ids": 0,
        "node_type_mismatches": 0,
        "duplicate_label_ids": 0,
        "duplicate_conflicts": 0,
        "node_type_counts": Counter(),
        "class_counts": Counter(),
        "class_counts_by_node_type": defaultdict(Counter),
        "samples": [],
        "id_to_class": {},
    }
    if not path.exists():
        return stats

    seen_label_ids: Dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n\r")
            parts = line.split("\t")
            if len(parts) < 4:
                stats["malformed"] += 1
                if len(stats["malformed_samples"]) < 5:
                    stats["malformed_samples"].append({"line": line_number, "raw": short_text(line)})
                continue

            node_id, node_name, node_type, class_id = parts[0], parts[1], parts[2], parts[3]
            stats["total"] += 1
            stats["unique_node_ids"].add(node_id)
            stats["node_type_counts"][node_type] += 1
            stats["class_counts"][class_id] += 1
            stats["class_counts_by_node_type"][node_type][class_id] += 1
            stats["id_to_class"][node_id] = class_id

            actual_type = node_id_to_type.get(node_id)
            if actual_type is None:
                stats["missing_node_ids"] += 1
            elif actual_type != node_type:
                stats["node_type_mismatches"] += 1

            if node_id in seen_label_ids:
                stats["duplicate_label_ids"] += 1
                if seen_label_ids[node_id] != class_id:
                    stats["duplicate_conflicts"] += 1
            seen_label_ids[node_id] = class_id

            if len(stats["samples"]) < 5:
                stats["samples"].append(
                    {
                        "node_id": node_id,
                        "name": short_text(node_name),
                        "node_type": node_type,
                        "class": class_id,
                    }
                )

    return stats


def degree_summary(counter: Counter[str], node_total: int) -> Dict[str, Any]:
    values = list(counter.values())
    zero_count = max(node_total - len(values), 0)
    if not values:
        return {
            "min": 0,
            "max": 0,
            "mean_all_nodes": "0.00",
            "mean_nonzero_nodes": "0.00",
            "zero_count": zero_count,
            "nonzero_count": 0,
            "top": [],
        }

    values_with_zero = values + [0] * zero_count
    return {
        "min": min(values_with_zero),
        "max": max(values),
        "mean_all_nodes": mean_value(sum(values), node_total),
        "mean_nonzero_nodes": mean_value(sum(values), len(values)),
        "zero_count": zero_count,
        "nonzero_count": len(values),
        "top": counter.most_common(10),
    }


def detect_reverse_edge_types(edge_specs: Dict[str, Dict[str, str]]) -> Dict[str, List[str]]:
    reverse_map: Dict[str, List[str]] = defaultdict(list)
    for left_type, left_spec in edge_specs.items():
        for right_type, right_spec in edge_specs.items():
            if left_type == right_type:
                continue
            if (
                left_spec.get("start") == right_spec.get("end")
                and left_spec.get("end") == right_spec.get("start")
            ):
                reverse_map[left_type].append(right_type)
    return {key: sorted_type_keys(value) for key, value in reverse_map.items()}


def label_split_overlap(label_stats: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    names = sorted(label_stats.keys())
    overlaps: List[Dict[str, Any]] = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            left_ids = label_stats[left]["unique_node_ids"]
            right_ids = label_stats[right]["unique_node_ids"]
            shared = left_ids.intersection(right_ids)
            conflicts = sum(
                1
                for node_id in shared
                if label_stats[left]["id_to_class"].get(node_id)
                != label_stats[right]["id_to_class"].get(node_id)
            )
            overlaps.append(
                {
                    "left": left,
                    "right": right,
                    "shared": len(shared),
                    "left_total": len(left_ids),
                    "right_total": len(right_ids),
                    "class_conflicts": conflicts,
                }
            )
    return overlaps


def parse_dataset(dataset_dir: Path) -> Dict[str, Any]:
    info_path = dataset_dir / "info.dat"
    meta_path = dataset_dir / "meta.dat"
    info, info_error = read_json_file(info_path) if info_path.exists() else (None, None)
    meta, meta_error = read_json_file(meta_path) if meta_path.exists() else (None, None)
    node_names = get_info_node_names(info)
    attr_dims = get_info_attribute_dims(info)
    edge_specs = get_info_edge_specs(info)
    label_names = get_info_label_names(info)

    files = scan_files(dataset_dir)
    node_stats = parse_node_file(dataset_dir / "node.dat")
    link_stats = parse_link_file(dataset_dir / "link.dat", node_stats["node_id_to_type"], edge_specs)

    label_stats: Dict[str, Dict[str, Any]] = {}
    for file_name in sorted(files.keys()):
        if file_name.startswith("label.dat"):
            label_stats[file_name] = parse_label_file(dataset_dir / file_name, node_stats["node_id_to_type"])

    node_total = int(node_stats["total"])
    link_stats["out_degree_summary"] = degree_summary(link_stats["out_degree"], node_total)
    link_stats["in_degree_summary"] = degree_summary(link_stats["in_degree"], node_total)
    link_stats["total_degree_summary"] = degree_summary(link_stats["degree"], node_total)

    return {
        "name": dataset_dir.name,
        "path": str(dataset_dir),
        "files": files,
        "info": info,
        "info_error": info_error,
        "meta": meta,
        "meta_error": meta_error,
        "node_names": node_names,
        "attr_dims": attr_dims,
        "edge_specs": edge_specs,
        "label_names": label_names,
        "reverse_edge_types": detect_reverse_edge_types(edge_specs),
        "node_stats": node_stats,
        "link_stats": link_stats,
        "label_stats": label_stats,
        "label_overlaps": label_split_overlap(label_stats),
    }


def feature_line(type_id: str, feature_stats: Dict[str, Any]) -> str:
    dims = ensure_counter_text(feature_stats["dims"])
    nonzero_min = feature_stats["nonzero_min"] if feature_stats["nonzero_min"] is not None else 0
    nonzero_max = feature_stats["nonzero_max"] if feature_stats["nonzero_max"] is not None else 0
    nonzero_mean = mean_value(feature_stats["nonzero_total"], feature_stats["with_feature"])
    return (
        f"| {type_id} | {feature_stats['records']} | {feature_stats['with_feature']} | "
        f"{feature_stats['empty_feature_records']} | {dims} | {nonzero_min} | {nonzero_mean} | "
        f"{nonzero_max} | {feature_stats['binary_like_records']} |"
    )


def write_markdown_report(datasets: List[Dict[str, Any]], root: Path) -> str:
    lines: List[str] = []
    lines.append("# Lisan_project 数据集详细读取报告")
    lines.append("")
    lines.append("本报告由 `scripts/audit_lisan_datasets.py` 只读扫描生成，用于评估当前数据集能否用于节点分类和链接预测。")
    lines.append("")
    lines.append(f"- 数据集根目录: `{root}`")
    lines.append(f"- 子数据集数量: {len(datasets)}")
    lines.append("- 读取约定: `node.dat` 通常为 `node_id<TAB>name<TAB>node_type<TAB>feature_vector`。")
    lines.append("- 读取约定: `link.dat` 通常为 `source_id<TAB>target_id<TAB>link_type<TAB>weight`，其中 weight 在部分数据集中不存在。")
    lines.append("- 读取约定: `label.dat*` 通常为 `node_id<TAB>name<TAB>node_type<TAB>class_id`。")
    lines.append("")

    lines.append("## 总览")
    lines.append("")
    lines.append("| 数据集 | 节点数 | 边数 | 标签文件 | info.dat | meta.dat | 可直接节点分类 | 可构造链接预测 |")
    lines.append("| --- | ---: | ---: | --- | --- | --- | --- | --- |")
    for dataset in datasets:
        node_total = dataset["node_stats"]["total"]
        edge_total = dataset["link_stats"]["total"]
        label_files = ", ".join(dataset["label_stats"].keys()) if dataset["label_stats"] else "无"
        direct_nc = "是" if "label.dat" in dataset["label_stats"] else "否"
        lp = "是" if edge_total > 0 else "否"
        lines.append(
            f"| {dataset['name']} | {node_total} | {edge_total} | {label_files} | "
            f"{'有' if dataset['info'] else '无'} | {'有' if dataset['meta'] else '无'} | {direct_nc} | {lp} |"
        )
    lines.append("")

    for dataset in datasets:
        node_stats = dataset["node_stats"]
        link_stats = dataset["link_stats"]
        node_names = dataset["node_names"]
        attr_dims = dataset["attr_dims"]
        edge_specs = dataset["edge_specs"]
        label_names = dataset["label_names"]
        reverse_edge_types = dataset["reverse_edge_types"]

        lines.append(f"## {dataset['name']}")
        lines.append("")
        lines.append(f"- 路径: `{dataset['path']}`")
        lines.append(f"- 节点总数: {node_stats['total']}")
        lines.append(f"- 边总数: {link_stats['total']}")
        lines.append(f"- 节点 ID 范围: {node_stats['id_min']} 到 {node_stats['id_max']}")
        lines.append(f"- node.dat 异常行: {node_stats['malformed']}")
        lines.append(f"- node.dat 重复节点 ID: {node_stats['duplicate_ids']}")
        lines.append(f"- link.dat 异常行: {link_stats['malformed']}")
        lines.append(f"- link.dat 重复边 `(source,target,type)`: {link_stats['duplicate_edges']}")
        lines.append(f"- link.dat 自环边: {link_stats['self_loops']}")
        lines.append(f"- link.dat 缺失 source 节点: {link_stats['missing_source']}")
        lines.append(f"- link.dat 缺失 target 节点: {link_stats['missing_target']}")
        lines.append("")

        lines.append("### 文件清单")
        lines.append("")
        lines.append("| 文件 | 行数 | 字节数 |")
        lines.append("| --- | ---: | ---: |")
        for file_name, file_info in dataset["files"].items():
            lines.append(f"| {file_name} | {file_info['lines']} | {file_info['bytes']} |")
        lines.append("")

        if dataset["info_error"]:
            lines.append(f"- info.dat 解析失败: `{dataset['info_error']}`")
            lines.append("")
        if dataset["meta_error"]:
            lines.append(f"- meta.dat 解析失败: `{dataset['meta_error']}`")
            lines.append("")

        if dataset["meta"]:
            lines.append("### meta.dat 原始统计")
            lines.append("")
            lines.append("| 字段 | 值 |")
            lines.append("| --- | ---: |")
            for key, value in dataset["meta"].items():
                lines.append(f"| {key} | {value} |")
            lines.append("")

        lines.append("### 节点类型与特征")
        lines.append("")
        lines.append("| 节点类型 | info 名称 | 实际数量 | info 特征维度 | 实际特征维度分布 |")
        lines.append("| --- | --- | ---: | ---: | --- |")
        for type_id in sorted_type_keys(node_stats["node_types"].keys()):
            lines.append(
                f"| {type_id} | {node_names.get(type_id, '未知')} | {node_stats['node_types'][type_id]} | "
                f"{attr_dims.get(type_id, '未声明')} | "
                f"{ensure_counter_text(node_stats['feature_by_type'][type_id]['dims'])} |"
            )
        lines.append("")

        lines.append("| 节点类型 | 记录数 | 有特征记录 | 空特征记录 | 维度分布 | 非零最小 | 非零均值 | 非零最大 | 二值向量记录 |")
        lines.append("| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |")
        for type_id in sorted_type_keys(node_stats["feature_by_type"].keys()):
            lines.append(feature_line(type_id, node_stats["feature_by_type"][type_id]))
        lines.append("")

        lines.append("### 节点样例")
        lines.append("")
        lines.append("| 节点类型 | 样例 |")
        lines.append("| --- | --- |")
        for type_id in sorted_type_keys(node_stats["samples_by_type"].keys()):
            sample_text = "; ".join(
                f"{sample['id']}={sample['name']}" for sample in node_stats["samples_by_type"][type_id]
            )
            lines.append(f"| {type_id} | {sample_text} |")
        lines.append("")

        lines.append("### 边类型与端点校验")
        lines.append("")
        lines.append("| 边类型 | info 含义 | info 端点 | 实际边数 | 唯一边数 | 实际端点类型分布 | 端点不匹配 | 可选链接预测正样本密度 | 反向边类型 |")
        lines.append("| --- | --- | --- | ---: | ---: | --- | ---: | --- | --- |")
        for edge_type in sorted_type_keys(link_stats["edge_types"].keys()):
            spec = edge_specs.get(edge_type, {})
            start = spec.get("start", "")
            end = spec.get("end", "")
            possible_pairs = 0
            density = "未知"
            if start in node_stats["node_types"] and end in node_stats["node_types"]:
                possible_pairs = node_stats["node_types"][start] * node_stats["node_types"][end]
                if start == end:
                    possible_pairs = max(possible_pairs - node_stats["node_types"][start], 0)
                density = f"{link_stats['edge_types'][edge_type]}/{possible_pairs} ({percent(link_stats['edge_types'][edge_type], possible_pairs)})"
            reverse = ", ".join(reverse_edge_types.get(edge_type, [])) or "无"
            lines.append(
                f"| {edge_type} | {spec.get('meaning', '未知')} | {start}->{end} | "
                f"{link_stats['edge_types'][edge_type]} | {len(link_stats['unique_edges_by_type'][edge_type])} | "
                f"{ensure_counter_text(link_stats['endpoint_pairs_by_edge_type'][edge_type])} | "
                f"{link_stats['expected_endpoint_mismatches'][edge_type]} | {density} | {reverse} |"
            )
        lines.append("")

        lines.append("### 边样例")
        lines.append("")
        lines.append("| 边类型 | 样例 |")
        lines.append("| --- | --- |")
        for edge_type in sorted_type_keys(link_stats["edge_samples_by_type"].keys()):
            sample_text = "; ".join(
                f"{item['source']}->{item['target']} w={item['weight'] or 'NA'}"
                for item in link_stats["edge_samples_by_type"][edge_type]
            )
            lines.append(f"| {edge_type} | {sample_text} |")
        lines.append("")

        lines.append("### 权重与度统计")
        lines.append("")
        lines.append(f"- 有权重边: {link_stats['weight_present']}")
        lines.append(f"- 无权重边: {link_stats['weight_missing']}")
        lines.append(f"- 非数值权重: {link_stats['weight_non_numeric']}")
        lines.append(
            f"- 数值权重范围: {link_stats['weight_min']} 到 {link_stats['weight_max']}，均值 "
            f"{mean_value(link_stats['weight_total'], link_stats['weight_numeric_count'])}"
        )
        for title, key in [
            ("出度", "out_degree_summary"),
            ("入度", "in_degree_summary"),
            ("总度", "total_degree_summary"),
        ]:
            summary = link_stats[key]
            lines.append(
                f"- {title}: 最小 {summary['min']}，最大 {summary['max']}，全节点均值 {summary['mean_all_nodes']}，"
                f"非零节点均值 {summary['mean_nonzero_nodes']}，零度节点 {summary['zero_count']}，非零节点 {summary['nonzero_count']}"
            )
        lines.append("")
        lines.append("| 度类型 | Top 节点 ID=度 |")
        lines.append("| --- | --- |")
        for title, key in [
            ("出度", "out_degree_summary"),
            ("入度", "in_degree_summary"),
            ("总度", "total_degree_summary"),
        ]:
            top_text = ", ".join(f"{node_id}={degree}" for node_id, degree in link_stats[key]["top"]) or "无"
            lines.append(f"| {title} | {top_text} |")
        lines.append("")

        lines.append("### 标签文件")
        lines.append("")
        if not dataset["label_stats"]:
            lines.append("- 未发现 `label.dat*` 文件，因此不能直接做监督节点分类。")
            lines.append("")
        else:
            if label_names:
                lines.append("#### info.dat 标签含义")
                lines.append("")
                lines.append("| 节点类型 | 类别映射 |")
                lines.append("| --- | --- |")
                for node_type in sorted_type_keys(label_names.keys()):
                    mapping = ", ".join(
                        f"{class_id}={name}"
                        for class_id, name in sorted(
                            label_names[node_type].items(),
                            key=lambda item: (int(item[0]) if item[0].isdigit() else math.inf, item[0]),
                        )
                    )
                    lines.append(f"| {node_type} | {mapping} |")
                lines.append("")

            for label_file, label_stat in dataset["label_stats"].items():
                lines.append(f"#### {label_file}")
                lines.append("")
                lines.append(f"- 标签行数: {label_stat['total']}")
                lines.append(f"- 唯一标签节点数: {len(label_stat['unique_node_ids'])}")
                lines.append(f"- 异常行: {label_stat['malformed']}")
                lines.append(f"- 标签节点不存在于 node.dat: {label_stat['missing_node_ids']}")
                lines.append(f"- 标签声明节点类型与 node.dat 不一致: {label_stat['node_type_mismatches']}")
                lines.append(f"- 重复标签节点 ID: {label_stat['duplicate_label_ids']}")
                lines.append(f"- 重复标签但类别冲突: {label_stat['duplicate_conflicts']}")
                lines.append(f"- 标签节点类型分布: {ensure_counter_text(label_stat['node_type_counts'])}")
                lines.append(f"- 标签类别分布: {ensure_counter_text(label_stat['class_counts'])}")
                if label_stat["samples"]:
                    sample_text = "; ".join(
                        f"{item['node_id']} type={item['node_type']} class={item['class']} name={item['name']}"
                        for item in label_stat["samples"]
                    )
                    lines.append(f"- 标签样例: {sample_text}")
                lines.append("")

            if dataset["label_overlaps"]:
                lines.append("#### 标签划分重叠检查")
                lines.append("")
                lines.append("| 文件 A | 文件 B | 共同节点 | A 节点数 | B 节点数 | 共同节点类别冲突 |")
                lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
                for item in dataset["label_overlaps"]:
                    lines.append(
                        f"| {item['left']} | {item['right']} | {item['shared']} | "
                        f"{item['left_total']} | {item['right_total']} | {item['class_conflicts']} |"
                    )
                lines.append("")

        lines.append("### 任务适配建议")
        lines.append("")
        if "label.dat" in dataset["label_stats"]:
            label_stat = dataset["label_stats"]["label.dat"]
            target_types = ", ".join(sorted_type_keys(label_stat["node_type_counts"].keys()))
            lines.append(
                f"- 节点分类: 可以直接使用 `label.dat` 作为训练标签，目标节点类型为 {target_types}。"
            )
            if "label.dat.test" in dataset["label_stats"]:
                lines.append("- 节点分类: 存在 `label.dat.test`，可作为测试标签。")
            if "label.dat.test_full" in dataset["label_stats"]:
                lines.append("- 节点分类: 存在 `label.dat.test_full`，但需要按实验协议确认它和 `label.dat.test` 的关系。")
        else:
            lines.append("- 节点分类: 当前缺少监督标签，不能直接做标准监督节点分类。")

        if link_stats["total"] > 0:
            lines.append("- 链接预测: 可以从 `link.dat` 的每种边类型构造正样本。")
            lines.append("- 链接预测: 当前未发现官方负样本或 train/valid/test 边划分，需要自行采负样本和划分。")
            if reverse_edge_types:
                lines.append("- 链接预测: 存在反向边类型，划分测试边时应同步处理反向边，避免信息泄漏。")
        else:
            lines.append("- 链接预测: 当前没有边，不能构造标准链接预测任务。")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Read and summarize Lisan_project heterogeneous graph datasets.")
    parser.add_argument(
        "--datasets",
        type=Path,
        default=Path("Datasets"),
        help="Path to the Datasets directory. Default: Datasets",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional Markdown output path. If omitted, the report is printed to stdout.",
    )
    args = parser.parse_args()

    root = args.datasets.resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Datasets directory not found: {root}")

    dataset_dirs = [path for path in sorted(root.iterdir(), key=lambda item: item.name.lower()) if path.is_dir()]
    datasets = [parse_dataset(dataset_dir) for dataset_dir in dataset_dirs]
    report = write_markdown_report(datasets, root)

    if args.output:
        output_path = args.output.resolve()
        output_path.write_text(report, encoding="utf-8")
        print(f"Wrote report: {output_path}")
    else:
        print(report)


if __name__ == "__main__":
    main()
