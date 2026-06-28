"""
summarize_ashin_results.py

ASHIN 结果汇总脚本。

扫描训练日志和 Optuna 最优参数，按模型、数据集、ASHIN 版本、seed 抽取
valid/test Macro-F1、Micro-F1，最后写成 Excel。
"""

import argparse
import json
import math
import statistics
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]


def _as_num(value):
    if value in ("", None):
        return ""
    try:
        return float(value)
    except Exception:
        return ""


def _mean(values):
    values = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return statistics.mean(values) if values else ""


def _std(values):
    values = [v for v in values if isinstance(v, (int, float)) and not math.isnan(v)]
    return statistics.stdev(values) if len(values) > 1 else ""


def _load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def _metric(metric, split, name):
    split_metric = metric.get(split, {}) if isinstance(metric, dict) else {}
    return _as_num(split_metric.get(name, split_metric.get(name.replace("Micro", "Mirco"), "")))


def collect_seed_results(models, datasets, versions):
    logs_root = ROOT / "openhgnn" / "output" / "ashin_logs"
    latest = {}
    if not logs_root.exists():
        return []
    for run_dir in logs_root.iterdir():
        if not run_dir.is_dir():
            continue
        args_path = run_dir / "args.json"
        metrics_path = run_dir / "metrics.json"
        if not args_path.exists() or not metrics_path.exists():
            continue
        args = _load_json(args_path)
        if not args.get("use_ashin") or not args.get("use_best_config"):
            continue
        model = args.get("model_name", args.get("model"))
        dataset = args.get("dataset_name", args.get("dataset"))
        version = args.get("ashin_version")
        seed = args.get("seed")
        if model not in models or dataset not in datasets or version not in versions:
            continue
        key = (model, dataset, version, seed)
        previous = latest.get(key)
        if previous is None or run_dir.stat().st_mtime > previous.stat().st_mtime:
            latest[key] = run_dir

    rows = []
    for (model, dataset, version, seed), run_dir in sorted(latest.items()):
        args = _load_json(run_dir / "args.json")
        metrics = _load_json(run_dir / "metrics.json")
        metric = metrics.get("metric", {}) if isinstance(metrics, dict) else {}
        rows.append({
            "model": model,
            "dataset": dataset,
            "ashin_version": version,
            "seed": seed,
            "valid_macro_f1": _metric(metric, "valid", "Macro_f1"),
            "valid_micro_f1": _metric(metric, "valid", "Micro_f1"),
            "test_macro_f1": _metric(metric, "test", "Macro_f1"),
            "test_micro_f1": _metric(metric, "test", "Micro_f1"),
            "best_epoch": metrics.get("epoch", ""),
            "ashin_base_version": args.get("ashin_base_version", ""),
            "ashin_dim": args.get("ashin_dim", ""),
            "ashin_norm": args.get("ashin_norm", ""),
            "run_dir": str(run_dir),
        })
    return rows


def collect_best_params(models, datasets, versions):
    rows = []
    root = ROOT / "openhgnn" / "output" / "optuna"
    for model in models:
        for dataset in datasets:
            for version in versions:
                study = root / f"ashin_{model}_{dataset}_{version}"
                best = _load_json(study / "best_params.json") if (study / "best_params.json").exists() else {}
                summary = _load_json(study / "study_summary.json") if (study / "study_summary.json").exists() else {}
                rows.append({
                    "model": model,
                    "dataset": dataset,
                    "ashin_version": version,
                    "best_value": _as_num(summary.get("best_value", "")),
                    "best_params_path": str(study / "best_params.json") if best else "",
                    "best_params_json": json.dumps(best, ensure_ascii=False, sort_keys=True) if best else "",
                })
    return rows


def aggregate(seed_rows):
    groups = {}
    for row in seed_rows:
        key = (row["model"], row["dataset"], row["ashin_version"])
        groups.setdefault(key, []).append(row)
    rows = []
    for key, items in sorted(groups.items()):
        model, dataset, version = key
        rows.append({
            "model": model,
            "dataset": dataset,
            "ashin_version": version,
            "num_seeds": len(items),
            "valid_macro_f1_mean": _mean([r["valid_macro_f1"] for r in items]),
            "valid_macro_f1_std": _std([r["valid_macro_f1"] for r in items]),
            "valid_micro_f1_mean": _mean([r["valid_micro_f1"] for r in items]),
            "valid_micro_f1_std": _std([r["valid_micro_f1"] for r in items]),
            "test_macro_f1_mean": _mean([r["test_macro_f1"] for r in items]),
            "test_macro_f1_std": _std([r["test_macro_f1"] for r in items]),
            "test_micro_f1_mean": _mean([r["test_micro_f1"] for r in items]),
            "test_micro_f1_std": _std([r["test_micro_f1"] for r in items]),
        })
    return rows


def missing_rows(models, datasets, versions, seed_rows, best_rows):
    have_result = {(r["model"], r["dataset"], r["ashin_version"]) for r in seed_rows}
    have_best = {(r["model"], r["dataset"], r["ashin_version"]) for r in best_rows if r["best_params_path"]}
    rows = []
    for model in models:
        for dataset in datasets:
            for version in versions:
                key = (model, dataset, version)
                if key not in have_best:
                    rows.append({"model": model, "dataset": dataset, "ashin_version": version, "missing": "best_params.json"})
                if key not in have_result:
                    rows.append({"model": model, "dataset": dataset, "ashin_version": version, "missing": "seed evaluation metrics"})
    return rows


def col_name(index):
    name = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        name = chr(65 + rem) + name
    return name


def sheet_xml(rows):
    parts = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>']
    parts.append('<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">')
    parts.append("<sheetData>")
    for r_idx, row in enumerate(rows, start=1):
        parts.append(f'<row r="{r_idx}">')
        for c_idx, value in enumerate(row):
            cell = f"{col_name(c_idx)}{r_idx}"
            if isinstance(value, (int, float)) and value != "":
                parts.append(f'<c r="{cell}"><v>{value}</v></c>')
            else:
                text = escape("" if value is None else str(value))
                parts.append(f'<c r="{cell}" t="inlineStr"><is><t>{text}</t></is></c>')
        parts.append("</row>")
    parts.append("</sheetData>")
    parts.append("</worksheet>")
    return "".join(parts)


def table_rows(dict_rows, preferred_headers):
    headers = preferred_headers[:]
    for row in dict_rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    return [headers] + [[row.get(h, "") for h in headers] for row in dict_rows]


def write_xlsx(path, sheets):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet_names = list(sheets.keys())
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
""" + "".join(
            f'<Override PartName="/xl/worksheets/sheet{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            for i in range(1, len(sheet_names) + 1)
        ) + "</Types>")
        zf.writestr("_rels/.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""")
        zf.writestr("xl/workbook.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets>""" + "".join(
            f'<sheet name="{escape(name)}" sheetId="{i}" r:id="rId{i}"/>'
            for i, name in enumerate(sheet_names, start=1)
        ) + "</sheets></workbook>")
        zf.writestr("xl/_rels/workbook.xml.rels", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">""" + "".join(
            f'<Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet{i}.xml"/>'
            for i in range(1, len(sheet_names) + 1)
        ) + f'<Relationship Id="rId{len(sheet_names) + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            + "</Relationships>")
        zf.writestr("xl/styles.xml", """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
<fills count="1"><fill><patternFill patternType="none"/></fill></fills>
<borders count="1"><border/></borders>
<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
<cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
</styleSheet>""")
        for idx, rows in enumerate(sheets.values(), start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", sheet_xml(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="openhgnn/output/overnight/ashin_results.xlsx")
    parser.add_argument("--models", nargs="+", default=["RGCN", "SimpleHGN"])
    parser.add_argument("--datasets", nargs="+", default=["ohgbn-acm", "ohgbn-imdb"])
    parser.add_argument("--versions", nargs="+", default=["E", "F", "B", "C"])
    args = parser.parse_args()

    seed_results = collect_seed_results(args.models, args.datasets, args.versions)
    best_params = collect_best_params(args.models, args.datasets, args.versions)
    summary = aggregate(seed_results)
    missing = missing_rows(args.models, args.datasets, args.versions, seed_results, best_params)
    meta = [{
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(ROOT),
        "note": "Metrics use latest use_best_config run per model/dataset/version/seed.",
    }]

    sheets = {
        "Summary": table_rows(summary, [
            "model", "dataset", "ashin_version", "num_seeds",
            "test_macro_f1_mean", "test_macro_f1_std", "test_micro_f1_mean", "test_micro_f1_std",
            "valid_macro_f1_mean", "valid_macro_f1_std", "valid_micro_f1_mean", "valid_micro_f1_std",
        ]),
        "Seed Results": table_rows(seed_results, [
            "model", "dataset", "ashin_version", "seed",
            "test_macro_f1", "test_micro_f1", "valid_macro_f1", "valid_micro_f1",
            "best_epoch", "ashin_base_version", "ashin_dim", "ashin_norm", "run_dir",
        ]),
        "Best Params": table_rows(best_params, [
            "model", "dataset", "ashin_version", "best_value", "best_params_path", "best_params_json",
        ]),
        "Missing": table_rows(missing, ["model", "dataset", "ashin_version", "missing"]),
        "Meta": table_rows(meta, ["generated_at", "root", "note"]),
    }
    write_xlsx(ROOT / args.output if not Path(args.output).is_absolute() else args.output, sheets)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
