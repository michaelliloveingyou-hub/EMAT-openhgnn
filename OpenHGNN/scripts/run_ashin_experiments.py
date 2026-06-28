"""
run_ashin_experiments.py

早期的小批量实验脚本。

默认顺序跑下面这些组合，并把结果汇总为 CSV：
1. RGCN baseline
2. RGCN + ASHIN-B
3. RGCN + ASHIN-C
4. SimpleHGN baseline
5. SimpleHGN + ASHIN-B
6. SimpleHGN + ASHIN-C

这个脚本不做超参数搜索，只顺序运行和汇总。summary.csv 会记录 test 指标用于报告，
但不会用 test 指标选择 best_params。正式的 Optuna objective 在 tune_ashin_common.py，
并且只用 valid 指标。

例子：
python scripts/run_ashin_experiments.py --dataset ohgbn-acm --gpu 0 --seeds 0
python scripts/run_ashin_experiments.py --dataset ohgbn-acm --gpu 0 --seeds 0 1 2 3 4
"""

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def latest_run_dir(run_name):
    # ASHIN run 目录会带时间戳；这里找到指定 run_name 对应的最新目录。
    root = ROOT / "openhgnn" / "output" / "ashin_logs"
    matches = sorted(root.glob(f"{run_name}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def read_metrics(run_dir, stdout):
    # 从 metrics.json 优先读取结构化指标；如果没有，再保留 stdout 作为辅助文本。
    data = {}
    if run_dir and (run_dir / "metrics.json").exists():
        data = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    text = stdout
    if run_dir and (run_dir / "train.log").exists():
        text += "\n" + (run_dir / "train.log").read_text(encoding="utf-8", errors="ignore")
    metric = data.get("metric", {}) if isinstance(data, dict) else {}
    valid = metric.get("valid", {})
    test = metric.get("test", {})
    row = {
        "val_acc": valid.get("Accuracy", ""),
        "test_acc": test.get("Accuracy", ""),
        "macro_f1": test.get("Macro_f1", ""),
        "micro_f1": test.get("Micro_f1", test.get("Mirco_f1", "")),
        "best_epoch": data.get("epoch", ""),
        "cache_path": "",
    }
    cache_match = re.search(r"cache_path['\"]?\s*[:=]\s*['\"]([^'\"]+)", text)
    if cache_match:
        row["cache_path"] = cache_match.group(1)
    return row


def run_one(model, dataset, ashin_version, seed, gpu, ashin_dim):
    # 拼出一条 main.py 命令并执行。baseline 不带 ASHIN 参数。
    run_name = f"{model}_{dataset}_ashin{ashin_version or 'baseline'}_seed{seed}_{int(time.time())}"
    cmd = [sys.executable, "main.py", "-m", model, "-d", dataset, "-t", "node_classification", "-g", str(gpu), "--seed", str(seed)]
    if ashin_version:
        cmd += ["--use_ashin", "--ashin_version", ashin_version, "--run_name", run_name]
        if ashin_version == "C":
            cmd += ["--ashin_dim", str(ashin_dim)]
    completed = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    run_dir = latest_run_dir(run_name) if ashin_version else None
    metrics = read_metrics(run_dir, completed.stdout + "\n" + completed.stderr)
    metrics.update({
        "model": model,
        "dataset": dataset,
        "ashin_version": ashin_version or "baseline",
        "seed": seed,
        "run_dir": str(run_dir) if run_dir else "",
        "returncode": completed.returncode,
    })
    if completed.returncode != 0:
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return metrics


def main():
    # 解析多 seed 参数，并按固定顺序运行六组实验。
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ohgbn-acm")
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--ashin_dim", type=int, default=128)
    parser.add_argument("--summary", default="openhgnn/output/ashin_logs/summary.csv")
    args = parser.parse_args()

    jobs = [
        ("RGCN", None),
        ("RGCN", "B"),
        ("RGCN", "C"),
        ("SimpleHGN", None),
        ("SimpleHGN", "B"),
        ("SimpleHGN", "C"),
    ]
    rows = []
    for seed in args.seeds:
        for model, version in jobs:
            rows.append(run_one(model, args.dataset, version, seed, args.gpu, args.ashin_dim))
    out = ROOT / args.summary
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "model", "dataset", "ashin_version", "seed", "val_acc", "test_acc",
            "macro_f1", "micro_f1", "best_epoch", "run_dir", "cache_path", "returncode"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
